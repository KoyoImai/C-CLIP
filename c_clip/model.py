"""
model.py — C-CLIP モデル本体

CCLIP は以下の 2 つの Config で学習パラメータを柔軟に制御できる。

■ LoRAConfig（lora.py）
  どの層に LoRA を適用するかを指定する。
  --lora_targets q,v,ffn  のようにコマンドラインから指定可能。

■ TrainableConfig（このファイル）
  LoRA 非適用パラメータのうち、どれを学習可能にするかを指定する。
  --trainable_params token_embedding,logit_scale のように指定可能。

  指定できるキー:
    token_embedding      : テキストエンコーダのトークン埋め込み      (25.3M)
    text_pos_embedding   : テキストエンコーダの位置埋め込み          (39K)
    text_projection      : テキストエンコーダの最終出力線形層        (262K)
    class_embedding      : 画像エンコーダの CLS トークン             (768)
    visual_pos_embedding : 画像エンコーダの位置埋め込み              (151K)
    visual_proj          : 画像エンコーダの最終出力線形層            (393K)
    logit_scale          : CLIP の学習可能温度パラメータ (スカラー)  (1)
"""

import copy
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip.clip import load
from c_clip.lora import inject_lora, count_lora_params, merge_lora, LoRAConfig


# ─────────────────────────────────────────────────────────────────────────────
# TrainableConfig
# ─────────────────────────────────────────────────────────────────────────────

# LoRA 非適用パラメータのキー → CLIP 内の名前パターン のマッピング
_TRAINABLE_KEY_TO_PATTERN = {
    "token_embedding":      "token_embedding",
    "text_pos_embedding":   "positional_embedding",   # テキストエンコーダ側
    "text_projection":      "text_projection",
    "class_embedding":      "class_embedding",
    "visual_pos_embedding": "positional_embedding",   # 画像エンコーダ側
    "visual_proj":          "visual.proj",
    "logit_scale":          "logit_scale",
}

# パラメータ数の概算（ViT-B/16、説明用）
_PARAM_COUNT_HINT = {
    "token_embedding":      "25,296,896",
    "text_pos_embedding":   "39,424",
    "text_projection":      "262,144",
    "class_embedding":      "768",
    "visual_pos_embedding": "151,296",
    "visual_proj":          "393,216",
    "logit_scale":          "1",
}

VALID_TRAINABLE_KEYS = set(_TRAINABLE_KEY_TO_PATTERN.keys())


@dataclass
class TrainableConfig:
    """
    LoRA 非適用パラメータのうち、学習可能にするものを指定する。

    デフォルト設定は既存実装と同じ:
      token_embedding, text_pos_embedding, class_embedding,
      visual_pos_embedding, logit_scale を学習可能にする。
      text_projection / visual_proj は固定。

    Attributes
    ----------
    keys : List[str]
        学習可能にするキーのリスト。
        有効なキーは VALID_TRAINABLE_KEYS を参照。
    """
    keys: List[str] = field(default_factory=lambda: [
        "token_embedding",
        "text_pos_embedding",
        "class_embedding",
        "visual_pos_embedding",
        "logit_scale",
    ])

    @classmethod
    def from_string(cls, s: str) -> "TrainableConfig":
        """
        カンマ区切り文字列から TrainableConfig を生成する。

        Parameters
        ----------
        s : str
            例: "token_embedding,logit_scale,visual_proj"

        Raises
        ------
        ValueError : 不明なキーが含まれる場合
        """
        keys    = [k.strip() for k in s.split(",") if k.strip()]
        unknown = set(keys) - VALID_TRAINABLE_KEYS
        if unknown:
            raise ValueError(
                f"--trainable_params に不明なキー {sorted(unknown)} が含まれています。\n"
                f"  有効なキー: {sorted(VALID_TRAINABLE_KEYS)}"
            )
        return cls(keys=keys)

    def summary(self) -> str:
        """設定内容の要約文字列"""
        if not self.keys:
            return "（なし）"
        lines = []
        for k in self.keys:
            hint = _PARAM_COUNT_HINT.get(k, "?")
            lines.append(f"{k} ({hint} params)")
        return "\n    ".join(lines)

    def should_train(self, param_name: str) -> bool:
        """
        CLIP の named_parameters() の name に対して学習可能にすべきか判定する。

        Parameters
        ----------
        param_name : str
            clip.named_parameters() から得られるパラメータ名
            例: "visual.proj", "token_embedding.weight", "positional_embedding"
        """
        for key in self.keys:
            pattern = _TRAINABLE_KEY_TO_PATTERN[key]

            # visual_pos_embedding と text_pos_embedding は同じパターン "positional_embedding" を
            # 使うため、CLIP の名前空間（visual か否か）で区別する
            if key == "visual_pos_embedding":
                if "visual" in param_name and pattern in param_name:
                    return True
            elif key == "text_pos_embedding":
                if "visual" not in param_name and pattern in param_name:
                    return True
            else:
                if pattern in param_name:
                    return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Projector h_ψ
# ─────────────────────────────────────────────────────────────────────────────
class Projector(nn.Module):
    """
    論文の h_ψ: D → D

    2 層 MLP + BatchNorm + ReLU。
    新モデルの特徴を旧空間と繋がりつつ同一でない空間へ変換し、
    CKC 損失で知識を保持する。
    """

    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim * 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# CCLIP
# ─────────────────────────────────────────────────────────────────────────────
class CCLIP(nn.Module):

    def __init__(self,
                 clip_model_name: str                = "ViT-B/16",
                 lora_rank: int                      = 16,
                 lora_alpha: Optional[int]           = None,
                 lora_dropout: float                 = 0.1,
                 merge_alpha: float                  = 0.5,
                 proj_hidden_dim: Optional[int]      = None,
                 device: str                         = "cuda",
                 lora_cfg: Optional[LoRAConfig]      = None,
                 trainable_cfg: Optional[TrainableConfig] = None,
                 ):
        super().__init__()

        self.lora_rank    = lora_rank
        self.lora_alpha   = lora_alpha or (2 * lora_rank)
        self.lora_dropout = lora_dropout
        self.merge_alpha  = merge_alpha
        self._device      = device

        # Config のデフォルト（既存実装と同じ挙動）
        self.lora_cfg      = lora_cfg      or LoRAConfig()
        self.trainable_cfg = trainable_cfg or TrainableConfig()

        # ── CLIP のロード ────────────────────────────────────────
        from clip.clip import load as clip_load
        clip_model, self.train_transform, self.val_transform = clip_load(
            clip_model_name, device=device, jit=False
        )
        self.clip = clip_model

        # ── 全パラメータを固定 ───────────────────────────────────
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # ── TrainableConfig に従いパラメータを解凍 ───────────────
        for name, param in self.clip.named_parameters():
            if self.trainable_cfg.should_train(name):
                param.requires_grad = True

        # ── Projector の構築 ─────────────────────────────────────
        embed_dim = self.clip.text_projection.shape[1]
        self.projector = Projector(embed_dim, proj_hidden_dim)

        # ── LoRA の注入 ──────────────────────────────────────────
        inject_lora(self.clip, self.lora_rank, self.lora_alpha,
                    self.lora_dropout, self.lora_cfg)

        # ── 旧モデル ─────────────────────────────────────────────
        self.old_clip: Optional[nn.Module] = None

    # ── 設定サマリ ─────────────────────────────────────────────────
    def print_config(self) -> None:
        """学習設定の概要を標準出力に表示する。"""
        print("=" * 60)
        print("  C-CLIP 学習パラメータ設定")
        print("=" * 60)
        print(f"  LoRA rank    : {self.lora_rank}")
        print(f"  LoRA alpha   : {self.lora_alpha}")
        print(f"  LoRA dropout : {self.lora_dropout}")
        print()
        print(f"  【LoRA 適用レイヤー】")
        print(f"    {self.lora_cfg.summary()}")
        print()
        print(f"  【LoRA 非適用・学習可能パラメータ】")
        print(f"    {self.trainable_cfg.summary()}")
        print()
        # パラメータ数の内訳
        lora_p    = sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and "projector" not in n
                        and any(x in n for x in ["lora_A", "lora_B"]))
        fixed_p   = sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and "projector" not in n
                        and not any(x in n for x in ["lora_A", "lora_B"]))
        proj_p    = sum(p.numel() for n, p in self.named_parameters()
                        if p.requires_grad and "projector" in n)
        total_p   = lora_p + fixed_p + proj_p
        print(f"  【学習可能パラメータ数】")
        print(f"    LoRA パラメータ              : {lora_p:>12,}")
        print(f"    LoRA 非適用・CLIP 本体       : {fixed_p:>12,}")
        print(f"    Projector h_ψ               : {proj_p:>12,}")
        print(f"    合計                         : {total_p:>12,}")
        print("=" * 60)

    # ── 埋め込み次元 ───────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return self.clip.text_projection.shape[1]

    # ── 特徴抽出 ───────────────────────────────────────────────────
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2 正規化した画像特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        return F.normalize(self.clip.encode_image(images).float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2 正規化したテキスト特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        return F.normalize(self.clip.encode_text(tokens).float(), dim=-1)

    def _encode_image_raw(self, images: torch.Tensor) -> torch.Tensor:
        """正規化前の生画像特徴を返す。Projector への入力専用。"""
        return self.clip.encode_image(images).float()

    def _encode_text_raw(self, tokens: torch.Tensor) -> torch.Tensor:
        """正規化前の生テキスト特徴を返す。Projector への入力専用。"""
        return self.clip.encode_text(tokens).float()

    @torch.no_grad()
    def encode_image_old(self, images: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化した画像特徴を返す (勾配なし)。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return F.normalize(self.old_clip.encode_image(images).float(), dim=-1)

    @torch.no_grad()
    def encode_text_old(self, tokens: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化したテキスト特徴を返す (勾配なし)。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return F.normalize(self.old_clip.encode_text(tokens).float(), dim=-1)

    @torch.no_grad()
    def _encode_image_raw_old(self, images: torch.Tensor) -> torch.Tensor:
        """旧モデルの正規化前の生画像特徴を返す (勾配なし)。CKCLoss 用。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return self.old_clip.encode_image(images).float()

    @torch.no_grad()
    def _encode_text_raw_old(self, tokens: torch.Tensor) -> torch.Tensor:
        """旧モデルの正規化前の生テキスト特徴を返す (勾配なし)。CKCLoss 用。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return self.old_clip.encode_text(tokens).float()

    # ── 順伝播 ────────────────────────────────────────────────────
    def forward(self,
                images: torch.Tensor,
                tokens: torch.Tensor,
                use_ckc: bool = False) -> dict:
        """
        Args:
            images  : (N, C, H, W)
            tokens  : (N, L) — clip.tokenize() の出力
            use_ckc : True の場合、CKC 損失に必要な追加特徴も計算する

        Returns dict:
            image_feat    (N, D) : 新モデル画像特徴  [L2 正規化済み / CLIPLoss 用]
            text_feat     (N, D) : 新モデルテキスト特徴 [L2 正規化済み / CLIPLoss 用]
            logit_scale          : CLIP の学習可能温度パラメータ
          use_ckc=True の場合追加:
            image_proj    (N, D) : 生特徴 → Projector 後 [正規化なし / CKCLoss 用]
            text_proj     (N, D) : 生特徴 → Projector 後 [正規化なし / CKCLoss 用]
            old_image_feat (N,D) : 旧モデルの生画像特徴  [正規化なし / CKCLoss 用]
            old_text_feat  (N,D) : 旧モデルの生テキスト特徴 [正規化なし / CKCLoss 用]
        """
        image_raw  = self._encode_image_raw(images)
        text_raw   = self._encode_text_raw(tokens)
        image_feat = F.normalize(image_raw, dim=-1)
        text_feat  = F.normalize(text_raw,  dim=-1)
        logit_scale = self.clip.logit_scale.exp()

        out = {
            "image_feat":  image_feat,
            "text_feat":   text_feat,
            "logit_scale": logit_scale,
        }

        if use_ckc:
            assert self.old_clip is not None, \
                "use_ckc=True には begin_task() で old_clip を設定してください。"
            out["image_proj"]     = self.projector(image_raw)
            out["text_proj"]      = self.projector(text_raw)
            out["old_image_feat"] = self._encode_image_raw_old(images)
            out["old_text_feat"]  = self._encode_text_raw_old(tokens)

        return out

    # ── タスクライフサイクル ───────────────────────────────────────
    def begin_task(self) -> None:
        """新タスク開始前に呼ぶ。現在の CLIP を deep copy して旧モデルとして保持する。"""
        self.old_clip = copy.deepcopy(self.clip)
        self.old_clip.eval()
        for p in self.old_clip.parameters():
            p.requires_grad_(False)

    def end_task(self) -> None:
        """
        タスク終了後に呼ぶ。
        LoRA デルタを backbone にマージし、次タスク用に新たな LoRA を注入する。
        """
        dev = next(self.clip.parameters()).device
        merge_lora(self.clip, alpha=self.merge_alpha)
        inject_lora(self.clip, self.lora_rank, self.lora_alpha,
                    self.lora_dropout, self.lora_cfg)
        self.clip.to(dev)
        self.old_clip = None

    # ── パラメータグループ ─────────────────────────────────────────
    def get_param_groups(self,
                         lr_image: float,
                         lr_text:  float,
                         lr_proj:  float) -> list:
        """
        Visual / Text / Projector で異なる LR を設定したパラメータグループを返す。
        論文: COCO タスクは text_lr = 80 × image_lr、他は 10 × image_lr。
        """
        visual_params, text_params, proj_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "projector" in name:
                proj_params.append(p)
            elif "clip.visual" in name:
                visual_params.append(p)
            else:
                text_params.append(p)

        return [
            {"params": visual_params, "lr": lr_image, "name": "visual"},
            {"params": text_params,   "lr": lr_text,  "name": "text"},
            {"params": proj_params,   "lr": lr_proj,  "name": "projector"},
        ]

    def trainable_params(self) -> int:
        return count_lora_params(self)