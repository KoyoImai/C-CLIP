
import copy
import torch.nn as nn
from typing import Optional


import torch
import torch.nn.functional as F

from clip.clip import load
from c_clip.lora import inject_lora, count_lora_params, merge_lora

# ─────────────────────────────────────────────────────────────────────────────
# Projector h_ψ
# ─────────────────────────────────────────────────────────────────────────────
class Projector(nn.Module):
    """
    論文の h_ψ: D → D

    2 層 MLP + BatchNorm + ReLU。
    新モデルの特徴を「旧空間と繋がりつつ同一でない」空間へ変換し、
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


class CCLIP(nn.Module):

    def __init__(self,
                 clip_model_name: str           = "ViT-B/16",
                 lora_rank: int                 = 16,
                 lora_alpha: Optional[int]      = None,
                 lora_dropout: float            = 0.1,
                 merge_alpha: float             = 0.5,
                 proj_hidden_dim: Optional[int] = None,
                 device: str                    = "cuda"
                 ):
        
        super().__init__()

        # 変数初期化
        self.lora_rank      = lora_rank
        self.lora_alpha     = lora_alpha or (2 * lora_rank)
        self.lora_dropout   = lora_dropout
        self.merge_alpha    = merge_alpha
        self._device        = device

        # -- CLIP の構築 & ロード ----------------------------------
        from clip.clip import load as clip_load
        clip_model, self.train_transform, self.val_transform = clip_load(
            clip_model_name, device=device, jit=False
        )
        self.clip = clip_model
        print(self.clip)

        # 全てのパラメータを固定
        for p in self.clip.parameters():
            p.requires_grad_(False)
        
        # おそらく，トークン埋め込みは学習可能にしいているので，この部分を学習可能にする（論文5.2節参照）
        # self.clip.token_embedding.weight.requires_grad_(True)
        for name, param in self.clip.named_parameters():
            if any(k in name for k in [
                "token_embedding",
                "positional_embedding",
                "class_embedding",
                # "text_projection",
                # "visual.proj",
                # "logit_scale",
            ]):
                param.requires_grad = True

        # -- ckc 損失のための Projector を構築 ----------------------
        embed_dim = self.clip.text_projection.shape[1]
        self.projector = Projector(embed_dim, proj_hidden_dim)
        print(self.projector)

        # -- LoRAモジュールの増築 -----------------------------------
        # どう実装するべきか？
        inject_lora(self.clip, self.lora_rank, self.lora_alpha, self.lora_dropout)

        # -- 旧モデル ---------------------------------------------
        self.old_clip: Optional[nn.Module] = None

    

    # -- 埋め込み次元 -----------------------------------------
    @property
    def embed_dim(self) -> int:
        return self.clip.text_projection.shape[1]

    # ── 特徴抽出 ─────────────────────────────────────────────────────

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2 正規化した画像特徴を返す。"""
        feats = self.clip.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2 正規化したテキスト特徴を返す。"""
        feats = self.clip.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_image_old(self, images: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化した画像特徴を返す (勾配なし)。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        feats = self.old_clip.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text_old(self, tokens: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化したテキスト特徴を返す (勾配なし)。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        feats = self.old_clip.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)


    # ── 順伝播 ────────────────────────────────────────────────────────
    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        use_ckc: bool = False,
    ) -> dict:
        """
        Args:
            images  : (N, C, H, W)
            tokens  : (N, L) — clip.tokenize() の出力
            use_ckc : True の場合、CKC 損失に必要な追加特徴も計算する

        Returns:
            dict:
                image_feat (N, D)   : 新モデル画像特徴
                text_feat  (N, D)   : 新モデルテキスト特徴
                logit_scale         : CLIP の学習可能温度
              use_ckc=True の場合追加:
                image_proj (N, D)   : Projector 適用後の画像特徴
                text_proj  (N, D)   : Projector 適用後のテキスト特徴
                old_image_feat (N,D): 旧モデル画像特徴
                old_text_feat  (N,D): 旧モデルテキスト特徴
        """
        image_feat = self.encode_image(images)
        text_feat  = self.encode_text(tokens)
        logit_scale = self.clip.logit_scale.exp()

        out = {
            "image_feat":  image_feat,
            "text_feat":   text_feat,
            "logit_scale": logit_scale,
        }

        if use_ckc:
            assert self.old_clip is not None, \
                "use_ckc=True には begin_task() で old_clip を設定してください。"
            out["image_proj"]     = self.projector(image_feat)
            out["text_proj"]      = self.projector(text_feat)
            out["old_image_feat"] = self.encode_image_old(images)
            out["old_text_feat"]  = self.encode_text_old(tokens)

        return out


    # ── タスクライフサイクル ──────────────────────────────────────────
    def begin_task(self) -> None:
        """
        新タスク開始前に呼ぶ。
        現在の CLIP を deep copy して旧モデルとして保持する。
        旧モデルは推論専用 (学習なし)。
        """
        self.old_clip = copy.deepcopy(self.clip)
        self.old_clip.eval()
        for p in self.old_clip.parameters():
            p.requires_grad_(False)

    def end_task(self) -> None:
        """
        タスク終了後に呼ぶ。
        LoRA デルタを backbone にマージし、次タスク用に新たな LoRA を注入する。
        Projector のパラメータはそのまま (タスク間で共有)。
        """

        # 現在の device を保存（追加したloraモジュールを同一デバイスに載せるため）
        dev = next(self.clip.parameters()).device

        merge_lora(self.clip, alpha=self.merge_alpha)
        inject_lora(self.clip, self.lora_rank, self.lora_alpha, self.lora_dropout)

        # 新しく挿した loraモジュール を GPU へ移動
        self.clip.to(dev)

        self.old_clip = None

    # -- 学習可能なパラメータ数 ---------------------------------
    def get_param_groups(
        self,
        lr_image: float,
        lr_text: float,
        lr_proj: float,
    ) -> list:
        """
        Visual LoRA / Text LoRA / Projector で異なる LR を設定したパラメータグループ。

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
            else:  # clip.transformer (text encoder)
                text_params.append(p)

        return [
            {"params": visual_params, "lr": lr_image, "name": "visual_lora"},
            {"params": text_params,   "lr": lr_text,  "name": "text_lora"},
            {"params": proj_params,   "lr": lr_proj,  "name": "projector"},
        ]
    
    def trainable_params(self) -> int:
        return count_lora_params(self)
