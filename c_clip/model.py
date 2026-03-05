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
                "logit_scale",
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
        """L2 正規化した画像特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        feats = self.clip.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2 正規化したテキスト特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        feats = self.clip.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    def _encode_image_raw(self, images: torch.Tensor) -> torch.Tensor:
        """正規化前の生画像特徴を返す。Projector への入力専用。

        L2 正規化は情報を破壊する:
          - ベクトルのノルム（大きさ）情報が消える
          - 全出力が単位球面上に射影され、分布が均一になる
          → Projector 内部の BatchNorm が有効に機能しなくなる

        対照学習の慣習 (SimCLR 等) に従い、backbone の生出力を
        Projector に渡し、L2 正規化は CKCLoss 内の cat 後に行う。
        """
        return self.clip.encode_image(images).float()

    def _encode_text_raw(self, tokens: torch.Tensor) -> torch.Tensor:
        """正規化前の生テキスト特徴を返す。Projector への入力専用。"""
        return self.clip.encode_text(tokens).float()

    @torch.no_grad()
    def encode_image_old(self, images: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化した画像特徴を返す (勾配なし)。CLIPLoss 用。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        feats = self.old_clip.encode_image(images)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text_old(self, tokens: torch.Tensor) -> torch.Tensor:
        """旧モデルで L2 正規化したテキスト特徴を返す (勾配なし)。CLIPLoss 用。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        feats = self.old_clip.encode_text(tokens)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def _encode_image_raw_old(self, images: torch.Tensor) -> torch.Tensor:
        """旧モデルの正規化前の生画像特徴を返す (勾配なし)。CKCLoss 用。

        CKCLoss では z̃_i = normalize([f_{t-1}(v), g_{t-1}(c)]) と定義されており、
        個別正規化ではなく cat 後にまとめて正規化する。
        したがって旧モデルの特徴も生のまま渡し、正規化は CKCLoss 内に委ねる。
        """
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return self.old_clip.encode_image(images).float()

    @torch.no_grad()
    def _encode_text_raw_old(self, tokens: torch.Tensor) -> torch.Tensor:
        """旧モデルの正規化前の生テキスト特徴を返す (勾配なし)。CKCLoss 用。"""
        if self.old_clip is None:
            raise RuntimeError("old_clip が未設定です。begin_task() を先に呼んでください。")
        return self.old_clip.encode_text(tokens).float()


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
                image_feat    (N, D) : 新モデル画像特徴  [L2 正規化済み / CLIPLoss 用]
                text_feat     (N, D) : 新モデルテキスト特徴 [L2 正規化済み / CLIPLoss 用]
                logit_scale          : CLIP の学習可能温度パラメータ
              use_ckc=True の場合追加:
                image_proj    (N, D) : 生特徴 → Projector 後 [正規化なし / CKCLoss 用]
                text_proj     (N, D) : 生特徴 → Projector 後 [正規化なし / CKCLoss 用]
                old_image_feat (N,D) : 旧モデルの生画像特徴  [正規化なし / CKCLoss 用]
                old_text_feat  (N,D) : 旧モデルの生テキスト特徴 [正規化なし / CKCLoss 用]

        【設計方針】
        CLIPLoss と CKCLoss で特徴の使われ方が異なる:
          - CLIPLoss : image_feat と text_feat の内積を取る
                       → 事前に L2 正規化して cos 類似度を計算するのが標準
          - CKCLoss  : h̃_i = normalize([h_ψ(f(v)), h_ψ(g(c))])
                             └── Projector への入力は生特徴 (ノルム情報を保持)
                       z̃_i = normalize([f_{t-1}(v), g_{t-1}(c)])
                             └── cat 後に一括正規化 → 個別正規化は不要
        したがって encode では正規化前の生特徴を Projector と CKC に渡し、
        L2 正規化は CLIPLoss 用の encode_image / encode_text にのみ適用する。
        """
        # ── 生特徴を1回だけ計算し、正規化版を派生させる ────────────────
        # use_ckc=True のとき encode_image と _encode_image_raw を別々に呼ぶと
        # エンコーダーが2回走り非効率。生特徴から正規化版を派生させる。
        image_raw   = self._encode_image_raw(images)   # (N, D) 正規化なし
        text_raw    = self._encode_text_raw(tokens)    # (N, D) 正規化なし
        image_feat  = F.normalize(image_raw, dim=-1)   # (N, D) 正規化済み
        text_feat   = F.normalize(text_raw,  dim=-1)   # (N, D) 正規化済み
        logit_scale = self.clip.logit_scale.exp()

        out = {
            "image_feat":  image_feat,
            "text_feat":   text_feat,
            "logit_scale": logit_scale,
        }

        if use_ckc:
            assert self.old_clip is not None, \
                "use_ckc=True には begin_task() で old_clip を設定してください。"

            # ── CKCLoss 用: 生特徴 → Projector (新モデル) ───────────────
            # 正規化前の backbone 出力を Projector に渡す。
            # ノルム情報を保持することで Projector の BatchNorm が有効に機能し、
            # より豊かな知識変換空間を学習できる。
            # （image_raw / text_raw は既に上で計算済みなので追加コストなし）
            out["image_proj"] = self.projector(image_raw)   # (N, D)
            out["text_proj"]  = self.projector(text_raw)    # (N, D)

            # ── CKCLoss 用: 旧モデルの生特徴 ────────────────────────────
            # CKCLoss 内で cat([old_image, old_text]) → normalize するため
            # 旧モデルの特徴も個別正規化なしの生のまま渡す。
            out["old_image_feat"] = self._encode_image_raw_old(images)   # (N, D)
            out["old_text_feat"]  = self._encode_text_raw_old(tokens)    # (N, D)

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