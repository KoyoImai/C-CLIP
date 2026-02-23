

import torch.nn as nn
from typing import Optional


import torch

from clip.clip import load
from c_clip.lora import inject_lora, count_lora_params

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
                 lora_rank: int                 = 12,
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
