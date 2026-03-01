"""
model_finetune.py — LoRA なし通常ファインチューニング用 CLIP モデル

CCLIP との対応表:
  CCLIP                    FinetuneCLIP
  ─────────────────────    ──────────────────────────────────────────
  inject_lora / merge_lora なし (全パラメータを直接更新)
  Projector h_ψ            なし (CKC 損失を使わないため不要)
  old_clip                 なし (知識蒸留なし)
  begin_task()             なし
  end_task()               なし
  forward(use_ckc=...)     forward() のみ (CLIP 損失に必要な出力だけ返す)
  get_param_groups(lr_image, lr_text, lr_proj)
                           get_param_groups(lr_image, lr_text)  ← proj なし

評価ループ (trainer.py) との共通インターフェース:
  encode_image(), encode_text(), embed_dim, trainable_params()
  train_transform, val_transform
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinetuneCLIP(nn.Module):
    """
    LoRA なしの通常ファインチューニング用 CLIP ラッパー。

    Parameters
    ----------
    clip_model_name : CLIP バックボーン ("ViT-B/16" 等)
    freeze_vision   : True の場合 Visual Encoder を固定し Text のみ学習
    freeze_text     : True の場合 Text Encoder を固定し Visual のみ学習
                      両方 False (デフォルト) で全エンコーダを学習
    device          : 初期デバイス
    """

    def __init__(
        self,
        clip_model_name: str  = "ViT-B/16",
        freeze_vision:   bool = False,
        freeze_text:     bool = False,
        device:          str  = "cuda",
    ):
        super().__init__()
        self._device = device

        # ── CLIP ロード ────────────────────────────────────────────────
        from clip.clip import load as clip_load
        clip_model, self.train_transform, self.val_transform = clip_load(
            clip_model_name, device=device, jit=False
        )
        self.clip = clip_model

        # ── 学習対象パラメータの設定 ───────────────────────────────────
        # デフォルト: 全パラメータを学習可能にする
        for p in self.clip.parameters():
            p.requires_grad_(True)

        if freeze_vision:
            for name, p in self.clip.named_parameters():
                if "visual" in name:
                    p.requires_grad_(False)

        if freeze_text:
            for name, p in self.clip.named_parameters():
                if "visual" not in name:
                    p.requires_grad_(False)

        # logit_scale は常に学習可能にする
        self.clip.logit_scale.requires_grad_(True)

    # ── 埋め込み次元 (CCLIP との互換) ─────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return self.clip.text_projection.shape[1]

    # ── 特徴抽出 (CCLIP と同一インターフェース) ────────────────────────
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2 正規化した画像特徴を返す。"""
        return F.normalize(self.clip.encode_image(images).float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2 正規化したテキスト特徴を返す。"""
        return F.normalize(self.clip.encode_text(tokens).float(), dim=-1)

    # ── 順伝播 ────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> dict:
        """
        CLIP 損失計算に必要な出力を返す。

        CCLIP.forward() と異なり use_ckc 引数・CKC 関連出力はない。

        Returns
        -------
        dict:
            image_feat  (N, D) : L2 正規化済み画像特徴
            text_feat   (N, D) : L2 正規化済みテキスト特徴
            logit_scale        : 学習可能温度パラメータ (exp 済み)
        """
        return {
            "image_feat":  self.encode_image(images),
            "text_feat":   self.encode_text(tokens),
            "logit_scale": self.clip.logit_scale.exp(),
        }

    # ── パラメータグループ ─────────────────────────────────────────────
    def get_param_groups(self, lr_image: float, lr_text: float) -> List[dict]:
        """
        Visual / Text で異なる LR を設定したパラメータグループを返す。

        CCLIP.get_param_groups() との違い: projector グループなし。
        """
        visual_params, text_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "visual" in name:
                visual_params.append(p)
            else:
                text_params.append(p)

        return [
            {"params": visual_params, "lr": lr_image, "name": "visual"},
            {"params": text_params,   "lr": lr_text,  "name": "text"},
        ]

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)