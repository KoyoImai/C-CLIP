"""
losses.py — C-CLIP 損失関数

CLIPLoss  : 論文 Eq.(6) — 標準的な対称交差エントロピー対照損失
CKCLoss   : 論文 Eq.(5) — Contrastive Knowledge Consolidation
              新モデルの projected 特徴と旧モデル特徴の間で対照学習
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    標準 CLIP 対称対照損失 (Symmetric Cross-Entropy)。

    Eq.(6):
      L_CLIP = 1/2 * (CE(I→T) + CE(T→I))

    - image_features / text_features は L2 正規化済みを想定
    - logit_scale は CLIP モデルが持つ学習可能温度パラメータ
    """

    def forward(
        self,
        image_features: torch.Tensor,   # (N, D) L2 正規化済み
        text_features: torch.Tensor,    # (N, D) L2 正規化済み
        logit_scale: torch.Tensor,      # scalar (CLIP.logit_scale.exp())
    ) -> torch.Tensor:
        logits_per_image = logit_scale * image_features @ text_features.t()  # (N, N)
        logits_per_text  = logits_per_image.t()

        labels = torch.arange(len(image_features), device=image_features.device)

        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text,  labels)

        return (loss_i2t + loss_t2i) / 2.0


class CKCLoss(nn.Module):
    """
    Contrastive Knowledge Consolidation Loss。

    Eq.(5):
      L_CKC = 1/2 * (CE(h̃→z̃) + CE(z̃→h̃))

    where:
      h̃_i = normalize([h_ψ(f_θt(v_i)),  h_ψ(g_φt(c_i))])  ← 新モデル + projector
      z̃_i = normalize([f_{θ(t-1)}(v_i), g_{φ(t-1)}(c_i)]) ← 旧モデル

    正例: 同一インデックス (同じ image-text ペアを新旧モデルで処理)
    負例: 異なるインデックス
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        new_image_proj: torch.Tensor,   # (N, D) 新モデル画像 → Projector 後
        new_text_proj: torch.Tensor,    # (N, D) 新モデルテキスト → Projector 後
        old_image_feat: torch.Tensor,   # (N, D) 旧モデル画像特徴 (L2 正規化済み)
        old_text_feat: torch.Tensor,    # (N, D) 旧モデルテキスト特徴 (L2 正規化済み)
    ) -> torch.Tensor:
        # 論文の記述通り image+text を cat してから正規化
        h = F.normalize(torch.cat([new_image_proj, new_text_proj], dim=-1), dim=-1)  # (N, 2D)
        z = F.normalize(torch.cat([old_image_feat,  old_text_feat],  dim=-1), dim=-1)  # (N, 2D)

        sim_h2z = h @ z.t() / self.temperature   # (N, N)
        sim_z2h = z @ h.t() / self.temperature   # (N, N)

        labels = torch.arange(len(h), device=h.device)

        loss_h2z = F.cross_entropy(sim_h2z, labels)
        loss_z2h = F.cross_entropy(sim_z2h, labels)

        return (loss_h2z + loss_z2h) / 2.0
