"""
model_lora.py — CLIP-LoRA モデル (継続学習ベースライン)

C-CLIP / FinetuneCLIP との対応表:

  CCLIP (model.py)         LoRACLIP
  ─────────────────────    ──────────────────────────────────────────────
  LoRA inject + merge      inject のみ (タスク間でマージ・再注入なし)
  Projector h_ψ            なし (CKC 損失を使わないため不要)
  old_clip                 なし (知識蒸留なし)
  begin_task() / end_task() なし
  forward(use_ckc=True)    forward() のみ — CLIP 損失に必要な出力だけ返す
  get_param_groups(3 引数)  get_param_groups(lr_image, lr_text) — proj なし

  FinetuneCLIP             LoRACLIP
  ─────────────────────    ──────────────────────────────────────────────
  全パラメータを直接更新    基底重みは固定、LoRA パラメータのみ更新
  パラメータ数: 大          パラメータ数: 小 (LoRA rank に依存)

評価インターフェース (eval_zeroshot.py / trainer.py と共通):
  encode_image(), encode_text(), embed_dim, trainable_params()
  train_transform, val_transform
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from c_clip.lora import inject_lora, count_lora_params


class LoRACLIP(nn.Module):
    """
    LoRA アダプタのみで CLIP を適応させるモデル。

    基底の CLIP 重みは完全に固定し、LoRA パラメータ (A, B 行列) と
    一部の特殊パラメータ (token_embedding, positional_embedding,
    class_embedding, logit_scale) のみを学習する。

    C-CLIP と異なり、タスク間で LoRA をマージ・再注入しないため
    LoRA 重みはタスクをまたいで蓄積される。これは継続学習における
    「LoRA fine-tuning」ベースラインの標準的な設定に対応する。

    Parameters
    ----------
    clip_model_name : CLIP バックボーン ("ViT-B/16" 等)
    lora_rank       : LoRA ランク r (デフォルト: 16)
    lora_alpha      : LoRA スケーリング係数 (デフォルト: 2 * rank)
    lora_dropout    : LoRA ドロップアウト率
    device          : 初期デバイス
    """

    def __init__(
        self,
        clip_model_name: str        = "ViT-B/16",
        lora_rank:       int        = 16,
        lora_alpha:      Optional[int] = None,
        lora_dropout:    float      = 0.1,
        device:          str        = "cuda",
    ):
        super().__init__()

        self.lora_rank    = lora_rank
        self.lora_alpha   = lora_alpha or (2 * lora_rank)
        self.lora_dropout = lora_dropout
        self._device      = device

        # ── CLIP ロード ────────────────────────────────────────────────
        from clip.clip import load as clip_load
        clip_model, self.train_transform, self.val_transform = clip_load(
            clip_model_name, device=device, jit=False
        )
        self.clip = clip_model

        # ── 全パラメータを固定 ─────────────────────────────────────────
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # ── 特殊パラメータを学習可能に (CCLIP と同設定) ───────────────
        # token_embedding / positional_embedding / class_embedding は
        # ドメイン適応で有効とされる軽量なパラメータ (論文 5.2 節参照)。
        # logit_scale は CLIP 損失の温度として常に学習可能にする。
        for name, param in self.clip.named_parameters():
            if any(k in name for k in [
                "token_embedding",
                "positional_embedding",
                "class_embedding",
                "logit_scale",
            ]):
                param.requires_grad = True

        # ── LoRA を注入 ────────────────────────────────────────────────
        # Vision Transformer / Text Transformer の全 ResidualAttentionBlock に
        # LoRAAttention (Q, V, out_proj) および MLP (c_fc, c_proj) を注入する。
        # CCLIP と同じ inject_lora 関数を使用し、アーキテクチャの一貫性を保つ。
        inject_lora(self.clip, self.lora_rank, self.lora_alpha, self.lora_dropout)

        print(f"  LoRACLIP: rank={self.lora_rank}, alpha={self.lora_alpha}, "
              f"dropout={self.lora_dropout}")
        print(f"  学習可能パラメータ: {self.trainable_params():,}")

    # ── 埋め込み次元 (CCLIP / FinetuneCLIP と同一インターフェース) ──
    @property
    def embed_dim(self) -> int:
        return self.clip.text_projection.shape[1]

    # ── 特徴抽出 ─────────────────────────────────────────────────────
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """L2 正規化した画像特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        return F.normalize(self.clip.encode_image(images).float(), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        """L2 正規化したテキスト特徴を返す。CLIPLoss・Recall@K 評価に使用。"""
        return F.normalize(self.clip.encode_text(tokens).float(), dim=-1)

    # ── 順伝播 ────────────────────────────────────────────────────────
    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> dict:
        """
        CLIP 損失計算に必要な出力を返す。

        FinetuneCLIP.forward() と同一インターフェース (use_ckc 引数なし)。

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
        Visual LoRA / Text LoRA で異なる LR を設定したパラメータグループを返す。

        CCLIP.get_param_groups() との違い:
          - projector グループなし (CKC を使わないため)
          - FinetuneCLIP.get_param_groups() と同一シグネチャ

        パラメータの分類基準:
          "clip.visual" を含む名前 → Visual LoRA グループ (lr_image)
          それ以外                  → Text LoRA グループ  (lr_text)
          ※ logit_scale / positional_embedding 等は Text グループに分類
        """
        visual_params: List[torch.Tensor] = []
        text_params:   List[torch.Tensor] = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "clip.visual" in name:
                visual_params.append(p)
            else:
                text_params.append(p)

        return [
            {"params": visual_params, "lr": lr_image, "name": "visual_lora"},
            {"params": text_params,   "lr": lr_text,  "name": "text_lora"},
        ]

    def trainable_params(self) -> int:
        """学習可能なパラメータ数を返す (LoRA + 特殊パラメータの合計)。"""
        return count_lora_params(self)