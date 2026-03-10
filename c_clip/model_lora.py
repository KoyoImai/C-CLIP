"""
model_lora.py — CLIP-LoRA モデル (継続学習ベースライン)

C-CLIP / FinetuneCLIP との対応表:

  CCLIP (model.py)          LoRACLIP
  ─────────────────────     ──────────────────────────────────────────────
  LoRA inject + merge       inject のみ (タスク間でマージ・再注入なし)
  Projector h_ψ             なし (CKC 損失を使わないため不要)
  old_clip                  なし (知識蒸留なし)
  begin_task() / end_task() なし
  forward(use_ckc=True)     forward() のみ — CLIP 損失に必要な出力だけ返す
  get_param_groups(3 引数)   get_param_groups(lr_image, lr_text) — proj なし
  lora_cfg / trainable_cfg  同じ設計で対応済み ← 追加

  FinetuneCLIP              LoRACLIP
  ─────────────────────     ──────────────────────────────────────────────
  全パラメータを直接更新     基底重みは固定、LoRA パラメータのみ更新
  パラメータ数: 大           パラメータ数: 小 (LoRA rank に依存)

評価インターフェース (eval_zeroshot.py / trainer.py と共通):
  encode_image(), encode_text(), embed_dim, trainable_params()
  train_transform, val_transform
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from c_clip.lora import inject_lora, count_lora_params, LoRAConfig
from c_clip.model import TrainableConfig


class LoRACLIP(nn.Module):
    """
    LoRA アダプタのみで CLIP を適応させるモデル。

    CCLIP と同じ LoRAConfig / TrainableConfig を受け取り、
    main_lora.py から --lora_targets / --trainable_params で制御できる。

    Parameters
    ----------
    clip_model_name : CLIP バックボーン ("ViT-B/16" 等)
    lora_rank       : LoRA ランク r (デフォルト: 16)
    lora_alpha      : LoRA スケーリング係数 (デフォルト: 2 * rank)
    lora_dropout    : LoRA ドロップアウト率
    device          : 初期デバイス
    lora_cfg        : どの層に LoRA を適用するかの設定 (None でデフォルト)
    trainable_cfg   : LoRA 非適用パラメータの学習可否設定 (None でデフォルト)
    """

    def __init__(
        self,
        clip_model_name: str                      = "ViT-B/16",
        lora_rank:       int                      = 16,
        lora_alpha:      Optional[int]            = None,
        lora_dropout:    float                    = 0.1,
        device:          str                      = "cuda",
        lora_cfg:        Optional[LoRAConfig]     = None,
        trainable_cfg:   Optional[TrainableConfig] = None,
    ):
        super().__init__()

        self.lora_rank    = lora_rank
        self.lora_alpha   = lora_alpha or (2 * lora_rank)
        self.lora_dropout = lora_dropout
        self._device      = device

        # Config のデフォルト（CCLIP と同じ挙動）
        self.lora_cfg      = lora_cfg      or LoRAConfig()
        self.trainable_cfg = trainable_cfg or TrainableConfig()

        # ── CLIP ロード ────────────────────────────────────────────────
        from clip.clip import load as clip_load
        clip_model, self.train_transform, self.val_transform = clip_load(
            clip_model_name, device=device, jit=False
        )
        self.clip = clip_model

        # ── 全パラメータを固定 ─────────────────────────────────────────
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # ── TrainableConfig に従いパラメータを解凍 ─────────────────────
        for name, param in self.clip.named_parameters():
            if self.trainable_cfg.should_train(name):
                param.requires_grad = True

        # ── LoRA を注入 ────────────────────────────────────────────────
        inject_lora(
            self.clip,
            self.lora_rank,
            self.lora_alpha,
            self.lora_dropout,
            self.lora_cfg,
        )

    # ── 設定サマリ ─────────────────────────────────────────────────────
    def print_config(self) -> None:
        """学習設定の概要を標準出力に表示する。"""
        print("=" * 60)
        print("  CLIP-LoRA 学習パラメータ設定")
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
        lora_p  = sum(p.numel() for n, p in self.named_parameters()
                      if p.requires_grad and any(x in n for x in ["lora_A", "lora_B"]))
        fixed_p = sum(p.numel() for n, p in self.named_parameters()
                      if p.requires_grad and not any(x in n for x in ["lora_A", "lora_B"]))
        print(f"  【学習可能パラメータ数】")
        print(f"    LoRA パラメータ         : {lora_p:>12,}")
        print(f"    LoRA 非適用・CLIP 本体  : {fixed_p:>12,}")
        print(f"    合計                    : {lora_p+fixed_p:>12,}")
        print("=" * 60)

    # ── 埋め込み次元 ───────────────────────────────────────────────────
    @property
    def embed_dim(self) -> int:
        return self.clip.text_projection.shape[1]

    # ── 特徴抽出 ───────────────────────────────────────────────────────
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
        Visual / Text で異なる LR を設定したパラメータグループを返す。

        CCLIP.get_param_groups() との違い: projector グループなし。
        FinetuneCLIP.get_param_groups() と同一シグネチャ。
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
            {"params": visual_params, "lr": lr_image, "name": "visual"},
            {"params": text_params,   "lr": lr_text,  "name": "text"},
        ]

    def trainable_params(self) -> int:
        """学習可能なパラメータ数を返す。"""
        return count_lora_params(self)