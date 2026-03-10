"""
lora.py — C-CLIP LoRA モジュール

LoRAConfig により「どの層に LoRA を適用するか」を外部から柔軟に指定できる。

■ LoRAConfig の指定項目
  attn_q   : Attention の Q_linear   に LoRA を適用する
  attn_k   : Attention の K_linear   に LoRA を適用する
  attn_v   : Attention の V_linear   に LoRA を適用する
  attn_out : Attention の out_proj   に LoRA を適用する
  ffn      : FFN の c_fc・c_proj     に LoRA を適用する（c_fc と c_proj は同時適用）

  ※ 視覚エンコーダ・テキストエンコーダの両方に同じ設定が適用される
  ※ visual.proj / text_projection は Parameter なので LoRA 非対応。
    学習可否は model.py の TrainableConfig で制御する。

■ コマンドライン指定例（--lora_targets）
  "q,v"         → Q + V のみ（LoRA 原論文準拠）        パラメータ数 ≈27.1M@R=16
  "q,ffn"       → Q + FFN（論文 Table6 一致）           パラメータ数 ≈29.1M@R=16
  "q,v,out,ffn" → Q + V + out + FFN（元実装と同じ）    パラメータ数 ≈30.1M@R=16
  "q,k,v,out,ffn" → 全 Attention + FFN
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# LoRAConfig
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LoRAConfig:
    """
    LoRA を適用するレイヤーの設定。
    デフォルト設定（Q, V, out_proj, FFN）は既存実装と同じ挙動。

    Attributes
    ----------
    attn_q   : Q_linear       に LoRA を適用 (デフォルト: True)
    attn_k   : K_linear       に LoRA を適用 (デフォルト: False)
    attn_v   : V_linear       に LoRA を適用 (デフォルト: True)
    attn_out : out_proj       に LoRA を適用 (デフォルト: True)
    ffn      : c_fc + c_proj  に LoRA を適用 (デフォルト: True)
    """
    attn_q:   bool = True
    attn_k:   bool = False
    attn_v:   bool = True
    attn_out: bool = True
    ffn:      bool = True

    # ── 文字列パーサ ──────────────────────────────────────────────
    @classmethod
    def from_string(cls, s: str) -> "LoRAConfig":
        """
        カンマ区切り文字列から LoRAConfig を生成する。

        Parameters
        ----------
        s : str
            有効なトークン: q / k / v / out / ffn
            例: "q,v"  →  attn_q=True, attn_v=True, 他は False

        Raises
        ------
        ValueError : 不明なトークンが含まれる場合
        """
        tokens  = {t.strip().lower() for t in s.split(",") if t.strip()}
        valid   = {"q", "k", "v", "out", "ffn"}
        unknown = tokens - valid
        if unknown:
            raise ValueError(
                f"--lora_targets に不明なトークン {sorted(unknown)} が含まれています。"
                f"  有効値: {sorted(valid)}"
            )
        return cls(
            attn_q   = "q"   in tokens,
            attn_k   = "k"   in tokens,
            attn_v   = "v"   in tokens,
            attn_out = "out" in tokens,
            ffn      = "ffn" in tokens,
        )

    # ── ユーティリティ ────────────────────────────────────────────
    def has_attn_lora(self) -> bool:
        """Attention 系に 1 つでも LoRA があるか"""
        return self.attn_q or self.attn_k or self.attn_v or self.attn_out

    def summary(self) -> str:
        """設定内容の要約文字列"""
        parts = []
        if self.attn_q:   parts.append("Q")
        if self.attn_k:   parts.append("K")
        if self.attn_v:   parts.append("V")
        if self.attn_out: parts.append("out_proj")
        if self.ffn:      parts.append("FFN(c_fc+c_proj)")
        return ", ".join(parts) if parts else "（なし）"


# ─────────────────────────────────────────────────────────────────────────────
# LoRALinear
# ─────────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """nn.Linear に LoRA (低ランク適応) を付加するラッパー。"""

    def __init__(self,
                 original: nn.Linear,
                 rank: int,
                 lora_alpha: int,
                 dropout: float = 0.0):
        super().__init__()
        self.in_features  = original.in_features
        self.out_features = original.out_features
        self.rank    = rank
        self.scaling = lora_alpha / rank

        # 元の重みは固定して保持
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # LoRA パラメータ: B=零初期化、A=Kaiming 初期化
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + lora * self.scaling

    def merge(self, alpha: float) -> nn.Linear:
        """W_new = W + alpha * (B @ A) * scaling として nn.Linear を返す。"""
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias is not None,
                           device=self.weight.device,
                           dtype=self.weight.dtype)
        merged.weight.data = self.weight.data + alpha * (self.lora_B @ self.lora_A) * self.scaling
        if self.bias is not None:
            merged.bias.data = self.bias.data.clone()
        return merged

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# LoRAAttention
# ─────────────────────────────────────────────────────────────────────────────
class LoRAAttention(nn.Module):
    """
    CLIP ResidualAttentionBlock.attn (MultiheadAttention) を置き換えるモジュール。
    LoRAConfig に従い Q / K / V / out_proj への LoRA 適用を切り替える。
    """

    def __init__(self,
                 original_mha: nn.MultiheadAttention,
                 rank: int,
                 lora_alpha: int,
                 dropout: float = 0.0,
                 lora_cfg: Optional[LoRAConfig] = None):
        super().__init__()

        cfg = lora_cfg or LoRAConfig()
        self.lora_cfg = cfg

        D = original_mha.embed_dim
        self.embed_dim  = D
        self.num_heads  = original_mha.num_heads
        self.head_dim   = D // self.num_heads
        self.scaling    = self.head_dim ** -0.5
        self.lora_scale = lora_alpha / rank

        # 元の in_proj_weight / bias を固定して保持
        self.in_proj_weight = nn.Parameter(
            original_mha.in_proj_weight.data.clone(), requires_grad=False)
        if original_mha.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(
                original_mha.in_proj_bias.data.clone(), requires_grad=False)
        else:
            self.in_proj_bias = None

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── Q LoRA ──────────────────────────────────────────────────
        if cfg.attn_q:
            self.q_lora_A = nn.Parameter(torch.empty(rank, D))
            self.q_lora_B = nn.Parameter(torch.zeros(D, rank))
            nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))

        # ── K LoRA ──────────────────────────────────────────────────
        if cfg.attn_k:
            self.k_lora_A = nn.Parameter(torch.empty(rank, D))
            self.k_lora_B = nn.Parameter(torch.zeros(D, rank))
            nn.init.kaiming_uniform_(self.k_lora_A, a=math.sqrt(5))

        # ── V LoRA ──────────────────────────────────────────────────
        if cfg.attn_v:
            self.v_lora_A = nn.Parameter(torch.empty(rank, D))
            self.v_lora_B = nn.Parameter(torch.zeros(D, rank))
            nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))

        # ── out_proj ────────────────────────────────────────────────
        # LoRA 適用 → LoRALinear に置き換え
        # 非適用   → 元の重みを固定した nn.Linear として保持
        if cfg.attn_out:
            self.out_proj = LoRALinear(original_mha.out_proj, rank, lora_alpha, dropout)
        else:
            out = nn.Linear(
                original_mha.out_proj.in_features,
                original_mha.out_proj.out_features,
                bias=original_mha.out_proj.bias is not None,
                device=original_mha.out_proj.weight.device,
                dtype=original_mha.out_proj.weight.dtype,
            )
            out.weight = nn.Parameter(
                original_mha.out_proj.weight.data.clone(), requires_grad=False)
            if original_mha.out_proj.bias is not None:
                out.bias = nn.Parameter(
                    original_mha.out_proj.bias.data.clone(), requires_grad=False)
            self.out_proj = out

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (L, N, D) — CLIP Transformer の形式 (seq_len, batch, dim)"""
        L, N, _ = x.shape
        D  = self.embed_dim
        H  = self.num_heads
        Dh = self.head_dim
        cfg = self.lora_cfg

        W_q = self.in_proj_weight[:D]
        W_k = self.in_proj_weight[D:2*D]
        W_v = self.in_proj_weight[2*D:]
        b_q = self.in_proj_bias[:D]    if self.in_proj_bias is not None else None
        b_k = self.in_proj_bias[D:2*D] if self.in_proj_bias is not None else None
        b_v = self.in_proj_bias[2*D:]  if self.in_proj_bias is not None else None

        q = F.linear(x, W_q, b_q)
        if cfg.attn_q:
            q = q + F.linear(F.linear(self.lora_dropout(x), self.q_lora_A),
                              self.q_lora_B) * self.lora_scale

        k = F.linear(x, W_k, b_k)
        if cfg.attn_k:
            k = k + F.linear(F.linear(self.lora_dropout(x), self.k_lora_A),
                              self.k_lora_B) * self.lora_scale

        v = F.linear(x, W_v, b_v)
        if cfg.attn_v:
            v = v + F.linear(F.linear(self.lora_dropout(x), self.v_lora_A),
                              self.v_lora_B) * self.lora_scale

        q = q.reshape(L, N * H, Dh).transpose(0, 1)
        k = k.reshape(L, N * H, Dh).transpose(0, 1)
        v = v.reshape(L, N * H, Dh).transpose(0, 1)

        attn_w = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        if attn_mask is not None:
            attn_w = attn_w + attn_mask
        attn_w = attn_w.softmax(dim=-1)

        out = torch.bmm(attn_w, v).transpose(0, 1).reshape(L, N, D)
        out = self.out_proj(out)
        return out

    def merge(self, alpha: float) -> nn.MultiheadAttention:
        """LoRA デルタを in_proj_weight と out_proj に統合して MultiheadAttention を返す。"""
        D   = self.embed_dim
        cfg = self.lora_cfg
        bias = self.in_proj_bias is not None

        merged = nn.MultiheadAttention(
            D, self.num_heads, bias=bias,
            device=self.in_proj_weight.device,
            dtype=self.in_proj_weight.dtype,
        )

        new_w = self.in_proj_weight.data.clone()
        if cfg.attn_q:
            new_w[:D]    += alpha * (self.q_lora_B @ self.q_lora_A) * self.lora_scale
        if cfg.attn_k:
            new_w[D:2*D] += alpha * (self.k_lora_B @ self.k_lora_A) * self.lora_scale
        if cfg.attn_v:
            new_w[2*D:]  += alpha * (self.v_lora_B @ self.v_lora_A) * self.lora_scale

        merged.in_proj_weight = nn.Parameter(new_w)
        if bias:
            merged.in_proj_bias = nn.Parameter(self.in_proj_bias.data.clone())

        if cfg.attn_out:
            m = self.out_proj.merge(alpha)
            merged.out_proj.weight = nn.Parameter(m.weight.data)
            if m.bias is not None:
                merged.out_proj.bias = nn.Parameter(m.bias.data)
        else:
            merged.out_proj.weight = nn.Parameter(self.out_proj.weight.data.clone())
            if self.out_proj.bias is not None:
                merged.out_proj.bias = nn.Parameter(self.out_proj.bias.data.clone())

        return merged


# ─────────────────────────────────────────────────────────────────────────────
# inject_lora / merge_lora
# ─────────────────────────────────────────────────────────────────────────────
def inject_lora(clip_model: nn.Module,
                rank: int = 16,
                lora_alpha: int = 32,
                dropout: float = 0.1,
                lora_cfg: Optional[LoRAConfig] = None) -> None:
    """
    CLIP の視覚・テキストエンコーダに LoRA を注入する。

    Parameters
    ----------
    lora_cfg : LoRAConfig | None
        None の場合はデフォルト設定（Q, V, out, FFN）を使用。
    """
    cfg = lora_cfg or LoRAConfig()
    _inject_transformer(clip_model.visual.transformer, rank, lora_alpha, dropout, cfg)
    _inject_transformer(clip_model.transformer,        rank, lora_alpha, dropout, cfg)


def _inject_transformer(transformer: nn.Module,
                        rank: int,
                        lora_alpha: int,
                        dropout: float,
                        cfg: LoRAConfig) -> None:

    for block in transformer.resblocks:

        # Attention 系に 1 つでも LoRA があるとき LoRAAttention に置換
        if cfg.has_attn_lora():
            block.attn = LoRAAttention(block.attn, rank, lora_alpha, dropout, cfg)
            _patch_attention_method(block)

        # FFN: c_fc と c_proj を同時に LoRALinear に置換
        if cfg.ffn:
            block.mlp.c_fc   = LoRALinear(block.mlp.c_fc,   rank, lora_alpha, dropout)
            block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank, lora_alpha, dropout)


def _patch_attention_method(block) -> None:
    """
    ResidualAttentionBlock.attention() を LoRAAttention に対応した実装で上書きする。

    DataParallel 対応のため動的サブクラス化を使用する。
    クロージャで block を捕捉すると DataParallel 複製時に GPU0 の block を
    参照し続けて RuntimeError が発生するため。
    """
    OriginalClass = type(block)
    if getattr(OriginalClass, '_lora_attention_patched', False):
        return

    class LoRAResidualAttentionBlock(OriginalClass):
        _lora_attention_patched = True

        def attention(self, x: torch.Tensor) -> torch.Tensor:
            mask = self.attn_mask
            if mask is not None:
                mask = mask.to(dtype=x.dtype, device=x.device)
            return self.attn(x, attn_mask=mask)

    block.__class__ = LoRAResidualAttentionBlock


def merge_lora(clip_model: nn.Module, alpha: float = 0.5) -> None:
    """
    LoRA デルタを CLIP backbone に統合する。
    論文 Eq.(2): θ^t = θ^(t-1) + α * θ_LoRA
    """
    _merge_transformer(clip_model.visual.transformer, alpha)
    _merge_transformer(clip_model.transformer,        alpha)


def _merge_transformer(transformer: nn.Module, alpha: float) -> None:
    for block in transformer.resblocks:
        if isinstance(block.attn, LoRAAttention):
            block.attn = block.attn.merge(alpha)
            _restore_attention_method(block)
        if isinstance(block.mlp.c_fc, LoRALinear):
            block.mlp.c_fc = block.mlp.c_fc.merge(alpha)
        if isinstance(block.mlp.c_proj, LoRALinear):
            block.mlp.c_proj = block.mlp.c_proj.merge(alpha)


def _restore_attention_method(block) -> None:
    """マージ後、block.__class__ を元の ResidualAttentionBlock に戻す。"""
    cls = type(block)
    if getattr(cls, '_lora_attention_patched', False):
        block.__class__ = cls.__bases__[0]


def count_lora_params(model: nn.Module) -> int:
    """学習可能なパラメータ数を返す。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)