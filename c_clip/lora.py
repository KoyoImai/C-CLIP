
import math
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# LoRALinear: 通常の nn.Linear に LoRA を付加
# ----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """
    nn.Linear に LoRA モジュールを追加するためのラッパー
    """
    def __init__(self,
                 original: nn.Linear,
                 rank: int,
                 lora_alphaa: int,
                 dropout: float = 0.0):
        
        super().__init__()

        self.in_features = original.in_features
        self.out_features = original.out_features

        self.rank = rank
        self.scaling = lora_alphaa / rank
        # self.scaling = lora_alphaa / sqrt(rank) <-- LoRACLIPだとスケーリングはこうなっているけど，LoRA公式的には上で正しいはず

        # 元の重みとバイアスを固定して保持する
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None
        
        # LoRA モジュールのパラメータ
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A は Kaiming 初期化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # 固定した通常モデルの forward 処理
        base = F.linear(x, self.weight, self.bias)

        # 学習可能な LoRA モジュール での forward 処理
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)

        return base + lora * self.scaling
    

    def merge(self, alpha: float) -> nn.Linear:
        """
        LoRA モジュールの重みを元の重みに統合して通常の nn.Linear を返す
        W_new = W + alpha * (B @ A) * scaling
        """

        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias is not None,
                           device=self.weight.device,
                           dtype=self.weight.dtype)
        
        delta = (self.lora_B @ self.lora_A) * self.scaling

        merged.weight.data = self.weight.data + alpha * delta
        if self.bias is not None:
            merged.bias.data = self.bias.data.clone()
        return merged
    
    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}")

# ----------------------------------------------------------------------------
# LoRAAttention: ResidualAttentionBlock の attention() を置換するクラス
# ----------------------------------------------------------------------------
class LoRAAttention(nn.Module):
    def __init__(self,
                 original_mha: nn.MultiheadAttention,
                 rank: int,
                 lora_alpha: int,
                 dropout: float = 0.0,
                 ):
        
        super().__init__()
        
        D = original_mha.embed_dim
        n_heads = original_mha.num_heads

        self.embed_dim  = D
        self.num_heads  = n_heads
        self.head_dim   = D // n_heads
        self.scaling    = (D // n_heads) ** -0.5
        self.lora_scale = lora_alpha / rank


        # 元の in_proj_weight / bias は固定して保持する
        self.in_proj_weight = nn.Parameter(
            original_mha.in_proj_weight.data.clone(), requires_grad=False
        )

        if original_mha.in_proj_bias is not None:
            self.in_proj_bias = nn.Parameter(
                original_mha.in_proj_bias.data.clone(), requires_grad=False
            )
        else:
            self.in_proj_bias = None

        # クエリ Q に対する LoRA（A: (r,D), B: (D,r)）
        self.q_lora_A = nn.Parameter(torch.empty(rank, D))
        self.q_lora_B = nn.Parameter(torch.zeros(D, rank))

        # バリュー V に対する　LoRA（A: (r,D), B: (D,r)） 
        self.v_lora_A = nn.Parameter(torch.empty(rank, D))
        self.v_lora_B = nn.Parameter(torch.zeros(D, rank))

        # outproj を LoRALinear に置換
        self.out_proj = LoRALinear(original_mha.out_proj, rank, lora_alpha, dropout)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # lora_A は Kaiming 初期化
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        """
        x: (L, N, D)  — CLIP Transformer の形式 (seq_len, batch, dim)
        """
        
        # 入力の形状を獲得
        L, N, D = x.shape
        D = self.embed_dim
        H = self.num_heads
        Dh = self.head_dim

        # -- Q/K/V を in_proj_weight / in_proj_bias から分割して計算 ----------------------
        W_q = self.in_proj_weight[:D]
        W_k = self.in_proj_weight[D:2*D]
        W_v = self.in_proj_weight[2*D:]

        b_q = self.in_proj_bias[:D]     if self.in_proj_bias is not None else None
        b_k = self.in_proj_bias[D:2*D]  if self.in_proj_bias is not None else None
        b_v = self.in_proj_bias[2*D:]   if self.in_proj_bias is not None else None

        # Q = W_q x + LoRA_q(x)
        q = F.linear(x, W_q, b_q)
        q = q + F.linear(F.linear(self.lora_dropout(x), self.q_lora_A), self.q_lora_B) * self.lora_scale

        # K = W_k x  (LoRA なし)
        k = F.linear(x, W_k, b_k)

        # V = W_v x + LoRA_v(x)
        v = F.linear(x, W_v, b_v)
        v = v + F.linear(F.linear(self.lora_dropout(x), self.v_lora_A), self.v_lora_B) * self.lora_scale

        # -- Scaled Dot-Product Attention ----------------------------------------------------
        # reshape: (L, N, D) → (N*H, L, Dh)
        q = q.reshape(L, N * H, Dh).transpose(0, 1)   # (N*H, L, Dh)
        k = k.reshape(L, N * H, Dh).transpose(0, 1)
        v = v.reshape(L, N * H, Dh).transpose(0, 1)

        attn_w = torch.bmm(q, k.transpose(1, 2)) * self.scaling  # (N*H, L, L)

        if attn_mask is not None:
            attn_w = attn_w + attn_mask  # ブロードキャスト: (N*H, L, L)

        attn_w = attn_w.softmax(dim=-1)

        out = torch.bmm(attn_w, v)                  # (N*H, L, Dh)
        out = out.transpose(0, 1).reshape(L, N, D)  # (L, N, D)

        # -- out_proj (LoRALinear) ------------------------------------------------------------
        out = self.out_proj(out)
        return out

    def merge(self, alpha: float) -> nn.MultiheadAttention:
        """
        LoRA デルタを in_proj_weight と out_proj.weight に統合し、
        通常の nn.MultiheadAttention を返す。
        """
        D = self.embed_dim
        bias = self.in_proj_bias is not None

        merged = nn.MultiheadAttention(
            D, self.num_heads,
            bias=bias,
            device=self.in_proj_weight.device,
            dtype=self.in_proj_weight.dtype,
        )

        new_in_proj = self.in_proj_weight.data.clone()

        # Q delta
        dq = (self.q_lora_B @ self.q_lora_A) * self.lora_scale
        new_in_proj[:D] = new_in_proj[:D] + alpha * dq

        # V delta
        dv = (self.v_lora_B @ self.v_lora_A) * self.lora_scale
        new_in_proj[2*D:] = new_in_proj[2*D:] + alpha * dv

        merged.in_proj_weight = nn.Parameter(new_in_proj)
        if bias:
            merged.in_proj_bias = nn.Parameter(self.in_proj_bias.data.clone())

        # out_proj delta
        out_merged = self.out_proj.merge(alpha)
        merged.out_proj.weight = nn.Parameter(out_merged.weight.data)
        if out_merged.bias is not None:
            merged.out_proj.bias = nn.Parameter(out_merged.bias.data)

        return merged
    


def inject_lora(clip_model: nn.Module,
                rank: int = 16,
                lora_alpha: int = 32,
                dropout: float = 0.1) -> None:
    
    """
    CLIP の Vision Transformer と Text Transformer に LoRA を入れる
    """

    # Visoin Encoder に LoRA を追加
    _inject_transformer(clip_model.visual.transformer, rank, lora_alpha, dropout)

    # Text Encoder に LoRA を追加
    _inject_transformer(clip_model.transformer, rank, lora_alpha, dropout)



def _inject_transformer(transformer: nn.Module,
                        rank: int,
                        lora_alpha: int,
                        dropout: float) -> None:
    
    for block in transformer.resblocks:

        # -- Attention 部分 -------------------------
        lora_attn = LoRAAttention(block.attn, rank, lora_alpha, dropout)
        block.attn = lora_attn

        # attention() メソッドを LoRAAttention.forward に差し替え
        #   (ResidualAttentionBlock の attention() は self.attn を呼ぶ形式を
        #    維持しつつ attn_mask を渡す)
        _patch_attention_method(block)

        # ── MLP c_fc ─────────────────────────────────────────────────
        block.mlp.c_fc   = LoRALinear(block.mlp.c_fc,   rank, lora_alpha, dropout)

        # ── MLP c_proj ────────────────────────────────────────────────
        block.mlp.c_proj = LoRALinear(block.mlp.c_proj, rank, lora_alpha, dropout)




def _patch_attention_method(block) -> None:
    """
    ResidualAttentionBlock.attention() を LoRAAttention.forward() を呼ぶように上書き
    """

    def lora_attention(x: torch.Tensor) -> torch.Tensor:
        mask = block.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return block.attn(x, attn_mask=mask)

    block.attention = lora_attention  # type: ignore[method-assign]

def merge_lora(clip_model: nn.Module, alpha: float = 0.5) -> None:
    """
    LoRA デルタを CLIP backbone に統合する。
    論文 Eq.(2): θ^t = θ^(t-1) + α * θ_LoRA

    呼び出し後、各ブロックは通常の nn.Linear / nn.MultiheadAttention に戻る。
    """
    _merge_transformer(clip_model.visual.transformer, alpha)
    _merge_transformer(clip_model.transformer,        alpha)


def _merge_transformer(transformer: nn.Module, alpha: float) -> None:
    for block in transformer.resblocks:
        # ── Attention マージ ──────────────────────────────────────────
        if isinstance(block.attn, LoRAAttention):
            merged_mha = block.attn.merge(alpha)
            block.attn = merged_mha
            _restore_attention_method(block)

        # ── MLP c_fc マージ ───────────────────────────────────────────
        if isinstance(block.mlp.c_fc, LoRALinear):
            block.mlp.c_fc = block.mlp.c_fc.merge(alpha)

        # ── MLP c_proj マージ ─────────────────────────────────────────
        if isinstance(block.mlp.c_proj, LoRALinear):
            block.mlp.c_proj = block.mlp.c_proj.merge(alpha)


def _restore_attention_method(block) -> None:
    """マージ後、attention() を標準の nn.MultiheadAttention 呼び出しに戻す。"""
    def std_attention(x: torch.Tensor) -> torch.Tensor:
        mask = block.attn_mask
        if mask is not None:
            mask = mask.to(dtype=x.dtype, device=x.device)
        return block.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    block.attention = std_attention  # type: ignore[method-assign]

    
def count_lora_params(model: nn.Module) -> int:
    """学習可能な LoRA パラメータ数を返す。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)