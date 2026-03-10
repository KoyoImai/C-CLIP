from collections import OrderedDict

def count_clip_parts(model, trainable_only=False):
    """
    LoRA を除いた CLIP base parameter 数を集計する。
    trainable_only=True にすると、requires_grad=True の base parameter だけ数える。
    """

    # DataParallel 対応
    if hasattr(model, "module"):
        model = model.module

    # LoRACLIP -> .clip を取り出す
    clip = model.clip if hasattr(model, "clip") else model

    result = {
        "text": OrderedDict([
            ("token_embedding", 0),
            ("positional_embedding", 0),
            ("text_projection", 0),
            ("q_linear", 0),
            ("k_linear", 0),
            ("v_linear", 0),
            ("out_projection", 0),
            ("ffn", 0),
        ]),
        "vision": OrderedDict([
            ("class_embedding", 0),
            ("positional_embedding", 0),
            ("visual_projection", 0),
            ("q_linear", 0),
            ("k_linear", 0),
            ("v_linear", 0),
            ("out_projection", 0),
            ("ffn", 0),
        ]),
        "other": OrderedDict([
            ("logit_scale", 0),
        ])
    }

    def add(bucket, key, tensor):
        if tensor is None:
            return
        if trainable_only and (not tensor.requires_grad):
            return
        result[bucket][key] += tensor.numel()

    # -----------------------------
    # text encoder: embeddings / projection
    # -----------------------------
    add("text", "token_embedding", clip.token_embedding.weight)
    add("text", "positional_embedding", clip.positional_embedding)
    add("text", "text_projection", clip.text_projection)

    # -----------------------------
    # vision encoder: embeddings / projection
    # -----------------------------
    add("vision", "class_embedding", clip.visual.class_embedding)
    add("vision", "positional_embedding", clip.visual.positional_embedding)
    add("vision", "visual_projection", clip.visual.proj)

    # -----------------------------
    # other
    # -----------------------------
    add("other", "logit_scale", clip.logit_scale)

    # -----------------------------
    # text transformer blocks
    # -----------------------------
    for block in clip.transformer.resblocks:
        attn = block.attn
        D = attn.embed_dim

        # Q / K / V は in_proj_weight, in_proj_bias の slice で数える
        add("text", "q_linear", attn.in_proj_weight[:D])
        add("text", "k_linear", attn.in_proj_weight[D:2*D])
        add("text", "v_linear", attn.in_proj_weight[2*D:])

        if attn.in_proj_bias is not None:
            add("text", "q_linear", attn.in_proj_bias[:D])
            add("text", "k_linear", attn.in_proj_bias[D:2*D])
            add("text", "v_linear", attn.in_proj_bias[2*D:])

        # out_proj は LoRALinear でも nn.Linear でも .weight/.bias を持つ
        add("text", "out_projection", attn.out_proj.weight)
        add("text", "out_projection", attn.out_proj.bias)

        # FFN = c_fc + c_proj （LoRA は除外して base weight/bias だけ数える）
        add("text", "ffn", block.mlp.c_fc.weight)
        add("text", "ffn", block.mlp.c_fc.bias)
        add("text", "ffn", block.mlp.c_proj.weight)
        add("text", "ffn", block.mlp.c_proj.bias)

    # -----------------------------
    # vision transformer blocks
    # -----------------------------
    for block in clip.visual.transformer.resblocks:
        attn = block.attn
        D = attn.embed_dim

        add("vision", "q_linear", attn.in_proj_weight[:D])
        add("vision", "k_linear", attn.in_proj_weight[D:2*D])
        add("vision", "v_linear", attn.in_proj_weight[2*D:])

        if attn.in_proj_bias is not None:
            add("vision", "q_linear", attn.in_proj_bias[:D])
            add("vision", "k_linear", attn.in_proj_bias[D:2*D])
            add("vision", "v_linear", attn.in_proj_bias[2*D:])

        add("vision", "out_projection", attn.out_proj.weight)
        add("vision", "out_projection", attn.out_proj.bias)

        add("vision", "ffn", block.mlp.c_fc.weight)
        add("vision", "ffn", block.mlp.c_fc.bias)
        add("vision", "ffn", block.mlp.c_proj.weight)
        add("vision", "ffn", block.mlp.c_proj.bias)

    return result


def print_clip_part_summary(summary, title="CLIP parameter summary"):
    print(f"\n=== {title} ===")
    for section in ["text", "vision", "other"]:
        print(f"\n[{section}]")
        subtotal = 0
        for k, v in summary[section].items():
            print(f"{k:20s}: {v:,}")
            subtotal += v
        print(f"{'subtotal':20s}: {subtotal:,}")


# # 使い方
# base_summary = count_clip_parts(model, trainable_only=False)
# print_clip_part_summary(base_summary, title="Base parameters (LoRA excluded)")

# trainable_base_summary = count_clip_parts(model, trainable_only=True)
# print_clip_part_summary(trainable_base_summary, title="Currently trainable base parameters (LoRA excluded)")