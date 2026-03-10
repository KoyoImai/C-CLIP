"""
main.py — C-CLIP エントリポイント

LoRA 適用範囲と学習可能パラメータ範囲をコマンドラインから指定できる。

■ LoRA 適用範囲の指定 (--lora_targets)
  有効トークン: q / k / v / out / ffn
  例:
    --lora_targets q,v          → Q + V のみ（LoRA 原論文準拠）
    --lora_targets q,ffn        → Q + FFN（論文 Table6 パラメータ数一致）
    --lora_targets q,v,out,ffn  → Q + V + out + FFN（元実装と同じ）

■ 学習可能パラメータ範囲の指定 (--trainable_params)
  有効キー:
    token_embedding      : テキストのトークン埋め込み    (25,296,896)
    text_pos_embedding   : テキストの位置埋め込み        (    39,424)
    text_projection      : テキストの最終出力線形層      (   262,144)
    text_ln              : テキストエンコーダの全 LN     (    25,600)
    class_embedding      : 画像の CLS トークン           (       768)
    visual_pos_embedding : 画像の位置埋め込み            (   151,296)
    visual_proj          : 画像の最終出力線形層          (   393,216)
    visual_ln            : 画像エンコーダの全 LN         (    39,936)
    visual_conv1         : 画像の patch embedding        (   589,824)
    logit_scale          : 温度パラメータ                (         1)
  例:
    --trainable_params token_embedding,logit_scale
    --trainable_params token_embedding,text_pos_embedding,class_embedding,visual_pos_embedding,logit_scale
    --trainable_params token_embedding,text_pos_embedding,text_projection,text_ln,class_embedding,visual_pos_embedding,visual_proj,visual_ln,visual_conv1,logit_scale
"""

import os
import yaml
import argparse

import torch
import torch.nn as nn

from clip.clip import tokenize

from c_clip import CCLIP
from c_clip.lora import LoRAConfig
from c_clip.model import TrainableConfig
from c_clip.dataset import build_vlcl_benchmark
from c_clip.trainer import VLCLTrainer
from c_clip.utils import set_seed, seed_worker, make_loader_generator


def parse_args():
    parser = argparse.ArgumentParser(description="C-CLIP: Multimodal Continual Learning")

    # ── データセット ──────────────────────────────────────────────
    parser.add_argument("--data_root", type=str, default="/home/kouyou/datasets/")
    parser.add_argument("--task_ids",  type=int, nargs="+", default=None,
                        help="学習するタスクID (省略時: 0〜7 の全8タスクを実行)")
    parser.add_argument(
        "--wikiart_captions_path", type=str,
        default="./wikiart_captions_out_blip2/wikiart_blip_captions.parquet",
        help="WikiArt タスク (task 5) に使用する BLIP2 生成キャプションの parquet パス。",
    )

    # ── モデル ────────────────────────────────────────────────────
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP のバックボーン")
    parser.add_argument("--lora_rank",    type=int,   default=16,
                        help="LoRA のランク r (デフォルト: 16)")
    parser.add_argument("--lora_alpha",   type=int,   default=None,
                        help="LoRA スケーリング (デフォルト: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--merge_alpha",  type=float, default=0.5,
                        help="LoRA 統合係数 α (デフォルト: 0.5)")

    # ── LoRA 適用範囲の指定（新規追加） ──────────────────────────
    parser.add_argument(
        "--lora_targets", type=str,
        default="q,v,out,ffn",
        help=(
            "LoRA を適用するレイヤーをカンマ区切りで指定。\n"
            "  有効値: q / k / v / out / ffn\n"
            "  例: --lora_targets q,v           # Q+V のみ（LoRA 原論文準拠）\n"
            "      --lora_targets q,ffn         # Q+FFN（論文 Table6 一致）\n"
            "      --lora_targets q,v,out,ffn   # 元実装と同じ（デフォルト）"
        ),
    )

    # ── 学習可能パラメータ範囲の指定 ─────────────────────────────
    parser.add_argument(
        "--trainable_params", type=str,
        default="token_embedding,text_pos_embedding,class_embedding,visual_pos_embedding,logit_scale",
        help=(
            "LoRA 非適用パラメータのうち学習可能にするものをカンマ区切りで指定。\n"
            "  有効値: token_embedding / text_pos_embedding / text_projection / text_ln /\n"
            "          class_embedding / visual_pos_embedding / visual_proj / visual_ln / visual_conv1 / logit_scale\n"
            "  例: --trainable_params token_embedding,logit_scale\n"
            "      --trainable_params token_embedding,text_pos_embedding,class_embedding,"
            "visual_pos_embedding,logit_scale  （デフォルト）"
        ),
    )

    # ── 学習ハイパーパラメータ ────────────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--batch_size",     type=int,   default=1024)
    parser.add_argument("--lr_image",       type=float, default=1e-5)
    parser.add_argument("--lr_image_coco",  type=float, default=5e-7)
    parser.add_argument("--lr_image_other", type=float, default=3e-5,
                        help="Pet/Lexica/Simpsons/WikiArt/Kream/Sketch の lr_image (論文 Appendix A.2: 3e-5)")
    parser.add_argument("--weight_decay",   type=float, default=0.2)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--temperature",    type=float, default=0.07)
    parser.add_argument("--grad_clip",      type=float, default=1.0)

    # ── その他 ────────────────────────────────────────────────────
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir",    type=str, default="./checkpoints")
    parser.add_argument("--device",      type=str, default=None,
                        help="デバイス指定 (デフォルト: CUDA があれば cuda)")
    parser.add_argument("--config",      type=str, default=None,
                        help="YAML 設定ファイルパス")
    parser.add_argument("--eval_only",   type=str, default=None,
                        help="評価のみ実行: チェックポイントパスを指定")
    parser.add_argument("--seed",        type=int, default=42,
                        help="乱数シード (Python/NumPy/PyTorch/cuDNN を一括固定)")

    return parser.parse_args()


def load_config(args):
    """コマンドライン引数 + YAML ファイルからコンフィグを構築。"""
    config = vars(args)
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        config.update(yaml_cfg)
    if config["device"] is None:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def main():
    args   = parse_args()
    config = load_config(args)

    # ── 乱数シードの固定（再現性確保） ───────────────────────────
    set_seed(config["seed"])

    # ── Config オブジェクトを生成 ────────────────────────────────
    lora_cfg      = LoRAConfig.from_string(config["lora_targets"])
    trainable_cfg = TrainableConfig.from_string(config["trainable_params"])

    print("\n" + "=" * 60)
    print("  C-CLIP Configuration")
    print("=" * 60)
    print(f"  clip_model   : {config['clip_model']}")
    print(f"  lora_rank    : {config['lora_rank']}")
    print(f"  lora_alpha   : {config['lora_alpha'] or config['lora_rank'] * 2}")
    print(f"  lora_targets : {config['lora_targets']}")
    print(f"    → {lora_cfg.summary()}")
    print(f"  trainable_params : {config['trainable_params']}")
    print(f"  seed         : {config['seed']}")
    print("=" * 60)

    # ── モデル構築 ───────────────────────────────────────────────
    print("\n[1] CLIP モデルを構築")
    model = CCLIP(
        clip_model_name = config["clip_model"],
        lora_rank       = config["lora_rank"],
        lora_alpha      = config["lora_alpha"],
        lora_dropout    = config["lora_dropout"],
        merge_alpha     = config["merge_alpha"],
        device          = config["device"],
        lora_cfg        = lora_cfg,
        trainable_cfg   = trainable_cfg,
    )
    model.print_config()

    # ── DataParallel / SyncBatchNorm の設定 ──────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  SyncBatchNorm: BatchNorm1d → SyncBatchNorm に変換します")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(f"  DataParallel: {n_gpus} GPUs を使用します")
        model = nn.DataParallel(model)
    else:
        print(f"  DataParallel: 無効 (GPU 数={n_gpus})、BatchNorm1d をそのまま使用")

    # ── データセット構築 ─────────────────────────────────────────
    print("\n[2] データセット構築")
    base_model      = model.module if isinstance(model, nn.DataParallel) else model
    train_transform = base_model.train_transform
    val_transform   = base_model.val_transform

    train_tasks = build_vlcl_benchmark(
        transform=train_transform,
        tokenizer=tokenize,
        split="train",
        cache_dir="/home/kouyou/datasets/HuggingFace",
        wikiart_captions_path=args.wikiart_captions_path,
    )
    val_tasks = build_vlcl_benchmark(
        transform=val_transform,
        tokenizer=tokenize,
        split="test",
        cache_dir="/home/kouyou/datasets/HuggingFace",
        wikiart_captions_path=args.wikiart_captions_path,
    )

    # ── トレーナー構築 ───────────────────────────────────────────
    trainer = VLCLTrainer(
        model=model,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        config=config,
        save_dir=config["save_dir"],
    )

    # ── 学習 or 評価 ─────────────────────────────────────────────
    if config.get("eval_only"):
        print(f"\n[評価のみ] チェックポイント: {config['eval_only']}")
        task_id = trainer.load_checkpoint(config["eval_only"])
        seen = list(range(task_id + 1))
        print("\n── タスク別評価 ──")
        trainer.evaluate_all(seen)
        print("\n── マージ評価 (論文準拠) ──")
        trainer.evaluate_merged(seen)
    else:
        print("\n[3] 継続学習を開始 ...")
        trainer.train_all_tasks()


if __name__ == "__main__":
    main()