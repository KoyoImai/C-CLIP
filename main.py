import os
import yaml
import argparse


import torch
import torch.nn as nn

from clip.clip import tokenize

from c_clip import CCLIP
from c_clip.dataset import build_vlcl_benchmark
from c_clip.trainer import VLCLTrainer


def parse_args():

    parser = argparse.ArgumentParser(description="C-CLIP: Multimodal Continual Learning")

    # データセット関係
    parser.add_argument("--data_root", type=str, default="/home/kouyou/datasets/")
    parser.add_argument("--task_ids",  type=int, nargs="+", default=None,
                        help="学習するタスクID (省略時: 0〜7 の全8タスクを実行)")

    # model 関係
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP のバックボーン")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA のランク r (デフォルト: 16)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA スケーリング (デフォルト: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--merge_alpha", type=float, default=0.5,
                        help="LoRA 統合係数 α (デフォルト: 0.5)")

    # 学習
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--lr_image",       type=float, default=1e-5)
    parser.add_argument("--lr_image_coco",  type=float, default=5e-7)
    parser.add_argument("--weight_decay",   type=float, default=0.2)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--temperature",    type=float, default=0.07)
    parser.add_argument("--grad_clip",      type=float, default=1.0)

    # その他いろいろ
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="デバイス指定 (デフォルト: CUDA があれば cuda)")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML 設定ファイルパス")
    parser.add_argument("--eval_only", type=str, default=None,
                        help="評価のみ実行: チェックポイントパスを指定")

    return parser.parse_args()


def load_config(args):
    """コマンドライン引数 + YAML ファイルからコンフィグを構築"""
    config = vars(args)

    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        config.update(yaml_cfg)

    # デバイス設定
    if config["device"] is None:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return config


def main():
    args = parse_args()
    config = load_config(args)

    print("\n" + "="*60)
    print("  C-CLIP Configuration")
    print("="*60)

    # -- モデルを構築 --------------------------
    print("[1]: CLIP モデルを構築")
    model = CCLIP(clip_model_name=config["clip_model"],
                  lora_rank=config["lora_rank"],
                  lora_alpha=config["lora_alpha"],
                  lora_dropout=config["lora_dropout"],
                  merge_alpha=config["merge_alpha"],
                  device=config["device"]
                 )

    print(f"  Embed dim : {model.embed_dim}")
    print(f"  学習可能パラメータ: {model.trainable_params():,}")

    # -- DataParallel の設定 -------------------
    # 利用可能な GPU が複数ある場合、DataParallel で並列化する。
    # DataParallel はモデルを各 GPU にコピーし、バッチを分割して処理する。
    # trainer 内では model.module (= _unwrapped) を通じて CCLIP の
    # begin_task / end_task / get_param_groups 等を呼び出す。
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  DataParallel: {n_gpus} GPUs を使用します")
        model = nn.DataParallel(model)
    else:
        print(f"  DataParallel: 無効 (GPU 数={n_gpus})")

    # -- データセット構築 -------------------------
    print("[2]: データセット構築")

    # DataParallel でラップされていても train/val_transform は model.module から取得する
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    train_transform = base_model.train_transform
    val_transform   = base_model.val_transform

    train_tasks = build_vlcl_benchmark(transform=train_transform,
                                       tokenizer=tokenize,
                                       split="train",
                                       cache_dir="/home/kouyou/datasets/HuggingFace"
                                       )
    val_tasks = build_vlcl_benchmark(transform=val_transform,
                                     tokenizer=tokenize,
                                     split="test",
                                     cache_dir="/home/kouyou/datasets/HuggingFace"
                                     )


    # -- トレーナー構築 ---------------------------
    trainer = VLCLTrainer(model=model,
                          train_tasks=train_tasks,
                          val_tasks=val_tasks,
                          config=config,
                          save_dir=config["save_dir"])


    # -- 学習の実行 -------------------------------
    if config.get("eval_only"):
        print(f"\n[評価のみ] チェックポイント: {config['eval_only']}")
        task_id = trainer.load_checkpoint(config["eval_only"])
        trainer.evaluate_all(list(range(task_id + 1)))
    else:
        print("\n[3] 継続学習を開始 ...")
        trainer.train_all_tasks()


if __name__ == "__main__":
    main()