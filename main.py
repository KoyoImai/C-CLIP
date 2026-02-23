

import os
import yaml
import argparse


import torch

from clip.clip import tokenize

from c_clip import CCLIP
from c_clip.dataset import build_vlcl_benchmark


def parse_args():

    parser = argparse.ArgumentParser(description="C-CLIP: Multimodal Continual Learning")

    # データセット関係
    parser.add_argument("--data_root", type=str, default="/home/kouyou/datasets/")
    parser.add_argument("--task_ids",  type=int, nargs="+", default=None,
                        help="学習するタスクID (省略時: 0〜7 全タスク)")

    # model 関係
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
                        help="CLIP バックボーン")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="LoRA のランク r (デフォルト: 16)")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA スケーリング (デフォルト: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--merge_alpha", type=float, default=0.5,
                        help="LoRA 統合係数 α (デフォルト: 0.5)")
    
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
    """
    ・LoRA（rank=16）とテキスト埋め込み層を学習可能にした場合
    学習可能パラメータ: 30,281,216（約28.88M）
    """


    # -- データセット構築 -------------------------
    print("[2]: データセット構築")

    # データ拡張
    train_transform = model.train_transform
    val_transform = model.val_transform

    train_tasks = build_vlcl_benchmark(data_root=config["data_root"],
                                       transform=train_transform,
                                       tokenizer=tokenize,
                                       split="train",
                                       task_ids=config["task_ids"]
                                       )

    val_tasks = None

    assert False
    


if __name__ == "__main__":
    main()



