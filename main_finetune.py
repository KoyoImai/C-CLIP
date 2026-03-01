"""
main_finetune.py — LoRA なし通常ファインチューニング エントリポイント

既存コードからの流用箇所:
  trainer.py  : _unwrapped パターン / evaluate_task / evaluate_all /
                _build_scheduler / _save_checkpoint / load_checkpoint
  main.py     : DataParallel 設定 / データセット構築 / parse_args / load_config

main.py との主な違い:
  モデル    : CCLIP (LoRA) → FinetuneCLIP (LoRA なし)
  損失      : CLIP 損失 + CKC 損失 → CLIP 損失のみ
  タスク管理: begin_task / end_task なし
  保存先    : checkpoints_finetune/finetune_task{N}.pt
"""

import os
import math
import yaml
import argparse
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from clip.clip import tokenize

from c_clip.model_finetune import FinetuneCLIP
from c_clip.dataset import (
    VLCLDataset, TASK_NAMES, build_vlcl_benchmark, compute_recall_at_k,
)
from c_clip.losses import CLIPLoss


# ─────────────────────────────────────────────────────────────────────────────
# FinetuneTrainer
# ─────────────────────────────────────────────────────────────────────────────
class FinetuneTrainer:
    """
    LoRA なし通常ファインチューニング用トレーナー。

    VLCLTrainer との対応:
      VLCLTrainer                  FinetuneTrainer
      ──────────────────────────   ──────────────────────────────────────
      _unwrapped プロパティ         同じパターンで実装 (FinetuneCLIP を返す)
      begin_task() / end_task()    なし (LoRA マージ・注入不要)
      _train_one_epoch (CKC あり)  CLIP 損失のみ / use_ckc 引数なし
      _build_optimizer             projector グループなし
      _build_scheduler             同一ロジックをそのまま流用
      evaluate_task / evaluate_all 同一ロジックをそのまま流用
      _save_checkpoint             "finetune_task{N}.pt" で保存
    """

    def __init__(
        self,
        model:       nn.Module,
        train_tasks: List[VLCLDataset],
        val_tasks:   List[VLCLDataset],
        config:      dict,
        save_dir:    str = "./checkpoints_finetune",
    ):
        self.model       = model
        self.train_tasks = train_tasks
        self.val_tasks   = val_tasks
        self.config      = config
        self.save_dir    = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)

        self.clip_loss = CLIPLoss()

        self.history: Dict[str, list] = {
            "task": [], "epoch": [], "clip_loss": [],
        }

    # ── DataParallel アンラップヘルパー (VLCLTrainer と同一パターン) ──
    @property
    def _unwrapped(self) -> FinetuneCLIP:
        """DataParallel でラップされていても生の FinetuneCLIP を返す。"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # ── 全タスク学習 ───────────────────────────────────────────────────
    def train_all_tasks(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Finetune CLIP (LoRA なし) on VLCL Benchmark")
        print(f"  Tasks     : {len(self.train_tasks)}")
        print(f"  Trainable : {self._unwrapped.trainable_params():,} params")
        print(f"  Device    : {self.device}")
        if isinstance(self.model, nn.DataParallel):
            print(f"  GPU 数    : {len(self.model.device_ids)}")
        print(f"{'='*60}")

        for task_id, train_ds in enumerate(self.train_tasks):
            name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else f"task_{task_id}"
            print(f"\n{'─'*60}")
            print(f"  [Task {task_id}: {name}]  samples={len(train_ds)}")
            print(f"{'─'*60}")

            optimizer = self._build_optimizer(task_id)
            scheduler = self._build_scheduler(optimizer)

            # データセット数がバッチサイズを下回る場合は自動調整
            # (VLCLTrainer と同一ロジック)
            configured_bs = self.config.get("batch_size", 256)
            actual_bs     = min(configured_bs, len(train_ds))
            if actual_bs < configured_bs:
                print(f"  [注意] データセット数 ({len(train_ds)}) < バッチサイズ ({configured_bs})")
                print(f"         バッチサイズを {actual_bs} に自動調整します。")

            loader = DataLoader(
                train_ds,
                batch_size=actual_bs,
                shuffle=True,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=False,
                collate_fn=VLCLDataset.collate_fn,
                drop_last=True,
            )

            n_epochs = self.config.get("epochs", 40)
            for epoch in range(n_epochs):
                self.model.train()
                loss_val = self._train_one_epoch(loader, optimizer)
                scheduler.step()

                self.history["task"].append(task_id)
                self.history["epoch"].append(epoch)
                self.history["clip_loss"].append(loss_val)

                if (epoch + 1) % self.config.get("log_every", 5) == 0:
                    print(
                        f"  Epoch [{epoch+1:3d}/{n_epochs}] "
                        f"clip={loss_val:.4f}"
                    )

            if self.val_tasks:
                print(f"  [Task {task_id}] 評価:")
                self.evaluate_all(list(range(task_id + 1)))

            self._save_checkpoint(task_id)

        print(f"\n{'='*60}")
        print("  全タスクの学習が完了しました。")
        print(f"{'='*60}\n")

    # ── 評価 (VLCLTrainer からそのまま流用) ───────────────────────────
    @torch.no_grad()
    def evaluate_task(self, dataset: VLCLDataset, task_name: str = "") -> dict:
        """1 タスク分の Recall@K を計算する。"""
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.config.get("eval_batch_size", 512),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=False,
            collate_fn=VLCLDataset.collate_fn,
        )

        img_feats, txt_feats = [], []
        for images, tokens, _ in loader:
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            # DataParallel 非対応のため _unwrapped 経由で呼ぶ
            img_feats.append(self._unwrapped.encode_image(images).cpu())
            txt_feats.append(self._unwrapped.encode_text(tokens).cpu())

        img_feats = torch.cat(img_feats)
        txt_feats = torch.cat(txt_feats)

        metrics = compute_recall_at_k(
            img_feats, txt_feats,
            n_captions_per_image=dataset.n_captions_per_image,
        )

        if task_name:
            print(
                f"    [{task_name:<12s}] "
                f"I2T R@1={metrics['I2T_R@1']:5.1f}%  "
                f"T2I R@1={metrics['T2I_R@1']:5.1f}%"
            )
        return metrics

    def evaluate_all(self, seen_task_ids: List[int]) -> dict:
        """見てきた全タスクで評価し平均を表示する。"""
        results = {}
        for tid in seen_task_ids:
            name = TASK_NAMES[tid] if tid < len(TASK_NAMES) else f"task_{tid}"
            results[name] = self.evaluate_task(self.val_tasks[tid], task_name=name)

        avg_i2t = sum(v["I2T_R@1"] for v in results.values()) / len(results)
        avg_t2i = sum(v["T2I_R@1"] for v in results.values()) / len(results)
        print(f"    [{'Average':<12s}] I2T R@1={avg_i2t:5.1f}%  T2I R@1={avg_t2i:5.1f}%")
        return results

    # ── 1 エポック学習 ─────────────────────────────────────────────────
    def _train_one_epoch(self, loader: DataLoader, optimizer: optim.Optimizer) -> float:
        """CLIP 損失のみで 1 エポック学習する。CKC 損失なし。"""
        total = 0.0
        n = 0

        for idx, (images, tokens, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            out = self.model(images, tokens)   # ← use_ckc 引数なし

            # DataParallel では logit_scale が (num_gpus,) になるため mean() でスカラーに戻す
            # (VLCLTrainer と同一の処理)
            logit_scale = out["logit_scale"]
            if logit_scale.dim() > 0:
                logit_scale = logit_scale.mean()

            loss = self.clip_loss(out["image_feat"], out["text_feat"], logit_scale)

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip", 1.0)
            )
            optimizer.step()

            total += loss.item()
            n += 1

            if idx % 20 == 0:
                print(idx, loss.item())

        return total / n

    # ── オプティマイザ (VLCLTrainer から流用、projector グループのみ削除) ──
    def _build_optimizer(self, task_id: int) -> optim.Optimizer:
        """
        Visual / Text で別 LR を設定。
          COCO タスク: text_lr = 80 × image_lr
          その他     : text_lr = 10 × image_lr
        """
        task_name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else ""

        if task_name == "coco":
            lr_img  = self.config.get("lr_image_coco", 5e-7)
            lr_text = lr_img * 80
        else:
            lr_img  = self.config.get("lr_image", 1e-5)
            lr_text = lr_img * 10

        # FinetuneCLIP.get_param_groups は projector なしで lr_image / lr_text の 2 引数
        param_groups = self._unwrapped.get_param_groups(lr_img, lr_text)

        return optim.AdamW(
            param_groups,
            betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.99)),
            weight_decay=self.config.get("weight_decay", 0.2),
        )

    # ── スケジューラ (VLCLTrainer からそのまま流用) ────────────────────
    def _build_scheduler(self, optimizer: optim.Optimizer):
        """線形 Warmup + Cosine Annealing スケジューラ。"""
        n_epochs = self.config.get("epochs", 40)
        warmup   = self.config.get("warmup_epochs", 5)

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(1, warmup)
            progress = (epoch - warmup) / max(1, n_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── チェックポイント (VLCLTrainer からそのまま流用、ファイル名のみ変更) ──
    def _save_checkpoint(self, task_id: int) -> None:
        path = self.save_dir / f"finetune_task{task_id}.pt"
        torch.save({
            "task_id":     task_id,
            "model_state": self._unwrapped.state_dict(),
            "config":      self.config,
            "history":     self.history,
        }, path)
        print(f"  [保存] {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self._unwrapped.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        task_id = ckpt["task_id"]
        print(f"  [ロード] {path}  (task_id={task_id})")
        return task_id


# ─────────────────────────────────────────────────────────────────────────────
# CLI (main.py と同構造)
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune CLIP (LoRA なし) on VLCL Benchmark"
    )

    # データセット (main.py と共通)
    parser.add_argument("--data_root", type=str, default="/home/kouyou/datasets/")
    parser.add_argument("--task_ids",  type=int, nargs="+", default=None,
                        help="学習するタスクID (省略時: 0〜7 の全タスク)")

    # モデル
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP バックボーン")
    parser.add_argument("--freeze_vision", action="store_true",
                        help="Visual Encoder を固定し Text Encoder のみ学習する")
    parser.add_argument("--freeze_text",   action="store_true",
                        help="Text Encoder を固定し Visual Encoder のみ学習する")

    # 学習 (main.py と同じデフォルト値)
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--lr_image",      type=float, default=1e-5)
    parser.add_argument("--lr_image_coco", type=float, default=5e-7)
    parser.add_argument("--weight_decay",  type=float, default=0.2)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--grad_clip",     type=float, default=1.0)

    # その他 (main.py と共通)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir",    type=str, default="./checkpoints_finetune")
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--config",      type=str, default=None,
                        help="YAML 設定ファイルパス")
    parser.add_argument("--eval_only",   type=str, default=None,
                        help="評価のみ実行: チェックポイントパスを指定")

    return parser.parse_args()


def load_config(args):
    """コマンドライン引数 + YAML ファイルからコンフィグを構築 (main.py と同一)"""
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

    print("\n" + "="*60)
    print("  Finetune CLIP (LoRA なし) Configuration")
    print("="*60)

    # ── [1] モデル構築 ─────────────────────────────────────────────────
    print("[1]: CLIP モデルを構築 (LoRA なし)")
    model = FinetuneCLIP(
        clip_model_name = config["clip_model"],
        freeze_vision   = config.get("freeze_vision", False),
        freeze_text     = config.get("freeze_text",   False),
        device          = config["device"],
    )
    print(f"  Embed dim         : {model.embed_dim}")
    print(f"  学習可能パラメータ: {model.trainable_params():,}")

    # ── [2] DataParallel (main.py と同一パターン) ─────────────────────
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  DataParallel: {n_gpus} GPUs を使用します")
        model = nn.DataParallel(model)
    else:
        print(f"  DataParallel: 無効 (GPU 数={n_gpus})")

    # ── [3] データセット構築 (main.py と同一パターン) ─────────────────
    print("[2]: データセット構築")
    base_model      = model.module if isinstance(model, nn.DataParallel) else model
    train_transform = base_model.train_transform
    val_transform   = base_model.val_transform

    train_tasks = build_vlcl_benchmark(
        transform = train_transform,
        tokenizer = tokenize,
        split     = "train",
        task_ids  = config.get("task_ids"),
        cache_dir = "/home/kouyou/datasets/HuggingFace",
    )
    val_tasks = build_vlcl_benchmark(
        transform = val_transform,
        tokenizer = tokenize,
        split     = "test",
        task_ids  = config.get("task_ids"),
        cache_dir = "/home/kouyou/datasets/HuggingFace",
    )

    # ── [4] トレーナー構築 ─────────────────────────────────────────────
    trainer = FinetuneTrainer(
        model       = model,
        train_tasks = train_tasks,
        val_tasks   = val_tasks,
        config      = config,
        save_dir    = config["save_dir"],
    )

    # ── [5] 学習 or 評価 ───────────────────────────────────────────────
    if config.get("eval_only"):
        print(f"\n[評価のみ] チェックポイント: {config['eval_only']}")
        task_id = trainer.load_checkpoint(config["eval_only"])
        trainer.evaluate_all(list(range(task_id + 1)))
    else:
        print("\n[3] ファインチューニングを開始 ...")
        trainer.train_all_tasks()


if __name__ == "__main__":
    main()