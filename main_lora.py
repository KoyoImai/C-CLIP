"""
main_lora.py — CLIP-LoRA 学習エントリポイント

継続学習ベースラインとして、LoRA アダプタのみで CLIP を各タスクに順次適応させる。
CKC 損失・知識蒸留・LoRA マージは使用しない (CLIP 損失のみ)。

既存コードとの対応:
  main_finetune.py → FinetuneTrainer    (全パラメータ更新)
  main_lora.py     → LoRATrainer        (LoRA パラメータのみ更新)  ← このファイル
  main.py          → VLCLTrainer (C-CLIP) (LoRA + CKC + マージ)

使い方
------
# 全 8 タスクを順次学習
python main_lora.py

# 評価のみ (タスク 5 まで学習済みのチェックポイントを指定)
python main_lora.py --eval_only ./checkpoints_lora/lora_task5.pt

# LoRA ランクを変更
python main_lora.py --lora_rank 8 --lora_alpha 16

# バックボーン・バッチサイズ変更
python main_lora.py --clip_model ViT-B/32 --batch_size 512

チェックポイント形式
-------------------
{
    "task_id":     int,
    "model_state": OrderedDict,   # LoRACLIP.state_dict()
    "config":      dict,
    "history":     dict,
}
保存先: ./checkpoints_lora/lora_task{N}.pt
"""

from __future__ import annotations

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

from c_clip.model_lora import LoRACLIP
from c_clip.dataset import (
    VLCLDataset, TASK_NAMES, build_vlcl_benchmark, compute_recall_at_k,
)
from c_clip.losses import CLIPLoss


# ─────────────────────────────────────────────────────────────────────────────
# LoRATrainer
# ─────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    """
    CLIP-LoRA 継続学習トレーナー。

    VLCLTrainer / FinetuneTrainer との対応:

      VLCLTrainer (C-CLIP)          LoRATrainer
      ──────────────────────────    ──────────────────────────────────────
      begin_task() / end_task()     なし (LoRA マージ・再注入しない)
      CLIP 損失 + CKC 損失           CLIP 損失のみ
      _build_optimizer (3 LR 群)    _build_optimizer (2 LR 群, proj なし)
      evaluate_merged()             同一ロジックをそのまま流用
      _save_checkpoint: cclip_*.pt  lora_task{N}.pt

      FinetuneTrainer               LoRATrainer
      ──────────────────────────    ──────────────────────────────────────
      FinetuneCLIP (全パラメータ)   LoRACLIP (LoRA パラメータのみ)
      get_param_groups(2 引数)      get_param_groups(2 引数) — 同一シグネチャ
      lora_task{N}.pt               lora_task{N}.pt
    """

    def __init__(
        self,
        model:       nn.Module,
        train_tasks: List[VLCLDataset],
        val_tasks:   List[VLCLDataset],
        config:      dict,
        save_dir:    str = "./checkpoints_lora",
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
    def _unwrapped(self) -> LoRACLIP:
        """DataParallel でラップされていても生の LoRACLIP を返す。"""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    # ── 全タスク学習 ───────────────────────────────────────────────────
    def train_all_tasks(self) -> None:
        print(f"\n{'='*60}")
        print(f"  CLIP-LoRA Continual Learning on VLCL Benchmark")
        print(f"  LoRA rank : {self._unwrapped.lora_rank}  "
              f"alpha: {self._unwrapped.lora_alpha}")
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

            # タスク専用オプティマイザ・スケジューラ
            optimizer = self._build_optimizer(task_id)
            scheduler = self._build_scheduler(optimizer)

            # データセット数がバッチサイズを下回る場合は自動調整 (既存コードと同一)
            configured_bs = self.config.get("batch_size", 256)
            actual_bs     = min(configured_bs, len(train_ds))
            if actual_bs < configured_bs:
                print(f"  [注意] データセット数 ({len(train_ds)}) < "
                      f"バッチサイズ ({configured_bs})")
                print(f"         バッチサイズを {actual_bs} に自動調整します。")

            loader = DataLoader(
                train_ds,
                batch_size  = actual_bs,
                shuffle     = True,
                num_workers = self.config.get("num_workers", 4),
                pin_memory  = False,
                collate_fn  = VLCLDataset.collate_fn,
                drop_last   = True,
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

            # タスク終了後: LoRA をマージせず、そのまま次タスクへ
            # (C-CLIP と異なり LoRA は蓄積し続ける)

            if self.val_tasks:
                print(f"  [Task {task_id}] タスク別評価:")
                self.evaluate_all(list(range(task_id + 1)))
                print(f"  [Task {task_id}] マージ評価 (論文準拠):")
                self.evaluate_merged(list(range(task_id + 1)))

            self._save_checkpoint(task_id)

        print(f"\n{'='*60}")
        print("  全タスクの学習が完了しました。")
        print(f"{'='*60}\n")

    # ── 評価 ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_task(self, dataset: VLCLDataset, task_name: str = "") -> dict:
        """1 タスク分の Recall@K を計算する。"""
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size  = self.config.get("eval_batch_size", 512),
            shuffle     = False,
            num_workers = self.config.get("num_workers", 4),
            pin_memory  = False,
            collate_fn  = VLCLDataset.collate_fn,
        )

        img_feats, txt_feats = [], []
        for images, tokens, _ in loader:
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            # encode_image / encode_text は DataParallel 非対応のため _unwrapped から呼ぶ
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

    @torch.no_grad()
    def evaluate_merged(self, seen_task_ids: List[int]) -> dict:
        """
        全タスクのテストサンプルを1つのプールに結合し、Recall@K を計算する。
        VLCLTrainer.evaluate_merged() と同一ロジック (論文 Table 3 準拠)。
        """
        self.model.eval()
        k_list = [1, 5, 10]

        per_task_img: List[torch.Tensor] = []
        per_task_txt: List[torch.Tensor] = []
        task_names:   List[str]          = []

        for tid in seen_task_ids:
            dataset = self.val_tasks[tid]
            n_cap   = dataset.n_captions_per_image
            name    = TASK_NAMES[tid] if tid < len(TASK_NAMES) else f"task_{tid}"

            loader = DataLoader(
                dataset,
                batch_size  = self.config.get("eval_batch_size", 512),
                shuffle     = False,
                num_workers = self.config.get("num_workers", 4),
                pin_memory  = False,
                collate_fn  = VLCLDataset.collate_fn,
            )

            img_feats, txt_feats = [], []
            for images, tokens, _ in loader:
                images = images.to(self.device)
                tokens = tokens.to(self.device)
                img_feats.append(self._unwrapped.encode_image(images).cpu())
                txt_feats.append(self._unwrapped.encode_text(tokens).cpu())

            img_feats = torch.cat(img_feats)
            txt_feats = torch.cat(txt_feats)

            # n_cap > 1 の場合: 先頭キャプションのみ使用 (1:1 マッチングに統一)
            if n_cap > 1:
                img_feats = img_feats[::n_cap]
                txt_feats = txt_feats[::n_cap]

            per_task_img.append(img_feats)
            per_task_txt.append(txt_feats)
            task_names.append(name)

        merged_img = torch.cat(per_task_img)
        merged_txt = torch.cat(per_task_txt)
        N_total    = len(merged_img)

        sim = merged_img @ merged_txt.t()

        results: dict = {}
        offset = 0

        for name, img_feats in zip(task_names, per_task_img):
            N_k      = len(img_feats)
            sim_i2t  = sim[offset : offset + N_k]
            sim_t2i  = sim.t()[offset : offset + N_k]
            labels   = torch.arange(offset, offset + N_k).unsqueeze(1)

            task_metrics: dict = {}
            for k in k_list:
                topk_i2t = sim_i2t.topk(min(k, N_total), dim=1).indices
                task_metrics[f"I2T_R@{k}"] = (
                    (topk_i2t == labels).any(dim=1).float().mean().item() * 100.0
                )
                topk_t2i = sim_t2i.topk(min(k, N_total), dim=1).indices
                task_metrics[f"T2I_R@{k}"] = (
                    (topk_t2i == labels).any(dim=1).float().mean().item() * 100.0
                )

            results[name] = task_metrics
            offset += N_k

        avg: dict = {}
        for k in k_list:
            avg[f"I2T_R@{k}"] = (
                sum(v[f"I2T_R@{k}"] for v in results.values()) / len(results)
            )
            avg[f"T2I_R@{k}"] = (
                sum(v[f"T2I_R@{k}"] for v in results.values()) / len(results)
            )
        results["average"] = avg

        # 結果表示 (VLCLTrainer と同一フォーマット)
        col_w  = 10
        header = f"  {'Dataset':<14s}" + "".join(
            f"{'I2T@' + str(k):>{col_w}s}{'T2I@' + str(k):>{col_w}s}"
            for k in k_list
        )
        sep = "  " + "─" * (len(header) - 2)
        print(f"\n{sep}")
        print(f"  Merged Retrieval  (全 {len(seen_task_ids)} タスク結合, "
              f"N_total={N_total:,})")
        print(sep)
        print(header)
        print(sep)
        for name in task_names:
            m   = results[name]
            row = f"  {name:<14s}"
            for k in k_list:
                row += f"{m[f'I2T_R@{k}']:>{col_w}.1f}{m[f'T2I_R@{k}']:>{col_w}.1f}"
            print(row)
        print(sep)
        m   = results["average"]
        row = f"  {'average':<14s}"
        for k in k_list:
            row += f"{m[f'I2T_R@{k}']:>{col_w}.1f}{m[f'T2I_R@{k}']:>{col_w}.1f}"
        print(row)
        print(f"{sep}\n")

        return results

    # ── 1 エポック学習 ─────────────────────────────────────────────────
    def _train_one_epoch(
        self, loader: DataLoader, optimizer: optim.Optimizer
    ) -> float:
        """CLIP 損失のみで 1 エポック学習する。CKC 損失なし。"""
        total = 0.0
        n     = 0

        for idx, (images, tokens, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            out = self.model(images, tokens)

            # DataParallel では logit_scale が (num_gpus,) になるため mean() でスカラー化
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
            n     += 1

            if idx % 20 == 0:
                print(idx, loss.item())

        return total / n

    # ── オプティマイザ (VLCLTrainer から流用、3 段階 LR を維持) ──────
    def _build_optimizer(self, task_id: int) -> optim.Optimizer:
        """
        論文 Appendix A.2 の LR 設定 (C-CLIP と同一):
          flickr30k : lr_image = 1e-5,  text = 10 × image
          coco      : lr_image = 5e-7,  text = 80 × image
          その他    : lr_image = 3e-5,  text = 10 × image

        projector グループは不要のため 2 グループのみ。
        """
        task_name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else ""

        if task_name == "coco":
            lr_img  = self.config.get("lr_image_coco", 5e-7)
            lr_text = lr_img * 80
        elif task_name == "flickr30k":
            lr_img  = self.config.get("lr_image", 1e-5)
            lr_text = lr_img * 10
        else:
            lr_img  = self.config.get("lr_image_other", 3e-5)
            lr_text = lr_img * 10

        param_groups = self._unwrapped.get_param_groups(lr_img, lr_text)

        return optim.AdamW(
            param_groups,
            betas        = (self.config.get("beta1", 0.9),
                            self.config.get("beta2", 0.99)),
            weight_decay = self.config.get("weight_decay", 0.2),
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

    # ── チェックポイント ───────────────────────────────────────────────
    def _save_checkpoint(self, task_id: int) -> None:
        path = self.save_dir / f"lora_task{task_id}.pt"
        torch.save({
            "task_id":     task_id,
            "model_state": self._unwrapped.state_dict(),
            "config":      self.config,
            "history":     self.history,
        }, path)
        print(f"  [保存] {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt    = torch.load(path, map_location=self.device)
        state   = ckpt.get("model_state", ckpt)
        missing, unexpected = self._unwrapped.load_state_dict(state, strict=False)
        if missing:
            print(f"  警告: missing keys = {len(missing)} 個")
        if unexpected:
            print(f"  警告: unexpected keys = {len(unexpected)} 個")
        self.history = ckpt.get("history", self.history)
        task_id = ckpt["task_id"]
        print(f"  [ロード] {path}  (task_id={task_id})")
        return task_id


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CLIP-LoRA Continual Learning on VLCL Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── データセット (main.py / main_finetune.py と共通) ──────────────
    parser.add_argument("--data_root", type=str,
                        default="/home/kouyou/datasets/")
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="学習するタスクID (省略時: 0〜7 の全8タスク)")
    parser.add_argument(
        "--wikiart_captions_path", type=str,
        default="./wikiart_captions_out_blip2/wikiart_blip_captions.parquet",
        help="WikiArt タスク (task 5) 用 BLIP2 生成キャプションの parquet パス",
    )

    # ── モデル ────────────────────────────────────────────────────────
    parser.add_argument("--clip_model", type=str, default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP バックボーン (main.py と揃えること)")
    parser.add_argument("--lora_rank",    type=int,   default=16,
                        help="LoRA ランク r")
    parser.add_argument("--lora_alpha",   type=int,   default=None,
                        help="LoRA スケーリング係数 (デフォルト: 2 * rank)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA ドロップアウト率")

    # ── 学習 (main.py と同じデフォルト値) ────────────────────────────
    parser.add_argument("--epochs",         type=int,   default=40)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--lr_image",       type=float, default=1e-5,
                        help="flickr30k の Visual LoRA LR")
    parser.add_argument("--lr_image_coco",  type=float, default=5e-7,
                        help="COCO の Visual LoRA LR")
    parser.add_argument("--lr_image_other", type=float, default=3e-5,
                        help="その他タスクの Visual LoRA LR")
    parser.add_argument("--weight_decay",   type=float, default=0.2)
    parser.add_argument("--warmup_epochs",  type=int,   default=5)
    parser.add_argument("--grad_clip",      type=float, default=1.0)

    # ── その他 ────────────────────────────────────────────────────────
    parser.add_argument("--num_workers", type=int,  default=8)
    parser.add_argument("--save_dir",    type=str,  default="./checkpoints_lora")
    parser.add_argument("--device",      type=str,  default=None,
                        help="デバイス (デフォルト: CUDA があれば cuda)")
    parser.add_argument("--config",      type=str,  default=None,
                        help="YAML 設定ファイルパス")
    parser.add_argument("--eval_only",   type=str,  default=None,
                        help="評価のみ実行: チェックポイントパスを指定")

    return parser.parse_args()


def load_config(args) -> dict:
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

    print("\n" + "="*60)
    print("  CLIP-LoRA Configuration")
    print("="*60)

    # ── [1] モデル構築 ─────────────────────────────────────────────────
    print("[1]: CLIP-LoRA モデルを構築")
    model = LoRACLIP(
        clip_model_name = config["clip_model"],
        lora_rank       = config["lora_rank"],
        lora_alpha      = config.get("lora_alpha"),
        lora_dropout    = config["lora_dropout"],
        device          = config["device"],
    )
    print(f"  Embed dim         : {model.embed_dim}")
    print(f"  学習可能パラメータ: {model.trainable_params():,}")

    # ── [2] DataParallel (main.py と同一パターン) ─────────────────────
    # LoRACLIP は Projector (BatchNorm1d) を持たないため SyncBatchNorm 変換不要
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
        transform              = train_transform,
        tokenizer              = tokenize,
        split                  = "train",
        task_ids               = config.get("task_ids"),
        cache_dir              = "/home/kouyou/datasets/HuggingFace",
        wikiart_captions_path  = args.wikiart_captions_path,
    )
    val_tasks = build_vlcl_benchmark(
        transform              = val_transform,
        tokenizer              = tokenize,
        split                  = "test",
        task_ids               = config.get("task_ids"),
        cache_dir              = "/home/kouyou/datasets/HuggingFace",
        wikiart_captions_path  = args.wikiart_captions_path,
    )

    # ── [4] トレーナー構築 ─────────────────────────────────────────────
    trainer = LoRATrainer(
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
        seen    = list(range(task_id + 1))
        print("\n── タスク別評価 ──")
        trainer.evaluate_all(seen)
        print("\n── マージ評価 (論文準拠) ──")
        trainer.evaluate_merged(seen)
    else:
        print("\n[3] CLIP-LoRA 継続学習を開始 ...")
        trainer.train_all_tasks()


if __name__ == "__main__":
    main()