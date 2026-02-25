
import math
from pathlib import Path
from typing import List, Optional, Dict


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from c_clip import CCLIP
from c_clip.dataset import VLCLDataset, TASK_NAMES, compute_recall_at_k
from c_clip.losses import CLIPLoss, CKCLoss

class VLCLTrainer:

    def __init__(self,
                 model: CCLIP,
                 train_tasks: List[VLCLDataset],
                 val_tasks: List[VLCLDataset],
                 config: dict,
                 save_dir: str = "./checkpoints"):

        self.model = model
        self.train_tasks = train_tasks
        self.val_tasks = val_tasks
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.model.to(self.device)


        # 損失関数
        self.clip_loss = CLIPLoss()
        self.ckc_loss  = CKCLoss(temperature=config.get("temperature", 0.07))

        # 学習履歴
        self.history: Dict[str, list] = {
            "task": [], "epoch": [], "total_loss": [], "clip_loss": [], "ckc_loss": []
        }

    def train_all_tasks(self) -> None:
        print(f"\n{'='*60}")
        print(f"  C-CLIP Continual Learning on VLCL Benchmark")
        print(f"  Backbone  : ViT-based ZSCL CLIP")
        try:
            print(f"  LoRA rank : {self.model.lora_rank}  alpha: {self.model.lora_alpha}")
            print(f"  Merge α   : {self.model.merge_alpha}")
        except:
            print(f"  LoRA rank : {self.model.module.lora_rank}  alpha: {self.model.module.lora_alpha}")
            print(f"  Merge α   : {self.model.module.merge_alpha}")
        print(f"  Tasks     : {len(self.train_tasks)}")
        try:
            print(f"  Trainable : {self.model.trainable_params():,} params")
        except:
            print(f"  Trainable : {self.model.module.trainable_params():,} params")
        print(f"  Device    : {self.device}")
        print(f"{'='*60}")


        for task_id, train_ds in enumerate(self.train_tasks):
            name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else f"task_{task_id}"
            print(f"\n{'─'*60}")
            print(f"  [Task {task_id}: {name}]  samples={len(train_ds)}")
            print(f"{'─'*60}")

            # 旧モデルを保存
            try:
                self.model.begin_task()
            except:
                self.model.module.begin_task()


            # タスク専用オプティマイザ・スケジューラ
            optimizer = self._build_optimizer(task_id)
            scheduler = self._build_scheduler(optimizer)

            loader = DataLoader(
                train_ds,
                batch_size=self.config.get("batch_size", 256),
                shuffle=True,
                num_workers=self.config.get("num_workers", 4),
                # pin_memory=True,
                pin_memory=False,
                collate_fn=VLCLDataset.collate_fn,
                drop_last=True,
            )

            n_epochs = self.config.get("epochs", 40)
            for epoch in range(n_epochs):
                self.model.train()
                losses = self._train_one_epoch(loader, optimizer, task_id)
                scheduler.step()

                self.history["task"].append(task_id)
                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(losses["total"])
                self.history["clip_loss"].append(losses["clip"])
                self.history["ckc_loss"].append(losses["ckc"])

                if (epoch + 1) % self.config.get("log_every", 5) == 0:
                    print(
                        f"  Epoch [{epoch+1:3d}/{n_epochs}] "
                        f"total={losses['total']:.4f}  "
                        f"clip={losses['clip']:.4f}  "
                        f"ckc={losses['ckc']:.4f}"
                    )

            # LoRA をマージして次タスクへ
            print(f"\n  [Task {task_id}] LoRA マージ (α={self.model.merge_alpha}) ...")
            self.model.end_task()

            # 全既存タスクで評価
            if self.val_tasks:
                print(f"  [Task {task_id}] 評価:")
                self.evaluate_all(list(range(task_id + 1)))

            self._save_checkpoint(task_id)
        
        print(f"\n{'='*60}")
        print("  全タスクの学習が完了しました。")
        print(f"{'='*60}\n")
    

    # ── 評価 ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_task(
        self, dataset: VLCLDataset, task_name: str = ""
    ) -> dict:
        """1 タスク分の Recall@K を計算する。"""
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.config.get("eval_batch_size", 512),
            shuffle=False,
            num_workers=self.config.get("num_workers", 4),
            collate_fn=VLCLDataset.collate_fn,
        )

        img_feats, txt_feats = [], []
        for images, tokens, _ in loader:
            images = images.to(self.device)
            tokens = tokens.to(self.device)
            img_feats.append(self.model.encode_image(images).cpu())
            txt_feats.append(self.model.encode_text(tokens).cpu())

        img_feats = torch.cat(img_feats)
        txt_feats = torch.cat(txt_feats)

        metrics = compute_recall_at_k(img_feats, txt_feats)

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
            m = self.evaluate_task(self.val_tasks[tid], task_name=name)
            results[name] = m

        avg_i2t = sum(v["I2T_R@1"] for v in results.values()) / len(results)
        avg_t2i = sum(v["T2I_R@1"] for v in results.values()) / len(results)
        print(f"    [{'Average':<12s}] I2T R@1={avg_i2t:5.1f}%  T2I R@1={avg_t2i:5.1f}%")
        return results


    # ── 1 エポック学習 ─────────────────────────────────────────────────
    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        task_id: int,
    ) -> dict:
        # 第 1 タスク (task_id=0) は CKC なし、第 2 タスク以降は CKC あり
        use_ckc = (task_id > 0)

        total = clip_sum = ckc_sum = 0.0
        n = 0

        for idx, (images, tokens, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            out = self.model(images, tokens, use_ckc=use_ckc)

            # ── CLIP 損失 ─────────────────────────────────────────
            l_clip = self.clip_loss(
                out["image_feat"], out["text_feat"], out["logit_scale"]
            )
            # print("l_clip: ", l_clip)

            # ── CKC 損失 ─────────────────────────────────────────
            if use_ckc:
                l_ckc = self.ckc_loss(
                    out["image_proj"],    out["text_proj"],
                    out["old_image_feat"], out["old_text_feat"],
                )
            else:
                l_ckc = torch.tensor(0.0, device=self.device)

            loss = l_clip + l_ckc

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip", 1.0)
            )
            optimizer.step()

            total    += loss.item()
            clip_sum += l_clip.item()
            ckc_sum  += l_ckc.item()
            n += 1

            if idx % 20 == 0:
                print(idx, loss.item(), l_clip.item(), l_ckc.item())

                if idx > 100:
                    break

        return {
            "total": total / n,
            "clip":  clip_sum / n,
            "ckc":   ckc_sum / n,
        }


    # ── オプティマイザ / スケジューラ ──────────────────────────────────
    def _build_optimizer(self, task_id: int) -> optim.Optimizer:
        """
        論文の LR 設定:
          - COCO (task 1): text_lr = 80 × image_lr,  image_lr = 5e-7
          - その他       : text_lr = 10 × image_lr,  image_lr = 1e-5

        ZSCL の実装を参照し、Visual と Text で別の LR を設定する。
        """
        task_name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else ""

        if task_name == "coco":
            lr_img  = self.config.get("lr_image_coco", 5e-7)
            lr_text = lr_img * 80
        else:
            lr_img  = self.config.get("lr_image", 1e-5)
            lr_text = lr_img * 10

        lr_proj = lr_img

        try:
            param_groups = self.model.get_param_groups(lr_img, lr_text, lr_proj)
        except:
            param_groups = self.model.module.get_param_groups(lr_img, lr_text, lr_proj)

        return optim.AdamW(
            param_groups,
            betas=(self.config.get("beta1", 0.9), self.config.get("beta2", 0.99)),
            weight_decay=self.config.get("weight_decay", 0.2),
        )

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
        path = self.save_dir / f"cclip_task{task_id}.pt"
        torch.save({
            "task_id":    task_id,
            "model_state": self.model.state_dict(),
            "config":     self.config,
            "history":    self.history,
        }, path)
        print(f"  [保存] {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history = ckpt.get("history", self.history)
        task_id = ckpt["task_id"]
        print(f"  [ロード] {path}  (task_id={task_id})")
        return task_id