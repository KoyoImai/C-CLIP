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

    # ── DataParallel アンラップヘルパー ────────────────────────────────
    @property
    def _unwrapped(self) -> CCLIP:
        """
        DataParallel でラップされていても生の CCLIP インスタンスを返す。
        begin_task / end_task / get_param_groups 等の呼び出しに使用する。
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def train_all_tasks(self) -> None:
        print(f"\n{'='*60}")
        print(f"  C-CLIP Continual Learning on VLCL Benchmark")
        print(f"  Backbone  : ViT-based CLIP")
        print(f"  LoRA rank : {self._unwrapped.lora_rank}  alpha: {self._unwrapped.lora_alpha}")
        print(f"  Merge α   : {self._unwrapped.merge_alpha}")
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

            # 旧モデルを保存  [Fix 5: try/except を廃止し _unwrapped 経由で呼ぶ]
            self._unwrapped.begin_task()

            # タスク専用オプティマイザ・スケジューラ
            optimizer = self._build_optimizer(task_id)
            scheduler = self._build_scheduler(optimizer)

            # データセット数がバッチサイズを下回る場合はデータセット数に合わせる
            # [Fix 2: actual_bs を DataLoader に渡す]
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

            # LoRA をマージして次タスクへ  [Fix 5: _unwrapped.end_task()]
            print(f"\n  [Task {task_id}] LoRA マージ (α={self._unwrapped.merge_alpha}) ...")
            self._unwrapped.end_task()

            # 全既存タスクで評価
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
    def evaluate_task(
        self, dataset: VLCLDataset, task_name: str = ""
    ) -> dict:
        """
        1 タスク分の Recall@K を計算する。
        [Fix 4] DataParallel は forward() しか並列化しないため、
        encode_image / encode_text は _unwrapped 経由で呼ぶ必要がある。
        """
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
            # encode_image / encode_text は DataParallel 非対応のため _unwrapped から呼ぶ
            img_feats.append(self._unwrapped.encode_image(images).cpu())
            txt_feats.append(self._unwrapped.encode_text(tokens).cpu())

        img_feats = torch.cat(img_feats)
        txt_feats = torch.cat(txt_feats)

        metrics = compute_recall_at_k(
            img_feats, txt_feats,
            n_captions_per_image=dataset.n_captions_per_image
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
            m = self.evaluate_task(self.val_tasks[tid], task_name=name)
            results[name] = m

        avg_i2t = sum(v["I2T_R@1"] for v in results.values()) / len(results)
        avg_t2i = sum(v["T2I_R@1"] for v in results.values()) / len(results)
        print(f"    [{'Average':<12s}] I2T R@1={avg_i2t:5.1f}%  T2I R@1={avg_t2i:5.1f}%")
        return results

    @torch.no_grad()
    def evaluate_merged(self, seen_task_ids: List[int]) -> dict:
        """
        全タスクのテストサンプルを1つのプールに結合し、その中での Recall@K をデータセットごとに計算する（論文 Table 3 準拠）。

        evaluate_all() はタスクごとに独立した (N_task × N_task) の小行列で評価するため、
        他タスクのサンプルが妨害として登場しない。一方この関数は全タスクのサンプルを
        結合した (N_total × N_total) の大行列で評価する。
        各画像が「全データセットの中から」自分のキャプションを検索できるかを測定するため、
        task ID を使用しない推論を前提とした論文値と直接比較可能な数値が得られる。

        評価手順:
          1. 全タスクのテスト特徴量を encode してデータセットごとに保持
          2. 全特徴量を結合し (N_total, D) の1つのプールを作成
          3. (N_total × N_total) 類似度行列を計算
          4. 各データセットのサンプルが属する行範囲を使い、
             N_total 個の候補の中での Recall@K をデータセットごとに算出
          5. 全データセットの平均を表示

        多重キャプションのタスク (Flickr30k, COCO: n_captions_per_image=5) は
        先頭キャプション (cap_idx=0) のみを使用し、全タスクで 1:1 マッチングに統一する。

        Parameters
        ----------
        seen_task_ids : 評価対象のタスク番号リスト

        Returns
        -------
        dict :
            キー = タスク名 (例: "flickr30k")
            値   = {"I2T_R@1": float, "I2T_R@5": float, "I2T_R@10": float,
                    "T2I_R@1": float, "T2I_R@5": float, "T2I_R@10": float}
            加えて "average" キーに全タスク平均を格納
        """
        self.model.eval()
        k_list = [1, 5, 10]

        # ── Step 1: 各タスクの特徴量を encode ────────────────────────────────
        per_task_img: List[torch.Tensor] = []
        per_task_txt: List[torch.Tensor] = []
        task_names:   List[str]          = []

        for tid in seen_task_ids:
            dataset = self.val_tasks[tid]
            n_cap   = dataset.n_captions_per_image
            name    = TASK_NAMES[tid] if tid < len(TASK_NAMES) else f"task_{tid}"

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
                img_feats.append(self._unwrapped.encode_image(images).cpu())
                txt_feats.append(self._unwrapped.encode_text(tokens).cpu())

            img_feats = torch.cat(img_feats)  # (N × n_cap, D) または (N, D)
            txt_feats = torch.cat(txt_feats)

            # n_cap > 1 (Flickr30k / COCO): 先頭キャプション (cap_idx=0) のみ使用
            # データセットのサンプル順: row0_cap0, row0_cap1, ..., row1_cap0, ...
            # → [::n_cap] で cap_idx=0 の行だけ取り出す
            if n_cap > 1:
                img_feats = img_feats[::n_cap]   # (N, D)
                txt_feats = txt_feats[::n_cap]   # (N, D)

            per_task_img.append(img_feats)
            per_task_txt.append(txt_feats)
            task_names.append(name)

        # ── Step 2: 全タスクを結合して1つのプールを作成 ───────────────────────
        merged_img = torch.cat(per_task_img)   # (N_total, D)
        merged_txt = torch.cat(per_task_txt)   # (N_total, D)
        N_total    = len(merged_img)

        # ── Step 3: (N_total × N_total) 類似度行列を計算 ─────────────────────
        # メモリ効率のため CPU 上で計算（N_total が大きい場合はチャンク処理も可）
        sim = merged_img @ merged_txt.t()   # (N_total, N_total): sim[i,j] = img[i]・txt[j]

        # ── Step 4: データセットごとに Recall@K を計算 ───────────────────────
        # 各データセットのサンプルは結合テンソルの [offset, offset+N_k) 行に対応する。
        # image[offset + i] の正解テキストは text[offset + i] (1:1 マッチング)。
        results: dict = {}
        offset = 0

        for name, img_feats in zip(task_names, per_task_img):
            N_k = len(img_feats)
            # このタスクの行スライス
            sim_i2t = sim[offset : offset + N_k]               # (N_k, N_total): I2T
            sim_t2i = sim.t()[offset : offset + N_k]           # (N_k, N_total): T2I

            # 正解ラベル: result[offset+i] の正解は offset+i
            # sim スライス内での列インデックスは offset+i なのでそのまま使う
            labels = torch.arange(offset, offset + N_k).unsqueeze(1)   # (N_k, 1)

            task_metrics: dict = {}
            for k in k_list:
                # I2T: image[i] が正解テキスト (offset+i) を top-k 内に含むか
                topk_i2t = sim_i2t.topk(min(k, N_total), dim=1).indices   # (N_k, k)
                task_metrics[f"I2T_R@{k}"] = (
                    (topk_i2t == labels).any(dim=1).float().mean().item() * 100.0
                )
                # T2I: text[i] が正解画像 (offset+i) を top-k 内に含むか
                topk_t2i = sim_t2i.topk(min(k, N_total), dim=1).indices   # (N_k, k)
                task_metrics[f"T2I_R@{k}"] = (
                    (topk_t2i == labels).any(dim=1).float().mean().item() * 100.0
                )

            results[name] = task_metrics
            offset += N_k

        # ── Step 5: 全タスク平均 ─────────────────────────────────────────────
        avg: dict = {}
        for k in k_list:
            avg[f"I2T_R@{k}"] = sum(v[f"I2T_R@{k}"] for v in results.values()) / len(results)
            avg[f"T2I_R@{k}"] = sum(v[f"T2I_R@{k}"] for v in results.values()) / len(results)
        results["average"] = avg

        # ── 結果表示 (論文 Table 3 レイアウト) ───────────────────────────────
        col_w = 10
        header = f"  {'Dataset':<14s}" + "".join(
            f"{'I2T@' + str(k):>{col_w}s}{'T2I@' + str(k):>{col_w}s}" for k in k_list
        )
        sep = "  " + "─" * (len(header) - 2)
        print(f"\n{sep}")
        print(f"  Merged Retrieval  (全 {len(seen_task_ids)} タスク結合, N_total={N_total:,})")
        print(sep)
        print(header)
        print(sep)
        for name in task_names:
            m = results[name]
            row = f"  {name:<14s}"
            for k in k_list:
                row += f"{m[f'I2T_R@{k}']:>{col_w}.1f}{m[f'T2I_R@{k}']:>{col_w}.1f}"
            print(row)
        print(sep)
        m = results["average"]
        row = f"  {'average':<14s}"
        for k in k_list:
            row += f"{m[f'I2T_R@{k}']:>{col_w}.1f}{m[f'T2I_R@{k}']:>{col_w}.1f}"
        print(row)
        print(f"{sep}\n")

        return results

    # ── 1 エポック学習 ─────────────────────────────────────────────────
    def _train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        task_id: int,
    ) -> dict:
        # 論文通り Task 0 を含む全タスクで CKC 損失を使用する。
        # Task 0 の学習開始前に begin_task() が呼ばれており、
        # old_clip には事前学習済み CLIP がコピーされているため
        # use_ckc=True でも old_clip=None にはならない。
        use_ckc = True

        total = clip_sum = ckc_sum = 0.0
        n = 0

        for idx, (images, tokens, _) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            tokens = tokens.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            out = self.model(images, tokens, use_ckc=use_ckc)

            # ── CLIP 損失 ─────────────────────────────────────────
            # DataParallel では各 GPU が logit_scale スカラーを返すため
            # gather 後に shape (num_gpus,) になる → .mean() でスカラーに戻す
            logit_scale = out["logit_scale"]
            if logit_scale.dim() > 0:
                logit_scale = logit_scale.mean()

            l_clip = self.clip_loss(
                out["image_feat"], out["text_feat"], logit_scale
            )

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

        return {
            "total": total / n,
            "clip":  clip_sum / n,
            "ckc":   ckc_sum / n,
        }

    # ── オプティマイザ / スケジューラ ──────────────────────────────────
    def _build_optimizer(self, task_id: int) -> optim.Optimizer:
        """
        論文 Appendix A.2 の LR 設定:
          - flickr30k (task 0): lr_image = 1e-5,  text = 10x image  -> text = 1e-4
          - coco      (task 1): lr_image = 5e-7,  text = 80x image  -> text = 4e-5
          - その他    (task 2+): lr_image = 3e-5, text = 10x image  -> text = 3e-4
        """
        task_name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else ""

        # [Fix 3: flickr30k / coco / others の 3 段階 LR]
        if task_name == "coco":
            lr_img  = self.config.get("lr_image_coco", 5e-7)
            lr_text = lr_img * 80
        elif task_name == "flickr30k":
            lr_img  = self.config.get("lr_image", 1e-5)
            lr_text = lr_img * 10
        else:
            # Pet, Lexica, Simpsons, WikiArt, Kream, Sketch
            lr_img  = self.config.get("lr_image_other", 3e-5)
            lr_text = lr_img * 10

        lr_proj = lr_img

        # [Fix 5: _unwrapped 経由で get_param_groups を呼ぶ]
        param_groups = self._unwrapped.get_param_groups(lr_img, lr_text, lr_proj)

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
        # [Fix 6: DataParallel でラップされている場合、_unwrapped の state_dict を保存]
        # これにより "module." プレフィックスなしで保存され、
        # ロード時に DataParallel の有無に関わらず使用できる。
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