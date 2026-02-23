

import os
import json
import random
from pathlib import Path
from typing import List, Optional, Callable, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader



# ─────────────────────────────────────────────────────────────────────────────
# タスク定義
# ─────────────────────────────────────────────────────────────────────────────
TASK_NAMES: List[str] = [
    "flickr30k",   # 0
    "coco",        # 1
    "pets",        # 2
    "lexica",      # 3
    "simpsons",    # 4
    "wikiart",     # 5
    "kream",       # 6
    "sketch",      # 7
]


class ImageCaptionDataset(Dataset):

    def __init__(self,
                 image_root: str,
                 samples: List[dict],
                 transform: Callable,
                 tokenizer: Callable,
                 task_id: int = 0):
        
        self.image_root = Path(image_root)
        self.samplesas = samples
        self.transform = transform
        self.tokenizer = tokenizer
        self.task_id = task_id
    
    def __len__(self) -> int:
        return len(self.samples)


def build_vlcl_benchmark(data_root: str,
                         transform: Callable,
                         tokenizer: Callable,
                         split: str = "train",
                         task_ids: Optional[List[int]] = None) -> List[ImageCaptionDataset]:
    
    """
    論文「https://proceedings.iclr.cc/paper_files/paper/2025/file/72fb9ab442fc60b7ae5d53bf6b478273-Paper-Conference.pdf」
    で提案された VLCL ベンチマークを作成する処理を実行

    Args:
        data_root  : 各タスクのデータが入った親ディレクトリ
        transform  : 画像前処理 (train_transform / val_transform)
        tokenizer  : clip.tokenize
        split      : "train" or "test"
        task_ids   : None の場合は全タスク (0〜7)

    Returns:
        List[ImageCaptionDataset]  — task_ids 順に並んだデータセット
    """

    if task_ids is None:
        task_ids = list(range(len(TASK_NAMES)))

    datasets = []
    for tid in task_ids:
        name = TASK_NAMES[tid]
        task_dir = Path(data_root) / name
        print(task_dir)

        if not task_dir.exists():
            print(f"  [警告] タスク {name} のデータが見つかりません: {task_dir}")
            # ダミーデータセット (テスト・デバッグ用)
            datasets.append(_make_dummy_dataset(transform, tokenizer, tid))
            continue
    

    assert False

def _make_dummy_dataset(transform, tokenizer, task_id: int) -> "DummyDataset":
    """データが存在しない場合のプレースホルダー"""
    return DummyDataset(transform, tokenizer, task_id)


class DummyDataset(Dataset):
    """データファイルが存在しない場合のダミー (テスト用)"""

    def __init__(self, transform, tokenizer, task_id):
        self.transform  = transform
        self.tokenizer  = tokenizer
        self.task_id    = task_id
        self._len       = 100  # ダミーサイズ

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img    = self.transform(Image.new("RGB", (224, 224)))
        tokens = self.tokenizer(["a photo of a dog"])[0]
        return img, tokens, self.task_id

    @staticmethod
    def collate_fn(batch):
        return ImageCaptionDataset.collate_fn(batch)


