

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader



# ------------------------------------------------------------------------------
# タスク定義 ＆ データ割合
# ------------------------------------------------------------------------------
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

_SPLIT_STRATEGY = {
    "flickr30k": "predefined",   # 1K テスト画像 (専用ファイル)
    "coco":      "predefined",   # val2017 (5K)
    "pets":      "predefined",   # 公式 test set
    "lexica":    "predefined",   # 公式 test set
    "simpsons":  0.8,            # 80/20
    "wikiart":   0.8,            # 80/20
    "sketch":    0.8,            # 80/20
    "kream":     0.5,            # 50/50  ← 旧実装は 80/20 で誤り
}

# ------------------------------------------------------------------------------
# データセットクラス
# ------------------------------------------------------------------------------
class ImageCaptionDataset(Dataset):
    """
    汎用 Image-Caption ペアデータセット。

    Args:
        image_root          : 画像ディレクトリのルート
        samples             : [{"image": <相対path>, "caption": <str>}, ...]
        transform           : 画像前処理
        tokenizer           : テキストトークナイザ (clip.tokenize 等)
        task_id             : タスクID
        n_captions_per_image: 1 画像あたりのキャプション数
                              (>1 の場合、samples は画像順にソート済みであること)
    """

    def __init__(
        self,
        image_root: str,
        samples: List[dict],
        transform: Callable,
        tokenizer: Callable,
        task_id: int = 0,
        n_captions_per_image: int = 1,
    ):
        self.image_root           = Path(image_root)
        self.samples              = samples
        self.transform            = transform
        self.tokenizer            = tokenizer
        self.task_id              = task_id
        self.n_captions_per_image = n_captions_per_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item     = self.samples[idx]
        img_path = self.image_root / item["image"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"画像が開けません: {img_path}") from e

        image  = self.transform(img)
        tokens = self.tokenizer([item["caption"]])[0]
        return image, tokens, self.task_id

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        tokens = torch.stack([b[1] for b in batch])
        tids   = [b[2] for b in batch]
        return images, tokens, tids


# ------------------------------------------------------------------------------
# Flickr30K 専用ローダー 
# ------------------------------------------------------------------------------
def _load_flickr30k_samples(data_dir: str, split: str) -> Tuple[str, List[dict], int]:
    """
    Flickr30K を読み込む。

    優先順位:
      1. annotations_train.json / annotations_test.json が存在する場合はそれを使用
      2. test_images.txt が存在する場合: annotations.json から train/test を分離
      3. いずれもない場合: annotations.json を 80/20 分割 (フォールバック)

    test set は 1K 画像 × 5 cap = 5K サンプル。
    """
    data_dir = Path(data_dir)

    # -- パターン１：専用ファイルが存在する ----------------------------
    split_file = data_dir / f"annotation_{split}.json"
    if split_file.exists():
        with open(split_file) as f:
            samples = json.load(f)
        n_cap = 5 if split == "test" else 1
        return str(data_dir), samples, n_cap
    
    # ── パターン 2: test_images.txt + annotations.json ──────────────
    ann_file       = data_dir / "annotations.json"
    test_list_file = data_dir / "test_images.txt"

    if ann_file.exists() and test_list_file.exists():
        with open(ann_file) as f:
            all_samples = json.load(f)
        with open(test_list_file) as f:
            test_images = set(line.strip() for line in f if line.strip())

        train_samples, test_samples = [], []
        # 画像ごとにキャプションを集める
        img_to_caps: Dict[str, List[str]] = defaultdict(list)
        for s in all_samples:
            img_to_caps[s["image"]].append(s["caption"])

        for img_path, caps in img_to_caps.items():
            fname = Path(img_path).name
            if fname in test_images:
                for cap in caps:
                    test_samples.append({"image": img_path, "caption": cap})
            else:
                for cap in caps:
                    train_samples.append({"image": img_path, "caption": cap})

        if split == "train":
            return str(data_dir), train_samples, 1
        else:
            return str(data_dir), test_samples, 5

    # ── パターン 3: フォールバック (80/20) ─────────────────────────
    if ann_file.exists():
        print(f"  [警告] flickr30k: test_images.txt が見つかりません．"
              f"annotations.json を 80/20 分割します．")
        return _load_random_split_samples(str(data_dir), split, train_ratio=0.8)

    raise FileNotFoundError(f"Flickr30K のアノテーションファイルが見つかりません: {data_dir}")


# ------------------------------------------------------------------------------
# COCO 専用ローダー
# ------------------------------------------------------------------------------
def _load_coco_samples(data_dir: str, split: str) -> Tuple[str, List[dict], int]:
    """
    COCO キャプションアノテーション形式を読み込む。
    Returns: (image_root, samples, n_captions_per_image)
      split="train" → captions_train2017.json (全ペア展開)
      split="test"  → captions_val2017.json   (5K 画像 × 5 cap)
    """
    coco_split = "train2017" if split == "train" else "val2017"
    ann_file   = Path(data_dir) / "annotations" / f"captions_{coco_split}.json"

    with open(ann_file) as f:
        data = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in data["images"]}

    # 画像IDごとにキャプションを集める (評価時に順序を保証するため)
    img_captions: Dict[int, List[str]] = defaultdict(list)
    for ann in data["annotations"]:
        img_captions[ann["image_id"]].append(ann["caption"])

    samples = []
    if split == "train":
        # 学習: (image, caption) を全ペア展開
        for img_id, captions in img_captions.items():
            fname = id2file[img_id]
            for cap in captions:
                samples.append({"image": f"images/{coco_split}/{fname}", "caption": cap})
        n_cap = 1  # 学習時は 1:1 ペアとして扱う
    else:
        # 評価: 画像ごとに 5 キャプションを順序通りに並べる
        for img_id, captions in img_captions.items():
            fname = id2file[img_id]
            for cap in captions[:5]:  # 最大 5 キャプション
                samples.append({"image": f"images/{coco_split}/{fname}", "caption": cap})
        n_cap = 5
    
    print("coco: samples[:5] ", samples[:5])

    return str(data_dir), samples, n_cap



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

        # 各データセットの基本情報
        name = TASK_NAMES[tid]
        task_dir = Path(data_root) / name
        strategy = _SPLIT_STRATEGY[name]

        if not task_dir.exists():
            print(f"  [警告] タスク {name} のデータが見つかりません: {task_dir}")
            datasets.append(_make_dummy_dataset(transform, tokenizer, tid))
            continue

        try:
            if name == "flickr30k":
                image_root, samples, n_cap = _load_flickr30k_samples(str(task_dir), split)
            elif name == "coco":
                image_root, samples, n_cap = _load_coco_samples(str(task_dir), split)
            elif strategy == "predefined":
                assert False
            else:
                # float: ランダム分割
                assert False

        except Exception as e:
            print(f"  [エラー] タスク {name} の読み込みに失敗: {e}")
            datasets.append(_make_dummy_dataset(transform, tokenizer, tid))


def _make_dummy_dataset(transform, tokenizer, task_id: int) -> "DummyDataset":
    return DummyDataset(transform, tokenizer, task_id)


class DummyDataset(Dataset):
    """データファイルが存在しない場合のダミー (テスト用)"""

    def __init__(self, transform, tokenizer, task_id):
        self.transform            = transform
        self.tokenizer            = tokenizer
        self.task_id              = task_id
        self.n_captions_per_image = 1
        self._len                 = 100

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        img    = self.transform(Image.new("RGB", (224, 224)))
        tokens = self.tokenizer(["a photo of a dog"])[0]
        return img, tokens, self.task_id

    @staticmethod
    def collate_fn(batch):
        return ImageCaptionDataset.collate_fn(batch)
    


