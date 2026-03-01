

"""
 ID  名前        HF リポジトリ                              image列    caption列              split 戦略
 ─── ─────────── ─────────────────────────────────────────  ─────────  ─────────────────────  ──────────────────────
  0  flickr30k   nlphuji/flickr30k                          image      caption (List×5)       predefined (内部split列)
  1  coco        Trickxter/COCO2017-captions                image      caption (str)          predefined (train/validation)
  2  pets        visual-layer/oxford-iiit-pet-vl-enriched   image      caption_enriched (str) predefined (train/test)
  3  lexica      vera365/lexica_dataset                     image      prompt (str)           predefined (train/test)
  4  simpsons    Norod78/simpsons-blip-captions             image      text (str)             random 80/20
  5  wikiart     fusing/wikiart_captions                    image      text (List×4)          random 80/20
  6  kream       hahminlew/kream-product-blip-captions      image      text (str)             random 50/50
  7  sketch      zoheb/sketch-scene                         image      text (str)             random 80/20
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

from dataclasses import dataclass, field

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


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace 設定レジストリ
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HFConfig:
    """
    1タスク分の HuggingFace ロード設定。

    Attributes
    ----------
    repo_id         : HuggingFace リポジトリ ID
    image_field     : PIL Image が格納されている列名
    caption_field   : キャプションが格納されている列名
                      str または List[str] どちらでも可
    caption_is_list : caption_field が List[str] の場合 True
    n_captions      : 1画像あたりのキャプション数 (評価時の Recall@K 計算に使用)
                      caption_is_list=True のテスト時にのみ n_captions > 1 を設定する

    hf_train_split  : train に使う HF split 名
    hf_test_split   : test  に使う HF split 名
                      (同名の場合は train_ratio でランダム分割)

    split_column       : HF split 内部にさらに列でtrain/testを分ける場合の列名 (例: "split")
    split_column_train : split_column が train を示す値
    split_column_test  : split_column が test  を示す値

    train_ratio     : float → 単一HF splitをランダムにtrain/testへ分割する比率
                      None  → predefined split をそのまま使用

    load_kwargs     : load_dataset() に追加で渡す引数
    """
    repo_id:        str
    image_field:    str
    caption_field:  str

    caption_is_list: bool  = False
    n_captions:      int   = 1

    hf_train_split:  str   = "train"
    hf_test_split:   str   = "test"

    split_column:         Optional[str] = None
    split_column_train:   str           = "train"
    split_column_test:    str           = "test"

    train_ratio:     Optional[float] = None

    load_kwargs:     dict = field(default_factory=dict)


# ─── 全8タスクの設定 ─────────────────────────────────────────────────────────

_HF_CONFIG: Dict[str, HFConfig] = {

    # ── Task 0: Flickr30K ────────────────────────────────────────────────────
    # repo   : nlphuji/flickr30k
    # 構造   : HF split は "test" に全31K行が入っている
    #          行の "split" 列 ("train"/"val"/"test") で論文上の分割を管理
    # caption: List[str] (5キャプション)
    # test   : "split"=="test" の約1,000画像 → 1画像×5cap → n_captions=5
    "flickr30k": HFConfig(
        repo_id         = "nlphuji/flickr30k",
        image_field     = "image",
        caption_field   = "caption",
        caption_is_list = True,
        n_captions      = 5,           # テスト時: 1画像×5キャプション評価
        hf_train_split  = "test",      # HF上の split 名 (全データが "test" に入っている)
        hf_test_split   = "test",
        split_column       = "split",  # 行内の "split" 列で論文上のsplitを区別
        split_column_train = "train",
        split_column_test  = "test",
    ),

    # ── Task 1: COCO ─────────────────────────────────────────────────────────
    # repo   : Trickxter/COCO2017-captions
    # 構造   : train(118k行) / validation(5k行)、1行=1キャプション
    # caption: str
    # test   : validation (5,000画像 × 1キャプション) → n_captions=1
    "coco": HFConfig(
        repo_id         = "Trickxter/COCO2017-captions",
        image_field     = "image",
        caption_field   = "caption",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "validation",
    ),

    # ── Task 2: Pets ─────────────────────────────────────────────────────────
    # repo   : visual-layer/oxford-iiit-pet-vl-enriched
    # 構造   : train(3,680) / test(3,669)、predefined split
    # caption: str (BLIP2生成)
    "pets": HFConfig(
        repo_id         = "visual-layer/oxford-iiit-pet-vl-enriched",
        image_field     = "image",
        caption_field   = "caption_enriched",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "test",
    ),

    # ── Task 3: Lexica ───────────────────────────────────────────────────────
    # repo   : vera365/lexica_dataset
    # 構造   : train / test、predefined split
    # caption: str (Stable Diffusion 生成プロンプト)
    "lexica": HFConfig(
        repo_id         = "vera365/lexica_dataset",
        image_field     = "image",
        caption_field   = "prompt",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "test",
    ),

    # ── Task 4: Simpsons ─────────────────────────────────────────────────────
    # repo   : Norod78/simpsons-blip-captions
    # 構造   : train のみ → 80/20 ランダム分割
    # caption: str (BLIP生成)
    "simpsons": HFConfig(
        repo_id         = "Norod78/simpsons-blip-captions",
        image_field     = "image",
        caption_field   = "text",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "train",   # 単一splitのため同名
        train_ratio     = 0.8,
    ),

    # ── Task 5: WikiArt ──────────────────────────────────────────────────────
    # repo   : fusing/wikiart_captions
    # 構造   : train のみ → 80/20 ランダム分割
    # caption: List[str] (4種テンプレートキャプション)
    #          訓練: 全4キャプションを独立サンプルに展開
    #          評価: 先頭キャプションのみ (n_captions=1)
    "wikiart": HFConfig(
        repo_id         = "fusing/wikiart_captions",
        image_field     = "image",
        caption_field   = "text",
        caption_is_list = True,
        n_captions      = 1,         # 評価は 1:1 マッチング
        hf_train_split  = "train",
        hf_test_split   = "train",
        train_ratio     = 0.8,
    ),

    # ── Task 6: Kream ────────────────────────────────────────────────────────
    # repo   : hahminlew/kream-product-blip-captions
    # 構造   : train のみ → 50/50 ランダム分割
    # caption: str (カテゴリ + 商品名 + BLIPキャプション)
    "kream": HFConfig(
        repo_id         = "hahminlew/kream-product-blip-captions",
        image_field     = "image",
        caption_field   = "text",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "train",
        train_ratio     = 0.5,
    ),

    # ── Task 7: Sketch ───────────────────────────────────────────────────────
    # repo   : zoheb/sketch-scene
    # 構造   : train のみ → 80/20 ランダム分割
    # caption: str
    "sketch": HFConfig(
        repo_id         = "zoheb/sketch-scene",
        image_field     = "image",
        caption_field   = "text",
        caption_is_list = False,
        n_captions      = 1,
        hf_train_split  = "train",
        hf_test_split   = "train",
        train_ratio     = 0.8,
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# データセットクラス
# ─────────────────────────────────────────────────────────────────────────────
class VLCLDataset(Dataset):
    """
    HuggingFace Dataset をラップした VLCL 統一データセット。

    全8タスクで同一インターフェースを提供します:
      __len__()           → int
      __getitem__(idx)    → (image_tensor, tokens_tensor, task_id: int)
      collate_fn(batch)   → (images_tensor, tokens_tensor, task_ids: list)
      n_captions_per_image → int  (Recall@K 計算に使用)
      task_id              → int

    Parameters
    ----------
    hf_dataset           : ロード・フィルタ済みの HuggingFace Dataset
    config               : HFConfig (このタスクの設定)
    transform            : 画像前処理 (PIL Image → Tensor)
    tokenizer            : テキストトークナイザ (clip.tokenize 等)
    task_id              : タスク番号 (0〜7)
    n_captions_per_image : Recall@K 計算用の 1画像あたりキャプション数
    is_train             : True=訓練モード / False=評価モード
                           caption_is_list=True のとき展開ルールが変わる
    """

    def __init__(
        self,
        hf_dataset,
        config: HFConfig,
        transform: Callable,
        tokenizer: Callable,
        task_id: int = 0,
        n_captions_per_image: int = 1,
        is_train: bool = True,
    ):
        self.hf_dataset           = hf_dataset
        self.config               = config
        self.transform            = transform
        self.tokenizer            = tokenizer
        self.task_id              = task_id
        self.n_captions_per_image = n_captions_per_image
        self.is_train             = is_train

        # ── フラットインデックス (row_idx, cap_idx) を構築 ────────────────
        # caption_field の列全体を一括取得（行単位ループより高速）
        self._index: List[Tuple[int, int]] = _build_index(
            hf_dataset   = hf_dataset,
            caption_field = config.caption_field,
            caption_is_list = config.caption_is_list,
            n_captions_per_image = n_captions_per_image,
            is_train     = is_train,
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row_idx, cap_idx = self._index[idx]
        row = self.hf_dataset[row_idx]

        # 画像 → Tensor
        img = row[self.config.image_field]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        image = self.transform(img.convert("RGB"))

        # キャプション → Tensor
        raw_cap = row[self.config.caption_field]
        caption = raw_cap[cap_idx] if self.config.caption_is_list else raw_cap
        if not isinstance(caption, str) or not caption.strip():
            caption = "a photo"   # 空キャプションのフォールバック
        tokens = self.tokenizer([caption])[0]

        return image, tokens, self.task_id

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, list]:
        images = torch.stack([b[0] for b in batch])
        tokens = torch.stack([b[1] for b in batch])
        tids   = [b[2] for b in batch]
        return images, tokens, tids


def _build_index(
    hf_dataset,
    caption_field: str,
    caption_is_list: bool,
    n_captions_per_image: int,
    is_train: bool) -> List[Tuple[int, int]]:
    """
    PyTroech Datasetのサンプル番号（idx）を，Hugging Face Datasetの行（row）とキャプション番号（cap）に対応づけるための参照リストを作成する関数

    (row_idx, cap_idx) のフラットなインデックスリストを構築する．

    caption_is_list=False の場合: 全行を (row_idx, 0) でインデックス
    caption_is_list=True  の場合:
        訓練時                   : 全キャプションを独立サンプルに展開
        評価・n_cap > 1 (Flickr) : 全キャプションを展開 (Recall@K 多重正解)
        評価・n_cap = 1          : 先頭キャプションのみ (1画像1サンプル)
    """
    n = len(hf_dataset)

    if not caption_is_list:
        # str キャプション: 1行 = 1サンプル
        return [(i, 0) for i in range(n)]

    # List[str] キャプション: 列全体を一括取得してキャプション数だけ把握
    all_caps = hf_dataset[caption_field]   # List[List[str]]

    if is_train or n_captions_per_image > 1:
        # 訓練 or 複数キャプション評価: 全キャプションを展開
        index = []
        for row_idx, caps in enumerate(all_caps):
            for cap_idx in range(len(caps)):
                index.append((row_idx, cap_idx))
        return index
    else:
        # 評価・1キャプション: 先頭のみ
        return [(i, 0) for i in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# ダミーデータセット (ロード失敗時のプレースホルダー)
# ─────────────────────────────────────────────────────────────────────────────
class DummyDataset(Dataset):
    """データロード失敗時に代替として使用するダミーデータセット。"""

    def __init__(self, transform: Callable, tokenizer: Callable, task_id: int):
        self.transform            = transform
        self.tokenizer            = tokenizer
        self.task_id              = task_id
        self.n_captions_per_image = 1

    def __len__(self) -> int:
        return 100

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img    = self.transform(Image.new("RGB", (224, 224)))
        tokens = self.tokenizer(["a photo"])[0]
        return img, tokens, self.task_id

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, list]:
        return VLCLDataset.collate_fn(batch)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace ローダー (内部関数)
# ─────────────────────────────────────────────────────────────────────────────
def _load_hf_split(
    cfg: HFConfig,
    split: str,
    seed: int = 42,
    cache_dir: Optional[str] = None,
):
    """
    HFConfig に基づいて HuggingFace Dataset をロードし，
    split ("train" / "test") に対応するサブセットを返す．

    Parameters
    ----------
    cfg       : タスクの HFConfig
    split     : "train" または "test"
    seed      : ランダム分割のシード
    cache_dir : HuggingFace データセットのキャッシュ保存先ディレクトリ．
                None の場合は HuggingFace のデフォルト
                (~/.cache/huggingface/datasets) が使用される．
                環境変数 HF_DATASETS_CACHE でも同様に設定可能．

    Returns
    -------
    hf_ds      : HuggingFace Dataset (フィルタ・分割済み)
    is_train   : True=訓練用 / False=評価用
    n_cap      : n_captions_per_image として設定すべき値
    """
    from datasets import load_dataset as hf_load

    is_train = (split == "train")

    # ── Step 1: HF split 名の決定 ────────────────────────────────────────────
    hf_split_name = cfg.hf_train_split if is_train else cfg.hf_test_split
    hf_ds = hf_load(cfg.repo_id, split=hf_split_name,
                    cache_dir=cache_dir, **cfg.load_kwargs)

    # ── Step 2: 内部 split 列によるフィルタリング ─────────────────────────────
    # 実質的には Flickr30k のみを対象とした訓練・テストデータの分割
    if cfg.split_column is not None:
        filter_val = cfg.split_column_train if is_train else cfg.split_column_test
        hf_ds = hf_ds.filter(
            lambda row: row[cfg.split_column] == filter_val,
            desc=f"filter {cfg.split_column}={filter_val}",
        )

    # ── Step 3: train_ratio によるランダム分割 ────────────────────────────────
    elif cfg.train_ratio is not None:
        test_size = 1.0 - cfg.train_ratio
        split_ds  = hf_ds.train_test_split(test_size=test_size, seed=seed)
        hf_ds     = split_ds["train"] if is_train else split_ds["test"]

    # ── Step 4: n_captions_per_image の決定 ──────────────────────────────────
    # テスト時かつ caption_is_list=True かつ n_captions > 1 → 多重キャプション評価
    if not is_train and cfg.caption_is_list and cfg.n_captions > 1:
        n_cap = cfg.n_captions
    else:
        n_cap = 1

    # 確認
    print("hf_ds.column_names: ", hf_ds.column_names)
    print("hf_ds.features: ", hf_ds.features)
    # assert False

    return hf_ds, is_train, n_cap


# ─────────────────────────────────────────────────────────────────────────────
# VLCL ベンチマーク構築 (公開 API)
# ─────────────────────────────────────────────────────────────────────────────
def build_vlcl_benchmark(
    transform: Callable,
    tokenizer: Callable,
    split: str = "train",
    task_ids: Optional[List[int]] = None,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> List[Dataset]:
    """
    VLCL ベンチマークの各タスクに対応するデータセットを構築して返す。

    全タスクを HuggingFace Hub からロードします。
    ロードに失敗したタスクは DummyDataset で代替されます。

    Parameters
    ----------
    transform  : 画像前処理 (train_transform / val_transform)
    tokenizer  : clip.tokenize
    split      : "train" または "test"
    task_ids   : ロードするタスク番号のリスト。None の場合は全タスク (0〜7)
    seed       : ランダム分割の乱数シード (再現性保証)
    cache_dir  : HuggingFace データセットのキャッシュ保存先ディレクトリ。
                 例: "/mnt/ssd/hf_cache" や "./datasets"
                 None の場合は HuggingFace のデフォルト
                 (~/.cache/huggingface/datasets) を使用する。
                 環境変数 HF_DATASETS_CACHE を設定することでも同様に制御可能

    Returns
    -------
    List[Dataset]  — task_ids 順に並んだ VLCLDataset (or DummyDataset) のリスト
    """
    if task_ids is None:
        task_ids = list(range(len(TASK_NAMES)))

    try:
        import datasets as _hf_datasets  # noqa: F401
    except ImportError:
        raise ImportError(
            "`datasets` ライブラリが未インストールです。\n"
            "  pip3 install datasets\n"
            "を実行してください。"
        )

    datasets_list: List[Dataset] = []

    for tid in task_ids:
        name = TASK_NAMES[tid]
        cfg  = _HF_CONFIG[name]

        print(f"  [Task {tid}: {name:<10s}] {split:5s} | "
              f"{cfg.repo_id} をロード中...", flush=True)

        try:

            # hugging face からデータセット情報をダウンロード
            hf_ds, is_train, n_cap = _load_hf_split(cfg, split, seed=seed,
                                                    cache_dir=cache_dir)

            # hugging face 形式のデータセット情報を torch のデータセットとして構築
            dataset = VLCLDataset(
                hf_dataset           = hf_ds,
                config               = cfg,
                transform            = transform,
                tokenizer            = tokenizer,
                task_id              = tid,
                n_captions_per_image = n_cap,
                is_train             = is_train,
            )

            n_images = len(hf_ds)
            n_total  = len(dataset)
            cap_factor = (n_total // n_images) if n_images > 0 else 1
            print(f"  [Task {tid}: {name:<10s}] {split:5s} | "
                  f"{n_images:>7,} 画像 × {cap_factor} cap "
                  f"= {n_total:>8,} サンプル  ✓")

        except Exception as e:
            print(f"  [Task {tid}: {name:<10s}] ロード失敗: {e}")
            print(f"  [Task {tid}: {name:<10s}] → DummyDataset で代替します")
            dataset = DummyDataset(transform, tokenizer, tid)

        datasets_list.append(dataset)

    return datasets_list


# ─────────────────────────────────────────────────────────────────────────────
# 評価: Recall@K
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_recall_at_k(
    image_features: torch.Tensor,
    text_features:  torch.Tensor,
    k_list: List[int] = [1, 5, 10],
    n_captions_per_image: int = 1,
) -> Dict[str, float]:
    """
    Image-to-Text (I2T) / Text-to-Image (T2I) の Recall@K を計算する。
    features は L2 正規化済みを想定。

    n_captions_per_image = 1 の場合 (1:1 マッチング):
        sim = (N, N)
        I2T: image[i] の正解は text[i]
        T2I: text[j]  の正解は image[j]

    n_captions_per_image > 1 の場合 (Flickr30K など多重キャプション評価):
        image_features : (N_img, D)  ※ (N_img × n_cap, D) で渡された場合は自動スライス
        text_features  : (N_img × n_cap, D)
        I2T: image[i] の正解は text[i*n_cap : i*n_cap + n_cap]
        T2I: text[j]  の正解は image[j // n_cap]

    Returns
    -------
    dict : {"I2T_R@1": float, "I2T_R@5": float, ..., "T2I_R@1": float, ...}
    """
    n_cap  = n_captions_per_image
    device = image_features.device

    # ── 1:1 マッチング ────────────────────────────────────────────────────────
    if n_cap == 1:
        N   = len(image_features)
        sim = image_features @ text_features.t()   # (N, N)

        labels  = torch.arange(N, device=device).unsqueeze(1)
        metrics: Dict[str, float] = {}

        # I2T
        for k in k_list:
            topk = sim.topk(min(k, N), dim=1).indices
            metrics[f"I2T_R@{k}"] = (
                (topk == labels).any(dim=1).float().mean().item() * 100.0
            )

        # T2I
        sim_t2i = sim.t()
        for k in k_list:
            topk = sim_t2i.topk(min(k, N), dim=1).indices
            metrics[f"T2I_R@{k}"] = (
                (topk == labels).any(dim=1).float().mean().item() * 100.0
            )

        return metrics

    # ── 多重キャプション評価 (Flickr30K) ─────────────────────────────────────
    N_txt = len(text_features)         # N_img × n_cap
    N_img = N_txt // n_cap

    # image_features が (N_img × n_cap, D) で渡された場合: 先頭1/n_capをスライス
    if len(image_features) == N_txt:
        image_features = image_features[::n_cap]   # (N_img, D)

    assert len(image_features) == N_img, (
        f"image_features の行数 ({len(image_features)}) が "
        f"N_txt ({N_txt}) / n_cap ({n_cap}) = {N_img} と一致しません。"
    )

    sim     = image_features @ text_features.t()   # (N_img, N_txt)
    metrics = {}

    # I2T: image[i] の正解は text[i*n_cap : i*n_cap + n_cap]
    correct_i2t = [
        set(range(i * n_cap, i * n_cap + n_cap)) for i in range(N_img)
    ]
    for k in k_list:
        topk = sim.topk(min(k, N_txt), dim=1).indices.tolist()
        hit  = torch.tensor(
            [any(j in correct_i2t[i] for j in topk[i]) for i in range(N_img)],
            dtype=torch.float,
        )
        metrics[f"I2T_R@{k}"] = hit.mean().item() * 100.0

    # T2I: text[j] の正解は image[j // n_cap]
    sim_t2i   = sim.t()                                       # (N_txt, N_img)
    correct_t = torch.arange(N_img, device=device).repeat_interleave(n_cap)
    for k in k_list:
        topk   = sim_t2i.topk(min(k, N_img), dim=1).indices  # (N_txt, k)
        labels = correct_t.unsqueeze(1)
        metrics[f"T2I_R@{k}"] = (
            (topk == labels).any(dim=1).float().mean().item() * 100.0
        )

    return metrics

