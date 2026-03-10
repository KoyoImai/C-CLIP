"""
utils.py — 実験再現性のためのユーティリティ

set_seed(seed)
    Python random / NumPy / PyTorch CPU・CUDA の乱数を一括固定する。
    cuDNN の非決定的アルゴリズムも無効化する。
    3 つのエントリポイント (main.py / main_lora.py / main_finetune.py) から
    学習開始前に必ず呼び出すこと。

seed_worker(worker_id)
    DataLoader の worker_init_fn として渡すことで、
    num_workers > 0 のときも各 worker の乱数状態を再現可能にする。

make_loader_generator(seed)
    DataLoader の generator 引数に渡す torch.Generator を生成する。
    shuffle=True のときのインデックス順を固定する。

使い方
------
from c_clip.utils import set_seed, seed_worker, make_loader_generator

set_seed(42)   # main() の最初に呼ぶ

g = make_loader_generator(42)
loader = DataLoader(
    dataset,
    shuffle        = True,
    worker_init_fn = seed_worker,
    generator      = g,
)
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    全乱数生成器のシードを固定する。

    対象:
        - Python 組み込み random
        - NumPy
        - PyTorch CPU / CUDA (全 GPU)
        - cuDNN (deterministic モード有効化・benchmark 無効化)

    Parameters
    ----------
    seed : int
        シード値。論文では 42 を使用。

    Notes
    -----
    cuDNN deterministic モードは再現性を保証するが、
    パフォーマンスがやや低下する場合がある。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN の非決定的アルゴリズムを無効化
    torch.backends.cudnn.deterministic = True
    # 実行ごとに最適アルゴリズムを探索する benchmark モードを無効化
    torch.backends.cudnn.benchmark = False

    # PYTHONHASHSEED: 辞書・集合の順序に影響する hash randomization を固定
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """
    DataLoader の worker_init_fn として使用する。

    num_workers > 0 のとき、各 worker プロセスは親プロセスから fork されるが
    乱数状態は共有されないため、worker ごとに独自のシードを設定する必要がある。

    PyTorch 公式ドキュメント推奨の実装に準拠。

    Parameters
    ----------
    worker_id : int
        DataLoader が自動的に渡す worker のインデックス (0-origin)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader_generator(seed: int) -> torch.Generator:
    """
    DataLoader の generator 引数に渡す torch.Generator を生成する。

    shuffle=True のときのインデックスシャッフル順序を固定する。

    Parameters
    ----------
    seed : int
        シード値

    Returns
    -------
    torch.Generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g