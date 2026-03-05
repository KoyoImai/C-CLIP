"""
plot_wikiart_captions.py
─────────────────────────
generate_wikiart_captions.py で生成した Parquet ファイルと
fusing/wikiart_captions の画像・メタデータを matplotlib で
グリッド表示し、PNG として保存するスクリプト。

【使い方】
  # ランダム12件を表示・保存
  python plot_wikiart_captions.py \
      --captions_path ./wikiart_captions_out/wikiart_blip_captions.parquet

  # 件数・シード指定
  python plot_wikiart_captions.py \
      --captions_path ./wikiart_captions_out/wikiart_blip_captions.parquet \
      --n_samples 20 --seed 123

  # アートスタイルで絞り込む
  python plot_wikiart_captions.py \
      --captions_path ./wikiart_captions_out/wikiart_blip_captions.parquet \
      --style "Impressionism"

  # image_idx を直接指定（カンマ区切り）
  python plot_wikiart_captions.py \
      --captions_path ./wikiart_captions_out/wikiart_blip_captions.parquet \
      --indices 0,5,42,100

  # 保存先を指定
  python plot_wikiart_captions.py \
      --captions_path ./wikiart_captions_out/wikiart_blip_captions.parquet \
      --output_path ./review.png

【出力レイアウト（1セルあたり）】
  ┌─────────────────────────────┐
  │         画像 (PIL)           │
  ├─────────────────────────────┤
  │ ✅ BLIP キャプション         │
  │ ─────────────────────────── │
  │ 🏷 Style / Genre / Artist   │
  │ ⚠️ 元テンプレートキャプション │
  └─────────────────────────────┘

【依存ライブラリ】
  pip install matplotlib datasets pandas pyarrow Pillow
"""

from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")           # GUI なし環境でも動作するよう Agg を使用
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# データ取得ユーティリティ
# ─────────────────────────────────────────────────────────────────────────────

def load_data(captions_path: str, cache_dir: Optional[str]):
    """Parquet と HuggingFace データセットをロードして返す。"""
    from datasets import load_dataset as hf_load

    captions_path = Path(captions_path)
    if not captions_path.exists():
        raise FileNotFoundError(
            f"キャプションファイルが見つかりません: {captions_path}\n"
            "先に generate_wikiart_captions.py を実行してください。"
        )

    print(f"キャプションファイルを読み込み中: {captions_path}")
    df = pd.read_parquet(captions_path)
    print(f"  件数: {len(df):,}")

    print("fusing/wikiart_captions を読み込み中...")
    hf_ds = hf_load("fusing/wikiart_captions", split="train", cache_dir=cache_dir)
    print(f"  HF 件数: {len(hf_ds):,}  /  カラム: {hf_ds.column_names}")

    return df, hf_ds


def detect_columns(hf_ds) -> dict:
    """スタイル・ジャンル・作者の列名を自動検出する。"""
    cols = hf_ds.column_names
    return {
        "style":  next((c for c in ["style",  "art_style", "Style"]  if c in cols), None),
        "genre":  next((c for c in ["genre",  "Genre"]               if c in cols), None),
        "artist": next((c for c in ["artist", "Artist", "author"]    if c in cols), None),
    }


def get_caption(df: pd.DataFrame, image_idx: int) -> str:
    """生成キャプションを返す。未生成の場合は代替文字列。"""
    rows = df[df["image_idx"] == image_idx]
    return str(rows.iloc[0]["caption"]) if not rows.empty else "(未生成)"


def get_meta(row: dict, col_map: dict) -> str:
    """スタイル / ジャンル / 作者のメタデータ文字列を組み立てる。"""
    parts = []
    for label, key in [("Style", "style"), ("Genre", "genre"), ("Artist", "artist")]:
        col = col_map[key]
        if col and row.get(col):
            parts.append(f"{label}: {row[col]}")
    return " | ".join(parts) if parts else ""


def get_orig_text(row: dict) -> str:
    """元データセットのテキスト列（テンプレートキャプション）を返す。"""
    raw = row.get("text", "")
    if isinstance(raw, list):
        # 4 テンプレートのうち最初の 1 件だけ表示（長すぎるため）
        return raw[0] if raw else ""
    return str(raw)


def sample_indices(
    df: pd.DataFrame,
    hf_ds,
    col_map: dict,
    n: int,
    seed: int,
    style_filter: Optional[str],
    genre_filter: Optional[str],
) -> List[int]:
    """フィルタ条件に合う image_idx をランダムサンプリングする。"""
    candidates = df["image_idx"].tolist()

    if style_filter or genre_filter:
        filtered = []
        for idx in candidates:
            row = hf_ds[int(idx)]
            if style_filter:
                if row.get(col_map["style"], "") != style_filter:
                    continue
            if genre_filter:
                if row.get(col_map["genre"], "") != genre_filter:
                    continue
            filtered.append(idx)
        candidates = filtered
        print(f"フィルタ後の候補件数: {len(candidates):,}")

    if not candidates:
        raise ValueError("フィルタ条件に合う画像が見つかりませんでした。")

    random.seed(seed)
    n = min(n, len(candidates))
    return random.sample(candidates, n)


# ─────────────────────────────────────────────────────────────────────────────
# グリッド描画
# ─────────────────────────────────────────────────────────────────────────────

# テキスト折り返しの幅（文字数）
WRAP_WIDTH_CAPTION = 40
WRAP_WIDTH_META    = 50

# フォントサイズ
FS_CAPTION  = 7.5
FS_META     = 6.5
FS_ORIG     = 6.0
FS_IDX      = 7.0

# セルごとの高さ比（画像部分 : テキスト部分）
IMG_RATIO   = 3
TEXT_RATIO  = 2


def wrap(text: str, width: int) -> str:
    """テキストを指定幅で折り返す。"""
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def plot_grid(
    indices: List[int],
    df: pd.DataFrame,
    hf_ds,
    col_map: dict,
    ncols: int,
    output_path: str,
    show: bool,
    dpi: int,
):
    """
    indices に対応する画像 + キャプションをグリッド描画して保存する。

    レイアウト:
      各セルを 2 行に分割
        行 0 (IMG_RATIO): PIL 画像
        行 1 (TEXT_RATIO): 生成キャプション / メタデータ / 元テキスト
    """
    n = len(indices)
    nrows = (n + ncols - 1) // ncols    # 切り上げ

    # Figure サイズ: 1 セルあたり幅 3.5 inch, 高さ 5.5 inch
    cell_w = 3.5
    cell_h = 5.5
    fig_w  = cell_w * ncols
    fig_h  = cell_h * nrows

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1e1e2e")

    # グリッド全体の GridSpec
    outer = gridspec.GridSpec(
        nrows, ncols,
        figure=fig,
        hspace=0.06,
        wspace=0.04,
        left=0.01, right=0.99,
        top=0.97,  bottom=0.01,
    )

    for pos, idx in enumerate(indices):
        row_pos = pos // ncols
        col_pos = pos % ncols

        # 各セルを画像用 / テキスト用に縦分割
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[row_pos, col_pos],
            height_ratios=[IMG_RATIO, TEXT_RATIO],
            hspace=0.0,
        )

        # ── 画像 ─────────────────────────────────────────────────────────────
        ax_img = fig.add_subplot(inner[0])
        ax_img.set_facecolor("#1e1e2e")

        hf_row = hf_ds[int(idx)]
        pil_img = hf_row["image"]
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(pil_img)
        ax_img.imshow(pil_img.convert("RGB"))
        ax_img.axis("off")

        # 左上に image_idx を表示
        ax_img.text(
            0.02, 0.97, f"#{idx}",
            transform=ax_img.transAxes,
            color="white", fontsize=FS_IDX,
            va="top", ha="left",
            bbox=dict(facecolor="black", alpha=0.55, pad=1.5, edgecolor="none"),
        )

        # ── テキスト ──────────────────────────────────────────────────────────
        ax_txt = fig.add_subplot(inner[1])
        ax_txt.set_facecolor("#13131f")
        ax_txt.axis("off")

        generated = get_caption(df, idx)
        meta      = get_meta(hf_row, col_map)
        orig      = get_orig_text(hf_row)

        # 3 種のテキストを縦に並べる
        # y 座標は 0 (下) 〜 1 (上) の相対座標
        y = 0.97

        # ① 生成キャプション（白・強調）
        cap_lines = wrap(f"✅ {generated}", WRAP_WIDTH_CAPTION)
        ax_txt.text(
            0.03, y, cap_lines,
            transform=ax_txt.transAxes,
            color="#a6e3a1",          # 緑系（Catppuccin Mocha "green"）
            fontsize=FS_CAPTION,
            va="top", ha="left",
            linespacing=1.3,
        )
        # 行数からオフセットを計算（概算）
        n_lines_cap = cap_lines.count("\n") + 1
        y -= (n_lines_cap * FS_CAPTION + 3) / (ax_txt.get_window_extent().height or 120)

        # 区切り線
        ax_txt.axhline(y=max(y + 0.04, 0.02), color="#45475a", linewidth=0.5)
        y -= 0.07

        # ② メタデータ（スタイル / ジャンル / 作者）
        if meta:
            meta_lines = wrap(f"🏷 {meta}", WRAP_WIDTH_META)
            ax_txt.text(
                0.03, max(y, 0.02), meta_lines,
                transform=ax_txt.transAxes,
                color="#89b4fa",      # 青系（"blue"）
                fontsize=FS_META,
                va="top", ha="left",
                linespacing=1.2,
            )
            n_lines_meta = meta_lines.count("\n") + 1
            y -= (n_lines_meta * FS_META + 3) / (ax_txt.get_window_extent().height or 120)
            y -= 0.05

        # ③ 元テンプレートキャプション（グレー・比較用）
        if orig:
            orig_lines = wrap(f"⚠ {orig}", WRAP_WIDTH_META)
            ax_txt.text(
                0.03, max(y, 0.01), orig_lines,
                transform=ax_txt.transAxes,
                color="#6c7086",      # グレー（"overlay1"）
                fontsize=FS_ORIG,
                va="top", ha="left",
                linespacing=1.2,
                style="italic",
            )

    # 空白セルを非表示
    for pos in range(n, nrows * ncols):
        row_pos = pos // ncols
        col_pos = pos % ncols
        ax = fig.add_subplot(outer[row_pos, col_pos])
        ax.set_visible(False)

    # タイトル
    filter_note = ""
    if args.style:
        filter_note += f"  Style={args.style}"
    if args.genre:
        filter_note += f"  Genre={args.genre}"
    fig.suptitle(
        f"WikiArt — BLIP Caption Review  (n={n}, seed={args.seed}){filter_note}",
        color="white", fontsize=10, y=0.995,
    )

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n保存しました: {output_path}  (dpi={dpi})")

    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WikiArt 生成キャプション グリッド表示 (matplotlib)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--captions_path",
        default="./wikiart_captions_out/wikiart_blip_captions.parquet",
        help="generate_wikiart_captions.py が出力した Parquet ファイルのパス",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="HuggingFace キャッシュディレクトリ",
    )

    # サンプリング
    parser.add_argument("--n_samples", type=int, default=12, help="表示件数")
    parser.add_argument("--seed",      type=int, default=42,  help="乱数シード")
    parser.add_argument(
        "--indices",
        default=None,
        help="image_idx をカンマ区切りで直接指定 (例: 0,5,42). 指定時は n_samples/seed を無視",
    )

    # フィルタ
    parser.add_argument("--style", default=None, help="アートスタイルで絞り込む (例: Impressionism)")
    parser.add_argument("--genre", default=None, help="ジャンルで絞り込む (例: landscape)")

    # レイアウト
    parser.add_argument("--ncols",   type=int, default=4,   help="グリッドの列数")
    parser.add_argument("--dpi",     type=int, default=150, help="出力 DPI")

    # 出力
    parser.add_argument(
        "--output_path",
        default="./wikiart_caption_review.png",
        help="保存先ファイルパス",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="保存後に plt.show() で画面表示する（GUI 環境のみ）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df, hf_ds = load_data(args.captions_path, args.cache_dir)
    col_map   = detect_columns(hf_ds)
    print(f"列マッピング: {col_map}")

    # インデックスの決定
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
        print(f"指定インデックス: {indices}")
    else:
        indices = sample_indices(
            df, hf_ds, col_map,
            n=args.n_samples,
            seed=args.seed,
            style_filter=args.style,
            genre_filter=args.genre,
        )
        print(f"サンプリング完了: {len(indices)} 件 (seed={args.seed})")

    plot_grid(
        indices=indices,
        df=df,
        hf_ds=hf_ds,
        col_map=col_map,
        ncols=args.ncols,
        output_path=args.output_path,
        show=args.show,
        dpi=args.dpi,
    )