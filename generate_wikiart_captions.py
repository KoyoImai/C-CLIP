"""
generate_wikiart_captions.py
────────────────────────────
fusing/wikiart_captions の各画像に対して BLIP / BLIP2 で
自然言語キャプションを生成し、Parquet 形式で保存するスクリプト。

【出力ファイル構成】
  <output_dir>/
    wikiart_blip_captions.parquet  ← 最終成果物 (image_idx, caption)
    progress.json                  ← 中断再開用チェックポイント
    chunks/
      chunk_0000.parquet           ← 途中結果を 1000件ずつ保存
      chunk_0001.parquet
      ...

【使い方】
  # BLIP (デフォルト、軽量)
  python generate_wikiart_captions.py --output_dir ./wikiart_captions_out

  # BLIP2 (高品質、VRAM 多め)
  python generate_wikiart_captions.py --model blip2 --output_dir ./wikiart_captions_out

  # バッチサイズ・GPU 指定
  python generate_wikiart_captions.py --batch_size 16 --device cuda:0

  # 途中から再開 (progress.json が存在すれば自動的に再開)
  python generate_wikiart_captions.py --output_dir ./wikiart_captions_out

  # サンプル確認（最初の 100件のみ）
  python generate_wikiart_captions.py --max_samples 100 --output_dir ./wikiart_captions_test

【依存ライブラリ】
  pip install transformers datasets torch Pillow pandas pyarrow tqdm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# モデルロード
# ─────────────────────────────────────────────────────────────────────────────

def load_blip(device: str):
    """
    Salesforce/blip-image-captioning-large をロードする。

    BLIP は軽量で消費 VRAM が小さい (約 2GB)。
    無条件キャプション (unconditional) で生成する。
    """
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_name = "Salesforce/blip-image-captioning-large"
    print(f"  BLIP モデルをロード中: {model_name}")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
    ).to(device)
    model.eval()
    return processor, model, "blip"


def load_blip2(device: str):
    """
    Salesforce/blip2-opt-2.7b をロードする。

    BLIP2 は高品質だが消費 VRAM が大きい (約 12GB)。

    multi-GPU 環境での device_map について:
      "auto" はモデルを複数 GPU に分散配置するが、
      language_model が複数デバイスに跨ると generate() 内で
      テンソルのデバイス不一致エラーが発生することがある。
      ここでは VRAM が足りる場合は単一 GPU に固定し、
      足りない場合のみ "balanced" で分散させる。
    """
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    model_name = "Salesforce/blip2-opt-2.7b"
    print(f"  BLIP2 モデルをロード中: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)

    # ── device_map の決定 ─────────────────────────────────────────────────
    # BLIP2-OPT-2.7B の重み量は fp16 で約 5.4GB。
    # 単一 GPU に十分な空き VRAM があれば cuda:0 に固定する。
    # 足りなければ "balanced" で GPU 間均等分散させる。
    if torch.cuda.is_available():
        n_gpus     = torch.cuda.device_count()
        free_vram0 = torch.cuda.mem_get_info(0)[0] / 1024**3   # GB
        print(f"  検出 GPU 数: {n_gpus}  /  cuda:0 空き VRAM: {free_vram0:.1f} GB")

        if free_vram0 >= 6.0:
            # 単一 GPU に載る → デバイス固定
            device_map = {"": 0}
            print("  device_map: cuda:0 に固定")
        elif n_gpus >= 2:
            # 複数 GPU で均等分散
            device_map = "balanced"
            print("  device_map: balanced (複数 GPU に分散)")
        else:
            # VRAM 不足だが GPU は 1 枚 → CPU オフロード
            device_map = "auto"
            print("  device_map: auto (CPU オフロード)")
    else:
        device_map = None   # CPU のみ
        print("  CPU モードで実行します")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    # モデルの実際の配置デバイスを確認してログ出力
    if hasattr(model, "hf_device_map"):
        print(f"  hf_device_map: {model.hf_device_map}")

    return processor, model, "blip2"


# ─────────────────────────────────────────────────────────────────────────────
# キャプション生成 (バッチ処理)
# ─────────────────────────────────────────────────────────────────────────────

def _get_first_device(model) -> torch.device:
    """
    device_map で分散配置されたモデルの「先頭モジュールが乗っているデバイス」
    を返す。pixel_values など最初に渡す入力の転送先として使用する。
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ── キャプション後処理 ────────────────────────────────────────────────────────

# BLIP が無条件生成時に出力しやすいノイズプレフィックス一覧。
# 条件付き生成プロンプト ("there is") を使っても残存する場合があるため後処理で除去する。
_NOISE_PREFIXES = [
    # BLIPの学習データに混入したノイズトークン（"a photograph of" の崩れ）
    "arafed ",
    "arafe ",
    "araf ",
    # 過度に絵画であることを明示するプレフィックス
    "a painting of ",
    "painting of ",
    "an oil painting of ",
    "a oil painting of ",
    "a watercolor painting of ",
    "a drawing of ",
    "a sketch of ",
    "a picture of ",
    "an image of ",
    "a photo of ",
    "a photograph of ",
    "a black and white photo of ",
    "a black and white painting of ",
    "a black and white drawing of ",
]

def clean_caption(text: str) -> str:
    """
    BLIP が出力するノイズプレフィックスを除去し、先頭を大文字にして返す。

    例:
      "arafed woman sitting on a bench"
        → "Woman sitting on a bench"
      "a painting of a landscape with mountains"
        → "A landscape with mountains"
      "painting of water lilies in a pond"
        → "Water lilies in a pond"

    プレフィックス除去後にテキストが空になる場合は元の文字列を返す。
    """
    lower = text.lower()
    for prefix in _NOISE_PREFIXES:
        if lower.startswith(prefix):
            cleaned = text[len(prefix):]
            # 除去後が空または極端に短い場合はスキップ
            if len(cleaned.strip()) < 4:
                continue
            # 先頭を大文字に・末尾に「.」を付与
            result = cleaned[0].upper() + cleaned[1:] if cleaned else text
            return result if result.endswith(".") else result + "."
        
    # プレフィックスに該当しない場合も先頭を大文字に揃えて「.」を付与
    if not text:
        return text
    result = text[0].upper() + text[1:]
    return result if result.endswith(".") else result + "."


def generate_captions_batch(
    images: List[Image.Image],
    processor,
    model,
    model_type: str,
    device: str,
    max_new_tokens: int = 60,
    min_new_tokens: int = 20,
    prompt: str = "there is",
    num_beams: int = 5,
    length_penalty: float = 1.2,
    repetition_penalty: float = 1.5,
) -> List[str]:
    """
    PIL Image のリストをバッチで処理してキャプション文字列のリストを返す。

    Parameters
    ----------
    images        : PIL Image のリスト (RGB に変換済みを想定)
    processor     : BLIP / BLIP2 のプロセッサ
    model         : BLIP / BLIP2 のモデル
    model_type    : "blip" または "blip2"
    device        : "cuda" / "cpu" 等 (BLIP 用。BLIP2 は model の配置から自動取得)
    max_new_tokens     : 生成トークン数の上限
    min_new_tokens     : 生成トークン数の下限 (短すぎるキャプションを防ぐ)
    prompt             : 条件付き生成のプロンプト文字列 (BLIP のみ使用)
    num_beams          : ビームサーチのビーム数 (大きいほど品質向上・低速)
    length_penalty     : >1.0 で長いシーケンスを優遇し、詳細な記述を促す
    repetition_penalty : >1.0 で同じ語句の繰り返しを抑制する

    Returns
    -------
    captions : 後処理済みキャプション文字列のリスト (len == len(images))

    Notes: プロンプトによるキャプション品質の向上
    ──────────────────────────────────────────────
    BLIP を無条件生成 (unconditional) で使うと、美術画像に対して
    "painting of ~" や "arafed ~" といったノイズが混入しやすい。
    条件付き生成 (conditional) で短いプロンプトを与えることで
    モデルを「シーン描写」方向に誘導できる。

      "there is"  → "There is a woman sitting by a window"  ← 推奨
      "the scene" → "The scene shows a landscape at sunset"
      ""          → 無条件生成（従来の動作）

    BLIP2 では pixel_values のみ渡す unconditional 生成を使用する
    (device_map 環境で input_ids のデバイス不一致が発生しやすいため)。

    Notes: BLIP2 のデバイス扱いについて
    ─────────────────────────────────────
    device_map で分散配置した BLIP2 では、全テンソルを
    「visual encoder (= 先頭モジュール) のデバイス」に渡す必要がある。
    accelerate がそれ以降のモジュール間転送を自動で行う。
    """
    if model_type == "blip2":
        input_device = _get_first_device(model)
    else:
        input_device = torch.device(device)

    # ── 画像入力の準備 ────────────────────────────────────────────────────
    if model_type == "blip" and prompt:
        # 条件付き生成: プロンプトを画像と一緒に processor に渡す
        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        )
    else:
        # 無条件生成 (BLIP2 / prompt="" 時)
        inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    with torch.no_grad():
        if model_type == "blip2":
            # BLIP2: unconditional 生成
            # (input_ids のデバイス不一致を避けるため pixel_values のみ渡す)
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
        else:
            # BLIP: 条件付き or 無条件生成
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )

    # トークン ID → 文字列に変換 (CPU に戻してデコード)
    raw_captions = processor.batch_decode(
        generated_ids.cpu(), skip_special_tokens=True
    )

    # 後処理: ノイズプレフィックスを除去して先頭を大文字に
    return [clean_caption(c.strip()) for c in raw_captions]


# ─────────────────────────────────────────────────────────────────────────────
# チェックポイント管理
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    進捗を JSON ファイルで管理する。
    中断した場合に最後に処理したインデックスから再開できる。
    """

    def __init__(self, path: Path):
        self.path = path
        if path.exists():
            with open(path) as f:
                self._data = json.load(f)
            print(f"  チェックポイントを読み込みました: {path}")
            print(f"  再開位置: {self._data['last_idx'] + 1} 件目から")
        else:
            self._data = {
                "last_idx": -1,
                "total_processed": 0,
                "chunk_count": 0,
            }

    @property
    def last_idx(self) -> int:
        return self._data["last_idx"]

    @property
    def chunk_count(self) -> int:
        return self._data["chunk_count"]

    def update(self, last_idx: int, total_processed: int, chunk_count: int):
        self._data["last_idx"] = last_idx
        self._data["total_processed"] = total_processed
        self._data["chunk_count"] = chunk_count
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace):

    # ── 出力ディレクトリの準備 ────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    chunks_dir = output_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(exist_ok=True)

    final_path   = output_dir / "wikiart_blip_captions.parquet"
    progress_path = output_dir / "progress.json"

    print("=" * 60)
    print("  WikiArt キャプション生成")
    print(f"  モデル      : {args.model}")
    print(f"  プロンプト        : {repr(args.prompt)}")
    print(f"  max_new_tokens    : {args.max_new_tokens}")
    print(f"  min_new_tokens    : {args.min_new_tokens}")
    print(f"  num_beams         : {args.num_beams}")
    print(f"  length_penalty    : {args.length_penalty}")
    print(f"  repetition_penalty: {args.repetition_penalty}")
    print(f"  バッチサイズ      : {args.batch_size}")
    print(f"  出力先      : {output_dir}")
    print("=" * 60)

    # ── チェックポイントの読み込み ─────────────────────────────────────────────
    tracker = ProgressTracker(progress_path)
    start_idx = tracker.last_idx + 1

    # ── HuggingFace データセットのロード ──────────────────────────────────────
    print("\n[1/4] fusing/wikiart_captions をロード中...")
    from datasets import load_dataset as hf_load

    hf_ds = hf_load(
        "fusing/wikiart_captions",
        split="train",
        cache_dir=args.cache_dir,
    )
    total = len(hf_ds)
    if args.max_samples is not None:
        total = min(total, args.max_samples)
        print(f"  max_samples={args.max_samples} を指定: {total} 件のみ処理します")
    else:
        print(f"  総画像数: {total:,} 件")

    if start_idx >= total:
        print("  全件処理済みです。最終ファイルを生成します。")
    else:
        # ── モデルのロード ────────────────────────────────────────────────────
        print(f"\n[2/4] {args.model.upper()} モデルをロード中...")
        device = args.device
        if not torch.cuda.is_available() and "cuda" in device:
            print("  警告: CUDA が利用できません。CPU に切り替えます。")
            device = "cpu"
        print(f"  使用デバイス: {device}")

        if args.model == "blip2":
            processor, model, model_type = load_blip2(device)
        else:
            processor, model, model_type = load_blip(device)
        print(f"  {args.model.upper()} のロード完了")

        # ── キャプション生成ループ ────────────────────────────────────────────
        print(f"\n[3/4] キャプション生成中 (開始: {start_idx} 件目)...")

        chunk_buffer: List[dict] = []    # 現在のチャンクに貯めるデータ
        chunk_count  = tracker.chunk_count
        total_processed = tracker.last_idx + 1  # 処理済み件数

        # バッチ処理: start_idx から total まで
        pbar = tqdm(range(start_idx, total), desc="生成中", unit="枚")

        i = start_idx
        while i < total:
            # バッチの終端インデックスを決定
            batch_end = min(i + args.batch_size, total)
            batch_indices = list(range(i, batch_end))

            # 画像を取得して RGB に変換
            images: List[Image.Image] = []
            for idx in batch_indices:
                row = hf_ds[idx]
                img = row[args.image_field]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                images.append(img.convert("RGB"))

            # キャプション生成
            try:
                captions = generate_captions_batch(
                    images, processor, model, model_type, device,
                    max_new_tokens=args.max_new_tokens,
                    min_new_tokens=args.min_new_tokens,
                    prompt=args.prompt,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    repetition_penalty=args.repetition_penalty,
                )
            except RuntimeError as e:
                # CUDA OOM などへの対処: バッチを半分にして再試行
                if "out of memory" in str(e).lower() and args.batch_size > 1:
                    print(f"\n  CUDA OOM 発生。バッチサイズを {args.batch_size // 2} に縮小して再試行...")
                    args.batch_size = args.batch_size // 2
                    torch.cuda.empty_cache()
                    continue   # ループを再実行
                else:
                    raise

            # バッファに追加
            for idx, caption in zip(batch_indices, captions):
                chunk_buffer.append({
                    "image_idx": idx,
                    "caption":   caption,
                })

            total_processed += len(batch_indices)
            pbar.update(len(batch_indices))

            # チャンクサイズに達したら中間保存
            if len(chunk_buffer) >= args.chunk_size:
                chunk_path = chunks_dir / f"chunk_{chunk_count:04d}.parquet"
                pd.DataFrame(chunk_buffer).to_parquet(chunk_path, index=False)
                print(f"\n  チャンク保存: {chunk_path.name} ({len(chunk_buffer)} 件)")
                chunk_count += 1
                chunk_buffer = []

            # チェックポイント更新 (50バッチごと)
            if (i // args.batch_size) % 50 == 0:
                tracker.update(batch_end - 1, total_processed, chunk_count)

            i = batch_end

        pbar.close()

        # 残りのバッファを保存
        if chunk_buffer:
            chunk_path = chunks_dir / f"chunk_{chunk_count:04d}.parquet"
            pd.DataFrame(chunk_buffer).to_parquet(chunk_path, index=False)
            print(f"  最終チャンク保存: {chunk_path.name} ({len(chunk_buffer)} 件)")
            chunk_count += 1

        tracker.update(total - 1, total_processed, chunk_count)
        print(f"\n  生成完了: {total_processed:,} 件")

    # ── チャンクを結合して最終 Parquet ファイルを生成 ─────────────────────────
    print("\n[4/4] チャンクを結合して最終ファイルを生成中...")
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))

    if not chunk_files:
        print("  エラー: チャンクファイルが見つかりません。")
        return

    dfs = [pd.read_parquet(f) for f in chunk_files]
    df_all = pd.concat(dfs, ignore_index=True)

    # image_idx でソートして整列
    df_all = df_all.sort_values("image_idx").reset_index(drop=True)

    # 重複除去 (再開時に重複が生じる可能性への対処)
    df_all = df_all.drop_duplicates(subset="image_idx", keep="last")

    df_all.to_parquet(final_path, index=False)

    print(f"\n  最終ファイル: {final_path}")
    print(f"  総件数     : {len(df_all):,} 件")
    print(f"\n  先頭5件のキャプション:")
    for _, row in df_all.head(5).iterrows():
        print(f"    [{int(row['image_idx']):05d}] {row['caption']}")

    print("\n✓ 完了")


# ─────────────────────────────────────────────────────────────────────────────
# 引数パーサー
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WikiArt 画像に BLIP/BLIP2 でキャプションを生成する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # モデル設定
    parser.add_argument(
        "--model",
        choices=["blip", "blip2"],
        default="blip",
        help="使用するキャプションモデル。blip=軽量, blip2=高品質",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推論デバイス (cuda / cuda:0 / cpu)",
    )

    # データ設定
    parser.add_argument(
        "--image_field",
        default="image",
        help="HuggingFace データセット内の画像列名",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="HuggingFace キャッシュディレクトリ",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="処理する最大サンプル数 (デバッグ用)",
    )

    # 生成設定
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="バッチサイズ (VRAM に応じて調整。OOM 時は自動縮小)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=60,
        help="生成するキャプションの最大トークン数",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=20,
        help=(
            "生成するキャプションの最小トークン数。"
            "短すぎるキャプション (例: 'A woman') を防ぐ。"
            "論文品質 (~15語) を得るには 20〜30 が目安。"
        ),
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help=(
            "ビームサーチのビーム数。"
            "大きいほど品質向上・速度低下。BLIP/BLIP2 ともに有効。"
            "推奨: 5 (デフォルト)。速度優先なら 3。"
        ),
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.2,
        help=(
            "長さペナルティ。>1.0 で長いシーケンスを優遇し詳細な記述を促す。"
            "ビームサーチ (num_beams>=2) と組み合わせて有効。"
            "推奨: 1.2〜1.5。"
        ),
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.5,
        help=(
            "繰り返しペナルティ。>1.0 で同じ語句の繰り返しを抑制する。"
            "推奨: 1.3〜1.5。"
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="there is",
        help=(
            "条件付き生成のプロンプト (BLIP のみ有効)。"
            "'there is' でシーン描写寄りになり 'painting of ~' を抑制できる。"
            "空文字列 "" で無条件生成。"
        ),
    )

    # 出力設定
    parser.add_argument(
        "--output_dir",
        default="./wikiart_captions_out",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="中間保存の件数単位 (大きいほど保存頻度が下がる)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)