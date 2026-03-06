"""
eval_zeroshot.py — CIFAR-10 / CIFAR-100 / ImageNet ゼロショット分類評価

C-CLIP の各チェックポイントを読み込み、CLIP スタイルのゼロショット分類を実行する。
プロンプトアンサンブル (CLIP 論文 Appendix B の 80 テンプレート) を使用。

使い方
------
# 単一チェックポイント + CIFAR-10/100
python eval_zeroshot.py \\
    --checkpoint ./checkpoints/cclip_task7.pt \\
    --datasets cifar10 cifar100

# チェックポイントディレクトリを指定 → task0〜7 を自動スキャン
python eval_zeroshot.py \\
    --checkpoint_dir ./checkpoints \\
    --datasets cifar10 cifar100 imagenet \\
    --imagenet_root /home/kouyou/datasets/imagenet/val

# vanilla CLIP (チェックポイントなし、事前学習済みのみ) で評価
python eval_zeroshot.py \\
    --no_checkpoint \\
    --datasets cifar10 cifar100

# バッチサイズ・モデルバックボーン変更
python eval_zeroshot.py \\
    --checkpoint ./checkpoints/cclip_task3.pt \\
    --clip_model ViT-B/16 \\
    --batch_size 256 \\
    --datasets cifar10

出力例
------
╔══════════════════════════════════════════════════════════════╗
║  Zero-Shot Classification Results                            ║
╠════════════════════╦═══════════╦═══════════╦════════════════╣
║  Checkpoint        ║  CIFAR-10 ║  CIFAR-100 ║  ImageNet     ║
║                    ║  Top-1    ║  Top-1     ║  Top-1 / Top-5 ║
╠════════════════════╬═══════════╬═══════════╬════════════════╣
║  (no checkpoint)   ║  88.2%    ║  64.8%    ║  68.3% / 89.4% ║
║  cclip_task0.pt    ║  87.5%    ║  63.9%    ║  68.1% / 89.2% ║
║  ...               ║  ...      ║  ...      ║  ...           ║
╚════════════════════╩═══════════╩═══════════╩════════════════╝
"""

from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CLIP プロンプトテンプレート (論文 Appendix B より 80 テンプレート)
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

# CIFAR 向け: より短いシンプルなテンプレートセット
CIFAR_TEMPLATES: List[str] = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a black and white photo of a {}.",
    "a low contrast photo of a {}.",
    "a high contrast photo of a {}.",
    "a bad photo of a {}.",
    "a good photo of a {}.",
    "a photo of a small {}.",
    "a photo of a big {}.",
    "a photo of a {}.",
    "a rendering of a {}.",
    "a {} in a video game.",
    "a cropped photo of a {}.",
    "the {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a photo of my {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a photo of one {}.",
    "a close-up photo of the {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a drawing of a {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a photo of the small {}.",
]


# ─────────────────────────────────────────────────────────────────────────────
# クラス名定義
# ─────────────────────────────────────────────────────────────────────────────

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR100_CLASSES: List[str] = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee",
    "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus",
    "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch",
    "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant",
    "flatfish", "forest", "fox", "girl", "hamster", "house",
    "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid", "otter",
    "palm tree", "pear", "pickup truck", "pine tree", "plain", "plate",
    "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray",
    "road", "rocket", "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow tree", "wolf",
    "woman", "worm",
]




# ─────────────────────────────────────────────────────────────────────────────
# モデルロード
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    clip_model_name: str,
    checkpoint_path: Optional[str],
    device: str,
) -> Tuple[object, object, object]:
    """
    CCLIP モデルと val_transform をロードする。

    Parameters
    ----------
    clip_model_name  : "ViT-B/16" 等
    checkpoint_path  : None = vanilla CLIP (チェックポイントなし)
    device           : "cuda" / "cpu"

    Returns
    -------
    model         : CCLIP インスタンス (eval 済み)
    val_transform : 画像前処理 (PIL → Tensor)
    tokenizer     : clip.tokenize
    """
    from c_clip import CCLIP
    from clip.clip import tokenize

    model = CCLIP(
        clip_model_name=clip_model_name,
        device=device,
    )

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        # チェックポイントの形式: {"model_state": ..., "task_id": ...}
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  警告: missing keys = {len(missing)} 個")
        if unexpected:
            print(f"  警告: unexpected keys = {len(unexpected)} 個")

    # DataParallel なしで eval モード
    model.to(device)
    model.eval()

    return model, model.val_transform, tokenize


# ─────────────────────────────────────────────────────────────────────────────
# テキスト特徴 (クラス重み行列) の構築
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_class_weights(
    model,
    tokenizer,
    class_names: List[str],
    templates: List[str],
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    プロンプトアンサンブルでクラスごとのテキスト特徴を構築する。

    各クラス名に全テンプレートを適用し、L2 正規化 → 平均 → 再正規化
    したベクトルを返す (CLIP 論文の "prompt ensembling" と同一処理)。

    Returns
    -------
    weights : (n_classes, embed_dim) — L2 正規化済みクラス重みベクトル
    """
    model.eval()
    weights = []

    for cls_name in tqdm(class_names, desc="  クラス重み構築", leave=False):
        # 全テンプレートにクラス名を埋め込んでトークン化
        texts = [tmpl.format(cls_name) for tmpl in templates]

        # バッチ処理 (テンプレート数が多いときのメモリ節約)
        feats_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = tokenizer(batch_texts).to(device)          # (B, L)
            feats  = model.encode_text(tokens)                  # (B, D) L2正規化済み
            feats_list.append(feats)

        # テンプレート間で平均 → 再正規化 (アンサンブル)
        feats_all = torch.cat(feats_list, dim=0)                # (n_tmpl, D)
        mean_feat = feats_all.mean(dim=0)                       # (D,)
        mean_feat = F.normalize(mean_feat, dim=0)               # L2 正規化
        weights.append(mean_feat)

    return torch.stack(weights, dim=0)                          # (n_cls, D)


# ─────────────────────────────────────────────────────────────────────────────
# データローダー構築
# ─────────────────────────────────────────────────────────────────────────────

def build_cifar_loader(
    dataset_name: str,
    transform,
    data_root: str = "./datasets",
    batch_size: int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, List[str]]:
    """
    CIFAR-10 / CIFAR-100 の test DataLoader を返す。

    torchvision が利用可能な場合はそちらを優先し、
    ない場合は HuggingFace datasets にフォールバックする。

    Returns
    -------
    loader      : DataLoader (image_tensor, label_int) を返す
    class_names : クラス名リスト (torchvision の classes 属性と一致)
    """
    try:
        import torchvision.datasets as tv_datasets

        if dataset_name == "cifar10":
            ds = tv_datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform
            )
            class_names = CIFAR10_CLASSES
        else:  # cifar100
            ds = tv_datasets.CIFAR100(
                root=data_root, train=False, download=True, transform=transform
            )
            class_names = CIFAR100_CLASSES

        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False  # multi-GPU 環境で pin_memory=True は cudaErrorInvalidValue を引き起こす
        )
        return loader, class_names

    except Exception as e:
        print(f"  torchvision ロード失敗 ({e})。HuggingFace へフォールバック...")
        return _build_cifar_loader_hf(dataset_name, transform, batch_size, num_workers)


def _build_cifar_loader_hf(
    dataset_name: str,
    transform,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[str]]:
    """HuggingFace datasets 経由で CIFAR をロードする (フォールバック)。"""
    from datasets import load_dataset as hf_load
    from torch.utils.data import Dataset as TorchDataset
    from PIL import Image as PILImage

    hf_name = "cifar10" if dataset_name == "cifar10" else "cifar100"
    label_col = "label" if dataset_name == "cifar10" else "fine_label"
    hf_ds = hf_load(hf_name, split="test")

    class HFCIFARDataset(TorchDataset):
        def __init__(self, hf_dataset, transform, label_col):
            self.ds = hf_dataset
            self.transform = transform
            self.label_col = label_col

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            row = self.ds[idx]
            img = row["img"]
            if not isinstance(img, PILImage.Image):
                img = PILImage.fromarray(img)
            image = self.transform(img.convert("RGB"))
            label = row[self.label_col]
            return image, label

    ds = HFCIFARDataset(hf_ds, transform, label_col)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False  # multi-GPU 環境で pin_memory=True は cudaErrorInvalidValue を引き起こす
    )
    class_names = CIFAR10_CLASSES if dataset_name == "cifar10" else CIFAR100_CLASSES
    return loader, class_names


def build_imagenet_loader(
    imagenet_root: str,
    transform,
    batch_size: int = 256,
    num_workers: int = 8,
) -> Tuple[DataLoader, List[str]]:
    """
    ImageNet バリデーションセットの DataLoader を返す。

    ディレクトリ構造:
        {imagenet_root}/n01440764/ILSVRC2012_val_00000001.JPEG
        ...

    torchvision.datasets.ImageFolder を使用する。

    Returns
    -------
    loader      : DataLoader
    class_names : 人間可読クラス名 (synset ID を変換したもの)
    """
    import torchvision.datasets as tv_datasets

    ds = tv_datasets.ImageFolder(root=imagenet_root, transform=transform)

    # synset → 人間可読名への変換を試みる
    class_names = _get_imagenet_classnames(ds.classes)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False  # multi-GPU 環境で pin_memory=True は cudaErrorInvalidValue を引き起こす
    )
    return loader, class_names



# ─────────────────────────────────────────────────────────────────────────────
# ImageNet synset → 人間可読クラス名の埋め込み辞書 (CLIP 公式クラス名セット)
# ─────────────────────────────────────────────────────────────────────────────
# ImageFolder は val フォルダの synset サブディレクトリをアルファベット順に並べる。
# このため「synset→名前」の辞書を使い、ImageFolder.classes の各 synset を変換する。
#
# 参照: openai/CLIP Prompt Engineering for ImageNet ノートブック
_IMAGENET_SYNSET_TO_NAME: Dict[str, str] = {
    "n01440764": "tench", "n01443537": "goldfish", "n01484850": "great white shark",
    "n01491361": "tiger shark", "n01494475": "hammerhead shark", "n01496331": "electric ray",
    "n01498041": "stingray", "n01514668": "rooster", "n01514859": "hen",
    "n01518878": "ostrich", "n01530575": "brambling", "n01531178": "goldfinch",
    "n01532829": "house finch", "n01534433": "junco", "n01537544": "indigo bunting",
    "n01558993": "American robin", "n01560419": "bulbul", "n01580077": "jay",
    "n01582220": "magpie", "n01592084": "chickadee", "n01601694": "American dipper",
    "n01608432": "kite", "n01614925": "bald eagle", "n01616318": "vulture",
    "n01622779": "great grey owl", "n01629819": "fire salamander",
    "n01630670": "smooth newt", "n01631663": "newt", "n01632458": "spotted salamander",
    "n01632777": "axolotl", "n01641577": "American bullfrog", "n01644373": "tree frog",
    "n01644900": "tailed frog", "n01664065": "loggerhead sea turtle",
    "n01665541": "leatherback sea turtle", "n01667114": "mud turtle",
    "n01667778": "terrapin", "n01669191": "box turtle", "n01675722": "banded gecko",
    "n01677366": "green iguana", "n01682714": "Carolina anole",
    "n01685808": "desert grassland whiptail lizard", "n01687978": "agama",
    "n01688243": "frilled-necked lizard", "n01689811": "alligator lizard",
    "n01692333": "Gila monster", "n01693334": "European green lizard",
    "n01694178": "chameleon", "n01695060": "Komodo dragon",
    "n01697457": "Nile crocodile", "n01698640": "American alligator",
    "n01704323": "triceratops", "n01728572": "worm snake",
    "n01728920": "ring-necked snake", "n01729322": "eastern hog-nosed snake",
    "n01729977": "smooth green snake", "n01734418": "kingsnake",
    "n01735189": "garter snake", "n01737021": "water moccasin",
    "n01739381": "rattle snake", "n01740131": "indigo snake",
    "n01742172": "diamondback rattlesnake", "n01744401": "sidewinder rattlesnake",
    "n01748264": "trilobite", "n01749939": "harvestman", "n01751748": "scorpion",
    "n01753488": "yellow garden spider", "n01755581": "barn spider",
    "n01756291": "European garden spider", "n01768244": "southern black widow",
    "n01770081": "tarantula", "n01770393": "wolf spider", "n01773157": "tick",
    "n01773549": "centipede", "n01773797": "black grouse", "n01774384": "ptarmigan",
    "n01774750": "ruffed grouse", "n01775062": "prairie grouse", "n01776313": "peacock",
    "n01784675": "quail", "n01795545": "partridge", "n01796340": "African grey parrot",
    "n01797886": "macaw", "n01798484": "sulphur-crested cockatoo",
    "n01806143": "lorikeet", "n01806567": "coucal", "n01807496": "bee eater",
    "n01817953": "hornbill", "n01818515": "hummingbird", "n01819313": "jacamar",
    "n01820546": "toucan", "n01824575": "duck", "n01828970": "red-breasted merganser",
    "n01829413": "goose", "n01833805": "black swan", "n01843065": "tusker",
    "n01843383": "echidna", "n01847000": "platypus", "n01855032": "wallaby",
    "n01855672": "koala", "n01860187": "wombat", "n01871265": "jellyfish",
    "n01872401": "sea anemone", "n01873310": "brain coral", "n01877812": "flatworm",
    "n01882714": "nematode", "n01883070": "conch", "n01910747": "snail",
    "n01914609": "slug", "n01917289": "sea slug", "n01924916": "chiton",
    "n01930112": "chambered nautilus", "n01943899": "Dungeness crab",
    "n01944390": "rock crab", "n01945685": "fiddler crab",
    "n01950731": "red king crab", "n01955084": "American lobster",
    "n01968897": "spiny lobster", "n01978287": "crayfish",
    "n01978455": "hermit crab", "n01980166": "isopod", "n01981276": "white stork",
    "n01983481": "black stork", "n01984695": "spoonbill", "n01985128": "flamingo",
    "n01986214": "little blue heron", "n01990800": "great egret",
    "n02002556": "bittern bird", "n02002724": "crane bird", "n02006656": "limpkin",
    "n02007683": "common gallinule", "n02009229": "American coot",
    "n02009912": "bustard", "n02011460": "ruddy turnstone", "n02012849": "dunlin",
    "n02013706": "common redshank", "n02017213": "dowitcher",
    "n02018207": "oystercatcher", "n02018795": "pelican",
    "n02025239": "king penguin", "n02027492": "albatross",
    "n02028035": "grey whale", "n02033041": "killer whale", "n02037110": "dugong",
    "n02051845": "sea lion", "n02056570": "Chihuahua",
    "n02058221": "Japanese Chin", "n02066245": "Maltese",
    "n02071294": "Pekingese", "n02074367": "Shih Tzu",
    "n02077923": "King Charles Spaniel", "n02085620": "Papillon",
    "n02085782": "toy terrier", "n02085936": "Rhodesian Ridgeback",
    "n02086079": "Afghan Hound", "n02086240": "Basset Hound",
    "n02086646": "Beagle", "n02086910": "Bloodhound",
    "n02087046": "Bluetick Coonhound", "n02087394": "Black and Tan Coonhound",
    "n02088094": "Treeing Walker Coonhound", "n02088238": "English foxhound",
    "n02088364": "Redbone Coonhound", "n02088466": "borzoi",
    "n02088632": "Irish Wolfhound", "n02089078": "Italian Greyhound",
    "n02089867": "Whippet", "n02089973": "Ibizan Hound",
    "n02090379": "Norwegian Elkhound", "n02090622": "Otterhound",
    "n02090721": "Saluki", "n02091032": "Scottish Deerhound",
    "n02091134": "Weimaraner", "n02091244": "Staffordshire Bull Terrier",
    "n02091467": "American Staffordshire Terrier",
    "n02091635": "Bedlington Terrier", "n02091831": "Border Terrier",
    "n02092002": "Kerry Blue Terrier", "n02092339": "Irish Terrier",
    "n02093256": "Norfolk Terrier", "n02093428": "Norwich Terrier",
    "n02093647": "Yorkshire Terrier", "n02093754": "Wire Fox Terrier",
    "n02093859": "Lakeland Terrier", "n02093991": "Sealyham Terrier",
    "n02094114": "Airedale Terrier", "n02094258": "Cairn Terrier",
    "n02094433": "Australian Terrier", "n02095314": "Dandie Dinmont Terrier",
    "n02095570": "Boston Terrier", "n02095889": "Miniature Schnauzer",
    "n02096051": "Giant Schnauzer", "n02096177": "Standard Schnauzer",
    "n02096294": "Scottish Terrier", "n02096437": "Tibetan Terrier",
    "n02096585": "Australian Silky Terrier",
    "n02097047": "Soft-coated Wheaten Terrier",
    "n02097130": "West Highland White Terrier", "n02097209": "Lhasa Apso",
    "n02097298": "Flat-Coated Retriever", "n02097474": "Curly-coated Retriever",
    "n02097658": "Golden Retriever", "n02098105": "Labrador Retriever",
    "n02098286": "Chesapeake Bay Retriever",
    "n02098413": "German Shorthaired Pointer",
    "n02099267": "Vizsla", "n02099429": "English Setter",
    "n02099601": "Irish Setter", "n02099712": "Gordon Setter",
    "n02099849": "Brittany dog", "n02100236": "Clumber Spaniel",
    "n02100583": "English Springer Spaniel",
    "n02100735": "Welsh Springer Spaniel", "n02100877": "Cocker Spaniel",
    "n02101006": "Sussex Spaniel", "n02101388": "Irish Water Spaniel",
    "n02101556": "Kuvasz", "n02102040": "Schipperke",
    "n02102177": "Groenendael dog", "n02102318": "Malinois",
    "n02102480": "Briard", "n02102973": "Australian Kelpie",
    "n02104029": "Komondor", "n02104365": "Old English Sheepdog",
    "n02105056": "Shetland Sheepdog", "n02105162": "collie",
    "n02105251": "Border Collie", "n02105412": "Bouvier des Flandres dog",
    "n02105505": "Rottweiler", "n02105641": "German Shepherd Dog",
    "n02105855": "Dobermann", "n02106030": "Miniature Pinscher",
    "n02106166": "Greater Swiss Mountain Dog",
    "n02106382": "Bernese Mountain Dog",
    "n02106550": "Appenzeller Sennenhund",
    "n02106662": "Entlebucher Sennenhund", "n02107142": "Boxer",
    "n02107312": "Bullmastiff", "n02107574": "Tibetan Mastiff",
    "n02107683": "French Bulldog", "n02107908": "Great Dane",
    "n02108000": "St. Bernard", "n02108089": "husky",
    "n02108422": "Alaskan Malamute", "n02108551": "Siberian Husky",
    "n02108915": "Dalmatian", "n02109047": "Affenpinscher",
    "n02109525": "Basenji", "n02109961": "pug", "n02110063": "Leonberger",
    "n02110185": "Newfoundland dog", "n02110341": "Great Pyrenees dog",
    "n02110627": "Samoyed", "n02110806": "Pomeranian",
    "n02110958": "Chow Chow", "n02111129": "Keeshond",
    "n02111277": "brussels griffon", "n02111500": "Pembroke Welsh Corgi",
    "n02111889": "Cardigan Welsh Corgi", "n02112018": "Toy Poodle",
    "n02112137": "Miniature Poodle", "n02112350": "Standard Poodle",
    "n02112706": "Mexican hairless dog", "n02113023": "grey wolf",
    "n02113186": "Alaskan tundra wolf", "n02113624": "red wolf",
    "n02113712": "coyote", "n02113799": "dingo", "n02113978": "dhole",
    "n02114367": "African wild dog", "n02114548": "hyena",
    "n02114712": "red fox", "n02114855": "kit fox", "n02115641": "Arctic fox",
    "n02115913": "grey fox", "n02116738": "tabby cat",
    "n02117135": "tiger cat", "n02119022": "Persian cat",
    "n02119789": "Siamese cat", "n02120079": "Egyptian Mau",
    "n02120505": "cougar", "n02123045": "lynx", "n02123159": "leopard",
    "n02123394": "snow leopard", "n02123597": "jaguar", "n02124075": "lion",
    "n02125311": "tiger", "n02127052": "cheetah", "n02128385": "brown bear",
    "n02128757": "American black bear", "n02128925": "polar bear",
    "n02129165": "sloth bear", "n02129604": "mongoose", "n02130308": "meerkat",
    "n02132136": "tiger beetle", "n02133161": "ladybug",
    "n02134084": "ground beetle", "n02134418": "longhorn beetle",
    "n02137549": "leaf beetle", "n02138441": "dung beetle",
    "n02165105": "rhinoceros beetle", "n02165456": "weevil",
    "n02167151": "fly", "n02168699": "bee", "n02172182": "ant",
    "n02174001": "grasshopper", "n02177972": "cricket insect",
    "n02190166": "stick insect", "n02206856": "cockroach",
    "n02219486": "praying mantis", "n02226429": "cicada",
    "n02229544": "leafhopper", "n02231487": "lacewing",
    "n02233338": "dragonfly", "n02236044": "damselfly",
    "n02256656": "admiral butterfly", "n02259212": "ringlet butterfly",
    "n02264363": "monarch butterfly", "n02268443": "small white butterfly",
    "n02268853": "sulphur butterfly", "n02276258": "gossamer-winged butterfly",
    "n02277742": "starfish", "n02279972": "sea urchin",
    "n02280649": "sea cucumber", "n02281406": "cottontail rabbit",
    "n02281787": "hare", "n02317335": "Angora rabbit",
    "n02319095": "hamster", "n02321529": "porcupine",
    "n02325366": "fox squirrel", "n02326432": "marmot",
    "n02328150": "beaver", "n02342885": "guinea pig",
    "n02346627": "common sorrel horse", "n02356798": "zebra",
    "n02361337": "pig", "n02363005": "wild boar", "n02364673": "warthog",
    "n02389026": "hippopotamus", "n02391049": "ox",
    "n02395406": "water buffalo", "n02396427": "bison",
    "n02397096": "ram", "n02398521": "bighorn sheep",
    "n02403003": "Alpine ibex", "n02408429": "hartebeest",
    "n02410509": "impala", "n02412080": "gazelle",
    "n02415577": "arabian camel", "n02417914": "llama",
    "n02422106": "weasel", "n02422699": "mink",
    "n02423022": "European polecat", "n02437312": "black-footed ferret",
    "n02437616": "otter", "n02441942": "skunk", "n02442845": "badger",
    "n02443114": "armadillo", "n02443484": "three-toed sloth",
    "n02444819": "orangutan", "n02445715": "gorilla",
    "n02447366": "chimpanzee", "n02454379": "gibbon",
    "n02457408": "siamang", "n02480495": "guenon",
    "n02480855": "patas monkey", "n02481823": "baboon",
    "n02483362": "macaque", "n02483708": "langur",
    "n02484975": "black-and-white colobus", "n02486261": "proboscis monkey",
    "n02486410": "marmoset", "n02487347": "white-headed capuchin",
    "n02488291": "howler monkey", "n02488702": "titi monkey",
    "n02489166": "Geoffroy\'s spider monkey",
    "n02490219": "common squirrel monkey", "n02492035": "ring-tailed lemur",
    "n02492660": "indri", "n02493509": "Asian elephant",
    "n02493793": "African bush elephant", "n02494079": "red panda",
    "n02497673": "giant panda", "n02500267": "snoek fish",
    "n02504013": "eel", "n02504458": "silver salmon",
    "n02509815": "rock beauty fish", "n02510455": "clownfish",
    "n02514041": "sturgeon", "n02526121": "gar fish",
    "n02536864": "lionfish", "n02606052": "pufferfish",
    "n02607072": "abacus", "n02640242": "abaya",
    "n02641379": "academic gown", "n02643566": "accordion",
    "n02655020": "acoustic guitar", "n02666196": "aircraft carrier",
    "n02667093": "airliner", "n02669723": "airship",
    "n02672831": "altar", "n02676566": "ambulance",
    "n02687172": "amphibious vehicle", "n02690373": "analog clock",
    "n02692877": "apiary", "n02699494": "apron",
    "n02701002": "trash can", "n02704792": "assault rifle",
    "n02708093": "backpack", "n02727426": "bakery",
    "n02730930": "balance beam", "n02747177": "balloon",
    "n02749479": "ballpoint pen", "n02769748": "Band-Aid",
    "n02776631": "banjo", "n02777292": "baluster",
    "n02782093": "barbell", "n02783161": "barber chair",
    "n02786058": "barbershop", "n02787622": "barn",
    "n02788148": "barometer", "n02790996": "barrel",
    "n02791124": "wheelbarrow", "n02791270": "baseball",
    "n02793495": "basketball", "n02794156": "bassinet",
    "n02795169": "bassoon", "n02797295": "swimming cap",
    "n02799071": "bath towel", "n02802426": "bathtub",
    "n02804414": "station wagon", "n02804610": "lighthouse",
    "n02807133": "beaker", "n02808304": "military hat",
    "n02808440": "beer bottle", "n02814533": "beer glass",
    "n02814860": "bell tower", "n02815834": "baby bib",
    "n02817516": "tandem bicycle", "n02823428": "bikini",
    "n02823750": "ring binder", "n02825657": "binoculars",
    "n02834397": "birdhouse", "n02835271": "boathouse",
    "n02837789": "bobsleigh", "n02840245": "bolo tie",
    "n02841315": "poke bonnet", "n02843684": "bookcase",
    "n02859443": "bookstore", "n02860847": "bottle cap",
    "n02865351": "hunting bow", "n02869837": "bow tie",
    "n02879718": "brass memorial plaque", "n02883205": "bra",
    "n02892201": "breakwater", "n02892767": "breastplate",
    "n02894605": "broom", "n02895154": "bucket",
    "n02906734": "buckle", "n02909870": "bulletproof vest",
    "n02910353": "high-speed train", "n02916936": "butcher shop",
    "n02917067": "taxicab", "n02927161": "cauldron",
    "n02930766": "candle", "n02939185": "cannon",
    "n02948072": "canoe", "n02950826": "can opener",
    "n02951358": "cardigan", "n02951585": "car mirror",
    "n02963159": "carousel", "n02965783": "tool kit",
    "n02966193": "cardboard box", "n02966687": "car wheel",
    "n02971356": "automated teller machine", "n02974003": "cassette",
    "n02977058": "cassette player", "n02978881": "castle",
    "n02979186": "catamaran", "n02980441": "CD player",
    "n02981792": "cello", "n02988304": "mobile phone",
    "n02992211": "chain", "n02992529": "chain-link fence",
    "n02999410": "chain mail", "n03000134": "chainsaw",
    "n03000247": "chest", "n03000684": "chiffonier",
    "n03014705": "chime", "n03016953": "china cabinet",
    "n03017168": "Christmas stocking", "n03018349": "church",
    "n03026506": "movie theater", "n03028079": "cleaver",
    "n03032252": "cliff dwelling", "n03041632": "cloak",
    "n03042490": "clogs", "n03045698": "cocktail shaker",
    "n03047690": "coffee mug", "n03062245": "coffeemaker",
    "n03063599": "spiral", "n03063689": "combination lock",
    "n03065424": "computer keyboard", "n03075370": "candy store",
    "n03085013": "container ship", "n03089624": "convertible",
    "n03095699": "corkscrew", "n03100240": "cornet",
    "n03109150": "cowboy boot", "n03110669": "cowboy hat",
    "n03124043": "cradle", "n03124170": "construction crane",
    "n03125729": "crash helmet", "n03126707": "crate",
    "n03127747": "infant bed", "n03127925": "Crock Pot",
    "n03131574": "croquet ball", "n03133878": "crutch",
    "n03134739": "cuirass", "n03141823": "dam",
    "n03146219": "desk", "n03160309": "desktop computer",
    "n03179701": "rotary dial telephone", "n03180011": "diaper",
    "n03187595": "digital clock", "n03188531": "digital watch",
    "n03196217": "dining table", "n03197337": "dishcloth",
    "n03201208": "dishwasher", "n03207743": "disc brake",
    "n03207941": "dock", "n03208938": "dog sled",
    "n03216828": "dome", "n03218198": "doormat",
    "n03220513": "drilling rig", "n03223299": "drum",
    "n03240683": "drumstick", "n03249569": "dumbbell",
    "n03255030": "Dutch oven", "n03259280": "electric fan",
    "n03261776": "electric guitar", "n03271574": "electric locomotive",
    "n03272010": "entertainment center", "n03272562": "envelope",
    "n03290653": "espresso machine", "n03291819": "face powder",
    "n03297495": "feather boa", "n03314780": "filing cabinet",
    "n03325584": "fireboat", "n03337140": "fire truck",
    "n03344393": "fire screen", "n03345487": "flagpole",
    "n03347037": "flute", "n03355925": "folding chair",
    "n03372029": "football helmet", "n03376595": "forklift",
    "n03379051": "fountain", "n03384352": "fountain pen",
    "n03388043": "four-poster bed", "n03388183": "freight car",
    "n03388549": "French horn", "n03393912": "frying pan",
    "n03394916": "fur coat", "n03400231": "garbage truck",
    "n03404251": "gas mask", "n03417042": "gas pump",
    "n03424325": "goblet", "n03425413": "go-kart",
    "n03443371": "golf ball", "n03444034": "golf cart",
    "n03445777": "gondola", "n03445924": "gong",
    "n03447447": "gown", "n03447721": "grand piano",
    "n03450230": "greenhouse", "n03452741": "radiator grille",
    "n03457902": "grocery store", "n03459775": "guillotine",
    "n03461385": "hair clip", "n03467068": "hair spray",
    "n03476684": "half-track", "n03476991": "hammer",
    "n03478589": "hamper", "n03481172": "hair dryer",
    "n03482405": "hand-held computer", "n03483316": "handkerchief",
    "n03485407": "hard disk drive", "n03485794": "harmonica",
    "n03492542": "harp", "n03494278": "combine harvester",
    "n03495258": "hatchet", "n03496892": "holster",
    "n03498962": "home theater", "n03527444": "honeycomb",
    "n03529860": "hook", "n03530642": "hoop skirt",
    "n03532672": "gymnastic horizontal bar",
    "n03534580": "horse-drawn vehicle", "n03535780": "hourglass",
    "n03538406": "iPod", "n03544143": "clothes iron",
    "n03584254": "carved pumpkin", "n03584829": "jeans",
    "n03590841": "jeep", "n03594734": "T-shirt",
    "n03594945": "jigsaw puzzle", "n03595614": "rickshaw",
    "n03598930": "joystick", "n03617480": "kimono",
    "n03623198": "knee pad", "n03627232": "knot",
    "n03630383": "lab coat", "n03633091": "ladle",
    "n03637318": "lampshade", "n03642806": "laptop computer",
    "n03649909": "lawn mower", "n03657121": "lens cap",
    "n03658185": "letter opener", "n03661043": "library",
    "n03662601": "lifeboat", "n03666591": "lighter",
    "n03670208": "limousine", "n03673027": "ocean liner",
    "n03676483": "lipstick", "n03680355": "slip-on shoe",
    "n03690938": "lotion", "n03691459": "music speaker",
    "n03692522": "loupe magnifying glass", "n03697007": "sawmill",
    "n03706229": "magnetic compass", "n03709823": "messenger bag",
    "n03710193": "mailbox", "n03710637": "maillot",
    "n03710721": "one-piece bathing suit", "n03717622": "manhole cover",
    "n03720891": "maraca", "n03721384": "marimba",
    "n03724870": "mask", "n03729826": "matchstick",
    "n03733131": "maypole", "n03733281": "maze",
    "n03733805": "measuring cup", "n03742115": "medicine cabinet",
    "n03743016": "megalith", "n03759954": "microphone",
    "n03761084": "microwave oven", "n03763968": "military uniform",
    "n03764736": "milk can", "n03769881": "minibus",
    "n03770439": "miniskirt", "n03770679": "minivan",
    "n03773504": "missile", "n03775071": "mitten",
    "n03775546": "mixing bowl", "n03776460": "mobile home",
    "n03777568": "Model T", "n03777754": "modem",
    "n03781244": "monastery", "n03782006": "monitor",
    "n03785016": "moped", "n03786901": "mortar and pestle",
    "n03787032": "graduation cap", "n03788195": "mosque",
    "n03788365": "mosquito net", "n03791053": "vespa",
    "n03792782": "mountain bike", "n03792972": "tent",
    "n03793489": "computer mouse", "n03794056": "mousetrap",
    "n03796401": "moving van", "n03803284": "muzzle",
    "n03804744": "metal nail", "n03814639": "neck brace",
    "n03814906": "necklace", "n03825788": "baby pacifier",
    "n03832673": "notebook computer", "n03837869": "obelisk",
    "n03838899": "oboe", "n03840681": "ocarina",
    "n03841143": "odometer", "n03843555": "oil filter",
    "n03854065": "pipe organ", "n03857828": "oscilloscope",
    "n03866082": "overskirt", "n03868242": "bullock cart",
    "n03868863": "oxygen mask", "n03871628": "product packet",
    "n03873416": "paddle", "n03874293": "paddle wheel",
    "n03874599": "padlock", "n03876231": "paintbrush",
    "n03877472": "pajamas", "n03877845": "palace",
    "n03884397": "pan flute", "n03887697": "paper towel",
    "n03888257": "parachute", "n03888605": "parking meter",
    "n03891251": "party popper", "n03891332": "passenger car",
    "n03895866": "patio", "n03899768": "payphone",
    "n03902125": "pedestal", "n03903868": "pencil case",
    "n03908618": "pencil sharpener", "n03908714": "perfume",
    "n03916031": "Petri dish", "n03920288": "photocopier",
    "n03924679": "plectrum", "n03929660": "Pickelhaube",
    "n03929855": "picket fence", "n03930313": "pier",
    "n03930630": "piggy bank", "n03933933": "pill bottle",
    "n03935335": "pillow", "n03942813": "ping-pong ball",
    "n03944341": "pinwheel", "n03947888": "pirate ship",
    "n03950228": "drink pitcher", "n03954731": "block plane",
    "n03956157": "planetarium", "n03958227": "plastic bag",
    "n03961711": "plate rack", "n03967562": "farm plow",
    "n03970156": "plunger", "n03976467": "Polaroid camera",
    "n03976657": "pole", "n03977966": "police van",
    "n03980874": "poncho", "n03982430": "pool table",
    "n03983396": "soda bottle", "n03991062": "plant pot",
    "n03992509": "potter\'s wheel", "n03995372": "power drill",
    "n03998194": "prayer rug", "n04004767": "printer",
    "n04005630": "prison", "n04008634": "projectile",
    "n04009552": "projector", "n04019541": "hockey puck",
    "n04023962": "punching bag", "n04026417": "purse",
    "n04033901": "quill", "n04033995": "race car",
    "n04037443": "racket", "n04039381": "radiator",
    "n04040759": "radio", "n04041544": "radio telescope",
    "n04044716": "rain barrel", "n04049303": "recreational vehicle",
    "n04065272": "fishing casting reel", "n04067472": "reflex camera",
    "n04069434": "refrigerator", "n04070727": "remote control",
    "n04074963": "restaurant", "n04081281": "revolver",
    "n04086273": "rifle", "n04090263": "rocking chair",
    "n04099969": "rotisserie", "n04111531": "eraser",
    "n04116512": "rugby ball", "n04118538": "ruler measuring stick",
    "n04118776": "sneaker", "n04120489": "safe",
    "n04125021": "safety pin", "n04127249": "salt shaker",
    "n04131690": "sandal", "n04133789": "sarong",
    "n04136333": "saxophone", "n04141076": "scabbard",
    "n04141327": "weighing scale", "n04141975": "school bus",
    "n04146614": "schooner", "n04147183": "scoreboard",
    "n04149813": "CRT monitor", "n04152593": "screw",
    "n04153751": "screwdriver", "n04154565": "seat belt",
    "n04162706": "sewing machine", "n04179913": "shield",
    "n04192698": "shoe store", "n04200800": "shoji screen",
    "n04201297": "shopping basket", "n04204238": "shopping cart",
    "n04204347": "shovel", "n04208210": "shower cap",
    "n04209133": "shower curtain", "n04209239": "ski",
    "n04209783": "balaclava ski mask", "n04210120": "sleeping bag",
    "n04210392": "slide rule", "n04213440": "sliding door",
    "n04216452": "slot machine", "n04217882": "snorkel",
    "n04225987": "snowmobile", "n04228054": "snowplow",
    "n04229816": "soap dispenser", "n04235860": "soccer ball",
    "n04238763": "sock", "n04239074": "solar thermal collector",
    "n04243546": "sombrero", "n04251144": "soup bowl",
    "n04252077": "keyboard space bar", "n04252225": "space heater",
    "n04254120": "space shuttle", "n04254680": "spatula",
    "n04254777": "motorboat", "n04258138": "spider web",
    "n04259630": "spindle", "n04261969": "sports car",
    "n04263257": "spotlight", "n04264628": "stage",
    "n04265275": "steam locomotive", "n04266014": "through arch bridge",
    "n04270147": "steel drum", "n04273569": "stethoscope",
    "n04275548": "scarf", "n04277352": "stone wall",
    "n04285008": "stopwatch", "n04286575": "stove",
    "n04296562": "strainer", "n04310018": "tram",
    "n04311004": "stretcher", "n04311174": "couch",
    "n04317175": "stupa", "n04325704": "submarine",
    "n04326547": "suit", "n04328186": "sundial",
    "n04330267": "sunglass", "n04332243": "sunglasses",
    "n04335435": "sunscreen", "n04336792": "suspension bridge",
    "n04344873": "mop", "n04346328": "sweatshirt",
    "n04347754": "swim trunks", "n04350905": "swing",
    "n04355338": "switch", "n04355933": "syringe",
    "n04356056": "table lamp", "n04366367": "tank",
    "n04367480": "tape player", "n04370456": "teapot",
    "n04371430": "teddy bear", "n04371774": "television",
    "n04372370": "tennis ball", "n04376876": "thatched roof",
    "n04380533": "front curtain", "n04389033": "thimble",
    "n04392985": "threshing machine", "n04398044": "throne",
    "n04399382": "tile roof", "n04404412": "toaster",
    "n04409515": "tobacco shop", "n04417672": "toilet seat",
    "n04418357": "torch", "n04423845": "totem pole",
    "n04428191": "tow truck", "n04429376": "toy store",
    "n04435653": "tractor", "n04442312": "semi-trailer truck",
    "n04443257": "tray", "n04447861": "trench coat",
    "n04476259": "tricycle", "n04479046": "trimaran",
    "n04482393": "tripod", "n04483307": "triumphal arch",
    "n04485082": "trolleybus", "n04486054": "trombone",
    "n04487081": "hot tub", "n04487394": "turnstile",
    "n04493381": "typewriter keyboard", "n04501370": "umbrella",
    "n04507155": "unicycle", "n04509417": "upright piano",
    "n04515003": "vacuum cleaner", "n04517823": "vase",
    "n04522168": "vending machine", "n04523525": "vestment",
    "n04525038": "viaduct", "n04525305": "violin",
    "n04532106": "volleyball", "n04532670": "waffle iron",
    "n04536866": "wall clock", "n04540053": "wallet",
    "n04542943": "wardrobe", "n04548280": "military aircraft",
    "n04548362": "sink", "n04550184": "washing machine",
    "n04552348": "water bottle", "n04553703": "water jug",
    "n04554684": "water tower", "n04557648": "whiskey jug",
    "n04560804": "whistle", "n04562935": "hair wig",
    "n04579145": "window screen", "n04579432": "window shade",
    "n04584207": "Windsor tie", "n04589890": "wine bottle",
    "n04590129": "airplane wing", "n04591157": "wok",
    "n04591713": "wooden spoon", "n04592741": "wool",
    "n04596742": "split-rail fence", "n04597913": "shipwreck",
    "n04599235": "sailboat", "n04604644": "yurt",
    "n04606251": "website", "n07248320": "comic book",
    "n07695742": "crossword", "n07695954": "traffic sign",
    "n07697313": "traffic light", "n07697537": "dust jacket",
    "n07711569": "menu", "n07714571": "plate",
    "n07714990": "guacamole", "n07715103": "consomme",
    "n07716358": "hot pot", "n07716906": "trifle",
    "n07717410": "ice cream", "n07717556": "popsicle",
    "n07718472": "baguette", "n07718747": "bagel",
    "n07720875": "pretzel", "n07730033": "cheeseburger",
    "n07734744": "hot dog", "n07742313": "mashed potatoes",
    "n07745940": "cabbage", "n07747607": "broccoli",
    "n07749582": "cauliflower", "n07753113": "zucchini",
    "n07753275": "spaghetti squash", "n07753592": "acorn squash",
    "n07754684": "butternut squash", "n07768694": "cucumber",
    "n07930864": "artichoke", "n07932039": "bell pepper",
    "n09193705": "cardoon", "n09229709": "mushroom",
    "n09246464": "Granny Smith apple", "n09256479": "strawberry",
    "n09288635": "orange", "n09332890": "lemon",
    "n09399592": "fig", "n09421951": "pineapple",
    "n09428293": "banana", "n09468604": "jackfruit",
    "n09472597": "cherimoya", "n09835506": "pomegranate",
    "n10148035": "hay", "n10565667": "carbonara",
    "n11879895": "chocolate syrup", "n11939491": "dough",
    "n12057211": "meatloaf", "n12144580": "pizza",
    "n12267677": "potpie", "n13037406": "burrito",
    "n13040303": "red wine", "n13044778": "espresso",
    "n13052670": "tea cup", "n13054560": "eggnog",
    "n13133613": "mountain", "n15075141": "bubble",
}


def _get_imagenet_classnames(synsets: List[str]) -> List[str]:
    """
    ImageFolder.classes (synset ID のリスト) を人間可読クラス名に変換する。

    優先順位:
      1. torchvision >= 0.13 の ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
         → 1000 クラス名が正しい順序で取得できる
      2. ローカルの synset_words.txt / words.txt を読む
         → ImageNet データセットに同梱されていることが多い
      3. 埋め込み済み辞書 _IMAGENET_SYNSET_TO_NAME をフォールバックとして使用
         → synset ID をキーに名前を引く

    Parameters
    ----------
    synsets : ImageFolder.classes (アルファベット順にソートされた synset ID のリスト)

    Returns
    -------
    class_names : synset に対応する人間可読クラス名のリスト (同じ順序)
    """

    # ── 方法1: torchvision weights API (>= 0.13) ────────────────────────────
    try:
        from torchvision.models import ResNet50_Weights
        categories = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
        # categories は synset アルファベット順の 1000 クラス名リスト
        # len が一致すれば信頼できる
        if len(categories) == len(synsets):
            print(f"  クラス名取得: torchvision weights API ({len(categories)} クラス)")
            return list(categories)
    except Exception:
        pass

    # ── 方法2: synset_words.txt をローカルから読む ───────────────────────────
    # ImageNet データセットに同梱されていることが多い
    # フォーマット: "n01440764 tench, Tinca tinca"
    candidate_paths = [
        Path(synsets[0]).parent / "synset_words.txt" if synsets else None,
        Path("/home/kouyou/datasets/imagenet/synset_words.txt"),
        Path("/home/kouyou/datasets/imagenet/words.txt"),
        Path("/home/kouyou/datasets/imagenet/LOC_synset_mapping.txt"),
    ]
    for p in candidate_paths:
        if p is None or not p.exists():
            continue
        try:
            mapping = {}
            for line in p.read_text().strip().splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    sid, names = parts
                    # "tench, Tinca tinca" → 先頭の一般名のみ使用
                    mapping[sid] = names.split(",")[0].strip()
            if mapping:
                result = [mapping.get(s, s) for s in synsets]
                print(f"  クラス名取得: {p} ({len(mapping)} エントリ)")
                return result
        except Exception:
            continue

    # ── 方法3: 埋め込み辞書フォールバック ────────────────────────────────────
    n_found  = sum(1 for s in synsets if s in _IMAGENET_SYNSET_TO_NAME)
    n_total  = len(synsets)
    n_missing = n_total - n_found
    if n_missing > 0:
        print(f"  警告: {n_missing}/{n_total} クラスが辞書に未登録です。")
        print(f"  未登録 synset の例: {[s for s in synsets if s not in _IMAGENET_SYNSET_TO_NAME][:5]}")
        print(f"  未登録クラスは synset ID (例: n04008634) のままテキストプロンプトに使われます。")
    else:
        print(f"  クラス名取得: 埋め込み辞書フォールバック ({n_found}/{n_total} クラス一致)")
    return [_IMAGENET_SYNSET_TO_NAME.get(s, s) for s in synsets]



# ─────────────────────────────────────────────────────────────────────────────
# 精度計算
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_zeroshot(
    model,
    loader: DataLoader,
    class_weights: torch.Tensor,
    device: str,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    ゼロショット分類の Top-K 精度を計算する。

    Parameters
    ----------
    model         : CCLIP (eval 済み)
    loader        : (image_tensor, label_int) を返す DataLoader
    class_weights : (n_classes, D) L2 正規化済みクラス重みベクトル
    device        : "cuda" / "cpu"
    top_k         : 計算する K のリスト

    Returns
    -------
    {"Top-1": float, "Top-5": float, ...}  (パーセンテージ)
    """
    model.eval()
    class_weights = class_weights.to(device)    # (n_cls, D)

    correct = {k: 0 for k in top_k}
    total   = 0

    for images, labels in tqdm(loader, desc="  推論", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # 画像特徴 (L2 正規化済み)
        img_feats = model.encode_image(images)              # (B, D)

        # 類似度スコア: (B, n_cls)
        # class_weights は (n_cls, D) → transpose して (D, n_cls) にして内積
        logits = img_feats @ class_weights.t()              # (B, n_cls)

        # Top-K 精度
        batch_size = labels.size(0)
        max_k      = max(top_k)
        _, pred    = logits.topk(max_k, dim=1, largest=True, sorted=True)  # (B, max_k)
        pred       = pred.t()                               # (max_k, B)
        correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))  # (max_k, B)

        for k in top_k:
            correct[k] += correct_mask[:k].any(dim=0).sum().item()
        total += batch_size

    return {f"Top-{k}": correct[k] / total * 100.0 for k in top_k}


# ─────────────────────────────────────────────────────────────────────────────
# チェックポイント一覧の取得
# ─────────────────────────────────────────────────────────────────────────────

def collect_checkpoints(
    checkpoint: Optional[str],
    checkpoint_dir: Optional[str],
    no_checkpoint: bool,
) -> List[Tuple[str, Optional[str]]]:
    """
    評価するチェックポイントのリスト [(label, path), ...] を返す。

    - no_checkpoint=True   → [("vanilla CLIP", None)]
    - checkpoint 指定      → [("vanilla CLIP", None), (filename, path)]
    - checkpoint_dir 指定  → vanilla + ディレクトリ内の cclip_task*.pt をソート順
    """
    ckpts: List[Tuple[str, Optional[str]]] = []

    # vanilla CLIP は常に含める (ベースライン)
    ckpts.append(("vanilla CLIP", None))

    if no_checkpoint:
        return ckpts

    if checkpoint:
        p = Path(checkpoint)
        ckpts.append((p.name, str(p)))

    elif checkpoint_dir:
        d = Path(checkpoint_dir)
        found = sorted(d.glob("cclip_task*.pt"))
        if not found:
            print(f"  警告: {d} に cclip_task*.pt が見つかりません")
        for p in found:
            ckpts.append((p.name, str(p)))

    return ckpts


# ─────────────────────────────────────────────────────────────────────────────
# 結果テーブル出力
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(
    results: List[Dict],
    datasets: List[str],
) -> None:
    """評価結果を整形されたテーブルで出力する。"""
    # ヘッダー幅
    COL_W = 22
    K_W   = 10

    dataset_cols = []
    for ds in datasets:
        if ds == "imagenet":
            dataset_cols.append(f"ImageNet (Top-1/5)")
        elif ds == "cifar10":
            dataset_cols.append("CIFAR-10 (Top-1)")
        elif ds == "cifar100":
            dataset_cols.append("CIFAR-100 (Top-1)")

    header = f"{'Checkpoint':<{COL_W}}" + "".join(
        f"  {c:<{K_W+8}}" for c in dataset_cols
    )
    sep = "─" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for row in results:
        name = row["checkpoint"][:COL_W - 1]
        line = f"{name:<{COL_W}}"
        for ds in datasets:
            # ds キーが存在しない、または評価失敗で空 dict の場合は "N/A" を表示
            if ds not in row or not row[ds]:
                line += f"  {'N/A':<{K_W+8}}"
                continue
            m = row[ds]
            if "Top-5" in m:
                val = f"{m['Top-1']:5.2f}% / {m['Top-5']:5.2f}%"
            elif "Top-1" in m:
                val = f"{m['Top-1']:5.2f}%"
            else:
                val = "N/A"
            line += f"  {val:<{K_W+8}}"
        print(line)

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# メイン処理
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C-CLIP ゼロショット分類評価 (CIFAR-10/100, ImageNet)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # モデル / チェックポイント
    parser.add_argument("--clip_model", default="ViT-B/16",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14"],
                        help="CLIP バックボーン (main.py と揃えること)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--checkpoint", type=str, default=None,
                     help="単一チェックポイントファイルのパス (例: ./checkpoints/cclip_task7.pt)")
    grp.add_argument("--checkpoint_dir", type=str, default=None,
                     help="チェックポイントディレクトリ。cclip_task*.pt を自動スキャン")
    grp.add_argument("--no_checkpoint", action="store_true",
                     help="vanilla CLIP (事前学習済みのみ) で評価")

    
    # 評価データセット
    parser.add_argument("--datasets", nargs="+",
                        default=["cifar10", "cifar100"],
                        choices=["cifar10", "cifar100", "imagenet"],
                        help="評価するデータセット")
    parser.add_argument("--imagenet_root", type=str,
                        default="/home/kouyou/datasets/ImageNet/val",
                        help="ImageNet バリデーションセットのルートディレクトリ")
    parser.add_argument("--cifar_root", type=str,
                        default="/home/kouyou/datasets",
                        help="CIFAR データセットのキャッシュ先")

    # 実行設定
    parser.add_argument("--device", type=str, default=None,
                        help="デバイス (デフォルト: CUDA があれば cuda)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)

    # プロンプトテンプレート
    parser.add_argument("--template", type=str,
                        default="ensemble",
                        choices=["ensemble", "simple"],
                        help=(
                            "ensemble: 80テンプレートアンサンブル (推奨), "
                            "simple: 'a photo of a {}.' のみ"
                        ))

    # 出力
    parser.add_argument("--output_csv", type=str, default=None,
                        help="結果を CSV に保存するパス (省略時: 保存しない)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── デバイス設定 ──────────────────────────────────────────────────────────
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    print(f"\n使用デバイス: {device}")
    if "cuda" in device:
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── チェックポイント一覧 ──────────────────────────────────────────────────
    ckpt_list = collect_checkpoints(
        args.checkpoint, args.checkpoint_dir, args.no_checkpoint
    )
    print(f"\n評価チェックポイント数: {len(ckpt_list)}")
    for label, path in ckpt_list:
        print(f"  {label}  →  {path or '(なし: vanilla CLIP)'}")

    # ── テンプレート選択 ──────────────────────────────────────────────────────
    if args.template == "simple":
        templates_cifar    = ["a photo of a {}."]
        templates_imagenet = ["a photo of a {}."]
    else:
        templates_cifar    = CIFAR_TEMPLATES
        templates_imagenet = IMAGENET_TEMPLATES

    # ── 評価ループ ────────────────────────────────────────────────────────────
    all_results = []

    for ckpt_label, ckpt_path in ckpt_list:
        print(f"\n{'='*60}")
        print(f"  チェックポイント: {ckpt_label}")
        print(f"{'='*60}")

        # モデルロード
        t0 = time.time()
        model, val_transform, tokenizer = load_model(
            args.clip_model, ckpt_path, device
        )
        print(f"  モデルロード: {time.time() - t0:.1f}s")

        row = {"checkpoint": ckpt_label}

        # ── 各データセットで評価 ──────────────────────────────────────────────
        for ds_name in args.datasets:
            print(f"\n  [{ds_name}] 評価中...")

            try:
                # DataLoader 構築
                if ds_name == "imagenet":
                    loader, class_names = build_imagenet_loader(
                        args.imagenet_root, val_transform,
                        args.batch_size, args.num_workers,
                    )
                    templates = templates_imagenet
                    top_k = (1, 5)
                else:
                    loader, class_names = build_cifar_loader(
                        ds_name, val_transform,
                        args.cifar_root, args.batch_size, args.num_workers,
                    )
                    templates = templates_cifar
                    top_k = (1,)

                print(f"  クラス数: {len(class_names)}  |  "
                      f"テンプレート数: {len(templates)}")

                # クラス重みベクトル構築
                t1 = time.time()
                class_weights = build_class_weights(
                    model, tokenizer, class_names, templates, device
                )
                print(f"  クラス重み構築: {time.time() - t1:.1f}s")

                # 精度計算
                t2 = time.time()
                metrics = evaluate_zeroshot(
                    model, loader, class_weights, device, top_k=top_k
                )
                elapsed = time.time() - t2

                # 結果表示
                result_str = "  ".join(
                    f"{k}: {v:.2f}%" for k, v in metrics.items()
                )
                print(f"  結果: {result_str}  ({elapsed:.1f}s)")

                row[ds_name] = metrics

            except Exception as e:
                print(f"  エラー: {e}")
                import traceback
                traceback.print_exc()
                row[ds_name] = {}

        all_results.append(row)

        # メモリ解放
        del model
        if "cuda" in device:
            torch.cuda.empty_cache()

    # ── 結果テーブル出力 ──────────────────────────────────────────────────────
    print_results_table(all_results, args.datasets)

    # ── CSV 保存 ──────────────────────────────────────────────────────────────
    if args.output_csv:
        _save_csv(all_results, args.datasets, args.output_csv)
        print(f"\n結果を保存しました: {args.output_csv}")


def _save_csv(
    results: List[Dict],
    datasets: List[str],
    path: str,
) -> None:
    """評価結果を CSV に保存する。"""
    import csv

    # ヘッダー構築
    headers = ["checkpoint"]
    for ds in datasets:
        headers.append(f"{ds}_top1")
        if ds == "imagenet":
            headers.append(f"{ds}_top5")

    rows = []
    for row in results:
        r = {"checkpoint": row["checkpoint"]}
        for ds in datasets:
            if ds in row and row[ds]:
                r[f"{ds}_top1"] = f"{row[ds].get('Top-1', ''):.4f}"
                if ds == "imagenet":
                    r[f"{ds}_top5"] = f"{row[ds].get('Top-5', ''):.4f}"
            else:
                r[f"{ds}_top1"] = ""
                if ds == "imagenet":
                    r[f"{ds}_top5"] = ""
        rows.append(r)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()