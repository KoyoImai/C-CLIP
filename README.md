# C-CLIP
[C-CLIP: MULTIMODAL CONTINUAL LEARNING FOR VISION-LANGUAGE MODEL](https://proceedings.iclr.cc/paper_files/paper/2025/file/72fb9ab442fc60b7ae5d53bf6b478273-Paper-Conference.pdf)の再現実装です．

## ディレクトリ構造


## データセットの用意
基本的には実行するだけです．Hugging Face経由で自動でダウンロードされます．
COCOとWikiartは，画像データとキャプションデータを別で用意する必要があるので注意してください．
以下にCOCOとWikiartの準備手順を記します．

### COCO
COCOは，画像データを指定されるディレクトリに配置してください．キャプションなどはHugging Face経由で用意されます．

### Wikiart
Wikiartは，データセットにキャプションが存在しないため，事前に用意する必要があります．
キャプションの作成にはblip2を使用し，画像から自動でキャプションを生成します．
以下のコマンドを実行してキャプションを生成してください．
実行すると，`./wikiart_captions_out_blip2`に各画像のキャプション情報が生成されます．
```
python3 generate_wikiart_captions.py --model blip2 --output_dir ./wikiart_captions_out_blip2
```
また，キャプションと画像を同時に表示したい場合は，wikiart_caption_viewer.ipynbで可視化できます．

### その他データセット
学習・評価を実行すると自動でダウンロードされます．


## 学習の実行
C-CLIPの学習は以下を実行してください．
学習可能なパラメータは`--trainable_params`，LoRAの適用箇所は`--lora_targets`で指定できます．
論文には学習可能パラメータとLoRAの適用先に関する詳細がないため，学習可能なパラメータ数から推測して選択しています．
```
python3 main.py  \
        --lora_targets q,ffn  \
        --trainable_params token_embedding,text_pos_embedding,text_projection,class_embedding,visual_pos_embedding,visual_proj,logit_scale  \
        --batch_size 1024
```
CLIPのFinetuneを行う場合は，以下を実行してください．
```
python3 main_finetune.py --batch_size 1024
```
CLIP-LoRAの学習を行う場合は以下を実行してください．
学習可能なパラメータは`--trainable_params`，LoRAの適用箇所は`--lora_targets`で指定できます．
以下の実行例は，C-CLIPと同じ箇所を学習可能＆LoRA適用しています．
```
python3 main_lora.py  \
        --lora_targets q,ffn  \
        --trainable_params token_embedding,text_pos_embedding,text_projection,class_embedding,visual_pos_embedding,visual_proj,logit_scale  \
        --batch_size 1024
```

また，各タスクの学習終了後に自動でt2iとi2tの検索佐での評価が実行されます．
ゼロショット分類の評価は行われませんので，必要があれば別途評価を実行して下さい．

## 評価の実行
ゼロショット分類性能評価を行う場合は`eval_zeroshot.py`を実行してください．

- 事前学習済みCLIP をベースラインとして CIFAR-10/100 を評価
    ```
    python3 eval_zeroshot.py --no_checkpoint --datasets cifar10 cifar1
    ```

- task7 チェックポイントで評価
    ```
    python eval_zeroshot.py \
        --checkpoint ./checkpoints/cclip_task7.pt \
        --datasets cifar10 cifar100
    ```

- 全チェックポイントを一括評価（忘却の推移を確認）
    ```
    python eval_zeroshot.py \
        --checkpoint_dir ./checkpoints \
        --datasets cifar10 cifar100 imagenet \
        --imagenet_root /home/kouyou/datasets/imagenet/val \
        --output_csv ./results/zeroshot.csv
    ```


## 評価結果
本プログラムで学習・評価を行なった場合の精度一覧です．

### 事前学習済みCLIPの性能
本実験では，OpenAI CLIPの公開重みを初期化に使用しています．以下の精度は，OpenAI CLIPの重みで初期化した時の精度で，学習はしていません．
- CIFAR10，CIFAR100，ImageNetのゼロショット分類性能

    |  Method          |   CIFAR10   |  CIFAR100  |  ImageNet  |
    |------------------|-------------|------------|------------|
    |  CLIP(Vit-B/16)  |   90.65     |   68.38    |   65.28    |


- flickr30k，coco，pets，lexica，simpsons，sikiart，kream，sketchに対する検索タスクの性能（タスク毎）

    |  Method             |   flickr30k |  coco      |   pets     |   lexica   |  simpsons  |   wikiart  |  kream     |  sketch    | Average    |
    |---------------------|-------------|------------|------------|------------|------------|------------|------------|------------|------------|
    |  CLIP(Vit-B/16) I2T |   82.2      |  51.7      |  11.0      |  37.8      |  16.6      |  17.9      |  20.4      |  4.3       |  24.5      |
    |  CLIP(Vit-B/16) T2I |   62.1      |  32.7      |  11.2      |  34.9      |  13.9      |  19.9      |  21.8      |  5.1       |  25.2      |


### 継続学習後の性能
flickr30kやcocoなど，8つのデータセットで順番に学習した時の精度を以下に示します．
0 0.8058196902275085
20 0.7707797288894653