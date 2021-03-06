# atmaCup11
[atmCup11](https://www.guruguru.science/competitions/17/) プログラム共有用

テーマ：「美術作品の画像やメタデータからその作品がどの時代に作成されたのかを予測する」

## Result
・CV = 0.6590
・Public  Score = 0.6674
・Private Score = 0.6631
<p align="center">
<img src='imgs/result.png' width="1000px"/>
 
## DataSet
・Image
 - num：9856
 - widge, height：75〜225, 75〜225
 - channel：3
 
・Labels
 - object_id (str)：作品に一意につけられたID
 - sorting_date (int)：作品の代表年 (西暦) 
 - art_series_id (str)：シリーズ物の作品に対してユニークなID
 - target (choice([0,1,2,3]))：作品の年代カテゴリ, 予測対象
 
・DataArgumentation：CosineAnnealingWarmRestarts (Ranger)
 
## Network
1. 事前学習 (SimSiam & ESViT)
 - ともにmaterials + techniquesのマルチラベルをMSE Lossにより学習
  
2. アンサンブル学習
 -  targetの予測（ResNet18d, ResNet34d, Swin_tiny, Swin_small）
 -  sorting_dateの予測 (Swin_small)
 
<p align="center">
<img src='imgs/network.png' width="1000px"/>


## Reference
・[SimSiam](https://arxiv.org/pdf/2011.10566.pdf) 「Exploring Simple Siamese Representation Learning」
 
・[ESViT](https://arxiv.org/pdf/2106.09785v1.pdf)　「Efficient Self-supervised Vision Transformers for
Representation Learning」
 
・[Swin](https://arxiv.org/pdf/2103.14030.pdf)　「Swin Transformer: Hierarchical Vision Transformer using Shifted Windows」
