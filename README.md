# CMAAN：双阶段跨模态注意力虚假新闻检测

复现 Yang, Xu, Lan, Zhu 的 CSCWD 2026 论文 *Dual-Stage Cross-Modal Attention Network for Multimodal Fake News Detection*。模型名称是 **CMAAN**（Cross-Modal Aligned Attention Network）。

## 代码结构

```text
cscwd27/
├── src/
│   ├── workers/
│   │   ├── __init__.py        # 导出下面这些类
│   │   ├── clip_encoder.py    # 冻结的 CLIP 文本 token 与图像 patch
│   │   ├── fusion.py          # MS-Fuse：按通道 softmax 门控后拼接
│   │   ├── attention.py       # 第一阶段交叉注意力，第二阶段共享双线性注意力
│   │   └── detector.py        # 三层全连接分类器
│   ├── model.py               # 把上面四个模块串成 CMAAN
│   ├── config.py              # 扁平实验配置
│   ├── data.py                # JSONL 数据集与 CLIP batch
│   ├── rawdata.py             # 微博、Twitter 原始文件解析
│   └── utils.py               # 池化、指标、checkpoint
├── eval/
│   ├── train.py               # 训练、早停、官方测试集
│   ├── evaluate.py            # 加载 checkpoint 做测试
│   └── sanity.py              # 不下载 CLIP，检查公式和形状
├── scripts/
│   ├── train_weibo.sh
│   ├── train_twitter.sh
│   ├── eval_weibo.sh
│   └── eval_twitter.sh
├── data/preprocess.py         # 原始数据写成统一 jsonl
└── configs/                   # weibo.yaml、twitter.yaml
```

训练和测试入口：

| 脚本 | 作用 | 启动 |
|---|---|---|
| `eval/train.py` | 训练并在验证集上早停，最后测官方测试集 | `scripts/train_weibo.sh` 或 `scripts/train_twitter.sh` |
| `eval/evaluate.py` | 只加载 `best.pt` 做测试 | `scripts/eval_weibo.sh` 或 `scripts/eval_twitter.sh` |
| `eval/sanity.py` | 检查融合、两阶段注意力和分类头 | `python eval/sanity.py` |

标签约定：`1` 是 fake，`0` 是 real。JSONL 每行是 `id`、`text`、`image`、`label`。

## 模型

1. CLIP 编码器冻结。微博用 Chinese CLIP ViT-B/16，Twitter 用 OpenAI CLIP ViT-B/32。文本用全部 token，图像去掉 CLS，只留 patch，并投到 CLIP 的共享空间。隐藏维度 512，注意力 8 头。
2. MS-Fuse 对两路序列做掩码平均池化，在模态维上 softmax。每个通道的两个门相加为 1，再乘回各自的 token。融合序列是门控后的文本 token 和图像 token 沿序列维拼接。式 (1) 把同一个 softmax 写了两次，而且拼接后的 2d 维向量不能直接和 d 维特征相乘；按模态维切分 softmax 才能同时保住门控的形状，并让第一阶段的 key 不止一个位置。
3. 第一阶段是 4 层后置 LayerNorm 的交叉注意力。文本和图像各自做 query，融合序列做 key 和 value。每一层包含多头注意力和前馈网络，两段都有残差。文本塔和图像塔各有 4 层。
4. 第二阶段用一个共享矩阵 `W_b`：`S = T' W_b (V')^T`。图像到文本、文本到图像都对这同一个 `S` 做 softmax，再聚合对方特征。
5. 两个阶段的文本和图像表示分别做掩码平均，拼接后送进三层全连接。前两层是 Linear、BatchNorm、ReLU，最后一层线性投影到二维 logits。损失是两类交叉熵。

`use_fusion: false` 时融合退化为直接拼接。`use_stage1` 和 `use_stage2` 对应论文里的 w/o Cross、w/o Bidirectional。

论文给出的训练设置是 Adam、50 个 epoch、早停、dropout 0.2、batch size 128、权重衰减 `1e-4`，微博学习率 `0.001`，Twitter 学习率 `0.0005`。早停的耐心轮数和前馈网络的扩展倍数论文没有写，配置里分别是 `patience: 10` 和 `ffn_ratio: 4`。验证集从训练集按 `val_ratio` 分层划出，不改动官方测试集。

## 环境

```bash
cd /root/cscwd27
pip install -r requirements.txt
```

国内拉取 Hugging Face 权重时可以先设：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

不下载 CLIP、只检查模块：

```bash
python eval/sanity.py
```

## 数据

微博原始目录需要包含 `train_rumor.txt`、`train_nonrumor.txt`、`test_rumor.txt`、`test_nonrumor.txt`，以及对应图片。图片文件名是 URL 最后一段，大小写不敏感，允许后缀不一致。

```bash
python data/preprocess.py weibo --root /path/to/weibo_raw --out data/weibo
```

然后把 `configs/weibo.yaml` 里的 `image_root` 改成这个原始目录。

Twitter 支持带表头的 CSV/TSV，也支持无表头的 MediaEval 七列：`tweetId, tweetText, userId, imageId, username, timestamp, label`。`humor` 默认丢弃。

```bash
python data/preprocess.py twitter --root /path/to/twitter_raw --table /path/to/train.txt --out data/twitter
python data/preprocess.py twitter --root /path/to/twitter_raw --table /path/to/test.txt --out data/twitter
```

## 训练和测试

```bash
bash scripts/train_weibo.sh
bash scripts/eval_weibo.sh
bash scripts/train_twitter.sh
bash scripts/eval_twitter.sh
```

也可以直接：

```bash
python eval/train.py --config configs/weibo.yaml
python eval/evaluate.py --checkpoint outputs/weibo/best.pt --jsonl data/weibo/test.jsonl --image-root data/weibo_raw
```
