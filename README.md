# LoopFuse: Loop-wise Cross-modal Fusion for Multimodal Fake News Detection

LoopFuse 用共享权重把 CLIP 的文本塔和图像塔各循环多次，保留每一圈的池化向量，再沿 loop 轴做双向跨模态融合，用最后一圈的特征判断图文真伪。英文数据用 CLIP，中文数据用 Chinese CLIP。论文稿在 `paper/main.tex`。

```bash
torchrun --nproc_per_node=2 trainer.py --config=weibo_3090
torchrun --nproc_per_node=2 trainer.py --config=twitter_3090
```

`model_type: chinese` 或 `clip` 就是 LoopFuse。`--config` 写 yaml 的文件名（不带扩展名），后面可以跟 Hydra 覆盖项。

## 同骨干 baseline

这些模型走同一个 `trainer.py`、同一份 JSONL、同一份 CLIP 或 Chinese CLIP 权重，以及同一套投影和分类头。当前代码循环的是整座塔，所以对照也按整座塔实现，用来分开「多循环」「权值共享」和「融合头」各自的作用。论文里的 Stacked 写的是只复制上 K 个 block；那一版要等编码器改成只循环上层之后再对齐。

| `model_type` | 编码方式 | 对应工作 |
| --- | --- | --- |
| `clip` / `chinese` | 整座塔共享权重，循环 `num_loops` 次，再融合 | LoopFuse |
| `clip_head` | 只编码 1 次，融合头与 LoopFuse 相同 | 论文中的 CLIP-Head |
| `stacked` | 整座 encoder 复制 `num_loops` 份，权重不共享，计算量和 LoopFuse 对齐 | 论文中的 Stacked |
| `layer_trajectory` | 单次前向，取最后 `num_loops` 层的池化向量，再走同一个融合头 | 论文中的 Layer-Trajectory，对照 [RINE](https://arxiv.org/abs/2402.19091) |
| `relaxed` | 共享循环加上每一圈单独的 attention LoRA，`lora_rank` 默认 8 | [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) |
| `clip_concat` | 只编码 1 次，拼接两个投影向量后接 MLP | 补充的无轨迹对照 |
| `clip_interact` | 只编码 1 次，使用两向量、逐元素积、绝对差和余弦 | 同骨干上的交互特征，思路来自 [FND-CLIP](https://doi.org/10.1109/ICME55011.2023.00480) |

骨干由权重路径决定：路径里带 `chinese` 就用 Chinese CLIP，否则用 CLIP。两边都训练时，命令只改 `model_type` 和输出目录。`clip_head`、`clip_concat`、`clip_interact` 固定为 1 次编码。

```bash
torchrun --nproc_per_node=2 trainer.py --config=weibo_3090 \
  model_type=clip_head output_dir=outputs/weibo_clip_head run_name=weibo-clip-head

torchrun --nproc_per_node=2 trainer.py --config=weibo_3090 \
  model_type=stacked output_dir=outputs/weibo_stacked run_name=weibo-stacked

torchrun --nproc_per_node=2 trainer.py --config=weibo_3090 \
  model_type=layer_trajectory output_dir=outputs/weibo_layers run_name=weibo-layers

torchrun --nproc_per_node=2 trainer.py --config=weibo_3090 \
  model_type=relaxed output_dir=outputs/weibo_relaxed run_name=weibo-relaxed

torchrun --nproc_per_node=2 trainer.py --config=twitter_3090 \
  model_type=clip_interact output_dir=outputs/twitter_interact run_name=twitter-interact
```

`stacked` 会把 encoder 复制 `num_loops` 份。3090 上如果显存不够，把 `batch_size` 调小。`relaxed` 在 `freeze_backbone: true` 时仍然训练 LoRA、投影、融合头和分类器。要改 LoRA 秩，在 yaml 里写 `lora_rank`，或在命令行加 `+lora_rank=4`。

## 论文中的已发表检测器

`paper/main.tex` 还列出了一批已发表的多模态检测器。它们使用各自论文里的骨干和官方实现。划分与原文一致时直接引用原文数字；划分不同时重跑官方代码。这里没有另写一套 TextCNN、VGG 或 ResNet。

| 模型 | 论文 |
| --- | --- |
| EANN | [KDD 2018](https://doi.org/10.1145/3219819.3219903) |
| MVAE | [WWW 2019](https://doi.org/10.1145/3308558.3313552) |
| SpotFake | [IEEE BigMM 2019](https://doi.org/10.1109/BigMM.2019.00-44) |
| MCAN | [ACL-IJCNLP 2021 Findings](https://doi.org/10.18653/v1/2021.findings-acl.226) |
| CAFE | [WWW 2022](https://doi.org/10.1145/3485447.3511968) |
| FND-CLIP | [ICME 2023](https://doi.org/10.1109/ICME55011.2023.00480) |
| BMR | [AAAI 2023](https://doi.org/10.1609/aaai.v37i4.25670) |
| MIMoE | [WWW 2025](https://doi.org/10.1145/3696410.3714522) |
| CIMDD | [arXiv:2504.09163](https://arxiv.org/abs/2504.09163) |
| MIAN | [arXiv:2502.01699](https://arxiv.org/abs/2502.01699) |
| DCCF | [arXiv:2512.20670](https://arxiv.org/abs/2512.20670) |
| DIVER | [arXiv:2601.07178](https://arxiv.org/abs/2601.07178) |

骨干论文：[CLIP](https://arxiv.org/abs/2103.00020)，[Chinese CLIP](https://arxiv.org/abs/2211.01335)。LoRA 的形式来自 [Hu et al.](https://arxiv.org/abs/2106.09685)。
