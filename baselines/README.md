# Baselines

同骨干对照由 `trainer.py` 通过 `model_type` 调用。说明、命令和论文链接在项目根目录的 [README.md](../README.md)。

| `model_type` | 模块 | 论文 |
| --- | --- | --- |
| `clip` / `chinese` | LoopFuse | `paper/main.tex` |
| `clip_head` | `CLIPHead`，L=1，融合头与 LoopFuse 相同 | 论文中的 CLIP-Head |
| `stacked` | `Stacked`，整座 encoder 复制 L 份 | 论文中的 Stacked |
| `layer_trajectory` | `LayerTrajectory`，单次前向的最后 L 层 | 论文中的 Layer-Trajectory；[RINE](https://arxiv.org/abs/2402.19091) |
| `relaxed` | `Relaxed`，循环上的 attention LoRA | [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672) |
| `clip_concat` | `CLIPConcat`，拼接两个投影向量 | 补充对照 |
| `clip_interact` | `CLIPInteract`，积、绝对差、余弦 | [FND-CLIP](https://doi.org/10.1109/ICME55011.2023.00480) |

已发表、且使用另一套骨干的检测器不在这里重写。链接如下。

| 模型 | 论文 |
| --- | --- |
| EANN | https://doi.org/10.1145/3219819.3219903 |
| MVAE | https://doi.org/10.1145/3308558.3313552 |
| SpotFake | https://doi.org/10.1109/BigMM.2019.00-44 |
| MCAN | https://doi.org/10.18653/v1/2021.findings-acl.226 |
| CAFE | https://doi.org/10.1145/3485447.3511968 |
| FND-CLIP | https://doi.org/10.1109/ICME55011.2023.00480 |
| BMR | https://doi.org/10.1609/aaai.v37i4.25670 |
| MIMoE | https://doi.org/10.1145/3696410.3714522 |
| CIMDD | https://arxiv.org/abs/2504.09163 |
| MIAN | https://arxiv.org/abs/2502.01699 |
| DCCF | https://arxiv.org/abs/2512.20670 |
| DIVER | https://arxiv.org/abs/2601.07178 |
