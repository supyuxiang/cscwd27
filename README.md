# LoopFuse: Loop-wise Cross-modal Fusion for Multimodal Fake News Detection

LoopFuse 做图文假新闻检测。文本和图像各自用一套权重共享的 CLIP encoder 循环编码，留下每一圈的 pooled hidden state。这些圈向量组成一条序列，经双向跨模态注意力融合，取最后一圈的特征判断真假。

```bash
python train_dp.py --config configs/weibo.yaml
torchrun --nproc_per_node=4 train_ddp.py --config configs/twitter.yaml
```
