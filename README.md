# LoopFuse: Loop-wise Cross-modal Fusion for Multimodal Fake News Detection

LoopFuse detects image-text fake news. Text and image are each encoded by a weight-shared CLIP encoder with looped encoding, retaining the pooled hidden state from every loop. These loop-wise vectors form a sequence, which is fused via bidirectional cross-modal attention, and the final-loop features are used to predict authenticity.

```bash
python train_dp.py --config configs/weibo.yaml
torchrun --nproc_per_node=4 train_ddp.py --config configs/twitter.yaml
```
