"""Shape and formula checks for fusion and dual-stage attention. Does not download CLIP."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rawdata import load_weibo_splits
from src.workers.attention import BidirectionalAttention, CrossAttentionBlock, CrossAttentionEncoder
from src.workers.detector import Detector
from src.workers.fusion import MSFuse


def main() -> None:
    torch.manual_seed(0)
    check_fusion()
    check_stage1()
    check_stage2()
    check_detector()
    check_weibo_parser()
    print("sanity ok")


def check_fusion() -> None:
    fusion = MSFuse()
    text = torch.randn(4, 5, 16)
    image = torch.randn(4, 3, 16)
    pad = torch.zeros(4, 5, dtype=torch.bool)
    pad[:, -1] = True
    fused, fused_pad = fusion(text, image, pad)
    if fused.shape != (4, 8, 16) or fused_pad.shape != (4, 8):
        raise AssertionError((fused.shape, fused_pad.shape))
    if not torch.equal(fused_pad[:, :5], pad) or bool(fused_pad[:, 5:].any()):
        raise AssertionError("fused padding mask is wrong")
    if int(torch.count_nonzero(fused[:, 4])) != 0:
        raise AssertionError("padded text tokens leaked into the fused sequence")
    shifted = fusion(text, image + 3, pad)[0]
    if torch.allclose(fused[:, :4], shifted[:, :4]):
        raise AssertionError("text gates did not depend on the image")
    pooled_text = text.masked_fill(pad.unsqueeze(-1), 0).sum(dim=1) / (~pad).sum(dim=1).clamp(min=1).unsqueeze(-1)
    pooled_image = image.mean(dim=1)
    weights = torch.softmax(torch.stack([pooled_text, pooled_image], dim=1), dim=1)
    if not torch.allclose(weights.sum(dim=1), torch.ones(4, 16), atol=1e-5):
        raise AssertionError("per-channel gates do not sum to 1")
    gated_text = text * weights[:, 0].unsqueeze(1)
    gated_text = gated_text.masked_fill(pad.unsqueeze(-1), 0.0)
    gated_image = image * weights[:, 1].unsqueeze(1)
    expected = torch.cat([gated_text, gated_image], dim=1)
    if not torch.allclose(fused, expected, atol=1e-5):
        raise AssertionError("fused sequence does not match the gated concatenation")


def check_stage1() -> None:
    block = CrossAttentionBlock(32, 4, dropout=0.0, ffn_ratio=4)
    block.eval()
    query_a = torch.randn(2, 6, 32)
    query_b = torch.randn(2, 6, 32)
    memory = torch.randn(2, 5, 32)
    with torch.no_grad():
        _, weights_a = block.attn(query_a, memory, memory, average_attn_weights=False)
        _, weights_b = block.attn(query_b, memory, memory, average_attn_weights=False)
    if weights_a.shape[-1] != 5:
        raise AssertionError(weights_a.shape)
    if torch.allclose(weights_a, torch.ones_like(weights_a)):
        raise AssertionError("stage-1 attention collapsed to a single key")
    if torch.allclose(weights_a, weights_b):
        raise AssertionError("stage-1 attention ignored the query")
    encoder = CrossAttentionEncoder(32, 4, n_layers=4, dropout=0.0, ffn_ratio=4)
    encoder.eval()
    pad = torch.zeros(2, 6, dtype=torch.bool)
    pad[:, -2:] = True
    with torch.no_grad():
        hidden = encoder(query_a, memory, query_pad=pad)
    if hidden.shape != query_a.shape or int(torch.count_nonzero(hidden[:, -2:])) != 0:
        raise AssertionError("stage-1 padding positions were not cleared")


def check_stage2() -> None:
    layer = BidirectionalAttention(16)
    layer.eval()
    text = torch.randn(2, 4, 16)
    image = torch.randn(2, 3, 16)
    text_hat, image_hat = layer(text, image)
    scores = layer.similarity(text, image)
    expected_text = torch.softmax(scores, dim=-1) @ image
    expected_image = torch.softmax(scores.transpose(-1, -2), dim=-1) @ text
    if text_hat.shape != text.shape or image_hat.shape != image.shape:
        raise AssertionError((text_hat.shape, image_hat.shape))
    if not torch.allclose(text_hat, expected_text, atol=1e-5):
        raise AssertionError("image-to-text aggregation does not reuse S")
    if not torch.allclose(image_hat, expected_image, atol=1e-5):
        raise AssertionError("text-to-image aggregation does not reuse S")
    pad = torch.tensor([[False, False, True, True], [False, False, False, True]])
    text_hat, _ = layer(text, image, pad)
    if int(torch.count_nonzero(text_hat[0, 2:])) != 0 or int(torch.count_nonzero(text_hat[1, 3:])) != 0:
        raise AssertionError("padded text positions leaked out of stage 2")
    text_hat.sum().backward()
    if layer.weight.grad is None or not torch.isfinite(layer.weight.grad).all():
        raise AssertionError("shared bilinear map did not receive a finite gradient")


def check_detector() -> None:
    head = Detector(64, 32, 2, dropout=0.0)
    head.eval()
    logits = head(torch.randn(4, 64))
    if logits.shape != (4, 2):
        raise AssertionError(logits.shape)
    if sum(1 for module in head.net if isinstance(module, torch.nn.Linear)) != 3:
        raise AssertionError("detector does not have three linear layers")


def check_weibo_parser() -> None:
    rumor = (
        "1|user|http://t|http://u|1|true|0|0|0|9|0|0|0|0|web\n"
        "http://img.example/a.jpg|null\n"
        "谣言文本\n"
    )
    real = (
        "2|user|http://t|http://u|1|true|0|0|0|9|0|0|0|0|web\n"
        "http://img.example/b.jpg|null\n"
        "真实文本\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "rumor_images").mkdir()
        (root / "nonrumor_images").mkdir()
        Image.new("RGB", (8, 8), "red").save(root / "rumor_images" / "a.jpg")
        Image.new("RGB", (8, 8), "blue").save(root / "nonrumor_images" / "b.jpg")
        for name, content in (
            ("train_rumor.txt", rumor),
            ("test_rumor.txt", rumor),
            ("train_nonrumor.txt", real),
            ("test_nonrumor.txt", real),
        ):
            (root / name).write_text(content, encoding="utf-8")
        splits = load_weibo_splits(root)
    labels = {sample["label"] for sample in splits["train"]}
    texts = {sample["text"] for sample in splits["train"]}
    if len(splits["train"]) != 2 or labels != {0, 1} or texts != {"谣言文本", "真实文本"}:
        raise AssertionError(splits["train"])
    if splits["train_missing_images"] != 0:
        raise AssertionError(splits["train_missing_images"])


if __name__ == "__main__":
    main()
