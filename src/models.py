from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from pathlib import Path
from collections import defaultdict
import contextlib


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        clip_path: str | Path,
        d_model: int,
        freeze: bool = True,
    ):
        super().__init__()
        self.freeze = freeze
        self.d_model = d_model

        self._load_clip(clip_path) # self.model, self.processor
        self.clip_dim = self.model.config.projection_dim

        self.text_proj = nn.Linear(self.clip_dim, self.d_model)
        self.image_proj = nn.Linear(self.clip_dim, self.d_model)


    def _load_clip(self, clip_path: str | Path):
        is_chinese = ('chinese' in clip_path.split('/')[-1].lower())
        if is_chinese:
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor
            model = ChineseCLIPModel.from_pretrained(clip_path, trust_remote_code=True)
            processor = ChineseCLIPProcessor.from_pretrained(clip_path, trust_remote_code=True)
        else:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(clip_path, trust_remote_code=True)
            processor = CLIPProcessor.from_pretrained(clip_path, trust_remote_code=True)

        self.model = model
        self.processor = processor

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool=True):
        super().train(mode)
        if self.freeze:
            self.clip.eval()
        return self

    def _tokenize(
        self,
        texts: list[str]
    ):
        pass

    def _process_images(
        self,
        images: list[str]
    ):
        pass

    def forward(
        self,
        input_ids: torch.Tensor, # for text
        attention_mask: torch.Tensor, # for text
        pixel_values: torch.Tensor # for image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass





class GQA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int
    ):
        super().__init__()
        assert num_heads_q % num_heads_kv == 0, (num_heads_q, num_heads_kv)
        assert d_model % num_heads_q == 0, d_model

        self.d_model = d_model
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv

        self.d_k = self.d_model // self.num_heads_q

        self.wq = nn.Linear(d_model, self.d_k * self.num_heads_q, bias=False) # d_model = d_k * num_heads_q
        self.wk = nn.Linear(d_model, self.d_k * self.num_heads_kv, bias=False)
        self.wv = nn.Linear(d_model, self.d_k * self.num_heads_kv, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=True)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x: batch_size, seq_len ,d_model
        # check
        assert len(q.shape) == len(k.shape) == len(v.shape) == 3, (q.shape, k.shape, v.shape)
        assert q.size(-1) == k.size(-1) == v.size(-1) == self.d_model, (q.size(-1), k.size(-1), v.size(-1))
        assert k.size(1) == v.size(1), (k.size(1), v.size(1))
        bsz, seq_len_q, seq_len_kv = q.size(0), q.size(1), k.size(1)
        use_flash_ctx = (self.d_k <= 256) # and attn_mask is None
        # bsz, seq_len_q, d_model -> bsz, seq_len_q, num_heads_q, d_k -> bsz, num_heads_q, seq_len_q, d_k
        q = self.wq(q).view(bsz, seq_len_q, self.num_heads_q, self.d_k).transpose(1,2)
        # bsz, seq_len_kv, d_model -> bsz, seq_len_kv, num_heads_kv, d_k -> bsz, num_heads_kv, seq_len_kv, d_k
        k = self.wk(k).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1,2)
        v = self.wv(v).view(bsz, seq_len_kv, self.num_heads_kv, self.d_k).transpose(1,2)

        # scaled_dot_product_attention with flash_attn_v2
        with (
            torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION])
            if use_flash_ctx else contextlib.nullcontext()
        ):
            attn_out = F.scaled_dot_product_attention(
                q,k,v,attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=True
            ) # bsz, num_heads_q, seq_len_q, d_k
        # bsz, seq_len_q, num_heads_q, d_k -> bsz, seq_len_q, d_model
        attn_out = attn_out.transpose(1,2).contiguous().view(bsz, seq_len_q, -1)
        out = self.wo(attn_out)
        return out
        


class FFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        dropout_p: float=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x: torch.Tensor):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class DenseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_experts: int,
        dropout_p: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts

        self.experts = nn.ModuleList([
            FFN(
                d_model, d_ffn, dropout_p
            ) for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:  bsz, seq_len, d_model

        # gate 权重
        gate_logits = self.gate(x) # bsz, seq_len, num_experts
        gate_weights = F.softmax(gate_logits, dim=-1)

        # 每个 expert 的输出
        expert_outputs = []
        for expert in self.experts:
            out = expert(x) # bsz, seq_len, d_model
            expert_outputs.append(out) 

        expert_outputs = torch.stack(expert_outputs, dim=2) # bsz, seq_len, num_experts, d_model

        gate_weights = gate_weights.unsqueeze(-1) # bsz, seq_len, num_experts, 1
        output = (gate_weights * expert_outputs).sum(dim=2) # bsz, seq_len, d_model

        return output


class SparseMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        num_experts: int,
        top_k: int = 2,
        dropout_p: float = 0.1
    ):
        super().__init__()
        assert top_k <= num_experts

        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = nn.ModuleList([
            FFN(d_model, d_ffn, dropout_p)
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(d_model, num_experts, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: bsz, seq_len, d_model
        # gate 分数
        gate_logits = self.gate(x)                     # [bsz, seq_len, num_experts]

        # top-k routing
        topk_weights, topk_indices = torch.topk(
            gate_logits, self.top_k, dim=-1
        )                                              # 各 [bsz, seq_len, top_k]

        # softmax 归一化权重
        topk_weights = F.softmax(topk_weights, dim=-1)

        # 初始化输出
        output = torch.zeros_like(x)

        # 对每个 expert，收集需要处理的 token
        for i in range(self.num_experts):
            # 找出所有被路由到 expert i 的 token
            expert_mask = (topk_indices == i)           # [bsz, seq_len, top_k]
            if not expert_mask.any():
                continue

            # 收集对应的权重
            expert_weights = topk_weights[expert_mask]  # [num_tokens_routed]

            # 找到对应 token 的位置
            # expert_mask 是 bool，直接用来索引
            # 但 topk_indices 是 [bsz, seq_len, top_k]，需要展开
            # 我们用 flatten 方式处理
            flat_mask = expert_mask.any(dim=-1)        # [bsz, seq_len]
            tokens_for_expert = x[flat_mask]           # [num_tokens, d_model]

            if tokens_for_expert.numel() == 0:
                continue

            # expert 前向
            expert_out = self.experts[i](tokens_for_expert)

            # 加权加回输出
            # 需要把权重 broadcast 到 d_model 维度
            expert_weights = expert_weights.unsqueeze(-1)
            weighted_out = expert_weights * expert_out

            # scatter 回原位置
            output[flat_mask] += weighted_out.squeeze(-1)

        return output
    


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int,
        d_ffn: int,
        dropout_p: float=0.1,
    ):
        super().__init__()
        
        self.gqa = GQA(
            d_model=d_model,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
        )
        self.ffn = FFN(
            d_model=d_model,
            d_ffn=d_ffn,
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

        nn.init.ones_(self.ln1.weight)
        nn.init.zeros_(self.ln1.bias)
        nn.init.ones_(self.ln2.weight)
        nn.init.zeros_(self.ln2.bias)


    def forward(self, x: torch.Tensor):
        # x: bsz, seq_len, d_model
        residual = x
        x = self.ln1(x)
        x = self.gqa(x,x,x)
        x = self.dropout1(residual + x)

        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout2(residual + x)

        return x



class LoopedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int,
        d_ffn: int,
        num_layers: int,
        looped_indices: list[int] | None = None,
        dropout_p: float=0.1,
        decoder: nn.Module | None = None
    ):
        super().__init__()
        assert (
            looped_indices is None or (isinstance(looped_indices, list) and len(looped_indices <= num_layers))
        ), looped_indices

        if looped_indices == None:
            self.looped_indices = []
        self.looped_indices = looped_indices

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    d_ffn=d_ffn,
                    dropout_p=dropout_p
                ) for _ in range(num_layers)
            ]
        )

        self.decoder = decoder


    def forward(self, x :torch.Tensor, num_loops: int=2):
        assert num_loops >= 1 and isinstance(num_loops, int)

        _cache = defaultdict(list)

        # x: bsz, seq_len, d_model
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            _cache[idx].append(x)
            if idx in self.looped_indices:
                for _ in range(num_loops -1):
                    x = layer(x)
                    
        if self.decoder is not None and isinstance(self.decoder, nn.Module):
            x = self.decoder(x)
        return x
    


class OurModel(nn.Module):
    def __init__(
        self,
        clip_path: str | Path,
        d_model: int,
        num_heads_q: int,
        num_heads_kv: int,
        d_ffn: int,
        num_layer: int,
        looped_indices: list[int] | None = None,
        dropout_p: float=0.1,
        freeze_clip: bool=True,
    ):
        super().__init__()

        self.clip_encoder = CLIPEncoder(
            clip_path=clip_path,
            d_model=d_model,
            freeze=freeze_clip,
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(d_model, 1)           
        )
        

    def forward(
        self,
        texts: list[str] | str | None = None,
        images: list[torch.Tensor] | torch.Tensor | None = None
    ):
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(images, torch.Tensor):
            images = [torch.Tensor]

        if texts is not None and images is not None:
            assert len(texts) == len(images), (len(texts), len(images))

        return 














# ----------------------

# ============================================================
# 公共组件：逐 token 生成 α, β, S
# ============================================================
class ReadWriteMixGenerator(nn.Module):
    """从 X (B, n, d) 生成 α (B, n), β (B, n), S (B, n, n)"""
    def __init__(self, d, n):
        super().__init__()
        self.w_alpha = nn.Linear(d, n)
        self.w_beta  = nn.Linear(d, n)
        self.w_s     = nn.Linear(d, n * n)
        self.n = n

    def forward(self, X):
        # 对 n 条流做平均池化，得到每个 token 的聚合表示 (B, d)
        h = X.mean(dim=1) # b, n, d -> b, d
        # 逐 token 生成
        alpha = self.w_alpha(h) # b, n
        beta  = self.w_beta(h) # b, n
        S     = self.w_s(h).view(-1, self.n, self.n)  # b, n * n -> b, n, n
        return alpha, beta, S


# ============================================================
# HC：无约束混合
# ============================================================
class HCBlock(nn.Module):
    def __init__(self, d, n, sublayer):
        super().__init__()
        self.gen = ReadWriteMixGenerator(d, n)
        self.sublayer = sublayer              # 任意 d -> d 的子层（Attention/FFN）
        self.n = n

    def forward(self, X):
        # X: (B, n, d)
        alpha, beta, S = self.gen(X)          # (B,n), (B,n), (B,n,n)

        # --- 读取：h = αᵀ X ---
        # alpha.unsqueeze(-1): (B, n, 1)，与 X 做加权和
        h = (alpha.unsqueeze(-1) * X).sum(dim=1)   # (B, d)

        # --- 子层计算 ---
        y = self.sublayer(h)                        # (B, d)

        # --- 混合：B X ---
        # S: (B, n, n), X: (B, n, d) -> (B, n, d)
        mixed = torch.bmm(S, X)

        # --- 写入：+ β yᵀ ---
        # beta.unsqueeze(-1) * y.unsqueeze(1): (B, n, d)
        write = beta.unsqueeze(-1) * y.unsqueeze(1)

        X_next = mixed + write                      # (B, n, d)
        return X_next


# ============================================================
# mHC：Sinkhorn 约束的混合
# ============================================================
def sinkhorn(log_alpha, n_iters=5):
    """
    log_alpha: (B, n, n) 的打分矩阵（S）
    返回：双随机矩阵 B (B, n, n)
    """
    B = log_alpha.exp()                          # 保证非负
    for _ in range(n_iters):
        # 行归一化
        B = B / B.sum(dim=2, keepdim=True).clamp_min(1e-8)
        # 列归一化
        B = B / B.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return B


class mHCBlock(nn.Module):
    def __init__(self, d, n, sublayer, n_sinkhorn_iters=5):
        super().__init__()
        self.gen = ReadWriteMixGenerator(d, n)
        self.sublayer = sublayer
        self.n = n
        self.T = n_sinkhorn_iters

    def forward(self, X):
        alpha, beta, S = self.gen(X)

        # --- 读取 ---
        h = (alpha.unsqueeze(-1) * X).sum(dim=1)

        # --- 子层 ---
        y = self.sublayer(h)

        # --- mHC 的关键差异：Sinkhorn 投影 ---
        B = sinkhorn(S, self.T)                     # (B, n, n)，双随机

        # --- 混合 + 写入 ---
        mixed = torch.bmm(B, X)
        write = beta.unsqueeze(-1) * y.unsqueeze(1)

        X_next = mixed + write
        return X_next


# ============================================================
# 快速验证：数值稳定性对比
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, n, d = 2, 4, 16
    depth = 50

    sublayer = nn.Sequential(
        nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)
    )

    # 初始多流状态
    x0 = torch.randn(B, d)
    X_init = x0.unsqueeze(1).repeat(1, n, 1)        # (B, n, d)

    # ---- HC ----
    X = X_init.clone()
    hc = HCBlock(d, n, sublayer)
    for _ in range(depth):
        X = hc(X)
    print("[HC]  输出范数:", X.norm().item())

    # ---- mHC ----
    X = X_init.clone()
    mhc = mHCBlock(d, n, sublayer)
    for _ in range(depth):
        X = mhc(X)
    print("[mHC] 输出范数:", X.norm().item())

    # ---- 验证 Sinkhorn 输出确实是双随机 ----
    _, _, S = mhc.gen(X)
    B_mat = sinkhorn(S, n_iters=5)
    print("\n双随机校验 (应接近 1):")
    print("  行和:", B_mat.sum(dim=2)[0])   # 应为 [1,1,1,1]
    print("  列和:", B_mat.sum(dim=1)[0])



    


    


