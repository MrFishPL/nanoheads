"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Nanohead configuration
    nanohead_proportion: float = 0.0  # proportion of attention QKV params for nanoheads (0 = disabled)
    nanohead_dim: int = 3  # dimension of each nanohead


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def calculate_nanohead_split(n_embd, n_head, n_kv_head, head_dim, nanohead_proportion, nanohead_dim):
    """
    Calculate split between normal and nano heads based on QKV parameter proportion.

    Returns: (normal_heads, normal_kv_heads, normal_head_dim,
              nano_heads, nano_kv_heads, nano_head_dim)
    """
    if nanohead_proportion == 0:
        return n_head, n_kv_head, head_dim, 0, 0, nanohead_dim

    # GQA ratio (how many query heads per KV head)
    gqa_ratio = n_head / n_kv_head

    # We want: nano_qkv_params / total_qkv_params ≈ nanohead_proportion
    # Start with approximate split based on output dimensions
    target_nano_dim = round(nanohead_proportion * n_embd / nanohead_dim) * nanohead_dim
    target_nano_dim = max(nanohead_dim, target_nano_dim)  # At least one nanohead
    target_nano_dim = min(target_nano_dim, n_embd - 2)  # Leave at least dim 2 for normal heads

    # Try to find a valid split where:
    # 1. normal_head_dim is even (for RoPE)
    # 2. total dimensions sum to exactly n_embd
    # 3. nano_output_dim is a multiple of nanohead_dim

    best_config = None
    best_diff = float('inf')

    # Try different nano_output_dims around the target
    # nano_dim must be a multiple of nanohead_dim
    # normal_dim must be even (so it can have even head_dim)
    # nano_dim + normal_dim must equal n_embd

    # This means we need: nano_dim = k * nanohead_dim for some integer k
    # and normal_dim = n_embd - k * nanohead_dim must be even

    # If nanohead_dim is odd and n_embd is even, then k * nanohead_dim must be even
    # which means k must be even (since odd * odd = odd)
    step = nanohead_dim if nanohead_dim % 2 == 0 else 2 * nanohead_dim

    for nano_dim_candidate in range(step, n_embd - 1, step):
        normal_dim_candidate = n_embd - nano_dim_candidate

        # Verify normal_dim is even
        if normal_dim_candidate % 2 != 0:
            continue

        # Verify nano_dim is a multiple of nanohead_dim
        if nano_dim_candidate % nanohead_dim != 0:
            continue

        # Find best even divisor for normal_dim_candidate (closest to target head_dim)
        for candidate_head_dim in range(2, normal_dim_candidate + 1, 2):
            if normal_dim_candidate % candidate_head_dim == 0:
                diff = abs(nano_dim_candidate - target_nano_dim)
                if diff < best_diff:
                    best_diff = diff
                    best_config = (normal_dim_candidate, candidate_head_dim, nano_dim_candidate)
                break  # Take the first (smallest) even divisor

    assert best_config is not None, f"Could not find valid split for n_embd={n_embd}, proportion={nanohead_proportion}"

    normal_output_dim, normal_head_dim, nano_output_dim = best_config

    # Calculate number of heads
    normal_heads = normal_output_dim // normal_head_dim
    normal_kv_heads = max(1, round(normal_heads / gqa_ratio))
    nano_heads = nano_output_dim // nanohead_dim
    nano_kv_heads = max(1, round(nano_heads / gqa_ratio))

    # Verify the split
    actual_total = normal_heads * normal_head_dim + nano_heads * nanohead_dim
    assert actual_total == n_embd, \
        f"Output dimension mismatch: {normal_heads}*{normal_head_dim} + {nano_heads}*{nanohead_dim} = {actual_total} != {n_embd}"
    assert normal_head_dim % 2 == 0, f"Normal head dimension must be even for RoPE, got {normal_head_dim}"

    return normal_heads, normal_kv_heads, normal_head_dim, nano_heads, nano_kv_heads, nanohead_dim

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = config.n_embd
        self.config = config

        # Calculate split between normal and nano heads
        default_head_dim = config.n_embd // config.n_head
        (self.normal_heads, self.normal_kv_heads, self.normal_head_dim,
         self.nano_heads, self.nano_kv_heads, self.nano_head_dim) = calculate_nanohead_split(
            config.n_embd, config.n_head, config.n_kv_head, default_head_dim,
            config.nanohead_proportion, config.nanohead_dim
        )

        self.has_nanoheads = self.nano_heads > 0

        # Normal head projections
        if self.normal_heads > 0:
            self.c_q_normal = nn.Linear(self.n_embd, self.normal_heads * self.normal_head_dim, bias=False)
            self.c_k_normal = nn.Linear(self.n_embd, self.normal_kv_heads * self.normal_head_dim, bias=False)
            self.c_v_normal = nn.Linear(self.n_embd, self.normal_kv_heads * self.normal_head_dim, bias=False)

        # Nano head projections (no RoPE, so different handling)
        if self.has_nanoheads:
            self.c_q_nano = nn.Linear(self.n_embd, self.nano_heads * self.nano_head_dim, bias=False)
            self.c_k_nano = nn.Linear(self.n_embd, self.nano_kv_heads * self.nano_head_dim, bias=False)
            self.c_v_nano = nn.Linear(self.n_embd, self.nano_kv_heads * self.nano_head_dim, bias=False)

        # Output projection (same for all heads)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Value embeddings
        self.ve_gate_channels = 32
        total_kv_heads = self.normal_kv_heads + self.nano_kv_heads
        self.ve_gate = nn.Linear(self.ve_gate_channels, total_kv_heads, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        outputs = []

        # Process normal heads (with RoPE)
        if self.normal_heads > 0:
            q_normal = self.c_q_normal(x).view(B, T, self.normal_heads, self.normal_head_dim)
            k_normal = self.c_k_normal(x).view(B, T, self.normal_kv_heads, self.normal_head_dim)
            v_normal = self.c_v_normal(x).view(B, T, self.normal_kv_heads, self.normal_head_dim)

            # Value residual for normal heads
            if ve is not None:
                ve_normal_dim = self.normal_kv_heads * self.normal_head_dim
                ve_normal = ve[..., :ve_normal_dim].view(B, T, self.normal_kv_heads, self.normal_head_dim)
                gate_normal = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])[..., :self.normal_kv_heads])
                v_normal = v_normal + gate_normal.unsqueeze(-1) * ve_normal

            # Apply RoPE to normal heads
            cos, sin = cos_sin
            q_normal = apply_rotary_emb(q_normal, cos, sin)
            k_normal = apply_rotary_emb(k_normal, cos, sin)
            q_normal, k_normal = norm(q_normal), norm(k_normal)

            # Flash attention for normal heads
            if kv_cache is None:
                y_normal = flash_attn.flash_attn_func(q_normal, k_normal, v_normal, causal=True, window_size=window_size)
            else:
                # TODO: Handle KV cache for split heads
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y_normal = flash_attn.flash_attn_with_kvcache(
                    q_normal, k_cache, v_cache,
                    k=k_normal, v=v_normal,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

            outputs.append(y_normal.contiguous().view(B, T, -1))

        # Process nano heads (no RoPE!)
        if self.has_nanoheads:
            q_nano = self.c_q_nano(x).view(B, T, self.nano_heads, self.nano_head_dim)
            k_nano = self.c_k_nano(x).view(B, T, self.nano_kv_heads, self.nano_head_dim)
            v_nano = self.c_v_nano(x).view(B, T, self.nano_kv_heads, self.nano_head_dim)

            # Value residual for nano heads
            if ve is not None:
                ve_nano_start = self.normal_kv_heads * self.normal_head_dim
                ve_nano_dim = self.nano_kv_heads * self.nano_head_dim
                ve_nano = ve[..., ve_nano_start:ve_nano_start + ve_nano_dim].view(B, T, self.nano_kv_heads, self.nano_head_dim)
                gate_nano = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels])[..., self.normal_kv_heads:])
                v_nano = v_nano + gate_nano.unsqueeze(-1) * ve_nano

            # NO RoPE for nano heads, just QK norm
            q_nano, k_nano = norm(q_nano), norm(k_nano)

            # Flash attention for nano heads
            if kv_cache is None:
                y_nano = flash_attn.flash_attn_func(q_nano, k_nano, v_nano, causal=True, window_size=window_size)
            else:
                # TODO: Handle KV cache for nano heads properly
                # For now, simplified version
                y_nano = flash_attn.flash_attn_func(q_nano, k_nano, v_nano, causal=True, window_size=window_size)

            outputs.append(y_nano.contiguous().view(B, T, -1))

        # Concatenate all head outputs and project
        y = torch.cat(outputs, dim=-1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        # For nanoheads, we need to calculate the total KV dimension properly
        # Each layer can have different splits, but for simplicity we use the first layer's config
        dummy_attn = CausalSelfAttention(config, 0)
        kv_dim = dummy_attn.normal_kv_heads * dummy_attn.normal_head_dim + dummy_attn.nano_kv_heads * dummy_attn.nano_head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            # Initialize normal head projections
            if block.attn.normal_heads > 0:
                torch.nn.init.uniform_(block.attn.c_q_normal.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k_normal.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v_normal.weight, -s, s)
            # Initialize nano head projections
            if block.attn.has_nanoheads:
                torch.nn.init.uniform_(block.attn.c_q_nano.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k_nano.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v_nano.weight, -s, s)
            # Output projection and MLP
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
