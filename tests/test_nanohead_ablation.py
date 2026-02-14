"""
Tests for classic-only nanohead ablation mode.
"""

import pytest
import torch
import torch.nn.functional as F

import nanochat.flash_attention as fa_module
from nanochat.gpt import GPT, GPTConfig, CausalSelfAttention, calculate_nanohead_split


def _supports_sdpa_enable_gqa():
    """Return True if this torch build supports the enable_gqa keyword."""
    q = torch.randn(1, 1, 2, 8)
    k = torch.randn(1, 1, 2, 8)
    v = torch.randn(1, 1, 2, 8)
    try:
        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
        return True
    except TypeError:
        return False


def test_classic_split_preserved_and_nano_disabled():
    cfg_full = GPTConfig(
        n_embd=512,
        n_head=4,
        n_kv_head=4,
        nanohead_proportion=0.8,
        nanohead_dim=3,
        nanohead_ablation=False,
    )
    expected = calculate_nanohead_split(
        cfg_full.n_embd,
        cfg_full.n_head,
        cfg_full.n_kv_head,
        cfg_full.n_embd // cfg_full.n_head,
        cfg_full.nanohead_proportion,
        cfg_full.nanohead_dim,
    )
    exp_normal_heads, exp_normal_kv_heads, exp_normal_head_dim, *_ = expected

    cfg_ablation = GPTConfig(
        n_embd=512,
        n_head=4,
        n_kv_head=4,
        nanohead_proportion=0.8,
        nanohead_dim=3,
        nanohead_ablation=True,
    )
    attn = CausalSelfAttention(cfg_ablation, layer_idx=0)

    assert attn.normal_heads == exp_normal_heads
    assert attn.normal_kv_heads == exp_normal_kv_heads
    assert attn.normal_head_dim == exp_normal_head_dim
    assert attn.nano_heads == 0
    assert attn.nano_kv_heads == 0
    assert attn.has_nanoheads is False


def test_projection_width_matches_effective_attention_width():
    cfg_ablation = GPTConfig(
        n_embd=512,
        n_head=4,
        n_kv_head=4,
        nanohead_proportion=0.8,
        nanohead_dim=3,
        nanohead_ablation=True,
    )
    attn = CausalSelfAttention(cfg_ablation, layer_idx=0)
    expected_in = attn.normal_heads * attn.normal_head_dim + attn.nano_heads * attn.nano_head_dim

    assert attn.c_proj.in_features == expected_in
    assert attn.c_proj.in_features < cfg_ablation.n_embd


def test_baseline_unchanged_when_proportion_zero():
    cfg_base = GPTConfig(
        n_embd=512,
        n_head=4,
        n_kv_head=4,
        nanohead_proportion=0.0,
        nanohead_dim=3,
        nanohead_ablation=False,
    )
    cfg_ablation = GPTConfig(
        n_embd=512,
        n_head=4,
        n_kv_head=4,
        nanohead_proportion=0.0,
        nanohead_dim=3,
        nanohead_ablation=True,
    )

    attn_base = CausalSelfAttention(cfg_base, layer_idx=0)
    attn_ablation = CausalSelfAttention(cfg_ablation, layer_idx=0)

    assert attn_base.normal_heads == attn_ablation.normal_heads
    assert attn_base.normal_kv_heads == attn_ablation.normal_kv_heads
    assert attn_base.normal_head_dim == attn_ablation.normal_head_dim
    assert attn_base.nano_heads == attn_ablation.nano_heads == 0
    assert attn_base.nano_kv_heads == attn_ablation.nano_kv_heads == 0
    assert attn_base.has_nanoheads is False
    assert attn_ablation.has_nanoheads is False
    assert attn_base.c_proj.in_features == attn_ablation.c_proj.in_features == cfg_base.n_embd


@pytest.mark.skipif(not _supports_sdpa_enable_gqa(), reason="requires torch SDPA enable_gqa support")
def test_forward_backward_sanity_ablation_mode_cpu():
    cfg = GPTConfig(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=512,
        window_pattern="L",
        nanohead_proportion=0.8,
        nanohead_dim=3,
        nanohead_ablation=True,
    )
    model = GPT(cfg)
    model.init_weights()

    idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)
    targets = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long)

    # Force SDPA path for this CPU test in environments where FA3 is available on CUDA.
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss).item()

        loss.backward()
        assert model.transformer.h[0].attn.c_proj.weight.grad is not None
    finally:
        fa_module._override_impl = None
