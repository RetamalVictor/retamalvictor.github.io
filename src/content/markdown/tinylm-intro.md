# TinyLM Lab: A Hackable Transformer for Research Prototyping

~6,800 lines of Python under `tinylm/`. Two architecture presets. A transformer stack you can read end-to-end.

---

After five years as an ML engineer, I finally had to work with LLMs. My new position demanded it, and I was behind.

The papers made sense. The math clicked. But when I opened a production codebase to understand how transformers actually work, I hit a wall. Thousands of files. Layers of abstraction. Configuration systems that required their own documentation.

I didn't need to deploy a model. I needed to *understand* one.

So I built TinyLM Lab: a transformer framework small enough to read in an afternoon, complete enough to train real models, and modular enough to test research ideas without fighting infrastructure.

The first time I swapped RoPE for learned positions and watched training behavior change, it finally *clicked* why architecture "details" aren't details.

What started as a learning exercise is now my go-to testbed for prototyping. When a paper catches my attention, I implement it in TinyLM first. If it works at tiny scale, it's worth pursuing further—though there are exceptions.

---

## Who It's For (and Not For)

TinyLM Lab is for researchers, students, and engineers who want to *edit the model itself*—not just fine-tune one.

**Good fit:**
- Implementing paper ideas at small scale
- Learning transformer internals by modifying them
- Quick architecture experiments before committing to large runs

**Not a good fit:**
- Production deployment at scale
- Maximum inference throughput
- Using pretrained weights (use HuggingFace)

If you're optimizing serving or training at billions of parameters, you'll be happier in a production framework. They're built for scale and safety; TinyLM Lab is built for comprehension and iteration.

---

## What TinyLM Lab Is

A transformer framework optimized for clarity first; scale is explicitly out of scope.

**The numbers:**
- ~6,800 lines of Python under `tinylm/` (measured: `find tinylm -name "*.py" -not -path "*__pycache__*" | xargs wc -l`)
- 2 architecture presets (LLaMA-style, GPT-style post-norm)
- Full training pipeline with gradient accumulation and checkpointing
- KV-cache generation
- Custom CUDA kernel for RMSNorm

**What's included:**

| Component | Current | Code Location |
|-----------|---------|---------------|
| Architectures | LLaMA, GPT (post-norm) | [`tinylm/architectures/__init__.py`][arch] |
| Attention | MHA, GQA, MQA | [`tinylm/components/attention/mha.py:17-18`][mha] |
| Attention Backends | standard, flash, memory_efficient | [`tinylm/components/attention/ops/`][ops] |
| Positional | RoPE, Learned | [`tinylm/components/positional/`][pos] |
| Normalization | RMSNorm (with CUDA), LayerNorm | [`tinylm/components/normalization/`][norm] |
| MLP | Standard, Gated (SwiGLU) | [`tinylm/components/mlp/`][mlp] |
| Quantization | Ternary (via BitTorch) | [`tinylm/quant/`][quant] |

[arch]: https://github.com/RetamalVictor/TinyLM-Lab/blob/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/architectures/__init__.py
[mha]: https://github.com/RetamalVictor/TinyLM-Lab/blob/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/components/attention/mha.py#L17-L18
[ops]: https://github.com/RetamalVictor/TinyLM-Lab/tree/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/components/attention/ops
[pos]: https://github.com/RetamalVictor/TinyLM-Lab/tree/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/components/positional
[norm]: https://github.com/RetamalVictor/TinyLM-Lab/tree/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/components/normalization
[mlp]: https://github.com/RetamalVictor/TinyLM-Lab/tree/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/components/mlp
[quant]: https://github.com/RetamalVictor/TinyLM-Lab/tree/e55b30cb7482017852846116ea72fa4fbd12dda6/tinylm/quant

**Note:** The attention backends use PyTorch's `scaled_dot_product_attention` (SDPA), which selects among available kernels (Flash, memory-efficient, math) depending on your PyTorch build, GPU, dtype, and tensor shapes. The "flash" and "memory_efficient" ops configure SDPA's kernel preferences—they *encourage* a specific backend, but don't guarantee it. See [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for details.

No hidden config loading. No framework magic. The constructor tells you what you're building:

```python
from tinylm import TinyLM

model = TinyLM(
    vocab_size=32000,
    dim=512,
    n_layers=8,
    n_heads=8,
    architecture="llama",  # or "gpt" (classic post-norm decoder)
)
```

### Code Map (Start Here)

| Path | What's There |
|------|--------------|
| `tinylm/model/transformer.py` | Top-level model assembly |
| `tinylm/architectures/` | LLaMA vs GPT preset configs |
| `tinylm/components/attention/mha.py` | MHA/GQA/MQA implementation |
| `tinylm/components/attention/ops/` | SDPA backend preferences |
| `tinylm/components/normalization/` | RMSNorm (with CUDA), LayerNorm |
| `tinylm/training/trainer.py` | Training loop + checkpointing |
| `tinylm/cli/` | Train, infer, evaluate commands |

---

## Why It Exists

### The Problem

Production frameworks are powerful but opaque. When I wanted to understand how RoPE actually works, I had to trace through five files and three abstraction layers. When I wanted to add a custom attention variant, I spent more time understanding the framework than implementing the idea.

There's a gap between tutorial code and real implementations. Tutorials show you the math. Production code shows you the engineering. Neither shows you both in a way you can modify.

### What I Needed

1. **Implement papers quickly** - New attention variant? Add one file, register it, done.
2. **See what's happening** - No magic. Every tensor transformation is explicit.
3. **Test ideas at small scale** - If an idea can't show a signal at small scale, it's usually not worth paying 7B-scale rent (unless the idea is explicitly about scaling behavior).

### The Solution

Build the minimum viable transformer. Compact enough to read in a sitting. Complete enough to train real models. Modular enough to swap any component.

---

## The Architecture System

The difference between LLaMA-style and GPT-style (post-norm) is one config:

```python
# tinylm/architectures/__init__.py

ARCHITECTURES = {
    "llama": ArchitectureConfig(
        norm_type="rmsnorm",
        norm_position="pre",
        pos_emb_type="rope",
        activation="silu",
        mlp_type="gated",      # SwiGLU
        use_bias=False,
    ),
    "gpt": ArchitectureConfig(
        norm_type="layernorm",
        norm_position="post",
        pos_emb_type="learned",
        activation="gelu",
        mlp_type="standard",
        use_bias=True,
    ),
}
```

### What This Means

| Aspect | LLaMA-style | GPT-style (post-norm) |
|--------|-------------|----------------------|
| Normalization | RMSNorm (computationally simpler, no mean-centering) | LayerNorm |
| Norm Position | Pre-norm (before attn/MLP) | Post-norm (after) |
| Position Encoding | RoPE (rotary, adds relative-position behavior inside attention) | Learned (absolute, added to embeddings) |
| Activation | SiLU | GELU |
| MLP | Gated FFN variant with extra projection (SwiGLU) | Standard (2 matrices) |
| Bias | No | Yes |

Note: The "GPT" preset uses post-norm, which is closer to the original Transformer than GPT-2/3 (which use pre-norm). I kept the name for simplicity, but it's really "classic post-norm decoder."

---

## Experiment: LLaMA vs GPT on TinyStories

<div id="training-comparison-demo" class="my-8 not-prose"></div>

In my runs (single seed, config below), the LLaMA-style preset converged faster on TinyStories.

**Results after 25,000 steps:**

| Architecture | Val Loss | Val PPL |
|--------------|----------|---------|
| **LLaMA-style** | **1.25** | **3.49** |
| GPT-style (post-norm) | 1.33 | 3.79 |

<details>
<summary><strong>Reproducibility: Full Config</strong></summary>

```yaml
# Both runs used identical settings except architecture
model:
  name: small
  dim: 384
  n_layers: 6
  n_heads: 6
  dropout: 0.1
  max_seq_len: 4096

training:
  steps: 50000  # stopped at 25000
  batch_size: 32
  seq_len: 512
  lr: 0.0003
  weight_decay: 0.1
  betas: [0.9, 0.95]
  lr_schedule: cosine
  warmup_steps: 500
  grad_clip: 1.0
  grad_accum_steps: 2
  mixed_precision: true

data:
  name: tinystories

tokenizer:
  type: bytelevel  # ByteLevel BPE via HuggingFace tokenizers
  vocab_size: 4096

seed: 42
hardware: Single GPU (RTX 3090)
```

**Commit:** [`e55b30c`](https://github.com/RetamalVictor/TinyLM-Lab/commit/e55b30cb7482017852846116ea72fa4fbd12dda6) on `feature/browser-lm` branch

**Commands:**
```bash
# LLaMA run
uv run python -m tinylm.cli.train model=small model.architecture=llama data=tinystories training=long

# GPT run
uv run python -m tinylm.cli.train model=small model.architecture=gpt data=tinystories training=long
```

**Training curves:** Embedded in the interactive chart above. Raw data available in the blog repo under `src/data/training_curves.json`.

</details>

This is the kind of experiment TinyLM Lab makes trivial. Change one line in the config, retrain, compare.

---

## The Registry Pattern

Every component type has a registry:

```python
from tinylm.components.registry import NORM_REGISTRY

@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Usage anywhere
from tinylm.components import build_norm
norm = build_norm("rmsnorm", dim=512)
```

### Why This Matters

1. **Swap components without touching model code** - Change `norm_type` in config, done
2. **Add new variants in one file** - Register it, use it
3. **Config-driven experimentation** - No code changes for architecture sweeps

### Available Registries

| Registry | Components |
|----------|------------|
| `NORM_REGISTRY` | rmsnorm, layernorm |
| `ATTENTION_REGISTRY` | mha, gqa, mqa |
| `ATTENTION_OP_REGISTRY` | standard, flash, memory_efficient |
| `MLP_REGISTRY` | standard, gated |
| `POS_EMB_REGISTRY` | rope, learned |

---

## Adding a New Component

Say you read a paper with a new attention variant. Here's how to add it:

**Step 1: Create the op**

```python
# tinylm/components/attention/ops/sliding_window.py

from tinylm.components.registry import ATTENTION_OP_REGISTRY
from tinylm.components.attention.ops.base import AttentionOp

@ATTENTION_OP_REGISTRY.register("sliding_window")
class SlidingWindowAttention(AttentionOp):
    """Sliding window attention (toy implementation for clarity).

    Note: This is O(seq²) and allocates a new mask on every call.
    For production, you'd implement this as a fused kernel or block-sparse op.
    """

    def __init__(self, window_size: int = 4096, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def __call__(self, q, k, v, attn_mask=None, is_causal=True, training=False):
        B, H, T, D = q.shape

        # Create sliding window mask (O(seq²) - not efficient!)
        device = q.device
        positions = torch.arange(T, device=device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        window_mask = distance.abs() <= self.window_size

        if is_causal:
            causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            window_mask = window_mask & causal_mask

        # Standard scaled dot-product
        scale = D ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = attn.masked_fill(~window_mask, -1e9)  # Use -1e9 for fp16 safety
        attn = torch.softmax(attn, dim=-1)

        return torch.matmul(attn, v)
```

**Step 2: Use it**

```python
# Via architecture config
from tinylm.architectures import ArchitectureConfig

config = ArchitectureConfig(
    attention_op="sliding_window",
    # ... rest of config
)

# Or swap at runtime
from tinylm.components.attention import build_attention_op
model.blocks[0].attn.attention_op = build_attention_op(
    "sliding_window",
    window_size=2048
)
```

One file, one decorator, one config change.

---

## What's Good About TinyLM Lab

### 1. Readable

The core model fits in ~240 lines; attention in ~210 (including GQA/MQA). You can set breakpoints without spelunking through framework glue.

When something breaks, you can find it. When you want to understand a component, you can trace it end-to-end.

### 2. Modular

Every component is swappable. Architecture is config, not code.

### 3. Complete

Despite being small, TinyLM Lab includes:
- Training loop with gradient accumulation
- Checkpointing (save/load/resume)
- KV-cache for efficient generation
- Hydra configs for experiment management
- Custom CUDA kernel (RMSNorm)
- Attention via PyTorch SDPA (FlashAttention when available)
- Gradient checkpointing for memory efficiency

### 4. Extensible

The quantization system (BitTorch integration) shows how to extend TinyLM Lab for research. Ternary weights, straight-through estimators, packed inference—all plugged in via the registry pattern.

---

## Limitations

### Not Tested at Scale

I don't have the budget to validate at 7B+ parameters. TinyLM Lab is designed for tiny-to-small models (1M - 500M params).

If you need production scale, use something else.

### Missing Features

- No tensor parallelism (yet)
- No FSDP integration (yet)
- Limited tokenizer support (uses HuggingFace tokenizers)

### Planned / In Progress

| Feature | Status |
|---------|--------|
| Distributed training (DDP) | Planned |
| Cloud training (RunPod) | Planned |
| More architectures | Planned |

---

## Future Plans

### Coming First: Distributed Training

Cloud training support (likely RunPod or similar) with proper distributed architecture. DDP first, then FSDP.

### Then: Multi-Modality

Vision and audio encoders for robotics applications. The goal is using language models as world models for robotic control.

### Later: New Architectures

State space models like Falcon H1. The registry pattern makes adding new architecture types straightforward.

---

## Getting Started

```bash
git clone https://github.com/RetamalVictor/TinyLM-Lab.git
cd TinyLM-Lab
uv sync

# Optional: build CUDA RMSNorm
uv run python setup.py build_ext --inplace

# Prepare data
uv run python scripts/prepare_tinyshakespeare.py
uv run python scripts/prepare_tinystories.py

# Train LLaMA-style (default)
uv run python -m tinylm.cli.train model=small

# Train GPT-style
uv run python -m tinylm.cli.train model=small model.architecture=gpt

# Inference
uv run python -m tinylm.cli.infer \
    --ckpt outputs/.../best.pt \
    --prompt "Once upon a time"
```

---

## What's Next

This is the first post in a series:

1. **This post**: What TinyLM Lab is and why
2. **Next**: Building the browser inference engine (WebGPU + ternary weights)
3. **Then**: Training quantized models with TinyLM Lab + BitTorch

If you're implementing papers, prototyping ideas, or just want to understand transformers better, give TinyLM Lab a try. The code is the documentation.

---

## Receipts

| Item | Value |
|------|-------|
| Repo | [github.com/RetamalVictor/TinyLM-Lab](https://github.com/RetamalVictor/TinyLM-Lab) |
| Commit | [`e55b30c`](https://github.com/RetamalVictor/TinyLM-Lab/commit/e55b30cb7482017852846116ea72fa4fbd12dda6) |
| LOC | 6,796 (`find tinylm -name "*.py" -not -path "*__pycache__*" \| xargs wc -l`) |
| Training data | Embedded chart above; raw JSON in blog repo |

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - LLaMA architecture
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - SwiGLU

