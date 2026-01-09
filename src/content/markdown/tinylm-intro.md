---
title: "Inside TinyLM: How I Built a Transformer I Could Actually Read"
date: "2026-01-08"
tags: ["machine-learning", "transformers", "pytorch", "cuda"]
summary: "A walkthrough of building a transformer from scratch - RMSNorm vs LayerNorm, RoPE's rotation trick, the KV cache that makes generation not suck, and what training curves actually tell you."
readTime: "15 min"
featured: true
---

# Inside TinyLM: How I Built a Transformer I Could Actually Read

I wanted to understand how transformers work, not the attention diagram everyone draws, but the code that runs when you call `model.forward()`.

So I opened the LLaMA source. Then Hugging Face's implementation. Then GPT-NeoX. Each time I hit the same wall: thousands of files, abstraction layers, configuration systems that needed their own documentation. I could follow the math on paper, but I couldn't point to the line where Q meets K.

So I built my own: ~6,800 lines of Python, two architecture presets, and a transformer small enough to read in an afternoon.

This isn't a tutorial. It's what I learned building it: what clicked, what surprised me, and enough implementation detail that you could build your own.

---

## The Core Loop

Every transformer forward pass is basically the same loop. Here's a simplified version of the real forward path:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [batch, seq_len] token indices

    h = self.embedding(x)  # [batch, seq_len, dim]

    for i, block in enumerate(self.blocks):
        h = block(h, pos_ctx, cache, layer_idx=i, start_pos=start_pos)

    h = self.norm(h)           # Final normalization
    logits = self.head(h)      # [batch, seq_len, vocab_size]
    return logits
```

Structurally it's boring: embed → repeat block → norm → head. The real complexity lives inside `block()`:

```python
def forward(self, x, pos_ctx, cache, layer_idx, start_pos):
    # Pre-norm: normalize before each sublayer
    x = x + self.attn(self.norm1(x), pos_ctx, cache, layer_idx, start_pos)
    x = x + self.mlp(self.norm2(x))
    return x
```

Two residual branches. Two normalizations. One attention and one MLP. Repeat N times.

What makes this hard in practice isn't the shape of the loop. It's the constant fight against instability, bandwidth, and latency.

---

## The Parts That Surprised Me

### [RMSNorm](https://arxiv.org/abs/1910.07467): Fewer Moving Parts

LayerNorm does two things: <span style="color: #f97316">centers</span> (subtract mean) and <span style="color: #22d3ee">scales</span> (divide by std). RMSNorm drops the centering:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; overflow-x: auto; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># LayerNorm: two reductions (mean, then variance)</span>
y = (<span style="color: #f97316">x - mean(x)</span>) / std(x) * <span style="color: #22d3ee">γ</span> + <span style="color: #9ca3af">β</span>

<span style="color: #6c7086"># RMSNorm: one reduction (mean of squares)</span>
y = x / sqrt(mean(x²) + ε) * <span style="color: #22d3ee">γ</span></pre>
</div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.85rem; flex-wrap: wrap;">
  <span><span style="color: #f97316">■</span> centering</span>
  <span><span style="color: #22d3ee">■</span> learned scale</span>
  <span><span style="color: #9ca3af">■</span> learned bias</span>
</div>

Why does dropping mean subtraction work? The [RMSNorm paper](https://arxiv.org/abs/1910.07467)'s core claim is empirical: you can drop mean-centering and still train well. What matters is controlling activation scale. Without normalization, residual streams tend to drift and variance accumulates with depth.

In practical terms: RMSNorm gives you a stabilizer with one reduction instead of two.

The CUDA kernel makes the dataflow concrete:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1.25rem; font-family: monospace; margin: 1rem 0; overflow-x: auto;"><pre style="margin: 0; color: #cdd6f4; line-height: 1.5; font-size: 0.85rem;"><span style="color: #6c7086">// One block per row, templatized for fp16/fp32</span>
template&lt;typename scalar_t&gt;
__global__ void rmsnorm_fwd_kernel(
    const scalar_t* __restrict__ x, const scalar_t* __restrict__ w,
    scalar_t* y, float* <span style="color: #f9e2af">inv_rms_out</span>, int hidden, float eps) {

  int row = blockIdx.x;
  const scalar_t* x_row = x + row * hidden;

  float sumsq = 0.f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
      sumsq += <span style="color: #a6e3a1">to_float</span>(x_row[i]) * <span style="color: #a6e3a1">to_float</span>(x_row[i]);
  float reduced = <span style="color: #89b4fa">blockReduceSum</span>&lt;float&gt;(sumsq);  <span style="color: #6c7086">// warp shuffles</span>

  <span style="color: #f9e2af">__shared__</span> float <span style="color: #f9e2af">s_inv_rms</span>;
  if (threadIdx.x == 0) {
      <span style="color: #f9e2af">s_inv_rms</span> = <span style="color: #a6e3a1">rsqrtf</span>(reduced / hidden + eps);
      <span style="color: #f9e2af">inv_rms_out</span>[row] = <span style="color: #f9e2af">s_inv_rms</span>;  <span style="color: #6c7086">// cache for backward</span>
  }
  <span style="color: #f9e2af">__syncthreads</span>();

  scalar_t* y_row = y + row * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x)
      y_row[i] = <span style="color: #a6e3a1">from_float</span>&lt;scalar_t&gt;(
          <span style="color: #a6e3a1">to_float</span>(x_row[i]) * <span style="color: #f9e2af">s_inv_rms</span> * <span style="color: #a6e3a1">to_float</span>(<span style="color: #a78bfa">w</span>[i]));
}</pre></div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.8rem; flex-wrap: wrap;">
  <span><span style="color: #89b4fa">■</span> parallel reduce</span>
  <span><span style="color: #f9e2af">■</span> shared mem / cached for bwd</span>
  <span><span style="color: #a6e3a1">■</span> type conversion (fp16/fp32)</span>
  <span><span style="color: #a78bfa">■</span> learned scale γ</span>
</div>

On NVIDIA GPUs, `rsqrtf` is a fast device intrinsic (SFU-backed), which is why [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) recommend it for normalization. We compute it once, broadcast via shared memory, and reuse it across threads. The reduction uses warp shuffles (no atomics). Global memory traffic is the unavoidable part (read `x`, read `weight`, write `y`), but we avoid extra global traffic just to compute the reduction.

I cache `inv_rms` for backward. The backward needs the same value, and recomputing it would burn bandwidth for no gain.

---

### [RoPE](https://arxiv.org/abs/2104.09864) (Rotary Position Embedding): Position as Rotation

Most positional encodings add a position vector to the embedding. RoPE does something cleaner: it rotates query and key vectors as a function of position.

A 512-dim query becomes 256 independent 2D rotations:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1.25rem; font-family: monospace; margin: 1rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #cdd6f4; line-height: 1.6; font-size: 0.85rem;"><span style="color: #6c7086">q = [</span><span style="color: #89b4fa">q₀, q₁</span><span style="color: #6c7086">,</span> <span style="color: #a78bfa">q₂, q₃</span><span style="color: #6c7086">,</span> <span style="color: #f9e2af">q₄, q₅</span><span style="color: #6c7086">, ...]</span>      <span style="color: #6c7086"># 512 dimensions</span>
       <span style="color: #89b4fa">└──┘</span>   <span style="color: #a78bfa">└──┘</span>   <span style="color: #f9e2af">└──┘</span>
       <span style="color: #89b4fa">2D</span>     <span style="color: #a78bfa">2D</span>     <span style="color: #f9e2af">2D</span>          <span style="color: #6c7086"># 256 pairs</span>

<span style="color: #cdd6f4">At position </span><span style="color: #4ade80">t</span><span style="color: #cdd6f4">, rotate each pair by </span><span style="color: #4ade80">t</span><span style="color: #cdd6f4"> × </span><span style="color: #f38ba8">θᵢ</span><span style="color: #cdd6f4">:</span>

  <span style="color: #89b4fa">pair 0</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₀</span>     <span style="color: #6c7086"># fast rotation</span>
  <span style="color: #a78bfa">pair 1</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₁</span>     <span style="color: #6c7086"># medium</span>
  <span style="color: #f9e2af">pair 2</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₂</span>     <span style="color: #6c7086"># slower</span>
  <span style="color: #6c7086">...                           # inv_freq[i] = 1 / base^(2i/d) (angles are t * inv_freq[i])</span>
  <span style="color: #6c7086">pair 255</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₂₅₅</span>  <span style="color: #6c7086"># very slow</span></pre>
</div>

**Why this gives relative position.** The rotations "subtract" in the dot product: a query rotated by `t_q θ` dotted with a key rotated by `t_k θ` depends on `(t_q - t_k)θ`. Attention scores become a function of how far apart tokens are, not just their absolute indices.

**Why multiple frequencies.** Some pairs rotate quickly (good for local precision), others rotate slowly (good for long-range structure). It's the same intuition as Fourier features.

Implementation-wise, RoPE is mostly precompute + a cheap per-token rotate:

```python
def precompute(self, max_seq_len, device):
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))
    t = torch.arange(max_seq_len)
    freqs = torch.einsum('t,f->tf', t, inv_freq)

    self.sin = torch.sin(torch.cat([freqs, freqs], dim=-1))
    self.cos = torch.cos(torch.cat([freqs, freqs], dim=-1))
```

```python
def apply(self, x, start_pos):
    sin = self.sin[start_pos:start_pos + seq_len]
    cos = self.cos[start_pos:start_pos + seq_len]

    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

No learned parameters, no additive interference with content embeddings, and it often extrapolates better than learned absolute embeddings, though performance can still degrade at lengths far beyond training.

---

### [SwiGLU](https://arxiv.org/abs/2002.05202): The Third Projection

A standard transformer MLP is "up → activation → down." It works, but the nonlinearity is doing too much: it decides what to compute *and* what to keep.

SwiGLU splits those roles:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; overflow-x: auto; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #a78bfa">gate</span> = silu(W_gate @ x)     <span style="color: #6c7086"># should we use this?</span>
<span style="color: #22d3ee">up</span>   = W_up @ x             <span style="color: #6c7086"># what value do we compute?</span>
out  = <span style="color: #f97316">W_down</span> @ (<span style="color: #a78bfa">gate</span> * <span style="color: #22d3ee">up</span>)</pre>
</div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.8rem;">
  <span><span style="color: #a78bfa">■</span> gate (controls flow)</span>
  <span><span style="color: #22d3ee">■</span> up (computes values)</span>
  <span><span style="color: #f97316">■</span> down (projects back)</span>
</div>

The two projections see the same input but learn different functions. A dimension only contributes if both agree: `gate * up`.

Three projections sounds like 50% more parameters. The trick is to shrink the hidden size:

<div style="background: #1e1e2e; border-radius: 8px; padding: 0.75rem 1.25rem; font-family: monospace; margin: 1rem 0; font-size: 0.85rem;">
<pre style="margin: 0; color: #cdd6f4;">Standard: 2 × dim × <span style="color: #fab387">4d</span>   = <span style="color: #a6e3a1">8d²</span>
SwiGLU:   3 × dim × <span style="color: #fab387">8d/3</span> = <span style="color: #a6e3a1">8d²</span>  <span style="color: #6c7086">← same parameter count (with this hidden size)</span></pre>
</div>

Then round for GPU-friendly alignment:

```python
hidden_dim = int(dim * 4 * 2 / 3)
hidden_dim = 256 * ((hidden_dim + 255) // 256)
```

---

## Making Generation Not Suck

### The KV Cache Problem

Training sees full sequences. Generation emits one token at a time. If you naively call the model on the growing prefix, you recompute key/value projections for the entire history on every step.

But keys and values for past tokens don't change. Cache them once, reuse them forever.

<span style="color: #f38ba8">**Without caching**</span>

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># </span><span style="color: #f38ba8">O(n²) total</span><span style="color: #6c7086">: recompute K,V for the whole prefix each step</span>
for i in range(n):
    logits = model(<span style="color: #f38ba8">all_tokens[:i+1]</span>)</pre>
</div>

<span style="color: #a6e3a1">**With caching**</span>

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># Pre-allocate</span>
<span style="color: #89b4fa">K</span>[max_seq, dim], <span style="color: #f9e2af">V</span>[max_seq, dim] = zeros(...)

<span style="color: #6c7086"># Each token:</span>
<span style="color: #89b4fa">K</span>[pos], <span style="color: #f9e2af">V</span>[pos] = project(new_token)
attn = Q @ <span style="color: #89b4fa">K</span>[:pos].T</pre>
</div>

Now you're roughly **O(n)** per token instead of O(n²) total recompute.

---

### Pre-allocation Matters

Pre-allocating `[batch, n_kv_heads, max_seq_len, head_dim]` means zero allocations during decoding: just indexed writes. If you grow tensors with `torch.cat`, you'll pay allocator overhead and eventually fragment memory.

Pre-allocation costs you reserved memory, but decoding typically has a known max length anyway.

---

### [GQA](https://arxiv.org/abs/2305.13245) (Grouped-Query Attention): Trading KV Capacity for Memory

KV cache is often the inference memory bottleneck. The size is easy to compute:

```
KV cache bytes = 2 × B × n_kv_heads × T × d_head × bytes_per_elem
```

For LLaMA-2 7B (fp16, batch=1, seq=2048, n_kv=32, d_head=128):
- **Per layer**: ~32 MB
- **Across 32 layers**: ~1 GB total

GQA reduces `n_kv_heads`. Multiple query heads share each KV head:

```python
# MHA (Multi-Head Attention): n_heads = n_kv_heads = 32
# GQA (Grouped-Query Attention): n_heads = 32, n_kv_heads = 8
# MQA (Multi-Query Attention): n_heads = 32, n_kv_heads = 1
```

For a 7B model (32 layers, fp16, 2048 ctx):

| Variant | Per Layer | Total Model |
|---------|-----------|-------------|
| MHA (32 KV heads) | ~32 MB | ~1 GB |
| GQA (8 KV heads) | ~8 MB | ~256 MB |
| MQA (1 KV head) | ~1 MB | ~32 MB |

These scale linearly with batch size, context length, KV heads, and dtype bytes (fp32 doubles them).

[LLaMA 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json) uses GQA with 8 KV heads for 64 Q heads, an 8× reduction in KV cache, and a big reason GQA shows up in large models.

The implementation is just "repeat KV heads to match Q heads":

```python
def _repeat_kv(self, kv):
    """Expand KV heads to match Q heads."""
    if self.n_rep == 1:
        return kv
    return kv.repeat_interleave(self.n_rep, dim=1)
```

---

## What the Training Curves Tell You

<div id="training-comparison-demo" class="my-8 not-prose"></div>

I trained two identical models on TinyStories, same size, data, and hyperparameters. Only difference: LLaMA-style (pre-norm, RMSNorm, RoPE, SwiGLU) vs GPT-style (post-norm, LayerNorm, learned positions, standard MLP).

After 25,000 steps:

| Architecture | Val Loss | Val PPL |
|--------------|----------|---------|
| LLaMA-style | 1.25 | 3.49 |
| GPT-style | 1.33 | 3.79 |

The more interesting part is the shape: LLaMA-style pulls ahead early and then the curves run roughly parallel. That's a hint the architecture changes are affecting optimization dynamics more than capacity.

None of this is shocking if you've read the papers. What surprised me is how clearly it shows up at tiny scale.

<details>
<summary><strong>Full training config</strong></summary>

```yaml
model:
  dim: 384
  n_layers: 6
  n_heads: 6
  max_seq_len: 4096

training:
  batch_size: 32
  seq_len: 512
  lr: 0.0003
  weight_decay: 0.1
  warmup_steps: 500
  grad_clip: 1.0

hardware: Single RTX 3090
```

</details>

---

## If You Want to Build Your Own

I'm not going to tell you to use my code. But if you're starting a model project from scratch, this is what would have saved me time.

### Start with the forward pass

Write the dumbest possible version first: no cache, no mixed precision, no fused kernels.

```python
def forward(x):
    h = embed(x)
    for block in blocks:
        h = h + attention(norm(h))
        h = h + mlp(norm(h))
    return head(norm(h))
```

Make gradients and shapes sane. Then optimize.

### The registry pattern pays for itself

Registries look like over-engineering until the first time you want to swap one component without touching model code.

```python
@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module):
    ...

norm = NORM_REGISTRY.build("rmsnorm", dim=512)
```

### What I'd do differently

**Write tests earlier.** I had a RoPE bug that only showed up at sequence lengths > 512. A tiny reference-implementation test would've caught it.

**Profile before optimizing.** I spent a week on a CUDA kernel for a few percent end-to-end speedup. The real bottleneck was attention, and PyTorch's `scaled_dot_product_attention` (SDPA) already handles that well. (Note: SDPA picks Flash/memory-efficient/math kernels depending on your PyTorch build, GPU, dtype, and tensor shapes.)

**Keep configs boring.** Hydra is powerful, but it adds conceptual overhead. For many research codebases, dataclasses + argparse is enough.

---

## The Code

If you want to browse the implementation: [github.com/RetamalVictor/TinyLM-Lab](https://github.com/RetamalVictor/TinyLM-Lab)

The main files:
- `tinylm/model/transformer.py`: the forward pass
- `tinylm/components/normalization/rmsnorm.py`: RMSNorm dispatch
- `tinylm/components/positional/rope.py`: RoPE
- `tinylm/components/attention/mha.py`: MHA/GQA/MQA
- `csrc/rmsnorm_cuda.cu`: RMSNorm kernel

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): The original transformer
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971): LLaMA architecture
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467): RMSNorm
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864): RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): SwiGLU / gated MLPs
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245): Grouped-Query Attention
