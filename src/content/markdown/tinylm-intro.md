# Inside TinyLM: How I Built a Transformer I Could Actually Read

I wanted to understand how transformers work. Not the attention diagram everyone draws—the actual code that runs when you call `model.forward()`.

So I opened the LLaMA source. Then Hugging Face's implementation. Then GPT-NeoX. Each time I hit the same wall: thousands of files, layers of abstraction, configuration systems that needed their own documentation. I could trace the math on paper, but I couldn't point to the line where Q meets K.

So I built my own. ~6,800 lines of Python. Two architecture presets. A transformer small enough to read in an afternoon.

This isn't a tutorial. It's a walkthrough of what I learned building it—the parts that clicked, the parts that surprised me, and enough detail that you could build your own if you wanted to.

---

## The Core Loop

Every transformer forward pass is the same loop. Here's the actual code path:

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

That's it. Everything else is implementation detail inside `block()`:

```python
def forward(self, x, pos_ctx, cache, layer_idx, start_pos):
    # Pre-norm: normalize before each sublayer
    x = x + self.attn(self.norm1(x), pos_ctx, cache, layer_idx, start_pos)
    x = x + self.mlp(self.norm2(x))
    return x
```

Two residual connections. Two normalizations. One attention, one MLP. Repeat N times.

The complexity isn't in the structure—it's in making each piece fast and stable. That's where it gets interesting.

---

## The Parts That Surprised Me

### RMSNorm: Simpler Than I Expected

LayerNorm does two things: <span style="color: #f97316">centers the data</span> (subtract mean) and <span style="color: #22d3ee">scales it</span> (divide by std). RMSNorm drops the centering:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; overflow-x: auto; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># LayerNorm: 4 ops per element</span>
y = (<span style="color: #f97316">x - mean(x)</span>) / std(x) * <span style="color: #22d3ee">γ</span> + <span style="color: #9ca3af">β</span>

<span style="color: #6c7086"># RMSNorm: 2 ops per element</span>
y = x / sqrt(mean(x²) + ε) * <span style="color: #22d3ee">γ</span></pre>
</div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.85rem; flex-wrap: wrap;">
  <span><span style="color: #f97316">■</span> centering</span>
  <span><span style="color: #22d3ee">■</span> learned scale</span>
  <span><span style="color: #9ca3af">■</span> learned bias</span>
</div>

Why does dropping mean subtraction work? The [RMSNorm paper](https://arxiv.org/abs/1910.07467) argues that re-centering is redundant when you already have bias terms in subsequent layers. The network learns to compensate. What *does* matter is controlling activation magnitudes—without normalization, residual connections cause exponential growth through deep networks.

RMSNorm gives you the stability benefit with fewer operations: one reduction (sum of squares) instead of two (sum for mean, then sum of squared deviations).

The CUDA kernel makes this concrete—three phases, each with a different parallelism pattern:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1.25rem; font-family: monospace; margin: 1rem 0; overflow-x: auto;"><pre style="margin: 0; color: #cdd6f4; line-height: 1.5; font-size: 0.85rem;"><span style="color: #6c7086">// One CUDA block per row (row = one token's hidden state)</span>
__global__ void rmsnorm_fwd_kernel(...) {
  float sumsq = 0.f;
  for (int i = tid; i < hidden; i += stride)
      sumsq += x[i] * x[i];
  float reduced = <span style="color: #89b4fa">blockReduceSum</span>(sumsq);  <span style="color: #6c7086">// warp shuffle, no atomics</span>

  <span style="color: #f9e2af">__shared__</span> float <span style="color: #f9e2af">s_inv_rms</span>;
  if (tid == 0)
      <span style="color: #f9e2af">s_inv_rms</span> = <span style="color: #a6e3a1">rsqrtf</span>(reduced / hidden + eps);  <span style="color: #6c7086">// 1 HW instruction</span>
  <span style="color: #f9e2af">__syncthreads</span>();  <span style="color: #6c7086">// all threads now see s_inv_rms</span>

  for (int i = tid; i < hidden; i += stride)
      y[i] = x[i] * <span style="color: #f9e2af">s_inv_rms</span> * <span style="color: #a78bfa">weight</span>[i];
}</pre></div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.8rem; flex-wrap: wrap;">
  <span><span style="color: #89b4fa">■</span> parallel reduce</span>
  <span><span style="color: #f9e2af">■</span> shared memory</span>
  <span><span style="color: #a6e3a1">■</span> rsqrtf (1 instruction)</span>
  <span><span style="color: #a78bfa">■</span> learned scale γ</span>
</div>

`rsqrtf` (reciprocal square root) is a single hardware instruction on NVIDIA GPUs. We compute it once, broadcast via shared memory, and every thread reads it. The reduction uses warp shuffles—no atomics, no global memory traffic.

I cache `inv_rms` for the backward pass. Computing gradients through the normalization requires the same value, and recomputing it would double the memory bandwidth.

### RoPE: Position as Rotation

Most position encodings *add* a position vector to the embedding. RoPE does something geometrically cleaner: it *rotates* query and key vectors based on their position.

Your 512-dim query vector? RoPE treats it as 256 separate 2D vectors and rotates each one:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1.25rem; font-family: monospace; margin: 1rem 0; overflow-x: auto;">
<pre style="margin: 0; color: #cdd6f4; line-height: 1.6; font-size: 0.85rem;"><span style="color: #6c7086">q = [</span><span style="color: #89b4fa">q₀, q₁</span><span style="color: #6c7086">,</span> <span style="color: #a78bfa">q₂, q₃</span><span style="color: #6c7086">,</span> <span style="color: #f9e2af">q₄, q₅</span><span style="color: #6c7086">, ...]</span>      <span style="color: #6c7086"># 512 dimensions</span>
       <span style="color: #89b4fa">└──┘</span>   <span style="color: #a78bfa">└──┘</span>   <span style="color: #f9e2af">└──┘</span>
       <span style="color: #89b4fa">2D</span>     <span style="color: #a78bfa">2D</span>     <span style="color: #f9e2af">2D</span>          <span style="color: #6c7086"># 256 pairs</span>

<span style="color: #cdd6f4">At position </span><span style="color: #4ade80">t</span><span style="color: #cdd6f4">, rotate each pair by </span><span style="color: #4ade80">t</span><span style="color: #cdd6f4"> × </span><span style="color: #f38ba8">θᵢ</span><span style="color: #cdd6f4">:</span>

  <span style="color: #89b4fa">pair 0</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₀</span>     <span style="color: #6c7086"># θ₀ = 1.0        (fast rotation)</span>
  <span style="color: #a78bfa">pair 1</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₁</span>     <span style="color: #6c7086"># θ₁ = 0.68       (medium)</span>
  <span style="color: #f9e2af">pair 2</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₂</span>     <span style="color: #6c7086"># θ₂ = 0.46       (slower)</span>
  <span style="color: #6c7086">...                           # θᵢ = 10000^(-2i/d)</span>
  <span style="color: #6c7086">pair 255</span>: rotate by <span style="color: #4ade80">t</span> × <span style="color: #f38ba8">θ₂₅₅</span>  <span style="color: #6c7086"># θ₂₅₅ ≈ 0.0001  (very slow)</span></pre>
</div>

**Why rotation encodes *relative* position**: When you compute `q · k`, the rotations combine. If `q` at position 5 rotated by `5θ` and `k` at position 3 rotated by `3θ`, the dot product only sees the *difference*: `(5-3)θ = 2θ`. The attention score depends on "2 tokens apart"—not "positions 5 and 3."

**Why multiple frequencies**: Low-index pairs rotate fast—position 1 vs 2 looks very different. High-index pairs rotate slowly—position 1 vs 2 looks almost identical, but position 1 vs 100 is distinct. The model gets both local precision and global structure, like Fourier features.

The implementation precomputes sin/cos tables:

```python
def precompute(self, max_seq_len, device):
    # Inverse frequencies: θ_i = base^(-2i/d)
    inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2) / self.dim))

    # Position indices
    t = torch.arange(max_seq_len)

    # Frequency matrix: freqs[t, i] = t * θ_i
    freqs = torch.einsum('t,f->tf', t, inv_freq)

    # Cache sin and cos
    self.sin = torch.sin(torch.cat([freqs, freqs], dim=-1))
    self.cos = torch.cos(torch.cat([freqs, freqs], dim=-1))
```

During forward, applying the rotation is just:

```python
def apply(self, x, start_pos):
    sin = self.sin[start_pos:start_pos + seq_len]
    cos = self.cos[start_pos:start_pos + seq_len]

    # x1 = even indices (first element of each pair)
    # x2 = odd indices (second element of each pair)
    x1, x2 = x[..., ::2], x[..., 1::2]

    # Standard 2D rotation: (x,y) → (x·cos - y·sin, x·sin + y·cos)
    return torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

No learned parameters, no additive interference with the content embedding, and it extrapolates to longer sequences than seen during training (with some degradation).

### SwiGLU: The Third Projection

Standard transformer MLPs: project up to 4× width, apply activation, project back down. Simple, but the activation (ReLU, GELU) is a blunt instrument—it decides "on or off" based purely on magnitude.

SwiGLU separates *what* to compute from *whether* to use it:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; overflow-x: auto; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #a78bfa">gate</span> = silu(W_gate @ x)     <span style="color: #6c7086"># learned "should we use this?"</span>
<span style="color: #22d3ee">up</span>   = W_up @ x             <span style="color: #6c7086"># learned "what's the value?"</span>
out  = <span style="color: #f97316">W_down</span> @ (<span style="color: #a78bfa">gate</span> * <span style="color: #22d3ee">up</span>)  <span style="color: #6c7086"># element-wise gating</span></pre>
</div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.8rem;">
  <span><span style="color: #a78bfa">■</span> gate (controls flow)</span>
  <span><span style="color: #22d3ee">■</span> up (computes values)</span>
  <span><span style="color: #f97316">■</span> down (projects back)</span>
</div>

The <span style="color: #a78bfa">gate</span> and <span style="color: #22d3ee">up</span> projections see the same input but learn different functions. The gate learns *which* hidden dimensions are relevant for this input; up learns *what* values to produce. The element-wise product `gate * up` means a dimension only contributes if both agree it should.

SiLU (Swish) is `x * sigmoid(x)`—smooth, allows gradients everywhere, and lets the gate output negative values (unlike ReLU).

Three projections means 50% more parameters, right? Here's the trick—use a smaller hidden dimension:

<div style="background: #1e1e2e; border-radius: 8px; padding: 0.75rem 1.25rem; font-family: monospace; margin: 1rem 0; font-size: 0.85rem;">
<pre style="margin: 0; color: #cdd6f4;">Standard: 2 × dim × <span style="color: #fab387">4d</span>   = <span style="color: #a6e3a1">8d²</span>
SwiGLU:   3 × dim × <span style="color: #fab387">8d/3</span> = <span style="color: #a6e3a1">8d²</span>  <span style="color: #6c7086">← same!</span></pre>
</div>

Same parameter count, more expressiveness. The implementation rounds to multiples of 256 for GPU efficiency:

```python
hidden_dim = int(dim * 4 * 2 / 3)                 # 8/3 ≈ 2.67x instead of 4x
hidden_dim = 256 * ((hidden_dim + 255) // 256)    # align for tensor cores
```

---

## Making Generation Not Suck

### The KV Cache Problem

Training sees whole sequences; generation produces one token at a time. Naively, you'd recompute everything for each new token—but attention has a useful property: past tokens' keys and values don't change.

Think about what attention computes: `softmax(Q @ K.T) @ V`. The new token needs Q (its query), but K and V come from *all* positions. Past tokens' K and V projections are deterministic functions of their embeddings—they're the same every time. Only Q changes.

<span style="color: #f38ba8">**Without caching**</span>—reproject K, V for all n tokens on every step:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># </span><span style="color: #f38ba8">O(n²) total</span><span style="color: #6c7086"> — projecting K,V grows linearly, n steps = quadratic</span>
for i in range(n):
    logits = model(<span style="color: #f38ba8">all_tokens[:i+1]</span>)  <span style="color: #6c7086"># reproject ALL K,V</span></pre>
</div>

<span style="color: #a6e3a1">**With caching**</span>—project K, V once, store forever:

<div style="background: #1e1e2e; border-radius: 8px; padding: 1rem 1.25rem; font-family: monospace; margin: 1rem 0; line-height: 1.5; font-size: 0.9rem;">
<pre style="margin: 0; color: #cdd6f4;"><span style="color: #6c7086"># Pre-allocate</span>
<span style="color: #89b4fa">K</span>[max_seq, dim], <span style="color: #f9e2af">V</span>[max_seq, dim] = zeros(...)

<span style="color: #6c7086"># On each token:</span>
<span style="color: #89b4fa">K</span>[<span style="color: #a6e3a1">pos</span>], <span style="color: #f9e2af">V</span>[<span style="color: #a6e3a1">pos</span>] = project(new_token)  <span style="color: #6c7086"># store</span>
attn = Q @ <span style="color: #89b4fa">K</span>[:pos].T                    <span style="color: #6c7086"># attend to all cached</span></pre>
</div>

<div style="display: flex; gap: 1.5rem; margin: 0.5rem 0 1rem; font-size: 0.8rem;">
  <span><span style="color: #89b4fa">■</span> K cache</span>
  <span><span style="color: #f9e2af">■</span> V cache</span>
  <span><span style="color: #a6e3a1">■</span> position</span>
</div>

The new token's Q attends to all cached <span style="color: #89b4fa">K</span>,<span style="color: #f9e2af">V</span>—now <span style="color: #a6e3a1">O(n) per token</span> instead of O(n²).

### Pre-allocation Matters

Why pre-allocate the full cache upfront? Two reasons: fragmentation and speed.

If you grow tensors dynamically (`torch.cat` on each step), you fragment GPU memory. The allocator finds gaps, copies data, frees old blocks. After enough tokens, you can't allocate contiguous blocks even with plenty of total free memory.

Pre-allocating `[batch, n_kv_heads, max_seq_len, head_dim]` means zero allocations during generation—just pointer arithmetic to write at the next position. The cost is reserving memory you might not use, but for generation you usually know your max length upfront.

### GQA: Trading KV Capacity for Speed

The KV cache is often the memory bottleneck during inference. For a 7B model generating 2048 tokens, you're storing 537 MB of key-value pairs—per layer.

Grouped-Query Attention (GQA) observes that K and V don't need as much capacity as Q. Multiple Q heads can share the same K,V head:

```python
# MHA: n_heads = n_kv_heads = 32    (every Q head has its own KV)
# GQA: n_heads = 32, n_kv_heads = 8  (4 Q heads share each KV head)
# MQA: n_heads = 32, n_kv_heads = 1  (all Q heads share one KV head)
```

The KV cache scales with `n_kv_heads`, not `n_heads`:

| Variant | KV Cache Size (2048 ctx) | Quality |
|---------|-------------------------|---------|
| MHA (32 KV heads) | 537 MB | Baseline |
| GQA (8 KV heads) | 134 MB | ~Same |
| MQA (1 KV head) | 17 MB | Slight loss |

LLaMA 2 70B uses GQA with 8 KV heads for 64 Q heads—an 8× reduction in KV cache with no measurable quality loss. The intuition: Q needs capacity to ask diverse questions, but K and V just need to store the context. The implementation handles all three variants:

```python
def _repeat_kv(self, kv):
    """Expand KV heads to match Q heads."""
    if self.n_rep == 1:  # MHA: no expansion needed
        return kv
    # GQA/MQA: repeat each KV head n_rep times
    return kv.repeat_interleave(self.n_rep, dim=1)
```

One function, three attention variants.

---

## What the Training Curves Tell You

<div id="training-comparison-demo" class="my-8 not-prose"></div>

I trained two identical models on TinyStories—same size, same data, same hyperparameters. Only difference: LLaMA-style (pre-norm, RMSNorm, RoPE, SwiGLU) vs GPT-style (post-norm, LayerNorm, learned positions, standard MLP).

Results after 25,000 steps:

| Architecture | Val Loss | Val PPL |
|--------------|----------|---------|
| LLaMA-style | 1.25 | 3.49 |
| GPT-style | 1.33 | 3.79 |

The curves tell a more interesting story than the final numbers. LLaMA-style pulls ahead early and stays ahead. The gap opens in the first few thousand steps, then the curves run roughly parallel.

This suggests the architectural differences affect optimization dynamics more than model capacity. Pre-norm gives cleaner gradient flow. RMSNorm is simpler to optimize. RoPE's relative position encoding might generalize better to the sequence lengths in TinyStories.

None of these are surprising if you've read the papers. What surprised me was how clearly they showed up at tiny scale. A 6-layer, 384-dim model is enough to see architectural effects that matter at 7B.

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

I'm not going to tell you to use my code. But here's what I learned that might save you time:

### Start with the forward pass

Write the dumbest possible implementation first. No KV cache, no mixed precision, no kernel fusion. Just:

```python
def forward(x):
    h = embed(x)
    for block in blocks:
        h = h + attention(norm(h))
        h = h + mlp(norm(h))
    return head(norm(h))
```

Make it produce plausible gradients. Then optimize.

### The registry pattern is worth it

Every component type (norm, attention, MLP, position encoding) has a registry:

```python
@NORM_REGISTRY.register("rmsnorm")
class RMSNorm(nn.Module):
    ...

# Later:
norm = NORM_REGISTRY.build("rmsnorm", dim=512)
```

This looks like over-engineering until you want to test a new attention variant. Then it's "add a file, add a decorator, change a config string."

### What I'd do differently

**More tests earlier.** I caught a RoPE bug that only showed up at sequence lengths > 512. A simple "outputs match reference implementation" test would have caught it immediately.

**Profile before optimizing.** I spent a week on a CUDA kernel that improved end-to-end speed by 3%. The real bottleneck was attention—and PyTorch's `scaled_dot_product_attention` already handles that.

**Simpler configs.** I used Hydra for configuration, which is powerful but adds complexity. For a research codebase, dataclasses + argparse might be enough.

---

## The Code

If you want to look at the implementation: [github.com/RetamalVictor/TinyLM-Lab](https://github.com/RetamalVictor/TinyLM-Lab)

The interesting files:
- `tinylm/model/transformer.py` — the forward pass
- `tinylm/components/normalization/rmsnorm.py` — RMSNorm with kernel dispatch
- `tinylm/components/positional/rope.py` — RoPE implementation
- `tinylm/components/attention/mha.py` — unified MHA/GQA/MQA
- `csrc/rmsnorm_cuda.cu` — the CUDA kernel

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — LLaMA architecture
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU and gated MLPs
