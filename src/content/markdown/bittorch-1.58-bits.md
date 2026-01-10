---
title: "1.58 Bits Per Weight: Implementing BitNet from Paper to CUDA"
date: "2025-12-15"
tags: ["machine-learning", "cuda", "quantization", "pytorch"]
summary: "The story of building BitTorch: ternary neural networks from paper to CUDA. Why my first kernel was 20× slower than PyTorch, the gradient bug that cost a week, and what I learned about low-precision ML."
readTime: "18 min"
featured: true
---

# 1.58 Bits Per Weight: Implementing BitNet from Paper to CUDA

The first time I ran my CUDA kernel, it was **20× slower than PyTorch**.

I'd spent a week reading the BitNet paper, another week implementing quantization logic, and a weekend fighting `nvcc` compiler errors. Eventually, it compiled. I ran the benchmark expecting *something*. Maybe 2× faster? The paper promised multiplication-free inference, after all.

```
nn.Linear FP32:     0.28 ms
TernaryLinearCUDA:  5.9 ms
```

Over 20× slower. Not a typo.

I seriously considered stopping there. But a 7B-parameter model needs 28 GB in FP32, and my laptop has 8 GB of VRAM. If ternary weights actually worked, I could run models that had no business running on my hardware.

So I kept going.

Six kernel versions, 185 tests, one spectacular gradient bug, and a deep dive into memory bandwidth later, I got **single-batch inference within ~10% of cuBLAS** on favorable shapes. Not faster, but close enough that the 16× memory reduction actually matters. This is the story of building **BitTorch**: ternary neural networks from paper to CUDA.

---

## Live Demo

Want to see ternary inference in action? The demo below runs a character-level language model entirely in your browser (requires a modern browser with WebGPU enabled). The model uses 1.58-bit ternary weights, the same technique described in this post.

<div id="ternary-lm-demo" class="my-8 not-prose"></div>

Type a prompt and click "Generate" to watch character-by-character text generation powered by {-1, 0, +1} weights. Click "Under the Hood" to see memory savings and model architecture.

---

### What we'll cover

* Why $\log_2(3) \approx 1.58$ bits is the *theoretical* lower bound per weight
* The Straight-Through Estimator trick that makes training possible
* Packing ternary weights: 4 values per byte
* Writing CUDA kernels (and why your first one will be embarrassingly slow)
* The optimization journey: from 20× slower to 0.9× cuBLAS
* Honest benchmarks: what worked, what didn't, and why

Let's start at the beginning.

---

## Part 1: Why 1.58 Bits?

Before touching code, it's worth understanding what we're actually compressing.

Standard neural network weights are 32-bit floats, about four billion possible values per weight. The BitNet paper makes a provocative claim: for many tasks, you can get away with **three** values:

$$
\lbrace -1, 0, +1 \rbrace
$$

That's it. No decimals. No fine-grained precision.

From an information-theoretic perspective, three values require:

$$
\log_2(3) \approx 1.58 \text{ bits}
$$

Compared to 32 bits, that's a **20× theoretical compression**. A 7B-parameter model drops from 28 GB to roughly 1.4 GB. That's the difference between "needs a datacenter" and "runs on a laptop."

### From Theory to Bytes: Packed Ternary Format

The 1.58 bits is a theoretical limit. In practice, we pack 4 ternary values into each byte using 2-bit encoding:

| Value | Binary |
|-------|--------|
| 0     | `00`   |
| +1    | `01`   |
| -1    | `10`   |
| (reserved) | `11` |

Four weights per byte means **0.25 bytes per weight**, or **2 bits per weight in storage**. Not quite the 1.58-bit entropy bound, but close enough for real compression:

| Format | Bytes/Weight | 4096×4096 Matrix |
|--------|--------------|------------------|
| FP32   | 4.0          | 64 MB            |
| FP16   | 2.0          | 32 MB            |
| INT8   | 1.0          | 16 MB            |
| **Ternary (packed)** | **0.25** | **4 MB** |

That's **16× actual compression** vs FP32 (plus a few KB for per-channel scales).

The packing layout is straightforward: weights are stored **LSB-first**, four per byte:

```
Byte: [w3 w3 | w2 w2 | w1 w1 | w0 w0]
       bits    bits    bits    bits
       6-7     4-5     2-3     0-1
```

### Quantization in Practice

To map real-valued weights to $\lbrace -1,0,+1 \rbrace$, we use thresholding:

$$
w_q =
\begin{cases}
+1 & w > \tau \cr
-1 & w < -\tau \cr
0 & \text{otherwise}
\end{cases}
$$

The threshold $\tau$ controls sparsity. Too high, and everything becomes zero. Too low, and everything becomes ±1.

In practice:

$$
\tau = \lambda \cdot \max(|w|)
$$

with $\lambda = 0.05$. In plain English: *weights smaller than 5% of the largest weight in the layer get zeroed*.

### Scaling: The Detail That Matters

Quantized weights alone aren't enough. You need to rescale them during inference:

$$
W_{\text{effective}} = \alpha \cdot W_q
$$

Using a single global scale works, but it hurts accuracy. Different output channels naturally learn different magnitudes. Crushing them into one scale penalizes the smaller ones.

Per-channel scaling works better:

$$
\alpha_j = \max_i |w_{j,i}|
$$

It's a small implementation detail with a real accuracy impact.

### Forward Pass

Putting it together:

$$
y_j = \alpha_j \sum_i x_i \cdot w_{q,ij} + b_j
$$

Here's the theoretical appeal: multiplying by $w_q \in \lbrace -1,0,+1 \rbrace$ isn't really multiplication.

$$
w_q \cdot x = \begin{cases}
+x & \text{if } w_q = +1 \cr
-x & \text{if } w_q = -1 \cr
0 & \text{if } w_q = 0
\end{cases}
$$

In theory, we've replaced floating-point multiplies with conditional adds. In practice... we'll get there.

---

## Part 2: Making It Train

Compression is easy. Training is not.

### The Gradient Wall

Quantization is a step function. Step functions have zero gradient almost everywhere. Standard backprop gives you:

$$
\frac{\partial w_q}{\partial w} = 0
$$

No gradient means no learning. I stared at a flat loss curve for longer than I care to admit before internalizing this.

### The Straight-Through Estimator

The fix is a hack known as the **Straight-Through Estimator (STE)**:

* **Forward pass:** use quantized weights
* **Backward pass:** pretend quantization didn't happen

Formally:

$$
\frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial w_q}
$$

This isn't mathematically rigorous. It works anyway.

Intuition: even though $w_q$ is discrete, the underlying $w$ is continuous. Gradients accumulate until a threshold flips.

### PyTorch Implementation

PyTorch doesn't expose "use X forward, Y backward," but you can fake it:

```python
(w_eff - w).detach() + w
```

Forward returns `w_eff`. Backward routes gradients to `w`. Once this pattern clicks, a lot of quantization code suddenly makes sense.

### The Bug That Cost a Week

My first version computed scale like this:

```python
scale = w.abs().max(dim=1, keepdim=True)[0]
```

That scale depends on `w`. Gradients flowed through it. The result was subtle, silent corruption: my CUDA and Python gradients had a cosine similarity of 0.29.

The full gradient was something like:

```
dL/d(weight) = dL/d(w_effective) * scale                    # STE path
             + dL/d(w_effective) * w_tern * d(scale)/d(weight)  # scale path
```

The second term involves the gradient through `max()`, which is sparse (only fires at the argmax position). It added noise more than signal.

The fix:

```python
scale = w.abs().max(dim=1, keepdim=True)[0].detach()
```

Scale is a calibration statistic, not a learnable parameter. One missing `.detach()` was enough to break everything.

Lesson learned: **quantization code demands explicit gradient boundaries**. If something is a calibration quantity (scale, mean, std), think hard about whether gradients should flow through it.

### The Full `TernaryLinear` Module

Here's the complete PyTorch implementation. This is a drop-in replacement for `nn.Linear`: same API, same initialization, same optimizers. The network doesn't "know" it's quantized.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, threshold_factor=0.05):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.threshold_factor = threshold_factor

        # Match nn.Linear initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        # Per-channel scale (detached: no gradients!)
        scale = self.weight.abs().max(dim=1, keepdim=True)[0].detach()
        threshold = self.threshold_factor * scale

        # Ternary quantization
        w_q = torch.where(
            self.weight > threshold,
            torch.ones_like(self.weight),
            torch.where(
                self.weight < -threshold,
                -torch.ones_like(self.weight),
                torch.zeros_like(self.weight),
            ),
        )

        # Straight-Through Estimator
        w_eff = (w_q * scale - self.weight).detach() + self.weight

        return F.linear(x, w_eff, self.bias)
```

This is the minimal version that:

* trains end-to-end,
* produces correct gradients,
* and matches the math described above.

Everything else, CUDA kernels, packing, performance work, builds on this.

### Results

On MNIST (784 → 256 → 128 → 10):

| Model   | Accuracy | Weight Memory |
| ------- | -------- | ------------- |
| FP32    | ~97%     | 100%          |
| Ternary | ~95%     | ~5%           |

Two points of accuracy for ~20× theoretical compression. Not free, but often worth it.

Language models told a similar story: roughly 33% worse perplexity. Acceptable for some deployments, unacceptable for others.

Now comes the hard part.

---

## Part 3: CUDA Reality

At this point the model trains correctly. The CUDA work is about **inference**, not training.

### Why Write Custom Kernels?

`F.linear` eventually calls cuBLAS. cuBLAS is absurdly fast. But it doesn't know our weights are ternary. It loads FP32 weights and performs FP32 multiplies.

The theoretical upside of ternary kernels:

* Less memory traffic (packed weights are 16× smaller)
* No multiplies (add/negate/skip)
* Potential bandwidth savings

So I wrote my own.

### Version 1: The Naive Baseline (21× Slower)

One thread per output element. Each thread reads K elements from global memory.

```
Shape (64, 1024, 4096):
- 262K threads, each reading 1024 floats
- ~1GB of redundant global memory reads
```

Result: **21× slower than cuBLAS**. I assumed the problem was arithmetic overhead. I was wrong.

### The Profiler Doesn't Lie

Running Nsight Compute revealed the real problem:

| Metric | My Kernel | cuBLAS | Ratio |
|--------|-----------|--------|-------|
| **DRAM throughput** | **1.77%** | 48.22% | 27× worse |
| SM throughput | 27.56% | 62.17% | 2.3× worse |
| Occupancy | 66.66% | 31.22% | 2× better (!) |

I had **1.77% memory throughput**. The A2000 has ~288 GB/s bandwidth. I was using ~5 GB/s.

The GPU wasn't slow at compute. It was *starving* for data.

### The First Bug: Uncoalesced Memory Access

GPU memory coalescing requires threads in a warp to access consecutive addresses (within 128 bytes). My kernel did this:

```cpp
// Each thread loads weights for its output channel
uint8_t packed = W_packed[n * K_bytes + byte_idx];
```

Adjacent threads had different `n` values, so they accessed addresses `K_bytes` apart (1024 bytes for K=4096).

**Every single load was uncoalesced.** Instead of one 128-byte transaction serving 32 threads, I was doing 32 separate transactions. This alone explained the 27× memory throughput gap.

### Version 2: Shared Memory Tiling (Still 6-12× Slower)

The classic GEMM fix: tile the matrices, cooperatively load into shared memory, compute from there.

```cpp
constexpr int TILE_M = 32;  // Batch tile
constexpr int TILE_N = 32;  // Output tile
constexpr int TILE_K = 32;  // Reduction tile

__shared__ float X_tile[TILE_M][TILE_K];
__shared__ int8_t W_tile[TILE_N][TILE_K];
```

Result: **1.4-1.9× faster than baseline**. Progress, but still nowhere near cuBLAS.

### The Second Bug: 73-Way Bank Conflicts

The profiler showed something alarming:

```
derived__memory_l1_conflicts_shared_nway: 73
derived__memory_l1_wavefronts_shared_excessive: 117,440,512
```

**73-way bank conflicts.** Normal is 1-4.

Shared memory has 32 banks. Address `A` maps to bank `A % 32`. My inner loop did:

```cpp
for (int k = 0; k < TILE_K; k++) {
    float x_val = X_tile[ty][k];   // row access - OK
    int8_t w_val = W_tile[tx][k];  // column access - BAD
}
```

For `W_tile[tx][k]`:
- Thread 0 reads address `(0 * 32 + k)` → bank `k`
- Thread 1 reads address `(1 * 32 + k)` → bank `k`
- Thread 31 reads address `(31 * 32 + k)` → bank `k`

**All 32 threads hit the same bank.** Every access serialized.

The fix was embarrassingly simple:

```cpp
// Before: 32-way bank conflict
__shared__ int8_t W_tile[TILE_N][TILE_K];     // [32][32]

// After: no conflicts
__shared__ int8_t W_tile[TILE_N][TILE_K + 1]; // [32][33]
```

The padding changes the stride, spreading threads across different banks.

### Version 3-5: The Kernel Museum

I tried everything:
- Different memory layouts (`[N, K]` vs `[K, N]`)
- Branchless ternary decode (LUT vs arithmetic)
- Larger tile sizes (`TILE_K=128` for more data reuse)
- Transposed weight formats

Each experiment taught something. Most made things worse. Some insights from one version would help another. I had 5 experimental kernels at one point. That's not a library, it's a museum.

### Version 6: Consolidation

The winning insight: **different batch sizes need different kernels.**

```
ternary_linear_packed_forward()
    │
    ├── B <= 32 ──► small_batch kernel
    │               TILE_K=128 (more K-reuse)
    │               Optimized for inference
    │
    └── B > 32  ──► large_batch kernel
                    TILE_K=32 (more parallelism)
                    Optimized for training
```

I deleted V2, V2.1, V2.2. They donated their insights and got removed. Ship two kernels, not five.

### Final Performance

GPU: NVIDIA RTX A2000 8GB

| Shape | Time (ms) | vs cuBLAS | Kernel |
|-------|-----------|-----------|--------|
| Single-batch (1×768→768) | 0.03 | **0.9×** | small |
| MLP (32×1024→4096) | 0.97 | 6.4× | small |
| Training (128×768→768) | 0.54 | 5.1× | large |

**The key win**: single-batch inference nearly matches cuBLAS. This is the deployment case that matters most.

For training batches, we're still 5-8× slower. That's acceptable because training happens once; inference happens millions of times.

### What's Still On The Table

1. **Vectorized loads**: cuBLAS uses `float4` (128-bit) loads. We load one value at a time.
2. **Warp-level primitives**: `__shfl_sync` for register reductions instead of shared memory.
3. **Tensor Core exploration**: INT8 tensor cores could theoretically help, but the mapping from ternary is non-trivial.
4. **True ternary arithmetic**: We still multiply by {-1, 0, +1}. A branchless add/negate/skip path might help on some architectures.

The honest summary: matching cuBLAS is *hard*. CUTLASS exists for a reason. But for single-batch inference with packed weights, we're close enough that the 16× memory reduction dominates.

### Lessons From the Kernel Grind

* **Profile first, hypothesize second**: I assumed warp divergence was the problem. The profiler said memory throughput.
* **1.77% throughput is a bug, not a feature**: If you're below 10% of peak bandwidth, something is fundamentally wrong with your access pattern.
* **Bank conflicts are silent killers**: 73-way conflicts don't crash. They just make everything 30× slower.
* **Padding fixes bank conflicts**: `[32][32]` → `[32][33]` is the cheapest optimization I've ever made.
* **Delete dead code**: A library isn't a museum. Ship the kernels that work, delete the experiments.

---

## Part 4: Where Ternary Actually Makes Sense

Ternary isn't about beating FP16 on A100s.

It's about **fitting models where they otherwise wouldn't fit**.

| Format      | Bits / Weight | 7B Model    |
| ----------- | ------------- | ----------- |
| FP32        | 32            | 28 GB       |
| FP16        | 16            | 14 GB       |
| INT8        | 8             | 7 GB        |
| INT4        | 4             | 3.5 GB      |
| **Ternary** | **2** (packed) | **~1.75 GB** (+ scales) |

That last row enables deployment on Jetsons, embedded devices, and very ordinary laptops.

### When It's a Good Idea

* Memory-constrained inference
* Edge deployment
* Single-batch / low-latency scenarios (where we match cuBLAS)
* Willing to trade some accuracy

### When It Isn't

* Accuracy is paramount
* Memory isn't tight
* Large training batches (we're still slower there)
* Model is already small

Know your constraints.

---

## Closing

I started this project convinced I could beat cuBLAS. I didn't, not on large batches. That expectation didn't survive first contact with reality.

But I got closer than I expected. Single-batch inference at 0.9× cuBLAS means the 16× memory reduction actually matters for deployment. The model fits where it couldn't before, and inference isn't painfully slow.

What I built along the way: a correct ternary linear layer, working CUDA kernels with auto-dispatch, proper gradient handling, and a much clearer mental model of why high-performance kernels look the way they do.

The biggest takeaway wasn't "ternary is slow" or "cuBLAS is unbeatable," but **where the real bottlenecks actually are**: memory layout, bandwidth, and data movement, not arithmetic. Skipping multiplies doesn't help when you're waiting on DRAM. Packing and data movement matter more than arithmetic once you leave the whiteboard.

This project also reinforced a quieter lesson: correctness comes first. Quantization code is deceptively easy to get *almost* right, and "almost right" is worse than broken. Detaching the wrong tensor, letting a statistic receive gradients, or trusting intuition over profiling can quietly invalidate weeks of work.

If any of this sounds familiar, you're probably the intended audience.

---

### Try it yourself

The code lives at [github.com/RetamalVictor/bittorch](https://github.com/RetamalVictor/bittorch):

```bash
git clone https://github.com/RetamalVictor/bittorch.git
cd bittorch
uv sync && uv build
uv run python examples/mnist_mlp_ternary.py --compare
```

The `--compare` flag runs both ternary and FP32 models, producing a summary like:

```
COMPARISON SUMMARY
============================================================
Metric               |      Ternary |         FP32 |       Diff
------------------------------------------------------------
Test Accuracy        |        ~95%  |        ~97%  |      ~2%
```

If you're exploring low-precision ML, I hope this saves you time, or at least makes the failure modes clearer. This stuff really is more fun when it's shared.

---

## References

* **BitNet b1.58: Scaling 1-bit Transformers**
  [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)
* **CUTLASS GEMM Tutorial**
  [https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)
* **Bengio et al., 2013: Straight-Through Estimator**
  [https://arxiv.org/abs/1308.3432](https://arxiv.org/abs/1308.3432)
