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

Six versions, 185 tests, and one spectacular "oh no, gradients are flowing through the scale computation" bug later, I had something that worked. It's still not faster than cuBLAS—I'll be upfront about that. But it compresses weights by ~20× *in theory*, trains end-to-end with correct gradients, and taught me more about GPU programming than any course ever did.

This is the story of building **BitTorch**: ternary neural networks from paper to CUDA.

---

### What we'll cover

* Why $\log_2(3) \approx 1.58$ bits is the *theoretical* lower bound per weight
* The Straight-Through Estimator trick that makes training possible
* Writing CUDA kernels (and why your first one will be embarrassingly slow)
* Honest benchmarks: what worked, what didn't, and why

Let's start at the beginning.

---

## Part 1: Why 1.58 Bits?

Before touching code, it's worth understanding what we're actually compressing.

Standard neural network weights are 32-bit floats—about four billion possible values per weight. The BitNet paper makes a provocative claim: for many tasks, you can get away with **three** values:

$$
\lbrace -1, 0, +1 \rbrace
$$

That's it. No decimals. No fine-grained precision.

From an information-theoretic perspective, three values require:

$$
\log_2(3) \approx 1.58 \text{ bits}
$$

Compared to 32 bits, that's a **20× theoretical compression**. A 7B-parameter model drops from 28 GB to roughly 1.4 GB. That's the difference between "needs a datacenter" and "runs on a laptop."

Important caveat up front: **this is a theoretical limit**. We don't physically store 1.58 bits per weight yet. But it's the target.

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

In theory, we've replaced floating-point multiplies with conditional adds. In practice… we'll get there.

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

That scale depends on `w`. Gradients flowed through it. The result was subtle, silent corruption—my CUDA and Python gradients had a cosine similarity of 0.29.

The fix:

```python
scale = w.abs().max(dim=1, keepdim=True)[0].detach()
```

Scale is a statistic, not a learnable parameter. One missing `.detach()` was enough to break everything.

Lesson learned: **quantization code demands explicit gradient boundaries**.

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

Everything else—CUDA kernels, packing, performance work—builds on this.

### Results

On MNIST (784 → 256 → 128 → 10):

| Model   | Accuracy | Weight Memory |
| ------- | -------- | ------------- |
| FP32    | 97.3%    | 100%          |
| Ternary | 94.3%    | ~5%           |

Three points of accuracy for ~20× theoretical compression. Not free, but often worth it.

Language models told a similar story: roughly 33% worse perplexity. Acceptable for some deployments, unacceptable for others.

Now comes the hard part.

---

## Part 3: CUDA Reality

At this point the model trains correctly. The CUDA work is about **inference**, not training.

### Why Write Custom Kernels?

`F.linear` eventually calls cuBLAS. cuBLAS is absurdly fast. But it doesn't know our weights are ternary. It loads FP32 weights and performs FP32 multiplies.

The theoretical upside of ternary kernels:

* Less memory traffic
* No multiplies
* Potential bandwidth savings

So I wrote my own.

### What Happened

A naive kernel was **20× slower than cuBLAS**. Tiling, shared memory, and cooperative loads helped—about **1.4–1.9× faster** than baseline—but still nowhere near competitive.

Profiling made the situation clear: this kernel is **memory-bandwidth bound**. When the GPU is waiting on memory, skipping multiplies doesn't matter.

### Lessons Learned

* cuBLAS represents decades of optimization (CUTLASS exists for a reason).
* Ternary arithmetic doesn't help if you're bandwidth-bound.
* Storing ternary weights as `int8` wastes 6 bits per weight.
* Packing is mandatory to see real gains.

The kernel works. It's correct. It's just not competitive yet.

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
| **Ternary** | **1.58**      | **~1.4 GB** |

That last row enables deployment on Jetsons, embedded devices, and very ordinary laptops.

### When It's a Good Idea

* Memory-constrained inference
* Edge deployment
* Willing to trade some accuracy

### When It Isn't

* Accuracy is paramount
* Memory isn't tight
* Model is already small

Know your constraints.

---

## Closing

I started this project convinced I could beat cuBLAS. I didn't. That expectation didn't survive first contact with reality.

What I did build is a correct ternary linear layer, a working CUDA forward pass, and a much clearer mental model of why high-performance kernels look the way they do. The biggest takeaway wasn't "ternary is slow" or "cuBLAS is unbeatable," but **where the real bottlenecks actually are**: memory layout, bandwidth, and data movement—not arithmetic.

Ternary networks aren't a drop-in replacement for FP16 on datacenter GPUs. They shine when memory is the constraint, not throughput—edge devices, embedded systems, and hardware that simply can't fit large models otherwise. In that regime, a 3–5% accuracy drop can be a very reasonable trade.

This project also reinforced a quieter lesson: correctness comes first. Quantization code is deceptively easy to get *almost* right, and "almost right" is worse than broken. Detaching the wrong tensor, letting a statistic receive gradients, or trusting intuition over profiling can quietly invalidate weeks of work.

If any of this sounds familiar, you're probably the intended audience.

---

## Live Demo

Want to see ternary inference in action? The demo below runs a character-level language model entirely in your browser using WebGPU. The model uses 1.58-bit ternary weights—the same technique described above.

<div id="ternary-lm-demo" class="my-8 not-prose"></div>

Type a prompt and click "Generate" to watch character-by-character text generation powered by {-1, 0, +1} weights. Click "Under the Hood" to see memory savings and model architecture.

---

### Try it yourself

The code lives here:

```bash
git clone https://github.com/RetamalVictor/bittorch.git
cd bittorch
uv sync && uv build
uv run python examples/mnist_mlp_ternary.py --compare
```

If you're exploring low-precision ML, I hope this saves you time—or at least makes the failure modes clearer. This stuff really is more fun when it's shared.

---

## References

* **BitNet b1.58: Scaling 1-bit Transformers**
  [https://arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764)
* **Bengio et al., 2013 — Straight-Through Estimator**
  [https://arxiv.org/abs/1308.3432](https://arxiv.org/abs/1308.3432)
