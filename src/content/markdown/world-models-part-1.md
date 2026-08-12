---
title: "World Models From Scratch, Part 1: A Ball, a VAE, and a Number That Would Not Move"
date: "2026-08-11"
tags: ["machine-learning", "world-models", "vae", "jax", "representation-learning"]
summary: "A VAE can reconstruct every frame perfectly and still know nothing about how the world moves. Building a toy environment where I know the ground truth, then using a linear probe to catch the latent out. Posterior collapse, the price of a nat, and why a dead model scored 0.74."
readTime: "22 min"
notebook: ""
featured: true
---

# World Models From Scratch, Part 1: A Ball, a VAE, and a Number That Would Not Move

<figure class="wm-fig wm-hero wm-pixel">
 <img src="/images/world-models/vae-recon-rollout.gif" alt="Animated, side by side. Left, a real episode of a white ball bouncing inside a dark box. Right, the same episode reconstructed frame by frame by the VAE, tracking it closely throughout.">
 <figcaption>Left, a real episode. Right, the same episode rebuilt frame by frame from sixteen numbers each, by the model this post is about. It never misses. It also has no idea which way the ball is going.</figcaption>
</figure>

A model can reconstruct every frame you show it, pixel for pixel, and still know nothing about how the world moves. I found that easy to nod along to and hard to actually believe until it happened in my own code, on a dataset small enough to hold in my head.

So this series builds the whole thing from scratch, on a world I wrote myself. Three models, in the order the field discovered them: a VAE, then Ha and Schmidhuber's frozen-encoder recurrent model from *World Models* (2018), then Dreamer's RSSM. All three run on the same dataset under the same evaluation protocol, and one number decides who wins.

That number is $R^2 = 0.00$, and this post is about earning the right to trust it.

**What you'll be able to do by the end of the three parts:** derive the ELBO and say which term does what to a learned code; explain why a single-frame encoder cannot represent velocity, and prove it on your own data with a linear probe; read an open-loop drift curve against calibrated reference levels instead of squinting at it; and say precisely what joint training buys over the frozen-encoder recipe, in $R^2$ and in pixels.

**What I assume you have:** comfort with convolutional networks and maximum likelihood, and enough JAX to read a `lax.scan` without flinching. Everything else gets built here. Code is at [github.com/RetamalVictor/world-models](https://github.com/RetamalVictor/world-models); every number below comes from a seeded run in that repo.

Part 1 is the environment and the VAE: the measuring instrument, and the first thing measured.

---

## Why not just use a real benchmark

Every model in this series is graded on one question.

> ### Did the latent learn the dynamics?

On Atari, or on DOOM, you cannot answer it. You can look at reconstructions, you can look at imagined rollouts, you can watch the video and go "hm, plausible". What you cannot do is ask the model where the ball is going, because nobody will tell you where the ball was actually going. The ground truth is inside the emulator, and the emulator is not in the business of exporting it.

So I wrote a simulator that is.

<figure class="wm-fig wm-swap">
 <img class="wm-light" src="/images/world-models/diagrams/measurement-loop-light.png" alt="The simulator emits a frame and a true state; the model turns the frame into a latent; a ridge probe maps the latent back to the true state and reports R squared.">
 <img class="wm-dark" src="/images/world-models/diagrams/measurement-loop-dark.png" alt="The simulator emits a frame and a true state; the model turns the frame into a latent; a ridge probe maps the latent back to the true state and reports R squared.">
 <figcaption><strong>Figure 1.</strong> The measurement loop. Ground truth exists here only because the simulator is mine.</figcaption>
</figure>

Figure 1 is the whole apparatus. The simulator emits two things at every step: the frame $o_t$ the model gets to see, and the true state $s_t = (x, y, v_x, v_y)$ the model never sees. The model turns frames into some internal state. Then a linear probe tries to read the true state back out of it, and reports an $R^2$.

**Definition (the probe).** Fit ridge regression from a model's state to a ground-truth quantity on one set of held-out episodes, and report $R^2$ on a second, disjoint set:

$$R^2 = 1 - \frac{\sum_t \lVert y_t - \hat{y}_t \rVert^2}{\sum_t \lVert y_t - \bar{y} \rVert^2}$$

One is perfect, zero is "no better than predicting the mean of $y$", and negative means the fit generalized worse than a constant. The whole probe is four lines, deliberately not `sklearn`, because a reader should be able to see all of it:

```python
def ridge_fit(X, y, lam: float = 1e-3):
 """Ridge regression with intercept. X: (N, D), y: (N, K) -> (w, b)."""
 x_mean = X.mean(axis=0)
 y_mean = y.mean(axis=0)
 Xc = X - x_mean
 yc = y - y_mean
 d = X.shape[1]
 w = jnp.linalg.solve(Xc.T @ Xc + lam * jnp.eye(d), Xc.T @ yc)
 b = y_mean - x_mean @ w
 return w, b
```

Two decisions in there bite later, so I'll defend them now. The probe is **linear** on purpose: a nonlinear probe with enough capacity recovers position from almost anything, including a randomly initialized network, so it measures the probe rather than the model. And it is fit on one set of *episodes* and scored on another. Frames inside one episode are near-duplicates of their neighbours; split at the frame level and you have leaked your test set into your training set through a wall of correlated pixels.

A warning that will get its own section later: this instrument lies in specific, learnable ways. It gets caught lying before this post ends.

## The world: a ball in a box

State is $(x, y, v_x, v_y)$ in continuous coordinates. The action is a small nudge $(a_x, a_y)$ added to the velocity. The observation is a 32x32 grayscale image with a Gaussian blob centred on the ball.

Here it is running. This is not a video: the physics and the renderer are the same forty lines the training data came from, ported to TypeScript and executing in your browser.

<div id="ball-env-demo" class="my-8 not-prose"></div>

Watch the wall bounces rather than the straight stretches. Between bounces the motion is a linear extrapolation any model can do. At a bounce, a half-pixel error in position decides whether the reflection happens this frame or the next, and two trajectories that agreed perfectly diverge at full speed. Everything hard about this environment is concentrated in those moments.

Then pause it, and look at a single frame with the ground truth hidden. That is exactly the input a VAE gets, and the question this post ends on is whether anything in that picture tells you which way the ball is moving.

The physics fits in one screen, and every line of it is a decision:

```python
vx = state.vx + ax
vy = state.vy + ay

speed = jnp.sqrt(vx ** 2 + vy ** 2)
scale = jnp.where(speed > params.max_speed, params.max_speed / speed, 1.0)
vx = vx * scale
vy = vy * scale

x = state.x + vx * params.dt
y = state.y + vy * params.dt

x, vx = _reflect(x, vx, jnp.float32(params.img_w))
y, vy = _reflect(y, vy, jnp.float32(params.img_h))
```

**The blob is Gaussian, not a hard circle.** A hard-edged sprite quantizes position to the pixel grid: move the ball a third of a pixel and the image does not change at all. A Gaussian blob shifts intensity smoothly, so the image encodes sub-pixel position, and the reconstruction loss gets a gradient everywhere instead of a staircase. This one choice is why the position probe can score above 0.9 in principle at all.

**Walls reflect, they don't clamp.** Velocity flips sign and the overshoot folds back, so energy is conserved and there is something in the dynamics worth modelling. One floor and one ceiling reflection per axis is enough only because `max_speed` (2.0) is far below the box size (32). That assumption is a comment in the code, and it would need revisiting if the speed cap ever approached the box dimensions.

**Speed is clamped by rescaling, not clipping.** Actions keep injecting energy; without a cap the ball accelerates forever. Rescaling the velocity vector preserves direction, while clipping components would quietly bend every fast trajectory toward the diagonals.

**The true state rides along in `info`.** Every `step` returns `{"true_x", "true_y", "true_vx", "true_vy"}`. That is the environment's entire reason to exist, so the ground truth is generated *with* each trajectory rather than reconstructed afterwards.

The API is gymnax-shaped, with `reset(key, params)` and `step(key, state, action, params)` as pure functions and params as a static `NamedTuple`, so the training code never learns that it is talking to a toy. Swapping in ViZDoom later does not touch the model code, which is the only reason the DOOM leg of this project is tractable at all.

## The dataset

One fixed file, generated once, shared by all three models:

```
uv run make-data # 500 episodes x 200 steps, random_nudge_policy(0.3), seed 0
```

500 episodes of 200 steps, frames stored as `uint8` (about 100 MB), actions and true state as `float32`. The split is at the episode level: episodes 0 to 399 train, 400 to 449 validation, 450 to 499 test. The probe fits on 450 to 474 and scores on 475 to 499, always, for every model in this series, so the numbers stay comparable by construction rather than by discipline.

Two statistics from `data/ball.stats.json` that turn out to matter more than they look:

| statistic | value |
|---|---|
| pixel mean | 0.0221 |
| pixel std | 0.1045 |

The frames are 98% black. Hold onto that.

## The VAE, informally

An autoencoder squeezes an image through a bottleneck and asks for it back. Whatever survives the squeeze is what the model considered worth keeping. That is already a decent theory of representation learning, and if you stop there you get a code that works but has no particular shape. Nothing in the objective cares how the 16 numbers are arranged.

A variational autoencoder adds one thing: the bottleneck is a **noisy channel**. The encoder does not emit a code, it emits a *distribution* over codes, and a sample from that distribution is what the decoder receives. Now sloppiness has a price. If two different frames map to overlapping distributions, the decoder cannot tell them apart. If the encoder wants to be certain, it has to pay, and the currency is the KL term.

This is the picture to keep: **the encoder is buying bandwidth, and $\beta$ is the price per nat.**

## The VAE, precisely

We want to maximize $\log p(o)$ for a latent-variable model $p(o) = \int p(o \mid z) p(z) dz$, which is intractable. Introduce any distribution $q(z \mid o)$ and apply Jensen's inequality:

$$\log p(o) = \log \mathbb{E}_{q} \left[\frac{p(o \mid z) p(z)}{q(z \mid o)}\right] \ge \underbrace{\mathbb{E}_{q}\big[\log p(o \mid z)\big] - \mathrm{KL}\big(q(z \mid o) \Vert p(z)\big)}_{\text{ELBO}}$$

The gap between the two sides is exactly $\mathrm{KL}(q(z \mid o) \Vert p(z \mid o))$, so the bound is tight when the encoder matches the true posterior. Maximizing the ELBO therefore does two jobs at once: it pushes up the likelihood and it drags $q$ toward the posterior it can never compute.

Three concrete choices turn that into code. The prior is $p(z) = \mathcal{N}(0, I)$ with $z \in \mathbb{R}^{16}$, since the ball has four true degrees of freedom and 16 leaves headroom without letting the probe flatter the model. The encoder is a diagonal Gaussian, $q(z \mid o) = \mathcal{N}(\mu(o), \mathrm{diag} \sigma^2(o))$, so the KL has a closed form. And the decoder is a Gaussian with fixed variance, which makes the reconstruction term a plain squared error:

$$-\log p(o \mid z) = \frac{1}{2\sigma_{\text{dec}}^2} \lVert o - \hat{o} \rVert^2 + \text{const}$$

Minimizing the negative ELBO therefore means minimizing $\frac{1}{2\sigma_{\text{dec}}^2}\lVert o - \hat{o}\rVert^2 + \mathrm{KL}$. Multiply through by $2\sigma_{\text{dec}}^2$, which changes nothing about where the minimum is, and you get the loss that is actually implemented:

$$\mathcal{L} = \lVert o - \hat{o} \rVert^2 + \beta \mathrm{KL}\big(q(z \mid o) \Vert \mathcal{N}(0, I)\big), \qquad \beta = 2\sigma_{\text{dec}}^2$$

**Remark.** $\beta$ is not a free knob bolted onto the ELBO, at least not at first. It *is* the decoder's assumed observation noise. $\beta = 1$ asserts $\sigma_{\text{dec}} \approx 0.71$ per pixel, on frames whose own standard deviation is 0.1045. You are telling the model that the image is mostly noise. Remember that when the first run collapses.

(Once you tune $\beta$ freely you have left the ELBO and entered $\beta$-VAE territory, where the KL is a disentanglement pressure rather than a likelihood term. That turns out to be a feature, and the probe section below is where it gets cashed in.)

### The reparameterization trick

We need $\nabla_\phi \mathbb{E}_{q_\phi}[ \cdot]$, and the parameters sit inside the distribution we are sampling from. Write the sample as a deterministic function of the parameters and a fixed noise source:

$$z = \mu_\phi(o) + \sigma_\phi(o) \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

Now the randomness is an input, the path from $\phi$ to $z$ is differentiable, and a single sample gives a low-variance gradient. In Flax it is one line, which rather undersells it:

```python
def __call__(self, x, key):
 mu, logvar = self.encoder(x)
 std = jnp.exp(0.5 * logvar)
 z = mu + std * jax.random.normal(key, mu.shape) # reparameterization
 recon = self.decoder(z)
 return recon, mu, logvar
```

### The network

<figure class="wm-fig wm-swap">
 <img class="wm-light" src="/images/world-models/diagrams/vae-blocks-light.png" alt="Encoder: three stride-2 convolutions from 32x32 to 4x4, widening 32 to 64 to 128 channels, a dense layer to 256, then mu and log variance heads. A sampling node produces z. Decoder mirrors the encoder back to 32x32.">
 <img class="wm-dark" src="/images/world-models/diagrams/vae-blocks-dark.png" alt="Encoder: three stride-2 convolutions from 32x32 to 4x4, widening 32 to 64 to 128 channels, a dense layer to 256, then mu and log variance heads. A sampling node produces z. Decoder mirrors the encoder back to 32x32.">
 <figcaption><strong>Figure 2.</strong> The VAE end to end. Note where the two loss terms attach: reconstruction across the whole round trip, KL at the latent.</figcaption>
</figure>

Figure 2 is the model in full: three stride-2 convolutions take 32x32 down to 4x4 while widening 32 to 64 to 128 channels, a dense layer to 256, then two heads for $\mu$ and $\log \sigma^2$. The decoder mirrors it. SiLU activations throughout; the final layer is linear, because its output is a Gaussian mean, not a probability.

Note where the two loss terms attach in the figure. Reconstruction is summed over the 1024 pixels; the KL is summed over the 16 latent dimensions and averaged over the batch. Both are *sums*, not means, over their respective axes, and that is what keeps $\beta$ interpretable as a price rather than an arbitrary scalar.

Training: adam at $10^{-3}$, gradients clipped at global norm 1.0, batch 128, 20k steps. On the GPU that is about two minutes, which is the real reason a toy environment is worth building. You can afford to be wrong six times before lunch.

## Three runs, one of them dead

I ran the textbook configuration first: $\beta = 1$, constant, from step zero.

| run | $\beta$ schedule | KL (nats) | val recon | position $R^2$ | velocity $R^2$ |
|---|---|---|---|---|---|
| `base` | 1.0 from step 0 | 0.00 | 11.17 | 0.74 | -0.005 |
| `beta01` | 0.1 fixed | ~9.6 | 0.06 | 0.35 | -0.013 |
| `warmup5k` | 1.0, 5k linear warmup | ~5.4 | 0.28 | 0.75 | -0.013 |

Units, because they bite for the rest of the series: reconstruction is squared error **summed over all 1024 pixels** and averaged over the batch, decoding the posterior mean. KL is in nats, summed over the 16 latent dimensions. Part 2 works in *per-pixel* error instead, which is the same quantity divided by 1024, and I will say so again when it matters.

`base` is posterior collapse, the canonical VAE failure: the KL hit zero inside the first thousand steps and never came back, and the reconstruction grid is a row of black squares.

A zero KL sounds like efficiency, since the code costs nothing. It is fatal for a specific reason. Against a unit Gaussian prior the KL vanishes in exactly one configuration:

$$\mathrm{KL}\big(q(z \mid o) \Vert \mathcal{N}(0, I)\big) = 0 \text{for every } o \quad\Longleftrightarrow\quad \mu(o) \equiv 0, \sigma(o) \equiv 1$$

The posterior is *the same distribution for every frame*. Whatever the encoder computes on its way there, none of it survives to the output. And the expected KL upper-bounds the mutual information between the frame and the sampled latent,

$$I(O; Z) \le \mathbb{E}_o\big[\mathrm{KL}\big(q(z \mid o) \Vert p(z)\big)\big]$$

so a KL of zero nats means $z$ carries zero information about the frame it came from. The decoder is being handed noise rather than a compressed picture, and the best thing any function of pure noise can do under squared error is ignore its input and emit a constant.

Which constant? The reconstruction number sat at **11.17** for twenty thousand steps and refused to move, and 11.17 turns out to be a very particular value:

$$\min_c \mathbb{E}\lVert o - c \rVert^2 = \sum_{\text{pixels}} \mathrm{Var}(o_i) = 1024 \times (0.1045)^2 \approx 11.17$$

<figure class="wm-fig">
 <img src="/images/world-models/vae-recon-collapsed.png" alt="Top row: eight frames with a white ball. Bottom row: eight near-black squares, the collapsed model's reconstructions.">
 <figcaption><strong>Figure 3.</strong> The collapsed run. Top row data, bottom row the model's best effort.</figcaption>
</figure>

There it is. On a dataset that is 98% black, the mean image *is* a black square, so the failure is almost invisible unless you know what number to look at.

The loss was not stuck. It had *converged*, to the best answer available to a decoder that ignores its input: paint the mean image. On frames that are 98% black, that is a comfortable place to sit, and at $\beta = 1$ from step zero the exit is too expensive to find. Every nat of information the encoder buys costs a full unit of squared error before it has a decoder good enough to spend it on.

The fix keeps $\beta = 1$ as the target and ramps it linearly over the first 5k steps:

```
uv run train-vae --run-name warmup5k --beta 1.0 --beta-warmup-steps 5000
```

Early on, information is nearly free, so the decoder learns to use the latent while nobody is charging for it. By the time the full price kicks in, "ignore the latent" is no longer the optimum it was at initialization.

<figure class="wm-fig">
 <img src="/images/world-models/vae-loss-curves-warmup.png" alt="Two panels. Left: training reconstruction rising to about 1.7 while validation reconstruction sits near 0.28. Right: KL falling from 30 nats to about 5.4 by step 5000 and staying flat.">
 <figcaption><strong>Figure 4.</strong> Training curves for <code>warmup5k</code>. The KL panel on the right is the one to read.</figcaption>
</figure>

The KL falls from about 30 nats to about 5.4 by step 5000, exactly when the ramp ends, and then *stays there* for 15,000 more steps. No re-collapse. The equilibrium is real: the model has found the amount of information about a frame that is worth paying full price for.

The left panel contains a trap I want to flag, because it confused me for a minute and it will confuse you. Training reconstruction (~1.7) sits *above* validation reconstruction (~0.28), which looks like the world's most broken generalization gap. It isn't. The training curve decodes a *sample* $z = \mu + \sigma\epsilon$, while the validation curve decodes the posterior *mean*. The gap is the sampling noise the model is trained to tolerate, not overfitting in reverse. Two different quantities with the same name on the same axis.

### What the budget buys

Take $\beta$ seriously as a price and the two healthy runs stop looking like arbitrary hyperparameter choices. The model minimizes $\text{recon} + \beta \cdot \text{KL}$, so a nat is worth buying only if it saves at least $\beta$ units of squared error. The total available saving is bounded: 11.17, the cost of knowing nothing.

| run | $\beta$ | nats bought | recon (summed) | error saved | per nat |
|---|---|---|---|---|---|
| `beta01` | 0.1 | 9.6 | 0.06 | 11.11 | 1.16 |
| `warmup5k` | 1.0 | 5.4 | 0.28 | 10.89 | 2.02 |

Cheap nats get spent freely; expensive nats get spent carefully. And the 5.4-nat figure has a physical reading. Pixel-precise position in a 32x32 box costs about $\ln 32 + \ln 32 \approx 6.9$ nats, so a 5.4-nat budget is position at slightly-under-pixel precision, which is exactly what those reconstructions look like.

(Honesty note: these are two separate optimization runs, not two points measured on one rate-distortion frontier, so read the "per nat" column as an average, not as a marginal rate. Getting the real frontier would mean sweeping $\beta$ over a dozen values and plotting the curve. I didn't.)

<figure class="wm-fig">
 <img src="/images/world-models/vae-recon-warmup.png" alt="Top row: eight data frames. Bottom row: eight reconstructions, each ball in the right place and slightly softer than the original.">
 <figcaption><strong>Figure 5.</strong> Held-out reconstructions from <code>warmup5k</code>, rebuilt from 16 numbers each.</figcaption>
</figure>

Every ball lands where it should, slightly softer than the original: the sub-pixel precision the KL budget could not afford.

<figure class="wm-fig">
 <img src="/images/world-models/vae-prior-samples.png" alt="Sixteen decoded samples from a unit Gaussian prior. Most are a single round ball somewhere in the box; a few show a second fainter blob.">
 <figcaption><strong>Figure 6.</strong> Sixteen draws of z from the prior, decoded. The generative check a reconstruction grid cannot give you.</figcaption>
</figure>

Most are single round balls somewhere in the box, which says the aggregate posterior actually covers the prior. A code that only decoded properly on its own training inputs would produce mush here. A handful show a second, fainter blob, which is the fine print on "approximately covers": there are regions of the prior that no frame maps to, and the decoder has never been asked what lives there.

Neither grid shows the model doing the one thing this project is about: holding on to something across time. It cannot, of course, because every frame here is encoded independently. Here is the clip from the top of the post again, with the stakes attached:

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/vae-recon-rollout.gif" alt="Animated side by side. Left, a real episode of the ball bouncing. Right, the same episode reconstructed frame by frame, tracking it closely.">
 <figcaption><strong>Figure 7.</strong> The opening rollout again, <code>warmup5k</code> on the right. Two hundred frames, each one encoded with no knowledge of the last.</figcaption>
</figure>

It tracks perfectly, and it tracks *statelessly*. Nothing carries over between those frames. A model that follows a ball this well for 200 frames is about to score zero on the one question that needs two of them.

## The probe disagrees with the reconstructions

Now the interrogation. `beta01` reconstructs *better* than `warmup5k`, 0.06 against 0.28 summed error, which looks like this:

<figure class="wm-fig">
 <img src="/images/world-models/vae-recon-beta01.png" alt="Top row: eight data frames. Bottom row: eight reconstructions from the beta 0.1 model, visually almost identical to the data.">
 <figcaption><strong>Figure 8.</strong> The <code>beta01</code> model reconstructing held-out frames. Near pixel-perfect.</figcaption>
</figure>

Every ball lands in the right place, at the right size, with the right softness. Whatever those 16 numbers are, the ball's position is in there; the decoder is drawing it. So the position probe should be a formality.

<figure class="wm-fig">
 <img src="/images/world-models/probe-scatter-beta01-vs-warmup.png" alt="Two scatter plots of predicted against true position. Left, beta 0.1, R squared 0.35, with predictions snaking around the diagonal in filaments. Right, beta 1 with warmup, R squared 0.75, a clean diagonal band.">
 <figcaption><strong>Figure 9.</strong> Linear probe predictions against truth, 5,000 held-out points per panel.</figcaption>
</figure>

Left, $\beta = 0.1$: $R^2 = 0.35$, and look at the *shape* of the failure. Predictions snake around the diagonal in smooth filaments, doubling back on themselves near $x \approx 15$. That is not noise; it is a curved code being read by a straight instrument. Right, $\beta = 1$ with warmup: $R^2 = 0.75$, the cloud is a diagonal band, and the residual structure near the walls is where the blob gets clipped.

Same architecture, same data, same 20k steps. Better reconstructions, half the probe score.

Here is why reconstruction can never settle this question, and the argument is airtight rather than empirical. Take any trained VAE with encoder $e$ and decoder $D$, and any smooth invertible map $f$ on the latent space. The pair $(f \circ e, D \circ f^{-1})$ reconstructs *identically*, because the warp and the unwarp cancel before the decoder draws a pixel. So the reconstruction term is completely blind to the coordinate system of the code. Straight, curled, braided: all the same to it.

A linear probe is precisely a measurement of that coordinate system. It lives or dies on whether $y$ is an affine function of $z$, and almost any choice of $f$ destroys that property while leaving reconstruction untouched.

> **"The information is present" is invariant under reparameterization.
> "The information is linearly accessible" is not.**

Which leaves one question: if the reconstruction term can't prefer straight codes, what can? Look at the loss again. There is only one other term, and against a factorized prior it decomposes per coordinate:

$$\mathrm{KL} = \tfrac{1}{2} \sum_{i=1}^{16} \left( \mu_i^2 + \sigma_i^2 - 1 - \log \sigma_i^2 \right)$$

That expression has opinions. It charges each axis separately for deviating from $\mathcal{N}(0,1)$, so among all the reconstruction-equivalent codes it prefers ones that are zero-centred, unit-scaled and axis-factorized. The KL is the only part of a VAE that cares what the code *looks like*, and $\beta$ is the volume knob on that opinion. This is the $\beta$-VAE result from Higgins et al. (2017). I knew it as a citation before I watched it double a number I cared about in a two-minute run.

One more reason to care about the layout, since the decoder plainly copes either way. Part 2 bolts a recurrent dynamics model onto these 16 numbers with the encoder *frozen*, and a GRU is nonlinear too, so in principle it can learn $f^{-1}$ itself, straightening the encoder's private coordinates before it predicts anything. In practice "first untangle, then predict" is a tax, paid in capacity and in data. I would rather measure that tax than assume it away, and this is the number it gets measured against.

## Where the instrument lies

You now have a probe you trust. Time to break that trust, because the failure is instructive and it took me by surprise.

Run the evaluation suite on the corpse: the collapsed `base` model, KL pinned at zero, decoder painting black squares. Its position probe reads **0.74**. The healthy `warmup5k` model reads **0.75**.

One hundredth of $R^2$ between a working world model and a dead one.

That should be impossible, and by an inequality I wrote down myself two sections ago. KL $\approx 0$ bounds $I(O; Z)$ at zero; the latent contains nothing; a probe cannot read 0.74 out of nothing.

It is not a violation, and seeing why is the real lesson of this section. The bound governs $I(O; Z)$ where $Z \sim q(z \mid o)$ is the *sampled* latent, the thing the decoder eats. The probe never touches $Z$. It reads $\mu(o)$, a deterministic function of the input, and $I(O; \mu(O))$ appears in no inequality anywhere in the ELBO.

<figure class="wm-fig">
 <img src="/images/world-models/dead-latent-probe.png" alt="Left panel: the collapsed model's probe predictions against truth, forming a real diagonal, R squared 0.74. Right panel: per-dimension spread of posterior means on a log axis, healthy model near 1.0 falling off after nine dimensions, collapsed model never above 0.01.">
 <figcaption><strong>Figure 10.</strong> How a dead latent scores 0.74. Left, its predictions. Right, the microscopic spread the probe amplified.</figcaption>
</figure>

The left panel is the dead model's actual predictions against truth. That diagonal is real. The right panel explains it: per-dimension spread of the posterior means, healthy model against collapsed one, sorted and on a log axis. The healthy model's first nine dimensions run from 1.0 down to 0.2, then dimension ten falls off a cliff to 0.04 and the rest idle near the prior, which is nine active dimensions for a world with four degrees of freedom. The collapsed model's largest dimension is 0.01.

It never went to zero either, and the KL is the reason it didn't have to. For small $\mu$ and $\sigma \approx 1$ the per-dimension cost is $\tfrac{1}{2}\mu_i^2$, so at $\mu \sim 0.005$ the encoder keeps input-correlated structure for about $10^{-5}$ nats a dimension. "KL = 0.000" in my logs was never zero. It was cheap.

And ridge regression does not care about magnitude: rescale a code by any $\varepsilon \neq 0$ and the fit becomes $W/\varepsilon$, leaving predictions unchanged, and therefore leaving $R^2$ unchanged too, since it is built entirely from correlations. The probe multiplied those microscopic wiggles by two hundred and read position straight back out.

Why can't the model itself cash that in? Signal-to-noise. The decoder does not see $\mu(o)$; it sees $z = \mu(o) + \sigma\epsilon$ with $\sigma \approx 1$. Its input SNR is $\lVert\mu\rVert/\sigma \approx 0.005$, so the position signal is two hundred times below the sampling noise. The probe reads $\mu$ directly, at infinite SNR. Same encoder, two different channels: one noiseless and informative, one noisy and empty. Both measurements are correct. They measure different things.

This rhymes with an older result: linear probes on *randomly initialized* convolutional networks beat chance comfortably, because a stack of convolutions is a feature extractor before it has learned anything at all. My collapsed encoder is that result's trained-then-flattened cousin. If architecture alone can carry a probe, then a probe alone cannot certify learning.

The rule this bought, which every later post in the series obeys:

> **A probe $R^2$ never travels alone.** Report it with the KL (how much
> the sampled latent could carry) and the reconstruction error (whether
> the model cashes it in). Any one of the three can flatter a broken
> model. It is much harder for all three to coordinate on the same lie.

## The zero this post existed to produce

Velocity probe, all three runs, healthy and dead: $R^2 = 0.00$.

This is a property of the renderer rather than a bug or a tuning failure, and you can read it off the environment code. The observation is `_render(x, y, params)`, a function of position alone. Velocity does not appear. Formally, $I(o_t; v_t \mid x_t, y_t) = 0$: given the position, the frame carries exactly zero additional information about the motion. No single-frame encoder, at any capacity, with any $\beta$, trained for any number of steps, can do better than chance here.

Velocity lives in the *difference between frames*, and a VAE never sees two frames at once.

That zero is the baseline. Everything the next two posts do is an attempt to move it, and the fair way to grade a dynamics model is to ask what it recovers that a single frame provably cannot contain.

## Where this breaks

Before the handoff, the limits of what I just showed you.

**One seed.** Every number in the table is a single run at seed 0. The gap between 0.35 and 0.75 is large enough that I believe it. The gap between 0.74 and 0.75 is not a measurement of anything, which is rather the point of that section.

**One environment, and an easy one.** Static background, one object, no occlusion, no distractors. The 98%-black statistic that made collapse so easy is a property of this dataset. On a busier scene, "paint the mean image" is a much worse local optimum and $\beta = 1$ may behave completely differently.

**The latent size is a choice, not a discovery.** At 16 dimensions the model uses about nine. At 4 it would be forced to be efficient; at 64 the probe would look better for reasons that have nothing to do with the model, because a linear map from 64 dimensions has more room to fit. Capacity-matching across models matters, and the next two posts hold $h = 128$ fixed for exactly that reason.

**The position expectation was wrong and I revised it.** The success criterion I wrote down before running anything said position $R^2 > 0.9$. It landed at 0.75. The honest conclusion is that about 0.75 is what $\beta = 1$ buys at this latent size, not that the model failed, but I wrote the criterion first and missed it, and that belongs in the record rather than in a quietly edited design doc.

## Try it yourself

Everything below runs on CPU, slower but identical to a few decimals.

```
uv sync
uv run make-data
uv run train-vae --run-name warmup5k --beta 1.0 --beta-warmup-steps 5000
```

Artifacts land in `runs/vae/warmup5k/`: `config.json`, `metrics.jsonl`, reconstruction and prior-sample grids, and `probe.json` with the numbers.

1. **Reproduce the collapse.** Run `train-vae --run-name base` with the default constant $\beta = 1$ and watch `recon` in `metrics.jsonl` flatten. Predict the value it flattens at *before* you look, from the pixel std in `data/ball.stats.json`.
2. **Move the price.** Sweep $\beta$ over 0.03, 0.1, 0.3, 1 and 3, and plot KL against reconstruction error. That plot is the rate-distortion frontier I admitted to not measuring above. Where does the position probe peak, and is it at the same $\beta$ that reconstructs best?
3. **Break the probe on purpose.** Encode the test set, apply an invertible nonlinear warp to the latents, say $z \mapsto z + 0.5 \tanh(z)$ elementwise, and re-run the probe. Reconstruction through the inverse is unchanged; watch $R^2$ fall. This is the reparameterization argument, made experimental.

---

Next: [part 2](/blog/world-models-part-2) freezes this encoder, bolts a GRU onto its latents in the *World Models* (2018) style, and asks the zero to move. It moves to 0.81, and then to 0.96 once the encoder is allowed to learn from the prediction loss too. The four-model grid I ran along the way turns out to be more interesting than either number, because one of the four detonates.
