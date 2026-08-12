---
title: "World Models From Scratch, Part 2: Frozen Versus Joint"
date: "2026-08-12"
tags: ["machine-learning", "world-models", "rssm", "dreamer", "jax"]
summary: "The zero from part 1 becomes 0.81 with a GRU bolted onto a frozen encoder, and 0.96 when the encoder is allowed to learn from the prediction loss. Along the way: a model that posts the best one-step error in the experiment and leaves the data manifold entirely by the thirtieth prediction, and an agent trained on nothing but hallucinations."
readTime: "34 min"
notebook: ""
featured: true
---

# World Models From Scratch, Part 2: Frozen Versus Joint

<div id="chase-hero" class="my-8 not-prose"></div>

That is the end of this post, running live, and both networks are the trained weights executing in your browser.

The white ball is steered by the policy this post builds, which receives those 32x32 pixels and nothing else, never the simulator's state. Put the goal wherever you like. It was trained on goals that cruise and bounce, so a goal that teleports under your cursor is a situation it never saw, and it copes.

The strip underneath is the part worth staring at. Those frames are not rendered by anything. Starting from what the model currently believes, it imagines the next sixteen steps with no further pixels arriving, and the decoder paints each one. That imagination is not a readout: it is the environment the policy was trained in. All 5,000 actor updates happened inside it, and the policy never saw a real transition until it was evaluated.

[Part 1](/blog/world-models-part-1) ended with a zero. A linear probe reading velocity out of my VAE's 16-dimensional latent scored $R^2 = 0.00$, and I argued that no amount of training could have fixed it: the renderer is a function of position alone, so a single frame carries exactly zero information about motion. Velocity lives in the difference between frames, and a VAE never sees two frames at once.

This post gives a model two frames at once. Then thirty-three of them. Then it does the whole thing again a second way, because there are two ways to attach memory to an encoder and the difference between them is the entire point of the series.

The first way freezes the encoder from part 1 and trains a recurrent network on its latents, which is the Ha and Schmidhuber *World Models* recipe from 2018. That takes the zero to **0.81**. The second way throws the seam away and trains everything against one loss, which is Dreamer's RSSM, and takes it to **0.96**.

The numbers are not the interesting part. The interesting parts are a model that posts the best one-step error in the entire experiment and then leaves the space of ball images altogether by its thirtieth prediction, and a comparison that looks like an eleven-fold win until you subtract the decoders.

**What this post assumes:** part 1's VAE, its probe protocol, and the dataset splits. **What you'll be able to do afterwards:** derive calibrated reference levels for a pixel-error curve so the numbers mean something; explain why teacher-forced training and open-loop evaluation sample from different input distributions; implement KL balancing and test that it does what you think; and say precisely what joint training buys, in $R^2$, in pixels, and in the parts where it buys nothing.

---

## The recipe

Ha and Schmidhuber's *World Models* (2018) has three parts: **V**, a VAE that compresses frames to latents; **M**, a recurrent network that predicts the next latent; and **C**, a tiny controller evolved with CMA-ES. I am building V and M and skipping C, because the comparison here is between world models rather than between controllers, and the last section replaces C with something better anyway.

The defining property of the recipe is not the GRU. It is the **seam**.

<figure class="wm-fig wm-swap">
  <img class="wm-light" src="/images/world-models/diagrams/ha-pipeline-light.png" alt="Phase 1 trains a VAE on single frames. Phase 2 freezes the encoder and trains a GRU on latent sequences, with a crossed-out arrow showing that the dynamics loss never reaches the encoder.">
  <img class="wm-dark" src="/images/world-models/diagrams/ha-pipeline-dark.png" alt="Phase 1 trains a VAE on single frames. Phase 2 freezes the encoder and trains a GRU on latent sequences, with a crossed-out arrow showing that the dynamics loss never reaches the encoder.">
  <figcaption><strong>Figure 1.</strong> Ha and Schmidhuber's split: V, then M. The crossed arrow is the whole design.</figcaption>
</figure>

Two training phases, deliberately separate. Phase 1 trains the VAE on single frames; phase 2 freezes it, encodes every trajectory into latent sequences, and trains a recurrent model on those. The crossed arrow in figure 1 is the design: the dynamics loss never reaches the encoder. Whatever information reconstruction happened to preserve is all the dynamics model will ever get, and however the encoder happened to arrange that information is the coordinate system the dynamics model has to work in.

Part 1 measured both of those costs in advance. The encoder keeps about 5.4 nats per frame, and its code is curled enough that a linear probe reads position at 0.75 rather than 0.95. Now we find out what a recurrent model does with that.

Why split at all? Because in 2018 it was the thing that worked. Training a big conv encoder jointly with a recurrent model was unstable, and decoupling them turns one hard optimization into two easy ones. It also makes the VAE reusable across tasks. The cost is what the second half of this post measures.

## The model

One cell, thirty lines, no tricks:

$$h_t = \mathrm{GRU}\big(h_{t-1}, [z_{t-1}, a_t]\big), \qquad \mu_t, \sigma_t = \mathrm{MLP}(h_t)$$

$$\mathcal{L} = -\log \mathcal{N}\big(z_t \mid \mu_t, \sigma_t\big)$$

```python
class GRUDynamics(nn.Module):
    latent_dim: int = 16
    action_dim: int = 2
    hidden: int = 128
    residual: bool = False

    @nn.compact
    def __call__(self, h, z_prev, action):
        x = jnp.concatenate([z_prev, action], axis=-1)
        h, _ = nn.GRUCell(features=self.hidden)(h, x)
        y = nn.silu(nn.Dense(128)(h))
        mu = nn.Dense(self.latent_dim)(y)
        sigma = jax.nn.softplus(nn.Dense(self.latent_dim)(y)) + 1e-3
        if self.residual:
            mu = z_prev + mu
        return h, mu, sigma
```

Four details in there that a paper would compress into half a sentence, each of which cost me a decision:

**The hidden state is 128, and stays 128 for the rest of the post.** The RSSM later uses the same 128 for its deterministic state. Probe comparisons across models are only meaningful if the states being probed have the same dimension, because a linear map from a wider vector fits better for free.

**$\sigma$ has a floor of $10^{-3}$.** Gaussian NLL is unbounded below: a model that drives $\sigma \to 0$ on an easy target earns arbitrarily large likelihood and takes the loss with it. `softplus(raw) + 1e-3` makes that impossible. Every likelihood-based model in this project has a floor somewhere, and it is the cheapest instability insurance there is.

**One Gaussian, not a mixture.** Ha and Schmidhuber use an MDN-RNN, and the mixture earns its place when the next-latent distribution is genuinely multimodal. For a ball in a box it mostly isn't. The one candidate is a wall bounce, where "does it hit this step or next step" is a real either/or. I left the MDN as an ablation to run if the single Gaussian visibly underfit at bounces. It didn't, so I didn't.

**Latents are standardized per dimension before the GRU sees them**, with the mean and std fit on the training episodes and stored in the checkpoint. The VAE's 16 dimensions have wildly different scales, since part 1 showed nine active and seven idling near the prior. Without standardization the NLL is dominated by whichever dimensions happen to have large variance, and the model spends its capacity there.

### Where velocity comes from

Before running anything: why should this work at all? Nothing in the architecture is smarter than the VAE. The GRU cannot see pixels, only the same frozen 16 numbers per frame that scored zero on velocity.

The answer is that the GRU state is the first object in the pipeline that ever sees a *sequence*. Velocity is $v_t \approx (p_t - p_{t-1})/\Delta t$: a function of two frames, computable from a history, absent from any single frame. The encoder maps one frame at a time and structurally cannot represent it. $h_t$ is a function of everything up to $t$, so it can.

*Can* is not *will*. The reason it must is in the loss. The NLL is minimized by predicting the conditional expectation,

$$\mu^\star(h_t) = \mathbb{E}\big[z_t \mid z_{0:t-1}, a_{1:t}\big]$$

and for a ball with momentum that expectation depends on position *and* velocity. A state that dropped velocity would predict the same next frame for a ball moving left as for one moving right, and pay for it in likelihood at every step of every sequence. The objective does not merely permit a velocity estimate. It sculpts one.

## Training, and the two switches

Subsequences of 33 frames (32 transitions), sampled uniformly from the training episodes, batch 64, adam at $10^{-3}$, 20k steps, one `lax.scan` over time. The speed cap is 2 pixels per step and the dataset's mean speed is 1.41, so a crossing of the 32-pixel box takes somewhere between 16 and 23 steps. Thirty-two transitions contain a couple of bounces, which is long enough that the model cannot get away with linear extrapolation.

Two implementation choices worth naming:

**$h_0 = 0$, and the first four transitions are excluded from the loss.** With no history the velocity is genuinely unknowable, so those steps would only teach the model to hedge, predicting a blurry average over directions it has no way to distinguish. Masking them is not cheating; it is refusing to train on a question with no answer. The same warm-up of 4 is used at evaluation time, for both models in this post, so the protocols line up exactly.

**Training is teacher forced.** At every step the GRU receives the *true* $z_{t-1}$, never its own prediction.

That last one is standard, and it sets up the most useful failure in this post, so hold it in view:

<figure class="wm-fig wm-swap">
  <img class="wm-light" src="/images/world-models/diagrams/teacher-openloop-light.png" alt="Two unrolled GRU chains. In training every input is a real encoded frame. In open-loop evaluation the model's own predictions feed back in as inputs after a four-step warm-up.">
  <img class="wm-dark" src="/images/world-models/diagrams/teacher-openloop-dark.png" alt="Two unrolled GRU chains. In training every input is a real encoded frame. In open-loop evaluation the model's own predictions feed back in as inputs after a four-step warm-up.">
  <figcaption><strong>Figure 2.</strong> Two different input distributions. The orange feedback path exists only at evaluation time.</figcaption>
</figure>

Figure 2 shows the two regimes side by side. In training (top) every input box is a real encoded frame. In open-loop evaluation (bottom) only the warm-up inputs are real; after that the model eats its own output, and the orange arrows are the feedback path that does not exist during training. The model is being evaluated on the *one* input distribution it was never trained on: slightly-wrong latents. Keep that picture, because it explains three of the four results below.

Now the switches. Designing this, I hit two forks that papers usually pick without comment:

1. **Direct or residual?** Should the head predict $z_t$, or the change $z_t - z_{t-1}$? Residual parameterization is standard wisdom for smooth dynamics, since the change is small and easier to fit than the absolute value.
2. **Means or samples?** The frozen encoder emits $q(z \mid o) = \mathcal{N}(\mu, \sigma)$. Do we encode the dataset to posterior means, or to samples?

Instead of choosing, I ran the 2x2. Four GRUs, identical except for those switches, about three GPU-minutes each:

```
uv run train-gru --predict direct   --latent-source mean
uv run train-gru --predict residual --latent-source mean
uv run train-gru --predict direct   --latent-source sample
uv run train-gru --predict residual --latent-source sample
```

One detail keeps the comparison honest. "Residual" is implemented as a mean *offset*, so the likelihood is $\mathcal{N}(z_t \mid z_{t-1} + \mu_\Delta, \sigma)$ and all four models put a density on the same variable, making their likelihoods directly comparable. Predicting a density over $\Delta z$ instead would need a change-of-variables term before any NLL column meant anything.

## Why a small error becomes a large one

Before the numbers, the mechanism that makes this environment hard. A dynamics model rolled out open loop accumulates error, and in a world with walls that error does not grow smoothly. Two trajectories that agree to within a fraction of a pixel stay together until a bounce, and a bounce turns a small disagreement about *where* into a total disagreement about *when*.

Give the two balls below a nudge apart and watch what the wall does to it:

<div id="divergence-demo" class="my-8 not-prose"></div>

Nothing in that demo is a neural network. It is the environment from part 1, run twice from almost the same state, and it puts a floor under every drift curve in this post: before the first mispredicted bounce, errors grow slowly, and after it the two timelines are unrelated.

## How to read a pixel-error curve

The evaluation is an **open-loop rollout**: warm up on 4 real transitions, then run the model forward on its own mean predictions for 30 steps, decoding each predicted latent, and measure pixel MSE against the real continuation. `rollout.py` implements it once and both models in this post call it, so "drift at horizon $k$" is the same quantity for both by construction.

But a pixel MSE of 0.018 means nothing until you know what the scale is. Two reference levels make every number readable, and both are worth deriving.

**Level 1, blind but honest.** A model that gives up and paints the dataset's mean image has per-pixel error equal to the pixel variance:

$$\mathbb{E}\lVert o - \bar{o} \rVert^2 / P = \mathrm{Var}(o_i) = 0.1045^2 \approx 0.011$$

**Level 2, hallucinating confidently.** A model that draws a perfectly good ball in an *uncorrelated* place does twice as badly. For independent $A$ and $B$ with the same statistics,

$$\mathbb{E}\lVert A - B \rVert^2 = \mathbb{E}\lVert A - \bar{o}\rVert^2 + \mathbb{E}\lVert B - \bar{o}\rVert^2 \approx 2 \times 0.011$$

the cross term vanishing in expectation. So about 0.011 is "knows nothing, admits it", about 0.022 is "knows nothing, insists otherwise", and anything far above 0.022 means the model has left the space of ball images altogether.

There is a third level, and it is specific to each pipeline rather than to the data. Every GRU prediction is decoded by the frozen decoder from latents produced by the frozen encoder, so no prediction can beat that autoencoding round trip. Part 1's `warmup5k` reconstruction error was 0.28 summed over 1024 pixels:

$$\text{floor} = 0.28 / 1024 \approx 0.00027 \text{ per pixel}$$

That is the best a perfect dynamics model could do on this pipeline. Hold onto it, because the RSSM's floor is completely different and the comparison at the end is meaningless without both.

It also heads off a units trap that would otherwise make this whole section look like a regression against part 1: that post's headline reconstruction number was 0.28, this one's headline drift number is 0.0004, and they are neither the same quantity nor the same exam. One is summed over 1024 pixels and graded with the answer in hand. The other is per pixel, on a frame the model has never seen.

## Four GRUs

| run | val NLL | drift@1 | drift@5 | drift@15 | drift@30 | position $R^2$ | velocity $R^2$ |
|---|---|---|---|---|---|---|---|
| `direct-mean` | -21.6 | 0.00041 | 0.0060 | 0.0178 | 0.0210 | 0.979 | **0.811** |
| `residual-mean` | -20.7 | **0.00035** | **0.0053** | 0.0196 | 0.139 | 0.976 | 0.766 |
| `direct-sample` | +19.4 | 0.0037 | 0.0113 | 0.0180 | 0.0191 | 0.964 | 0.756 |
| `residual-sample` | +19.5 | 0.0041 | 0.0098 | **0.0162** | **0.0177** | 0.971 | 0.742 |

Bold marks the best value in each column, and one number in that table is bad enough to spot without help. Here is the same data as curves, with the two reference levels drawn in as dashes:

<figure class="wm-fig">
  <img src="/images/world-models/gru-drift-comparison.png" alt="Left panel: four drift curves on a log axis against open-loop horizon, three flattening between two dashed reference lines and one climbing past both. Right panel: a bar chart of velocity probe R squared, 0.81, 0.77, 0.76, 0.74.">
  <figcaption><strong>Figure 3.</strong> Open-loop drift for the four variants, and the velocity probe this step existed to move.</figcaption>
</figure>

Three of the four curves rise and flatten between the dashed lines. The fourth keeps going. Taking those in order.

### The result this half existed for

Velocity is linearly readable: 0.74 to 0.81 across all four variants, against exactly 0.00 from the encoder alone. The recurrence integrates motion, as the objective argument said it had to.

The one-step number deserves a second look too. `direct-mean` predicts the next frame at 0.00041 per pixel against a pipeline floor of 0.00027, so **one step into the future costs about fifty percent more than repainting the present**. For a model that has never seen a pixel in its life, that is a startlingly good short-horizon predictor.

Position is the sneaky one: $h$ probes at 0.979 where the raw latent managed 0.75. The GRU beats its own input. Two mechanisms and one caveat. Filtering first: each frame's encoding is a noisy measurement of position, and a recurrence that has integrated many of them can average, with estimation error falling roughly like $1/T$, which is Kalman logic with the matrices learned rather than derived. Untangling second: part 1 showed position is present but curled in the latent, and a GRU that must *use* position to predict re-represents it in coordinates of its own, where 128 dimensions give the representation room to lie flat.

The caveat: a 128-dimensional feature vector flatters a linear probe relative to a 16-dimensional one on capacity alone. The 0.979-against-0.75 comparison mixes "better organized" with "more room to be organized in", and I cannot separate them with this experiment. It is precisely why $h$ stays pinned at 128 for the RSSM, where the comparison is between two 128-dimensional states and the confound cancels.

### The switch that detonates

**Residual prediction, trained on means, posts the best one-step error in the grid, and then explodes.** By horizon 30 it sits at 0.139, more than six times above the hallucination ceiling. On a log axis it is the orange curve that never flattens.

<figure class="wm-fig">
  <img src="/images/world-models/gru-filmstrip-residual-mean.png" alt="Two rows of eight frames at increasing horizons. The top row is real. The bottom row matches early, then develops a vertical smear at the left edge and an oversized over-bright blob by the last frames.">
  <figcaption><strong>Figure 4.</strong> The residual-mean rollout. Top row real, bottom row open loop. By k=22 the bottom row contains things that are not balls.</figcaption>
</figure>

Through $k=6$ the two rows are indistinguishable. By $k=22$ there is a vertical smear at the left edge that is not a ball and has never been a ball; by $k=30$ the predicted blob is oversized, over-bright and still accompanied by its streak. That is what decoding an off-manifold latent looks like: the decoder was trained on encoder outputs, and it is being handed a vector no encoder would ever produce.

The mechanism is the oldest one in numerical integration. Open loop, the residual model computes $\hat{z}_{t+1} = \hat{z}_t + \mu_\Delta$, a running sum. Every step's small bias accumulates, nothing pulls the sum back toward the set of latents the decoder has seen, and the state random-walks out of the data distribution. The direct model's update is $\hat{z}_{t+1} = \mu(h_t)$, whose output *is* a fresh prediction of a plausible latent at every step, so errors saturate at "plausible but wrong" instead of compounding to "impossible".

Same objective. Nearly identical teacher-forced scores (-20.7 against -21.6 NLL). Completely different failure geometry. **One-step metrics cannot see this**, which is the practical lesson: if you are going to roll a model out for 30 steps, you have to measure it at 30 steps.

### The switch that rescues it

**Training on samples costs you the short game and wins the long one.** The sample-trained runs are about 9 times worse at horizon 1, because the encoder's posterior noise $z = \mu + \sigma\epsilon$ is an error floor no model can beat, but both plateau *below* every mean-trained run by horizon 30, and `residual-sample` goes from 0.139 to 0.0177, best in the grid.

Figure 2 is the explanation. Teacher forcing trains the model on inputs from the real data distribution; open loop feeds it its own slightly-wrong outputs, which are inputs it has never seen, *unless* it trained on noisy latents, in which case "slightly wrong" has been in-distribution all along. Noise injection at training time buys robustness exactly where open-loop rollouts spend it.

It is not a coincidence that Dreamer, the architecture in the second half of this post, feeds its dynamics model samples rather than means. I had read that as an implementation detail for years. The grid says it is load-bearing.

One column in that table is a trap, and I want to name it rather than quietly not print it. The NLLs are only comparable *within* a latent source: sampled targets carry the posterior's irreducible entropy, so their likelihood floor sits about 40 nats above the mean-trained runs'. The mean-trained models are not "better" at -21.6 versus +19.4; they are being graded on an easier exam. Metrics with different floors are not one leaderboard.

### Where they all end up

All four curves bend up toward the 0.022 line by $k \approx 15$ to $20$ and flatten there. That plateau is the frozen-encoder recipe's ceiling, and it has a physical meaning: about one bounce of validity.

<figure class="wm-fig">
  <img src="/images/world-models/gru-filmstrip-direct-mean.png" alt="Two rows of eight frames at increasing horizons. The top row is real. The bottom row tracks it closely at first, then drifts to a different part of the box while still drawing a clean round ball.">
  <figcaption><strong>Figure 5.</strong> The best frozen model failing. Every frame in the bottom row is a perfectly good ball, on a timeline that stopped being ours.</figcaption>
</figure>

Through $k=10$ it tracks. At $k=15$ the real ball is centre-left and the imagined one is drifting low and right. At $k=22$ the real ball has reached the bottom edge and the imagined one is in the bottom-right corner. By $k=30$ they share nothing but a colour scheme, and the imagined ball is still a *perfect ball*, round, correctly sized, sitting exactly where a ball could plausibly be.

In motion, real on the left and imagination on the right, you can watch the exact moment the two worlds part company:

<figure class="wm-fig wm-pixel">
  <img src="/images/world-models/gru-rollout-direct-mean.gif" alt="Animated side by side. Left, the real episode. Right, the open-loop imagination, tracking for a while and then diverging after a bounce.">
  <figcaption><strong>Figure 6.</strong> Thirty steps of open loop, played out.</figcaption>
</figure>

This environment makes compounding vicious, and the reason generalizes to every physical world model: a small error in position or velocity changes *when and whether the next bounce happens*, exactly as the two-ball demo showed. Before the first mispredicted bounce, errors grow smoothly. After it, the real and imagined trajectories separate at full speed and never re-converge. The drift curve is not measuring gradual blur. It is measuring how long the model's timeline stays synchronized with ours.

---

## Removing the seam

So the frozen recipe works, and it has a ceiling: every variant is drawing a confident ball in the wrong place by twenty steps. The question the rest of this post exists to answer is how much of that ceiling is the frozenness itself.

There are two candidate explanations sitting in the results above. The GRU spends capacity untangling and denoising a code that was shaped by reconstruction alone, which part 1 called the untangling tax. And it cannot push information *back* into the encoder, so anything reconstruction discarded is gone for good. Both are properties of the seam, not of recurrence.

Dreamer's answer is to delete the seam. One model, one loss, encoder trained by the prediction objective from the first gradient step.

## Predict, then correct

Every filtering algorithm ever written has the same two beats. **Predict** where the world should be, from what you knew and what you did. Then **correct** that prediction with what you actually observe. A Kalman filter does it with matrices; an RSSM does it with a GRU and two small MLPs, and learns all of them.

The state comes in two pieces, and the split is the design:

- $h_t$, deterministic, 128 dimensions, carried by a GRU. This is memory. Given the past, it is a known quantity.
- $z_t$, stochastic, 16 dimensions, resampled every step. This is everything the frame told you that memory could not have predicted.

Why both? A purely deterministic state cannot represent uncertainty, and this world has genuine uncertainty in it, since the random action nudges are unpredictable by construction. A purely stochastic state has to re-derive the entire past at every step through a sampling bottleneck. Splitting them lets the model route predictable structure through $h$, where it is cheap and exact, and reserve the noisy channel for what actually needs it.

The other half of the design is that the model gets asked the same question twice, once with the answer hidden and once with it visible. The **prior** $p(z_t \mid h_t)$ must guess the stochastic state from memory alone. The **posterior** $q(z_t \mid h_t, o_t)$ gets to look at the frame. Forcing the first to agree with the second is how dynamics get learned, and that forcing is a KL term.

## The cell

<figure class="wm-fig wm-swap">
  <img class="wm-light" src="/images/world-models/diagrams/rssm-cell-light.png" alt="One RSSM step. A GRU core takes the previous state and action to produce h. A dashed path goes to the prior head with no frame. A solid path takes the frame through an encoder to the posterior head. A KL bracket links the two heads. The sampled z and h feed a decoder.">
  <img class="wm-dark" src="/images/world-models/diagrams/rssm-cell-dark.png" alt="One RSSM step. A GRU core takes the previous state and action to produce h. A dashed path goes to the prior head with no frame. A solid path takes the frame through an encoder to the posterior head. A KL bracket links the two heads. The sampled z and h feed a decoder.">
  <figcaption><strong>Figure 7.</strong> One RSSM step. The KL bracket is the only thing connecting prediction to perception.</figcaption>
</figure>

One step, given frame $o_t$ and action $a_t$:

$$h_t = \mathrm{GRU}\big(h_{t-1}, [z_{t-1}, a_t]\big)$$

$$\text{prior:}\quad p(z_t \mid h_t) = \mathcal{N}\big(\mu_p(h_t), \sigma_p(h_t)\big)$$

$$\text{posterior:}\quad q(z_t \mid h_t, e_t) = \mathcal{N}\big(\mu_q(h_t, e_t), \sigma_q(h_t, e_t)\big), \quad e_t = \mathrm{enc}(o_t)$$

$$z_t \sim q, \qquad \hat{o}_t = \mathrm{dec}\big([h_t, z_t]\big), \qquad \hat{r}_t = r\big([h_t, z_t]\big)$$

Trace the two paths in figure 7. The dashed one goes up: $h_t$ to the prior head, no frame anywhere near it. The solid one goes down: the frame enters through the encoder, joins $h_t$, and produces the posterior. They meet at the orange bracket, which is the only thing in the architecture connecting prediction to perception.

Three choices in there that the first half of this post paid for:

**The decoder sees $[h_t, z_t]$, not $z_t$ alone.** Reconstruction can lean on the deterministic path, which frees $z$ to carry only what $h$ could not predict. This is Dreamer's arrangement, and it is most of the explanation for the reconstruction result further down.

**$z_t$ is always sampled, never the mean.** The 2x2 grid above showed that feeding a dynamics model noisy inputs is what makes it robust to its own errors in open loop. Here it is not a training trick; it is what the architecture does.

**$\sigma$ has a floor of 0.1 on both heads**, ten times higher than the GRU's. It is the Dreamer default, and it doubles as insurance against the $\sigma$-collapse failure mode described above.

The encoder and decoder are the same conv stacks as part 1's VAE, and $h$ is the same 128 as the GRU's. That is deliberate, and it holds up when you count: the RSSM is 1.57M parameters, and the frozen pipeline it replaces is 1.39M (VAE) plus 0.077M (GRU), so 1.47M. Seven percent apart. Whatever is about to happen to the drift curve, it is not because I brought a bigger model. **Nothing about the networks got smarter. Only the training signal changed.**

## The loss

$$\mathcal{L} = \sum_t \Big[ \lVert o_t - \hat{o}_t \rVert^2 + \lVert r_t - \hat{r}_t \rVert^2 + \beta \mathrm{KL}\big(q(z_t \mid h_t, o_t) \Vert p(z_t \mid h_t)\big) \Big]$$

The reward term is switched off for everything in the comparison below: the plain ball has no reward, so its targets are all zeros and the head contributes nothing. It exists because an agent will need it, and it earns its keep in the last section, since a world model that cannot predict reward cannot host a policy.

Structurally this is part 1's ELBO with the sequence wrapped around it, same reconstruction term, same KL. One thing changed, and it changes everything: **the KL is against a learned prior**, not against $\mathcal{N}(0, I)$.

That single difference is why the collapse that ate part 1's first run never happened here. I trained with $\beta = 1$ from step zero, warmup knob untouched, and the KL settled at about 1.4 nats and stayed alive for 20k steps. The reason is a change in what "ignore the latent" costs. Against a fixed prior, the model can drive the KL to zero by matching $\mathcal{N}(0,I)$ and pay nothing, which on 98%-black frames is a comfortable optimum. Against a prior that can *chase* the posterior, the KL is only large when the frame carried something the dynamics genuinely failed to predict, so carrying information is cheap from the first gradient step and there is no collapsed configuration to fall into.

<figure class="wm-fig">
  <img src="/images/world-models/rssm-loss-curves.png" alt="Three panels: total loss falling and flattening, reconstruction falling with validation tracking training, and KL falling from about 8 nats to about 1.4 without reaching zero.">
  <figcaption><strong>Figure 8.</strong> RSSM training. The KL panel on the right is the one to read: it settles, and it never touches zero.</figcaption>
</figure>

What I look for in the third panel is the shape: KL falls from about 8 nats and settles at 1.4 without ever touching zero, and validation tracks training the whole way. A KL pinned at zero is the collapse we know by name from part 1; a KL that oscillates or climbs is an unstable prior. This one is neither. The spike at step 4000 is real, in both the train and validation traces, and it recovers within a few hundred steps. I have no explanation for it beyond a hard batch, which is the honest thing to say about a bump that never recurred.

### The one trick: KL balancing

$\mathrm{KL}(q \Vert p)$ has two arguments and one gradient, and left alone it pulls both ways at once: the prior moves toward the posterior *and* the posterior gets dragged toward whatever the prior currently believes. Early in training the prior believes nothing useful, so that second pull is a regularizer toward ignorance.

Dreamer's fix is to split the gradient with stop-gradients while leaving the *value* alone:

$$\mathrm{KL}_t = \alpha \mathrm{KL}\big(\mathrm{sg}(q) \Vert p\big) + (1 - \alpha) \mathrm{KL}\big(q \Vert \mathrm{sg}(p)\big), \qquad \alpha = 0.8$$

<figure class="wm-fig wm-swap">
  <img class="wm-light" src="/images/world-models/diagrams/kl-balancing-light.png" alt="A thick arrow carrying 80 percent of the gradient pulls the prior toward a stop-gradiented posterior. A thin arrow carrying 20 percent regularizes the posterior toward a stop-gradiented prior.">
  <img class="wm-dark" src="/images/world-models/diagrams/kl-balancing-dark.png" alt="A thick arrow carrying 80 percent of the gradient pulls the prior toward a stop-gradiented posterior. A thin arrow carrying 20 percent regularizes the posterior toward a stop-gradiented prior.">
  <figcaption><strong>Figure 9.</strong> KL balancing. Same KL value either way; only the gradient is split.</figcaption>
</figure>

```python
def kl_balanced(mu_q, sig_q, mu_p, sig_p, alpha: float = 0.8):
    sg = jax.lax.stop_gradient
    toward_prior = kl_gauss(sg(mu_q), sg(sig_q), mu_p, sig_p)
    toward_post = kl_gauss(mu_q, sig_q, sg(mu_p), sg(sig_p))
    return alpha * toward_prior + (1.0 - alpha) * toward_post
```

Read it as a negotiation with deliberately asymmetric pressure. Eighty percent of the gradient pulls the prior toward what perception concluded, which is how the dynamics get learned. Twenty percent regularizes perception toward what was predictable, enough to keep the posterior from inventing detail the prior can never match, not enough to collapse it into the prior's ignorance.

This is the single most load-bearing stabilization trick in the build, and it is four lines. It is also easy to get backwards, so there is a test: at $\alpha = 1$ the posterior parameters receive exactly zero KL gradient, checked with `jax.grad`. Write that test. Sign errors here produce a model that trains, converges, and is quietly wrong.

### Training

The whole per-sequence computation is one `lax.scan` with carry $(h, z)$, which is also, and this is the part that pays off in the last section, *exactly* the loop that generates imagination once you swap the posterior for the prior:

```python
def step(carry, xs):
    h, z = carry
    e_t, o_t, a_t, r_t, n_t = xs[:5]
    h = model.apply(params, h, z, a_t, method=RSSM.core_step)
    mu_p, sig_p = model.apply(params, h, method=RSSM.prior_dist)
    mu_q, sig_q = model.apply(params, h, e_t, method=RSSM.post_dist)
    z = mu_q + sig_q * n_t
    o_hat = model.apply(params, h, z, method=RSSM.decode)
    recon = ((o_hat - o_t) ** 2).sum(axis=(1, 2, 3)).mean()
    klb = kl_balanced(mu_q, sig_q, mu_p, sig_p, config.alpha).mean()
    ...
```

Elided: the reward head's MSE term and the carry's return value. The full function is `make_losses` in `train_rssm.py`.

Inputs are pixel sequences now, not precomputed latents: batches of 32 sequences of 33 frames, with the encoder inside the training graph. Everything else is pinned to the frozen run, with the same dataset file, the same splits, the same 32 transitions, the same warm-up 4 and horizon 30 for evaluation, and the same ridge probe.

One protocol difference worth explaining. The GRU masked the first four transitions out of the loss because with $h_0 = 0$ the velocity was unknowable. Here there is no mask: the posterior sees $o_t$ at every step, so early states are *corrected* rather than blind. The evaluation still uses warm-up 4, for comparability.

It trained in one shot. No learning-rate drop, no beta warmup, no instability, and I had lined up both levers in the design doc and used neither. RSSMs have a reputation for being touchy, and I record the non-event here so that the next person's expectations are calibrated by something other than my anxiety.

## The comparison

| metric | VAE latent $z$ | best frozen GRU ($h$) | joint RSSM ($h$) |
|---|---|---|---|
| velocity $R^2$ | 0.00 | 0.811 | **0.96** |
| position $R^2$ | 0.75 | 0.979 | **0.998** |
| drift @1 | n/a | 0.00035 | **0.000031** |
| drift @5 | n/a | 0.0053 | **0.00039** |
| drift @15 | n/a | 0.0162 | **0.0036** |
| drift @30 | n/a | 0.0177 | **0.0129** |
| reconstruction (summed) | 0.28 | inherits the VAE | **0.033** |

The middle column is generous on purpose: it takes the best of the four GRU variants *per row*, so the frozen recipe is represented by `residual-mean` on the short horizons, `residual-sample` on the long ones and `direct-mean` on the probes. RSSM probes are on $h$ alone, 128 dimensions, capacity-matched against the GRU's $h$, which is the comparison this whole post was built to make fair. Adding $z$ to the probe input buys about 0.003 more.

<figure class="wm-fig">
  <img src="/images/world-models/three-model-comparison.png" alt="Left: three drift curves on a log axis, the joint model an order of magnitude below the two frozen ones and still under the upper reference line at horizon 30. Right: a bar chart of velocity probe R squared reading 0.00, 0.81 and 0.96.">
  <figcaption><strong>Figure 10.</strong> Frozen against joint, on the drift curve and on the probe.</figcaption>
</figure>

The blue curve enters an order of magnitude below both frozen models: 11 times better at one step, 13 times at five. Before reading that as a dynamics result, subtract the decoders.

Each pipeline's drift is bounded below by its own reconstruction error, meaning what that decoder manages when it *is* handed the frame, and those two floors are nothing alike: 0.00027 per pixel for the frozen stack, 0.000032 for the joint one. Measured against its own floor, the frozen GRU's best one-step prediction sits about 30% above it. The RSSM's sits *at* it: 0.000031 against a floor of 0.000032, which is as close as two separately-measured numbers ever get. So the faithful one-step statement is not "11 times better". It is that predicting one step ahead costs the RSSM nothing measurable beyond redrawing the frame it already has, while the frozen model pays about 30% extra for the same step, and that the 11 times is that fact multiplied by a better decoder.

The horizons where the floors stop mattering are the interesting ones anyway. By $k = 30$ the frozen model sits 65 times above its floor and the joint one 400 times above its own, so the decoders have dropped out of the comparison entirely. There the RSSM reads 0.0129, just above the mean-image line at 0.011, and nowhere near the 0.022 ceiling where every frozen variant had already parked. Read against the reference levels derived earlier, that says the joint model still retains genuine information about the true trajectory at a horizon where the frozen recipe had fully decorrelated into "a perfect ball in an uncorrelated place".

<figure class="wm-fig">
  <img src="/images/world-models/rssm-filmstrip.png" alt="Two rows of eight frames at increasing horizons. The rows are nearly identical through the middle of the sequence, including a wall bounce at k equals 22, and separate slightly by the last frame.">
  <figcaption><strong>Figure 11.</strong> Thirty steps of RSSM imagination. Look at k=22, where both balls are at the bottom edge mid-bounce.</figcaption>
</figure>

Through $k=15$ the rows are hard to tell apart. At $k=22$, and this is the frame I keep coming back to, the imagined ball is at the bottom edge, *mid-bounce*, where the real one is. No frozen variant ever predicted a bounce correctly that far out, and a bounce is where the dynamics are least linear and the smallest position error flips the outcome. By $k=30$ the two have finally separated, which is exactly what drift@30 = 0.0129 says: not tracking any more, not decorrelated either.

### Two things worth understanding about why

**The encoder stopped being an adversary.** In the frozen pipeline the GRU spent capacity untangling and denoising a code that had been shaped by reconstruction alone. Here the encoder is shaped by the prediction loss from the start, so the latent arrives prediction-ready.

The cleanest evidence is the reconstruction row, and it is the result I would have bet against: **0.033 summed error, against the standalone VAE's 0.28**. Joint training reconstructs the frames *better* than a model that does nothing but reconstruct frames. The two numbers are not measured identically, and the difference runs *against* the result rather than for it: the VAE's figure decodes the posterior mean, while the RSSM's decodes a posterior sample, so the joint model is being graded with its own sampling noise included and still wins by more than eight times. I expected a reconstruction-versus-dynamics trade and got the opposite, because $h$ hands the decoder temporal context that no single-frame encoder ever had, and knowing where the ball was for the last thirty frames is extremely useful information for drawing where it is now. The untangling tax from part 1 simply is not levied when the code is prediction-shaped from birth.

**The KL is a task-complexity meter.** Its equilibrium value reads as "how many nats does each frame carry that the dynamics could not predict". For this environment it settles at about 1.4, roughly the injected action noise plus sub-pixel residue, which is the genuinely unpredictable part of the world. Do not compare that against part 1's 5.4 nats: same units, different question. The VAE's KL is measured against a fixed $\mathcal{N}(0, I)$ and prices the *whole* content of the code, position included; this one is measured against a learned prior that already knows where the ball was going, and prices only the surprise.

Train the same architecture on the scene with a static goal ball in it and the KL settles near 4; let the goal move and it goes to 5 or 6. The number knew the difficulty ordering of the three tasks before I did, and it costs nothing to log.

## Then we let it dream

A world model earns the name by hosting training. So: a new version of the environment with a red goal ball rendered *into the pixels*, because the reward depends on the goal and if the goal were invisible no reward head could learn it from state. The observation goes RGB, and the reward is a Gaussian kernel on the agent-to-goal distance. One knob covers two tasks: the goal either sits still (**hover**) or cruises and bounces at half the agent's speed cap (**follow**).

The recipe is offline Dreamer. Freeze the world model, then train an actor and a critic entirely on imagined rollouts:

$$a_i = a_{\max} \tanh\big(\mu_\pi(s_i) + \sigma_\pi(s_i) \epsilon\big), \qquad s = [h, z]$$

Start from real posterior states filtered out of the replay data, roll the **prior** forward 15 steps under the actor, read rewards from the reward head, and score with TD($\lambda$):

$$R_i = r_{i+1} + \gamma\big[(1-\lambda) V(s_{i+1}) + \lambda R_{i+1}\big], \qquad \gamma = \lambda = 0.95$$

The critic regresses onto $\mathrm{sg}(R)$; the actor maximizes $R$ directly. Because every piece is reparameterized, with Gaussian latents, tanh-Gaussian actions and a differentiable reward head, the value gradient flows *through the dynamics themselves*. No REINFORCE, no target network, no environment interaction. The policy improves by backpropagating through its own imagination.

<figure class="wm-fig">
  <img src="/images/world-models/ac-training-eval.png" alt="Left: imagined reward sitting near 0.2 while real reward climbs past 0.8 over training. Right: a bar chart of final evaluation, actor at 0.65 and 0.88 against baselines near 0.05, with a large error bar on hover and a small one on follow.">
  <figcaption><strong>Figure 12.</strong> Training in imagination, evaluated in reality. The gap in the left panel is not what I expected to find.</figcaption>
</figure>

Evaluated in the real environment, 100 fresh episodes, against zero-action and random-nudge baselines:

| task | actor | zero | random | actor std |
|---|---|---|---|---|
| hover | 0.652 | 0.055 | 0.055 | 0.228 |
| follow | **0.879** | 0.048 | 0.049 | **0.025** |

<figure class="wm-fig">
  <img src="/images/world-models/ac-rollout-strips.png" alt="Two rows of frames. Top: the agent dives at a static goal and parks on it. Bottom: the agent catches a moving goal and rides it through bounces.">
  <figcaption><strong>Figure 13.</strong> Top row hover, bottom row follow. Frame seven of the bottom row catches the one moment it loses the lock.</figcaption>
</figure>

The follow task in motion, which is the shot this whole series was built to take:

<figure class="wm-fig wm-pixel">
  <img src="/images/world-models/ac-follow-rollout.gif" alt="Animated. A white ball steers toward a red one and then stays with it as it bounces around the box.">
  <figcaption><strong>Figure 14.</strong> A policy whose entire training experience was 15-step hallucinations, running in the real environment.</figcaption>
</figure>

Two findings from that experiment are worth more than the scores.

**The imagination was pessimistic, not exploitative.** The left panel of figure 12 plots imagined reward against real reward. I built it expecting to catch the actor conning its own reward head, the classic model-based failure. The opposite happened: imagined reward sits near 0.2 while real reward climbs past 0.8. The cause is horizon bias. Imagination only ever sees 15-step windows starting from random replay states, most of which are far from the goal, so it mostly measures transit time; real 200-step episodes let the agent park on the goal and collect. **Imagined return is not an estimate of episode return. It is an estimate of local improvement.**

**The harder task produced the better policy, because it produced the better world model.** Follow beats hover 0.879 to 0.652, and the standard deviations are the tell: 0.025 against 0.228. The hover world model overfit its 400 static goal positions, the first train/validation gap of the entire project, and represents the goal weakly, so on some episodes it simply mislocates the target and the actor inherits the error. The follow model never had that option: 400 goal *trajectories* forced it to track, and it probes goal position at 0.99. Its actor locks on every single episode.

The policy is a mirror held up to the world model. Every flaw transfers.

And one more entry for the file part 1 opened on probes lying in specific ways: the hover model's goal probe read 0.12, which looks damning until you notice that a static goal means the probe's fit set contains 25 effectively distinct target values dressed up as thousands of rows. Ridge memorizes 25 points in 128 dimensions and whiffs on the unseen 25. **A probe needs variance in its target within its fit distribution, or it measures nothing.**

## Where this breaks

The parts of all this I would not defend in a paper.

**One seed, one environment, 32x32 pixels.** The RSSM beat the frozen models by margins large enough that seed noise cannot explain them, but every number in the tables is a single run. The gap between 0.998 and 0.979 on the position probe is not a result I would defend; the gap between 0.96 and 0.81 on velocity is. Likewise, the residual-mean divergence is a factor of six and I trust it, while the difference between 0.811 and 0.766 across the four GRUs is four runs at one seed apiece.

**None of the four GRUs was tuned individually.** A residual model with gradient clipping tuned for it, or a lower learning rate, might not detonate. What I showed is that the default recipe detonates, not that the parameterization is unfixable.

**The objective is reconstruction, and that is a real limitation.** Both models learn what they need to draw the frame. On a ball in an empty box, what you need to draw is what you need to predict, so the two objectives agree perfectly. Add a distractor background, waving trees or a scrolling texture or TV static, and reconstruction spends its capacity on the irrelevant pixels because they occupy most of the image. That is the motivation for reconstruction-free objectives, and this environment is built in a way that cannot see the problem. I plan a distractor-background version specifically to make it visible.

**Continuous Gaussian latents.** DreamerV2 and V3 use categorical latents with straight-through gradients, which are better at multimodal futures. For this world the Gaussian is fine, since I never saw a case that needed a mixture, but "fine on a bouncing ball" is a weak claim about anything else.

**The comparison is capacity-matched, not compute-matched.** All the models got 20k steps. The RSSM does far more work per step, with an encoder and decoder inside a 33-step scan versus a GRU over precomputed latents, so it consumed more compute for the same step count. If your constraint is wall-clock rather than architecture, the frozen recipe is cheaper per unit of progress, and its two phases can be debugged independently.

**And the honest scoreboard on the ceiling question.** I asked how much of the frozen recipe's 20-step wall was frozenness. The RSSM at horizon 30 is at 0.0129 and still climbing toward the same reference lines. Joint training did not remove the wall; it moved it. The frozen models cross the mean-image level at $k \approx 10$, the RSSM at $k \approx 28$, so a factor of about three in horizon, plus an order of magnitude in error everywhere short of that. Large practically, not qualitatively: both models eventually lose the timeline. That is what a stochastic environment does to open-loop prediction, and nothing here escapes it.

## Try it yourself

```
uv run make-data
uv run train-gru  --predict direct --latent-source mean
uv run train-rssm --run-name base

uv run make-data --task follow
uv run train-rssm --data data/ball_follow.npz --run-name follow
uv run train-ac --wm-run runs/rssm/follow --data data/ball_follow.npz \
                --goal-speed 1.0 --run-name follow
```

1. **Predict before you measure.** Before running the residual-mean variant, write down what you expect its horizon-30 drift to be. Then look. Nothing else in this project produced a bigger gap between my prediction and the measurement.
2. **Break KL balancing on purpose.** Train the RSSM with `--alpha 0.5` and `--alpha 1.0`. At 1.0 the posterior gets no KL gradient at all. Predict what that does to the KL curve and to the drift before you look.
3. **Take the prior's uncertainty seriously.** The drift curves above roll *prior means*. Roll prior *samples* instead and plot both. The sampled curve sits slightly higher, and the gap is the model's own uncertainty rendered in pixels, which is a free calibration check on whether the $\sigma_p$ it reports is honest.
4. **Kill the action channel.** Retrain the GRU with the action input zeroed. The ball's motion is mostly ballistic, so the drift curve should barely move at short horizons and degrade at long ones, where accumulated nudges matter. If it doesn't degrade at all, your dataset has the bug part 1 described.

## The arc

<figure class="wm-fig wm-swap">
  <img class="wm-light" src="/images/world-models/diagrams/three-models-light.png" alt="Three columns comparing the VAE, the frozen VAE plus GRU, and the joint RSSM, with their losses and velocity probe scores of 0.00, 0.81 and 0.96.">
  <img class="wm-dark" src="/images/world-models/diagrams/three-models-dark.png" alt="Three columns comparing the VAE, the frozen VAE plus GRU, and the joint RSSM, with their losses and velocity probe scores of 0.00, 0.81 and 0.96.">
  <figcaption><strong>Figure 15.</strong> What each architecture lets the prediction loss touch, and what it reads on the probe.</figcaption>
</figure>

Velocity $R^2$: 0.00, then 0.81, then 0.96. Open-loop error at five steps: an order of magnitude. Reconstruction: better from the model that was not only reconstructing. And at the end, an agent that learned to chase a moving target from 500 offline episodes and about five GPU-minutes of dreaming, with pixels in, nudges out, and ground truth touched only to grade it.

The through-line is one sentence, and it is worth more than any of the numbers: **a representation is shaped by the loss that reaches it.** Train an encoder on reconstruction and you get a code that knows what a frame looks like. Train it on prediction and you get a code that knows what the world does next, and, as it turns out, one that also knows better what the frame looks like.

Next stop is the environment this was all rehearsal for: the same recipe, pointed at DOOM's `take_cover`, where nobody hands you the true state and every measurement in this series has to be earned a different way.
