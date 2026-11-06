---
title: "World Models From Scratch, Part 3: The Head That Learned the Wrong Death"
date: "2026-08-16"
tags: ["machine-learning", "world-models", "rssm", "dreamer", "reinforcement-learning", "vizdoom"]
summary: "The same recipe pointed at ViZDoom take_cover, where nobody hands you the true state. A death predictor with a near-perfect loss that almost never fired inside a dream, the one-step test that caught it, and an agent trained in imagination that survives 2.7x longer than random."
readTime: "15 min"
notebook: ""
featured: true
---

# World Models From Scratch, Part 3: The Head That Learned the Wrong Death

<figure class="wm-fig wm-hero wm-pixel">
 <img src="/images/world-models/doom-agent-best.gif" alt="Animated 64x64 grayscale DOOM from the agent's point of view. It strafes left and right, dodging fireballs, for 367 steps.">
 <figcaption>The agent this post builds, on its best of ten episodes: 367 steps, about five times the random average. Every update to its policy happened inside its own world model's dreams.</figcaption>
</figure>

<aside class="tldr">
<p class="tldr-label">TL;DR</p>

- **Same recipe as part 2, pointed at ViZDoom `take_cover`, minus the true state.** The probe that graded the ball cannot be built here. This post is about what replaced it.
- **A death predictor can learn a cue that dreams never show it.** Mine reached a near-perfect loss by detecting a frozen frame. Dreams never freeze, so in imagination it almost never predicted a death. A one-step prior test catches this in seconds.
- **Rare labels get diluted twice**, once per frame and again by the training window, and an easy shortcut can hide that until you remove it.
- **Survival is heavy-tailed, so three evaluation episodes say almost nothing.** Over 100 episodes, the final agent survives 2.7x longer than random: 187 steps against 70.

*In a hurry: read the bold line that opens each section, and the tables. Boxes hold implementation notes and exercises; the post reads with every box closed.*

</aside>

[Part 2](/blog/world-models-part-2) ended with an agent that learned to chase a ball without touching the real environment, and every claim about it was easy to check: the simulator handed over the true position and velocity with each frame, and a linear probe turned any question about the model into a number. This post points the same recipe at DOOM. DOOM gives you a 64x64 screen buffer and nothing else, and on the frame where you die it does not even give you that.

**That missing frame cost me the first world model. Its death predictor reached a loss of 0.0001 by learning that death is the moment the picture stops moving. In the model's own dreams the picture never stops, so the predictor read "alive" on nearly every imagined step, including the ones where a fireball hits, and none of the curves I logged showed a problem.**

Most of this post is about the three checks that replaced the probe: sampled dreams compared with reality by their statistics, a one-step test for the death predictor, and an evaluation large enough for a heavy-tailed score.

**What you need:** part 2's RSSM and the λ-return its actor trains on. Code is at [github.com/RetamalVictor/world-models-from-scratch](https://github.com/RetamalVictor/world-models-from-scratch); the commands are in the box at the end.

---

## 1. What changes on DOOM

**Result of this block: three changes are plumbing and one takes away the probe. The piece that matters is a new head that predicts death, because it is the only way dying reaches the actor.**

The scenario is `take_cover` from Ha and Schmidhuber's *World Models* (2018). You stand at one end of a room, monsters at the other end throw fireballs, and you strafe left or right until one hits you. It ships with the `vizdoom` package. Moving part 2's recipe over took four changes:

- **Bigger frames.** 64x64 grayscale instead of 32x32, with the latent and the GRU state widened to match.
- **Discrete actions.** Left, right, or nothing. Part 2's Gaussian actor becomes a categorical one, kept differentiable with a straight-through sample: a hard one-hot on the forward pass, the softmax's gradient on the backward. Each policy step holds its button for four engine tics.
- **A stateful engine.** ViZDoom is a C++ process on the CPU, so collection is a plain Python loop. Everything after the replay buffer stays jitted.
- **No true state.** Nobody exports the fireball positions, so the probe that graded every model on the ball has nothing to regress onto.

### Where death enters the dream

**The reward is the same on every step, so everything the actor learns about dying comes from a new head on the world model, through one factor in the return.**

The reward is a flat amount per step survived, so the reward head learns a constant and carries no information about danger. That information comes from the continue head, which reads the state and outputs $c$, the probability that the episode carries on past this step. It enters part 2's λ-return as a factor on the discount:

$$R_i = r_{i+1} + \gamma c_{i+1} \big[(1-\lambda) V(s_{i+1}) + \lambda R_{i+1}\big], \qquad \gamma = 0.99, \lambda = 0.95$$

When $c$ drops toward zero the return is cut off at that step, and the actions that led there lose all the reward that would have followed. When $c$ stays at one the return bootstraps straight through. No other term in the actor's objective depends on death. If the head reads 1.0 on every imagined step, walking into a fireball costs nothing and the actor has no reason to dodge.

---

## 2. Getting the death head to work in dreams

**Result of this block: a head can reach a near-perfect loss on a cue that never appears in dreams. Finding that, and fixing it, took two new instruments and three pretrains that each changed one thing.**

Every world model in this block comes from the same pretrain: random play only, 20k world-model updates, no actor, about half an hour on the GPU. I wrote down each run's pass conditions before launching it.

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-dream.gif" alt="Side by side 64x64 DOOM frames. Left, a real episode. Right, the model's own continuation of it, with fireballs it invented.">
 <figcaption><strong>Figure 1.</strong> Left, thirty real frames. Right, the model's continuation after four warm-up frames, with no engine behind it and fireballs it invented. This is the world the agent trains in.</figcaption>
</figure>

### How to look at a dream

**A prior-mean rollout averages the fireballs away. Roll out samples, and compare them with reality by their statistics.**

The first dream filmstrip I made looked fine. The room stayed a room for thirty imagined steps, with none of the drift into gray mush I had braced for. But its warm-up happened to contain no fireball, so it never exercised the one thing the actor depends on, and it rolled out the prior *mean*, which turned out to be the bigger problem.

Part 2 rolled out prior means too, and on the ball that was reasonable, because the motion is close to deterministic and the mean future looks like a likely one. In `take_cover`, whether a monster fires now, in ten steps, or not at all is random. The average over those futures has at most a faint smear where each fireball might have been. A sampled rollout draws one of the futures in full.

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-spawn-blog.png" alt="Four rows of DOOM frames from one fireball-free warm-up. The real row spawns a fireball. The prior-mean row stays empty. Two sampled rollouts spawn their own.">
 <figcaption><strong>Figure 2.</strong> One warm-up with no fireball, rolled out four ways. The real future spawns one at step 21. The prior mean never does, while both sampled rollouts spawn fireballs of their own.</figcaption>
</figure>

Over 32 fireball-free warm-ups, a fireball appears within 30 imagined steps in 3% of prior-mean rollouts, 72% of sampled ones, and 84% of real continuations. The model had been producing fireballs all along. Sampling also changes how a dream should be judged. Two rollouts from the same warm-up disagree, since each draws a different future, so comparing a dream with the real continuation frame by frame means little after a few steps. What should match is the statistics, such as how often fireballs spawn and how long they persist. Every dream in this post is sampled, as are the ones the actor trains in.

<details>
<summary>How I measured fireballs. Open if you will reuse a pixel heuristic.</summary>

Every fireball number passes through one threshold: a pixel over 170 that beats the background median by 40, six connected pixels for a fireball, 800 for an impact. Before trusting any aggregate I spent twenty minutes on contact sheets until the masks landed on fireballs and nothing else. Dreamed fireballs track the real ones for three to five steps, which at four tics per step is most of the time a fireball spends on screen.

</details>

### v1: a perfect loss for the wrong feature

**The v1 continue head reached a loss of 0.0001, against about 0.08 for a head that always says "alive". It was detecting the picture freezing.**

ViZDoom drops its screen buffer the instant you die. My wrapper had nothing to return on the terminal step, so it repeated the last live frame and stored that with the death label. Every death episode in the buffer therefore ended on a frame pixel-identical to the one before it, which made a repeated frame the cheapest feature separating dead from alive.

On real data that feature separates the classes perfectly, which is why the loss looked so good. In imagination it never occurs, because the prior does not generate two identical frames in a row. When I rolled dreams forward from real death windows, the head's median reading stayed above 0.99 at every step, dreamed impacts included.

I knew about the engine quirk. It was in my journal from the week before as a debugging note, and I had not asked what feature it would teach the model. This is shortcut learning (Geirhos et al., 2020): a cue that is perfectly predictive in training and absent at test time. For a world model's heads, test time is imagination. They train on states the posterior built from real frames and are read on states the prior generates, so a cue is only useful if it exists in both.

### The one-step prior test

**Filter the posterior up to the last real frame, take one prior step, and read the head on both states. If the reading changes, the cue is absent from dreams.**

The fix for v1 was one branch in the collection loop, which stops storing the repeated frame. The death label now lands on the last frame the engine rendered, and that frame shows the incoming fireball, something the prior can generate. I'll call the model retrained on this data v2. Checking that its head could use the new cue in a dream needed a test the loss curve could not provide.

A dream is a chain of prior steps, and the test checks one of them. Take a real death window, filter the posterior up to its last frame, and read the continue head. Then take a single prior step, with the real action and no frame, and read the head again. With one step there is no rollout to drift, so any change in the reading comes from moving the state from the posterior to the prior. It runs in seconds. The table shows medians over 256 death windows.

<figure class="wm-fig wm-diagram">
 <img src="/images/world-models/diagrams/one-step-prior.svg" alt="The posterior filters real frames into a state at the death frame. One prior step with no frame produces a second state. The continue head is read on both.">
 <figcaption><strong>Figure 3.</strong> The one-step prior test. Posterior state at the death frame, one prior step, head read on both.</figcaption>
</figure>

| model | at the death frame | one prior step later |
|---|---|---|
| v1, repeated frame | 0.05 | 0.99 |
| v2, last rendered frame | 0.95 | 0.96 |

v1's signal vanishes as soon as the state comes from the prior. v2's reading is unchanged, and that separates two failures that look the same in a rollout, since in both the head reads "alive" all the way to the impact. A head that learned a cue the prior cannot reach needs different data. A head that never learned the cue needs a stronger training signal. v2 is the second kind: posterior and prior agree, and the value they agree on, for the frame where the agent dies, is 0.95.

### The shortcut was hiding the imbalance

**With the artifact gone, the real problem showed. Deaths are about one loss term in a thousand, so a head that says "alive" everywhere is already close to optimal.**

The dilution happens twice. An episode carries one death label in about 70 frames. The model then trains on 33-frame windows sampled from the buffer, and a window contains the label only if it ends on the episode's last frame, one start position in about forty. Even that window holds the label on one frame of 33, so roughly one loss term in 1,300 is a death.

<figure class="wm-fig wm-diagram">
 <img src="/images/world-models/diagrams/death-dilution.svg" alt="One episode as a long bar with a single death label at the end. Four 33-frame windows below it; only the last one reaches the label.">
 <figcaption><strong>Figure 4.</strong> Why deaths are so rare in the loss. The label sits on one frame, and a window only contains it by ending there.</figcaption>
</figure>

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-death-dream-v2-strip.png" alt="Two rows of ten DOOM frames, real on top and dreamed below, with P(continue) under each. A fireball closes to point blank and the head reads 1.00 almost the whole way.">
 <figcaption><strong>Figure 5.</strong> v2 on one sampled dream. A fireball closes to point blank and the head reads 1.00 nearly the whole way. The dynamics imagined the death, and the head did not respond to it.</figcaption>
</figure>

v1 had the same imbalance and never showed it. The frozen-frame cue was so easy that the head learned it despite the tiny class weight, and the loss looked converged. Had I fixed the labels and the weighting in the same retrain, I could not have said which change did what.

### v3: pay the rare class

**One change: weight the death terms 64x, roughly the inverse of the per-frame death rate. v3 is the first head that predicts death in dreams as well as on data.**

```python
def continue_bce(logits, targets, death_weight: float = 1.0):
    bce = (jnp.maximum(logits, 0.0) - logits * targets
           + jnp.log1p(jnp.exp(-jnp.abs(logits))))
    return bce * (1.0 + (death_weight - 1.0) * (1.0 - targets))
```

Focal loss and oversampling the windows that end in death are the alternatives. Weighting is one line and leaves the meaning of the target intact, so it went first. The v3 buffer is byte-identical to v2's, so the comparison isolates the weight.

| $P(\text{continue})$ | v1 | v2 | v3 |
|---|---|---|---|
| on the real death frame | 0.05 (artifact) | 0.95 | **0.14** |
| one frame before death | 1.00 | 0.99 | **0.72** |
| after one prior step | 0.99 | 0.96 | **0.20** |

In the bottom row, v3 is the first head that still reads "dead" after a prior step. The middle row shows something neither earlier head did: a drop before the death frame itself.

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-death-dream-v3-strip.png" alt="Two rows of ten DOOM frames, real on top and dreamed below, with P(continue) under each. In the dream the value drops as the fireball closes and reads 0.00 at impact.">
 <figcaption><strong>Figure 6.</strong> Same probe on v3. Along the dream, $P$ drops as the fireball closes and reads 0.00 at impact.</figcaption>
</figure>

---

## 3. Training the agent, and scoring it

**Result of this block: an online loop takes the actor to 1.5x random survival. The loop's own three-episode evaluations could not show that, and it took 100 episodes to measure.**

### The loop

**Stage one trains the world model on random play. Stage two adds the actor, which collects one real episode per round and trains only in imagination.**

Parts 1 and 2 trained on a fixed dataset. Here the actor's own episodes feed back into the data. Each round collects an episode with the current actor, trains the world model on the buffer, and trains the actor in imagination.

<figure class="wm-fig wm-diagram">
 <img src="/images/world-models/diagrams/online-loop.svg" alt="A cycle: the environment sends one episode per round to the replay buffer, the buffer trains the world model, the world model trains the actor in imagination, and the actor collects the next episode.">
 <figcaption><strong>Figure 7.</strong> One round. Reality enters through the one episode the actor collects. The actor trains only on the dashed side, inside the model.</figcaption>
</figure>

Stage one is 1000 random episodes and 200 rounds of world-model updates, about an hour on a rented 4090. Stage two runs 300 rounds. By the end the buffer holds 95k frames, about a quarter of them collected by the learning actor.

### Why three episodes are too few

**Survival in `take_cover` is heavy-tailed, because deaths cluster at fireball volleys, and three episodes can all land in one mode.**

The loop's built-in evaluation runs three episodes every 25 rounds. Over one stretch it read 99, 47, 49, 35, 48 and 100, and by the halfway point I had told myself three different stories about the actor, each overturned by the next point. Episode length here is set mostly by when the first unlucky volley arrives, and three draws of that say very little about the mean.

So I made 30 episodes the minimum for anything quotable, wrote the rule into my notes, and walked into the same trap one experiment later. An actor retrained from scratch on the final world model came back from its built-in evaluation at 41 ± 8 steps. I read the tiny spread as a collapsed policy dying at the same moment every time, and had a story about it exploiting the dreams half written when the 30-episode rerun came back at 86 ± 44. All three episodes had landed in the same mode of the distribution.

From here on every number is 100 episodes, and ± is the spread across episodes. Measured that way, the actor survives 106 steps against 70 for random play, a ratio of 1.5 with about five standard errors between them. Its policy was updated only on imagined rollouts, where the sole signal about dying is the continue head from block 2.

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-play-loop-actor.gif" alt="Animated 64x64 grayscale DOOM. The trained policy strafes in one direction, reverses as a fireball commits to a lane, and survives 73 steps.">
 <figcaption><strong>Figure 8.</strong> The loop's actor on its median episode, 73 steps. It holds a strafe and reverses when a fireball heads for its lane.</figcaption>
</figure>

---

## 4. Making dreams harder to exploit

**Result of this block: on the same 95k frames, noisier dreams and more actor training per frame each add about 45 steps, and together they take the agent to 2.7x random.**

### Temperature

**Sample dreams with the prior's spread inflated, so the actor cannot rely on the model's errors. The 2018 paper's value did nothing here, and a larger one gave the biggest single gain in the project.**

An actor trained in imagination is optimized against the world model's version of the game. Wherever the model is wrong in the actor's favor, a fireball that fizzles or a dodge that works too easily, the gradient pushes the policy toward that error, and the gain disappears in the real game. Ha and Schmidhuber reported this on `take_cover`, with a controller that scored well in its dreams and barely beat random in reality. Their fix was temperature, a multiplier $\tau$ on the prior's $\sigma$ when sampling imagined latents. The extra noise makes any one erroneous trajectory harder to repeat, so a policy that depends on it scores worse in the dream as well. In the imagination loop it is one factor:

```python
# tau touches only these dream samples; posterior filtering
# and the filmstrip's rollout_prior stay at temperature 1
z = mu_p + sig_p * config.imagination_temperature * pn
```

The paper used $\tau = 1.15$. I tried it cheaply first, training a fresh actor against the frozen final world model at each value and scoring it over 30 episodes: 86 steps at 1.0, 76 at 1.15 and 110 at 1.30. My guess is that an RSSM trained with KL balancing has a narrower prior than a 2018 MDN-RNN, so the same multiplier adds less noise. I did not test that. If you port the trick, sweep the value.

The full loop at 1.30 reached 151 steps against the baseline's 106, and the three-episode evaluations during that run never separated from the baseline's. One confound stays open: noisier dreams also mean more exploration, and this experiment cannot tell the two effects apart.

### More training per frame, then both

**Four times the actor updates on the same frames, with the resets and slow critic that a high replay ratio needs, matches temperature's gain by a different route. The two together reach 187 steps.**

The blunt lever for sample efficiency is more gradient steps per real frame. Push it far and the actor overfits its early data and stays stuck there, which the literature calls primacy bias, so a high replay ratio usually ships with treatments (Nikishin et al., 2022; D'Oro et al., 2023): periodic resets of the actor and critic, and a slow copy of the critic to bootstrap returns from. I ran the package as one change, 400 actor updates per round instead of 100 with two resets along the way, and it reached 153 steps, level with temperature. The resets show up even in the three-episode evaluations, as a collapse at each reset followed within about 25 rounds by numbers the baseline never posted.

| run | steps survived | vs random |
|---|---|---|
| random | 70 ± 32 | 1.0x |
| plain loop | 106 ± 66 | 1.5x |
| temperature 1.30 | 151 ± 99 | 2.2x |
| replay bundle | 153 ± 77 | 2.2x |
| both | **187 ± 128** | **2.7x** |

The two gains nearly add up, +45 and +47 alone and +81 together, which suggests they address mostly different failures. The spread is also the widest yet. In the footage the composed policy commits to a direction on each volley, so a run of good choices produces the 367-step episode at the top of the post and a bad one early gives the 35-step episode in Figure 9.

<figure class="wm-fig wm-pixel">
 <img src="/images/world-models/doom-agent-worst.gif" alt="Animated 64x64 grayscale DOOM. The policy commits to a strafe, a fireball arrives in that lane, and the episode ends after 35 steps.">
 <figcaption><strong>Figure 9.</strong> Same policy, worst of ten episodes: 35 steps.</figcaption>
</figure>

## Where this breaks

- **One seed per row.** Every result is one training run, evaluated over 100 episodes. The 2.7x gap is large enough to believe. The two steps between temperature and the replay bundle are inside the noise.
- **Each lever's mechanism is unverified.** The replay bundle is three changes in one row, and temperature is confounded with exploration.
- **The continue head does not track threat.** On ordinary frames its output barely correlates with fireball size, and it drops only around the death frame. The head that went into the loop was also trained on a smaller buffer, which oversamples deaths, so it errs the other way and calls death on about a fifth of calm frames. How much of the actor's gain runs through the head, rather than through the dynamics alone, these experiments cannot say. Backing the 64x weight off is the next experiment, and I have not run it.
- **One pre-declared check failed.** I required v3 to score dreamed impact flashes well below 0.9, and the median came in at 0.97. The check was weak, since the agent survives some flashes that size, but I wrote the rule before the run and the verdict stands.
- **Reconstruction is still the objective.** In `take_cover` the fireballs are the brightest thing in the image, so reconstruction happens to spend its capacity on them. On a scene where the objects that matter are dim, the same objective would spend it elsewhere.

<details>
<summary>Try it yourself: commands and four exercises. Open if you want to run it.</summary>

Linux with an NVIDIA GPU; 12 GB is plenty. On Windows use WSL2, since JAX has no native Windows CUDA path. The v1 to v3 pretrains ran on an RTX 5070 Ti under WSL2, and the loop runs on a rented RTX 4090.

```
uv sync --extra doom --extra cuda

uv run train-doom --wm-only --seed-episodes 1000 \
    --episodes-per-round 0 --rounds 200 --run-name pretrain

uv run train-doom --resume --run-name pretrain --rounds 500 \
    --imagination-temperature 1.30 --ac-updates 400 \
    --ac-reset-every 100 --critic-ema 0.98 --final-ac-steps 5000

uv run python scripts/doom_eval.py --run runs/doom/pretrain --episodes 100
uv run python scripts/doom_diag.py --run runs/doom/pretrain
```

The pretrain is about an hour on a 4090 and the composed loop about 3.5 more. Expect about 187 ± 128 survival steps against a random arm at 70 ± 32. The random arm reproduces exactly; the actor arm will land inside the reported spread rather than on the mean. To run one ablation at a time, copy the run directory first, since `--resume` advances counters in place.

1. **Break the continue head on purpose.** Set `continue_death_weight` to 1.0, rerun the pretrain, and run `doom_diag.py` on both. Predict the loss curve and the death-frame $P(\text{continue})$ separately before you look.
2. **Reproduce the small-n trap.** Run `doom_eval.py --episodes 3` at five seeds on the same finished run, then once at 100. Mine spanned a factor of three.
3. **Sweep temperature.** Resume a finished run with `--final-ac-steps 20000` at temperatures 1.0, 1.15 and 1.30, and score the fresh actor at 30 episodes. Predict the ordering first.
4. **Split the bundle.** Run `--ac-updates 400` alone, then add `--ac-reset-every 100`, then `--critic-ema 0.98`. This is the ablation I could not afford.

</details>

## Imagination is test time

The v1 head and part 1's dead VAE failed in the same way. In part 1 the probe read the encoder's mean while the decoder received a noisy sample, so a collapsed model scored 0.74 on a channel it could not use. Here the loss was computed on posterior states while the actor read the head on prior ones. In both cases the number was correct for the place it was measured, and the model was being used somewhere else. What caught each was a second, cheap measurement taken where the model is used: the KL next to the probe score in part 1, and the one-step prior test next to the loss curve here.

The one-step test needs nothing from DOOM except a model with a posterior, a prior and a head to read, so it should carry over to other world models. Mine is in `scripts/doom_diag.py`.
