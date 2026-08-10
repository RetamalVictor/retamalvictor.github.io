# MAPF-GNN browser demo — build spec

Self-contained spec for building an interactive multi-robot path planning demo
that runs entirely client-side. Written so an agent with no prior context can
execute it.

**Source project:** `C:\Users\victo\workspace_code\MAPF-GNN` (branch
`perf/gpu-batched-rollouts`). Read `docs/findings.md` there for the results the
demo is meant to convey, and `grid/env_vectorized.py` for the environment being
ported — it is the reference implementation and is pinned to the original by an
equivalence test.

---

## 1. What this is

A fleet of robots on a grid, each with its own goal, each seeing only a 5×5
patch around itself. A small graph neural network lets each robot exchange one
feature vector with nearby robots before choosing a move. The policy was trained
by imitating an optimal centralised planner on ten robots; it runs on hundreds.

The demo exists to make three things tangible:

1. **Communication matters, but only at scale.** Up to ~40 robots a
   non-communicating policy does just as well. At 100 it collapses (α 0.21)
   while the communicating one holds (α 0.60–0.70).
2. **A deterministic policy deadlocks forever.** 98% of failures are a single
   joint state repeating. Taking the argmax in a deterministic environment means
   two robots wanting the same cell collide, revert, and repeat eternally.
3. **Sampling fixes it.** Drawing from the policy instead of taking its max
   raises α at 40 robots from 0.015 to 0.980. Uniform random scores 0.000, so
   the policy does the work and the noise only breaks symmetry.

α ("alpha") throughout means: fraction of episodes where **every** robot reaches
its goal.

## 2. Site conventions to follow

This repo is Vite + TypeScript, no framework. Demos are classes that take a
`containerId` and are lazy-imported.

**Study `src/components/ternary/TernaryLMDemo.ts` first** — it is the closest
precedent: hand-written inference, a weights fetch, controls, and an "under the
hood" panel.

Mount by adding to `initializeEmbeddedDemos()` in `src/pages/BlogPost.ts`:

```ts
const mapf = document.getElementById('mapf-demo');
if (mapf) {
  import('../components/mapf').then(({ MapfDemo }) => {
    new MapfDemo({
      containerId: 'mapf-demo',
      modelPath: '/assets/models/mapf',
      defaultAgents: 40,
      defaultTemperature: 3.0,
    });
  }).catch(err => console.error('Failed to load MAPF demo:', err));
}
```

The blog post then contains `<div id="mapf-demo" class="my-8 not-prose"></div>`.

Weights live in `public/assets/models/mapf/`, alongside the existing
`depth/`, `pelvis/`, and `transformer_new/`.

## 3. The model

64,117 parameters. 125 KB at fp16. Six weight groups, in this order in
`weights.bin` (fp16, little-endian):

| # | Tensor | Shape | Notes |
|---|---|---|---|
| 1 | conv1 w, b | (16,2,3,3), (16) | stride 1, pad 1 |
| 2 | conv2 w, b | (16,16,3,3), (16) | |
| 3 | conv3 w, b | (16,16,3,3), (16) | |
| 4 | encoder w, b | (64,400), (64) | |
| 5 | `W`, `W2` | (64,3,128), (64,128) | graph filter, K=3; `W2` is the self term |
| 6 | policy w, b | (5,128), (5) | |

**BatchNorm is folded into the preceding convolution at export.** The demo runs
in eval mode only, so it does not appear at runtime.

### Forward pass

Per robot, given its 2×5×5 patch:

```
x = relu(conv1(x))            # 2  -> 16, 3x3, pad 1   → 16×5×5
x = relu(conv2(x))            # 16 -> 16               → 16×5×5
x = relu(conv3(x))            # 16 -> 16               → 16×5×5
h = relu(encoder(flatten(x))) # 400 -> 64
```

Then over the fleet, with `H` the 64-dim features stacked as `N × 64` and `A`
the adjacency:

```
Â  = A + I
d  = rowsum(Â)
Ã  = Â * d^-½ (rows) * d^-½ (cols)      # symmetric normalisation
                                        # d = 0 → contribute 0, not NaN

Z  = concat_k( Ã^k · H )  for k = 0..K-1     # N × (K·64), K = 3
H' = relu( Z · W_flat  +  H · W2 )           # N × 128
                                             # W_flat is W reshaped to (K·64, 128)
logits = H' · policy_wᵀ + policy_b           # N × 5
```

`W2` is present only in the message-passing variant, which is the shipped
checkpoint. Weights are shared across robots — the fleet size is never baked in,
which is why the same weights run any N.

### Action selection

```
T = 0   → argmax(logits)
T > 0   → sample from softmax(logits / T)
```

Actions are `0 idle, 1 right (+x), 2 up (+y), 3 left (−x), 4 down (−y)`.

Sampling needs **per-robot reproducible randomness** — a small PCG or xorshift
seeded once and advanced each step — so a seed replays identically and the
deadlock preset is stable.

## 4. The environment

Port from `grid/env_vectorized.py`. **Order matters**; getting it wrong produces
a demo that behaves unlike every number in the article.

### State

- `positions: i32[N][2]` (x, y)
- `goals: i32[N][2]`
- `board: f32[H][W]` — `0` free, `1` robot, `2` obstacle

### Reset

Board holds **obstacles only**. Robots are *not* stamped onto the board until
the first step — so the very first field of view shows no other robots. This is
a quirk of the reference implementation, and it must be reproduced.

### Step, in this exact order

1. Save `prev = positions`
2. `new = positions + delta[action]`
3. **Obstacles before bounds.** A move off the board is simply not in the
   obstacle set, so test `in_bounds && obstacle[new]` and revert those robots to
   `prev`
4. **Clamp** to `[0, board-1]`
5. **Collisions.** Every robot sharing a cell reverts, including groups of three
   or more. Reverting is *not* re-checked for new collisions
6. **Board update.** Clear all `prev` cells to 0 *first*, then stamp all current
   positions to 1 — so a robot moving into a cell another just vacated survives
   the clear
7. Recompute the graph

### Graph

```
d = pairwise euclidean distance
d[d >= sensing_range] = 0                 # sensing_range = 4
# keep only the four nearest non-zero neighbours per robot:
masked  = where(d > 0, d, +inf)
thresh  = sort(masked, axis=1)[:, 3]      # +inf padding means fewer than four
d[d > thresh] = 0                         #   neighbours drops nothing
A = (d != 0)
```

The `+inf` padding is load-bearing: a robot with fewer than four neighbours must
keep all of them.

### Field of view

Two channels, 5×5, `pad = 3`.

**Channel 0 — local map.** Pad the board by 3 on each side, then take rows
`y+1 … y+5` and columns `x+1 … x+5`, and **flip the rows** (the reference uses
`np.flip(..., axis=0)`).

**Channel 1 — goal bearing.** The goal is almost never inside a 5×5 patch, so it
is projected onto the patch edge. With `gp = goal + pad`, `px = x + pad`,
`py = y + pad`:

```
inside_x = (gp.x < px + pad - 1) && (gp.x > px - pad + 1)
goal_x   = inside_x ? (gp.x - px) + pad - 1
                    : (gp.x <= px - pad + 1 ? 0 : 1 + pad)

inside_y = (gp.y < py + pad - 1) && (gp.y >= py - pad + 1)
goal_y   = inside_y ? (py - gp.y) + pad - 1
                    : (gp.y <= py - pad + 1 ? 1 + pad : 0)

fov[robot][1][goal_y][goal_x] = 3.0        # note the index order: y then x
```

Everything else in channel 1 is zero. The value is `3.0`, not `1.0`.

## 5. Architecture

```
src/components/mapf/
  index.ts            export { MapfDemo }
  MapfDemo.ts         lifecycle, DOM, controls, RAF loop
  types.ts            Config, WorldState, Weights
  weights.ts          fetch + parse weights.bin, fp16 → f32
  env.ts              TS reference environment
  policy.ts           TS reference forward pass
  render2d.ts         Canvas2D renderer (fallback path)
  gpu/
    device.ts         adapter/device init, feature detection
    buffers.ts        allocation and layout
    pipeline.ts       shader modules, bind groups, dispatch order
    render.ts         instanced WebGPU renderer
    shaders/
      fov.wgsl        board + positions + goals → N×2×5×5
      adjacency.wgsl  distances, threshold, nearest-4
      encoder.wgsl    conv ×3 + 400→64
      gnn.wgsl        normalise, K-hop, W and W2
      action.wgsl     128→5, temperature sample
      step.wgsl       dynamics
```

Per frame, both backends run the same six stages:

```
adjacency → fov → encoder → gnn → action → step
```

State stays in GPU buffers across frames; only the render pass reads back.

### Buffer sizes at N = 500

| Buffer | Size |
|---|---|
| positions, prev, goals | 12 KB |
| board (64²) | 16 KB |
| adjacency (N×N f32) | 1 MB |
| fov (N×50) | 100 KB |
| features (N×64, N×128) | 384 KB |
| logits, actions, rng | 14 KB |
| weights | 250 KB |

Under 2 MB. No memory pressure.

### Collision resolution in WGSL

The revert-everyone-sharing-a-cell rule is a scatter with contention. Two-pass:
atomically increment an occupancy grid keyed by `y * W + x`, then each robot
reads its own cell's count and reverts if it exceeds one.

## 6. Controls

| Control | Range | Purpose |
|---|---|---|
| Communication on/off | toggle | Two panels, same instance, one with the graph and one without |
| Fleet size | 10 – 500 | Identical at 20, diverges past 100 |
| Temperature | 0 – 5 | **The key interaction.** 0 freezes the fleet permanently; ~3 resolves it; 5 makes it wander uselessly |
| Deadlock preset | button | Loads a known-jamming instance |
| Stats | — | robots home, step, mean neighbours, fps |

Defaults **must** match the article: `T = 3`, horizon `3 × board width`. If the
demo and the post disagree, the reader believes the demo.

## 7. Phases

**Phase 1 — TypeScript reference (~1 day).** Export script and assets; weights
loader; `env.ts`; `policy.ts`; parity test; Canvas2D renderer; the three
controls.
*Done when:* 40 robots at 30 fps, logits match the fixture within 1e-3, and the
temperature slider visibly freezes and unfreezes the fleet.

**Phase 2 — WebGPU (~2 days).** Device and buffers; the six shaders; parity
against the TS path; instanced rendering.
*Done when:* 500 robots at 60 fps, and GPU matches TS step-for-step at T=0 on
the same seed.

**Phase 3 — polish (~½ day).** Deadlock preset, under-the-hood panel,
`prefers-reduced-motion`, mobile sizing.

Ship after Phase 1 if time is short — a 40-robot demo with a temperature slider
already carries the strongest finding.

## 8. Assets to request from the source repo

Three files into `public/assets/models/mapf/`. They are the entire coupling
between the two projects.

- **`weights.bin`** — the six groups above, fp16, BatchNorm folded
- **`config.json`** — channels, K, sensing range, FOV size, pad, action
  encoding, board defaults, recommended temperature
- **`fixture.json`** — one `(observation, adjacency) → logits` triple, plus a
  five-step rollout of positions and actions from a fixed seed

## 9. Tests

| Test | Guards |
|---|---|
| Logits match `fixture.json` within 1e-3 | The forward pass |
| Five-step rollout matches the fixture at T=0 | Environment dynamics |
| GPU matches TS, same seed, 100 steps | WGSL drift |
| Adjacency matches a brute-force implementation | The nearest-4 rule |
| Demo defaults equal the article's | The demo contradicting the post |

## 10. Gotchas

- Robots are **not** on the board at reset — only obstacles.
- Obstacles are checked **before** the boundary clamp.
- Collision reverts are **not** re-checked.
- The FOV window is **row-flipped**.
- The goal marker value is **3.0**, and indexed `[goal_y][goal_x]`.
- Degree-zero robots must contribute 0 in the normalisation, not NaN.
- Fewer than four neighbours must keep all of them — hence the `+inf` padding.
- Sampling temperature has a real optimum: too little and the fleet deadlocks,
  too much and it wanders. Do not "improve" it to pure argmax.

## 11. Out of scope

Training in the browser, editing the architecture, custom map upload. The demo
ships one checkpoint and shows what it does.
