# MAPF-GNN browser demo

A fleet of robots on a grid, each with its own goal, each seeing only a 5×5
patch around itself. A small graph neural network lets each robot exchange one
feature vector with nearby robots before choosing a move. The policy was trained
by imitating an optimal centralised planner on ten robots; it runs on hundreds,
client-side.

**Status: shipped and running.** The TypeScript path is complete and pinned to
PyTorch by a parity test. This document is both the record of how it works and
the reference for extending it.

**Source project:** `C:\Users\victo\workspace_code\MAPF-GNN`. `docs/findings.md`
there has the results the demo conveys; `grid/env_vectorized.py` is the
environment this ports.

---

## 1. What it shows

Three things, all of them things a reader can do rather than read.

1. **Communication matters, but only at scale.** Up to ~40 robots a
   non-communicating policy does just as well. At 100 it collapses (α 0.21)
   while the communicating one holds (α 0.60).
2. **A deterministic policy deadlocks forever.** 98% of failures are a single
   joint state repeating. Taking the argmax in a deterministic environment means
   two robots wanting the same cell collide, revert, and repeat eternally.
3. **Sampling fixes it, and there is a real optimum.** Drawing from the policy
   raises α at 40 robots from 0.015 to 0.980. Uniform random scores 0.000, so
   the policy does the work and the noise only breaks symmetry. At temperature 5
   it collapses again.

α throughout means: the fraction of episodes where **every** robot reaches its
goal.

The demo runs two panels on one instance — identical obstacles, starts, goals
*and* per-robot random streams — differing only in whether the policy may use
the graph. That is what makes (1) visible instead of asserted.

## 2. Layout

```
src/components/mapf/
  index.ts        public exports
  MapfDemo.ts     lifecycle, DOM, controls, RAF loop, episode bookkeeping
  types.ts        manifest and model types; mirrors the exporter
  weights.ts      fetch + parse; parsing is separate so node can test it
  env.ts          the grid world
  policy.ts       the forward pass
  render2d.ts     Canvas2D renderer
  test/
    parity.ts     checked against PyTorch      (npm run test:mapf)
    bench.ts      fleet size vs. steps/second

public/assets/models/mapf/
  config.json     manifest: shapes, offsets, defaults
  comm.bin        GCN K=3      55,829 float32   218 KB
  nocomm.bin      baseline     30,933 float32   121 KB
  fixture.json    ground-truth rollout
```

Mounted from `initializeEmbeddedDemos()` in `src/pages/BlogPost.ts`, like the
other demos. A post carries it with:

```html
<div id="mapf-demo" class="my-8 not-prose"></div>
```

## 3. The models

Two, both trained on 10 robots on a 20×20 board.

| Panel | Checkpoint | Architecture | Parameters |
|---|---|---|---|
| `nocomm` | `baseline_a10b20` | CNN → encoder → head | 30,933 |
| `comm` | `gnn_k3_a10b20` | CNN → encoder → graph filter K=3 → head | 55,829 |

**Not the message-passing variant.** Its self-term read a scrambled global
mixture of every robot's features rather than its own, so it was neither
permutation-equivariant nor decentralised — with an *empty* communication graph,
one robot's observation still moved another's logits. A demo about
decentralised control cannot run that. See `findings.md` §5; the exporter
refuses it.

### Weights

float32, little-endian, row-major, concatenated; `config.json` gives each
tensor's offset and shape. Two conventions, both applied at export so the
browser never has to reproduce PyTorch's storage:

- **BatchNorm is folded** into the preceding convolution. The demo is eval-only.
- **Matrices are pre-transposed** so every use is a plain `x @ W`.

float32 rather than float16 is deliberate: halving 340 KB is not worth risking
the parity tolerance, and folding BatchNorm can inflate a convolution's dynamic
range by whatever `1/sqrt(running_var + eps)` happens to be.

| Tensor | Shape | Notes |
|---|---|---|
| `conv1.w`, `conv1.b` | (16,2,3,3), (16) | stride 1, pad 1, ReLU |
| `conv2.w`, `conv2.b` | (16,16,3,3), (16) | |
| `conv3.w`, `conv3.b` | (16,16,3,3), (16) | |
| `enc.w`, `enc.b` | (400,64), (64) | ReLU |
| `gnn.w` | (192,128) | `comm` only; K·64 → 128, no bias |
| `act.w`, `act.b` | (128,5) or (64,5), (5) | |

### Forward pass

Per robot, from its 2×5×5 patch:

```
x = relu(conv3(relu(conv2(relu(conv1(x))))))    # 2 -> 16 -> 16 -> 16, 5x5
h = relu(enc(flatten(x)))                       # 400 -> 64
```

Flattening is channel-major: index `c*25 + y*5 + x`.

Then, for `comm` only, over the fleet with `H` as `N × 64`:

```
Â  = A + I
d  = rowsum(Â)
Ã  = Â * d^-½ (rows) * d^-½ (cols)

Z  = concat_k( (Ãᵀ)^k · H )   for k = 0..K-1     # N × 192, K = 3
H' = relu( Z · gnn.w )                           # N × 128
```

**`Ãᵀ`, not `Ã`.** `A` is not symmetric — nearest-four is a per-row rule, so `j`
can be among `i`'s four nearest while `i` is not among `j`'s — and the Python
computes `(F, N) @ (N, N)`, which aggregates over the transpose. A robot pulls
from every robot that considers *it* a neighbour. Getting this backwards still
produces a plausible-looking demo.

Then both panels:

```
logits = source · act.w + act.b        # source is H' (comm) or h (nocomm)
```

### Action selection

```
T = 0   → argmax
T > 0   → sample from softmax(logits / T)
```

Actions: `0 idle, 1 right (+x), 2 up (+y), 3 left (−x), 4 down (−y)`.

Sampling uses a per-robot seeded stream (`FleetRng`), so a seed replays
identically and both panels get the same draws.

## 4. The environment

Ported from `grid/env_vectorized.py`. **Order matters**; getting it wrong
produces a demo that behaves unlike every number in the article.

### State

- `posX`, `posY`, `goalX`, `goalY` — `Int32Array`, one entry per robot
- `cells` — `0` free, `1` robot, `2` obstacle, indexed `y * board + x`
- `obstacles` — kept separately, as the Python does

### Reset

The board holds **obstacles only**. Robots are not stamped onto it until the
first step, so the very first field of view shows no other robots. A quirk of
the reference implementation, reproduced.

### Step, in this exact order

1. Save `prev`
2. `new = pos + delta[action]`
3. **Obstacles before bounds.** A move off the board is not in the obstacle set,
   so test `inBounds && obstacle[new]` and revert those robots
4. **Clamp** to `[0, board-1]`
5. **Collisions.** Every robot sharing a cell reverts, including groups of three
   or more. Reverting is *not* re-checked for collisions it creates
6. **Board update.** Clear all `prev` cells first, *then* stamp all current
   positions — so a robot moving into a just-vacated cell survives the clear
7. Recompute the graph and the field of view

### Graph

```
d = pairwise euclidean distance, zeroed at or beyond sensing_range (4)
threshold_i = fourth smallest strictly-positive distance in row i  (+inf if fewer)
A[i][j] = 1  iff  0 < d[i][j] <= threshold_i
```

Comparing against the fourth-smallest rather than taking a fixed top-4 is what
makes a robot with fewer than four neighbours keep all of them, and ties at the
fourth distance all survive. The result is directed.

### Field of view

Two channels, 5×5, `pad = 3`.

**Channel 0 — local map.** The window is `board[posY-2+(4-r)][posX-2+c]` for
`r, c` in `0..4`, zero outside the board. Note the `(4-r)`: the Python
row-flips each patch.

**Channel 1 — goal bearing.** The goal is almost never inside a 5×5 patch, so it
is projected onto the patch edge — the robot is told a direction, not a
position. With `dx = goalX - posX`, `dy = goalY - posY`:

```
col = (-2 < dx < 2) ? dx + 2 : (dx <= -2 ? 0 : 4)
row = (-2 <= dy < 2) ? 2 - dy : (dy <= -2 ? 4 : 0)
fov[robot][1][row][col] = 3.0
```

The row test uses `>=` where the column test uses `>`. The asymmetry is in the
original. The marker value is `3.0`, not `1.0`.

## 5. Measured performance

Both panels, one policy step each, node on the development machine:

| robots | board | ms/step | steps/s |
|---|---|---|---|
| 10 | 20 | 3.0 | 329 |
| 40 | 40 | 11.7 | 86 |
| 100 | 63 | 29.3 | 34 |
| 200 | 89 | 75.4 | 13 |
| 500 | 141 | 181.3 | 5.5 |

The demo simulates at 12 steps/s and interpolates robot positions between cells,
so the frame rate is independent of the fleet size until the budget runs out.

**This is why there is no WebGPU path.** The proposal assumed plain JavaScript
would cap out around 40 robots and that the GPU was needed to reach the
hundreds. It reaches 100 — the fleet size where the whole claim lives — at 34
steps/s, and 200 comfortably. The fleet slider stops at 200. WebGPU would only
buy 350+, which is past where the demo has anything left to say.

## 6. Controls

| Control | Range | Purpose |
|---|---|---|
| Fleet size | 10 – 200 | Identical at 20, diverges past 100 |
| Temperature | 0 – 5 | 0 freezes the fleet permanently; ~3 resolves it; 5 wanders |
| Pause / New instance | — | |
| Under the hood | — | Architecture, parameter count, what a robot sees |

Per panel: robots home, step against horizon, α accumulated over episodes, and
mean neighbours for the communicating panel.

Defaults **must** match the article: `T = 3`, horizon `3 × board width`. If the
demo and the post disagree, the reader believes the demo.

## 7. Tests

`npm run test:mapf` — parity against PyTorch, from the same checkpoint the
browser loads.

| Check | Guards | Current |
|---|---|---|
| Field of view matches the fixture | Window, flip, goal projection | exact |
| Adjacency matches | The nearest-four rule | exact |
| Logits match within 1e-3 | The forward pass | 5e-6 |
| Actions match at T=0 | Decoding | 96/96, both models |
| Positions match, 9 steps | Step order, collisions | exact |
| The two policies disagree | That the fixture can detect a missing graph | 12/96 |

That last row matters. On a sparse board both policies choose identical actions,
so a fixture drawn from one would pass for an implementation that ignored
communication entirely. The fixture instance is deliberately crowded — 12 robots
on 12×12.

`npx tsx src/components/mapf/test/bench.ts` reproduces the table in §5.

## 8. Regenerating the assets

From the MAPF-GNN repository:

```
python scripts/export_web_model.py --out ../retamalvictor.github.io/public/assets/models/mapf
```

That writes all four files. Re-run `npm run test:mapf` afterwards — the fixture
is regenerated too, so a mismatch after an export means the port disagrees with
the new checkpoint, not that the fixture went stale.

## 9. Gotchas

- Robots are **not** on the board at reset — only obstacles.
- Obstacles are checked **before** the boundary clamp.
- Collision reverts are **not** re-checked.
- The FOV window is **row-flipped**.
- The goal marker is **3.0**, at `[row][col]`, and the row test's `>=` versus the
  column test's `>` is deliberate.
- The adjacency is **directed**, and aggregation is over its **transpose**.
- Degree-zero robots must contribute 0 in the normalisation, not NaN. (With
  `A + I` the degree is never zero, but a variant without the self-loop needs
  the guard.)
- Sampling temperature has a real optimum. Do not "improve" it to pure argmax:
  that is the deadlock the demo exists to show.
- Do not swap in the message-passing checkpoint. See §3.

## 10. If you did want the GPU path

Not needed for anything the demo currently claims, but the shape of it: six
compute shaders — `fov`, `adjacency`, `encoder`, `gnn`, `action`, `step` —
with state resident in GPU buffers across frames and only the render pass
reading back. Under 2 MB of buffers at 500 robots, the adjacency matrix being
almost all of it. Collision resolution is the one awkward kernel: reverting
every robot that shares a cell is a scatter with contention, done as two passes
over an atomically-incremented occupancy grid.

Keep `env.ts` and `policy.ts` as the reference and check the GPU against them
step-for-step at temperature 0 on the same seed.

## 11. Out of scope

Training in the browser, editing the architecture, custom map upload. The demo
ships two checkpoints and shows what they do.
