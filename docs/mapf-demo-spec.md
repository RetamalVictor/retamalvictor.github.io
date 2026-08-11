# MAPF-GNN browser demo

A hundred robots on a grid, each with its own goal, each seeing only a 5×5 patch
around itself and none of them holding a plan. A small graph neural network lets
each robot exchange one feature vector with every robot inside its radio range
before choosing a move. The policy was trained by imitating an optimal
centralised planner on ten robots; it runs on two hundred, client-side.

**Status: shipped and running**, as the *Robot Fleet* tab in the home page hero.
This document is both the record of how it works and the reference for changing
it.

**Source project:** `C:\Users\victo\workspace_code\MAPF-GNN`, branch
`feat/web-export`. `docs/findings.md` there has the results; `grid/env_vectorized.py`
is the environment this ports.

---

## 1. What it shows

One board, one policy. It used to race a non-communicating policy side by side,
and that comparison was retired: retrained on the same data the baseline goes
from α 0.21 to 0.79 at a hundred robots against 0.94 for the communicating one,
so most of the gap it displayed belonged to its training set rather than to
communication. A race that flattering to one side is not worth running, and one
board at twice the size shows the mesh far better than two did.

What is worth showing:

1. **A policy trained on ten robots drives two hundred.** The weights are shared
   and the fleet size appears nowhere in them.
2. **Every robot is nearly blind.** 5×5 of local map, plus a bearing to its goal
   projected onto the patch edge. No map, no plan, no identity.
3. **A deterministic policy deadlocks forever.** Temperature 0 freezes the fleet
   solid: shared weights make two robots in symmetric situations pick the same
   cell, collide, revert, and repeat. 98% of failures are one joint state
   repeating.

α throughout means: the fraction of episodes where **every** robot reaches its
goal.

## 2. Layout

```
src/components/mapf/
  index.ts        public exports
  MapfDemo.ts     lifecycle, DOM, controls, RAF loop, episode bookkeeping
  types.ts        manifest types (mirrors the exporter) and BoardRenderer
  weights.ts      fetch + parse; parsing is separate so node can test it
  env.ts          the grid world
  policy.ts       the forward pass
  render2d.ts     Canvas2D renderer
  render3d.ts     three.js renderer, lazily imported
  test/
    parity.ts     checked against PyTorch      (npm run test:mapf)
    bench.ts      fleet size vs. steps/second

public/assets/models/mapf/
  config.json     manifest: shapes, offsets, defaults, world settings
  comm.bin        GCN K=3      55,829 float32   218 KB
  nocomm.bin      baseline     30,933 float32   121 KB
  fixture.json    ground-truth rollout
```

Mounted from `initializeEmbeddedDemos()` in `src/pages/BlogPost.ts` for a post,
and from `DemoManager` for the hero. A post carries it with
`<div id="mapf-demo" class="my-8 not-prose"></div>`.

`nocomm.bin` is exported but never displayed. The parity fixture uses the
disagreement between the two policies to prove it is crowded enough to detect a
missing communication path; that check is worth more than the 121 KB.

## 3. The model

`gnn_k3_dense_r8` — GCN with K=3, 55,829 parameters, trained on 10 robots on a
20×20 board across **four obstacle densities** (2/5/10/15%), with a sensing
range of 8 and **no neighbour cap**.

Three things about that matter if you retrain or re-export:

- **No cap.** The graph joins a robot to *every* robot in range, not the four
  nearest. `D^-½(A+I)D^-½` is exactly what makes a varying degree safe to sum
  over, and the old cap discarded more than half the links at deployment scale.
  In the manifest, `max_neighbours: 0` means uncapped.
- **Range 8, not 4.** Only worth anything once the cap is gone; with the cap on,
  widening swaps near neighbours for far ones instead of adding any.
- **Four densities.** The demo has an obstacle slider. The previous model,
  trained at 2% only, scored exactly **0.000** past 40 robots at 10% obstacles —
  the slider was showing a cliff that belonged to the training set.

### Weights

float32, little-endian, row-major, concatenated; `config.json` gives each
tensor's offset and shape. Two conventions applied at export so the browser
never reproduces PyTorch's storage:

- **BatchNorm folded** into the preceding convolution (the demo is eval-only).
- **Matrices pre-transposed** so every use is a plain `x @ W`.

| Tensor | Shape | Notes |
|---|---|---|
| `conv1.w`, `conv1.b` | (16,2,3,3), (16) | stride 1, pad 1, ReLU |
| `conv2.w`, `conv2.b` | (16,16,3,3), (16) | |
| `conv3.w`, `conv3.b` | (16,16,3,3), (16) | |
| `enc.w`, `enc.b` | (400,64), (64) | ReLU |
| `gnn.w` | (192,128) | K·64 → 128, no bias |
| `act.w`, `act.b` | (128,5), (5) | |

### Forward pass

Per robot, from its 2×5×5 patch:

```
x = relu(conv3(relu(conv2(relu(conv1(x))))))    # 2 -> 16 -> 16 -> 16, 5x5
h = relu(enc(flatten(x)))                       # 400 -> 64
```

Flattening is channel-major: `c*25 + y*5 + x`.

Then over the fleet, with `H` as `N × 64`:

```
Â  = A + I
d  = rowsum(Â)
Ã  = Â * d^-½ (rows) * d^-½ (cols)
Z  = concat_k( (Ãᵀ)^k · H )   for k = 0..K-1     # N × 192, K = 3
H' = relu( Z · gnn.w )                           # N × 128
logits = H' · act.w + act.b
```

**`Ãᵀ`, not `Ã`.** `A` is not symmetric — nearest-k is a per-row rule and the
uncapped graph is symmetric only incidentally — and the Python computes
`(F, N) @ (N, N)`, which aggregates over the transpose. Getting this backwards
still produces a plausible-looking demo.

### Action selection

`T = 0` → argmax; `T > 0` → sample from `softmax(logits / T)`. Actions are
`0 idle, 1 right (+x), 2 up (+y), 3 left (−x), 4 down (−y)`. Sampling uses a
per-robot seeded stream so a seed replays identically.

## 4. The environment

Ported from `grid/env_vectorized.py`. **Order matters.**

**Reset.** The board holds obstacles only; robots are not stamped onto it until
the first step, so the first field of view shows no other robots.

**Step, in this exact order:**

1. Save `prev`
2. `new = pos + delta[action]`
3. **Obstacles before bounds** — a move off the board is not in the obstacle set
4. **Clamp** to `[0, board-1]`
5. **Collisions** — every robot sharing a cell reverts, including groups of three
   or more, and reverting is *not* re-checked
6. **Board update** — clear all `prev` cells first, *then* stamp current ones
7. Recompute the graph and the field of view

**Graph.** Pairwise distance, zeroed at or beyond `sensing_range`. If
`max_neighbours` is non-zero, keep only the k smallest strictly-positive
distances per row, comparing against the k-th smallest so ties all survive and a
robot with fewer than k neighbours keeps all of them. If it is zero, every robot
in range is a neighbour.

**Field of view.** Two channels, 5×5, `pad = 3`. Channel 0 is
`board[posY-2+(4-r)][posX-2+c]`, zero outside — note the `(4-r)`, the patch is
row-flipped. Channel 1 marks the goal projected onto the patch edge:

```
col = (-2 < dx < 2) ? dx + 2 : (dx <= -2 ? 0 : 4)
row = (-2 <= dy < 2) ? 2 - dy : (dy <= -2 ? 4 : 0)
fov[robot][1][row][col] = 3.0
```

The row test uses `>=` where the column test uses `>`; the asymmetry is in the
original. The marker is `3.0`, not `1.0`.

## 5. Rendering

Both views implement `BoardRenderer` from `types.ts` and are interchangeable.
**Nothing about the simulation changes between them** — the planner, the graph
and the policy are flat and stay flat; the third dimension is decoration.

```ts
draw(env, alpha, focus)   // focus < 0 draws every link, else only that robot's
```

`render2d.ts` caches everything that does not change per frame: the board,
obstacles, goals, palette and edge list are rebuilt only when their inputs do.
The simulation advances ~12 times a second while the display runs at 60.

`render3d.ts` imports three.js **lazily**, so a reader who never presses 3D never
downloads it. Obstacles extrude, robots are instanced spheres, goals are rings on
the floor, and the camera orbits by dragging. Four draw calls at 200 robots.

Two traps:

- Switching views **must replace the canvas element** — one that has handed out
  a 2D context will not return WebGL. That happens across an `await`, so the
  render loop has to be held off meanwhile.
- An `InstancedMesh` takes its per-instance tint from `instanceColor`, which
  three wires up on the first `setColorAt`. Setting `vertexColors: true` as well
  makes the shader look for a geometry colour attribute that does not exist, and
  every robot renders black.

## 6. Controls

| Control | Range | Purpose |
|---|---|---|
| Fleet size | 10 – 200 | Trained on 10 |
| Obstacles | 0 – 15% | Trained across 2–15% |
| Radio range | 2 – 14 | Trained at 8; the graph really does grow with it |
| Temperature | 0 – 5 | 0 freezes the fleet permanently; ~3 resolves it |
| Pause / New | — | |
| 2D / 3D | — | |
| All links / 1 robot | — | One robot's neighbourhood plus its radio disc |

Defaults come from the manifest and **must** match the article: `T = 3`, horizon
`3 × board width`, 100 robots, 2% obstacles.

## 7. Measured performance

One policy, one step, node on the development machine (the browser is
comparable):

| robots | board | ms/step | steps/s |
|---|---|---|---|
| 10 | 20 | 1.7 | 600 |
| 40 | 40 | 7.3 | 137 |
| 100 | 63 | 18.8 | 53 |
| 200 | 89 | 41.0 | 24 |

The demo simulates at up to 12 steps/s, backing off adaptively so the simulation
never eats more than ~40% of the wall clock, and interpolates robots between
cells. There is no WebGPU path and none is needed: 200 robots has a 2× margin.

## 8. Tests

`npm run test:mapf` — parity against PyTorch, from the same checkpoint the
browser loads.

| Check | Guards | Current |
|---|---|---|
| Field of view matches the fixture | Window, flip, goal projection | exact |
| Adjacency matches | The graph rule, capped or not | exact |
| Logits match within 1e-3 | The forward pass | 5e-6 |
| Actions match at T=0 | Decoding | 96/96 |
| Positions match, 9 steps | Step order, collisions | exact |
| The two policies disagree | That the fixture can detect a missing graph | 7/96 |

`npx tsx src/components/mapf/test/bench.ts` reproduces §7.

## 9. Regenerating the assets

From the MAPF-GNN repository:

```
python scripts/export_web_model.py --out ../retamalvictor.github.io/public/assets/models/mapf
```

Then re-run `npm run test:mapf`. The fixture is regenerated too, so a failure
means the port disagrees with the new checkpoint — not that the fixture aged.

## 10. Gotchas

- Robots are **not** on the board at reset — only obstacles.
- Obstacles are checked **before** the boundary clamp.
- Collision reverts are **not** re-checked.
- The FOV window is **row-flipped**; the goal marker is **3.0** at `[row][col]`.
- Aggregation is over the **transpose** of the normalised adjacency.
- `max_neighbours: 0` means uncapped, and it must be carried into every config
  that builds an environment. `VectorizedGraphEnv` defaults it to 4, so omitting
  it silently produces a capped graph for an uncapped model — this shipped once
  and the parity test caught it.
- Sampling temperature has a real optimum. Do not "improve" it to argmax; that
  is the deadlock the demo exists to show.

## 11. Out of scope

Training in the browser, editing the architecture, custom map upload.
