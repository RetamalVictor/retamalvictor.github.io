/**
 * The policy: a small CNN per robot, one round of graph aggregation across the
 * fleet, and a linear head.
 *
 * Every weight is shared across robots and every observation is local, so the
 * fleet size appears nowhere in this file except as a loop bound. That is the
 * whole reason a policy trained on ten robots runs on hundreds.
 *
 * Ported from `models/framework_gnn.py` and `models/networks/gnn.py`. Buffers
 * are allocated once and reused; at 500 robots this runs every frame.
 */

import { FleetRng } from './random';
import { NUM_ACTIONS } from './env';
import type { Model } from './types';

export class Policy {
    readonly model: Model;
    readonly numAgents: number;
    readonly side: number;

    private readonly convA: Float32Array;
    private readonly convB: Float32Array;
    private readonly encoded: Float32Array;
    private readonly hops: Float32Array;
    private readonly hopScratch: Float32Array;
    private readonly normalised: Float32Array;
    private readonly degree: Float32Array;
    private readonly node: Float32Array;
    readonly logits: Float32Array;
    readonly actions: Int32Array;
    private readonly probabilities: Float32Array;

    constructor(model: Model, numAgents: number, side = 5) {
        this.model = model;
        this.numAgents = numAgents;
        this.side = side;

        const plane = side * side;
        const widest = Math.max(...model.channels);
        this.convA = new Float32Array(widest * plane);
        this.convB = new Float32Array(widest * plane);

        const encoderDim = model.encoderDim;
        this.encoded = new Float32Array(numAgents * encoderDim);

        const k = model.filterNumber ?? 1;
        this.hops = new Float32Array(numAgents * k * encoderDim);
        this.hopScratch = new Float32Array(numAgents * encoderDim);
        this.normalised = new Float32Array(model.arch === 'gcn' ? numAgents * numAgents : 0);
        this.degree = new Float32Array(numAgents);
        this.node = new Float32Array(numAgents * (model.nodeDim ?? encoderDim));

        this.logits = new Float32Array(numAgents * NUM_ACTIONS);
        this.actions = new Int32Array(numAgents);
        this.probabilities = new Float32Array(NUM_ACTIONS);
    }

    /**
     * Run the fleet forward, filling `logits`.
     *
     * @param fov `N x 2 x side x side` observations
     * @param adjacency `N x N`, 1 where the column is a neighbour of the row
     */
    forward(fov: Float32Array, adjacency: Float32Array): Float32Array {
        this.encode(fov);
        if (this.model.arch === 'gcn') {
            this.aggregate(adjacency);
        }
        this.head();
        return this.logits;
    }

    /** Per-robot CNN and encoder — identical weights, independent robots. */
    private encode(fov: Float32Array): void {
        const { channels, encoderDim, tensors } = this.model;
        const plane = this.side * this.side;
        const flatSize = channels[channels.length - 1] * plane;

        const encW = tensors['enc.w'].data;
        const encB = tensors['enc.b'].data;

        for (let i = 0; i < this.numAgents; i++) {
            this.convA.set(fov.subarray(i * 2 * plane, (i + 1) * 2 * plane));

            let input = this.convA;
            let output = this.convB;
            for (let layer = 1; layer < channels.length; layer++) {
                this.conv3x3(
                    input,
                    output,
                    channels[layer - 1],
                    channels[layer],
                    tensors[`conv${layer}.w`].data,
                    tensors[`conv${layer}.b`].data
                );
                const swap = input;
                input = output;
                output = swap;
            }

            // Linear(flatSize -> encoderDim) then ReLU. `input` holds the last
            // convolution's output after the swap above.
            const base = i * encoderDim;
            for (let j = 0; j < encoderDim; j++) this.encoded[base + j] = encB[j];
            for (let a = 0; a < flatSize; a++) {
                const value = input[a];
                if (value === 0) continue;
                const row = a * encoderDim;
                for (let j = 0; j < encoderDim; j++) {
                    this.encoded[base + j] += value * encW[row + j];
                }
            }
            for (let j = 0; j < encoderDim; j++) {
                if (this.encoded[base + j] < 0) this.encoded[base + j] = 0;
            }
        }
    }

    /** 3x3 cross-correlation, stride 1, padding 1, followed by ReLU. */
    private conv3x3(
        input: Float32Array,
        output: Float32Array,
        inChannels: number,
        outChannels: number,
        weight: Float32Array,
        bias: Float32Array
    ): void {
        const side = this.side;
        const plane = side * side;

        for (let o = 0; o < outChannels; o++) {
            const outBase = o * plane;
            const b = bias[o];
            for (let y = 0; y < side; y++) {
                for (let x = 0; x < side; x++) {
                    let sum = b;
                    for (let c = 0; c < inChannels; c++) {
                        const inBase = c * plane;
                        const wBase = (o * inChannels + c) * 9;
                        for (let ky = 0; ky < 3; ky++) {
                            const yy = y + ky - 1;
                            if (yy < 0 || yy >= side) continue;
                            for (let kx = 0; kx < 3; kx++) {
                                const xx = x + kx - 1;
                                if (xx < 0 || xx >= side) continue;
                                sum += weight[wBase + ky * 3 + kx] * input[inBase + yy * side + xx];
                            }
                        }
                    }
                    output[outBase + y * side + x] = sum > 0 ? sum : 0;
                }
            }
        }
    }

    /**
     * K-hop aggregation over the communication graph.
     *
     * The normalisation is symmetric — D^-1/2 (A + I) D^-1/2 — but A itself is
     * not symmetric, because nearest-four is a per-row rule. The Python computes
     * `(F, N) @ (N, N)`, which in row-major node-feature terms is a multiply by
     * the *transpose*: a robot aggregates from every robot that considers it a
     * neighbour. Reproduced here rather than tidied into `A @ H`.
     */
    private aggregate(adjacency: Float32Array): void {
        const n = this.numAgents;
        const f = this.model.encoderDim;
        const k = this.model.filterNumber ?? 1;

        // Degrees of A + I. The self-loop guarantees a degree of at least one,
        // so the inverse square root is always finite here.
        for (let i = 0; i < n; i++) {
            let sum = 1;
            const row = i * n;
            for (let j = 0; j < n; j++) if (j !== i) sum += adjacency[row + j];
            this.degree[i] = 1 / Math.sqrt(sum);
        }

        for (let m = 0; m < n; m++) {
            const row = m * n;
            const sm = this.degree[m];
            for (let j = 0; j < n; j++) {
                const edge = j === m ? adjacency[row + j] + 1 : adjacency[row + j];
                this.normalised[row + j] = edge === 0 ? 0 : edge * sm * this.degree[j];
            }
        }

        // Hop 0 is the robot's own features, unmixed.
        for (let i = 0; i < n; i++) {
            const from = i * f;
            const to = i * k * f;
            for (let j = 0; j < f; j++) this.hops[to + j] = this.encoded[from + j];
        }

        for (let hop = 1; hop < k; hop++) {
            this.hopScratch.fill(0);
            for (let m = 0; m < n; m++) {
                const source = m * k * f + (hop - 1) * f;
                const row = m * n;
                for (let target = 0; target < n; target++) {
                    const weight = this.normalised[row + target];
                    if (weight === 0) continue;
                    const destination = target * f;
                    for (let j = 0; j < f; j++) {
                        this.hopScratch[destination + j] += weight * this.hops[source + j];
                    }
                }
            }
            for (let i = 0; i < n; i++) {
                const from = i * f;
                const to = i * k * f + hop * f;
                for (let j = 0; j < f; j++) this.hops[to + j] = this.hopScratch[from + j];
            }
        }

        // Linear over the concatenated hops, then ReLU.
        const gnnW = this.model.tensors['gnn.w'].data;
        const nodeDim = this.model.nodeDim as number;
        const width = k * f;
        this.node.fill(0);
        for (let i = 0; i < n; i++) {
            const zBase = i * width;
            const outBase = i * nodeDim;
            for (let a = 0; a < width; a++) {
                const value = this.hops[zBase + a];
                if (value === 0) continue;
                const wRow = a * nodeDim;
                for (let o = 0; o < nodeDim; o++) {
                    this.node[outBase + o] += value * gnnW[wRow + o];
                }
            }
            for (let o = 0; o < nodeDim; o++) {
                if (this.node[outBase + o] < 0) this.node[outBase + o] = 0;
            }
        }
    }

    /** Linear head onto five action logits. */
    private head(): void {
        const actW = this.model.tensors['act.w'].data;
        const actB = this.model.tensors['act.b'].data;
        const isGraph = this.model.arch === 'gcn';
        const source = isGraph ? this.node : this.encoded;
        const width = isGraph ? (this.model.nodeDim as number) : this.model.encoderDim;

        for (let i = 0; i < this.numAgents; i++) {
            const inBase = i * width;
            const outBase = i * NUM_ACTIONS;
            for (let a = 0; a < NUM_ACTIONS; a++) this.logits[outBase + a] = actB[a];
            for (let o = 0; o < width; o++) {
                const value = source[inBase + o];
                if (value === 0) continue;
                const row = o * NUM_ACTIONS;
                for (let a = 0; a < NUM_ACTIONS; a++) {
                    this.logits[outBase + a] += value * actW[row + a];
                }
            }
        }
    }

    /**
     * Choose actions from the current logits.
     *
     * At temperature zero this is the argmax, which is what the parity test
     * uses. Above zero it samples, which is not a detail: two robots running
     * identical weights on symmetric observations pick the same action, collide,
     * revert, and repeat forever. The noise breaks that symmetry. It does not
     * solve the problem — uniform random actions never finish an episode — and
     * too much of it drowns out the policy.
     */
    selectActions(temperature: number, rng: FleetRng | null): Int32Array {
        for (let i = 0; i < this.numAgents; i++) {
            const base = i * NUM_ACTIONS;

            if (temperature <= 0 || rng === null) {
                let best = 0;
                for (let a = 1; a < NUM_ACTIONS; a++) {
                    if (this.logits[base + a] > this.logits[base + best]) best = a;
                }
                this.actions[i] = best;
                continue;
            }

            let peak = -Infinity;
            for (let a = 0; a < NUM_ACTIONS; a++) {
                const scaled = this.logits[base + a] / temperature;
                this.probabilities[a] = scaled;
                if (scaled > peak) peak = scaled;
            }
            let total = 0;
            for (let a = 0; a < NUM_ACTIONS; a++) {
                const value = Math.exp(this.probabilities[a] - peak);
                this.probabilities[a] = value;
                total += value;
            }

            let threshold = rng.float(i) * total;
            let chosen = NUM_ACTIONS - 1;
            for (let a = 0; a < NUM_ACTIONS; a++) {
                threshold -= this.probabilities[a];
                if (threshold <= 0) {
                    chosen = a;
                    break;
                }
            }
            this.actions[i] = chosen;
        }
        return this.actions;
    }
}
