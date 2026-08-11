/**
 * The grid world, ported from `grid/env_vectorized.py` in the MAPF-GNN
 * repository (which is itself pinned to the original `GraphEnv` by an
 * equivalence test).
 *
 * The order of operations inside `step` is load-bearing and not obvious. Three
 * details in particular will silently produce a demo that behaves unlike every
 * published number if you get them wrong:
 *
 *   - obstacles are tested *before* the boundary clamp, because a move off the
 *     board is simply not in the obstacle set;
 *   - reverting a collision is *not* re-checked for collisions it may create;
 *   - vacated cells are cleared *before* new ones are stamped, so a robot
 *     moving into a cell another just left survives the clear.
 *
 * Two more live in the observation: robots are absent from the board until the
 * first step, and each field-of-view patch is row-flipped.
 */

import { Rng } from './random';
import {
    CELL_AGENT,
    CELL_FREE,
    CELL_OBSTACLE,
    GOAL_MARKER,
    type EnvConfig,
    type Instance,
} from './types';

/** Index order matches the Python `action_list`: idle, right, up, left, down. */
export const ACTION_DX = [0, 1, 0, -1, 0];
export const ACTION_DY = [0, 0, 1, 0, -1];
export const NUM_ACTIONS = 5;

let nextInstanceId = 0;

export class MapfEnv {
    /** Distinguishes one episode's instance from the next, for render caches. */
    instanceId = nextInstanceId++;

    readonly board: number;
    readonly numAgents: number;
    /** Assignable: the demo exposes it, and the graph is rebuilt every step. */
    sensingRange: number;
    readonly pad: number;
    /** Side of the field-of-view patch, `pad * 2 - 1`. */
    readonly side: number;

    /** Board contents: free, agent, or obstacle. Row-major, `y * board + x`. */
    readonly cells: Float32Array;
    /** Obstacles kept separately, as the Python does — `cells` also holds robots. */
    readonly obstacles: Uint8Array;

    readonly posX: Int32Array;
    readonly posY: Int32Array;
    readonly goalX: Int32Array;
    readonly goalY: Int32Array;

    /** Row-major `N x N`; 1 where `j` is a neighbour of `i`. Not symmetric. */
    readonly adjacency: Float32Array;
    /** `N x 2 x side x side`, the policy input. */
    readonly fov: Float32Array;

    /**
     * Consecutive steps a robot has failed to move while not on its goal.
     *
     * A single stationary step means nothing — idling is a legal action and
     * robots take it constantly. Repeated failure to move is the signature of
     * the deadlock this whole demo is about, so it is counted rather than
     * inferred from one frame.
     */
    readonly stalled: Int32Array;

    time = 0;
    /**
     * Bumped on every reset and step. The renderer caches per-step work — the
     * edge list above all — and uses this to know when to rebuild it.
     */
    version = 0;

    /** Positions before the last step. The renderer interpolates from these. */
    readonly prevX: Int32Array;
    readonly prevY: Int32Array;

    private readonly startX: Int32Array;
    private readonly startY: Int32Array;
    private readonly newX: Int32Array;
    private readonly newY: Int32Array;
    private readonly distances: Float32Array;
    private readonly occupancy: Int32Array;

    constructor(config: EnvConfig, instance: Instance) {
        this.board = instance.board;
        this.numAgents = instance.starts.length;
        this.sensingRange = config.sensingRange;
        this.pad = config.pad;
        this.side = config.pad * 2 - 1;

        if (instance.goals.length !== this.numAgents) {
            throw new Error('starts and goals must have the same length');
        }

        const n = this.numAgents;
        const area = this.board * this.board;

        this.cells = new Float32Array(area);
        this.obstacles = new Uint8Array(area);
        this.posX = new Int32Array(n);
        this.posY = new Int32Array(n);
        this.goalX = new Int32Array(n);
        this.goalY = new Int32Array(n);
        this.startX = new Int32Array(n);
        this.startY = new Int32Array(n);
        this.prevX = new Int32Array(n);
        this.prevY = new Int32Array(n);
        this.newX = new Int32Array(n);
        this.newY = new Int32Array(n);
        this.stalled = new Int32Array(n);
        this.adjacency = new Float32Array(n * n);
        this.distances = new Float32Array(n * n);
        this.occupancy = new Int32Array(area);
        this.fov = new Float32Array(n * 2 * this.side * this.side);

        this.load(instance);
    }

    /**
     * Swap in a new instance without reallocating.
     *
     * An episode lasts a few seconds, and rebuilding the environment for each
     * one means an N-by-N distance matrix, an N-by-N adjacency and the field-of-
     * view buffers all becoming garbage on a timer. Reusing them turns a
     * periodic collection — visible as a dropped frame — into nothing at all.
     */
    load(instance: Instance): void {
        if (instance.board !== this.board) {
            throw new Error(`instance board ${instance.board} does not match ${this.board}`);
        }
        if (instance.starts.length !== this.numAgents) {
            throw new Error('instance fleet size does not match this environment');
        }

        this.instanceId = nextInstanceId++;
        this.obstacles.fill(0);
        for (const [x, y] of instance.obstacles) {
            this.obstacles[y * this.board + x] = 1;
        }
        instance.starts.forEach(([x, y], i) => {
            this.startX[i] = x;
            this.startY[i] = y;
        });
        instance.goals.forEach(([x, y], i) => {
            this.goalX[i] = x;
            this.goalY[i] = y;
        });

        this.reset();
    }

    reset(): void {
        this.time = 0;
        this.version++;
        this.posX.set(this.startX);
        this.posY.set(this.startY);
        this.prevX.set(this.startX);
        this.prevY.set(this.startY);
        this.stalled.fill(0);

        // The board carries obstacles only. The Python stamps robots inside
        // updateBoard(), which runs during step, so the very first field of view
        // shows no other robots at all.
        this.cells.fill(CELL_FREE);
        for (let cell = 0; cell < this.obstacles.length; cell++) {
            if (this.obstacles[cell]) this.cells[cell] = CELL_OBSTACLE;
        }

        this.computeGraph();
        this.computeFov();
    }

    step(actions: Int32Array | number[]): void {
        const n = this.numAgents;
        const board = this.board;
        const last = board - 1;

        this.prevX.set(this.posX);
        this.prevY.set(this.posY);

        for (let i = 0; i < n; i++) {
            const action = actions[i];
            let x = this.posX[i] + ACTION_DX[action];
            let y = this.posY[i] + ACTION_DY[action];

            // Obstacles before the clamp: an out-of-bounds move is not in the
            // obstacle set, so it is left to the clamp below instead.
            const inBounds = x >= 0 && x <= last && y >= 0 && y <= last;
            if (inBounds && this.obstacles[y * board + x]) {
                x = this.prevX[i];
                y = this.prevY[i];
            }

            this.newX[i] = x < 0 ? 0 : x > last ? last : x;
            this.newY[i] = y < 0 ? 0 : y > last ? last : y;
        }

        // Every robot sharing a cell reverts, including groups of three or more.
        // Reverting is not re-checked, so a reverted robot may land on top of a
        // robot that did not move. The Python does the same.
        this.occupancy.fill(0);
        for (let i = 0; i < n; i++) {
            this.occupancy[this.newY[i] * board + this.newX[i]]++;
        }
        for (let i = 0; i < n; i++) {
            if (this.occupancy[this.newY[i] * board + this.newX[i]] > 1) {
                this.posX[i] = this.prevX[i];
                this.posY[i] = this.prevY[i];
            } else {
                this.posX[i] = this.newX[i];
                this.posY[i] = this.newY[i];
            }
        }

        for (let i = 0; i < n; i++) {
            const moved = this.posX[i] !== this.prevX[i] || this.posY[i] !== this.prevY[i];
            this.stalled[i] = moved || this.atGoal(i) ? 0 : this.stalled[i] + 1;
        }

        // Clear every vacated cell first, then stamp the new ones.
        for (let i = 0; i < n; i++) {
            this.cells[this.prevY[i] * board + this.prevX[i]] = CELL_FREE;
        }
        for (let i = 0; i < n; i++) {
            this.cells[this.posY[i] * board + this.posX[i]] = CELL_AGENT;
        }

        this.time++;
        this.version++;
        this.computeGraph();
        this.computeFov();
    }

    /**
     * Nearest-four communication graph within the sensing range.
     *
     * A robot with fewer than four neighbours keeps all of them, and ties at the
     * fourth distance are all kept — both follow from comparing against the
     * fourth-smallest distance rather than taking a fixed-size top-k.
     *
     * The result is *not* symmetric: `j` can be among `i`'s four nearest while
     * `i` is not among `j`'s.
     */
    private computeGraph(): void {
        const n = this.numAgents;
        const range = this.sensingRange;
        this.adjacency.fill(0);

        for (let i = 0; i < n; i++) {
            const xi = this.posX[i];
            const yi = this.posY[i];
            for (let j = i + 1; j < n; j++) {
                const dx = xi - this.posX[j];
                const dy = yi - this.posY[j];
                const d = Math.sqrt(dx * dx + dy * dy);
                const kept = d >= range ? 0 : d;
                this.distances[i * n + j] = kept;
                this.distances[j * n + i] = kept;
            }
            this.distances[i * n + i] = 0;
        }

        for (let i = 0; i < n; i++) {
            const row = i * n;
            // Four smallest strictly-positive distances. Distances of zero mean
            // "out of range" or "self", never "adjacent".
            let t0 = Infinity, t1 = Infinity, t2 = Infinity, t3 = Infinity;
            for (let j = 0; j < n; j++) {
                const d = this.distances[row + j];
                if (d <= 0) continue;
                if (d < t0) { t3 = t2; t2 = t1; t1 = t0; t0 = d; }
                else if (d < t1) { t3 = t2; t2 = t1; t1 = d; }
                else if (d < t2) { t3 = t2; t2 = d; }
                else if (d < t3) { t3 = d; }
            }
            for (let j = 0; j < n; j++) {
                const d = this.distances[row + j];
                if (d > 0 && d <= t3) this.adjacency[row + j] = 1;
            }
        }
    }

    /**
     * Two channels per robot: the local map, and a marker for where its goal is.
     *
     * The goal is almost never inside a 5x5 window, so it is projected onto the
     * window's edge — the robot is told a direction, not a position.
     */
    private computeFov(): void {
        const n = this.numAgents;
        const side = this.side;
        const pad = this.pad;
        const board = this.board;
        const plane = side * side;

        this.fov.fill(0);

        for (let i = 0; i < n; i++) {
            const base = i * 2 * plane;
            const originX = this.posX[i] + 1 - pad;
            const originY = this.posY[i] + 1 - pad;

            for (let r = 0; r < side; r++) {
                // The Python flips the rows of each patch.
                const y = originY + (side - 1 - r);
                if (y < 0 || y >= board) continue;
                for (let c = 0; c < side; c++) {
                    const x = originX + c;
                    if (x < 0 || x >= board) continue;
                    this.fov[base + r * side + c] = this.cells[y * board + x];
                }
            }

            const [goalRow, goalCol] = this.goalCell(i);
            this.fov[base + plane + goalRow * side + goalCol] = GOAL_MARKER;
        }
    }

    /**
     * Where in the patch to stamp the goal marker, as [row, column].
     *
     * Transcribed from `map_goal`. The padding cancels out of every comparison,
     * so this is written in board coordinates. Note that the row test uses `>=`
     * on one side where the column test uses `>` — the asymmetry is in the
     * original and is reproduced rather than tidied.
     */
    private goalCell(i: number): [number, number] {
        const pad = this.pad;
        const dx = this.goalX[i] - this.posX[i];
        const dy = this.goalY[i] - this.posY[i];

        let col: number;
        if (dx < pad - 1 && dx > -(pad - 1)) col = dx + pad - 1;
        else if (dx <= -(pad - 1)) col = 0;
        else col = 1 + pad;

        let row: number;
        if (dy < pad - 1 && dy >= -(pad - 1)) row = -dy + pad - 1;
        else if (dy <= -(pad - 1)) row = 1 + pad;
        else row = 0;

        return [row, col];
    }

    atGoal(i: number): boolean {
        return this.posX[i] === this.goalX[i] && this.posY[i] === this.goalY[i];
    }

    numAtGoal(): number {
        let count = 0;
        for (let i = 0; i < this.numAgents; i++) if (this.atGoal(i)) count++;
        return count;
    }

    /** Robots that have been unable to move for `threshold` consecutive steps. */
    numStuck(threshold: number): number {
        let count = 0;
        for (let i = 0; i < this.numAgents; i++) if (this.stalled[i] >= threshold) count++;
        return count;
    }

    allAtGoal(): boolean {
        return this.numAtGoal() === this.numAgents;
    }

    meanNeighbours(): number {
        let total = 0;
        for (let k = 0; k < this.adjacency.length; k++) total += this.adjacency[k];
        return this.numAgents === 0 ? 0 : total / this.numAgents;
    }
}

/**
 * Board side for a fleet, holding robots-per-cell fixed.
 *
 * The models were trained with 10 robots on 20x20; keeping density constant is
 * what makes a fleet-size sweep a test of the policy rather than of how empty
 * the board got. Reproduces the boards used in the article: 10 -> 20, 20 -> 28,
 * 40 -> 40, 60 -> 49, 100 -> 63.
 */
export function boardForAgents(agents: number): number {
    return Math.round(Math.sqrt(agents / 10) * 20);
}

/**
 * Draw obstacles, starts and goals from distinct cells.
 *
 * Distinctness matters: two robots starting on the same cell is a state the
 * environment cannot produce and the policy has never seen.
 */
export function sampleInstance(
    rng: Rng,
    agents: number,
    board: number,
    obstacleFraction: number
): Instance {
    const area = board * board;
    const numObstacles = Math.round(area * obstacleFraction);
    const needed = numObstacles + 2 * agents;
    if (needed > area) {
        throw new Error(`cannot place ${needed} distinct cells on a ${board}x${board} board`);
    }

    // Partial Fisher-Yates: only the prefix we consume is shuffled.
    const pool = new Int32Array(area);
    for (let i = 0; i < area; i++) pool[i] = i;
    for (let i = 0; i < needed; i++) {
        const j = i + rng.nextInt(area - i);
        const swap = pool[i];
        pool[i] = pool[j];
        pool[j] = swap;
    }

    const cell = (k: number): [number, number] => [pool[k] % board, Math.floor(pool[k] / board)];

    const obstacles: Array<[number, number]> = [];
    const starts: Array<[number, number]> = [];
    const goals: Array<[number, number]> = [];
    let k = 0;
    for (let i = 0; i < numObstacles; i++) obstacles.push(cell(k++));
    for (let i = 0; i < agents; i++) starts.push(cell(k++));
    for (let i = 0; i < agents; i++) goals.push(cell(k++));

    return { board, obstacles, starts, goals };
}
