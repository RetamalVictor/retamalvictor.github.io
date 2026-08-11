/**
 * Seeded randomness.
 *
 * Two things need it: generating an instance, and sampling actions. Both must
 * replay identically from a seed, so that the deadlock preset always deadlocks
 * and a reported result can be reproduced.
 *
 * xorshift32 is not a great generator, but breaking a tie between two robots
 * that want the same cell is not a demanding application, and it costs a few
 * integer operations per robot per step.
 */

const UINT32 = 4294967296;

function mix(seed: number): number {
    // Splitmix-style avalanche so that seeds 1, 2, 3 give unrelated streams
    // rather than nearly identical ones.
    let x = (seed | 0) ^ 0x9e3779b9;
    x = Math.imul(x ^ (x >>> 16), 0x21f0aaad);
    x = Math.imul(x ^ (x >>> 15), 0x735a2d97);
    x = x ^ (x >>> 15);
    return x === 0 ? 0x9e3779b9 : x;
}

export class Rng {
    private state: number;

    constructor(seed: number) {
        this.state = mix(seed);
    }

    nextUint32(): number {
        let x = this.state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.state = x | 0;
        return x >>> 0;
    }

    /** Uniform in [0, 1). */
    nextFloat(): number {
        return this.nextUint32() / UINT32;
    }

    /** Uniform integer in [0, bound). */
    nextInt(bound: number): number {
        return Math.floor(this.nextFloat() * bound);
    }
}

/**
 * One independent stream per robot.
 *
 * Kept as a flat array rather than an array of objects: at 500 robots this runs
 * every frame, and the whole point is that it stays cheap.
 */
export class FleetRng {
    private states: Uint32Array;

    constructor(count: number, seed: number) {
        this.states = new Uint32Array(count);
        this.reseed(seed);
    }

    reseed(seed: number): void {
        for (let i = 0; i < this.states.length; i++) {
            this.states[i] = mix(seed + i * 0x9e3779b9) >>> 0;
        }
    }

    /** Uniform in [0, 1) from robot `i`'s stream. */
    float(i: number): number {
        let x = this.states[i] | 0;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.states[i] = x >>> 0;
        return (x >>> 0) / UINT32;
    }

    get size(): number {
        return this.states.length;
    }
}
