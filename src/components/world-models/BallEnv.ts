/**
 * The bouncing ball environment, ported from the JAX original in
 * world_models/envs/bouncing_ball.py.
 *
 * This is not a re-imagining. The integration order, the reflection rule,
 * the speed clamp and the Gaussian renderer are the same operations in the
 * same sequence, so the frames here are the frames the VAE in the post was
 * trained on. Kept dependency-free and deterministic given a seed.
 */

export interface EnvParams {
    imgH: number;
    imgW: number;
    ballRadius: number;   // sigma of the Gaussian blob, in pixels
    maxSpeed: number;     // clamp on velocity magnitude
    maxNudge: number;     // action clamp
    dt: number;
}

export const DEFAULT_PARAMS: EnvParams = {
    imgH: 32,
    imgW: 32,
    ballRadius: 2.0,
    maxSpeed: 2.0,
    maxNudge: 0.3,
    dt: 1.0,
};

export interface EnvState {
    x: number;
    y: number;
    vx: number;
    vy: number;
    step: number;
}

/** Mulberry32: small, seeded, and identical across browsers. */
export function makeRng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Reflect position and velocity so the ball stays inside [0, limit).
 *
 * One floor and one ceiling reflection is enough only because a ball
 * cannot cross the box in a single step (maxSpeed is far below the box
 * size). That assumption is load bearing and it is the same one the
 * Python version documents.
 */
function reflect(pos: number, vel: number, limit: number): [number, number] {
    if (pos < 0) {
        pos = -pos;
        vel = -vel;
    }
    if (pos >= limit) {
        pos = 2 * limit - pos;
        vel = -vel;
    }
    return [pos, vel];
}

export function reset(rand: () => number, p: EnvParams): EnvState {
    const margin = 4.0;
    const angle = rand() * 2 * Math.PI;
    const speed = p.maxSpeed * 0.8;
    return {
        x: margin + rand() * (p.imgW - 2 * margin),
        y: margin + rand() * (p.imgH - 2 * margin),
        vx: speed * Math.cos(angle),
        vy: speed * Math.sin(angle),
        step: 0,
    };
}

export function step(s: EnvState, ax: number, ay: number, p: EnvParams): EnvState {
    ax = Math.max(-p.maxNudge, Math.min(p.maxNudge, ax));
    ay = Math.max(-p.maxNudge, Math.min(p.maxNudge, ay));

    let vx = s.vx + ax;
    let vy = s.vy + ay;

    // Rescale rather than clip, so direction survives the clamp.
    const speed = Math.hypot(vx, vy);
    if (speed > p.maxSpeed) {
        const scale = p.maxSpeed / speed;
        vx *= scale;
        vy *= scale;
    }

    let x = s.x + vx * p.dt;
    let y = s.y + vy * p.dt;

    [x, vx] = reflect(x, vx, p.imgW);
    [y, vy] = reflect(y, vy, p.imgH);

    return { x, y, vx, vy, step: s.step + 1 };
}

/** Gaussian blob, the reason position is sub-pixel rather than quantised. */
export function render(s: EnvState, p: EnvParams): Float32Array {
    const img = new Float32Array(p.imgH * p.imgW);
    const twoSigmaSq = 2 * p.ballRadius * p.ballRadius;
    for (let row = 0; row < p.imgH; row++) {
        const dy = row - s.y;
        for (let col = 0; col < p.imgW; col++) {
            const dx = col - s.x;
            img[row * p.imgW + col] = Math.exp(-(dx * dx + dy * dy) / twoSigmaSq);
        }
    }
    return img;
}

/** Uniform nudges in [-scale, scale]^2, the policy that made the dataset. */
export function randomNudge(rand: () => number, scale = 0.3): [number, number] {
    return [(rand() * 2 - 1) * scale, (rand() * 2 - 1) * scale];
}
