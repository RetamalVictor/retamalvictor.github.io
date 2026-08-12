/**
 * The goal environment, ported from world_models/envs/bouncing_ball_goal.py.
 *
 * Two entities in a box: the agent, which the policy nudges, and the goal,
 * which normally cruises and bounces. In the demo the goal is under the
 * reader's cursor instead, which is a case the trained agent never saw and
 * handles anyway.
 *
 * The renderer matters more than usual here. It is the only thing the
 * policy ever receives, so it has to match the training-time renderer
 * exactly: agent white, goal red, additive and clipped where they overlap.
 */

export interface GoalParams {
    imgH: number;
    imgW: number;
    ballRadius: number;
    goalRadius: number;
    maxSpeed: number;
    maxNudge: number;
    goalSpeed: number;
    rewardSigma: number;
    dt: number;
}

export const GOAL_PARAMS: GoalParams = {
    imgH: 32,
    imgW: 32,
    ballRadius: 2.0,
    goalRadius: 2.0,
    maxSpeed: 2.0,
    maxNudge: 0.3,
    goalSpeed: 1.0,
    rewardSigma: 3.0,
    dt: 1.0,
};

export interface GoalState {
    x: number; y: number; vx: number; vy: number;
    gx: number; gy: number; gvx: number; gvy: number;
    step: number;
}

function reflect(pos: number, vel: number, limit: number): [number, number] {
    if (pos < 0) { pos = -pos; vel = -vel; }
    if (pos >= limit) { pos = 2 * limit - pos; vel = -vel; }
    return [pos, vel];
}

export function goalReset(rand: () => number, p: GoalParams): GoalState {
    const margin = 4.0;
    const angle = rand() * 2 * Math.PI;
    const gangle = rand() * 2 * Math.PI;
    const speed = p.maxSpeed * 0.8;
    return {
        x: margin + rand() * (p.imgW - 2 * margin),
        y: margin + rand() * (p.imgH - 2 * margin),
        vx: speed * Math.cos(angle),
        vy: speed * Math.sin(angle),
        gx: margin + rand() * (p.imgW - 2 * margin),
        gy: margin + rand() * (p.imgH - 2 * margin),
        gvx: p.goalSpeed * Math.cos(gangle),
        gvy: p.goalSpeed * Math.sin(gangle),
        step: 0,
    };
}

/**
 * One step. `goalOverride` pins the goal where the reader put it, which
 * replaces the goal's own physics for that step.
 */
export function goalStep(s: GoalState, ax: number, ay: number, p: GoalParams,
                         goalOverride?: { gx: number; gy: number }): GoalState {
    ax = Math.max(-p.maxNudge, Math.min(p.maxNudge, ax));
    ay = Math.max(-p.maxNudge, Math.min(p.maxNudge, ay));

    let vx = s.vx + ax;
    let vy = s.vy + ay;
    const speed = Math.hypot(vx, vy);
    if (speed > p.maxSpeed) {
        const k = p.maxSpeed / speed;
        vx *= k;
        vy *= k;
    }
    let x = s.x + vx * p.dt;
    let y = s.y + vy * p.dt;
    [x, vx] = reflect(x, vx, p.imgW);
    [y, vy] = reflect(y, vy, p.imgH);

    let { gx, gy, gvx, gvy } = s;
    if (goalOverride) {
        gx = Math.max(0, Math.min(p.imgW - 1e-3, goalOverride.gx));
        gy = Math.max(0, Math.min(p.imgH - 1e-3, goalOverride.gy));
        gvx = 0;
        gvy = 0;
    } else {
        gx = gx + gvx * p.dt;
        gy = gy + gvy * p.dt;
        [gx, gvx] = reflect(gx, gvx, p.imgW);
        [gy, gvy] = reflect(gy, gvy, p.imgH);
    }

    return { x, y, vx, vy, gx, gy, gvx, gvy, step: s.step + 1 };
}

function blob(out: Float32Array, cx: number, cy: number, sigma: number,
              H: number, W: number): void {
    const twoSigmaSq = 2 * sigma * sigma;
    for (let row = 0; row < H; row++) {
        const dy = row - cy;
        for (let col = 0; col < W; col++) {
            const dx = col - cx;
            out[row * W + col] = Math.exp(-(dx * dx + dy * dy) / twoSigmaSq);
        }
    }
}

/** (H, W, 3) float32 in [0, 1]: agent white, goal red. */
export function goalRender(s: GoalState, p: GoalParams): Float32Array {
    const { imgH: H, imgW: W } = p;
    const agent = new Float32Array(H * W);
    const goal = new Float32Array(H * W);
    blob(agent, s.x, s.y, p.ballRadius, H, W);
    blob(goal, s.gx, s.gy, p.goalRadius, H, W);

    const out = new Float32Array(H * W * 3);
    for (let i = 0; i < H * W; i++) {
        const a = agent[i], g = goal[i];
        out[i * 3] = Math.min(1, a + g);
        out[i * 3 + 1] = Math.min(1, a + 0.15 * g);
        out[i * 3 + 2] = Math.min(1, a + 0.15 * g);
    }
    return out;
}

/** Dense Gaussian kernel on the agent-to-goal distance. */
export function goalReward(s: GoalState, p: GoalParams): number {
    const d2 = (s.x - s.gx) ** 2 + (s.y - s.gy) ** 2;
    return Math.exp(-d2 / (2 * p.rewardSigma * p.rewardSigma));
}

function goalRng(seed: number): () => number {
    let a = seed >>> 0;
    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export { goalRng as makeGoalRng };
