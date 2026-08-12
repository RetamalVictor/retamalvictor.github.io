/**
 * Does the TypeScript ball environment match the JAX original?
 *
 *   npm run test:world-models
 *
 * The post tells readers that the demo runs the same physics the training
 * data came from. This checks it: same start state, same action sequence,
 * and the trajectory plus the rendered frame have to agree with a reference
 * dumped from world_models/envs/bouncing_ball.py.
 *
 * Regenerate the reference with scratchpad/ball_reference.py after any
 * change to the Python env.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DEFAULT_PARAMS, render, step, type EnvState } from '../BallEnv.js';

const here = dirname(fileURLToPath(import.meta.url));
const ref = JSON.parse(
    readFileSync(join(here, 'ball_reference.json'), 'utf8'),
) as {
    start: [number, number, number, number];
    actions: [number, number][];
    states: [number, number, number, number][];
    final_frame: number[];
};

// float32 in JAX against float64 here, so exact equality is not the bar.
const TOL = 2e-5;
let worst = 0;
let worstWhat = '';
let failures = 0;

function check(label: string, got: number, want: number): void {
    const d = Math.abs(got - want);
    if (d > worst) {
        worst = d;
        worstWhat = label;
    }
    if (d > TOL) {
        failures++;
        if (failures <= 8) {
            console.log(`  FAIL ${label}  got ${got}  want ${want}`);
        }
    }
}

const [x, y, vx, vy] = ref.start;
let state: EnvState = { x, y, vx, vy, step: 0 };

for (let t = 0; t < ref.actions.length; t++) {
    const [ax, ay] = ref.actions[t];
    state = step(state, ax, ay, DEFAULT_PARAMS);
    const want = ref.states[t];
    check(`step ${t} x`, state.x, want[0]);
    check(`step ${t} y`, state.y, want[1]);
    check(`step ${t} vx`, state.vx, want[2]);
    check(`step ${t} vy`, state.vy, want[3]);
}

const frame = render(state, DEFAULT_PARAMS);
if (frame.length !== ref.final_frame.length) {
    console.log(`  FAIL frame length ${frame.length} vs ${ref.final_frame.length}`);
    failures++;
} else {
    for (let i = 0; i < frame.length; i++) {
        check(`frame px ${i}`, frame[i], ref.final_frame[i]);
    }
}

console.log(`steps checked: ${ref.actions.length}, plus a ${frame.length}px frame`);
console.log(`worst disagreement: ${worst.toExponential(3)} (${worstWhat})`);

if (failures) {
    console.log(`${failures} value(s) over tolerance ${TOL}`);
    process.exit(1);
}
console.log(`all within ${TOL}: the browser runs the same physics as the dataset`);
