/**
 * Does the browser agent agree with JAX?
 *
 *   npm run test:agent
 *
 * Replays the reference produced by scripts/export_agent_web.py: twelve
 * controlled steps on real frames, then six dreamed steps branched off
 * the state they reach. Everything is compared, including the actions,
 * because an actor that is subtly wrong still looks like it is playing.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { AgentModel, type AgentState } from '../AgentModel.js';

const here = dirname(fileURLToPath(import.meta.url));
// Deliberately the shipped weights, not a build directory: this checks
// the bytes the site actually serves.
const served = join(here, '..', '..', '..', '..',
                    'public', 'models', 'world-models', 'follow');

const manifest = JSON.parse(readFileSync(join(served, 'manifest.json'), 'utf8'));
const parity = JSON.parse(readFileSync(join(here, 'agent-parity.json'), 'utf8'));
const bin = readFileSync(join(served, 'weights.bin'));
const buffer = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);

const model = new AgentModel(manifest, buffer);

const TOL = 3e-4;
let worst = 0;
let worstWhat = '';
let failures = 0;

function compare(label: string, got: ArrayLike<number>, want: ArrayLike<number>) {
    let m = 0;
    for (let i = 0; i < want.length; i++) {
        const d = Math.abs(got[i] - want[i]);
        if (d > m) m = d;
    }
    if (m > worst) { worst = m; worstWhat = label; }
    if (m > TOL) {
        failures++;
        if (failures <= 6) {
            console.log(`  FAIL ${label.padEnd(24)} max|diff| ${m.toExponential(3)}`);
        }
    }
}

// --- the controlled loop, on real frames from the JAX environment ---
let state: AgentState = model.initialState();
for (let t = 0; t < parity.steps.length; t++) {
    const frame = Float32Array.from(parity.frames[t]);
    state = model.observeAndAct(state, frame);
    compare(`controlled ${t} h`, state.h, parity.steps[t].h);
    compare(`controlled ${t} z`, state.z, parity.steps[t].z);
    compare(`controlled ${t} action`, state.a, parity.steps[t].action);
}

// --- the dreamed loop, branched off the state the filter reached ---
let frame = new Float32Array(0);
for (let t = 0; t < parity.dream.length; t++) {
    const out = model.imagine(state);
    state = out.state;
    frame = out.frame;
    compare(`dreamed ${t} h`, state.h, parity.dream[t].h);
    compare(`dreamed ${t} action`, state.a, parity.dream[t].action);

    let mean = 0, min = Infinity, max = -Infinity;
    for (const v of frame) { mean += v; if (v < min) min = v; if (v > max) max = v; }
    mean /= frame.length;
    let varsum = 0;
    for (const v of frame) varsum += (v - mean) ** 2;
    compare(`dreamed ${t} frame stats`,
        [mean, Math.sqrt(varsum / frame.length), min, max],
        parity.dream[t].frame_stats);
}
compare('dreamed final frame', frame, parity.dream_final_frame);

console.log(`controlled steps: ${parity.steps.length}, dreamed: ${parity.dream.length}`);
console.log(`worst disagreement: ${worst.toExponential(3)} (${worstWhat})`);
if (failures) {
    console.log(`${failures} comparison(s) over tolerance ${TOL}`);
    process.exit(1);
}
console.log(`all within ${TOL}: the browser runs the trained model`);
