/**
 * How large a fleet can the TypeScript path drive?
 *
 * The demo decouples simulation from rendering — robots move between grid cells
 * over several frames — so the target is roughly 10 policy steps per second for
 * *both* panels, not 60. This measures where that budget runs out, which is what
 * decides the fleet-size slider's range and whether the WebGPU path is needed
 * for the claim the demo is making.
 *
 * Run with:  npx tsx src/components/mapf/test/bench.ts
 */

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { MapfEnv, boardForAgents, sampleInstance } from '../env';
import { Policy } from '../policy';
import { Rng } from '../random';
import { parseModel } from '../weights';
import type { DemoManifest } from '../types';

const ASSETS = join(process.cwd(), 'public', 'assets', 'models', 'mapf');
const FLEETS = [10, 20, 40, 60, 100, 200, 350, 500];
const STEPS = 30;
const REPEATS = 5;

function readBuffer(name: string): ArrayBuffer {
    const raw = readFileSync(join(ASSETS, name));
    return raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
}

const manifest = JSON.parse(
    readFileSync(join(ASSETS, 'config.json'), 'utf8')
) as DemoManifest;

const models = Object.fromEntries(
    Object.entries(manifest.models).map(([name, entry]) => [
        name,
        parseModel(entry, readBuffer(entry.file)),
    ])
);

console.log(`${STEPS} steps per fleet size, both panels, temperature 3\n`);
console.log(
    'robots  board    comm ms   nocomm ms   both ms/step   steps/s   ' +
    'verdict (10 steps/s budget)'
);

for (const agents of FLEETS) {
    const board = boardForAgents(agents);
    const rng = new Rng(1);
    const instance = sampleInstance(rng, agents, board, manifest.defaults.obstacle_fraction);
    const envConfig = {
        board,
        sensingRange: manifest.sensing_range,
        pad: manifest.pad,
        maxNeighbours: manifest.max_neighbours,
    };

    const timings: Record<string, number> = {};
    for (const [name, model] of Object.entries(models)) {
        const env = new MapfEnv(envConfig, instance);
        const policy = new Policy(model, agents, manifest.fov[0]);

        // Warm the JIT before measuring.
        for (let i = 0; i < 5; i++) {
            policy.forward(env.fov, env.adjacency);
            env.step(policy.selectActions(0, null));
        }

        // Several short runs, best one wins. A single run at small fleet sizes
        // is a handful of milliseconds, where a GC pause or a background task
        // shows up as a larger effect than the thing being measured — enough to
        // make 60 robots look faster than 40.
        let best = Infinity;
        for (let repeat = 0; repeat < REPEATS; repeat++) {
            const env2 = new MapfEnv(envConfig, instance);
            const start = performance.now();
            for (let i = 0; i < STEPS; i++) {
                policy.forward(env2.fov, env2.adjacency);
                env2.step(policy.selectActions(0, null));
            }
            best = Math.min(best, (performance.now() - start) / STEPS);
        }
        timings[name] = best;
    }

    const both = timings.comm + timings.nocomm;
    const rate = 1000 / both;
    const verdict = rate >= 30 ? 'comfortable' : rate >= 10 ? 'fine' : rate >= 5 ? 'marginal' : 'needs GPU';
    console.log(
        `${String(agents).padStart(6)}  ${String(board).padStart(5)}  ` +
        `${timings.comm.toFixed(2).padStart(8)}  ${timings.nocomm.toFixed(2).padStart(10)}  ` +
        `${both.toFixed(2).padStart(12)}  ${rate.toFixed(1).padStart(8)}   ${verdict}`
    );
}
