/**
 * MPC Hover Test
 *
 * Simple test: start at target, verify MPC maintains hover.
 * This isolates the core MPC behavior from trajectory tracking.
 *
 * Run with: npx tsx src/components/drone-racing/test/test-mpc-hover.ts
 */

import { MPC, DEFAULT_MPC_CONFIG } from '../control/MPC';
import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { DroneState, Waypoint } from '../types';

console.log('='.repeat(60));
console.log('MPC HOVER TEST');
console.log('='.repeat(60));

const g = DEFAULT_DYNAMICS_PARAMS.gravity;

// Test 1: Perfect hover - start exactly at target
console.log('\n--- TEST 1: Perfect Hover ---');
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    // Start exactly at target
    dynamics.setPosition(0, 2, 0);

    // Reference: hover at same position
    const getReference = (t: number): Waypoint => ({
        position: { x: 0, y: 2, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        acceleration: { x: 0, y: 0, z: 0 },
        jerk: { x: 0, y: 0, z: 0 },
        heading: 0,
        headingRate: 0,
        time: t,
    });

    console.log(`Target: hover at (0, 2, 0)`);
    console.log(`\nSimulating...`);

    let simTime = 0;
    const dt = 0.02;

    for (let step = 0; step < 50; step++) {
        const state = dynamics.getState();
        const cmd = mpc.computeControl(state, getReference, simTime);

        if (step % 10 === 0) {
            console.log(`t=${simTime.toFixed(2)}: pos=(${state.position.x.toFixed(4)}, ${state.position.y.toFixed(4)}, ${state.position.z.toFixed(4)}) T=${cmd.thrust.toFixed(2)} rates=(${cmd.rollRate.toFixed(3)}, ${cmd.pitchRate.toFixed(3)}, ${cmd.yawRate.toFixed(3)})`);
        }

        dynamics.step(cmd, dt);
        simTime += dt;
    }

    const finalState = dynamics.getState();
    const err = Math.sqrt(
        finalState.position.x ** 2 +
        (finalState.position.y - 2) ** 2 +
        finalState.position.z ** 2
    );
    console.log(`\nFinal error: ${err.toFixed(4)} m`);
    console.log(`PASS: ${err < 0.01 ? 'YES' : 'NO'}`);
}

// Test 2: Hover with small perturbation
console.log('\n--- TEST 2: Small Perturbation ---');
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    // Start with 10cm offset
    dynamics.setPosition(0.1, 2.0, 0.1);

    const getReference = (t: number): Waypoint => ({
        position: { x: 0, y: 2, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        acceleration: { x: 0, y: 0, z: 0 },
        jerk: { x: 0, y: 0, z: 0 },
        heading: 0,
        headingRate: 0,
        time: t,
    });

    console.log(`Initial: (0.1, 2.0, 0.1)`);
    console.log(`Target: (0, 2, 0)`);
    console.log(`\nSimulating 2 seconds...`);

    let simTime = 0;
    const dt = 0.02;
    const errors: number[] = [];

    for (let step = 0; step < 100; step++) {
        const state = dynamics.getState();
        const cmd = mpc.computeControl(state, getReference, simTime);
        const err = Math.sqrt(
            state.position.x ** 2 +
            (state.position.y - 2) ** 2 +
            state.position.z ** 2
        );
        errors.push(err);

        if (step % 20 === 0) {
            console.log(`t=${simTime.toFixed(2)}: err=${err.toFixed(4)} T=${cmd.thrust.toFixed(2)} rates=(${cmd.rollRate.toFixed(3)}, ${cmd.pitchRate.toFixed(3)}, ${cmd.yawRate.toFixed(3)})`);
        }

        dynamics.step(cmd, dt);
        simTime += dt;
    }

    const finalErr = errors[errors.length - 1];
    const maxErr = Math.max(...errors);
    console.log(`\nFinal error: ${finalErr.toFixed(4)} m`);
    console.log(`Max error: ${maxErr.toFixed(4)} m`);
    console.log(`Error monotonically decreasing: ${errors.every((e, i) => i === 0 || e <= errors[i - 1] * 1.01) ? 'YES' : 'NO'}`);
    console.log(`PASS: ${finalErr < 0.05 && maxErr < 0.2 ? 'YES' : 'NO'}`);
}

// Test 3: Debug single MPC step
console.log('\n--- TEST 3: Single Step Debug ---');
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    dynamics.setPosition(0.1, 2.1, 0);

    const getReference = (t: number): Waypoint => ({
        position: { x: 0, y: 2, z: 0 },
        velocity: { x: 0, y: 0, z: 0 },
        acceleration: { x: 0, y: 0, z: 0 },
        jerk: { x: 0, y: 0, z: 0 },
        heading: 0,
        headingRate: 0,
        time: t,
    });

    const state = dynamics.getState();
    console.log(`State: pos=(${state.position.x}, ${state.position.y}, ${state.position.z})`);
    console.log(`Target: pos=(0, 2, 0)`);
    console.log(`Error: x=${state.position.x.toFixed(3)}, y=${(state.position.y - 2).toFixed(3)} (above target)`);

    const cmd = mpc.computeControl(state, getReference, 0);
    console.log(`\nMPC Command:`);
    console.log(`  Thrust: ${cmd.thrust.toFixed(4)} (hover=${g.toFixed(4)})`);
    console.log(`  Thrust diff: ${(cmd.thrust - g).toFixed(4)} (should be < 0 to descend)`);
    console.log(`  Roll rate: ${cmd.rollRate.toFixed(4)} (should be negative to go -X)`);
    console.log(`  Pitch rate: ${cmd.pitchRate.toFixed(4)} (should be 0)`);
    console.log(`  Yaw rate: ${cmd.yawRate.toFixed(4)}`);

    // What is expected?
    console.log(`\nExpected behavior:`);
    console.log(`  - Thrust < gravity to descend 0.1m`);
    console.log(`  - Negative roll rate to move toward X=0`);
    console.log(`  - Commands should be small (small error)`);
}

console.log('\n' + '='.repeat(60));
console.log('HOVER TEST COMPLETE');
console.log('='.repeat(60));
