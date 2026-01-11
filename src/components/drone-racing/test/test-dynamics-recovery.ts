/**
 * Test DroneDynamics recovery from tilted state
 * Check if the dynamics can recover from a 30° roll
 */

import { DroneDynamics } from '../core/DroneDynamics';
import { MPCModel } from '../control/MPCModel';

const dynamics = new DroneDynamics();
const model = new MPCModel();
const g = 9.81;
const dt = 0.02;

console.log('='.repeat(70));
console.log('DYNAMICS RECOVERY TEST');
console.log('='.repeat(70));

// Test 1: Start with 30° roll, try to level out with roll rate command
console.log('\n--- Test 1: 30° roll, level out with max roll rate ---');
{
    const dyn = new DroneDynamics();
    dyn.setPosition(0, 2, 0);
    // Apply roll by stepping with positive roll rate
    for (let i = 0; i < 15; i++) {
        dyn.step({ thrust: g, rollRate: 2, pitchRate: 0, yawRate: 0, timestamp: 0 }, dt);
    }

    const state0 = dyn.getState();
    const mpcState0 = model.fromDroneState(state0.position, state0.velocity, state0.orientation);
    console.log(`Initial: roll=${(mpcState0.roll * 180 / Math.PI).toFixed(1)}°, y=${state0.position.y.toFixed(3)}m`);

    // Now command max negative roll rate to level out
    let lastRoll = mpcState0.roll;
    for (let i = 0; i < 50; i++) {
        dyn.step({ thrust: g * 1.2, rollRate: -10, pitchRate: 0, yawRate: 0, timestamp: 0 }, dt);
        const state = dyn.getState();
        const mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);

        if (i % 10 === 0) {
            console.log(`t=${(i * dt).toFixed(2)}s: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, y=${state.position.y.toFixed(3)}m, rollRate=${(mpcState.rollRate * 180 / Math.PI).toFixed(1)}°/s`);
        }
        lastRoll = mpcState.roll;
    }
    const finalState = dyn.getState();
    console.log(`Final: roll=${(lastRoll * 180 / Math.PI).toFixed(1)}°, y=${finalState.position.y.toFixed(3)}m`);
    console.log(`Expected: roll should decrease to 0°, y should stay near 2m`);
}

// Test 2: Start with 90° roll, check behavior
console.log('\n--- Test 2: 90° roll (thrust sideways), apply corrections ---');
{
    const dyn = new DroneDynamics();
    dyn.setPosition(0, 2, 0);
    // Apply roll by stepping with positive roll rate to get to 90°
    for (let i = 0; i < 45; i++) {
        dyn.step({ thrust: g, rollRate: 2, pitchRate: 0, yawRate: 0, timestamp: 0 }, dt);
    }

    const state0 = dyn.getState();
    const mpcState0 = model.fromDroneState(state0.position, state0.velocity, state0.orientation);
    console.log(`Initial: roll=${(mpcState0.roll * 180 / Math.PI).toFixed(1)}°, y=${state0.position.y.toFixed(3)}m`);

    // Compute thrust direction
    const sr = Math.sin(mpcState0.roll), cr = Math.cos(mpcState0.roll);
    const sp = Math.sin(mpcState0.pitch), cp = Math.cos(mpcState0.pitch);
    const sy = Math.sin(mpcState0.yaw), cy = Math.cos(mpcState0.yaw);
    console.log(`Thrust direction: (${(-sr*cy + cr*sp*sy).toFixed(3)}, ${(cr*cp).toFixed(3)}, ${(cr*sp*cy + sr*sy).toFixed(3)})`);

    // Try to recover with max negative roll rate and increased thrust
    for (let i = 0; i < 50; i++) {
        const state = dyn.getState();
        const mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);

        // Command to recover: negative roll rate and high thrust
        const rollRateCmd = mpcState.roll > 0 ? -10 : 10;
        dyn.step({ thrust: 20, rollRate: rollRateCmd, pitchRate: 0, yawRate: 0, timestamp: 0 }, dt);

        if (i % 10 === 0) {
            console.log(`t=${(i * dt).toFixed(2)}s: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, y=${state.position.y.toFixed(3)}m`);
        }
    }
    const finalState = dyn.getState();
    const finalMpc = model.fromDroneState(finalState.position, finalState.velocity, finalState.orientation);
    console.log(`Final: roll=${(finalMpc.roll * 180 / Math.PI).toFixed(1)}°, y=${finalState.position.y.toFixed(3)}m`);
}

// Test 3: Circular trajectory with open-loop feedforward
console.log('\n--- Test 3: Circular trajectory with feedforward (no MPC) ---');
{
    const dyn = new DroneDynamics();
    const radius = 5, speed = 3, height = 2;
    const omega = speed / radius;
    const centripetalAccel = speed * speed / radius;

    // Start at (5, 2, 0) with velocity (-0, 0, 3)... wait, at angle=0:
    // position = (radius, height, 0) = (5, 2, 0)
    // velocity = (-speed*sin(0), 0, speed*cos(0)) = (0, 0, 3)
    dyn.setPosition(5, 2, 0);
    dyn.setVelocity(0, 0, 3);
    dyn.setHeading(0);

    // Required roll for circular flight
    const requiredRoll = Math.atan(centripetalAccel / g);
    console.log(`Required bank angle: ${(requiredRoll * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Required thrust magnitude: ${Math.sqrt(g*g + centripetalAccel*centripetalAccel).toFixed(2)} m/s²`);

    // Simulate with feedforward only (no feedback)
    let simTime = 0;
    const period = 2 * Math.PI / omega;
    for (let i = 0; i < Math.ceil(period / dt); i++) {
        const angle = omega * simTime;

        // Reference values
        const refX = radius * Math.cos(angle);
        const refZ = radius * Math.sin(angle);
        const heading = -angle;

        // Feedforward thrust
        const thrustMag = Math.sqrt(g*g + centripetalAccel*centripetalAccel);

        // Feedforward roll rate (to maintain bank into turn)
        // For steady circular, roll is constant, so rollRate = 0
        // But yaw is changing at -omega, so we need:
        // Actually for a coordinated turn, body-frame angular velocity is:
        // omega_body = R^T * omega_world
        // where omega_world = (0, -omega, 0) for yaw rate around Y

        dyn.step({
            thrust: thrustMag,
            rollRate: 0,
            pitchRate: 0,
            yawRate: -omega,
            timestamp: 0
        }, dt);

        const state = dyn.getState();

        if (i % Math.floor(period / dt / 8) === 0) {
            const err = Math.sqrt((state.position.x - refX)**2 + (state.position.y - height)**2 + (state.position.z - refZ)**2);
            console.log(`t=${simTime.toFixed(2)}s: pos=(${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}, ${state.position.z.toFixed(2)}) ref=(${refX.toFixed(2)}, ${height}, ${refZ.toFixed(2)}) err=${err.toFixed(3)}m`);
        }

        simTime += dt;
    }
}

console.log('\n' + '='.repeat(70));
