/**
 * Test quaternion to Euler conversion
 */

import { DroneDynamics } from '../core/DroneDynamics';
import { MPCModel } from '../control/MPCModel';

const dynamics = new DroneDynamics();
const model = new MPCModel();

console.log('='.repeat(70));
console.log('EULER CONVERSION TEST');
console.log('='.repeat(70));

// Test 1: Identity (level flight)
console.log('\n--- Test 1: Level flight ---');
dynamics.setPosition(0, 2, 0);
dynamics.setHeading(0);
let state = dynamics.getState();
let mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`Heading: 0°`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);

// Test 2: Heading = -90° (looking in -Z direction)
console.log('\n--- Test 2: Heading -90° ---');
dynamics.setHeading(-Math.PI / 2);
state = dynamics.getState();
mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`Heading: -90°`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);

// Test 3: Heading = -180° (looking in +X direction, reversed)
console.log('\n--- Test 3: Heading -180° ---');
dynamics.setHeading(-Math.PI);
state = dynamics.getState();
mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`Heading: -180°`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);

// Test 4: Simulate tilted flight
console.log('\n--- Test 4: After applying roll rate ---');
dynamics.setHeading(0);
// Apply roll rate for a bit
for (let i = 0; i < 10; i++) {
    dynamics.step({ thrust: 9.81, rollRate: 1, pitchRate: 0, yawRate: 0, timestamp: 0 }, 0.02);
}
state = dynamics.getState();
mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`After 0.2s of rollRate=1:`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);
console.log(`Expected roll: ~11.5° (1 rad/s * 0.2s * 180/π)`);

// Test 5: After applying pitch rate
console.log('\n--- Test 5: After applying pitch rate ---');
dynamics.setHeading(0);
dynamics.setPosition(0, 2, 0);
// Reset orientation
const freshDynamics = new DroneDynamics();
for (let i = 0; i < 10; i++) {
    freshDynamics.step({ thrust: 9.81, rollRate: 0, pitchRate: 1, yawRate: 0, timestamp: 0 }, 0.02);
}
state = freshDynamics.getState();
mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`After 0.2s of pitchRate=1:`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);
console.log(`Expected pitch: ~11.5°`);

// Test 6: Simulate circular trajectory
console.log('\n--- Test 6: During circular trajectory at 90° ---');
const circDynamics = new DroneDynamics();
const radius = 5;
const speed = 3;
const omega = speed / radius;

circDynamics.setPosition(radius, 2, 0);
circDynamics.setVelocity(0, 0, speed);
circDynamics.setHeading(0);

// Simulate until angle = 90°
let simTime = 0;
const targetTime = Math.PI / 2 / omega;  // t = π/2 / ω = π/2 * 5/3 ≈ 2.62s
while (simTime < targetTime) {
    // Simple feedforward
    const cmd = {
        thrust: 10,  // Approximate
        rollRate: 0,
        pitchRate: 0,
        yawRate: -omega,
        timestamp: 0,
    };
    circDynamics.step(cmd, 0.02);
    simTime += 0.02;
}

state = circDynamics.getState();
mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
console.log(`After simulating to 90° point:`);
console.log(`Position: (${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}, ${state.position.z.toFixed(2)})`);
console.log(`Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);
console.log(`MPC Euler: roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);
console.log(`Expected: roll≈10.4°, pitch≈0°, yaw≈-90°`);

console.log('\n' + '='.repeat(70));
