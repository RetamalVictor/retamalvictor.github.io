/**
 * Test Euler conversion during circular trajectory
 */

import { MPC, DEFAULT_MPC_CONFIG } from '../control/MPC';
import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { MPCModel } from '../control/MPCModel';
import { Waypoint } from '../types';

console.log('='.repeat(70));
console.log('EULER CONVERSION DURING CIRCULAR TRAJECTORY');
console.log('='.repeat(70));

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const mpc = new MPC();
const dynamics = new DroneDynamics();
const model = new MPCModel();

const radius = 5;
const speed = 3;
const height = 2;
const omega = speed / radius;
const centripetalAccel = speed * speed / radius;

function createWaypoint(
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number,
    ax: number, ay: number, az: number,
    heading: number, headingRate: number,
    time: number
): Waypoint {
    return {
        position: { x: px, y: py, z: pz },
        velocity: { x: vx, y: vy, z: vz },
        acceleration: { x: ax, y: ay, z: az },
        jerk: { x: 0, y: 0, z: 0 },
        heading,
        headingRate,
        time,
    };
}

const getReference = (t: number): Waypoint => {
    const angle = omega * t;
    const heading = -angle;
    return createWaypoint(
        radius * Math.cos(angle), height, radius * Math.sin(angle),
        -speed * Math.sin(angle), 0, speed * Math.cos(angle),
        -centripetalAccel * Math.cos(angle), 0, -centripetalAccel * Math.sin(angle),
        heading, -omega, t
    );
};

// Start on the circle with correct velocity
const start = getReference(0);
dynamics.setPosition(start.position.x, start.position.y, start.position.z);
dynamics.setVelocity(start.velocity.x, start.velocity.y, start.velocity.z);
dynamics.setHeading(start.heading);

// Simulate to t=2.55s using MPC
let simTime = 0;
while (simTime < 2.55) {
    const state = dynamics.getState();
    const cmd = mpc.computeControl(state, getReference, simTime);
    dynamics.step(cmd, 0.02);
    simTime += 0.02;
}

// Now examine the state at t=2.55s
const droneState = dynamics.getState();
const fullState = dynamics.getFullState();

console.log('\n--- State at t=2.55s ---');
console.log(`\nPosition: (${droneState.position.x.toFixed(3)}, ${droneState.position.y.toFixed(3)}, ${droneState.position.z.toFixed(3)})`);
console.log(`Velocity: (${droneState.velocity.x.toFixed(3)}, ${droneState.velocity.y.toFixed(3)}, ${droneState.velocity.z.toFixed(3)})`);

console.log(`\nQuaternion: w=${droneState.orientation.w.toFixed(4)}, x=${droneState.orientation.x.toFixed(4)}, y=${droneState.orientation.y.toFixed(4)}, z=${droneState.orientation.z.toFixed(4)}`);
console.log(`Angular velocity: (${fullState.angularVelocity.x.toFixed(4)}, ${fullState.angularVelocity.y.toFixed(4)}, ${fullState.angularVelocity.z.toFixed(4)})`);
console.log(`Current rates: (${fullState.currentRates.x.toFixed(4)}, ${fullState.currentRates.y.toFixed(4)}, ${fullState.currentRates.z.toFixed(4)})`);

// Convert to MPC state
const mpcState = model.fromDroneState(droneState.position, droneState.velocity, droneState.orientation);

console.log(`\nMPC State Euler angles:`);
console.log(`  roll: ${(mpcState.roll * 180 / Math.PI).toFixed(1)}°`);
console.log(`  pitch: ${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°`);
console.log(`  yaw: ${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);

// What should the reference be at t=2.60s (with command delay)?
const refTime = 2.55 + DEFAULT_MPC_CONFIG.commandDelay;
const ref = getReference(refTime);
const angle = omega * refTime;

// Compute expected attitude from acceleration
const ax = ref.acceleration.x;
const ay = ref.acceleration.y + g;
const az = ref.acceleration.z;

const yaw = ref.heading;
const cy = Math.cos(yaw);
const sy = Math.sin(yaw);
const axBody = ax * cy - az * sy;
const ayBody = ay;
const azBody = ax * sy + az * cy;

const expectedRoll = -Math.atan2(axBody, ayBody);
const expectedPitch = Math.atan2(azBody, ayBody);

console.log(`\n--- Reference at t=${refTime.toFixed(2)}s ---`);
console.log(`Angle: ${(angle * 180 / Math.PI).toFixed(1)}°`);
console.log(`Heading: ${(ref.heading * 180 / Math.PI).toFixed(1)}°`);
console.log(`Expected attitude:`);
console.log(`  roll: ${(expectedRoll * 180 / Math.PI).toFixed(1)}°`);
console.log(`  pitch: ${(expectedPitch * 180 / Math.PI).toFixed(1)}°`);

console.log(`\n--- Residual (MPC state - reference) ---`);
console.log(`  roll diff: ${((mpcState.roll - expectedRoll) * 180 / Math.PI).toFixed(1)}°`);
console.log(`  pitch diff: ${((mpcState.pitch - expectedPitch) * 180 / Math.PI).toFixed(1)}°`);
console.log(`  yaw diff: ${((mpcState.yaw - ref.heading) * 180 / Math.PI).toFixed(1)}°`);

// Also check what the quaternion-to-Euler should give for a pure yaw rotation
console.log('\n--- Verify Euler extraction ---');
const testHeading = -89 * Math.PI / 180;  // -89°
const testQuat = {
    w: Math.cos(testHeading / 2),
    x: 0,
    y: Math.sin(testHeading / 2),
    z: 0,
};
const testMpc = model.fromDroneState({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, testQuat);
console.log(`Pure yaw=-89° quaternion: w=${testQuat.w.toFixed(4)}, x=${testQuat.x.toFixed(4)}, y=${testQuat.y.toFixed(4)}, z=${testQuat.z.toFixed(4)}`);
console.log(`Extracted: roll=${(testMpc.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(testMpc.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(testMpc.yaw * 180 / Math.PI).toFixed(1)}°`);

// Check what the actual drone quaternion represents
console.log('\n--- Decompose actual quaternion ---');
// The drone quaternion should be roughly: Yaw(heading) * Pitch * Roll
// Let's extract the yaw component and see what's left
const q = droneState.orientation;
const yawFromQuat = 2 * Math.atan2(q.y, q.w);  // Assuming small pitch/roll, yaw ≈ 2*atan2(qy, qw)
console.log(`Estimated yaw from quaternion: ${(yawFromQuat * 180 / Math.PI).toFixed(1)}°`);
console.log(`Expected yaw (circle trajectory): ${(-omega * 2.55 * 180 / Math.PI).toFixed(1)}°`);

// Check the quaternion norm
const qNorm = Math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
console.log(`Quaternion norm: ${qNorm.toFixed(6)} (should be 1)`);

// Check if there's significant roll/pitch in the quaternion
// For a yaw-only rotation: q = [cos(yaw/2), 0, sin(yaw/2), 0]
// The difference from this indicates roll/pitch
const pureYawQuat = {
    w: Math.cos(yawFromQuat / 2),
    x: 0,
    y: Math.sin(yawFromQuat / 2),
    z: 0,
};
const diffX = Math.abs(q.x - pureYawQuat.x);
const diffZ = Math.abs(q.z - pureYawQuat.z);
console.log(`Deviation from pure yaw: x=${diffX.toFixed(4)}, z=${diffZ.toFixed(4)}`);
console.log(`(Non-zero indicates actual roll/pitch tilt)`);

console.log('\n' + '='.repeat(70));
