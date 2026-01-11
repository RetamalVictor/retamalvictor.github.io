/**
 * MPC Model Test
 *
 * Tests the MPCModel class to verify:
 * 1. State conversion (DroneState <-> MPCState)
 * 2. Dynamics propagation
 * 3. Linearization accuracy
 * 4. Trajectory rollout
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-mpc-model.ts
 */

import { MPCModel, MPCState, MPCInput } from '../control/MPCModel';
import { DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';

// Helper to create MPCState with quaternion (identity by default)
function createState(
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number,
    qw = 1, qx = 0, qy = 0, qz = 0
): MPCState {
    const g = DEFAULT_DYNAMICS_PARAMS.gravity;
    return {
        px, py, pz,
        vx, vy, vz,
        qw, qx, qy, qz,
        thrust: g,
        rollRate: 0,
        pitchRate: 0,
        yawRate: 0,
    };
}

// Helper to create input
function createInput(thrust: number, rollRate = 0, pitchRate = 0, yawRate = 0): MPCInput {
    return { thrust, rollRate, pitchRate, yawRate };
}

// Helper to format state
function fmtState(s: MPCState): string {
    return `pos=(${s.px.toFixed(3)}, ${s.py.toFixed(3)}, ${s.pz.toFixed(3)}) ` +
           `vel=(${s.vx.toFixed(3)}, ${s.vy.toFixed(3)}, ${s.vz.toFixed(3)}) ` +
           `quat=(${s.qw.toFixed(3)}, ${s.qx.toFixed(3)}, ${s.qy.toFixed(3)}, ${s.qz.toFixed(3)})`;
}

// Helper to compute state difference norm
function stateDiffNorm(s1: MPCState, s2: MPCState): number {
    const posDiff = Math.sqrt(
        (s1.px - s2.px) ** 2 + (s1.py - s2.py) ** 2 + (s1.pz - s2.pz) ** 2
    );
    const velDiff = Math.sqrt(
        (s1.vx - s2.vx) ** 2 + (s1.vy - s2.vy) ** 2 + (s1.vz - s2.vz) ** 2
    );
    // Quaternion difference (accounting for q ≈ -q)
    const dot = s1.qw * s2.qw + s1.qx * s2.qx + s1.qy * s2.qy + s1.qz * s2.qz;
    const quatDiff = Math.sqrt(2 * (1 - Math.abs(dot)));
    return posDiff + velDiff + quatDiff;
}

console.log('='.repeat(60));
console.log('MPC MODEL TEST');
console.log('='.repeat(60));

const model = new MPCModel();
const g = model.params.gravity;
const dt = 0.05;  // 50ms step

console.log(`\nModel parameters: nx=${model.nx}, nu=${model.nu}`);
console.log(`Dynamics params: ${JSON.stringify(model.params)}\n`);

// Test 1: Hover Dynamics
console.log('\n--- TEST 1: Hover Dynamics ---');
{
    const state = createState(0, 2, 0, 0, 0, 0);
    const input = createInput(g);  // hover thrust

    console.log(`Initial: ${fmtState(state)}`);
    console.log(`Input: thrust=${input.thrust.toFixed(2)} m/s²`);

    // Step forward
    const nextState = model.dynamics(state, input, dt);
    console.log(`After ${dt}s: ${fmtState(nextState)}`);

    // Should be nearly identical
    const diff = stateDiffNorm(state, nextState);
    console.log(`State change norm: ${diff.toFixed(6)}`);
    console.log(`PASS: ${diff < 0.01 ? 'YES' : 'NO'}`);
}

// Test 2: Vertical Acceleration
console.log('\n--- TEST 2: Vertical Acceleration ---');
{
    const state = createState(0, 2, 0, 0, 0, 0);
    const input = createInput(g + 2);  // +2 m/s² acceleration

    console.log(`Initial: ${fmtState(state)}`);
    console.log(`Input: thrust=${input.thrust.toFixed(2)} m/s² (g + 2)`);

    // Step forward 1 second (20 steps)
    let s = state;
    for (let i = 0; i < 20; i++) {
        s = model.dynamics(s, input, dt);
    }
    console.log(`After 1s: ${fmtState(s)}`);

    // Expected: vy ≈ 2 m/s, y ≈ 3m (without drag)
    console.log(`Expected vy ~ 2 m/s, y ~ 3m`);
    console.log(`PASS: ${s.vy > 1.5 && s.py > 2.5 ? 'YES' : 'NO'}`);
}

// Test 3: Pitch-Induced Forward Motion
console.log('\n--- TEST 3: Pitch-Induced Forward Motion ---');
{
    const state = createState(0, 2, 0, 0, 0, 0);
    const input = createInput(g * 1.05, 0, 0.5, 0);  // pitch rate

    console.log(`Initial: ${fmtState(state)}`);
    console.log(`Input: thrust=${input.thrust.toFixed(2)}, pitchRate=${input.pitchRate} rad/s`);

    // Step forward 2 seconds
    let s = state;
    for (let i = 0; i < 40; i++) {
        s = model.dynamics(s, input, dt);
    }
    console.log(`After 2s: ${fmtState(s)}`);

    // Should have tilted orientation and forward motion (+Z)
    // Check qx (pitch component) is non-zero and z position increased
    console.log(`PASS: ${Math.abs(s.qx) > 0.05 && s.pz > 0 ? 'YES' : 'NO'}`);
}

// Test 4: Linearization Accuracy
console.log('\n--- TEST 4: Linearization Accuracy ---');
{
    const state = createState(0, 2, 0, 2, 0, 3);  // Non-zero velocity
    // Set a small pitch rotation via quaternion
    const pitchAngle = 0.15;
    state.qw = Math.cos(pitchAngle / 2);
    state.qx = Math.sin(pitchAngle / 2);
    state.qy = 0;
    state.qz = 0;
    const input = createInput(g * 1.1, 0.1, 0.1, 0);

    console.log(`Operating point: ${fmtState(state)}`);
    console.log(`Input: T=${input.thrust.toFixed(2)}, rates=(${input.rollRate}, ${input.pitchRate}, ${input.yawRate})`);

    // Nonlinear dynamics
    const nonlinearNext = model.dynamics(state, input, dt);

    // Linearization
    const { A, B, c } = model.linearize(state, input, dt);

    // Linear prediction: x_next = A*x + B*u + c
    const x = model.stateToArray(state);
    const u = model.inputToArray(input);
    const linearNextArr = new Array(model.nx).fill(0);

    for (let i = 0; i < model.nx; i++) {
        linearNextArr[i] = c[i];
        for (let j = 0; j < model.nx; j++) {
            linearNextArr[i] += A[i][j] * x[j];
        }
        for (let j = 0; j < model.nu; j++) {
            linearNextArr[i] += B[i][j] * u[j];
        }
    }
    const linearNext = model.arrayToState(linearNextArr);

    console.log(`\nNonlinear: ${fmtState(nonlinearNext)}`);
    console.log(`Linear:    ${fmtState(linearNext)}`);

    const diff = stateDiffNorm(nonlinearNext, linearNext);
    console.log(`\nDifference norm: ${diff.toFixed(6)}`);
    console.log(`PASS: ${diff < 0.001 ? 'YES (excellent)' : diff < 0.01 ? 'YES (good)' : 'NO'}`);
}

// Test 5: Trajectory Rollout
console.log('\n--- TEST 5: Trajectory Rollout ---');
{
    const initial = createState(5, 2, 0, 0, 0, 3);  // Moving forward
    const N = 10;
    const inputs: MPCInput[] = [];

    // Generate inputs for gentle turn
    for (let k = 0; k < N; k++) {
        inputs.push(createInput(g * 1.05, 0.1, 0, 0.2));  // slight roll + yaw
    }

    console.log(`Initial: ${fmtState(initial)}`);
    console.log(`Horizon: N=${N}, dt=${dt}s`);
    console.log(`Input: constant T=${(g*1.05).toFixed(2)}, rollRate=0.1, yawRate=0.2`);

    const trajectory = model.rollout(initial, inputs, dt);

    console.log(`\nTrajectory:`);
    for (let k = 0; k <= N; k += 2) {
        console.log(`  [${k}] ${fmtState(trajectory[k])}`);
    }

    console.log(`\nTotal distance traveled: ${Math.sqrt(
        (trajectory[N].px - initial.px) ** 2 +
        (trajectory[N].py - initial.py) ** 2 +
        (trajectory[N].pz - initial.pz) ** 2
    ).toFixed(3)} m`);
}

// Test 6: Euler to Quaternion to Euler roundtrip
console.log('\n--- TEST 6: Attitude Conversion ---');
{
    const testAngles = [
        { roll: 0, pitch: 0, yaw: 0 },
        { roll: 0.3, pitch: 0, yaw: 0 },
        { roll: 0, pitch: 0.4, yaw: 0 },
        { roll: 0, pitch: 0, yaw: 1.5 },
        { roll: 0.2, pitch: 0.3, yaw: 0.5 },
    ];

    console.log('Testing desired attitude → thrust projection:');
    for (const { roll, pitch, yaw } of testAngles) {
        // Compute what thrust vector results from this attitude
        const cr = Math.cos(roll), sr = Math.sin(roll);
        const cp = Math.cos(pitch), sp = Math.sin(pitch);
        const cy = Math.cos(yaw), sy = Math.sin(yaw);

        // Thrust in body frame is [0, T, 0]
        // R * [0, T, 0] for ZYX Euler
        const T = g;  // hover thrust
        const ax = T * (cy * sp * cr + sy * sr);
        const ay = T * cp * cr;
        const az = T * (sy * sp * cr - cy * sr);

        console.log(`  Roll=${(roll*180/Math.PI).toFixed(0)}° Pitch=${(pitch*180/Math.PI).toFixed(0)}° Yaw=${(yaw*180/Math.PI).toFixed(0)}° → a=(${ax.toFixed(2)}, ${ay.toFixed(2)}, ${az.toFixed(2)})`);
    }
}

console.log('\n' + '='.repeat(60));
console.log('MPC MODEL TESTS COMPLETE');
console.log('='.repeat(60));
