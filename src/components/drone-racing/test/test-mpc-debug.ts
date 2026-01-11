/**
 * MPC Debug Test
 *
 * Step-by-step debugging of the MPC to find where it goes wrong.
 *
 * Run with: npx tsx src/components/drone-racing/test/test-mpc-debug.ts
 */

import { MPCModel, MPCState, MPCInput } from '../control/MPCModel';
import { QPSolver, MatrixUtils } from '../control/QPSolver';
import { DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const model = new MPCModel();

console.log('='.repeat(70));
console.log('MPC DEBUG TEST');
console.log('='.repeat(70));

// Simple hover case: drone slightly above target
const x0: MPCState = {
    px: 0, py: 2.5, pz: 0,  // 0.5m above target
    vx: 0, vy: 0, vz: 0,
    roll: 0, pitch: 0, yaw: 0,
    thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0,
};

// Target: hover at y=2
const xRef: MPCState = {
    px: 0, py: 2, pz: 0,
    vx: 0, vy: 0, vz: 0,
    roll: 0, pitch: 0, yaw: 0,
    thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0,
};

const uRef: MPCInput = {
    thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0,
};

console.log('\n--- Step 1: Check Initial State ---');
console.log(`x0: py=${x0.py}, vy=${x0.vy}`);
console.log(`xRef: py=${xRef.py}, vy=${xRef.vy}`);
console.log(`Error: py_err = ${x0.py - xRef.py} (above target, need to descend)`);

console.log('\n--- Step 2: Single Dynamics Step ---');
{
    // What happens with hover thrust?
    const uHover: MPCInput = { thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0 };
    const x1_hover = model.dynamics(x0, uHover, 0.05);
    console.log(`With hover thrust (${g.toFixed(2)}): py=${x1_hover.py.toFixed(4)}, vy=${x1_hover.vy.toFixed(4)}`);

    // What about less thrust?
    const uLess: MPCInput = { thrust: g - 2, rollRate: 0, pitchRate: 0, yawRate: 0 };
    const x1_less = model.dynamics(x0, uLess, 0.05);
    console.log(`With reduced thrust (${(g-2).toFixed(2)}): py=${x1_less.py.toFixed(4)}, vy=${x1_less.vy.toFixed(4)}`);

    // What about more thrust?
    const uMore: MPCInput = { thrust: g + 2, rollRate: 0, pitchRate: 0, yawRate: 0 };
    const x1_more = model.dynamics(x0, uMore, 0.05);
    console.log(`With increased thrust (${(g+2).toFixed(2)}): py=${x1_more.py.toFixed(4)}, vy=${x1_more.vy.toFixed(4)}`);
}

console.log('\n--- Step 3: Linearization Check ---');
{
    const { A, B, c } = model.linearize(x0, uRef, 0.05);

    console.log('State Jacobian A (selected entries):');
    console.log(`  A[py][py] = ${A[1][1].toFixed(4)} (should be ~1)`);
    console.log(`  A[py][vy] = ${A[1][4].toFixed(4)} (should be ~dt = 0.05)`);
    console.log(`  A[vy][py] = ${A[4][1].toFixed(4)} (should be ~0)`);
    console.log(`  A[vy][vy] = ${A[4][4].toFixed(4)} (should be ~1-drag*dt)`);

    console.log('\nInput Jacobian B (selected entries):');
    console.log(`  B[py][thrust] = ${B[1][0].toFixed(6)} (should be small)`);
    console.log(`  B[vy][thrust] = ${B[4][0].toFixed(6)} (should be ~dt/mass = 0.05)`);

    console.log('\nAffine term c (selected):');
    console.log(`  c[py] = ${c[1].toFixed(6)}`);
    console.log(`  c[vy] = ${c[4].toFixed(6)}`);
}

console.log('\n--- Step 4: Simple QP Test ---');
{
    // Test the QP solver with a simple problem
    // min 0.5 * x^T * H * x + g^T * x
    // s.t. -1 <= x <= 1

    const H = [[2, 0], [0, 2]];  // Identity * 2
    const gVec = [1, -1];  // Push toward (-0.5, 0.5)
    const lb = [-1, -1];
    const ub = [1, 1];

    const solution = QPSolver.solve({ H, g: gVec, lb, ub });
    console.log(`Simple QP test:`);
    console.log(`  Solution: [${solution.x.map(v => v.toFixed(4)).join(', ')}]`);
    console.log(`  Expected: [-0.5, 0.5]`);
    console.log(`  PASS: ${Math.abs(solution.x[0] + 0.5) < 0.01 && Math.abs(solution.x[1] - 0.5) < 0.01 ? 'YES' : 'NO'}`);
}

console.log('\n--- Step 5: Cost Function Analysis ---');
{
    // What does the cost look like for different inputs?
    const N = 3;  // Short horizon
    const dt = 0.05;

    // Define cost weights
    const Qpos = 10;
    const Qvel = 5;
    const R = 0.1;

    console.log('Cost for different thrust values (starting from x0):');

    for (const thrustOffset of [-4, -2, 0, 2, 4]) {
        const thrust = g + thrustOffset;
        const input: MPCInput = { thrust, rollRate: 0, pitchRate: 0, yawRate: 0 };

        // Rollout
        let state = x0;
        let totalCost = 0;

        for (let k = 0; k < N; k++) {
            // State cost
            const posErr = (state.py - xRef.py) ** 2;
            const velErr = state.vy ** 2;
            totalCost += Qpos * posErr + Qvel * velErr;

            // Input cost
            const thrustErr = (thrust - g) ** 2;
            totalCost += R * thrustErr;

            state = model.dynamics(state, input, dt);
        }

        // Terminal cost
        const termPosErr = (state.py - xRef.py) ** 2;
        const termVelErr = state.vy ** 2;
        totalCost += 10 * (Qpos * termPosErr + Qvel * termVelErr);

        console.log(`  T=${thrust.toFixed(1)}: cost=${totalCost.toFixed(2)}, final_py=${state.py.toFixed(3)}, final_vy=${state.vy.toFixed(3)}`);
    }

    console.log('\nExpected: Lower thrust should have lower cost (to descend to y=2)');
}

console.log('\n--- Step 6: Manual MPC Step ---');
{
    // Implement a simple 1-step MPC manually to debug
    const N = 1;  // Just 1 step
    const dt = 0.05;

    // Linearize at current state
    const { A, B, c } = model.linearize(x0, uRef, dt);

    // State error: x0 - xRef
    const x0Arr = model.stateToArray(x0);
    const xRefArr = model.stateToArray(xRef);
    const xErr = x0Arr.map((x, i) => x - xRefArr[i]);

    console.log('Initial state error (selected):');
    console.log(`  py_err = ${xErr[1].toFixed(4)}`);
    console.log(`  vy_err = ${xErr[4].toFixed(4)}`);

    // For 1-step MPC with condensed formulation:
    // x1 = A*x0 + B*u + c
    // Cost = (x1 - xRef)^T Q (x1 - xRef) + u^T R u
    //      = (A*x0 + B*u + c - xRef)^T Q (A*x0 + B*u + c - xRef) + u^T R u
    // Let d = A*x0 + c - xRef (constant term)
    // Cost = (d + B*u)^T Q (d + B*u) + u^T R u
    //      = d^T Q d + 2 d^T Q B u + u^T (B^T Q B + R) u
    // H = B^T Q B + R
    // g = B^T Q d

    // Build Q (diagonal for position and velocity)
    const Q = MatrixUtils.zeros(model.nx, model.nx);
    Q[0][0] = Q[1][1] = Q[2][2] = 10;  // position
    Q[3][3] = Q[4][4] = Q[5][5] = 5;   // velocity
    Q[6][6] = Q[7][7] = Q[8][8] = 1;   // attitude

    // Build R
    const R = MatrixUtils.diag([0.1, 0.1, 0.1, 0.1]);

    // Compute d = A*x0 + c - xRef
    const d = new Array(model.nx).fill(0);
    for (let i = 0; i < model.nx; i++) {
        d[i] = c[i] - xRefArr[i];
        for (let j = 0; j < model.nx; j++) {
            d[i] += A[i][j] * x0Arr[j];
        }
    }

    console.log('\nConstant term d (selected):');
    console.log(`  d[py] = ${d[1].toFixed(6)} (predicted py - target py)`);
    console.log(`  d[vy] = ${d[4].toFixed(6)} (predicted vy)`);

    // Compute H = B^T Q B + R
    const BT = MatrixUtils.transpose(B);
    const QB = MatrixUtils.matMul(Q, B);
    const BTQB = MatrixUtils.matMul(BT, QB);
    const H = MatrixUtils.add(BTQB, R);

    // Compute g = B^T Q d
    const Qd = new Array(model.nx).fill(0);
    for (let i = 0; i < model.nx; i++) {
        for (let j = 0; j < model.nx; j++) {
            Qd[i] += Q[i][j] * d[j];
        }
    }
    const gVec = new Array(model.nu).fill(0);
    for (let i = 0; i < model.nu; i++) {
        for (let j = 0; j < model.nx; j++) {
            gVec[i] += BT[i][j] * Qd[j];
        }
    }

    console.log('\nQP gradient (g):');
    console.log(`  g[thrust] = ${gVec[0].toFixed(6)}`);
    console.log(`  g[rollRate] = ${gVec[1].toFixed(6)}`);

    // Bounds (relative to reference)
    const lb = [2 - g, -2, -2, -1.5];  // u_min - u_ref
    const ub = [18 - g, 2, 2, 1.5];    // u_max - u_ref

    // Solve
    const solution = QPSolver.solve({ H, g: gVec, lb, ub }, { maxIterations: 100 });

    console.log('\nQP Solution (delta from reference):');
    console.log(`  dThrust = ${solution.x[0].toFixed(4)}`);
    console.log(`  dRollRate = ${solution.x[1].toFixed(4)}`);
    console.log(`  dPitchRate = ${solution.x[2].toFixed(4)}`);
    console.log(`  dYawRate = ${solution.x[3].toFixed(4)}`);

    const optimalThrust = g + solution.x[0];
    console.log(`\nOptimal thrust: ${optimalThrust.toFixed(4)} (expected < 9.81 to descend)`);
    console.log(`Converged: ${solution.converged}`);
}

console.log('\n' + '='.repeat(70));
console.log('DEBUG COMPLETE');
console.log('='.repeat(70));
