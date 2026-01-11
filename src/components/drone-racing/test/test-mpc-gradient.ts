/**
 * Debug test to trace the gradient computation in MPC
 */

import { MPCModel, MPCState, MPCInput } from '../control/MPCModel';
import { QPSolver, MatrixUtils } from '../control/QPSolver';

console.log('='.repeat(70));
console.log('MPC GRADIENT DEBUG');
console.log('='.repeat(70));

const model = new MPCModel();
const N = 3;  // Short horizon for debugging
const nx = 13;
const nu = 4;
const dt = 0.05;
const g = model.params.gravity;

// Weights
const Q = MatrixUtils.zeros(nx, nx);
Q[0][0] = Q[1][1] = Q[2][2] = 400;  // position
Q[3][3] = Q[4][4] = Q[5][5] = 1;    // velocity
Q[6][6] = Q[7][7] = Q[8][8] = 10;   // attitude

const R = MatrixUtils.zeros(nu, nu);
R[0][0] = 0.5;   // thrust
R[1][1] = R[2][2] = R[3][3] = 1;  // rates

// Simple scenario: drone needs to reduce roll from 30° to 10°
console.log('\n--- Scenario: Reduce roll from 30° to 10° ---');

const currentRoll = 30 * Math.PI / 180;  // 30°
const targetRoll = 10 * Math.PI / 180;   // 10°

// Current state (drone at 30° roll)
const x0: MPCState = {
    px: 0, py: 2, pz: 0,
    vx: 0, vy: 0, vz: 0,
    roll: currentRoll, pitch: 0, yaw: 0,
    thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0,
};

// Reference states (drone at 10° roll)
const refStates: MPCState[] = [];
for (let k = 0; k <= N; k++) {
    refStates.push({
        px: 0, py: 2, pz: 0,
        vx: 0, vy: 0, vz: 0,
        roll: targetRoll, pitch: 0, yaw: 0,
        thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0,
    });
}

// Reference inputs (hover)
const refInputs: MPCInput[] = [];
for (let k = 0; k < N; k++) {
    refInputs.push({ thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0 });
}

// Cold start: rollout from x0 using reference inputs
const nominalStates = model.rollout(x0, refInputs, dt);
const nominalInputs = [...refInputs];

console.log('\nNominal states (first 3):');
for (let k = 0; k <= 2; k++) {
    const ns = nominalStates[k];
    console.log(`  k=${k}: roll=${(ns.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(ns.pitch * 180 / Math.PI).toFixed(1)}°`);
}

console.log('\nReference states:');
for (let k = 0; k <= 2; k++) {
    const rs = refStates[k];
    console.log(`  k=${k}: roll=${(rs.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rs.pitch * 180 / Math.PI).toFixed(1)}°`);
}

// Compute residuals
console.log('\nResiduals (nom - ref):');
for (let k = 0; k <= N; k++) {
    const residual = (nominalStates[k].roll - refStates[k].roll) * 180 / Math.PI;
    console.log(`  k=${k}: roll residual = ${residual.toFixed(1)}°`);
}

// Linearize dynamics
console.log('\nLinearization at k=0:');
const lin0 = model.linearize(nominalStates[0], nominalInputs[0], dt);
console.log('  B[6][0:4] (roll row):');
console.log(`    [${lin0.B[6].map(v => v.toFixed(6)).join(', ')}]`);
console.log('  Expected: B[6][1] > 0 (positive rollRate increases roll)');

// Compute Psi matrices
const nz = N * nu;
const Psi: number[][][] = [];

Psi.push(MatrixUtils.zeros(nx, nz));  // Psi[0] = 0

let PhiK = MatrixUtils.eye(nx);

for (let k = 0; k < N; k++) {
    const lin = model.linearize(nominalStates[k], nominalInputs[k], dt);

    // Psi_{k+1} = A_k * Psi_k + [0...0, B_k, 0...0]
    const APsi = k > 0 ? MatrixUtils.matMul(lin.A, Psi[k]) : MatrixUtils.zeros(nx, nz);
    const PsiNext = MatrixUtils.zeros(nx, nz);

    for (let i = 0; i < nx; i++) {
        for (let j = 0; j < nz; j++) {
            PsiNext[i][j] = APsi[i][j];
        }
        for (let j = 0; j < nu; j++) {
            PsiNext[i][k * nu + j] += lin.B[i][j];
        }
    }

    Psi.push(PsiNext);
    PhiK = MatrixUtils.matMul(lin.A, PhiK);
}

console.log('\nPsi[1][6][0:4] (effect on roll at k=1 from inputs at k=0):');
console.log(`  [${Psi[1][6].slice(0, 4).map(v => v.toFixed(6)).join(', ')}]`);
console.log('  Expected: Psi[1][6][1] > 0 (positive dU[rollRate@0] increases roll at k=1)');

// Compute gradient
const gradient = new Array(nz).fill(0);

for (let k = 0; k <= N; k++) {
    const Qk = k < N ? Q : Q;  // Same for terminal
    const PsiK = Psi[k];

    const xNomK = model.stateToArray(nominalStates[k]);
    const xRefK = model.stateToArray(refStates[k]);
    const residual = xNomK.map((x, i) => x - xRefK[i]);

    // g += Psi_k^T * Q * residual
    for (let i = 0; i < nz; i++) {
        for (let a = 0; a < nx; a++) {
            let QRes = 0;
            for (let b = 0; b < nx; b++) {
                QRes += Qk[a][b] * residual[b];
            }
            gradient[i] += PsiK[a][i] * QRes;
        }
    }
}

// Add input reference cost
for (let k = 0; k < N; k++) {
    const uRef = model.inputToArray(refInputs[k]);
    const uNom = model.inputToArray(nominalInputs[k]);
    for (let i = 0; i < nu; i++) {
        let RDu = 0;
        for (let j = 0; j < nu; j++) {
            RDu += R[i][j] * (uNom[j] - uRef[j]);
        }
        gradient[k * nu + i] += RDu;
    }
}

console.log('\nGradient (first 4 = inputs at k=0):');
console.log(`  g[thrust@0] = ${gradient[0].toFixed(4)}`);
console.log(`  g[rollRate@0] = ${gradient[1].toFixed(4)}`);
console.log(`  g[pitchRate@0] = ${gradient[2].toFixed(4)}`);
console.log(`  g[yawRate@0] = ${gradient[3].toFixed(4)}`);
console.log('\nExpected: g[rollRate@0] > 0 (so QP should decrease rollRate)');

// Solve QP
const H = MatrixUtils.zeros(nz, nz);

// R_bar contribution
for (let k = 0; k < N; k++) {
    for (let i = 0; i < nu; i++) {
        H[k * nu + i][k * nu + i] += R[i][i];
    }
}

// Psi^T * Q * Psi contribution
for (let k = 0; k <= N; k++) {
    const Qk = k < N ? Q : Q;
    const PsiK = Psi[k];

    for (let i = 0; i < nz; i++) {
        for (let j = 0; j < nz; j++) {
            for (let a = 0; a < nx; a++) {
                for (let b = 0; b < nx; b++) {
                    H[i][j] += PsiK[a][i] * Qk[a][b] * PsiK[b][j];
                }
            }
        }
    }
}

// Bounds (large for now)
const lb = new Array(nz).fill(-100);
const ub = new Array(nz).fill(100);

const solution = QPSolver.solve({ H, g: gradient, lb, ub }, {
    maxIterations: 100,
    tolerance: 1e-8,
});

console.log('\nQP Solution (first 4 = dU at k=0):');
console.log(`  dU[thrust@0] = ${solution.x[0].toFixed(4)}`);
console.log(`  dU[rollRate@0] = ${solution.x[1].toFixed(4)}`);
console.log(`  dU[pitchRate@0] = ${solution.x[2].toFixed(4)}`);
console.log(`  dU[yawRate@0] = ${solution.x[3].toFixed(4)}`);
console.log(`  Converged: ${solution.converged}, Iterations: ${solution.iterations}`);
console.log('\nExpected: dU[rollRate@0] < 0 (to reduce roll from 30° to 10°)');

// Final command
const finalRollRate = refInputs[0].rollRate + solution.x[1];
console.log(`\nFinal rollRate command: ${finalRollRate.toFixed(4)} rad/s = ${(finalRollRate * 180 / Math.PI).toFixed(1)}°/s`);
console.log('Expected: negative (to reduce roll)');

// Verify: simulate one step with this rollRate
const testInput: MPCInput = {
    thrust: refInputs[0].thrust + solution.x[0],
    rollRate: finalRollRate,
    pitchRate: refInputs[0].pitchRate + solution.x[2],
    yawRate: refInputs[0].yawRate + solution.x[3],
};
const nextState = model.dynamics(x0, testInput, dt);
console.log(`\nAfter one step with this command:`);
console.log(`  Roll: ${(x0.roll * 180 / Math.PI).toFixed(1)}° -> ${(nextState.roll * 180 / Math.PI).toFixed(1)}°`);
console.log(`  Expected: roll decreases toward 10°`);

console.log('\n' + '='.repeat(70));
