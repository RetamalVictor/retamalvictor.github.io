/**
 * Jacobian Validation Test
 *
 * Validates analytical Jacobians against numerical differentiation:
 * 1. Multi-epsilon finite differences (1e-4, 1e-5, 1e-6)
 * 2. Per-block relative error analysis
 * 3. Directional derivative checks
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-jacobians.ts
 */

import { MPCModel, MPCState, MPCInput } from '../control/MPCModel';
import { DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const dt = 0.05;  // 50ms step

// Helper to create test states at various operating points
function createTestState(
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number,
    pitchAngle = 0, yawAngle = 0, rollAngle = 0
): MPCState {
    // Create quaternion from Euler angles (ZYX order: roll first, then pitch, then yaw)
    const cr = Math.cos(rollAngle / 2), sr = Math.sin(rollAngle / 2);
    const cp = Math.cos(pitchAngle / 2), sp = Math.sin(pitchAngle / 2);
    const cy = Math.cos(yawAngle / 2), sy = Math.sin(yawAngle / 2);

    return {
        px, py, pz,
        vx, vy, vz,
        qw: cr * cp * cy + sr * sp * sy,
        qx: sr * cp * cy - cr * sp * sy,
        qy: cr * sp * cy + sr * cp * sy,
        qz: cr * cp * sy - sr * sp * cy,
        thrust: g * 1.1,
        rollRate: 0.1,
        pitchRate: 0.1,
        yawRate: 0.1,
    };
}

function createTestInput(): MPCInput {
    return {
        thrust: g * 1.15,
        rollRate: 0.2,
        pitchRate: 0.15,
        yawRate: 0.1,
    };
}

// Matrix utilities
function matrixFrobeniusNorm(M: number[][]): number {
    let sum = 0;
    for (const row of M) {
        for (const val of row) {
            sum += val * val;
        }
    }
    return Math.sqrt(sum);
}

function matrixRelativeError(A: number[][], B: number[][], eps = 1e-10): number {
    const normB = matrixFrobeniusNorm(B);
    if (normB < eps) return matrixFrobeniusNorm(A);

    let sumSq = 0;
    for (let i = 0; i < A.length; i++) {
        for (let j = 0; j < A[0].length; j++) {
            sumSq += (A[i][j] - B[i][j]) ** 2;
        }
    }
    return Math.sqrt(sumSq) / normB;
}

function extractBlock(M: number[][], rowStart: number, rowEnd: number, colStart: number, colEnd: number): number[][] {
    const block: number[][] = [];
    for (let i = rowStart; i < rowEnd; i++) {
        const row: number[] = [];
        for (let j = colStart; j < colEnd; j++) {
            row.push(M[i][j]);
        }
        block.push(row);
    }
    return block;
}

// Block indices
const BLOCK_INDICES = {
    position: { row: [0, 3], col: [0, 3] },
    velocity: { row: [3, 6], col: [3, 6] },
    quaternion: { row: [6, 10], col: [6, 10] },
    actuator: { row: [10, 14], col: [10, 14] },
    // Cross blocks
    p_v: { row: [0, 3], col: [3, 6] },  // position w.r.t. velocity
    v_q: { row: [3, 6], col: [6, 10] },  // velocity w.r.t. quaternion (THE RISKY ONE)
    v_thrust: { row: [3, 6], col: [10, 11] },  // velocity w.r.t. thrust actuator
    q_rates: { row: [6, 10], col: [11, 14] },  // quaternion w.r.t. rate actuators
};

function compareBlocks(
    namePrefix: string,
    analytical: number[][],
    numerical: number[][],
    threshold = 0.02  // 2% relative error
): { name: string; relError: number; pass: boolean }[] {
    const results: { name: string; relError: number; pass: boolean }[] = [];

    for (const [blockName, indices] of Object.entries(BLOCK_INDICES)) {
        const { row, col } = indices as { row: number[]; col: number[] };
        const blockA = extractBlock(analytical, row[0], row[1], col[0], col[1]);
        const blockN = extractBlock(numerical, row[0], row[1], col[0], col[1]);

        const relError = matrixRelativeError(blockA, blockN);
        const pass = relError < threshold || matrixFrobeniusNorm(blockN) < 1e-8;

        results.push({
            name: `${namePrefix}_${blockName}`,
            relError,
            pass
        });
    }

    return results;
}

// Directional derivative test
function directionalDerivativeTest(
    model: MPCModel,
    state: MPCState,
    input: MPCInput,
    dt: number,
    A: number[][],
    B: number[][],
    numTests = 5
): { maxAError: number; maxBError: number } {
    const nx = model.nx;
    const nu = model.nu;

    let maxAError = 0;
    let maxBError = 0;

    // Generate random directions and compare A*delta_x with finite difference
    for (let t = 0; t < numTests; t++) {
        // Random state direction
        const deltaX: number[] = [];
        for (let i = 0; i < nx; i++) {
            deltaX.push((Math.random() - 0.5) * 0.01);  // Small perturbation
        }

        // Normalize quaternion part
        const qNorm = Math.sqrt(
            (state.qw + deltaX[6]) ** 2 + (state.qx + deltaX[7]) ** 2 +
            (state.qy + deltaX[8]) ** 2 + (state.qz + deltaX[9]) ** 2
        );
        deltaX[6] = state.qw + deltaX[6] / qNorm - state.qw;
        deltaX[7] = state.qx + deltaX[7] / qNorm - state.qx;
        deltaX[8] = state.qy + deltaX[8] / qNorm - state.qy;
        deltaX[9] = state.qz + deltaX[9] / qNorm - state.qz;

        // Compute A*deltaX
        const Ad: number[] = new Array(nx).fill(0);
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < nx; j++) {
                Ad[i] += A[i][j] * deltaX[j];
            }
        }

        // Compute finite difference: (F(x+eps*dx) - F(x)) / eps
        const eps = 1e-5;
        const xBase = model.stateToArray(state);
        const xPerturbed = xBase.map((v, i) => v + eps * deltaX[i]);

        // Renormalize quaternion
        const qNormP = Math.sqrt(
            xPerturbed[6] ** 2 + xPerturbed[7] ** 2 +
            xPerturbed[8] ** 2 + xPerturbed[9] ** 2
        );
        xPerturbed[6] /= qNormP;
        xPerturbed[7] /= qNormP;
        xPerturbed[8] /= qNormP;
        xPerturbed[9] /= qNormP;

        const f0 = model.stateToArray(model.dynamics(state, input, dt));
        const fP = model.stateToArray(model.dynamics(model.arrayToState(xPerturbed), input, dt));

        const fdDiff: number[] = fP.map((v, i) => (v - f0[i]) / eps);

        // Compare
        let errorNorm = 0;
        for (let i = 0; i < nx; i++) {
            errorNorm += (Ad[i] - fdDiff[i]) ** 2;
        }
        maxAError = Math.max(maxAError, Math.sqrt(errorNorm));

        // Random input direction
        const deltaU: number[] = [];
        for (let i = 0; i < nu; i++) {
            deltaU.push((Math.random() - 0.5) * 0.1);
        }

        // Compute B*deltaU
        const Bd: number[] = new Array(nx).fill(0);
        for (let i = 0; i < nx; i++) {
            for (let j = 0; j < nu; j++) {
                Bd[i] += B[i][j] * deltaU[j];
            }
        }

        // Compute finite difference for input
        const uBase = model.inputToArray(input);
        const uPerturbed = uBase.map((v, i) => v + eps * deltaU[i]);

        const fPu = model.stateToArray(model.dynamics(state, model.arrayToInput(uPerturbed), dt));
        const fdDiffU: number[] = fPu.map((v, i) => (v - f0[i]) / eps);

        let errorNormB = 0;
        for (let i = 0; i < nx; i++) {
            errorNormB += (Bd[i] - fdDiffU[i]) ** 2;
        }
        maxBError = Math.max(maxBError, Math.sqrt(errorNormB));
    }

    return { maxAError, maxBError };
}

// Multi-epsilon convergence test
function multiEpsilonTest(
    model: MPCModel,
    state: MPCState,
    input: MPCInput,
    dt: number,
    analytical: { A: number[][]; B: number[][] }
): void {
    const epsilons = [1e-4, 1e-5, 1e-6];

    console.log('\n  Multi-epsilon convergence (relative errors should decrease):');

    for (const eps of epsilons) {
        // Compute numerical with this epsilon
        const numerical = numericalLinearize(model, state, input, dt, eps);

        const relErrorA = matrixRelativeError(analytical.A, numerical.A);
        const relErrorB = matrixRelativeError(analytical.B, numerical.B);

        console.log(`    eps=${eps.toExponential(0)}: A_err=${relErrorA.toExponential(2)}, B_err=${relErrorB.toExponential(2)}`);
    }
}

function numericalLinearize(
    model: MPCModel,
    state: MPCState,
    input: MPCInput,
    dt: number,
    eps: number
): { A: number[][]; B: number[][] } {
    const nx = model.nx;
    const nu = model.nu;

    const f0 = model.stateToArray(model.dynamics(state, input, dt));

    // A matrix
    const A: number[][] = [];
    for (let i = 0; i < nx; i++) {
        A.push(new Array(nx).fill(0));
    }

    for (let j = 0; j < nx; j++) {
        const stateArray = model.stateToArray(state);
        stateArray[j] += eps;

        // Renormalize quaternion if perturbed
        if (j >= 6 && j <= 9) {
            const qNorm = Math.sqrt(
                stateArray[6] ** 2 + stateArray[7] ** 2 +
                stateArray[8] ** 2 + stateArray[9] ** 2
            );
            stateArray[6] /= qNorm;
            stateArray[7] /= qNorm;
            stateArray[8] /= qNorm;
            stateArray[9] /= qNorm;
        }

        const fP = model.stateToArray(model.dynamics(model.arrayToState(stateArray), input, dt));
        for (let i = 0; i < nx; i++) {
            A[i][j] = (fP[i] - f0[i]) / eps;
        }
    }

    // B matrix
    const B: number[][] = [];
    for (let i = 0; i < nx; i++) {
        B.push(new Array(nu).fill(0));
    }

    for (let j = 0; j < nu; j++) {
        const inputArray = model.inputToArray(input);
        inputArray[j] += eps;

        const fP = model.stateToArray(model.dynamics(state, model.arrayToInput(inputArray), dt));
        for (let i = 0; i < nx; i++) {
            B[i][j] = (fP[i] - f0[i]) / eps;
        }
    }

    return { A, B };
}

// ============================================================
// MAIN TEST
// ============================================================

console.log('='.repeat(70));
console.log('ANALYTICAL JACOBIAN VALIDATION TEST');
console.log('='.repeat(70));

const model = new MPCModel();

// Test operating points
const testPoints = [
    { name: 'Hover (identity quaternion)', state: createTestState(0, 2, 0, 0, 0, 0) },
    { name: 'Forward flight', state: createTestState(0, 2, 0, 5, 0, 8, 0.1, 0, 0) },
    { name: 'Banked turn', state: createTestState(5, 3, 10, 3, 0, 5, 0.15, 0.2, 0.3) },
    { name: 'Aggressive maneuver', state: createTestState(0, 5, 0, 0, 2, 10, 0.4, 0.3, 0.2) },
];

let allPassed = true;
const input = createTestInput();

for (const { name, state } of testPoints) {
    console.log(`\n${'─'.repeat(70)}`);
    console.log(`Operating Point: ${name}`);
    console.log(`  State: pos=(${state.px.toFixed(1)}, ${state.py.toFixed(1)}, ${state.pz.toFixed(1)})`);
    console.log(`         vel=(${state.vx.toFixed(1)}, ${state.vy.toFixed(1)}, ${state.vz.toFixed(1)})`);
    console.log(`         quat=(${state.qw.toFixed(3)}, ${state.qx.toFixed(3)}, ${state.qy.toFixed(3)}, ${state.qz.toFixed(3)})`);
    console.log(`  Input: T=${input.thrust.toFixed(2)}, rates=(${input.rollRate}, ${input.pitchRate}, ${input.yawRate})`);

    // Get analytical and numerical Jacobians
    const analytical = model.linearizeAnalytical(state, input, dt);
    const numerical = model.linearize(state, input, dt);  // Uses eps=1e-6

    // Overall matrix comparison
    const relErrorA = matrixRelativeError(analytical.A, numerical.A);
    const relErrorB = matrixRelativeError(analytical.B, numerical.B);

    console.log(`\n  Overall relative errors:`);
    console.log(`    A matrix: ${(relErrorA * 100).toFixed(4)}%`);
    console.log(`    B matrix: ${(relErrorB * 100).toFixed(4)}%`);

    // Per-block comparison
    console.log(`\n  Per-block relative errors (threshold: 2%):`);
    const blockResults = compareBlocks('A', analytical.A, numerical.A);

    let pointPassed = true;
    for (const result of blockResults) {
        const status = result.pass ? '✓' : '✗';
        console.log(`    ${status} ${result.name}: ${(result.relError * 100).toFixed(4)}%`);
        if (!result.pass) {
            pointPassed = false;
            allPassed = false;
        }
    }

    // Directional derivative test
    console.log(`\n  Directional derivative tests:`);
    const dirTest = directionalDerivativeTest(model, state, input, dt, analytical.A, analytical.B);
    const dirPassA = dirTest.maxAError < 0.001;
    const dirPassB = dirTest.maxBError < 0.002;  // Relaxed threshold for B
    console.log(`    ${dirPassA ? '✓' : '✗'} A·δx max error: ${dirTest.maxAError.toExponential(3)}`);
    console.log(`    ${dirPassB ? '✓' : '✗'} B·δu max error: ${dirTest.maxBError.toExponential(3)}`);

    if (!dirPassA || !dirPassB) {
        pointPassed = false;
        allPassed = false;
    }

    // Multi-epsilon convergence
    multiEpsilonTest(model, state, input, dt, analytical);

    console.log(`\n  Point result: ${pointPassed ? 'PASS' : 'FAIL'}`);
}

// Performance comparison
console.log(`\n${'='.repeat(70)}`);
console.log('PERFORMANCE COMPARISON');
console.log('='.repeat(70));

const benchState = createTestState(0, 2, 0, 5, 0, 8, 0.15, 0.1, 0.05);
const benchInput = createTestInput();
const iterations = 1000;

// Warm up
for (let i = 0; i < 100; i++) {
    model.linearize(benchState, benchInput, dt);
    model.linearizeAnalytical(benchState, benchInput, dt);
}

// Benchmark numerical
const startNum = performance.now();
for (let i = 0; i < iterations; i++) {
    model.linearize(benchState, benchInput, dt);
}
const endNum = performance.now();
const numTime = endNum - startNum;

// Benchmark analytical
const startAna = performance.now();
for (let i = 0; i < iterations; i++) {
    model.linearizeAnalytical(benchState, benchInput, dt);
}
const endAna = performance.now();
const anaTime = endAna - startAna;

console.log(`\n${iterations} linearizations:`);
console.log(`  Numerical:  ${numTime.toFixed(2)} ms (${(numTime / iterations).toFixed(3)} ms/call)`);
console.log(`  Analytical: ${anaTime.toFixed(2)} ms (${(anaTime / iterations).toFixed(3)} ms/call)`);
console.log(`  Speedup:    ${(numTime / anaTime).toFixed(2)}x`);

console.log(`\n${'='.repeat(70)}`);
console.log(`FINAL RESULT: ${allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED'}`);
console.log('='.repeat(70));

process.exit(allPassed ? 0 : 1);
