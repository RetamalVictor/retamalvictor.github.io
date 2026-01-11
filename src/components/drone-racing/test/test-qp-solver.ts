/**
 * QP Solver Verification Test
 *
 * Tests the QP solver with known problems where we can verify the answer.
 */

import { QPSolver, MatrixUtils } from '../control/QPSolver';

console.log('='.repeat(70));
console.log('QP SOLVER VERIFICATION');
console.log('='.repeat(70));

// Test 1: Simple 2D unconstrained quadratic
// min 0.5 * (x^2 + y^2) + 2*x + 3*y
// Optimal: x = -2, y = -3
console.log('\n--- Test 1: Simple 2D unconstrained ---');
{
    const H = [[1, 0], [0, 1]];
    const g = [2, 3];

    const result = QPSolver.solve({ H, g }, { maxIterations: 100, tolerance: 1e-8 });

    console.log(`Solution: x=[${result.x.map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`Expected: x=[-2.000000, -3.000000]`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${Math.abs(result.x[0] + 2) < 0.01 && Math.abs(result.x[1] + 3) < 0.01 ? 'YES' : 'NO'}`);
}

// Test 2: 2D with box constraints
// min 0.5 * (x^2 + y^2) + 2*x + 3*y
// s.t. -1 <= x <= 1, -1 <= y <= 1
// Optimal: x = -1, y = -1 (at boundary)
console.log('\n--- Test 2: 2D with box constraints ---');
{
    const H = [[1, 0], [0, 1]];
    const g = [2, 3];
    const lb = [-1, -1];
    const ub = [1, 1];

    const result = QPSolver.solve({ H, g, lb, ub }, { maxIterations: 100, tolerance: 1e-8 });

    console.log(`Solution: x=[${result.x.map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`Expected: x=[-1.000000, -1.000000]`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${Math.abs(result.x[0] + 1) < 0.01 && Math.abs(result.x[1] + 1) < 0.01 ? 'YES' : 'NO'}`);
}

// Test 3: Asymmetric Hessian (cross terms)
// min 0.5 * (2*x^2 + 2*x*y + y^2) + x - y
// H = [[2, 1], [1, 1]], g = [1, -1]
// Optimal: solve H*x = -g => [2,1;1,1]*x = [-1,1]
// x = H^{-1} * [-1,1] = [1,-1;-1,2] * [-1,1] = [-2, 3]
console.log('\n--- Test 3: Asymmetric (cross terms) ---');
{
    const H = [[2, 1], [1, 1]];
    const g = [1, -1];

    const result = QPSolver.solve({ H, g }, { maxIterations: 200, tolerance: 1e-8 });

    console.log(`Solution: x=[${result.x.map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`Expected: x=[-2.000000, 3.000000]`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${Math.abs(result.x[0] + 2) < 0.01 && Math.abs(result.x[1] - 3) < 0.01 ? 'YES' : 'NO'}`);
}

// Test 4: Larger problem (similar to MPC: 60 variables = 15 steps * 4 inputs)
// min 0.5 * sum_i (x_i^2) + sum_i i*x_i
// Optimal: x_i = -i for all i
console.log('\n--- Test 4: Larger problem (60 variables) ---');
{
    const n = 60;
    const H = MatrixUtils.eye(n);
    const g = Array.from({ length: n }, (_, i) => i + 1);

    const result = QPSolver.solve({ H, g }, { maxIterations: 200, tolerance: 1e-6 });

    // Check first and last few elements
    const expected = g.map(v => -v);
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
        maxErr = Math.max(maxErr, Math.abs(result.x[i] - expected[i]));
    }

    console.log(`First 5: [${result.x.slice(0, 5).map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`Expected: [${expected.slice(0, 5).map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`Max error: ${maxErr.toFixed(6)}`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${maxErr < 0.01 ? 'YES' : 'NO'}`);
}

// Test 5: Larger problem with box constraints
console.log('\n--- Test 5: Larger problem with constraints ---');
{
    const n = 60;
    const H = MatrixUtils.eye(n);
    const g = Array.from({ length: n }, (_, i) => i + 1);
    const lb = new Array(n).fill(-10);
    const ub = new Array(n).fill(10);

    const result = QPSolver.solve({ H, g, lb, ub }, { maxIterations: 200, tolerance: 1e-6 });

    // Expected: x_i = max(min(-i, 10), -10) = min(-i, 10) since -i < 0
    // For i <= 10: x_i = -i
    // For i > 10: x_i = -10
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
        const expected = Math.max(-10, -(i + 1));
        maxErr = Math.max(maxErr, Math.abs(result.x[i] - expected));
    }

    console.log(`First 5: [${result.x.slice(0, 5).map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`Expected: [-1, -2, -3, -4, -5]`);
    console.log(`Last 5: [${result.x.slice(-5).map(v => v.toFixed(2)).join(', ')}]`);
    console.log(`Expected: [-10, -10, -10, -10, -10]`);
    console.log(`Max error: ${maxErr.toFixed(6)}`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${maxErr < 0.01 ? 'YES' : 'NO'}`);
}

// Test 6: Problem similar to actual MPC (dense Hessian)
console.log('\n--- Test 6: Dense Hessian (like MPC) ---');
{
    // Create a random positive definite matrix
    const n = 20;
    const A = MatrixUtils.zeros(n, n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            A[i][j] = Math.random() - 0.5;
        }
    }
    // H = A^T * A + I (guaranteed positive definite)
    const AT = MatrixUtils.transpose(A);
    let H = MatrixUtils.matMul(AT, A);
    for (let i = 0; i < n; i++) {
        H[i][i] += 1;
    }

    const g = Array.from({ length: n }, () => Math.random() - 0.5);

    const result = QPSolver.solve({ H, g }, { maxIterations: 500, tolerance: 1e-8 });

    // Verify by checking gradient at solution
    const grad = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            grad[i] += H[i][j] * result.x[j];
        }
        grad[i] += g[i];
    }
    const gradNorm = Math.sqrt(grad.reduce((s, v) => s + v * v, 0));

    console.log(`Gradient norm at solution: ${gradNorm.toFixed(8)}`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${gradNorm < 0.01 ? 'YES' : 'NO'}`);
}

// Test 7: Verify with constrained problem - check optimality conditions
console.log('\n--- Test 7: Verify KKT conditions ---');
{
    const H = [[4, 2], [2, 2]];
    const g = [1, 1];
    const lb = [0, 0];
    const ub = [Infinity, Infinity];

    const result = QPSolver.solve({ H, g, lb, ub }, { maxIterations: 100, tolerance: 1e-8 });

    // Compute gradient
    const grad = [
        H[0][0] * result.x[0] + H[0][1] * result.x[1] + g[0],
        H[1][0] * result.x[0] + H[1][1] * result.x[1] + g[1],
    ];

    console.log(`Solution: x=[${result.x.map(v => v.toFixed(6)).join(', ')}]`);
    console.log(`Gradient at solution: [${grad.map(v => v.toFixed(6)).join(', ')}]`);

    // KKT: gradient should be non-negative at lower bound, non-positive at upper bound
    let kktSatisfied = true;
    for (let i = 0; i < 2; i++) {
        if (result.x[i] <= lb[i] + 1e-6) {
            // At lower bound, gradient should be >= 0
            if (grad[i] < -0.01) kktSatisfied = false;
        } else if (result.x[i] >= ub[i] - 1e-6) {
            // At upper bound, gradient should be <= 0
            if (grad[i] > 0.01) kktSatisfied = false;
        } else {
            // Interior, gradient should be ~0
            if (Math.abs(grad[i]) > 0.01) kktSatisfied = false;
        }
    }

    console.log(`KKT conditions satisfied: ${kktSatisfied ? 'YES' : 'NO'}`);
    console.log(`Converged: ${result.converged}, Iterations: ${result.iterations}`);
    console.log(`PASS: ${kktSatisfied ? 'YES' : 'NO'}`);
}

console.log('\n' + '='.repeat(70));
console.log('QP SOLVER VERIFICATION COMPLETE');
console.log('='.repeat(70));
