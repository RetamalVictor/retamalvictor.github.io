/**
 * QP Solver for MPC
 *
 * Solves quadratic programs of the form:
 *
 *   min  0.5 * x^T * H * x + g^T * x
 *   s.t. lb <= x <= ub  (box constraints)
 *        A_eq * x = b_eq (optional equality constraints)
 *
 * Uses projected gradient descent with Nesterov acceleration
 * for efficiency in real-time MPC applications.
 */

export interface QPProblem {
    H: number[][];      // Hessian matrix (n x n), must be positive semi-definite
    g: number[];        // Linear term (n)
    lb?: number[];      // Lower bounds (n), default -Infinity
    ub?: number[];      // Upper bounds (n), default +Infinity
    Aeq?: number[][];   // Equality constraint matrix (m x n)
    beq?: number[];     // Equality constraint RHS (m)
}

export interface QPSolution {
    x: number[];        // Optimal solution
    cost: number;       // Optimal cost value
    iterations: number; // Number of iterations
    converged: boolean; // Whether solver converged
    residual: number;   // Final residual (gradient norm)
}

export interface QPOptions {
    maxIterations: number;  // Maximum iterations (default 100)
    tolerance: number;      // Convergence tolerance (default 1e-6)
    warmStart?: number[];   // Initial guess
    stepSize?: number;      // Step size (default: use line search)
}

const DEFAULT_OPTIONS: QPOptions = {
    maxIterations: 100,
    tolerance: 1e-6,
};

/**
 * Projected Gradient Descent QP Solver with Nesterov Acceleration
 */
export class QPSolver {
    /**
     * Solve a QP problem
     */
    public static solve(problem: QPProblem, options: Partial<QPOptions> = {}): QPSolution {
        const opts = { ...DEFAULT_OPTIONS, ...options };
        const n = problem.g.length;

        // Initialize solution
        let x = opts.warmStart ? [...opts.warmStart] : new Array(n).fill(0);
        let y = [...x];  // For Nesterov acceleration

        // Set default bounds
        const lb = problem.lb ?? new Array(n).fill(-Infinity);
        const ub = problem.ub ?? new Array(n).fill(Infinity);

        // Compute step size using largest eigenvalue estimate of H
        const stepSize = opts.stepSize ?? this.estimateStepSize(problem.H);

        let t = 1;
        let converged = false;
        let iterations = 0;
        let residual = Infinity;

        for (let iter = 0; iter < opts.maxIterations; iter++) {
            iterations = iter + 1;

            // Compute gradient at y: grad = H * y + g
            const grad = this.matVecMul(problem.H, y);
            for (let i = 0; i < n; i++) {
                grad[i] += problem.g[i];
            }

            // Gradient step
            const xNew: number[] = new Array(n);
            for (let i = 0; i < n; i++) {
                xNew[i] = y[i] - stepSize * grad[i];
            }

            // Project onto box constraints
            this.projectBox(xNew, lb, ub);

            // Handle equality constraints via projection (if any)
            if (problem.Aeq && problem.beq) {
                this.projectEquality(xNew, problem.Aeq, problem.beq);
                this.projectBox(xNew, lb, ub);  // Re-project to box
            }

            // Nesterov acceleration
            const tNew = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
            const beta = (t - 1) / tNew;

            for (let i = 0; i < n; i++) {
                y[i] = xNew[i] + beta * (xNew[i] - x[i]);
            }

            // Check convergence
            residual = 0;
            for (let i = 0; i < n; i++) {
                residual += (xNew[i] - x[i]) * (xNew[i] - x[i]);
            }
            residual = Math.sqrt(residual);

            if (residual < opts.tolerance) {
                converged = true;
                x = xNew;
                break;
            }

            x = xNew;
            t = tNew;
        }

        // Compute final cost
        const cost = this.evaluateCost(problem.H, problem.g, x);

        return {
            x,
            cost,
            iterations,
            converged,
            residual,
        };
    }

    /**
     * Solve unconstrained QP via Cholesky factorization
     * H * x = -g
     */
    public static solveUnconstrained(H: number[][], g: number[]): number[] {
        // Cholesky factorization: H = L * L^T
        const L = this.cholesky(H);

        // Solve L * y = -g
        const y = this.forwardSubstitution(L, g.map(v => -v));

        // Solve L^T * x = y
        const x = this.backwardSubstitution(L, y);

        return x;
    }

    /**
     * Estimate step size from Hessian (1 / L where L is Lipschitz constant)
     */
    private static estimateStepSize(H: number[][]): number {
        const n = H.length;

        // Use Gershgorin circle theorem for rough eigenvalue bound
        let maxEig = 0;
        for (let i = 0; i < n; i++) {
            let rowSum = 0;
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    rowSum += Math.abs(H[i][j]);
                }
            }
            maxEig = Math.max(maxEig, H[i][i] + rowSum);
        }

        // Add small regularization for numerical stability
        return 1 / (maxEig + 1e-8);
    }

    /**
     * Matrix-vector multiplication: result = A * x
     */
    private static matVecMul(A: number[][], x: number[]): number[] {
        const m = A.length;
        const n = x.length;
        const result: number[] = new Array(m).fill(0);

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                result[i] += A[i][j] * x[j];
            }
        }

        return result;
    }

    /**
     * Project vector onto box constraints
     */
    private static projectBox(x: number[], lb: number[], ub: number[]): void {
        for (let i = 0; i < x.length; i++) {
            x[i] = Math.max(lb[i], Math.min(ub[i], x[i]));
        }
    }

    /**
     * Project onto equality constraints via least squares
     * Aeq * x = beq
     * x_proj = x + Aeq^T * (Aeq * Aeq^T)^{-1} * (beq - Aeq * x)
     */
    private static projectEquality(x: number[], Aeq: number[][], beq: number[]): void {
        const m = Aeq.length;
        const n = x.length;

        if (m === 0) return;

        // Compute Aeq * x
        const Ax = this.matVecMul(Aeq, x);

        // Compute residual: r = beq - Aeq * x
        const r: number[] = new Array(m);
        for (let i = 0; i < m; i++) {
            r[i] = beq[i] - Ax[i];
        }

        // Compute Aeq * Aeq^T
        const AAT: number[][] = [];
        for (let i = 0; i < m; i++) {
            AAT.push(new Array(m).fill(0));
            for (let j = 0; j < m; j++) {
                for (let k = 0; k < n; k++) {
                    AAT[i][j] += Aeq[i][k] * Aeq[j][k];
                }
            }
        }

        // Solve AAT * lambda = r
        // Use simple Gaussian elimination for small systems
        const lambda = this.solveLinearSystem(AAT, r);

        // Update x: x = x + Aeq^T * lambda
        for (let j = 0; j < n; j++) {
            for (let i = 0; i < m; i++) {
                x[j] += Aeq[i][j] * lambda[i];
            }
        }
    }

    /**
     * Evaluate quadratic cost: 0.5 * x^T * H * x + g^T * x
     */
    private static evaluateCost(H: number[][], g: number[], x: number[]): number {
        const n = x.length;
        let cost = 0;

        // Quadratic term
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                cost += 0.5 * x[i] * H[i][j] * x[j];
            }
        }

        // Linear term
        for (let i = 0; i < n; i++) {
            cost += g[i] * x[i];
        }

        return cost;
    }

    /**
     * Cholesky factorization: A = L * L^T
     */
    private static cholesky(A: number[][]): number[][] {
        const n = A.length;
        const L: number[][] = [];

        for (let i = 0; i < n; i++) {
            L.push(new Array(n).fill(0));
        }

        for (let i = 0; i < n; i++) {
            for (let j = 0; j <= i; j++) {
                let sum = 0;
                for (let k = 0; k < j; k++) {
                    sum += L[i][k] * L[j][k];
                }

                if (i === j) {
                    const val = A[i][i] - sum;
                    L[i][j] = Math.sqrt(Math.max(0, val) + 1e-10);  // Add regularization
                } else {
                    L[i][j] = (A[i][j] - sum) / (L[j][j] + 1e-10);
                }
            }
        }

        return L;
    }

    /**
     * Forward substitution: solve L * x = b
     */
    private static forwardSubstitution(L: number[][], b: number[]): number[] {
        const n = b.length;
        const x: number[] = new Array(n);

        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let j = 0; j < i; j++) {
                sum += L[i][j] * x[j];
            }
            x[i] = (b[i] - sum) / (L[i][i] + 1e-10);
        }

        return x;
    }

    /**
     * Backward substitution: solve L^T * x = b
     */
    private static backwardSubstitution(L: number[][], b: number[]): number[] {
        const n = b.length;
        const x: number[] = new Array(n);

        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += L[j][i] * x[j];  // Note: L^T[i][j] = L[j][i]
            }
            x[i] = (b[i] - sum) / (L[i][i] + 1e-10);
        }

        return x;
    }

    /**
     * Solve linear system Ax = b using Gaussian elimination
     */
    private static solveLinearSystem(A: number[][], b: number[]): number[] {
        const n = b.length;

        // Make copies
        const M: number[][] = A.map(row => [...row]);
        const r: number[] = [...b];

        // Forward elimination
        for (let k = 0; k < n; k++) {
            // Find pivot
            let maxVal = Math.abs(M[k][k]);
            let maxRow = k;
            for (let i = k + 1; i < n; i++) {
                if (Math.abs(M[i][k]) > maxVal) {
                    maxVal = Math.abs(M[i][k]);
                    maxRow = i;
                }
            }

            // Swap rows
            [M[k], M[maxRow]] = [M[maxRow], M[k]];
            [r[k], r[maxRow]] = [r[maxRow], r[k]];

            // Eliminate
            for (let i = k + 1; i < n; i++) {
                const factor = M[i][k] / (M[k][k] + 1e-10);
                for (let j = k; j < n; j++) {
                    M[i][j] -= factor * M[k][j];
                }
                r[i] -= factor * r[k];
            }
        }

        // Back substitution
        const x: number[] = new Array(n);
        for (let i = n - 1; i >= 0; i--) {
            let sum = 0;
            for (let j = i + 1; j < n; j++) {
                sum += M[i][j] * x[j];
            }
            x[i] = (r[i] - sum) / (M[i][i] + 1e-10);
        }

        return x;
    }
}

/**
 * Matrix utilities for MPC
 */
export class MatrixUtils {
    /**
     * Create identity matrix
     */
    public static eye(n: number): number[][] {
        const I: number[][] = [];
        for (let i = 0; i < n; i++) {
            I.push(new Array(n).fill(0));
            I[i][i] = 1;
        }
        return I;
    }

    /**
     * Create zero matrix
     */
    public static zeros(m: number, n: number): number[][] {
        const Z: number[][] = [];
        for (let i = 0; i < m; i++) {
            Z.push(new Array(n).fill(0));
        }
        return Z;
    }

    /**
     * Create diagonal matrix from vector
     */
    public static diag(v: number[]): number[][] {
        const n = v.length;
        const D: number[][] = [];
        for (let i = 0; i < n; i++) {
            D.push(new Array(n).fill(0));
            D[i][i] = v[i];
        }
        return D;
    }

    /**
     * Matrix-matrix multiplication: C = A * B
     */
    public static matMul(A: number[][], B: number[][]): number[][] {
        const m = A.length;
        const n = B[0].length;
        const p = B.length;

        const C: number[][] = [];
        for (let i = 0; i < m; i++) {
            C.push(new Array(n).fill(0));
            for (let j = 0; j < n; j++) {
                for (let k = 0; k < p; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    /**
     * Matrix transpose
     */
    public static transpose(A: number[][]): number[][] {
        const m = A.length;
        const n = A[0].length;
        const AT: number[][] = [];

        for (let j = 0; j < n; j++) {
            AT.push(new Array(m));
            for (let i = 0; i < m; i++) {
                AT[j][i] = A[i][j];
            }
        }
        return AT;
    }

    /**
     * Matrix addition: C = A + B
     */
    public static add(A: number[][], B: number[][]): number[][] {
        const m = A.length;
        const n = A[0].length;
        const C: number[][] = [];

        for (let i = 0; i < m; i++) {
            C.push(new Array(n));
            for (let j = 0; j < n; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    /**
     * Scalar-matrix multiplication: B = s * A
     */
    public static scale(s: number, A: number[][]): number[][] {
        return A.map(row => row.map(v => s * v));
    }

    /**
     * Block diagonal matrix from list of matrices
     */
    public static blockDiag(blocks: number[][][]): number[][] {
        let totalRows = 0;
        let totalCols = 0;
        for (const block of blocks) {
            totalRows += block.length;
            totalCols += block[0].length;
        }

        const result = this.zeros(totalRows, totalCols);
        let rowOffset = 0;
        let colOffset = 0;

        for (const block of blocks) {
            const m = block.length;
            const n = block[0].length;
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    result[rowOffset + i][colOffset + j] = block[i][j];
                }
            }
            rowOffset += m;
            colOffset += n;
        }

        return result;
    }

    /**
     * Stack matrices vertically
     */
    public static vstack(matrices: number[][][]): number[][] {
        const result: number[][] = [];
        for (const M of matrices) {
            for (const row of M) {
                result.push([...row]);
            }
        }
        return result;
    }

    /**
     * Stack matrices horizontally
     */
    public static hstack(matrices: number[][][]): number[][] {
        if (matrices.length === 0) return [];
        const m = matrices[0].length;
        const result: number[][] = [];

        for (let i = 0; i < m; i++) {
            const row: number[] = [];
            for (const M of matrices) {
                row.push(...M[i]);
            }
            result.push(row);
        }
        return result;
    }
}
