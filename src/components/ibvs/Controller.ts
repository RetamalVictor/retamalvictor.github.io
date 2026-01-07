/**
 * Image-Based Visual Servoing (IBVS) Controller
 * Computes camera velocity commands from feature errors
 */
export class IBVSController {
    private gain: number;
    private focalLength: number;
    private clipThreshold: number;

    constructor(focalLength: number, gain: number = 2.0, clipThreshold: number = 0.5) {
        this.focalLength = focalLength;
        this.gain = gain;
        this.clipThreshold = clipThreshold;
    }

    /**
     * Compute velocity command from visual feature error
     * @param desiredFeatures - target 2D positions [u1,v1, u2,v2, ...]
     * @param currentFeatures - current 2D positions [u1,v1, u2,v2, ...]
     * @param depths - Z coordinates in camera frame for each point
     * @param cx - principal point x
     * @param cy - principal point y
     * @returns velocity command [vx, vy, vz, wx, wy, wz]
     */
    private debugFrame = 0;

    public computeControl(
        desiredFeatures: Float32Array,
        currentFeatures: Float32Array,
        depths: Float32Array,
        cx: number,
        cy: number
    ): Float32Array {
        const numPoints = desiredFeatures.length / 2;

        // Compute feature error: e = s* - s
        const error = new Float32Array(desiredFeatures.length);
        for (let i = 0; i < desiredFeatures.length; i++) {
            error[i] = desiredFeatures[i] - currentFeatures[i];
        }

        // Build interaction matrix L (2*numPoints x 6)
        const L = this.computeInteractionMatrix(currentFeatures, depths, cx, cy);

        // Compute pseudo-inverse of L
        const Lpinv = this.pseudoInverse(L, numPoints * 2, 6);

        // Compute velocity: v = λ * L_pinv * error
        // Since error = (desired - current), we use POSITIVE gain
        // (Standard IBVS uses v = -λ * L⁺ * (current - desired), which equals λ * L⁺ * (desired - current))
        const velocity = new Float32Array(6);
        for (let i = 0; i < 6; i++) {
            let sum = 0;
            for (let j = 0; j < numPoints * 2; j++) {
                sum += Lpinv[i * numPoints * 2 + j] * error[j];
            }
            velocity[i] = this.gain * sum;  // Positive sign because error = desired - current
        }

        // Debug logging
        this.debugFrame++;
        if (this.debugFrame % 60 === 1) {
            console.log('\n[Controller Debug]');
            console.log('  Error vector:', Array.from(error).map(v => v.toFixed(2)));
            console.log('  L matrix (first point, vz column):', L[2].toFixed(4), L[8].toFixed(4));
            console.log('  L_pinv (vz row, first 4 cols):',
                Lpinv[2 * numPoints * 2 + 0].toFixed(4),
                Lpinv[2 * numPoints * 2 + 1].toFixed(4),
                Lpinv[2 * numPoints * 2 + 2].toFixed(4),
                Lpinv[2 * numPoints * 2 + 3].toFixed(4)
            );
            console.log('  Raw velocity (before clip):', Array.from(velocity).map(v => v.toFixed(4)));
        }

        // Clip velocity to threshold
        for (let i = 0; i < 6; i++) {
            velocity[i] = Math.max(-this.clipThreshold, Math.min(this.clipThreshold, velocity[i]));
        }

        return velocity;
    }

    /**
     * Compute the interaction matrix L for all feature points
     * Each point contributes a 2x6 block:
     * L = [ [-f/Z,    0,   x/Z,   xy/f,    -(f + x²/f),   y  ]
     *       [  0,   -f/Z,  y/Z,   f+y²/f,   -xy/f,       -x  ] ]
     */
    private computeInteractionMatrix(
        features: Float32Array,
        depths: Float32Array,
        cx: number,
        cy: number
    ): Float32Array {
        const numPoints = features.length / 2;
        const L = new Float32Array(numPoints * 2 * 6);
        const f = this.focalLength;

        for (let i = 0; i < numPoints; i++) {
            // Image coordinates relative to principal point
            const x = features[i * 2] - cx;
            const y = features[i * 2 + 1] - cy;
            const Z = Math.max(depths[i], 0.1);  // Avoid division by zero

            const row1 = i * 2;
            const row2 = i * 2 + 1;

            // Row 1 (for x coordinate)
            L[row1 * 6 + 0] = -f / Z;                           // vx
            L[row1 * 6 + 1] = 0;                                 // vy
            L[row1 * 6 + 2] = x / Z;                             // vz
            L[row1 * 6 + 3] = (x * y) / f;                       // wx
            L[row1 * 6 + 4] = -(f + (x * x) / f);                // wy
            L[row1 * 6 + 5] = y;                                 // wz

            // Row 2 (for y coordinate)
            L[row2 * 6 + 0] = 0;                                 // vx
            L[row2 * 6 + 1] = -f / Z;                            // vy
            L[row2 * 6 + 2] = y / Z;                             // vz
            L[row2 * 6 + 3] = f + (y * y) / f;                   // wx
            L[row2 * 6 + 4] = -(x * y) / f;                      // wy
            L[row2 * 6 + 5] = -x;                                // wz
        }

        return L;
    }

    /**
     * Compute Moore-Penrose pseudo-inverse using SVD
     * For an m×n matrix A, returns n×m pseudo-inverse
     */
    private pseudoInverse(A: Float32Array, rows: number, cols: number): Float32Array {
        // For small matrices, we can use a simple approach
        // A_pinv = (A^T * A + λI)^-1 * A^T (Tikhonov regularization)
        const lambda = 1e-6;

        // Compute A^T * A (cols x cols)
        const AtA = new Float32Array(cols * cols);
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < cols; j++) {
                let sum = 0;
                for (let k = 0; k < rows; k++) {
                    sum += A[k * cols + i] * A[k * cols + j];
                }
                AtA[i * cols + j] = sum;
                // Add regularization to diagonal
                if (i === j) {
                    AtA[i * cols + j] += lambda;
                }
            }
        }

        // Compute (A^T * A + λI)^-1 using Gaussian elimination
        const AtAinv = this.invertMatrix(AtA, cols);

        // Compute A^T (cols x rows)
        const At = new Float32Array(cols * rows);
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                At[i * rows + j] = A[j * cols + i];
            }
        }

        // Compute (A^T * A + λI)^-1 * A^T (cols x rows)
        const result = new Float32Array(cols * rows);
        for (let i = 0; i < cols; i++) {
            for (let j = 0; j < rows; j++) {
                let sum = 0;
                for (let k = 0; k < cols; k++) {
                    sum += AtAinv[i * cols + k] * At[k * rows + j];
                }
                result[i * rows + j] = sum;
            }
        }

        return result;
    }

    /**
     * Invert a square matrix using Gauss-Jordan elimination
     */
    private invertMatrix(A: Float32Array, n: number): Float32Array {
        // Create augmented matrix [A | I]
        const aug = new Float32Array(n * n * 2);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                aug[i * (n * 2) + j] = A[i * n + j];
                aug[i * (n * 2) + n + j] = (i === j) ? 1 : 0;
            }
        }

        // Forward elimination
        for (let col = 0; col < n; col++) {
            // Find pivot
            let maxRow = col;
            let maxVal = Math.abs(aug[col * (n * 2) + col]);
            for (let row = col + 1; row < n; row++) {
                const val = Math.abs(aug[row * (n * 2) + col]);
                if (val > maxVal) {
                    maxVal = val;
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow !== col) {
                for (let j = 0; j < n * 2; j++) {
                    const temp = aug[col * (n * 2) + j];
                    aug[col * (n * 2) + j] = aug[maxRow * (n * 2) + j];
                    aug[maxRow * (n * 2) + j] = temp;
                }
            }

            // Scale pivot row
            const pivot = aug[col * (n * 2) + col];
            if (Math.abs(pivot) < 1e-10) {
                // Matrix is singular, return identity as fallback
                const identity = new Float32Array(n * n);
                for (let i = 0; i < n; i++) identity[i * n + i] = 1;
                return identity;
            }

            for (let j = 0; j < n * 2; j++) {
                aug[col * (n * 2) + j] /= pivot;
            }

            // Eliminate column
            for (let row = 0; row < n; row++) {
                if (row !== col) {
                    const factor = aug[row * (n * 2) + col];
                    for (let j = 0; j < n * 2; j++) {
                        aug[row * (n * 2) + j] -= factor * aug[col * (n * 2) + j];
                    }
                }
            }
        }

        // Extract inverse from augmented matrix
        const inv = new Float32Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                inv[i * n + j] = aug[i * (n * 2) + n + j];
            }
        }

        return inv;
    }

    /**
     * Compute the feature error magnitude
     */
    public computeError(desiredFeatures: Float32Array, currentFeatures: Float32Array): number {
        let sumSq = 0;
        for (let i = 0; i < desiredFeatures.length; i++) {
            const diff = desiredFeatures[i] - currentFeatures[i];
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq);
    }

    /**
     * Set controller gain
     */
    public setGain(gain: number): void {
        this.gain = gain;
    }

    /**
     * Get current gain
     */
    public getGain(): number {
        return this.gain;
    }
}
