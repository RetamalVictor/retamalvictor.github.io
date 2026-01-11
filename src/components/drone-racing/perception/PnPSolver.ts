import * as THREE from 'three';
import { GateDetection, GatePose, Point2D, Vector3, Quaternion } from '../types';

/**
 * Perspective-n-Point (PnP) Solver
 *
 * Estimates 3D gate pose from 2D corner detections.
 *
 * Based on the real system's pnp.hpp:
 * 1. Uses iterative PnP algorithm (DLT + refinement)
 * 2. Validates with reprojection error (10% threshold)
 * 3. Transforms from camera frame to world frame
 *
 * The 4 gate corners provide exactly the minimum points needed
 * for a unique solution (P4P problem).
 */
export class PnPSolver {
    // Gate model (corners in gate-local frame)
    private readonly gateSize: number;
    private readonly modelPoints: THREE.Vector3[];

    // Camera intrinsics
    private readonly fx: number;  // focal length x (pixels)
    private readonly fy: number;  // focal length y (pixels)
    private readonly cx: number;  // principal point x
    private readonly cy: number;  // principal point y

    // Validation threshold (relative to gate size in pixels)
    private readonly reprojectionThreshold = 0.10;  // 10%

    // Iteration parameters
    private readonly maxIterations = 10;
    private readonly convergenceThreshold = 1e-6;

    constructor(
        gateSize: number = 1.52,
        focalLength: number = 400,
        imageWidth: number = 640,
        imageHeight: number = 480
    ) {
        this.gateSize = gateSize;

        // Define gate corners in gate-local frame (Z = 0 plane)
        // Order: TL, TR, BR, BL
        const halfSize = gateSize / 2;
        this.modelPoints = [
            new THREE.Vector3(-halfSize, halfSize, 0),   // TL
            new THREE.Vector3(halfSize, halfSize, 0),    // TR
            new THREE.Vector3(halfSize, -halfSize, 0),   // BR
            new THREE.Vector3(-halfSize, -halfSize, 0),  // BL
        ];

        // Camera intrinsics (assuming square pixels)
        this.fx = focalLength;
        this.fy = focalLength;
        this.cx = imageWidth / 2;
        this.cy = imageHeight / 2;
    }

    /**
     * Solve for gate pose from detected corners
     *
     * @param detection - Gate detection with 2D corners
     * @param cameraPose - Current camera pose (for world frame transform)
     * @returns Gate pose in world frame, or null if solution is invalid
     */
    public solvePose(
        detection: GateDetection,
        cameraPose: { position: THREE.Vector3; quaternion: THREE.Quaternion }
    ): { pose: GatePose; reprojectionError: number } | null {
        const corners = detection.keypoints.corners;

        // Step 1: Solve PnP in camera frame
        const cameraFramePose = this.solvePnPCamera(corners);
        if (!cameraFramePose) {
            return null;
        }

        // Step 2: Validate with reprojection error
        const reprojError = this.computeReprojectionError(corners, cameraFramePose);
        const gateSizePixels = this.estimateGateSizePixels(corners);

        if (reprojError / gateSizePixels > this.reprojectionThreshold) {
            return null;  // Reject high-error solutions
        }

        // Step 3: Transform to world frame
        const worldPose = this.transformToWorld(cameraFramePose, cameraPose);

        return {
            pose: {
                gateId: detection.gateId,
                position: worldPose.position,
                orientation: worldPose.orientation,
                innerSize: this.gateSize,
            },
            reprojectionError: reprojError / gateSizePixels,
        };
    }

    /**
     * Solve PnP in camera frame using iterative algorithm
     */
    private solvePnPCamera(corners: [Point2D, Point2D, Point2D, Point2D]): {
        translation: THREE.Vector3;
        rotation: THREE.Matrix3;
    } | null {
        // Normalize image points (subtract principal point, divide by focal length)
        const normalizedPoints = corners.map(c => ({
            x: (c.u - this.cx) / this.fx,
            y: (c.v - this.cy) / this.fy,
        }));

        // Initial estimate using DLT (Direct Linear Transform)
        const initialPose = this.dltSolve(normalizedPoints);
        if (!initialPose) {
            return null;
        }

        // Refine using iterative optimization
        const refinedPose = this.refineIterative(normalizedPoints, initialPose);

        return refinedPose;
    }

    /**
     * Direct Linear Transform for initial pose estimate
     * Solves the homogeneous linear system derived from projection equations
     */
    private dltSolve(normalizedPoints: { x: number; y: number }[]): {
        translation: THREE.Vector3;
        rotation: THREE.Matrix3;
    } | null {
        // Build the DLT matrix (2n x 12 for n points, we have 4 points)
        // Each point gives 2 equations
        const A: number[][] = [];

        for (let i = 0; i < 4; i++) {
            const X = this.modelPoints[i].x;
            const Y = this.modelPoints[i].y;
            const Z = this.modelPoints[i].z;
            const u = normalizedPoints[i].x;
            const v = normalizedPoints[i].y;

            // Row 1: [X Y Z 1 0 0 0 0 -u*X -u*Y -u*Z -u]
            A.push([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]);
            // Row 2: [0 0 0 0 X Y Z 1 -v*X -v*Y -v*Z -v]
            A.push([0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]);
        }

        // Solve using SVD (simplified: use pseudo-inverse approach)
        const solution = this.solveSVD(A);
        if (!solution) {
            return null;
        }

        // Extract rotation and translation from solution
        // Solution is the projection matrix P = [R | t] flattened
        const P = [
            [solution[0], solution[1], solution[2], solution[3]],
            [solution[4], solution[5], solution[6], solution[7]],
            [solution[8], solution[9], solution[10], solution[11]],
        ];

        // Enforce rotation matrix orthonormality
        const R = this.enforceRotation([
            [P[0][0], P[0][1], P[0][2]],
            [P[1][0], P[1][1], P[1][2]],
            [P[2][0], P[2][1], P[2][2]],
        ]);

        // Scale factor from rotation matrix normalization
        const scale = (
            Math.sqrt(P[0][0] * P[0][0] + P[1][0] * P[1][0] + P[2][0] * P[2][0]) +
            Math.sqrt(P[0][1] * P[0][1] + P[1][1] * P[1][1] + P[2][1] * P[2][1]) +
            Math.sqrt(P[0][2] * P[0][2] + P[1][2] * P[1][2] + P[2][2] * P[2][2])
        ) / 3;

        const translation = new THREE.Vector3(
            P[0][3] / scale,
            P[1][3] / scale,
            P[2][3] / scale
        );

        // Ensure positive depth
        if (translation.z < 0) {
            translation.multiplyScalar(-1);
            R[0][0] *= -1; R[0][1] *= -1; R[0][2] *= -1;
            R[1][0] *= -1; R[1][1] *= -1; R[1][2] *= -1;
            R[2][0] *= -1; R[2][1] *= -1; R[2][2] *= -1;
        }

        const rotation = new THREE.Matrix3();
        rotation.set(
            R[0][0], R[0][1], R[0][2],
            R[1][0], R[1][1], R[1][2],
            R[2][0], R[2][1], R[2][2]
        );

        return { translation, rotation };
    }

    /**
     * Solve Ax = 0 using simplified SVD (power iteration for smallest singular value)
     */
    private solveSVD(A: number[][]): number[] | null {
        const m = A.length;
        const n = A[0].length;

        // Compute A^T * A
        const AtA: number[][] = [];
        for (let i = 0; i < n; i++) {
            AtA[i] = [];
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < m; k++) {
                    sum += A[k][i] * A[k][j];
                }
                AtA[i][j] = sum;
            }
        }

        // Power iteration to find smallest eigenvector
        let x = new Array(n).fill(1 / Math.sqrt(n));

        for (let iter = 0; iter < 100; iter++) {
            // Inverse power iteration: solve (A^T A) y = x
            // Use simplified approach: compute (A^T A + λI)^{-1} x with small λ
            const lambda = 1e-6;
            const y = this.solveLinearSystem(AtA, x, lambda);
            if (!y) return null;

            // Normalize
            const norm = Math.sqrt(y.reduce((sum, v) => sum + v * v, 0));
            if (norm < 1e-10) return null;
            x = y.map(v => v / norm);
        }

        return x;
    }

    /**
     * Solve (A + λI)x = b using Gauss-Jordan elimination
     */
    private solveLinearSystem(A: number[][], b: number[], lambda: number): number[] | null {
        const n = A.length;

        // Create augmented matrix [A + λI | b]
        const aug: number[][] = [];
        for (let i = 0; i < n; i++) {
            aug[i] = [...A[i]];
            aug[i][i] += lambda;
            aug[i].push(b[i]);
        }

        // Forward elimination with partial pivoting
        for (let col = 0; col < n; col++) {
            // Find pivot
            let maxRow = col;
            for (let row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > Math.abs(aug[maxRow][col])) {
                    maxRow = row;
                }
            }
            [aug[col], aug[maxRow]] = [aug[maxRow], aug[col]];

            if (Math.abs(aug[col][col]) < 1e-10) continue;

            // Eliminate below
            for (let row = col + 1; row < n; row++) {
                const factor = aug[row][col] / aug[col][col];
                for (let j = col; j <= n; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        const x = new Array(n).fill(0);
        for (let i = n - 1; i >= 0; i--) {
            let sum = aug[i][n];
            for (let j = i + 1; j < n; j++) {
                sum -= aug[i][j] * x[j];
            }
            x[i] = Math.abs(aug[i][i]) > 1e-10 ? sum / aug[i][i] : 0;
        }

        return x;
    }

    /**
     * Enforce rotation matrix orthonormality using SVD
     */
    private enforceRotation(R: number[][]): number[][] {
        // Simplified: use Gram-Schmidt orthonormalization
        const r1 = [R[0][0], R[1][0], R[2][0]];
        const r2 = [R[0][1], R[1][1], R[2][1]];
        const r3 = [R[0][2], R[1][2], R[2][2]];

        // Normalize r1
        const n1 = Math.sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]);
        r1[0] /= n1; r1[1] /= n1; r1[2] /= n1;

        // r2 = r2 - (r2.r1)r1, then normalize
        const d21 = r2[0] * r1[0] + r2[1] * r1[1] + r2[2] * r1[2];
        r2[0] -= d21 * r1[0]; r2[1] -= d21 * r1[1]; r2[2] -= d21 * r1[2];
        const n2 = Math.sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]);
        r2[0] /= n2; r2[1] /= n2; r2[2] /= n2;

        // r3 = r1 x r2 (cross product ensures right-handed)
        r3[0] = r1[1] * r2[2] - r1[2] * r2[1];
        r3[1] = r1[2] * r2[0] - r1[0] * r2[2];
        r3[2] = r1[0] * r2[1] - r1[1] * r2[0];

        return [
            [r1[0], r2[0], r3[0]],
            [r1[1], r2[1], r3[1]],
            [r1[2], r2[2], r3[2]],
        ];
    }

    /**
     * Refine pose estimate using iterative optimization
     */
    private refineIterative(
        normalizedPoints: { x: number; y: number }[],
        initialPose: { translation: THREE.Vector3; rotation: THREE.Matrix3 }
    ): { translation: THREE.Vector3; rotation: THREE.Matrix3 } {
        let t = initialPose.translation.clone();
        let R = initialPose.rotation.clone();

        for (let iter = 0; iter < this.maxIterations; iter++) {
            // Compute Jacobian and residual
            const { J, r } = this.computeJacobianAndResidual(normalizedPoints, t, R);

            // Solve normal equations: (J^T J) dx = J^T r
            const JtJ = this.multiplyAtA(J);
            const Jtr = this.multiplyAtb(J, r);

            // Add damping for stability
            for (let i = 0; i < 6; i++) {
                JtJ[i][i] += 0.01;
            }

            const dx = this.solveLinearSystem(JtJ, Jtr, 0);
            if (!dx) break;

            // Update translation
            t.x -= dx[0];
            t.y -= dx[1];
            t.z -= dx[2];

            // Update rotation (using Rodrigues formula)
            const dw = new THREE.Vector3(dx[3], dx[4], dx[5]);
            if (dw.length() > 1e-10) {
                const angle = dw.length();
                const axis = dw.clone().normalize();
                const dR = new THREE.Matrix4().makeRotationAxis(axis, -angle);
                const R4 = new THREE.Matrix4().setFromMatrix3(R);
                R4.premultiply(dR);
                R.setFromMatrix4(R4);
            }

            // Check convergence
            const norm = Math.sqrt(dx.reduce((s, v) => s + v * v, 0));
            if (norm < this.convergenceThreshold) break;
        }

        return { translation: t, rotation: R };
    }

    /**
     * Compute Jacobian and residual for iterative refinement
     */
    private computeJacobianAndResidual(
        normalizedPoints: { x: number; y: number }[],
        t: THREE.Vector3,
        R: THREE.Matrix3
    ): { J: number[][]; r: number[] } {
        const J: number[][] = [];
        const r: number[] = [];

        for (let i = 0; i < 4; i++) {
            const P = this.modelPoints[i];

            // Transform point: p = R * P + t
            const Rp = new THREE.Vector3(P.x, P.y, P.z).applyMatrix3(R);
            const p = Rp.clone().add(t);

            // Projection
            const invZ = 1 / p.z;
            const projU = p.x * invZ;
            const projV = p.y * invZ;

            // Residual
            r.push(normalizedPoints[i].x - projU);
            r.push(normalizedPoints[i].y - projV);

            // Jacobian (partial derivatives)
            const invZ2 = invZ * invZ;

            // d(u)/d(tx, ty, tz, wx, wy, wz)
            J.push([
                invZ, 0, -p.x * invZ2,
                -p.x * p.y * invZ2, 1 + p.x * p.x * invZ2, -p.y * invZ
            ]);
            // d(v)/d(tx, ty, tz, wx, wy, wz)
            J.push([
                0, invZ, -p.y * invZ2,
                -(1 + p.y * p.y * invZ2), p.x * p.y * invZ2, p.x * invZ
            ]);
        }

        return { J, r };
    }

    /**
     * Compute A^T * A
     */
    private multiplyAtA(A: number[][]): number[][] {
        const m = A.length;
        const n = A[0].length;
        const result: number[][] = [];

        for (let i = 0; i < n; i++) {
            result[i] = [];
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < m; k++) {
                    sum += A[k][i] * A[k][j];
                }
                result[i][j] = sum;
            }
        }

        return result;
    }

    /**
     * Compute A^T * b
     */
    private multiplyAtb(A: number[][], b: number[]): number[] {
        const m = A.length;
        const n = A[0].length;
        const result: number[] = [];

        for (let i = 0; i < n; i++) {
            let sum = 0;
            for (let k = 0; k < m; k++) {
                sum += A[k][i] * b[k];
            }
            result[i] = sum;
        }

        return result;
    }

    /**
     * Compute reprojection error
     */
    private computeReprojectionError(
        corners: [Point2D, Point2D, Point2D, Point2D],
        pose: { translation: THREE.Vector3; rotation: THREE.Matrix3 }
    ): number {
        let totalError = 0;

        for (let i = 0; i < 4; i++) {
            const P = this.modelPoints[i];

            // Transform point
            const Rp = new THREE.Vector3(P.x, P.y, P.z).applyMatrix3(pose.rotation);
            const p = Rp.add(pose.translation);

            // Project
            const projU = this.fx * (p.x / p.z) + this.cx;
            const projV = this.fy * (p.y / p.z) + this.cy;

            // Error
            const du = corners[i].u - projU;
            const dv = corners[i].v - projV;
            totalError += Math.sqrt(du * du + dv * dv);
        }

        return totalError / 4;  // Average error per corner
    }

    /**
     * Estimate gate size in pixels (max diagonal)
     */
    private estimateGateSizePixels(corners: [Point2D, Point2D, Point2D, Point2D]): number {
        const diag1 = Math.sqrt(
            Math.pow(corners[2].u - corners[0].u, 2) +
            Math.pow(corners[2].v - corners[0].v, 2)
        );
        const diag2 = Math.sqrt(
            Math.pow(corners[3].u - corners[1].u, 2) +
            Math.pow(corners[3].v - corners[1].v, 2)
        );
        return Math.max(diag1, diag2);
    }

    /**
     * Transform pose from camera frame to world frame
     */
    private transformToWorld(
        cameraPose: { translation: THREE.Vector3; rotation: THREE.Matrix3 },
        camera: { position: THREE.Vector3; quaternion: THREE.Quaternion }
    ): { position: Vector3; orientation: Quaternion } {
        // Camera rotation matrix
        const cameraRot = new THREE.Matrix4().makeRotationFromQuaternion(camera.quaternion);
        const cameraRotMat3 = new THREE.Matrix3().setFromMatrix4(cameraRot);

        // Gate position in camera frame
        const gatePosCamera = cameraPose.translation.clone();

        // Transform to world frame: p_world = R_camera * p_camera + t_camera
        const gatePosWorld = gatePosCamera.applyMatrix3(cameraRotMat3).add(camera.position);

        // Gate orientation in world frame
        const gateRotCamera = cameraPose.rotation.clone();
        const gateRotWorld = new THREE.Matrix3().multiplyMatrices(cameraRotMat3, gateRotCamera);

        // Convert rotation matrix to quaternion
        const rotMat4 = new THREE.Matrix4().setFromMatrix3(gateRotWorld);
        const quat = new THREE.Quaternion().setFromRotationMatrix(rotMat4);

        return {
            position: { x: gatePosWorld.x, y: gatePosWorld.y, z: gatePosWorld.z },
            orientation: { w: quat.w, x: quat.x, y: quat.y, z: quat.z },
        };
    }

    /**
     * Project model points using current pose (for visualization)
     */
    public reprojectCorners(pose: { translation: THREE.Vector3; rotation: THREE.Matrix3 }): Point2D[] {
        const projected: Point2D[] = [];

        for (const P of this.modelPoints) {
            const Rp = new THREE.Vector3(P.x, P.y, P.z).applyMatrix3(pose.rotation);
            const p = Rp.add(pose.translation);

            projected.push({
                u: this.fx * (p.x / p.z) + this.cx,
                v: this.fy * (p.y / p.z) + this.cy,
            });
        }

        return projected;
    }
}
