import { DroneState, ControlCommand, Waypoint } from '../types';
import { MPCModel, MPCState, MPCInput } from './MPCModel';
import { QPSolver, MatrixUtils } from './QPSolver';
import { DynamicsParams } from '../core/DroneDynamics';

/**
 * Model Predictive Controller using Sequential Quadratic Programming
 *
 * This is a TRUE MPC implementation that:
 * 1. Predicts future states using the dynamics model
 * 2. Formulates a QP to minimize tracking error + control effort
 * 3. Solves the QP with input constraints
 * 4. Applies the first control action (receding horizon)
 *
 * The SQP approach linearizes the nonlinear dynamics around the current
 * trajectory guess and iterates until convergence.
 */

export interface MPCConfig {
    // Horizon
    horizonSteps: number;      // Number of prediction steps (N)
    dt: number;                // Time step between predictions (s)

    // Cost weights (diagonal of Q and R matrices)
    positionWeight: number;    // Position tracking weight
    velocityWeight: number;    // Velocity tracking weight
    attitudeWeight: number;    // Roll/pitch (thrust direction) tracking weight
    yawWeight: number;         // Yaw tracking weight (separate for aggressive heading tracking)
    thrustWeight: number;      // Thrust effort weight
    rateWeight: number;        // Angular rate effort weight

    // Terminal cost multiplier
    terminalWeight: number;    // Multiplier for terminal state cost

    // Input constraints
    minThrust: number;         // Minimum thrust (m/s²)
    maxThrust: number;         // Maximum thrust (m/s²)
    maxRate: number;           // Maximum roll/pitch rate (rad/s)
    maxYawRate: number;        // Maximum yaw rate (rad/s)

    // SQP settings
    sqpIterations: number;     // Max SQP iterations per control step
    sqpTolerance: number;      // SQP convergence tolerance

    // Command delay compensation
    commandDelay: number;      // Delay to compensate for (s)
}

export const DEFAULT_MPC_CONFIG: MPCConfig = {
    // 15 nodes * 0.05s = 0.75s horizon
    horizonSteps: 15,
    dt: 0.05,

    // Weights tuned for stable trajectory tracking
    // Using thrust-direction cost for attitude to avoid gimbal lock
    positionWeight: 400.0,   // High position tracking priority
    velocityWeight: 1.0,     // Low velocity weight (allows movement)
    attitudeWeight: 10.0,    // Roll/pitch (thrust direction) penalty
    yawWeight: 10.0,         // Same as roll/pitch for now
    thrustWeight: 0.5,       // Input cost for thrust
    rateWeight: 1.0,         // Rate cost

    // Terminal cost - same weight as running cost
    terminalWeight: 1.0,

    // Reference uses: thrust=[0,70]N for 0.979kg drone ≈ [0,71] m/s²
    // But code generation uses [2, 20], let's use similar
    minThrust: 2.0,
    maxThrust: 20.0,
    // Reference uses ±10.22 rad/s for roll/pitch, ±3 rad/s for yaw
    maxRate: 10.0,
    maxYawRate: 3.0,

    sqpIterations: 3,
    sqpTolerance: 1e-4,

    // Reference uses 0.05s command delay
    commandDelay: 0.05,
};

/**
 * Reference trajectory for MPC
 */
export interface MPCReference {
    states: MPCState[];    // Reference states at each horizon step
    inputs: MPCInput[];    // Reference inputs (optional, for feedforward)
}

export class MPC {
    private config: MPCConfig;
    private model: MPCModel;

    // State and input dimensions
    private readonly nx: number;
    private readonly nu: number;
    private readonly N: number;  // Horizon length

    // Cost matrices
    private Q: number[][];      // State cost (nx x nx)
    private R: number[][];      // Input cost (nu x nu)
    private Qf: number[][];     // Terminal state cost (nx x nx)

    // Warm start from previous solution
    private prevInputs: MPCInput[] | null = null;
    private prevStates: MPCState[] | null = null;

    // Last computed trajectory for visualization
    private lastPredictedStates: MPCState[] = [];

    constructor(
        config: Partial<MPCConfig> = {},
        dynamicsParams: Partial<DynamicsParams> = {}
    ) {
        this.config = { ...DEFAULT_MPC_CONFIG, ...config };
        this.model = new MPCModel(dynamicsParams);

        this.nx = this.model.nx;
        this.nu = this.model.nu;
        this.N = this.config.horizonSteps;

        // Build cost matrices
        this.Q = this.buildStateCost();
        this.R = this.buildInputCost();
        this.Qf = MatrixUtils.scale(this.config.terminalWeight, this.Q);
    }

    /**
     * Build state cost matrix Q
     *
     * State indices (14D with quaternion):
     * - Position: [0, 1, 2] (px, py, pz)
     * - Velocity: [3, 4, 5] (vx, vy, vz)
     * - Quaternion: [6, 7, 8, 9] (qw, qx, qy, qz)
     * - Actuators: [10, 11, 12, 13] (thrust, rollRate, pitchRate, yawRate)
     */
    private buildStateCost(): number[][] {
        const Q = MatrixUtils.zeros(this.nx, this.nx);

        // Position weights (indices 0, 1, 2)
        Q[0][0] = this.config.positionWeight;
        Q[1][1] = this.config.positionWeight;
        Q[2][2] = this.config.positionWeight;

        // Velocity weights (indices 3, 4, 5)
        Q[3][3] = this.config.velocityWeight;
        Q[4][4] = this.config.velocityWeight;
        Q[5][5] = this.config.velocityWeight;

        // Quaternion weights (indices 6, 7, 8, 9)
        // qw is near 1 for small rotations, so we don't penalize it
        // qx, qy, qz represent half the rotation angle components
        Q[6][6] = 0;  // qw - no cost (it's near 1 for small errors)
        Q[7][7] = this.config.attitudeWeight;  // qx - pitch component
        Q[8][8] = this.config.yawWeight;       // qy - yaw component
        Q[9][9] = this.config.attitudeWeight;  // qz - roll component

        // Actuator state weights (indices 10, 11, 12, 13) - small
        Q[10][10] = 0.01;
        Q[11][11] = 0.01;
        Q[12][12] = 0.01;
        Q[13][13] = 0.01;

        return Q;
    }

    /**
     * Build input cost matrix R
     */
    private buildInputCost(): number[][] {
        const R = MatrixUtils.zeros(this.nu, this.nu);

        R[0][0] = this.config.thrustWeight;  // Thrust
        R[1][1] = this.config.rateWeight;    // Roll rate
        R[2][2] = this.config.rateWeight;    // Pitch rate
        R[3][3] = this.config.rateWeight;    // Yaw rate

        return R;
    }

    /**
     * Compute optimal control given current state and reference trajectory
     *
     * @param currentState - Current drone state
     * @param getReference - Function to get reference waypoint at time t
     * @param currentTime - Current time in trajectory
     * @returns Optimal control command
     */
    public computeControl(
        currentState: DroneState,
        getReference: (t: number) => Waypoint,
        currentTime: number
    ): ControlCommand {
        // Convert drone state to MPC state
        const x0 = this.droneStateToMPCState(currentState);

        // Sample reference trajectory
        const reference = this.sampleReference(getReference, currentTime);

        // Initialize trajectory guess (warm start or reference)
        const { states: initStates, inputs: initInputs } = this.initializeTrajectory(x0, reference);

        // Run SQP iterations
        let states = initStates;
        let inputs = initInputs;

        for (let sqpIter = 0; sqpIter < this.config.sqpIterations; sqpIter++) {
            // Linearize dynamics along current trajectory
            const linearizations = this.linearizeAlongTrajectory(states, inputs);

            // Build and solve QP
            const solution = this.solveQPSubproblem(x0, states, inputs, reference, linearizations);

            // Update trajectory
            const { newStates, newInputs, improvement } = this.updateTrajectory(
                x0, states, inputs, solution, linearizations
            );

            states = newStates;
            inputs = newInputs;

            // Check convergence
            if (improvement < this.config.sqpTolerance) {
                break;
            }
        }

        // Store for warm start and visualization
        this.prevStates = states;
        this.prevInputs = inputs;
        this.lastPredictedStates = states;

        // Return first control action
        return this.model.inputToCommand(inputs[0], currentState.timestamp);
    }

    /**
     * Convert DroneState to MPCState
     */
    private droneStateToMPCState(state: DroneState): MPCState {
        return this.model.fromDroneState(
            state.position,
            state.velocity,
            state.orientation,
            this.model.params.gravity,  // Default hover thrust
            { x: 0, y: 0, z: 0 }
        );
    }

    /**
     * Sample reference trajectory from waypoint function
     *
     * Generates quaternion-based reference states from waypoints.
     */
    private sampleReference(
        getReference: (t: number) => Waypoint,
        currentTime: number
    ): MPCReference {
        const states: MPCState[] = [];
        const inputs: MPCInput[] = [];
        const { dt, commandDelay } = this.config;
        const gravity = this.model.params.gravity;

        for (let k = 0; k <= this.N; k++) {
            const t = currentTime + commandDelay + k * dt;
            const wp = getReference(t);

            // Wrap heading to ±π
            let heading = wp.heading;
            while (heading > Math.PI) heading -= 2 * Math.PI;
            while (heading < -Math.PI) heading += 2 * Math.PI;

            // Compute desired quaternion from acceleration and heading
            const q = this.accelerationToQuaternion(wp.acceleration, heading, gravity);

            states.push({
                px: wp.position.x,
                py: wp.position.y,
                pz: wp.position.z,
                vx: wp.velocity.x,
                vy: wp.velocity.y,
                vz: wp.velocity.z,
                qw: q.w,
                qx: q.x,
                qy: q.y,
                qz: q.z,
                thrust: gravity,  // Will be updated by feedforward
                rollRate: 0,
                pitchRate: 0,
                yawRate: wp.headingRate,
            });

            if (k < this.N) {
                // Compute feedforward input from reference
                const thrust = this.computeFeedforwardThrust(wp, gravity);
                inputs.push({
                    thrust,
                    rollRate: 0,
                    pitchRate: 0,
                    yawRate: wp.headingRate,
                });
            }
        }

        return { states, inputs };
    }

    /**
     * Compute desired quaternion from desired acceleration and heading
     *
     * The quaternion represents the orientation needed to produce
     * the desired thrust vector while maintaining the given heading.
     */
    private accelerationToQuaternion(
        accel: { x: number; y: number; z: number },
        yaw: number,
        gravity: number
    ): { w: number; x: number; y: number; z: number } {
        // Desired thrust vector in world frame (must counteract gravity + provide accel)
        const ax = accel.x;
        const ay = accel.y + gravity;  // Compensate gravity
        const az = accel.z;

        // Compute body-frame tilt angles from thrust direction
        const cy = Math.cos(yaw);
        const sy = Math.sin(yaw);

        // Transform to body frame (rotate by -yaw around Y)
        const axBody = ax * cy + az * sy;
        const ayBody = ay;
        const azBody = -ax * sy + az * cy;

        // Body-frame tilt angles
        // Roll tilts thrust in local X direction, pitch tilts in local Z
        const bodyRoll = -Math.atan2(axBody, ayBody);
        const bodyPitch = Math.atan2(azBody, Math.sqrt(axBody * axBody + ayBody * ayBody));

        // Clamp to reasonable range
        const maxTilt = Math.PI / 4;  // 45 degrees
        const clampedRoll = Math.max(-maxTilt, Math.min(maxTilt, bodyRoll));
        const clampedPitch = Math.max(-maxTilt, Math.min(maxTilt, bodyPitch));

        // Build quaternion: q = q_yaw * q_pitch * q_roll (YXZ order)
        // q_yaw rotates around Y, q_pitch around X, q_roll around Z
        const halfYaw = yaw / 2;
        const halfPitch = clampedPitch / 2;
        const halfRoll = clampedRoll / 2;

        const cy2 = Math.cos(halfYaw);
        const sy2 = Math.sin(halfYaw);
        const cp = Math.cos(halfPitch);
        const sp = Math.sin(halfPitch);
        const cr = Math.cos(halfRoll);
        const sr = Math.sin(halfRoll);

        // q = q_yaw * q_pitch * q_roll
        // q_yaw = (cy2, 0, sy2, 0)
        // q_pitch = (cp, sp, 0, 0)
        // q_roll = (cr, 0, 0, sr)
        return {
            w: cy2 * cp * cr + sy2 * sp * sr,
            x: cy2 * sp * cr + sy2 * cp * sr,
            y: sy2 * cp * cr - cy2 * sp * sr,
            z: cy2 * cp * sr - sy2 * sp * cr,
        };
    }

    /**
     * Compute feedforward thrust from reference
     */
    private computeFeedforwardThrust(wp: Waypoint, gravity: number): number {
        const ax = wp.acceleration.x;
        const ay = wp.acceleration.y + gravity;
        const az = wp.acceleration.z;
        return Math.sqrt(ax * ax + ay * ay + az * az);
    }

    /**
     * Initialize trajectory for SQP (warm start or from reference)
     */
    private initializeTrajectory(
        x0: MPCState,
        reference: MPCReference
    ): { states: MPCState[]; inputs: MPCInput[] } {
        // Disabled warm-start - it hurts circular trajectory tracking without speed benefit
        const useWarmStart = false;

        if (useWarmStart && this.prevInputs && this.prevStates) {
            // Warm start: shift previous solution
            const states: MPCState[] = [x0];
            const inputs: MPCInput[] = [];

            // Shift inputs by one
            for (let k = 1; k < this.N; k++) {
                inputs.push(this.prevInputs[k]);
            }
            // Append last input or reference
            inputs.push(reference.inputs[this.N - 1] ?? this.model.createHoverInput());

            // Rollout states
            let state = x0;
            for (let k = 0; k < this.N; k++) {
                state = this.model.dynamics(state, inputs[k], this.config.dt);
                states.push(state);
            }

            return { states, inputs };
        }

        // Cold start: rollout from x0 using reference inputs
        // This ensures the nominal trajectory starts from actual current state
        const coldInputs = [...reference.inputs];
        const coldStates = this.model.rollout(x0, coldInputs, this.config.dt);
        return { states: coldStates, inputs: coldInputs };
    }

    /**
     * Linearize dynamics along trajectory
     */
    private linearizeAlongTrajectory(
        states: MPCState[],
        inputs: MPCInput[]
    ): { A: number[][]; B: number[][]; c: number[] }[] {
        const linearizations: { A: number[][]; B: number[][]; c: number[] }[] = [];

        for (let k = 0; k < this.N; k++) {
            const lin = this.model.linearize(states[k], inputs[k], this.config.dt);
            linearizations.push(lin);
        }

        return linearizations;
    }

    /**
     * Build and solve QP subproblem
     *
     * The QP is formulated with decision variables:
     * z = [dx_0, du_0, dx_1, du_1, ..., dx_N]
     *
     * Where dx_k = x_k - x_k^ref (state deviation)
     *       du_k = u_k - u_k^ref (input deviation)
     */
    private solveQPSubproblem(
        _x0: MPCState,
        nominalStates: MPCState[],  // Predicted states with nominal inputs
        inputs: MPCInput[],
        reference: MPCReference,
        linearizations: { A: number[][]; B: number[][]; c: number[] }[]
    ): number[] {
        const { N, nx, nu } = this;

        // Decision variable: dU = U - U_nom (input deviations from nominal)
        // The QP minimizes the cost with respect to dU
        const nz = N * nu;

        // Using condensed form: x_k = x_nom_k + Psi_k * dU
        // where x_nom_k is the predicted state with nominal inputs
        // and Psi_k tells us how state k changes with input perturbations

        // Compute prediction matrices (Psi matrices for sensitivity)
        const { Psi } = this.computePredictionMatrices(linearizations, nominalStates[0]);

        // Build Hessian: H = Psi^T * Q_bar * Psi + R_bar
        const H = MatrixUtils.zeros(nz, nz);
        const g = new Array(nz).fill(0);

        // R_bar contribution to Hessian (input cost)
        for (let k = 0; k < N; k++) {
            for (let i = 0; i < nu; i++) {
                for (let j = 0; j < nu; j++) {
                    H[k * nu + i][k * nu + j] += this.R[i][j];
                }
            }
        }

        // State cost contribution: sum over k
        for (let k = 0; k <= N; k++) {
            const Qk = k < N ? this.Q : this.Qf;
            const PsiK = Psi[k];  // (nx x nz)

            // Hessian: Psi_k^T * Q * Psi_k
            for (let i = 0; i < nz; i++) {
                for (let j = 0; j < nz; j++) {
                    for (let a = 0; a < nx; a++) {
                        for (let b = 0; b < nx; b++) {
                            H[i][j] += PsiK[a][i] * Qk[a][b] * PsiK[b][j];
                        }
                    }
                }
            }

            // Gradient: Psi_k^T * Q * (x_nom_k - x_ref_k)
            // The residual is the difference between predicted state and reference
            const xNomK = this.model.stateToArray(nominalStates[k]);
            const xRefK = this.model.stateToArray(reference.states[k]);

            // Extract quaternions for proper error computation
            const qNomW = xNomK[6], qNomX = xNomK[7], qNomY = xNomK[8], qNomZ = xNomK[9];
            let qRefW = xRefK[6], qRefX = xRefK[7], qRefY = xRefK[8], qRefZ = xRefK[9];

            // Ensure quaternions are in the same hemisphere (q and -q represent same rotation)
            const dot = qNomW * qRefW + qNomX * qRefX + qNomY * qRefY + qNomZ * qRefZ;
            if (dot < 0) {
                // Negate reference quaternion to get shortest path
                qRefW = -qRefW;
                qRefX = -qRefX;
                qRefY = -qRefY;
                qRefZ = -qRefZ;
            }

            // Compute residual using quaternion subtraction
            // For linearized MPC, direct subtraction in tangent space is valid
            // This avoids all gimbal lock issues of Euler angles
            const residual = xNomK.map((x, i) => {
                // For quaternion components (indices 6-9), use hemisphere-corrected reference
                if (i === 6) return qNomW - qRefW;
                if (i === 7) return qNomX - qRefX;
                if (i === 8) return qNomY - qRefY;
                if (i === 9) return qNomZ - qRefZ;
                return x - xRefK[i];
            });

            // g += Psi_k^T * Q * residual
            for (let i = 0; i < nz; i++) {
                for (let a = 0; a < nx; a++) {
                    let QRes = 0;
                    for (let b = 0; b < nx; b++) {
                        QRes += Qk[a][b] * residual[b];
                    }
                    g[i] += PsiK[a][i] * QRes;
                }
            }
        }

        // Input reference cost gradient
        for (let k = 0; k < N; k++) {
            const uRef = this.model.inputToArray(reference.inputs[k]);
            const uNom = this.model.inputToArray(inputs[k]);
            for (let i = 0; i < nu; i++) {
                let RDu = 0;
                for (let j = 0; j < nu; j++) {
                    RDu += this.R[i][j] * (uNom[j] - uRef[j]);
                }
                g[k * nu + i] += RDu;
            }
        }

        // Build input constraints
        const { lb, ub } = this.buildInputConstraints(inputs);

        // Solve QP
        const solution = QPSolver.solve({ H, g, lb, ub }, {
            maxIterations: 50,
            tolerance: 1e-6,
        });

        return solution.x;
    }

    /**
     * Compute prediction matrices for condensed QP
     *
     * x_k = Phi_k * x0 + Psi_k * U + offset_k
     */
    private computePredictionMatrices(
        linearizations: { A: number[][]; B: number[][]; c: number[] }[],
        _x0: MPCState  // Kept for API consistency
    ): {
        Phi: number[][][];     // Phi[k] is (nx x nx)
        Psi: number[][][];     // Psi[k] is (nx x N*nu)
        offsets: number[][];   // offsets[k] is (nx)
    } {
        const { N, nx, nu } = this;
        const nz = N * nu;

        // Initialize
        const Phi: number[][][] = [];
        const Psi: number[][][] = [];
        const offsets: number[][] = [];

        // k = 0: x_0 = x0
        Phi.push(MatrixUtils.eye(nx));
        Psi.push(MatrixUtils.zeros(nx, nz));
        offsets.push(new Array(nx).fill(0));

        // Recursion: x_{k+1} = A_k * x_k + B_k * u_k + c_k
        // = A_k * (Phi_k * x0 + Psi_k * U + offset_k) + B_k * u_k + c_k
        // = (A_k * Phi_k) * x0 + (A_k * Psi_k + [0...B_k...0]) * U + (A_k * offset_k + c_k)

        for (let k = 0; k < N; k++) {
            const { A, B, c } = linearizations[k];

            // Phi_{k+1} = A_k * Phi_k
            const PhiNext = MatrixUtils.matMul(A, Phi[k]);

            // Psi_{k+1} = A_k * Psi_k + [0...0, B_k, 0...0]
            const APsi = MatrixUtils.matMul(A, Psi[k]);
            const PsiNext = MatrixUtils.zeros(nx, nz);

            for (let i = 0; i < nx; i++) {
                for (let j = 0; j < nz; j++) {
                    PsiNext[i][j] = APsi[i][j];
                }
                // Add B_k at position k
                for (let j = 0; j < nu; j++) {
                    PsiNext[i][k * nu + j] += B[i][j];
                }
            }

            // offset_{k+1} = A_k * offset_k + c_k
            const offsetNext = new Array(nx).fill(0);
            for (let i = 0; i < nx; i++) {
                offsetNext[i] = c[i];
                for (let j = 0; j < nx; j++) {
                    offsetNext[i] += A[i][j] * offsets[k][j];
                }
            }

            Phi.push(PhiNext);
            Psi.push(PsiNext);
            offsets.push(offsetNext);
        }

        return { Phi, Psi, offsets };
    }

    /**
     * Build input constraint bounds
     */
    private buildInputConstraints(
        nominalInputs: MPCInput[]
    ): { lb: number[]; ub: number[] } {
        const { N, nu } = this;
        const { minThrust, maxThrust, maxRate, maxYawRate } = this.config;

        const lb = new Array(N * nu);
        const ub = new Array(N * nu);

        for (let k = 0; k < N; k++) {
            const uNom = this.model.inputToArray(nominalInputs[k]);

            // Thrust bounds (index 0)
            lb[k * nu + 0] = minThrust - uNom[0];
            ub[k * nu + 0] = maxThrust - uNom[0];

            // Roll rate bounds (index 1)
            lb[k * nu + 1] = -maxRate - uNom[1];
            ub[k * nu + 1] = maxRate - uNom[1];

            // Pitch rate bounds (index 2)
            lb[k * nu + 2] = -maxRate - uNom[2];
            ub[k * nu + 2] = maxRate - uNom[2];

            // Yaw rate bounds (index 3)
            lb[k * nu + 3] = -maxYawRate - uNom[3];
            ub[k * nu + 3] = maxYawRate - uNom[3];
        }

        return { lb, ub };
    }

    /**
     * Update trajectory with QP solution
     */
    private updateTrajectory(
        x0: MPCState,
        _states: MPCState[],  // Previous trajectory (for line search in future)
        inputs: MPCInput[],
        dU: number[],
        _linearizations: { A: number[][]; B: number[][]; c: number[] }[]  // For line search in future
    ): { newStates: MPCState[]; newInputs: MPCInput[]; improvement: number } {
        const { N, nu } = this;

        // Update inputs: u_new = u_old + du
        const newInputs: MPCInput[] = [];
        let improvement = 0;

        for (let k = 0; k < N; k++) {
            const uOld = this.model.inputToArray(inputs[k]);
            const duK = dU.slice(k * nu, (k + 1) * nu);

            const uNew = uOld.map((u, i) => u + duK[i]);
            newInputs.push(this.model.arrayToInput(uNew));

            // Track improvement
            improvement += duK.reduce((sum, d) => sum + d * d, 0);
        }
        improvement = Math.sqrt(improvement);

        // Rollout new states using nonlinear dynamics
        const newStates = this.model.rollout(x0, newInputs, this.config.dt);

        return { newStates, newInputs, improvement };
    }

    // =========================================
    // Public API for visualization
    // =========================================

    /**
     * Get predicted states for visualization
     */
    public getPredictedStates(): DroneState[] {
        return this.lastPredictedStates.map(s => ({
            position: { x: s.px, y: s.py, z: s.pz },
            orientation: { w: s.qw, x: s.qx, y: s.qy, z: s.qz },
            velocity: { x: s.vx, y: s.vy, z: s.vz },
            timestamp: 0,
        }));
    }

    /**
     * Get reference waypoints (for compatibility with old API)
     */
    public getReferenceWaypoints(): Waypoint[] {
        // Return empty - the demo should get these from trajectory generator
        return [];
    }

    /**
     * Get tracking error
     */
    public getTrackingError(state: DroneState, reference: Waypoint): number {
        const dx = reference.position.x - state.position.x;
        const dy = reference.position.y - state.position.y;
        const dz = reference.position.z - state.position.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /**
     * Reset controller state
     */
    public reset(): void {
        this.prevInputs = null;
        this.prevStates = null;
        this.lastPredictedStates = [];
    }
}
