import { Vector3, Quaternion, ControlCommand } from '../types';
import { DynamicsParams, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';

/**
 * MPC State Vector (14-dimensional with quaternion attitude)
 *
 * Using quaternions for attitude eliminates gimbal lock issues
 * that occur with Euler angles at certain orientations.
 *
 * - Position: [px, py, pz] (3)
 * - Velocity: [vx, vy, vz] (3)
 * - Quaternion: [qw, qx, qy, qz] (4) - attitude
 * - Actuator states: [thrust, roll_rate, pitch_rate, yaw_rate] (4)
 *
 * Total: 14 states
 */
export interface MPCState {
    // Position (world frame)
    px: number;
    py: number;
    pz: number;
    // Velocity (world frame)
    vx: number;
    vy: number;
    vz: number;
    // Quaternion (w, x, y, z) - body to world rotation
    qw: number;
    qx: number;
    qy: number;
    qz: number;
    // Actuator states
    thrust: number;
    rollRate: number;
    pitchRate: number;
    yawRate: number;
}

/**
 * MPC Input Vector (4-dimensional)
 */
export interface MPCInput {
    thrust: number;      // Commanded thrust (m/s²)
    rollRate: number;    // Commanded roll rate (rad/s)
    pitchRate: number;   // Commanded pitch rate (rad/s)
    yawRate: number;     // Commanded yaw rate (rad/s)
}

/**
 * Linearized dynamics matrices at an operating point
 *
 * x_{k+1} = A * x_k + B * u_k + c
 */
export interface LinearizedDynamics {
    A: number[][];  // State Jacobian (14x14)
    B: number[][];  // Input Jacobian (14x4)
    c: number[];    // Affine term (14)
}

/**
 * MPC Dynamics Model with Quaternion Attitude
 *
 * Provides:
 * - Nonlinear dynamics propagation using quaternion kinematics
 * - Linearization for SQP
 * - State/input conversion utilities
 */
export class MPCModel {
    public readonly params: DynamicsParams;

    // State dimension (14 with quaternion)
    public readonly nx = 14;
    // Input dimension
    public readonly nu = 4;

    constructor(params: Partial<DynamicsParams> = {}) {
        this.params = { ...DEFAULT_DYNAMICS_PARAMS, ...params };
    }

    /**
     * Nonlinear dynamics: x_{k+1} = f(x_k, u_k, dt)
     *
     * Uses quaternion kinematics for attitude propagation:
     * dq/dt = 0.5 * q ⊗ [0, ω_body]
     */
    public dynamics(state: MPCState, input: MPCInput, dt: number): MPCState {
        const { mass, gravity, linearDrag, tauThrust, tauRate } = this.params;

        // Actuator dynamics (first-order)
        const alphaThrust = 1 - Math.exp(-dt / tauThrust);
        const alphaRate = 1 - Math.exp(-dt / tauRate);

        const newThrust = state.thrust + alphaThrust * (input.thrust - state.thrust);
        const newRollRate = state.rollRate + alphaRate * (input.rollRate - state.rollRate);
        const newPitchRate = state.pitchRate + alphaRate * (input.pitchRate - state.pitchRate);
        const newYawRate = state.yawRate + alphaRate * (input.yawRate - state.yawRate);

        // Quaternion kinematics: dq/dt = 0.5 * q ⊗ [0, ω_body]
        // Body angular velocity: ω_body = [pitchRate, yawRate, rollRate] (X, Y, Z in body frame)
        // For Y-up: roll around Z, pitch around X, yaw around Y
        const wx = newPitchRate;  // Pitch around X
        const wy = newYawRate;    // Yaw around Y
        const wz = newRollRate;   // Roll around Z

        // Quaternion derivative: qdot = 0.5 * q ⊗ [0, wx, wy, wz]
        // q ⊗ p = [q.w*p.w - q.v·p.v, q.w*p.v + p.w*q.v + q.v×p.v]
        // For p = [0, wx, wy, wz]:
        const qdot_w = 0.5 * (-state.qx * wx - state.qy * wy - state.qz * wz);
        const qdot_x = 0.5 * (state.qw * wx + state.qy * wz - state.qz * wy);
        const qdot_y = 0.5 * (state.qw * wy + state.qz * wx - state.qx * wz);
        const qdot_z = 0.5 * (state.qw * wz + state.qx * wy - state.qy * wx);

        // Integrate quaternion
        let newQw = state.qw + qdot_w * dt;
        let newQx = state.qx + qdot_x * dt;
        let newQy = state.qy + qdot_y * dt;
        let newQz = state.qz + qdot_z * dt;

        // Normalize quaternion to maintain unit constraint
        const qNorm = Math.sqrt(newQw * newQw + newQx * newQx + newQy * newQy + newQz * newQz);
        newQw /= qNorm;
        newQx /= qNorm;
        newQy /= qNorm;
        newQz /= qNorm;

        // Compute thrust direction in world frame using quaternion rotation
        // Body thrust is [0, T, 0] (Y-up), rotate by quaternion
        // v_world = q * v_body * q^-1
        // For v_body = [0, T, 0]:
        const thrustWorld = this.rotateByQuaternion(
            { x: 0, y: newThrust, z: 0 },
            { w: newQw, x: newQx, y: newQy, z: newQz }
        );

        // Acceleration = (thrust - gravity - drag) / mass
        const ax = thrustWorld.x / mass - linearDrag * state.vx;
        const ay = thrustWorld.y / mass - gravity - linearDrag * state.vy;
        const az = thrustWorld.z / mass - linearDrag * state.vz;

        // Velocity update
        const newVx = state.vx + ax * dt;
        const newVy = state.vy + ay * dt;
        const newVz = state.vz + az * dt;

        // Position update
        const newPx = state.px + state.vx * dt + 0.5 * ax * dt * dt;
        const newPy = state.py + state.vy * dt + 0.5 * ay * dt * dt;
        const newPz = state.pz + state.vz * dt + 0.5 * az * dt * dt;

        return {
            px: newPx,
            py: newPy,
            pz: newPz,
            vx: newVx,
            vy: newVy,
            vz: newVz,
            qw: newQw,
            qx: newQx,
            qy: newQy,
            qz: newQz,
            thrust: newThrust,
            rollRate: newRollRate,
            pitchRate: newPitchRate,
            yawRate: newYawRate,
        };
    }

    /**
     * Rotate vector by quaternion: v' = q * v * q^-1
     */
    private rotateByQuaternion(v: Vector3, q: Quaternion): Vector3 {
        // Using quaternion rotation formula
        const { w, x, y, z } = q;
        const { x: vx, y: vy, z: vz } = v;

        // t = 2 * cross(q.xyz, v)
        const tx = 2 * (y * vz - z * vy);
        const ty = 2 * (z * vx - x * vz);
        const tz = 2 * (x * vy - y * vx);

        // result = v + w * t + cross(q.xyz, t)
        return {
            x: vx + w * tx + (y * tz - z * ty),
            y: vy + w * ty + (z * tx - x * tz),
            z: vz + w * tz + (x * ty - y * tx),
        };
    }

    /**
     * Linearize dynamics around operating point
     *
     * Computes Jacobians using numerical differentiation.
     * For quaternions, we use a 4-parameter perturbation and renormalize.
     */
    public linearize(stateOp: MPCState, inputOp: MPCInput, dt: number): LinearizedDynamics {
        const eps = 1e-6;

        // Compute f(x_op, u_op)
        const f0 = this.stateToArray(this.dynamics(stateOp, inputOp, dt));

        // Compute A = df/dx using finite differences
        const A: number[][] = [];
        for (let i = 0; i < this.nx; i++) {
            A.push(new Array(this.nx).fill(0));
        }

        for (let j = 0; j < this.nx; j++) {
            const stateArray = this.stateToArray(stateOp);
            stateArray[j] += eps;

            // Renormalize quaternion if we perturbed it
            if (j >= 6 && j <= 9) {
                const qNorm = Math.sqrt(
                    stateArray[6] * stateArray[6] +
                    stateArray[7] * stateArray[7] +
                    stateArray[8] * stateArray[8] +
                    stateArray[9] * stateArray[9]
                );
                stateArray[6] /= qNorm;
                stateArray[7] /= qNorm;
                stateArray[8] /= qNorm;
                stateArray[9] /= qNorm;
            }

            const statePerturbedNew = this.arrayToState(stateArray);
            const fPerturbed = this.stateToArray(this.dynamics(statePerturbedNew, inputOp, dt));

            for (let i = 0; i < this.nx; i++) {
                A[i][j] = (fPerturbed[i] - f0[i]) / eps;
            }
        }

        // Compute B = df/du using finite differences
        const B: number[][] = [];
        for (let i = 0; i < this.nx; i++) {
            B.push(new Array(this.nu).fill(0));
        }

        for (let j = 0; j < this.nu; j++) {
            const inputArray = this.inputToArray(inputOp);
            inputArray[j] += eps;
            const inputPerturbedNew = this.arrayToInput(inputArray);

            const fPerturbed = this.stateToArray(this.dynamics(stateOp, inputPerturbedNew, dt));

            for (let i = 0; i < this.nx; i++) {
                B[i][j] = (fPerturbed[i] - f0[i]) / eps;
            }
        }

        // Compute affine term: c = f(x_op, u_op) - A * x_op - B * u_op
        const xOp = this.stateToArray(stateOp);
        const uOp = this.inputToArray(inputOp);
        const c: number[] = new Array(this.nx).fill(0);

        for (let i = 0; i < this.nx; i++) {
            c[i] = f0[i];
            for (let j = 0; j < this.nx; j++) {
                c[i] -= A[i][j] * xOp[j];
            }
            for (let j = 0; j < this.nu; j++) {
                c[i] -= B[i][j] * uOp[j];
            }
        }

        return { A, B, c };
    }

    /**
     * Propagate state trajectory using nonlinear dynamics
     */
    public rollout(initialState: MPCState, inputs: MPCInput[], dt: number): MPCState[] {
        const states: MPCState[] = [initialState];
        let state = initialState;

        for (const input of inputs) {
            state = this.dynamics(state, input, dt);
            states.push(state);
        }

        return states;
    }

    // =========================================
    // Quaternion utilities
    // =========================================

    /**
     * Compute quaternion error: q_error = q_ref^* ⊗ q_actual
     *
     * For small errors, q_error ≈ [1, θx/2, θy/2, θz/2]
     * where θ is the axis-angle error in body frame.
     */
    public quaternionError(qActual: Quaternion, qRef: Quaternion): Quaternion {
        // q_ref^* (conjugate)
        const qRefConj = { w: qRef.w, x: -qRef.x, y: -qRef.y, z: -qRef.z };

        // q_error = q_ref^* ⊗ q_actual
        return this.quaternionMultiply(qRefConj, qActual);
    }

    /**
     * Quaternion multiplication: q1 ⊗ q2
     */
    public quaternionMultiply(q1: Quaternion, q2: Quaternion): Quaternion {
        return {
            w: q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            x: q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            y: q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,
            z: q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x,
        };
    }

    /**
     * Create quaternion from axis-angle representation
     */
    public axisAngleToQuaternion(axis: Vector3, angle: number): Quaternion {
        const halfAngle = angle / 2;
        const s = Math.sin(halfAngle);
        return {
            w: Math.cos(halfAngle),
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
        };
    }

    /**
     * Create quaternion from yaw angle (rotation around Y axis for Y-up)
     */
    public yawToQuaternion(yaw: number): Quaternion {
        const halfYaw = yaw / 2;
        return {
            w: Math.cos(halfYaw),
            x: 0,
            y: Math.sin(halfYaw),
            z: 0,
        };
    }

    /**
     * Extract yaw from quaternion (rotation around Y axis for Y-up)
     */
    public quaternionToYaw(q: Quaternion): number {
        // For Y-up, yaw is rotation around Y
        const sinYaw = 2 * (q.w * q.y + q.z * q.x);
        const cosYaw = 1 - 2 * (q.x * q.x + q.y * q.y);
        return Math.atan2(sinYaw, cosYaw);
    }

    // =========================================
    // Conversion utilities
    // =========================================

    /**
     * Convert MPCState to array (14 elements)
     */
    public stateToArray(state: MPCState): number[] {
        return [
            state.px, state.py, state.pz,
            state.vx, state.vy, state.vz,
            state.qw, state.qx, state.qy, state.qz,
            state.thrust, state.rollRate, state.pitchRate, state.yawRate,
        ];
    }

    /**
     * Convert array to MPCState
     */
    public arrayToState(arr: number[]): MPCState {
        return {
            px: arr[0],
            py: arr[1],
            pz: arr[2],
            vx: arr[3],
            vy: arr[4],
            vz: arr[5],
            qw: arr[6],
            qx: arr[7],
            qy: arr[8],
            qz: arr[9],
            thrust: arr[10],
            rollRate: arr[11],
            pitchRate: arr[12],
            yawRate: arr[13],
        };
    }

    /**
     * Convert MPCInput to array
     */
    public inputToArray(input: MPCInput): number[] {
        return [input.thrust, input.rollRate, input.pitchRate, input.yawRate];
    }

    /**
     * Convert array to MPCInput
     */
    public arrayToInput(arr: number[]): MPCInput {
        return {
            thrust: arr[0],
            rollRate: arr[1],
            pitchRate: arr[2],
            yawRate: arr[3],
        };
    }

    /**
     * Convert ControlCommand to MPCInput
     */
    public commandToInput(cmd: ControlCommand): MPCInput {
        return {
            thrust: cmd.thrust,
            rollRate: cmd.rollRate,
            pitchRate: cmd.pitchRate,
            yawRate: cmd.yawRate,
        };
    }

    /**
     * Convert MPCInput to ControlCommand
     */
    public inputToCommand(input: MPCInput, timestamp: number = 0): ControlCommand {
        return {
            thrust: input.thrust,
            rollRate: input.rollRate,
            pitchRate: input.pitchRate,
            yawRate: input.yawRate,
            timestamp,
        };
    }

    /**
     * Create MPCState from DroneState (quaternion is used directly)
     */
    public fromDroneState(
        position: Vector3,
        velocity: Vector3,
        orientation: Quaternion,
        actuatorThrust: number = this.params.gravity,
        actuatorRates: Vector3 = { x: 0, y: 0, z: 0 }
    ): MPCState {
        return {
            px: position.x,
            py: position.y,
            pz: position.z,
            vx: velocity.x,
            vy: velocity.y,
            vz: velocity.z,
            qw: orientation.w,
            qx: orientation.x,
            qy: orientation.y,
            qz: orientation.z,
            thrust: actuatorThrust,
            rollRate: actuatorRates.z,  // Roll around Z
            pitchRate: actuatorRates.x, // Pitch around X
            yawRate: actuatorRates.y,   // Yaw around Y
        };
    }

    /**
     * Create a hover state at given position
     */
    public createHoverState(position: Vector3, yaw: number = 0): MPCState {
        const q = this.yawToQuaternion(yaw);
        return {
            px: position.x,
            py: position.y,
            pz: position.z,
            vx: 0,
            vy: 0,
            vz: 0,
            qw: q.w,
            qx: q.x,
            qy: q.y,
            qz: q.z,
            thrust: this.params.gravity,
            rollRate: 0,
            pitchRate: 0,
            yawRate: 0,
        };
    }

    /**
     * Create hover input
     */
    public createHoverInput(): MPCInput {
        return {
            thrust: this.params.gravity,
            rollRate: 0,
            pitchRate: 0,
            yawRate: 0,
        };
    }

    /**
     * Get thrust direction from quaternion state
     */
    public getThrustDirection(state: MPCState): Vector3 {
        const q = { w: state.qw, x: state.qx, y: state.qy, z: state.qz };
        // Body thrust is [0, 1, 0] (unit Y), rotate by quaternion
        return this.rotateByQuaternion({ x: 0, y: 1, z: 0 }, q);
    }
}
