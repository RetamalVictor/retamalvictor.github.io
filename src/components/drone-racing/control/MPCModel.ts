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
     * Linearize dynamics using analytical Jacobians (Stage A)
     *
     * Computes most Jacobian blocks analytically for performance.
     * A_vq (velocity w.r.t. quaternion) is kept numerical due to
     * sensitivity to quaternion convention errors.
     *
     * State indices:
     * - [0-2]: position (px, py, pz)
     * - [3-5]: velocity (vx, vy, vz)
     * - [6-9]: quaternion (qw, qx, qy, qz)
     * - [10-13]: actuators (thrust, rollRate, pitchRate, yawRate)
     */
    public linearizeAnalytical(stateOp: MPCState, inputOp: MPCInput, dt: number): LinearizedDynamics {
        const { mass, linearDrag, tauThrust, tauRate } = this.params;

        // Actuator dynamics coefficients (exponential discretization)
        const alphaThrust = 1 - Math.exp(-dt / tauThrust);
        const alphaRate = 1 - Math.exp(-dt / tauRate);

        // Initialize matrices
        const A: number[][] = [];
        const B: number[][] = [];
        for (let i = 0; i < this.nx; i++) {
            A.push(new Array(this.nx).fill(0));
            B.push(new Array(this.nu).fill(0));
        }

        // Extract state components
        const { qw, qx, qy, qz, rollRate, pitchRate, yawRate } = stateOp;

        // Compute new actuator states (used in quaternion block)
        const newRollRate = rollRate + alphaRate * (inputOp.rollRate - rollRate);
        const newPitchRate = pitchRate + alphaRate * (inputOp.pitchRate - pitchRate);
        const newYawRate = yawRate + alphaRate * (inputOp.yawRate - yawRate);

        // Body angular velocity
        const wx = newPitchRate;  // Pitch around X
        const wy = newYawRate;    // Yaw around Y
        const wz = newRollRate;   // Roll around Z

        // ============================================
        // Position block (trivial): p_{k+1} = p_k + v_k*dt + 0.5*a*dt²
        // ============================================
        // A_pp = I₃
        A[0][0] = 1; A[1][1] = 1; A[2][2] = 1;
        // A_pv = dt*I₃ (ignoring 0.5*a*dt² contribution for simplicity)
        A[0][3] = dt; A[1][4] = dt; A[2][5] = dt;

        // ============================================
        // Velocity block
        // v_{k+1} = v_k + (thrust_world/mass - g - drag*v_k) * dt
        // ============================================
        // A_vv = I₃ - dt*drag*I₃
        const dragTerm = 1 - dt * linearDrag;
        A[3][3] = dragTerm; A[4][4] = dragTerm; A[5][5] = dragTerm;

        // A_vq and A_v_thrust: computed analytically after quaternion block (need qhat)

        // ============================================
        // Quaternion block
        // q_{k+1} = normalize(q_k + 0.5*dt * q_k ⊗ [0, ω])
        // ============================================
        // A_qq_unnorm = I₄ + 0.5*dt*Ω(ω)
        // Normalization projection: ∂(q/||q||)/∂q = (I - q̂q̂ᵀ) for ||q||≈1
        // Full: A_qq = (I - q_new*q_new^T) * A_qq_unnorm
        //
        // Ω(ω) = [  0  -wx -wy -wz ]
        //        [ wx   0  wz -wy ]
        //        [ wy -wz   0  wx ]
        //        [ wz  wy -wx   0 ]
        const halfDt = 0.5 * dt;

        // Build unnormalized A_qq
        const Aqq_unnorm: number[][] = [
            [1,          -halfDt*wx, -halfDt*wy, -halfDt*wz],
            [halfDt*wx,  1,          halfDt*wz,  -halfDt*wy],
            [halfDt*wy,  -halfDt*wz, 1,          halfDt*wx],
            [halfDt*wz,  halfDt*wy,  -halfDt*wx, 1]
        ];

        // Compute the unnormalized quaternion after update (q_u = q + dt*qdot)
        // This matches the dynamics code exactly before normalization
        const qdot_w = 0.5 * (-qx * wx - qy * wy - qz * wz);
        const qdot_x = 0.5 * (qw * wx + qy * wz - qz * wy);
        const qdot_y = 0.5 * (qw * wy + qz * wx - qx * wz);
        const qdot_z = 0.5 * (qw * wz + qx * wy - qy * wx);

        const qu_w = qw + qdot_w * dt;
        const qu_x = qx + qdot_x * dt;
        const qu_y = qy + qdot_y * dt;
        const qu_z = qz + qdot_z * dt;

        const qu_norm = Math.sqrt(qu_w*qu_w + qu_x*qu_x + qu_y*qu_y + qu_z*qu_z) || 1;
        const qhat = [qu_w/qu_norm, qu_x/qu_norm, qu_y/qu_norm, qu_z/qu_norm];

        // Build normalization Jacobian: Jnorm = (I - qhat*qhat^T) / ||qu||
        const Jnorm: number[][] = [];
        for (let i = 0; i < 4; i++) {
            Jnorm.push([]);
            for (let j = 0; j < 4; j++) {
                Jnorm[i][j] = ((i === j ? 1 : 0) - qhat[i] * qhat[j]) / qu_norm;
            }
        }

        // Apply normalization Jacobian to A_qq: A_qq = Jnorm * Aqq_unnorm
        const A_qq: number[][] = [[], [], [], []];
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let s = 0;
                for (let k = 0; k < 4; k++) {
                    s += Jnorm[i][k] * Aqq_unnorm[k][j];
                }
                A[6+i][6+j] = s;
                A_qq[i][j] = s;
            }
        }

        // ============================================
        // A_vq: Analytical velocity derivative w.r.t. quaternion state
        // Chain: q_state → q_new → thrustDir → v_new
        // A_vq = (dt/mass) * newThrust * dDir_dq * A_qq
        // ============================================
        // Thrust direction at qhat: d = R(qhat)*[0,1,0] = [2(xy-wz), 1-2(x²+z²), 2(wx+yz)]
        const qhw = qhat[0], qhx = qhat[1], qhy = qhat[2], qhz = qhat[3];
        const thrustDirQhat = {
            x: 2 * (qhx * qhy - qhw * qhz),
            y: 1 - 2 * (qhx * qhx + qhz * qhz),
            z: 2 * (qhw * qhx + qhy * qhz)
        };

        // Analytical ∂d/∂qhat (3x4 matrix)
        const dDir_dqhat = [
            [-2*qhz,  2*qhy,  2*qhx, -2*qhw],  // ∂dx/∂(w,x,y,z)
            [     0, -4*qhx,      0, -4*qhz],  // ∂dy/∂(w,x,y,z)
            [ 2*qhx,  2*qhw,  2*qhz,  2*qhy]   // ∂dz/∂(w,x,y,z)
        ];

        // newThrust at operating point
        const newThrustA = (1 - alphaThrust) * stateOp.thrust + alphaThrust * inputOp.thrust;
        const velScale = dt / mass;

        // A_vq = velScale * newThrust * dDir_dqhat * A_qq  (3x4 = 3x4 * 4x4)
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 4; j++) {
                let s = 0;
                for (let k = 0; k < 4; k++) {
                    s += dDir_dqhat[i][k] * A_qq[k][j];
                }
                A[3+i][6+j] = velScale * newThrustA * s;
            }
        }

        // A_v_thrust: ∂v/∂T_state = (dt/mass) * (1-αT) * thrustDir(qhat)
        const dv_dT = velScale * (1 - alphaThrust);
        A[3][10] = dv_dT * thrustDirQhat.x;
        A[4][10] = dv_dT * thrustDirQhat.y;
        A[5][10] = dv_dT * thrustDirQhat.z;

        // A_q_ω: quaternion derivative w.r.t. actuator rate states
        // In dynamics: newRate = (1-αR)*rate + αR*rate_cmd
        // So ∂(newRate)/∂(state.rate) = (1-αR)
        // dq_unnorm/dω = 0.5*dt*Q(q) where ω = [wx, wy, wz] = [newPitchRate, newYawRate, newRollRate]
        // Q(q) = [ -qx -qy -qz ]
        //        [  qw  qz -qy ]
        //        [ -qz  qw  qx ]
        //        [  qy -qx  qw ]
        // Actuator indices: 10=thrust, 11=rollRate, 12=pitchRate, 13=yawRate
        // wx = pitchRate (idx 12), wy = yawRate (idx 13), wz = rollRate (idx 11)
        // Apply Jnorm to get normalized derivative
        // Q(q) = [∂qdot/∂wx, ∂qdot/∂wy, ∂qdot/∂wz] (columns for each ω component)
        // From qdot = 0.5*q⊗[0,ω]:
        //   ∂qdot/∂wx = 0.5*[-qx, qw, qz, -qy]
        //   ∂qdot/∂wy = 0.5*[-qy, -qz, qw, qx]
        //   ∂qdot/∂wz = 0.5*[-qz, qy, -qx, qw]
        const Qq = [
            [-qx, -qy, -qz],   // qdot_w row
            [ qw, -qz,  qy],   // qdot_x row
            [ qz,  qw, -qx],   // qdot_y row
            [-qy,  qx,  qw]    // qdot_z row
        ];
        const dq_dw_state = halfDt * (1 - alphaRate);  // For state rate derivatives
        const dq_dw_input = halfDt * alphaRate;        // For input rate derivatives

        // Build unnormalized derivatives (4 rows x 3 rate columns)
        // Column order: [rollRate(idx11/wz), pitchRate(idx12/wx), yawRate(idx13/wy)]
        const Aq_rates_unnorm: number[][] = [];
        for (let i = 0; i < 4; i++) {
            Aq_rates_unnorm.push([
                dq_dw_state * Qq[i][2],  // d(qi)/d(rollRate) via wz
                dq_dw_state * Qq[i][0],  // d(qi)/d(pitchRate) via wx
                dq_dw_state * Qq[i][1],  // d(qi)/d(yawRate) via wy
            ]);
        }

        // Apply normalization Jacobian: A_q_rates = Jnorm * Aq_rates_unnorm
        for (let j = 0; j < 3; j++) {  // For each rate column
            for (let i = 0; i < 4; i++) {
                let s = 0;
                for (let k = 0; k < 4; k++) {
                    s += Jnorm[i][k] * Aq_rates_unnorm[k][j];
                }
                A[6+i][11+j] = s;
            }
        }

        // ============================================
        // Actuator block (first-order exponential discretization)
        // T_{k+1} = (1-α)*T_k + α*T_cmd
        // ============================================
        A[10][10] = 1 - alphaThrust;  // dT/dT
        A[11][11] = 1 - alphaRate;    // dRollRate/dRollRate
        A[12][12] = 1 - alphaRate;    // dPitchRate/dPitchRate
        A[13][13] = 1 - alphaRate;    // dYawRate/dYawRate

        // ============================================
        // B matrix (input Jacobian)
        // ============================================
        // Actuator rows
        B[10][0] = alphaThrust;  // dT/dT_cmd
        B[11][1] = alphaRate;    // dRollRate/dRollRate_cmd
        B[12][2] = alphaRate;    // dPitchRate/dPitchRate_cmd
        B[13][3] = alphaRate;    // dYawRate/dYawRate_cmd

        // ============================================
        // B matrix: Use qhat thrust direction (thrustDirQhat, dDir_dqhat already computed above)
        // ============================================

        // Velocity w.r.t. input thrust: dv/d(T_cmd) = (dt/mass) * d(qhat) * αT
        const dv_dTcmd = (dt / mass) * alphaThrust;
        B[3][0] = dv_dTcmd * thrustDirQhat.x;
        B[4][0] = dv_dTcmd * thrustDirQhat.y;
        B[5][0] = dv_dTcmd * thrustDirQhat.z;

        // Position w.r.t. input thrust: dp/d(T_cmd) = 0.5*dt²/mass * d(qhat) * αT
        const dp_dTcmd = 0.5 * dt * dt / mass * alphaThrust;
        B[0][0] = dp_dTcmd * thrustDirQhat.x;
        B[1][0] = dp_dTcmd * thrustDirQhat.y;
        B[2][0] = dp_dTcmd * thrustDirQhat.z;

        // Quaternion rows: dq/d(input_rate_cmd) = Jnorm * dq_unnorm/d(newRate) * d(newRate)/d(input)
        // d(newRate)/d(input) = αR
        // Input order: [thrust, rollRate_cmd, pitchRate_cmd, yawRate_cmd]
        const Bq_rates_unnorm: number[][] = [];
        for (let i = 0; i < 4; i++) {
            Bq_rates_unnorm.push([
                dq_dw_input * Qq[i][2],  // d(qi)/d(rollRate_cmd) via wz
                dq_dw_input * Qq[i][0],  // d(qi)/d(pitchRate_cmd) via wx
                dq_dw_input * Qq[i][1],  // d(qi)/d(yawRate_cmd) via wy
            ]);
        }

        // Apply normalization Jacobian to B quaternion rows
        for (let j = 0; j < 3; j++) {  // For each input rate column (1,2,3 in B)
            for (let i = 0; i < 4; i++) {
                let s = 0;
                for (let k = 0; k < 4; k++) {
                    s += Jnorm[i][k] * Bq_rates_unnorm[k][j];
                }
                B[6+i][1+j] = s;
            }
        }

        // ============================================
        // Propagate rate_cmd through quaternion to velocity/position
        // Chain: rate_cmd → qhat → thrust direction → acceleration → v, p
        // ∂thrust/∂rate_cmd = T_new * (∂d/∂qhat) * (∂qhat/∂rate_cmd)
        // ============================================
        const newThrust = (1 - alphaThrust) * stateOp.thrust + alphaThrust * inputOp.thrust;

        // For each rate command column j (1=roll, 2=pitch, 3=yaw in B)
        for (let j = 0; j < 3; j++) {
            // Get dqhat/du_rate from B quaternion rows (already computed above)
            const dq_du = [B[6][1+j], B[7][1+j], B[8][1+j], B[9][1+j]];

            // Compute d(thrustWorld)/d(rate_cmd) = T_new * (dDir/dq) * (dq/du)
            const dthrustWorld_du = [0, 0, 0];
            for (let i = 0; i < 3; i++) {
                for (let k = 0; k < 4; k++) {
                    dthrustWorld_du[i] += newThrust * dDir_dqhat[i][k] * dq_du[k];
                }
            }

            // Velocity rows: dv/du = (dt/mass) * d(thrustWorld)/du
            B[3][1+j] += (dt / mass) * dthrustWorld_du[0];
            B[4][1+j] += (dt / mass) * dthrustWorld_du[1];
            B[5][1+j] += (dt / mass) * dthrustWorld_du[2];

            // Position rows: dp/du = 0.5*dt²/mass * d(thrustWorld)/du
            B[0][1+j] += 0.5 * dt * dt / mass * dthrustWorld_du[0];
            B[1][1+j] += 0.5 * dt * dt / mass * dthrustWorld_du[1];
            B[2][1+j] += 0.5 * dt * dt / mass * dthrustWorld_du[2];
        }

        // ============================================
        // Compute affine term: c = f(x_op, u_op) - A * x_op - B * u_op
        // ============================================
        const f0 = this.stateToArray(this.dynamics(stateOp, inputOp, dt));
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
