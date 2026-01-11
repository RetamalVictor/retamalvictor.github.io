import { DroneState, ControlCommand, Vector3, Quaternion } from '../types';

/**
 * Pure Drone Dynamics Model
 *
 * This class contains ONLY the physics simulation - no visualization.
 * Used by both:
 * - RacingDrone (for simulation with visualization)
 * - MPC (for state prediction in optimization)
 *
 * State: [px, py, pz, qw, qx, qy, qz, vx, vy, vz] (10-state)
 * Input: [thrust, roll_rate, pitch_rate, yaw_rate] (4-input)
 *
 * Dynamics:
 * - Quaternion-based orientation
 * - First-order actuator dynamics
 * - Linear drag model
 * - Y-up coordinate system (Three.js convention)
 */
export interface DynamicsParams {
    mass: number;           // kg
    gravity: number;        // m/s²
    linearDrag: number;     // Linear drag coefficient
    tauThrust: number;      // Thrust time constant (s)
    tauRate: number;        // Angular rate time constant (s)
}

export const DEFAULT_DYNAMICS_PARAMS: DynamicsParams = {
    mass: 1.0,
    gravity: 9.81,
    linearDrag: 0.3,
    tauThrust: 0.04,   // 40ms
    tauRate: 0.03,     // 30ms
};

/**
 * Internal state representation for dynamics integration
 */
export interface DynamicsState {
    position: Vector3;
    velocity: Vector3;
    orientation: Quaternion;
    angularVelocity: Vector3;
    // Actuator states (for smooth transitions)
    currentThrust: number;
    currentRates: Vector3;
}

export class DroneDynamics {
    private params: DynamicsParams;
    private state: DynamicsState;

    constructor(params: Partial<DynamicsParams> = {}) {
        this.params = { ...DEFAULT_DYNAMICS_PARAMS, ...params };
        this.state = this.createInitialState();
    }

    /**
     * Create initial state at hover
     */
    private createInitialState(): DynamicsState {
        return {
            position: { x: 0, y: 2, z: 0 },
            velocity: { x: 0, y: 0, z: 0 },
            orientation: { w: 1, x: 0, y: 0, z: 0 },
            angularVelocity: { x: 0, y: 0, z: 0 },
            currentThrust: this.params.gravity,
            currentRates: { x: 0, y: 0, z: 0 },
        };
    }

    /**
     * Step dynamics forward by dt seconds
     *
     * @param command - Control command (thrust, rates)
     * @param dt - Time step in seconds
     */
    public step(command: ControlCommand, dt: number): void {
        // Limit dt to prevent instability
        dt = Math.min(dt, 0.05);

        // Apply first-order actuator dynamics
        this.updateActuators(command, dt);

        // Update orientation using quaternion integration
        this.integrateOrientation(dt);

        // Compute acceleration and update velocity/position
        this.integrateTranslation(dt);
    }

    /**
     * Apply first-order dynamics to actuators
     */
    private updateActuators(command: ControlCommand, dt: number): void {
        const alphaThrust = 1 - Math.exp(-dt / this.params.tauThrust);
        const alphaRate = 1 - Math.exp(-dt / this.params.tauRate);

        this.state.currentThrust += alphaThrust * (command.thrust - this.state.currentThrust);

        // Map rates to axes (Y-up convention):
        // Roll around Z, Pitch around X, Yaw around Y
        this.state.currentRates.x += alphaRate * (command.pitchRate - this.state.currentRates.x);
        this.state.currentRates.y += alphaRate * (command.yawRate - this.state.currentRates.y);
        this.state.currentRates.z += alphaRate * (command.rollRate - this.state.currentRates.z);

        // Angular velocity directly from rates
        this.state.angularVelocity = { ...this.state.currentRates };
    }

    /**
     * Integrate quaternion using angular velocity
     */
    private integrateOrientation(dt: number): void {
        const w = this.state.angularVelocity;
        const wMag = Math.sqrt(w.x * w.x + w.y * w.y + w.z * w.z);

        if (wMag < 1e-10) return;

        // Quaternion derivative: dq/dt = 0.5 * q * [0, w]
        const halfAngle = wMag * dt / 2;
        const sinHalf = Math.sin(halfAngle);
        const cosHalf = Math.cos(halfAngle);

        // Rotation quaternion for this timestep
        const dq: Quaternion = {
            w: cosHalf,
            x: w.x / wMag * sinHalf,
            y: w.y / wMag * sinHalf,
            z: w.z / wMag * sinHalf,
        };

        // Apply rotation: q_new = q * dq
        this.state.orientation = this.multiplyQuaternions(this.state.orientation, dq);
        this.normalizeQuaternion(this.state.orientation);
    }

    /**
     * Compute acceleration and integrate position/velocity
     */
    private integrateTranslation(dt: number): void {
        // Thrust in body frame (Y-up)
        const thrustBody: Vector3 = { x: 0, y: this.state.currentThrust, z: 0 };

        // Rotate to world frame
        const thrustWorld = this.rotateByQuaternion(thrustBody, this.state.orientation);

        // Acceleration: (thrust - gravity - drag) / mass
        const accel: Vector3 = {
            x: thrustWorld.x / this.params.mass - this.params.linearDrag * this.state.velocity.x,
            y: thrustWorld.y / this.params.mass - this.params.gravity - this.params.linearDrag * this.state.velocity.y,
            z: thrustWorld.z / this.params.mass - this.params.linearDrag * this.state.velocity.z,
        };

        // Update velocity: v = v + a*dt
        this.state.velocity.x += accel.x * dt;
        this.state.velocity.y += accel.y * dt;
        this.state.velocity.z += accel.z * dt;

        // Update position: p = p + v*dt
        this.state.position.x += this.state.velocity.x * dt;
        this.state.position.y += this.state.velocity.y * dt;
        this.state.position.z += this.state.velocity.z * dt;
    }

    /**
     * Multiply two quaternions: result = q1 * q2
     */
    private multiplyQuaternions(q1: Quaternion, q2: Quaternion): Quaternion {
        return {
            w: q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
            x: q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
            y: q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
            z: q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        };
    }

    /**
     * Normalize quaternion in-place
     */
    private normalizeQuaternion(q: Quaternion): void {
        const mag = Math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
        if (mag > 1e-10) {
            q.w /= mag;
            q.x /= mag;
            q.y /= mag;
            q.z /= mag;
        }
    }

    /**
     * Rotate vector by quaternion: v' = q * v * q^-1
     */
    private rotateByQuaternion(v: Vector3, q: Quaternion): Vector3 {
        // For unit quaternion, q^-1 = conjugate
        const qConj: Quaternion = { w: q.w, x: -q.x, y: -q.y, z: -q.z };

        // v as quaternion with w=0
        const vQuat: Quaternion = { w: 0, x: v.x, y: v.y, z: v.z };

        // q * v * q^-1
        const tmp = this.multiplyQuaternions(q, vQuat);
        const result = this.multiplyQuaternions(tmp, qConj);

        return { x: result.x, y: result.y, z: result.z };
    }

    // =========================================
    // Public API
    // =========================================

    /**
     * Get current state as DroneState
     */
    public getState(): DroneState {
        return {
            position: { ...this.state.position },
            orientation: { ...this.state.orientation },
            velocity: { ...this.state.velocity },
            timestamp: Date.now() * 1000,
        };
    }

    /**
     * Get full internal state (for MPC warm-starting)
     */
    public getFullState(): DynamicsState {
        return {
            position: { ...this.state.position },
            velocity: { ...this.state.velocity },
            orientation: { ...this.state.orientation },
            angularVelocity: { ...this.state.angularVelocity },
            currentThrust: this.state.currentThrust,
            currentRates: { ...this.state.currentRates },
        };
    }

    /**
     * Set state (for initialization or MPC rollouts)
     */
    public setState(state: Partial<DynamicsState>): void {
        if (state.position) this.state.position = { ...state.position };
        if (state.velocity) this.state.velocity = { ...state.velocity };
        if (state.orientation) this.state.orientation = { ...state.orientation };
        if (state.angularVelocity) this.state.angularVelocity = { ...state.angularVelocity };
        if (state.currentThrust !== undefined) this.state.currentThrust = state.currentThrust;
        if (state.currentRates) this.state.currentRates = { ...state.currentRates };
    }

    /**
     * Reset to initial state
     */
    public reset(position?: Vector3): void {
        this.state = this.createInitialState();
        if (position) {
            this.state.position = { ...position };
        }
    }

    /**
     * Set position directly
     */
    public setPosition(x: number, y: number, z: number): void {
        this.state.position = { x, y, z };
    }

    /**
     * Set velocity directly
     */
    public setVelocity(vx: number, vy: number, vz: number): void {
        this.state.velocity = { x: vx, y: vy, z: vz };
    }

    /**
     * Set heading (yaw) - rotation around Y axis
     */
    public setHeading(yaw: number): void {
        const halfYaw = yaw / 2;
        this.state.orientation = {
            w: Math.cos(halfYaw),
            x: 0,
            y: Math.sin(halfYaw),
            z: 0,
        };
    }

    /**
     * Get speed magnitude
     */
    public getSpeed(): number {
        const v = this.state.velocity;
        return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    /**
     * Get dynamics parameters (for MPC to match)
     */
    public getParams(): DynamicsParams {
        return { ...this.params };
    }

    // =========================================
    // Static utilities for MPC prediction
    // =========================================

    /**
     * Predict state forward without modifying internal state
     * Useful for MPC rollouts
     *
     * @param state - Initial state
     * @param command - Control command
     * @param dt - Time step
     * @param params - Dynamics parameters
     * @returns New state after dt
     */
    public static predict(
        state: DynamicsState,
        command: ControlCommand,
        dt: number,
        params: DynamicsParams
    ): DynamicsState {
        // Create a temporary dynamics instance
        const dynamics = new DroneDynamics(params);
        dynamics.setState(state);
        dynamics.step(command, dt);
        return dynamics.getFullState();
    }

    /**
     * Predict trajectory over horizon
     *
     * @param initialState - Starting state
     * @param commands - Sequence of commands
     * @param dt - Time step between commands
     * @param params - Dynamics parameters
     * @returns Array of predicted states
     */
    public static predictTrajectory(
        initialState: DynamicsState,
        commands: ControlCommand[],
        dt: number,
        params: DynamicsParams
    ): DynamicsState[] {
        const states: DynamicsState[] = [initialState];
        const dynamics = new DroneDynamics(params);
        dynamics.setState(initialState);

        for (const cmd of commands) {
            dynamics.step(cmd, dt);
            states.push(dynamics.getFullState());
        }

        return states;
    }
}
