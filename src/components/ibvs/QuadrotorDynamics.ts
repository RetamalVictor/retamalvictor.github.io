/**
 * Underactuated Quadrotor Dynamics Model
 *
 * State: [x, y, z, vx, vy, vz, rotX, rotY, rotZ, wx, wy, wz]
 *
 * Coordinate frames (Three.js convention):
 * - World: X right, Y up, Z forward
 * - rotation.x (rotX): tilt forward/backward (positive = nose down)
 * - rotation.y (rotY): yaw/heading (positive = turn left)
 * - rotation.z (rotZ): roll left/right (positive = roll left)
 *
 * The quadrotor is underactuated: 4 inputs (thrust + 3 torques) for 6 DOF.
 * - Forward motion (+Z) requires positive rotX (tilt forward)
 * - Rightward motion (+X) requires negative rotZ (roll right)
 * - Altitude (Y) controlled by thrust
 * - Heading (rotY) controlled directly
 */
export class QuadrotorDynamics {
    // Physical parameters
    private mass: number = 1.0;           // kg
    private gravity: number = 9.81;       // m/s²

    // State vector [x, y, z, vx, vy, vz, rotX, rotY, rotZ, wx, wy, wz]
    private state: Float32Array;

    // Controller gains
    // Velocity to tilt mapping
    private kvp: number = 0.5;            // feedforward: desired velocity to tilt
    private kvd: number = 0.3;            // damping: opposes current velocity

    // Attitude controller (inner loop) - tracks desired roll/pitch/yaw
    private kp_att: number = 10.0;        // attitude P gain
    private kd_att: number = 5.0;         // attitude D gain (critically damped)

    // Altitude controller
    private kp_alt: number = 2.0;         // altitude P gain
    private kd_alt: number = 3.0;         // altitude D gain (overdamped)

    // Yaw controller
    private kp_yaw: number = 1.0;
    private kd_yaw: number = 0.5;

    // Limits
    private maxTilt: number = Math.PI / 6;      // 30 degrees max tilt
    private maxThrust: number = 20.0;           // N
    private minThrust: number = 0.0;            // N
    private maxAngularRate: number = Math.PI;   // rad/s

    // Drag coefficients (simplified air resistance)
    private linearDrag: number = 0.5;
    private angularDrag: number = 0.3;

    constructor() {
        // Initialize state at origin, hovering
        this.state = new Float32Array(12);
        this.reset();
    }

    /**
     * Reset to initial state
     */
    public reset(x: number = 0, y: number = 0, z: number = -3): void {
        this.state.fill(0);
        this.state[0] = x;    // x
        this.state[1] = y;    // y
        this.state[2] = z;    // z
        // All velocities and angles start at 0
    }

    /**
     * Get current state
     */
    public getState(): Float32Array {
        return this.state;
    }

    /**
     * Get position
     */
    public getPosition(): [number, number, number] {
        return [this.state[0], this.state[1], this.state[2]];
    }

    /**
     * Get orientation (Euler angles)
     */
    public getOrientation(): [number, number, number] {
        return [this.state[6], this.state[7], this.state[8]];
    }

    /**
     * Get velocity
     */
    public getVelocity(): [number, number, number] {
        return [this.state[3], this.state[4], this.state[5]];
    }

    /**
     * Update dynamics given desired velocities from IBVS
     *
     * @param desiredVelocity - [vx, vy, vz, wx, wy, wz] in camera frame
     *   vx: right, vy: down (camera), vz: forward, wz: yaw rate
     * @param dt - time step in seconds
     */
    public update(desiredVelocity: Float32Array, dt: number): void {
        // Current state extraction (only extract what we need)
        const vx = this.state[3];
        const vy = this.state[4];
        const vz = this.state[5];
        const rotX = this.state[6];   // rotation.x = forward/back tilt
        const rotZ = this.state[8];   // rotation.z = roll (left/right tilt)
        const wx = this.state[9];     // rotation.x rate
        const wy = this.state[10];    // rotation.y rate
        const wz = this.state[11];    // rotation.z rate

        // Transform desired velocities from camera frame to world frame
        // Camera: X right, Y down, Z forward
        // World: X right, Y up, Z forward
        const vx_des_world = desiredVelocity[0];      // camera X → world X (right)
        const vy_des_world = -desiredVelocity[1];     // camera Y down → world Y up (negate)
        const vz_des_world = desiredVelocity[2];      // camera Z → world Z (forward)
        const wyaw_des = desiredVelocity[5];          // yaw rate

        // ========================================
        // VELOCITY → TILT MAPPING
        // ========================================

        // Feedforward from desired velocity + damping from current velocity
        // tilt_des = kvp * v_des - kvd * v_current
        //
        // - Feedforward (kvp * v_des): tilt to accelerate toward desired velocity
        // - Damping (kvd * v_current): tilt back to slow down when moving fast

        const vy_err = vy_des_world - vy;  // For altitude control

        // Forward tilt: feedforward + damping
        let rotX_des = this.kvp * vz_des_world - this.kvd * vz;

        // Roll: feedforward + damping (signs flipped for roll direction)
        let rotZ_des = -this.kvp * vx_des_world + this.kvd * vx;

        // Clamp desired tilt to max
        rotX_des = Math.max(-this.maxTilt, Math.min(this.maxTilt, rotX_des));
        rotZ_des = Math.max(-this.maxTilt, Math.min(this.maxTilt, rotZ_des));

        // Desired yaw rate from IBVS (direct passthrough with limit)
        let wy_des = Math.max(-this.maxAngularRate, Math.min(this.maxAngularRate, wyaw_des));

        // ========================================
        // INNER LOOP: Attitude Controller
        // ========================================

        // Attitude errors
        const rotX_err = rotX_des - rotX;
        const rotZ_err = rotZ_des - rotZ;

        // Angular accelerations (torques / I)
        const wx_dot_cmd = this.kp_att * rotX_err - this.kd_att * wx;
        const wy_dot_cmd = this.kp_yaw * (wy_des - wy) - this.kd_yaw * wy;
        const wz_dot_cmd = this.kp_att * rotZ_err - this.kd_att * wz;

        // ========================================
        // ALTITUDE CONTROLLER
        // ========================================

        // Thrust needed to hover + control altitude
        const thrust_hover = this.mass * this.gravity;
        const thrust_control = this.mass * (this.kp_alt * vy_err - this.kd_alt * vy);
        let thrust = thrust_hover + thrust_control;

        // Clamp thrust
        thrust = Math.max(this.minThrust, Math.min(this.maxThrust, thrust));

        // ========================================
        // DYNAMICS: Compute accelerations
        // ========================================

        // Small angle approximation for thrust components
        // Thrust is along body +Y, rotated by tilt angles
        // For small angles: sin(θ) ≈ θ, cos(θ) ≈ 1
        //
        // rotX > 0 (tilt forward): thrust gets +Z component
        // rotZ < 0 (tilt right): thrust gets +X component

        const thrust_world_x = -thrust * rotZ;  // Negative rotZ (roll right) → +X thrust
        const thrust_world_y = thrust;          // Main thrust up
        const thrust_world_z = thrust * rotX;   // Positive rotX (tilt forward) → +Z thrust

        // Linear accelerations (world frame)
        const ax = thrust_world_x / this.mass - this.linearDrag * vx;
        const ay = thrust_world_y / this.mass - this.gravity - this.linearDrag * vy;
        const az = thrust_world_z / this.mass - this.linearDrag * vz;

        // Angular accelerations (with drag)
        const wx_dot = wx_dot_cmd - this.angularDrag * wx;
        const wy_dot = wy_dot_cmd - this.angularDrag * wy;
        const wz_dot = wz_dot_cmd - this.angularDrag * wz;

        // ========================================
        // INTEGRATION: Euler method
        // ========================================

        // Update velocities
        this.state[3] += ax * dt;
        this.state[4] += ay * dt;
        this.state[5] += az * dt;

        // Update positions
        this.state[0] += this.state[3] * dt;
        this.state[1] += this.state[4] * dt;
        this.state[2] += this.state[5] * dt;

        // Update angular rates
        this.state[9] += wx_dot * dt;
        this.state[10] += wy_dot * dt;
        this.state[11] += wz_dot * dt;

        // Update angles
        this.state[6] += this.state[9] * dt;
        this.state[7] += this.state[10] * dt;
        this.state[8] += this.state[11] * dt;

        // Normalize yaw to [-π, π]
        while (this.state[7] > Math.PI) this.state[7] -= 2 * Math.PI;
        while (this.state[7] < -Math.PI) this.state[7] += 2 * Math.PI;
    }

    /**
     * Set controller gains
     */
    public setGains(params: {
        kvp?: number;
        kvd?: number;
        kp_att?: number;
        kd_att?: number;
        kp_alt?: number;
        kd_alt?: number;
    }): void {
        if (params.kvp !== undefined) this.kvp = params.kvp;
        if (params.kvd !== undefined) this.kvd = params.kvd;
        if (params.kp_att !== undefined) this.kp_att = params.kp_att;
        if (params.kd_att !== undefined) this.kd_att = params.kd_att;
        if (params.kp_alt !== undefined) this.kp_alt = params.kp_alt;
        if (params.kd_alt !== undefined) this.kd_alt = params.kd_alt;
    }

    /**
     * Get current desired attitude for visualization
     */
    public getDesiredAttitude(): [number, number, number] {
        // This would need to be stored during update for visualization
        // For now, return current attitude
        return [this.state[6], this.state[7], this.state[8]];
    }
}
