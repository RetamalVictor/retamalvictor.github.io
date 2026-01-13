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

    // Controller gains - Differential Flatness approach
    // Velocity tracking gain (desired velocity → desired acceleration)
    private kv: number = 2.0;             // velocity tracking gain

    // Attitude controller (inner loop) - tracks flatness-derived attitude
    private kp_att: number = 25.0;        // attitude P gain (aggressive)
    private kd_att: number = 8.0;         // attitude D gain

    // Yaw controller
    private kp_yaw: number = 3.0;
    private kd_yaw: number = 1.5;

    // Limits
    private maxTilt: number = Math.PI / 3;      // 60 degrees max tilt (aggressive)
    private maxThrust: number = 30.0;           // N (3g capability)
    private minThrust: number = 1.0;            // N (never zero thrust)
    private maxAngularRate: number = Math.PI * 2;  // rad/s (fast rotations)

    // Drag coefficients (simplified air resistance)
    private linearDrag: number = 0.2;           // low drag for responsive motion
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
     * Update dynamics using Differential Flatness
     *
     * Instead of mapping velocity → tilt heuristically, we:
     * 1. Compute desired acceleration from velocity error
     * 2. Use flatness: acceleration → thrust vector → attitude
     * 3. Track the flatness-derived attitude with inner loop
     *
     * @param desiredVelocity - [vx, vy, vz, wx, wy, wz] in camera frame
     * @param dt - time step in seconds
     */
    public update(desiredVelocity: Float32Array, dt: number): void {
        // Current state
        const vx = this.state[3];
        const vy = this.state[4];
        const vz = this.state[5];
        const rotX = this.state[6];   // pitch (forward/back tilt)
        const rotZ = this.state[8];   // roll (left/right tilt)
        const wx = this.state[9];
        const wy = this.state[10];
        const wz = this.state[11];

        // Transform desired velocities: Camera (X right, Y down, Z forward) → World (X right, Y up, Z forward)
        const vx_des = desiredVelocity[0];
        const vy_des = -desiredVelocity[1];  // flip Y
        const vz_des = desiredVelocity[2];
        const wyaw_des = desiredVelocity[5];

        // ========================================
        // DIFFERENTIAL FLATNESS: Velocity → Acceleration → Thrust & Attitude
        // ========================================

        // Step 1: Compute desired acceleration (velocity tracking)
        const ax_des = this.kv * (vx_des - vx);
        const ay_des = this.kv * (vy_des - vy);
        const az_des = this.kv * (vz_des - vz);

        // Step 2: Total thrust vector must provide desired accel + counteract gravity
        // F/m = a_des + g  →  F = m * (a_des + [0, g, 0])
        const Fx = this.mass * ax_des;
        const Fy = this.mass * (ay_des + this.gravity);
        const Fz = this.mass * az_des;

        // Step 3: Thrust magnitude
        let thrust = Math.sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
        thrust = Math.max(this.minThrust, Math.min(this.maxThrust, thrust));

        // Step 4: Desired attitude from thrust direction
        // Thrust vector defines body Y-axis direction (for Y-up convention)
        // Normalize thrust direction
        const F_norm = Math.sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
        const zB_x = F_norm > 0.01 ? Fx / F_norm : 0;
        const zB_y = F_norm > 0.01 ? Fy / F_norm : 1;  // default to pointing up
        const zB_z = F_norm > 0.01 ? Fz / F_norm : 0;

        // Extract desired roll (rotZ) and pitch (rotX) from thrust direction
        // For Y-up: pitch = atan2(Fz, Fy), roll = atan2(-Fx, Fy)
        let rotX_des = Math.atan2(zB_z, zB_y);   // pitch: forward component
        let rotZ_des = Math.atan2(-zB_x, zB_y);  // roll: lateral component

        // Clamp to max tilt
        rotX_des = Math.max(-this.maxTilt, Math.min(this.maxTilt, rotX_des));
        rotZ_des = Math.max(-this.maxTilt, Math.min(this.maxTilt, rotZ_des));

        // Desired yaw rate from IBVS
        const wy_des = Math.max(-this.maxAngularRate, Math.min(this.maxAngularRate, wyaw_des));

        // ========================================
        // ATTITUDE CONTROLLER (Inner Loop)
        // ========================================

        // Attitude errors
        const rotX_err = rotX_des - rotX;
        const rotZ_err = rotZ_des - rotZ;

        // Angular accelerations (PD control)
        const wx_dot_cmd = this.kp_att * rotX_err - this.kd_att * wx;
        const wy_dot_cmd = this.kp_yaw * (wy_des - wy) - this.kd_yaw * wy;
        const wz_dot_cmd = this.kp_att * rotZ_err - this.kd_att * wz;

        // ========================================
        // PHYSICS: Compute actual accelerations
        // ========================================

        // Thrust components in world frame (exact, not small-angle approx)
        const cosX = Math.cos(rotX);
        const sinX = Math.sin(rotX);
        const cosZ = Math.cos(rotZ);
        const sinZ = Math.sin(rotZ);

        // Rotation of thrust from body to world (simplified for roll-pitch only)
        // Body Y-axis rotated by rotX (pitch) then rotZ (roll)
        const thrust_world_x = -thrust * sinZ * cosX;
        const thrust_world_y = thrust * cosZ * cosX;
        const thrust_world_z = thrust * sinX;

        // Linear accelerations
        const ax = thrust_world_x / this.mass - this.linearDrag * vx;
        const ay = thrust_world_y / this.mass - this.gravity - this.linearDrag * vy;
        const az = thrust_world_z / this.mass - this.linearDrag * vz;

        // Angular accelerations
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
        kv?: number;
        kp_att?: number;
        kd_att?: number;
        kp_yaw?: number;
        kd_yaw?: number;
    }): void {
        if (params.kv !== undefined) this.kv = params.kv;
        if (params.kp_att !== undefined) this.kp_att = params.kp_att;
        if (params.kd_att !== undefined) this.kd_att = params.kd_att;
        if (params.kp_yaw !== undefined) this.kp_yaw = params.kp_yaw;
        if (params.kd_yaw !== undefined) this.kd_yaw = params.kd_yaw;
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
