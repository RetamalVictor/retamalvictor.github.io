import * as THREE from 'three';
import { DroneState, ControlCommand, Waypoint, Quaternion, RacingConfig, DEFAULT_CONFIG } from '../types';

/**
 * Simplified Model Predictive Controller
 *
 * Based on the MPC from drone-racing-control:
 * - 10-state vector: [px, py, pz, qw, qx, qy, qz, vx, vy, vz]
 * - 4-input vector: [thrust, roll_rate, pitch_rate, yaw_rate]
 * - State prediction for command delay compensation
 * - Feedforward from trajectory + feedback correction
 *
 * Simplifications from real system:
 * - Explicit control law instead of QP solver
 * - 10-node horizon instead of 30
 * - First-order dynamics approximation
 */
export class SimplifiedMPC {
    private config: RacingConfig;

    // Controller gains (conservative for stability after trajectory fixes)
    private readonly kp_pos = 1.0;      // Position P gain (reduced from 1.5)
    private readonly kd_pos = 1.5;      // Position D gain - more damping (was 1.0)
    private readonly kp_att = 3.0;      // Attitude P gain (reduced from 5.0)

    // Physical parameters
    private readonly gravity = 9.81;    // m/s²

    // Input limits - CONSERVATIVE for stability
    private readonly minThrust = 5.0;   // m/s² - don't go too low
    private readonly maxThrust = 15.0;  // m/s² - don't go too high
    private readonly maxRate = 1.5;     // rad/s for roll/pitch - REDUCED for stability
    private readonly maxYawRate = 1.0;  // rad/s
    private readonly maxTiltAngle = 0.4; // rad (~23 degrees) - conservative

    // State prediction
    private readonly tauThrust = 0.04;  // Thrust time constant (40ms)

    // Command queue for state prediction
    private commandQueue: ControlCommand[] = [];
    private currentThrust = 9.81;  // Start at hover thrust

    // Debug logging
    private debugFrameCount = 0;


    // Prediction horizon
    private readonly horizonNodes = 10;
    private readonly horizonTime: number;

    // Last computed values for visualization
    private lastPredictedStates: DroneState[] = [];
    private lastReferenceWaypoints: Waypoint[] = [];

    constructor(config: Partial<RacingConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.horizonTime = this.config.predictionHorizon;
    }

    /**
     * Compute control command given current state and reference trajectory
     *
     * @param currentState - Current drone state estimate
     * @param getReference - Function to get reference waypoint at time t
     * @param currentTime - Current time in trajectory
     * @returns Control command (thrust + body rates)
     */
    public computeControl(
        currentState: DroneState,
        getReference: (t: number) => Waypoint,
        currentTime: number
    ): ControlCommand {
        // Step 1: Predict state after command delay
        const predictedState = this.predictState(currentState, this.config.commandDelay);

        // Step 2: Sample reference trajectory at prediction times
        const references: Waypoint[] = [];
        const dt = this.horizonTime / (this.horizonNodes - 1);
        for (let i = 0; i < this.horizonNodes; i++) {
            references.push(getReference(currentTime + this.config.commandDelay + i * dt));
        }

        // Step 3: Compute feedforward from reference trajectory
        const feedforward = this.computeFeedforward(references[0]);

        // Step 4: Compute feedback correction
        const feedback = this.computeFeedback(predictedState, references[0]);

        // Step 5: Combine feedforward and feedback
        const command = this.combineCommands(feedforward, feedback, currentState);

        // Debug logging for first 60 frames
        this.debugFrameCount++;
        if (this.debugFrameCount <= 60 && this.debugFrameCount % 10 === 0) {
            const ref = references[0];
            console.log(`=== MPC FRAME ${this.debugFrameCount} ===`);
            console.log(`Ref accel: (${ref.acceleration.x.toFixed(3)}, ${ref.acceleration.y.toFixed(3)}, ${ref.acceleration.z.toFixed(3)})`);
            console.log(`FF: thrust=${feedforward.thrust.toFixed(3)}, rollDes=${(feedforward.rollDes*180/Math.PI).toFixed(2)}°, pitchDes=${(feedforward.pitchDes*180/Math.PI).toFixed(2)}°`);
            console.log(`FB: thrust=${feedback.thrust.toFixed(3)}, rollDes=${(feedback.rollDes*180/Math.PI).toFixed(2)}°, pitchDes=${(feedback.pitchDes*180/Math.PI).toFixed(2)}°`);
            console.log(`Combined: thrust=${command.thrust.toFixed(3)}, rollRate=${command.rollRate.toFixed(3)}, pitchRate=${command.pitchRate.toFixed(3)}`);
        }

        // Store for visualization
        this.lastPredictedStates = this.predictHorizon(currentState, command);
        this.lastReferenceWaypoints = references;

        // Step 6: Queue command for future state prediction
        this.pushCommand(command);

        return command;
    }

    /**
     * Predict state after given time using command queue
     */
    private predictState(state: DroneState, dt: number): DroneState {
        // Simple integration with current velocity
        const predicted: DroneState = {
            position: {
                x: state.position.x + state.velocity.x * dt,
                y: state.position.y + state.velocity.y * dt,
                z: state.position.z + state.velocity.z * dt,
            },
            orientation: { ...state.orientation },
            velocity: { ...state.velocity },
            timestamp: state.timestamp + dt * 1e6,
        };

        return predicted;
    }

    /**
     * Compute feedforward from reference trajectory
     * Uses reference acceleration to compute required thrust and desired attitude
     *
     * Returns desired attitude (not rates) - rates are computed in combineCommands
     * where we have access to current state
     *
     * CRITICAL: The acceleration is in world frame, but roll/pitch are body-frame
     * angles. We must rotate the acceleration into the body frame (accounting for
     * heading/yaw) before computing roll and pitch.
     */
    private computeFeedforward(reference: Waypoint): {
        thrust: number;
        rollDes: number;
        pitchDes: number;
        yawRate: number;
    } {
        // Desired acceleration in world frame (from trajectory + gravity compensation)
        const ax_world = reference.acceleration.x;
        const ay_world = reference.acceleration.y + this.gravity;  // Compensate gravity
        const az_world = reference.acceleration.z;

        // Total thrust magnitude (invariant under rotation)
        const thrust = Math.sqrt(ax_world * ax_world + ay_world * ay_world + az_world * az_world);

        // Rotate world-frame acceleration into body frame using heading
        // The heading is yaw rotation around Y-axis
        // Body frame: after yaw rotation, body +Z aligns with velocity direction
        //
        // Rotation matrix for yaw around Y:
        //   R_y(ψ) = | cos(ψ)   0   sin(ψ) |
        //            |   0      1     0    |
        //            |-sin(ψ)   0   cos(ψ) |
        //
        // To go from world to body: multiply by R_y(-ψ) = R_y(ψ)^T
        //   ax_body = ax_world * cos(ψ) - az_world * sin(ψ)
        //   ay_body = ay_world (Y is unchanged)
        //   az_body = ax_world * sin(ψ) + az_world * cos(ψ)
        const yaw = reference.heading;
        const cosYaw = Math.cos(yaw);
        const sinYaw = Math.sin(yaw);

        const ax_body = ax_world * cosYaw - az_world * sinYaw;
        const ay_body = ay_world;
        const az_body = ax_world * sinYaw + az_world * cosYaw;

        // Now compute roll and pitch from body-frame acceleration
        // These formulas are correct because we're now in body frame where yaw = 0
        //
        // Roll(φ) around body Z: [0,T,0] → [-Tsinφ, Tcosφ, 0] → positive roll gives -X thrust
        // Pitch(θ) around body X: [0,T,0] → [0, Tcosθ, Tsinθ] → positive pitch gives +Z thrust
        //
        // To achieve thrust components [ax_body, ay_body, az_body]:
        //   roll = -atan2(ax_body, ay_body)
        //   pitch = atan2(az_body, ay_body)
        const rollDes = -Math.atan2(ax_body, ay_body);
        const pitchDes = Math.atan2(az_body, ay_body);

        return { thrust, rollDes, pitchDes, yawRate: reference.headingRate };
    }

    /**
     * Compute feedback correction based on tracking error
     * Returns desired attitude corrections (not rates)
     *
     * CRITICAL: Like feedforward, the position/velocity errors are in world frame,
     * but the resulting roll/pitch corrections must be in body frame.
     */
    private computeFeedback(state: DroneState, reference: Waypoint): {
        thrust: number;
        rollDes: number;
        pitchDes: number;
        yawRate: number;
    } {
        // Position error in world frame - CLAMP to prevent aggressive corrections
        const maxPosError = 2.0;  // meters
        let ex_world = reference.position.x - state.position.x;
        let ey_world = reference.position.y - state.position.y;
        let ez_world = reference.position.z - state.position.z;

        // Clamp position errors
        ex_world = Math.max(-maxPosError, Math.min(maxPosError, ex_world));
        ey_world = Math.max(-maxPosError, Math.min(maxPosError, ey_world));
        ez_world = Math.max(-maxPosError, Math.min(maxPosError, ez_world));

        // Velocity error in world frame - also clamp
        const maxVelError = 3.0;  // m/s
        let evx_world = reference.velocity.x - state.velocity.x;
        let evy_world = reference.velocity.y - state.velocity.y;
        let evz_world = reference.velocity.z - state.velocity.z;

        evx_world = Math.max(-maxVelError, Math.min(maxVelError, evx_world));
        evy_world = Math.max(-maxVelError, Math.min(maxVelError, evy_world));
        evz_world = Math.max(-maxVelError, Math.min(maxVelError, evz_world));

        // PD control for position → desired acceleration (world frame)
        const ax_world = this.kp_pos * ex_world + this.kd_pos * evx_world;
        const ay_world = this.kp_pos * ey_world + this.kd_pos * evy_world;
        const az_world = this.kp_pos * ez_world + this.kd_pos * evz_world;

        // Convert to thrust correction (Y is up, independent of heading)
        const thrust_fb = ay_world;

        // Rotate world-frame acceleration correction into body frame using current heading
        // This ensures roll/pitch corrections are computed in body coordinates
        const currentHeading = this.toEulerYXZ(state.orientation).yaw;
        const cosYaw = Math.cos(currentHeading);
        const sinYaw = Math.sin(currentHeading);

        const ax_body = ax_world * cosYaw - az_world * sinYaw;
        const az_body = ax_world * sinYaw + az_world * cosYaw;

        // Attitude corrections (small angle approximation) now in body frame
        //   +roll → -X_body acceleration, so for +X_body accel need -roll
        //   +pitch → +Z_body acceleration, so for +Z_body accel need +pitch
        const rollDes = -ax_body / this.gravity;
        const pitchDes = az_body / this.gravity;

        // Heading error for yaw rate
        let headingError = reference.heading - currentHeading;
        while (headingError > Math.PI) headingError -= 2 * Math.PI;
        while (headingError < -Math.PI) headingError += 2 * Math.PI;

        const yawRate = this.kp_att * headingError;

        return { thrust: thrust_fb, rollDes, pitchDes, yawRate };
    }

    /**
     * Combine feedforward and feedback commands with limiting
     *
     * Key insight: ff and fb give DESIRED attitudes, not rates.
     * We compute: rate = kp_att * (desired_attitude - current_attitude)
     * This is proper PD control on attitude.
     */
    private combineCommands(
        ff: { thrust: number; rollDes: number; pitchDes: number; yawRate: number },
        fb: { thrust: number; rollDes: number; pitchDes: number; yawRate: number },
        state: DroneState
    ): ControlCommand {
        // Get current roll/pitch from quaternion using YXZ Euler order
        const euler = this.toEulerYXZ(state.orientation);
        const currentRoll = euler.roll;
        const currentPitch = euler.pitch;

        // Combine feedforward and feedback to get total desired attitude
        const desiredRoll = ff.rollDes + fb.rollDes;
        const desiredPitch = ff.pitchDes + fb.pitchDes;

        // Clamp desired attitude to max tilt
        const clampedRoll = Math.max(-this.maxTiltAngle, Math.min(this.maxTiltAngle, desiredRoll));
        const clampedPitch = Math.max(-this.maxTiltAngle, Math.min(this.maxTiltAngle, desiredPitch));

        // Compute attitude ERROR, then apply gain to get rate
        const rollError = clampedRoll - currentRoll;
        const pitchError = clampedPitch - currentPitch;

        let rollRate = this.kp_att * rollError;
        let pitchRate = this.kp_att * pitchError;
        let yawRate = ff.yawRate + fb.yawRate;

        // Combine thrust
        let thrust = ff.thrust + fb.thrust;

        // Apply first-order dynamics (smooth transitions)
        const alpha_thrust = 1 - Math.exp(-1 / 60 / this.tauThrust);  // At 60Hz
        thrust = (1 - alpha_thrust) * this.currentThrust + alpha_thrust * thrust;
        this.currentThrust = thrust;

        // Apply standard limits
        thrust = Math.max(this.minThrust, Math.min(this.maxThrust, thrust));
        rollRate = Math.max(-this.maxRate, Math.min(this.maxRate, rollRate));
        pitchRate = Math.max(-this.maxRate, Math.min(this.maxRate, pitchRate));
        yawRate = Math.max(-this.maxYawRate, Math.min(this.maxYawRate, yawRate));

        return {
            thrust,
            rollRate,
            pitchRate,
            yawRate,
            timestamp: state.timestamp,
        };
    }

    /**
     * Convert quaternion to Euler angles using YXZ order (Three.js Y-up convention)
     *
     * CRITICAL: Must use 'YXZ' order everywhere in the codebase!
     * This gives aerospace-style decomposition:
     *   - Yaw first (Y), then Pitch (X), then Roll (Z)
     *
     * @param q - Quaternion in our { w, x, y, z } format
     * @returns roll (Z), pitch (X), yaw (Y) angles in radians
     */
    private toEulerYXZ(q: Quaternion): { roll: number; pitch: number; yaw: number } {
        // Three.js Quaternion constructor order: (x, y, z, w)
        const tq = new THREE.Quaternion(q.x, q.y, q.z, q.w);
        const euler = new THREE.Euler().setFromQuaternion(tq, 'YXZ');
        return {
            roll: euler.z,   // Roll around Z-axis
            pitch: euler.x,  // Pitch around X-axis
            yaw: euler.y,    // Yaw around Y-axis
        };
    }

    /**
     * Predict state over horizon for visualization
     */
    private predictHorizon(state: DroneState, command: ControlCommand): DroneState[] {
        const states: DroneState[] = [{ ...state }];
        const dt = this.horizonTime / (this.horizonNodes - 1);

        let currentState = { ...state };

        for (let i = 1; i < this.horizonNodes; i++) {
            // Simple kinematic prediction
            currentState = {
                position: {
                    x: currentState.position.x + currentState.velocity.x * dt,
                    y: currentState.position.y + currentState.velocity.y * dt,
                    z: currentState.position.z + currentState.velocity.z * dt,
                },
                orientation: currentState.orientation,
                velocity: {
                    x: currentState.velocity.x,
                    y: currentState.velocity.y + (command.thrust - this.gravity) * dt,
                    z: currentState.velocity.z,
                },
                timestamp: currentState.timestamp + dt * 1e6,
            };
            states.push({ ...currentState });
        }

        return states;
    }

    /**
     * Push command to queue for state prediction
     */
    private pushCommand(command: ControlCommand): void {
        this.commandQueue.push(command);

        // Keep only recent commands
        const maxQueueSize = 20;
        if (this.commandQueue.length > maxQueueSize) {
            this.commandQueue.shift();
        }
    }

    /**
     * Get last predicted states for visualization
     */
    public getPredictedStates(): DroneState[] {
        return this.lastPredictedStates;
    }

    /**
     * Get last reference waypoints for visualization
     */
    public getReferenceWaypoints(): Waypoint[] {
        return this.lastReferenceWaypoints;
    }

    /**
     * Compute tracking error (position magnitude)
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
        this.commandQueue = [];
        this.currentThrust = this.gravity;
        this.lastPredictedStates = [];
        this.lastReferenceWaypoints = [];
        this.debugFrameCount = 0;
    }
}
