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

    // Controller gains
    private readonly kp_pos = 2.0;      // Position P gain
    private readonly kd_pos = 1.5;      // Position D gain (velocity feedback)
    private readonly kp_att = 8.0;      // Attitude P gain

    // Physical parameters
    private readonly gravity = 9.81;    // m/s²

    // Input limits (from real system)
    private readonly minThrust = 2.0;   // m/s² (normalized)
    private readonly maxThrust = 20.0;  // m/s²
    private readonly maxRate = 3.0;     // rad/s for roll/pitch
    private readonly maxYawRate = 1.0;  // rad/s

    // State prediction
    private readonly tauThrust = 0.04;  // Thrust time constant (40ms)

    // Command queue for state prediction
    private commandQueue: ControlCommand[] = [];
    private currentThrust = 9.81;  // Start at hover thrust

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
     * Uses reference acceleration to compute required thrust and attitude
     */
    private computeFeedforward(reference: Waypoint): {
        thrust: number;
        rollRate: number;
        pitchRate: number;
        yawRate: number;
    } {
        // Desired acceleration (from trajectory + gravity compensation)
        const ax_des = reference.acceleration.x;
        const ay_des = reference.acceleration.y + this.gravity;  // Compensate gravity
        const az_des = reference.acceleration.z;

        // Total thrust magnitude
        const thrust = Math.sqrt(ax_des * ax_des + ay_des * ay_des + az_des * az_des);

        // Desired attitude from thrust direction
        // thrust_world = R * [0, thrust, 0] (Y is up)
        // So: roll = -atan2(ax, ay), pitch = atan2(az, ay)
        const roll_des = -Math.atan2(ax_des, ay_des);
        const pitch_des = Math.atan2(az_des, ay_des);

        // Convert desired attitude to rate commands (assume we want to reach it quickly)
        const rollRate = roll_des * this.kp_att;
        const pitchRate = pitch_des * this.kp_att;
        const yawRate = reference.headingRate;

        return { thrust, rollRate, pitchRate, yawRate };
    }

    /**
     * Compute feedback correction based on tracking error
     */
    private computeFeedback(state: DroneState, reference: Waypoint): {
        thrust: number;
        rollRate: number;
        pitchRate: number;
        yawRate: number;
    } {
        // Position error
        const ex = reference.position.x - state.position.x;
        const ey = reference.position.y - state.position.y;
        const ez = reference.position.z - state.position.z;

        // Velocity error
        const evx = reference.velocity.x - state.velocity.x;
        const evy = reference.velocity.y - state.velocity.y;
        const evz = reference.velocity.z - state.velocity.z;

        // PD control for position
        const ax_fb = this.kp_pos * ex + this.kd_pos * evx;
        const ay_fb = this.kp_pos * ey + this.kd_pos * evy;
        const az_fb = this.kp_pos * ez + this.kd_pos * evz;

        // Convert to thrust and attitude corrections
        const thrust_fb = ay_fb;  // Y is up

        // Attitude corrections (small angle approximation)
        const roll_fb = -ax_fb / this.gravity;   // Roll to move in X
        const pitch_fb = az_fb / this.gravity;   // Pitch to move in Z

        // Heading error
        const currentHeading = this.quaternionToYaw(state.orientation);
        let headingError = reference.heading - currentHeading;
        while (headingError > Math.PI) headingError -= 2 * Math.PI;
        while (headingError < -Math.PI) headingError += 2 * Math.PI;

        const yaw_fb = this.kp_att * headingError;

        return {
            thrust: thrust_fb,
            rollRate: roll_fb * this.kp_att,
            pitchRate: pitch_fb * this.kp_att,
            yawRate: yaw_fb,
        };
    }

    /**
     * Combine feedforward and feedback commands with limiting
     */
    private combineCommands(
        ff: { thrust: number; rollRate: number; pitchRate: number; yawRate: number },
        fb: { thrust: number; rollRate: number; pitchRate: number; yawRate: number },
        state: DroneState
    ): ControlCommand {
        // Combine
        let thrust = ff.thrust + fb.thrust;
        let rollRate = ff.rollRate + fb.rollRate;
        let pitchRate = ff.pitchRate + fb.pitchRate;
        let yawRate = ff.yawRate + fb.yawRate;

        // Apply first-order dynamics (smooth transitions)
        const alpha_thrust = 1 - Math.exp(-1 / 60 / this.tauThrust);  // At 60Hz

        thrust = (1 - alpha_thrust) * this.currentThrust + alpha_thrust * thrust;
        this.currentThrust = thrust;

        // Limit commands
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
     * Convert quaternion to yaw angle
     */
    private quaternionToYaw(q: Quaternion): number {
        // Yaw (rotation around Y axis)
        const siny_cosp = 2 * (q.w * q.y + q.z * q.x);
        const cosy_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
        return Math.atan2(siny_cosp, cosy_cosp);
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
    }
}
