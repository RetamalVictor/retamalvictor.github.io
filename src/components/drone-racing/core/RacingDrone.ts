import * as THREE from 'three';
import { DroneState, ControlCommand, Vector3 } from '../types';
import { DroneDynamics, DynamicsParams, DynamicsState } from './DroneDynamics';
import { DroneVisualization, DroneVisualizationConfig } from '../visualization/DroneVisualization';

/**
 * Racing Drone - Composition of Dynamics + Visualization
 *
 * This class composes:
 * - DroneDynamics: Pure physics simulation
 * - DroneVisualization: Pure 3D rendering
 *
 * This allows the dynamics to be used independently by MPC
 * while keeping a clean interface for the simulation demo.
 */
export interface RacingDroneConfig {
    dynamics?: Partial<DynamicsParams>;
    visualization?: Partial<DroneVisualizationConfig>;
}

export class RacingDrone {
    // Components
    private dynamics: DroneDynamics;
    private visualization: DroneVisualization;

    // Gravity constant for thrust normalization
    private readonly gravity: number;

    constructor(config: RacingDroneConfig = {}) {
        this.dynamics = new DroneDynamics(config.dynamics);
        this.visualization = new DroneVisualization(config.visualization);
        this.gravity = this.dynamics.getParams().gravity;

        // Initial sync
        this.syncVisualization();
    }

    /**
     * Get the Three.js mesh for scene addition
     */
    public get mesh(): THREE.Group {
        return this.visualization.mesh;
    }

    /**
     * Update dynamics and visualization with control command
     */
    public update(command: ControlCommand, dt: number): void {
        // Step dynamics
        this.dynamics.step(command, dt);

        // Update visualization
        this.syncVisualization(command.thrust);
    }

    /**
     * Sync visualization with current dynamics state
     */
    private syncVisualization(thrust?: number): void {
        const state = this.dynamics.getState();
        const thrustNormalized = (thrust ?? this.gravity) / this.gravity;
        this.visualization.update(state, thrustNormalized);
    }

    /**
     * Get current state
     */
    public getState(): DroneState {
        return this.dynamics.getState();
    }

    /**
     * Get full internal dynamics state (for MPC)
     */
    public getFullState(): DynamicsState {
        return this.dynamics.getFullState();
    }

    /**
     * Get camera pose (world frame)
     */
    public getCameraPose(): { position: THREE.Vector3; rotation: THREE.Euler } {
        return this.visualization.getCameraPose(this.dynamics.getState());
    }

    /**
     * Get position as THREE.Vector3
     */
    public getPosition(): THREE.Vector3 {
        const state = this.dynamics.getState();
        return new THREE.Vector3(state.position.x, state.position.y, state.position.z);
    }

    /**
     * Get velocity magnitude
     */
    public getSpeed(): number {
        return this.dynamics.getSpeed();
    }

    /**
     * Reset to initial state
     */
    public reset(position?: Vector3): void {
        this.dynamics.reset(position);
        this.syncVisualization();
    }

    /**
     * Set position directly
     */
    public setPosition(x: number, y: number, z: number): void {
        this.dynamics.setPosition(x, y, z);
        this.syncVisualization();
    }

    /**
     * Set heading (yaw) directly
     */
    public setHeading(yaw: number): void {
        this.dynamics.setHeading(yaw);
        this.syncVisualization();
    }

    /**
     * Set velocity directly
     */
    public setVelocity(vx: number, vy: number, vz: number): void {
        this.dynamics.setVelocity(vx, vy, vz);
    }

    /**
     * Show/hide velocity arrow
     */
    public showVelocityArrow(show: boolean): void {
        this.visualization.showVelocityArrow(show);
    }

    /**
     * Get dynamics instance (for MPC to access parameters)
     */
    public getDynamics(): DroneDynamics {
        return this.dynamics;
    }

    /**
     * Get dynamics parameters (for MPC to match)
     */
    public getDynamicsParams(): DynamicsParams {
        return this.dynamics.getParams();
    }

    /**
     * Dispose of resources
     */
    public dispose(): void {
        this.visualization.dispose();
    }
}
