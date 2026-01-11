/**
 * Trajectory Generator for Drone Racing Demo
 *
 * Generates various racing trajectories:
 * - Circle: Simple circular path
 * - Figure-8 (Lemniscate): Infinity-shaped curve
 * - Hairpin: Racing-style tight turns
 * - Snake: Serpentine weaving pattern
 * - Race Track: Multi-gate race course
 */

import { Waypoint } from '../types';

export type TrajectoryType = 'circle' | 'figure8' | 'hairpin' | 'snake' | 'racetrack';

export interface TrajectoryConfig {
    type: TrajectoryType;
    speed: number;          // Base speed in m/s
    scale: number;          // Scale factor for trajectory size
    height: number;         // Flight altitude
}

export const DEFAULT_TRAJECTORY_CONFIG: TrajectoryConfig = {
    type: 'circle',
    speed: 15.0,
    scale: 1.0,
    height: 4.0,
};

/**
 * Trajectory Generator
 *
 * All trajectories are parameterized by time and return smooth
 * position, velocity, acceleration, and heading information.
 */
export class TrajectoryGenerator {
    private config: TrajectoryConfig;

    constructor(config: Partial<TrajectoryConfig> = {}) {
        this.config = { ...DEFAULT_TRAJECTORY_CONFIG, ...config };
    }

    /**
     * Get trajectory configuration
     */
    public getConfig(): TrajectoryConfig {
        return { ...this.config };
    }

    /**
     * Update trajectory configuration
     */
    public setConfig(config: Partial<TrajectoryConfig>): void {
        this.config = { ...this.config, ...config };
    }

    /**
     * Get the period (lap time) for the current trajectory
     */
    public getPeriod(): number {
        switch (this.config.type) {
            case 'circle':
                return this.getCirclePeriod();
            case 'figure8':
                return this.getFigure8Period();
            case 'hairpin':
                return this.getHairpinPeriod();
            case 'snake':
                return this.getSnakePeriod();
            case 'racetrack':
                return this.getRacetrackPeriod();
            default:
                return this.getCirclePeriod();
        }
    }

    /**
     * Get waypoint at time t for current trajectory
     */
    public getWaypoint(t: number): Waypoint {
        switch (this.config.type) {
            case 'circle':
                return this.getCircleWaypoint(t);
            case 'figure8':
                return this.getFigure8Waypoint(t);
            case 'hairpin':
                return this.getHairpinWaypoint(t);
            case 'snake':
                return this.getSnakeWaypoint(t);
            case 'racetrack':
                return this.getRacetrackWaypoint(t);
            default:
                return this.getCircleWaypoint(t);
        }
    }

    /**
     * Get trajectory points for visualization
     */
    public getTrajectoryPoints(numPoints: number = 200): { x: number; y: number; z: number }[] {
        const period = this.getPeriod();
        const points: { x: number; y: number; z: number }[] = [];

        for (let i = 0; i <= numPoints; i++) {
            const t = (i / numPoints) * period;
            const wp = this.getWaypoint(t);
            points.push({
                x: wp.position.x,
                y: wp.position.y,
                z: wp.position.z,
            });
        }

        return points;
    }

    // =========================================
    // Circle Trajectory
    // =========================================

    private getCircleRadius(): number {
        return 20.0 * this.config.scale;
    }

    private getCirclePeriod(): number {
        const radius = this.getCircleRadius();
        return (2 * Math.PI * radius) / this.config.speed;
    }

    private getCircleWaypoint(t: number): Waypoint {
        const radius = this.getCircleRadius();
        const speed = this.config.speed;
        const height = this.config.height;
        const omega = speed / radius;
        const angle = omega * t;

        const x = radius * Math.cos(angle);
        const z = radius * Math.sin(angle);
        const vx = -speed * Math.sin(angle);
        const vz = speed * Math.cos(angle);
        const centripetalAccel = speed * speed / radius;
        const ax = -centripetalAccel * Math.cos(angle);
        const az = -centripetalAccel * Math.sin(angle);
        const heading = Math.atan2(vx, vz);

        return {
            position: { x, y: height, z },
            velocity: { x: vx, y: 0, z: vz },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate: -omega,
            time: t,
        };
    }

    // =========================================
    // Figure-8 (Lemniscate) Trajectory
    // =========================================

    private getFigure8Scale(): number {
        return 15.0 * this.config.scale;
    }

    private getFigure8Period(): number {
        // Approximate arc length of lemniscate: ~5.24 * a
        const a = this.getFigure8Scale();
        const arcLength = 5.24 * a;
        return arcLength / this.config.speed;
    }

    private getFigure8Waypoint(t: number): Waypoint {
        const a = this.getFigure8Scale();
        const speed = this.config.speed;
        const height = this.config.height;
        const period = this.getFigure8Period();

        // Normalize t to [0, 2π]
        const phase = (t / period) * 2 * Math.PI;

        // Lemniscate of Bernoulli: r² = a² * cos(2θ)
        // Parametric form: x = a*cos(θ)/(1+sin²(θ)), z = a*sin(θ)*cos(θ)/(1+sin²(θ))
        // Simplified form for smooth figure-8:
        const x = a * Math.sin(phase);
        const z = a * Math.sin(phase) * Math.cos(phase);

        // Velocity (derivative of position w.r.t. phase, scaled by phase rate)
        const phaseRate = (2 * Math.PI) / period;
        const vx = a * Math.cos(phase) * phaseRate;
        const vz = a * (Math.cos(phase) * Math.cos(phase) - Math.sin(phase) * Math.sin(phase)) * phaseRate;

        // Normalize velocity to desired speed
        const vMag = Math.sqrt(vx * vx + vz * vz);
        const vxNorm = (vx / vMag) * speed;
        const vzNorm = (vz / vMag) * speed;

        // Acceleration (centripetal + tangential)
        // For smooth visualization, compute from curvature
        const ax = -a * Math.sin(phase) * phaseRate * phaseRate;
        const az = -2 * a * Math.sin(phase) * Math.cos(phase) * phaseRate * phaseRate;

        // Heading from velocity direction
        const heading = Math.atan2(vxNorm, vzNorm);

        // Heading rate (approximate)
        const dt = 0.001;
        const phase2 = ((t + dt) / period) * 2 * Math.PI;
        const vx2 = a * Math.cos(phase2) * phaseRate;
        const vz2 = a * (Math.cos(phase2) * Math.cos(phase2) - Math.sin(phase2) * Math.sin(phase2)) * phaseRate;
        const heading2 = Math.atan2(vx2, vz2);
        let headingRate = (heading2 - heading) / dt;
        // Wrap heading rate
        while (headingRate > Math.PI / dt) headingRate -= 2 * Math.PI / dt;
        while (headingRate < -Math.PI / dt) headingRate += 2 * Math.PI / dt;

        return {
            position: { x, y: height, z },
            velocity: { x: vxNorm, y: 0, z: vzNorm },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
        };
    }

    // =========================================
    // Hairpin Trajectory (Double Ellipse)
    // =========================================

    private getHairpinRadius(): number {
        return 8.0 * this.config.scale;
    }

    private getHairpinStretch(): number {
        return 3.0;  // How stretched the hairpin is
    }

    private getHairpinPeriod(): number {
        const r = this.getHairpinRadius();
        const stretch = this.getHairpinStretch();
        // Two ellipses: approximate arc length
        const ellipseArc = Math.PI * (3 * (r + r * stretch) - Math.sqrt((3 * r + r * stretch) * (r + 3 * r * stretch)));
        return (2 * ellipseArc) / this.config.speed;
    }

    private getHairpinWaypoint(t: number): Waypoint {
        const r = this.getHairpinRadius();
        const stretch = this.getHairpinStretch();
        const speed = this.config.speed;
        const height = this.config.height;
        const period = this.getHairpinPeriod();

        // Normalize t to [0, 1]
        const normalized = (t % period) / period;

        // Two hairpin turns: first half and second half
        let x: number, z: number, vx: number, vz: number;

        if (normalized < 0.5) {
            // First hairpin (right turn)
            const phase = normalized * 2 * Math.PI;
            x = r * stretch * Math.cos(phase);
            z = r * Math.sin(phase);
        } else {
            // Second hairpin (left turn, offset)
            const phase = (normalized - 0.5) * 2 * Math.PI;
            x = -r * stretch * Math.cos(phase);
            z = r * Math.sin(phase) + 2 * r;
        }

        // Compute velocity numerically
        const dt = 0.001;
        const t2 = t + dt;
        const normalized2 = (t2 % period) / period;
        let x2: number, z2: number;

        if (normalized2 < 0.5) {
            const phase2 = normalized2 * 2 * Math.PI;
            x2 = r * stretch * Math.cos(phase2);
            z2 = r * Math.sin(phase2);
        } else {
            const phase2 = (normalized2 - 0.5) * 2 * Math.PI;
            x2 = -r * stretch * Math.cos(phase2);
            z2 = r * Math.sin(phase2) + 2 * r;
        }

        vx = (x2 - x) / dt;
        vz = (z2 - z) / dt;

        // Normalize to speed
        const vMag = Math.sqrt(vx * vx + vz * vz);
        if (vMag > 0.01) {
            vx = (vx / vMag) * speed;
            vz = (vz / vMag) * speed;
        }

        // Centripetal acceleration (approximate)
        const curvature = 1 / r;  // Simplified
        const ax = -vz * curvature * speed;
        const az = vx * curvature * speed;

        const heading = Math.atan2(vx, vz);

        // Heading rate
        const heading2 = Math.atan2((x2 - x), (z2 - z));
        let headingRate = (heading2 - heading) / dt;
        while (headingRate > Math.PI / dt) headingRate -= 2 * Math.PI / dt;
        while (headingRate < -Math.PI / dt) headingRate += 2 * Math.PI / dt;

        return {
            position: { x, y: height, z },
            velocity: { x: vx, y: 0, z: vz },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
        };
    }

    // =========================================
    // Snake (Serpentine) Trajectory
    // =========================================

    private getSnakeAmplitude(): number {
        return 10.0 * this.config.scale;
    }

    private getSnakeWavelength(): number {
        return 25.0 * this.config.scale;
    }

    private getSnakeLength(): number {
        return 3 * this.getSnakeWavelength();  // 3 full waves
    }

    private getSnakePeriod(): number {
        // Approximate arc length of sine wave
        const amp = this.getSnakeAmplitude();
        const wavelength = this.getSnakeWavelength();
        const length = this.getSnakeLength();
        // Arc length ≈ length * sqrt(1 + (2πA/λ)²) for small amplitudes
        const arcLength = length * Math.sqrt(1 + Math.pow(2 * Math.PI * amp / wavelength, 2) * 0.5);
        return (2 * arcLength) / this.config.speed;  // Back and forth
    }

    private getSnakeWaypoint(t: number): Waypoint {
        const amp = this.getSnakeAmplitude();
        const wavelength = this.getSnakeWavelength();
        const length = this.getSnakeLength();
        const speed = this.config.speed;
        const height = this.config.height;
        const period = this.getSnakePeriod();

        // Normalize t to [0, 1]
        const normalized = (t % period) / period;

        // Forward pass (0 to 0.5), backward pass (0.5 to 1)
        let progress: number;
        let direction: number;

        if (normalized < 0.5) {
            progress = normalized * 2;
            direction = 1;
        } else {
            progress = (1 - normalized) * 2;
            direction = -1;
        }

        const z = (progress - 0.5) * length;
        const x = amp * Math.sin(2 * Math.PI * z / wavelength);

        // Velocity
        const dxdz = amp * (2 * Math.PI / wavelength) * Math.cos(2 * Math.PI * z / wavelength);
        const vMag = speed;
        const vzBase = vMag / Math.sqrt(1 + dxdz * dxdz);
        const vxBase = dxdz * vzBase;

        const vx = vxBase * direction;
        const vz = vzBase * direction;

        // Acceleration (curvature-based)
        const d2xdz2 = -amp * Math.pow(2 * Math.PI / wavelength, 2) * Math.sin(2 * Math.PI * z / wavelength);
        const curvature = Math.abs(d2xdz2) / Math.pow(1 + dxdz * dxdz, 1.5);
        const centripetalAccel = speed * speed * curvature;

        // Perpendicular to velocity
        const ax = -vz * centripetalAccel / vMag * Math.sign(d2xdz2);
        const az = vx * centripetalAccel / vMag * Math.sign(d2xdz2);

        const heading = Math.atan2(vx, vz);

        // Heading rate (approximate)
        const dt = 0.001;
        const progress2 = normalized + dt / period < 0.5
            ? (normalized + dt / period) * 2
            : (1 - normalized - dt / period) * 2;
        const z2 = (progress2 - 0.5) * length;
        const dxdz2 = amp * (2 * Math.PI / wavelength) * Math.cos(2 * Math.PI * z2 / wavelength);
        const heading2 = Math.atan2(dxdz2 * direction, direction);
        let headingRate = (heading2 - heading) / dt;
        while (headingRate > Math.PI / dt) headingRate -= 2 * Math.PI / dt;
        while (headingRate < -Math.PI / dt) headingRate += 2 * Math.PI / dt;

        return {
            position: { x, y: height, z },
            velocity: { x: vx, y: 0, z: vz },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
        };
    }

    // =========================================
    // Race Track (Multi-Gate Course)
    // =========================================

    private getRacetrackScale(): number {
        return 15.0 * this.config.scale;
    }

    private getRacetrackPeriod(): number {
        const s = this.getRacetrackScale();
        // Approximate track length: 2 straights + 2 semicircles + some corners
        const trackLength = 4 * s + 2 * Math.PI * s * 0.5 + 4 * s;
        return trackLength / this.config.speed;
    }

    private getRacetrackWaypoint(t: number): Waypoint {
        const s = this.getRacetrackScale();
        const speed = this.config.speed;
        const height = this.config.height;
        const period = this.getRacetrackPeriod();

        // Normalize t to [0, 1]
        const normalized = (t % period) / period;

        // Race track segments:
        // 0.00-0.20: Straight 1 (forward +Z)
        // 0.20-0.35: Turn 1 (right 180°)
        // 0.35-0.55: Straight 2 (backward -Z)
        // 0.55-0.70: Turn 2 (right 180°)
        // 0.70-0.85: Diagonal
        // 0.85-1.00: Final turn back to start

        let x: number, z: number, heading: number;
        const dt = 0.001;

        if (normalized < 0.20) {
            // Straight 1: from (s, h, -s) to (s, h, s)
            const progress = normalized / 0.20;
            x = s;
            z = -s + progress * 2 * s;
            heading = 0;
        } else if (normalized < 0.35) {
            // Turn 1: 180° right turn
            const progress = (normalized - 0.20) / 0.15;
            const angle = progress * Math.PI;
            x = s - s * 0.5 * (1 - Math.cos(angle));
            z = s + s * 0.5 * Math.sin(angle);
            heading = -angle;
        } else if (normalized < 0.55) {
            // Straight 2: backward
            const progress = (normalized - 0.35) / 0.20;
            x = 0;
            z = s + s * 0.5 - progress * 2 * s;
            heading = -Math.PI;
        } else if (normalized < 0.70) {
            // Turn 2: 180° right turn
            const progress = (normalized - 0.55) / 0.15;
            const angle = progress * Math.PI;
            x = s * 0.5 * (1 - Math.cos(angle));
            z = -s * 0.5 - s * 0.5 * Math.sin(angle);
            heading = -Math.PI - angle;
        } else if (normalized < 0.85) {
            // Diagonal back
            const progress = (normalized - 0.70) / 0.15;
            x = s * progress;
            z = -s + progress * 0;
            heading = Math.atan2(s, 0);
        } else {
            // Final approach
            x = s;
            z = -s;
            heading = 0;
        }

        // Compute velocity numerically
        const normalized2 = ((t + dt) % period) / period;
        let x2: number, z2: number;

        if (normalized2 < 0.20) {
            const progress = normalized2 / 0.20;
            x2 = s;
            z2 = -s + progress * 2 * s;
        } else if (normalized2 < 0.35) {
            const progress = (normalized2 - 0.20) / 0.15;
            const angle = progress * Math.PI;
            x2 = s - s * 0.5 * (1 - Math.cos(angle));
            z2 = s + s * 0.5 * Math.sin(angle);
        } else if (normalized2 < 0.55) {
            const progress = (normalized2 - 0.35) / 0.20;
            x2 = 0;
            z2 = s + s * 0.5 - progress * 2 * s;
        } else if (normalized2 < 0.70) {
            const progress = (normalized2 - 0.55) / 0.15;
            const angle = progress * Math.PI;
            x2 = s * 0.5 * (1 - Math.cos(angle));
            z2 = -s * 0.5 - s * 0.5 * Math.sin(angle);
        } else if (normalized2 < 0.85) {
            const progress = (normalized2 - 0.70) / 0.15;
            x2 = s * progress;
            z2 = -s;
        } else {
            x2 = s;
            z2 = -s;
        }

        let vx = (x2 - x) / dt;
        let vz = (z2 - z) / dt;

        // Normalize to speed
        const vMag = Math.sqrt(vx * vx + vz * vz);
        if (vMag > 0.01) {
            vx = (vx / vMag) * speed;
            vz = (vz / vMag) * speed;
        }

        // Recompute heading from velocity
        heading = Math.atan2(vx, vz);

        // Approximate acceleration
        const ax = 0;
        const az = 0;

        // Heading rate (computed from velocity direction)
        const headingRate = 0;

        return {
            position: { x, y: height, z },
            velocity: { x: vx, y: 0, z: vz },
            acceleration: { x: ax, y: 0, z: az },
            jerk: { x: 0, y: 0, z: 0 },
            heading,
            headingRate,
            time: t,
        };
    }
}
