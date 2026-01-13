/**
 * Base Trajectory Interface and Abstract Class
 *
 * Defines the contract for all trajectory types and provides
 * common utility methods for trajectory generation.
 */

import { Waypoint, GatePosition } from '../types';

/**
 * Trajectory configuration shared by all trajectory types
 */
export interface TrajectoryParams {
    speed: number;      // Target speed in m/s (set by trajectory generator based on path)
    height: number;     // Flight altitude in m
}

/**
 * Abstract base class for all trajectory types
 *
 * Provides:
 * - Common interface for waypoint generation
 * - Numerical differentiation utilities
 * - Progress-based (arc-length) parameterization for MPC
 * - Trajectory visualization helpers
 */
export abstract class Trajectory {
    protected speed: number;
    protected height: number;

    // Arc-length parameterization (computed lazily)
    private _arcLength: number | null = null;
    private _phaseToArcTable: { phase: number; arc: number }[] | null = null;
    private readonly ARC_TABLE_RESOLUTION = 500;  // Number of samples for arc-length table

    // Variable speed profile (computed lazily)
    private _speedTable: number[] | null = null;
    private readonly SPEED_TABLE_RESOLUTION = 200;
    private readonly speedSmoothingAlpha = 0.2;  // 0..1 (higher = less smoothing)
    private readonly maxCentripetalAccel = 15;   // m/s² max centripetal acceleration

    constructor(params: TrajectoryParams) {
        this.speed = params.speed;
        this.height = params.height;
    }

    // ============================================
    // Arc-Length / Progress-Based API
    // ============================================

    /**
     * Get the total arc length of the trajectory
     * Computed once and cached.
     */
    public get arcLength(): number {
        if (this._arcLength === null) {
            this.computeArcLengthTable();
        }
        return this._arcLength!;
    }

    /**
     * Whether this trajectory is periodic (loops back to start)
     * Override in subclasses for non-periodic trajectories (e.g., point-to-point paths)
     * Default: true (most racing tracks are loops)
     */
    public isPeriodic(): boolean {
        return true;
    }

    /**
     * Compute arc-length lookup table using numerical integration
     */
    private computeArcLengthTable(): void {
        const n = this.ARC_TABLE_RESOLUTION;
        this._phaseToArcTable = [];
        let cumulativeArc = 0;
        let prevPos = this.getPositionAtPhase(0);
        let prevY = prevPos.y !== undefined ? prevPos.y : this.height;

        this._phaseToArcTable.push({ phase: 0, arc: 0 });

        for (let i = 1; i <= n; i++) {
            const phase = i / n;
            const pos = this.getPositionAtPhase(phase);
            const y = pos.y !== undefined ? pos.y : this.height;

            // Compute segment length
            const dx = pos.x - prevPos.x;
            const dy = y - prevY;
            const dz = pos.z - prevPos.z;
            const segmentLength = Math.sqrt(dx * dx + dy * dy + dz * dz);

            cumulativeArc += segmentLength;
            this._phaseToArcTable.push({ phase, arc: cumulativeArc });

            prevPos = pos;
            prevY = y;
        }

        this._arcLength = cumulativeArc;
    }

    /**
     * Convert arc-length progress to phase using lookup table
     * Handles wrapping for periodic trajectories, clamping for non-periodic.
     */
    protected progressToPhase(s: number): number {
        if (this._phaseToArcTable === null) {
            this.computeArcLengthTable();
        }
        const table = this._phaseToArcTable!;
        const totalArc = this._arcLength!;

        // Handle s bounds based on periodicity
        let sWrapped: number;
        if (this.isPeriodic()) {
            // Wrap s to [0, totalArc) for periodic trajectories
            sWrapped = s % totalArc;
            if (sWrapped < 0) sWrapped += totalArc;
        } else {
            // Clamp s to [0, totalArc] for non-periodic trajectories
            sWrapped = Math.max(0, Math.min(totalArc, s));
        }

        // Binary search for the interval
        let lo = 0;
        let hi = table.length - 1;
        while (lo < hi - 1) {
            const mid = Math.floor((lo + hi) / 2);
            if (table[mid].arc <= sWrapped) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation within interval
        const arc0 = table[lo].arc;
        const arc1 = table[hi].arc;
        const phase0 = table[lo].phase;
        const phase1 = table[hi].phase;

        if (Math.abs(arc1 - arc0) < 1e-10) {
            return phase0;
        }

        const t = (sWrapped - arc0) / (arc1 - arc0);
        return phase0 + t * (phase1 - phase0);
    }

    /**
     * Convert phase to arc-length progress
     */
    protected phaseToProgress(phase: number): number {
        if (this._phaseToArcTable === null) {
            this.computeArcLengthTable();
        }
        const table = this._phaseToArcTable!;

        // Wrap phase to [0, 1)
        let phaseWrapped = phase % 1;
        if (phaseWrapped < 0) phaseWrapped += 1;

        // Binary search for the interval
        let lo = 0;
        let hi = table.length - 1;
        while (lo < hi - 1) {
            const mid = Math.floor((lo + hi) / 2);
            if (table[mid].phase <= phaseWrapped) {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation within interval
        const phase0 = table[lo].phase;
        const phase1 = table[hi].phase;
        const arc0 = table[lo].arc;
        const arc1 = table[hi].arc;

        if (Math.abs(phase1 - phase0) < 1e-10) {
            return arc0;
        }

        const t = (phaseWrapped - phase0) / (phase1 - phase0);
        return arc0 + t * (arc1 - arc0);
    }

    /**
     * Get position at arc-length progress s
     */
    public getPositionAtProgress(s: number): { x: number; y: number; z: number } {
        const phase = this.progressToPhase(s);
        const pos = this.getPositionAtPhase(phase);
        const y = pos.y !== undefined ? pos.y : this.height;
        return { x: pos.x, y, z: pos.z };
    }

    /**
     * Get unit tangent vector at arc-length progress s
     */
    public getTangentAtProgress(s: number): { x: number; y: number; z: number } {
        // Use central difference for tangent
        // Bounded ds: 1cm minimum (numerical stability), 50cm maximum (corner accuracy)
        const ds = Math.max(0.01, Math.min(0.5, this.arcLength * 1e-4));
        const p1 = this.getPositionAtProgress(s - ds);
        const p2 = this.getPositionAtProgress(s + ds);

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const dz = p2.z - p1.z;
        const mag = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (mag < 1e-10) {
            return { x: 0, y: 0, z: 1 };  // Default tangent
        }

        return { x: dx / mag, y: dy / mag, z: dz / mag };
    }

    /**
     * Get full waypoint at arc-length progress s
     */
    public getWaypointAtProgress(s: number): Waypoint {
        const phase = this.progressToPhase(s);
        const period = this.getPeriod();
        const t = phase * period;
        return this.getWaypoint(t);
    }

    /**
     * Find closest progress to a position with windowed search and continuity bias
     *
     * @param pos - Query position
     * @param prevS - Previous progress estimate (for continuity)
     * @param window - Search window size in arc-length units (default: arcLength/4)
     * @param lambda - Continuity bias weight (default: 0.01)
     * @returns Closest progress value
     */
    public findClosestProgress(
        pos: { x: number; y: number; z: number },
        prevS?: number,
        window?: number,
        lambda: number = 0.01
    ): number {
        const totalArc = this.arcLength;
        const searchWindow = window ?? totalArc / 4;
        const numSamples = 100;  // Samples within the search window

        // Determine search range
        let sMin: number, sMax: number;
        if (prevS !== undefined) {
            // Windowed search around previous progress
            sMin = prevS - searchWindow;
            sMax = prevS + searchWindow;
        } else {
            // Full trajectory search
            sMin = 0;
            sMax = totalArc;
        }

        let bestS = prevS ?? 0;
        let bestCost = Infinity;

        for (let i = 0; i <= numSamples; i++) {
            const s = sMin + (i / numSamples) * (sMax - sMin);
            const trajPos = this.getPositionAtProgress(s);

            // Distance cost
            const dx = pos.x - trajPos.x;
            const dy = pos.y - trajPos.y;
            const dz = pos.z - trajPos.z;
            const distSq = dx * dx + dy * dy + dz * dz;

            // Continuity bias (tie-breaker for self-intersections)
            const continuityTerm = prevS !== undefined ? lambda * (s - prevS) * (s - prevS) : 0;

            const cost = distSq + continuityTerm;

            if (cost < bestCost) {
                bestCost = cost;
                bestS = s;
            }
        }

        // Refine with golden section search
        bestS = this.refineClosestProgress(pos, bestS, totalArc / numSamples, prevS, lambda);

        return bestS;
    }

    /**
     * Refine closest progress using golden section search
     */
    private refineClosestProgress(
        pos: { x: number; y: number; z: number },
        initialS: number,
        initialRange: number,
        prevS: number | undefined,
        lambda: number
    ): number {
        const phi = (1 + Math.sqrt(5)) / 2;
        let a = initialS - initialRange;
        let b = initialS + initialRange;

        const costFunc = (s: number): number => {
            const trajPos = this.getPositionAtProgress(s);
            const dx = pos.x - trajPos.x;
            const dy = pos.y - trajPos.y;
            const dz = pos.z - trajPos.z;
            const distSq = dx * dx + dy * dy + dz * dz;
            const continuityTerm = prevS !== undefined ? lambda * (s - prevS) * (s - prevS) : 0;
            return distSq + continuityTerm;
        };

        let c = b - (b - a) / phi;
        let d = a + (b - a) / phi;

        for (let i = 0; i < 20; i++) {  // ~20 iterations for good precision
            if (costFunc(c) < costFunc(d)) {
                b = d;
            } else {
                a = c;
            }
            c = b - (b - a) / phi;
            d = a + (b - a) / phi;
        }

        return (a + b) / 2;
    }

    /**
     * Get the period (lap time) of the trajectory in seconds
     */
    abstract getPeriod(): number;

    /**
     * Get the name of this trajectory type
     */
    abstract getName(): string;

    /**
     * Get position at normalized time t ∈ [0, 1]
     * Returns { x, y, z } - subclasses can override y for 3D trajectories
     * Default y = this.height for 2D trajectories
     */
    protected abstract getPositionAtPhase(phase: number): { x: number; y?: number; z: number };

    /**
     * Get waypoint at time t
     * Uses variable speed based on curvature for smooth cornering
     */
    public getWaypoint(t: number): Waypoint {
        const period = this.getPeriod();
        let phase = (t % period) / period;
        if (phase < 0) phase += 1;

        // Get position (y defaults to this.height for 2D trajectories)
        const pos = this.getPositionAtPhase(phase);
        const y = pos.y !== undefined ? pos.y : this.height;

        // Get target speed at this phase (curvature-based)
        const targetSpeed = this.getSpeedAtPhase(phase);

        // Compute velocity using phaseDot-consistent neighbor sampling
        // phase ± (phaseDot * dt) for time↔phase consistency
        const dt = 0.0001;
        const phaseDot = this.getPhaseDot(phase);
        const dPhase = phaseDot * dt;

        let phase1 = phase - dPhase;
        let phase2 = phase + dPhase;
        if (this.isPeriodic()) {
            phase1 = ((phase1 % 1) + 1) % 1;
            phase2 = phase2 % 1;
        } else {
            phase1 = Math.max(0, phase1);
            phase2 = Math.min(1, phase2);
        }

        const pos1 = this.getPositionAtPhase(phase1);
        const pos2 = this.getPositionAtPhase(phase2);

        const y1 = pos1.y !== undefined ? pos1.y : this.height;
        const y2 = pos2.y !== undefined ? pos2.y : this.height;

        let vx = (pos2.x - pos1.x) / (2 * dt);
        let vy = (y2 - y1) / (2 * dt);
        let vz = (pos2.z - pos1.z) / (2 * dt);

        // Normalize velocity to target speed (variable, based on curvature)
        const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vMag > 1e-6) {
            const scale = targetSpeed / vMag;
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        // Compute acceleration using central difference on velocity
        const dt2 = 0.001;
        const wp1 = this.getVelocityAtPhase(phase - this.getPhaseDot(phase) * dt2);
        const wp2 = this.getVelocityAtPhase(phase + this.getPhaseDot(phase) * dt2);
        const ax = (wp2.vx - wp1.vx) / (2 * dt2);
        const ay = (wp2.vy - wp1.vy) / (2 * dt2);
        const az = (wp2.vz - wp1.vz) / (2 * dt2);

        // Compute jerk using central difference on acceleration (with boundary handling)
        const dt3 = 0.002;
        let jx = 0, jy = 0, jz = 0;
        if (t > dt3 && t < period - dt3) {
            const a1 = this.getAccelerationAtPhase(phase - this.getPhaseDot(phase) * dt3);
            const a2 = this.getAccelerationAtPhase(phase + this.getPhaseDot(phase) * dt3);
            jx = (a2.ax - a1.ax) / (2 * dt3);
            jy = (a2.ay - a1.ay) / (2 * dt3);
            jz = (a2.az - a1.az) / (2 * dt3);
        } else {
            // Boundary: use forward/backward difference
            const a0 = this.getAccelerationAtPhase(phase);
            const a1 = this.getAccelerationAtPhase(Math.min(1, phase + this.getPhaseDot(phase) * dt3));
            jx = (a1.ax - a0.ax) / dt3;
            jy = (a1.ay - a0.ay) / dt3;
            jz = (a1.az - a0.az) / dt3;
        }

        // Clamp jerk to physically reasonable bounds (numerical diff can be noisy)
        const maxJerk = 100; // m/s³ - reasonable for aggressive racing maneuvers
        const jerkMag = Math.sqrt(jx * jx + jy * jy + jz * jz);
        if (jerkMag > maxJerk) {
            const scale = maxJerk / jerkMag;
            jx *= scale;
            jy *= scale;
            jz *= scale;
        }

        // Compute heading and heading rate (heading is in XZ plane)
        const heading = Math.atan2(vx, vz);
        const heading1 = Math.atan2(wp1.vx, wp1.vz);
        const heading2 = Math.atan2(wp2.vx, wp2.vz);
        let headingRate = this.wrapAngle(heading2 - heading1) / (2 * dt2);

        // Clamp heading rate to physically reasonable limits (max ~5 rad/s = 286°/s)
        const maxHeadingRate = 5.0;  // rad/s
        headingRate = Math.max(-maxHeadingRate, Math.min(maxHeadingRate, headingRate));

        return {
            position: { x: pos.x, y, z: pos.z },
            velocity: { x: vx, y: vy, z: vz },
            acceleration: { x: ax, y: ay, z: az },
            jerk: { x: jx, y: jy, z: jz },
            heading,
            headingRate,
            time: t,
        };
    }

    /**
     * Helper to get velocity at a specific phase (for numerical differentiation)
     * Uses variable speed based on curvature
     */
    private getVelocityAtPhase(phase: number): { vx: number; vy: number; vz: number } {
        // Wrap phase to [0, 1)
        let phaseWrapped = phase;
        if (this.isPeriodic()) {
            phaseWrapped = ((phase % 1) + 1) % 1;
        } else {
            phaseWrapped = Math.max(0, Math.min(1, phase));
        }

        // Get target speed at this phase
        const targetSpeed = this.getSpeedAtPhase(phaseWrapped);

        // Compute velocity direction using phaseDot-consistent neighbor sampling
        const dt = 0.0001;
        const phaseDot = this.getPhaseDot(phaseWrapped);
        const dPhase = phaseDot * dt;

        let phase1 = phaseWrapped - dPhase;
        let phase2 = phaseWrapped + dPhase;
        if (this.isPeriodic()) {
            phase1 = ((phase1 % 1) + 1) % 1;
            phase2 = phase2 % 1;
        } else {
            phase1 = Math.max(0, phase1);
            phase2 = Math.min(1, phase2);
        }

        const pos1 = this.getPositionAtPhase(phase1);
        const pos2 = this.getPositionAtPhase(phase2);

        const y1 = pos1.y !== undefined ? pos1.y : this.height;
        const y2 = pos2.y !== undefined ? pos2.y : this.height;

        let vx = (pos2.x - pos1.x) / (2 * dt);
        let vy = (y2 - y1) / (2 * dt);
        let vz = (pos2.z - pos1.z) / (2 * dt);

        // Normalize velocity to target speed
        const vMag = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (vMag > 1e-6) {
            const scale = targetSpeed / vMag;
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        return { vx, vy, vz };
    }

    /**
     * Helper to get acceleration at a specific phase (for jerk computation)
     * Uses central difference on velocity
     */
    private getAccelerationAtPhase(phase: number): { ax: number; ay: number; az: number } {
        // Wrap phase to [0, 1)
        let phaseWrapped = phase;
        if (this.isPeriodic()) {
            phaseWrapped = ((phase % 1) + 1) % 1;
        } else {
            phaseWrapped = Math.max(0, Math.min(1, phase));
        }

        const dt2 = 0.001;
        const phaseDot = this.getPhaseDot(phaseWrapped);
        const dPhase = phaseDot * dt2;

        const v1 = this.getVelocityAtPhase(phaseWrapped - dPhase);
        const v2 = this.getVelocityAtPhase(phaseWrapped + dPhase);

        return {
            ax: (v2.vx - v1.vx) / (2 * dt2),
            ay: (v2.vy - v1.vy) / (2 * dt2),
            az: (v2.vz - v1.vz) / (2 * dt2),
        };
    }

    // ============================================
    // Variable Speed Profile (Curvature-Based)
    // ============================================

    /**
     * Get curvature at a given phase using centered derivatives
     * κ = |r' × r''| / |r'|³
     */
    protected getCurvatureAtPhase(phase: number): number {
        const eps = 1e-4;

        // Handle phase wrapping
        let pm = phase - eps;
        let pp = phase + eps;
        if (this.isPeriodic()) {
            pm = ((pm % 1) + 1) % 1;
            pp = pp % 1;
        } else {
            pm = Math.max(0, pm);
            pp = Math.min(1, pp);
        }

        const posMinus = this.getPositionAtPhase(pm);
        const pos0 = this.getPositionAtPhase(phase);
        const posPlus = this.getPositionAtPhase(pp);

        const ym = posMinus.y !== undefined ? posMinus.y : this.height;
        const y0 = pos0.y !== undefined ? pos0.y : this.height;
        const yp = posPlus.y !== undefined ? posPlus.y : this.height;

        // r' ≈ (p(φ+ε) - p(φ-ε)) / (2ε)
        const r1x = (posPlus.x - posMinus.x) / (2 * eps);
        const r1y = (yp - ym) / (2 * eps);
        const r1z = (posPlus.z - posMinus.z) / (2 * eps);

        // r'' ≈ (p(φ+ε) - 2p(φ) + p(φ-ε)) / (ε²)
        const r2x = (posPlus.x - 2 * pos0.x + posMinus.x) / (eps * eps);
        const r2y = (yp - 2 * y0 + ym) / (eps * eps);
        const r2z = (posPlus.z - 2 * pos0.z + posMinus.z) / (eps * eps);

        // Cross product r1 × r2
        const crossX = r1y * r2z - r1z * r2y;
        const crossY = r1z * r2x - r1x * r2z;
        const crossZ = r1x * r2y - r1y * r2x;

        const crossMag = Math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ);
        const r1Mag = Math.sqrt(r1x * r1x + r1y * r1y + r1z * r1z);

        // κ = |r' × r''| / |r'|³
        return crossMag / (r1Mag * r1Mag * r1Mag + 1e-10);
    }

    /**
     * Build smoothed speed table from curvature constraints
     * Uses forward + backward exponential filtering
     */
    private buildSpeedTable(): number[] {
        const N = this.SPEED_TABLE_RESOLUTION;
        const raw: number[] = [];

        // 1. Compute raw speed from curvature constraint: v = min(vMax, sqrt(aMax / κ))
        for (let i = 0; i <= N; i++) {
            const phase = i / N;
            const kappa = this.getCurvatureAtPhase(phase);
            const vFromCurvature = Math.sqrt(this.maxCentripetalAccel / (kappa + 1e-6));
            raw.push(Math.min(this.speed, vFromCurvature));
        }

        // 2. Smooth with exponential filter (forward + backward pass)
        const smooth = [...raw];
        const alpha = this.speedSmoothingAlpha;

        // Forward pass
        for (let i = 1; i <= N; i++) {
            smooth[i] = alpha * raw[i] + (1 - alpha) * smooth[i - 1];
        }
        // Backward pass
        for (let i = N - 1; i >= 0; i--) {
            smooth[i] = alpha * smooth[i] + (1 - alpha) * smooth[i + 1];
        }

        return smooth;
    }

    /**
     * Get target speed at a given phase (from smoothed speed table)
     */
    protected getSpeedAtPhase(phase: number): number {
        // Build speed table on first call
        if (!this._speedTable) {
            this._speedTable = this.buildSpeedTable();
        }

        // Wrap phase to [0, 1)
        let phaseWrapped = phase % 1;
        if (phaseWrapped < 0) phaseWrapped += 1;

        // Interpolate from table
        const N = this._speedTable.length - 1;
        const idx = phaseWrapped * N;
        const i0 = Math.floor(idx);
        const i1 = Math.min(i0 + 1, N);
        const frac = idx - i0;

        return this._speedTable[i0] * (1 - frac) + this._speedTable[i1] * frac;
    }

    /**
     * Get phase rate: φ̇ = v(φ) / |p'(φ)|
     * This ensures velocity magnitude matches getSpeedAtPhase()
     */
    protected getPhaseDot(phase: number): number {
        const eps = 1e-4;

        // Handle phase wrapping
        let pm = phase - eps;
        let pp = phase + eps;
        if (this.isPeriodic()) {
            pm = ((pm % 1) + 1) % 1;
            pp = pp % 1;
        } else {
            pm = Math.max(0, pm);
            pp = Math.min(1, pp);
        }

        const posMinus = this.getPositionAtPhase(pm);
        const posPlus = this.getPositionAtPhase(pp);

        const ym = posMinus.y !== undefined ? posMinus.y : this.height;
        const yp = posPlus.y !== undefined ? posPlus.y : this.height;

        // p'(φ) ≈ (p(φ+ε) - p(φ-ε)) / (2ε)
        const dpx = (posPlus.x - posMinus.x) / (2 * eps);
        const dpy = (yp - ym) / (2 * eps);
        const dpz = (posPlus.z - posMinus.z) / (2 * eps);
        const dpMag = Math.sqrt(dpx * dpx + dpy * dpy + dpz * dpz);

        const v = this.getSpeedAtPhase(phase);
        return v / (dpMag + 1e-9);
    }

    /**
     * Wrap angle to [-π, π]
     */
    protected wrapAngle(angle: number): number {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }

    /**
     * Get trajectory points for visualization
     */
    public getVisualizationPoints(numPoints: number = 200): { x: number; y: number; z: number }[] {
        const points: { x: number; y: number; z: number }[] = [];

        for (let i = 0; i <= numPoints; i++) {
            const phase = i / numPoints;
            const pos = this.getPositionAtPhase(phase);
            const y = pos.y !== undefined ? pos.y : this.height;
            points.push({ x: pos.x, y, z: pos.z });
        }

        return points;
    }

    /**
     * Get initial state for starting the trajectory
     */
    public getInitialState(): { position: { x: number; y: number; z: number }; heading: number } {
        const wp = this.getWaypoint(0);
        return {
            position: wp.position,
            heading: wp.heading,
        };
    }

    /**
     * Get gate positions for this trajectory
     * Default implementation places gates at regular intervals
     * Subclasses can override to place gates at specific points
     */
    public getGatePositions(numGates: number = 4): GatePosition[] {
        const gates: GatePosition[] = [];
        const period = this.getPeriod();

        for (let i = 0; i < numGates; i++) {
            const phase = i / numGates;
            const t = phase * period;
            const wp = this.getWaypoint(t);

            gates.push({
                position: { ...wp.position },
                heading: wp.heading,
            });
        }

        return gates;
    }
}
