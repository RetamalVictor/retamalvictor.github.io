/**
 * Trajectory Analysis Tool
 *
 * Analyzes trajectory feasibility by computing required accelerations
 * and comparing them to drone capabilities.
 */

import { createTrajectory, TrajectoryType } from '../trajectory';

const DRONE_MAX_ACCEL = 30.0;  // m/s² (what MPC can command)
const SAMPLE_DT = 0.01;        // 100 Hz sampling

interface AnalysisPoint {
    t: number;
    phase: number;
    position: { x: number; y: number; z: number };
    velocity: { x: number; y: number; z: number };
    speed: number;
    acceleration: { x: number; y: number; z: number };
    accelMag: number;
    curvature: number;  // 1/r
    requiredCentripetalAccel: number;  // v²/r
}

function analyzeTrajectory(name: string, type: TrajectoryType) {
    console.log('\n' + '='.repeat(80));
    console.log(`TRAJECTORY ANALYSIS: ${name}`);
    console.log('='.repeat(80));

    const trajectory = createTrajectory(type);
    const period = trajectory.getPeriod();

    console.log(`\nTrajectory period: ${period.toFixed(2)} s`);
    console.log(`Sampling at ${1/SAMPLE_DT} Hz`);

    const points: AnalysisPoint[] = [];
    const numSamples = Math.ceil(period / SAMPLE_DT);

    // Sample trajectory
    for (let i = 0; i < numSamples; i++) {
        const t = i * SAMPLE_DT;
        const wp = trajectory.getWaypoint(t);

        const speed = Math.sqrt(
            wp.velocity.x ** 2 + wp.velocity.y ** 2 + wp.velocity.z ** 2
        );
        const accelMag = Math.sqrt(
            wp.acceleration.x ** 2 + wp.acceleration.y ** 2 + wp.acceleration.z ** 2
        );

        points.push({
            t,
            phase: t / period,
            position: { ...wp.position },
            velocity: { ...wp.velocity },
            speed,
            acceleration: { ...wp.acceleration },
            accelMag,
            curvature: 0,
            requiredCentripetalAccel: 0,
        });
    }

    // Compute curvature using finite differences
    // curvature = |dv/ds| = |a_perp| / v²
    for (let i = 1; i < points.length - 1; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const next = points[i + 1];

        const dvx = (next.velocity.x - prev.velocity.x) / (2 * SAMPLE_DT);
        const dvy = (next.velocity.y - prev.velocity.y) / (2 * SAMPLE_DT);
        const dvz = (next.velocity.z - prev.velocity.z) / (2 * SAMPLE_DT);

        const vNorm = curr.speed > 0.1 ? curr.speed : 0.1;
        const vx = curr.velocity.x / vNorm;
        const vy = curr.velocity.y / vNorm;
        const vz = curr.velocity.z / vNorm;

        const aParallel = dvx * vx + dvy * vy + dvz * vz;
        const aPerpX = dvx - aParallel * vx;
        const aPerpY = dvy - aParallel * vy;
        const aPerpZ = dvz - aParallel * vz;
        const aPerpMag = Math.sqrt(aPerpX * aPerpX + aPerpY * aPerpY + aPerpZ * aPerpZ);

        const curvature = curr.speed > 0.1 ? aPerpMag / (curr.speed * curr.speed) : 0;
        const requiredAccel = curr.speed * curr.speed * curvature;

        curr.curvature = curvature;
        curr.requiredCentripetalAccel = requiredAccel;
    }

    const maxAccel = Math.max(...points.map(p => p.requiredCentripetalAccel));
    const maxSpeed = Math.max(...points.map(p => p.speed));
    const avgSpeed = points.reduce((s, p) => s + p.speed, 0) / points.length;
    const maxCurvature = Math.max(...points.map(p => p.curvature));
    const minRadius = maxCurvature > 0 ? 1 / maxCurvature : Infinity;

    console.log('\n--- STATISTICS ---');
    console.log(`Max speed: ${maxSpeed.toFixed(2)} m/s (${(maxSpeed * 3.6).toFixed(0)} km/h)`);
    console.log(`Avg speed: ${avgSpeed.toFixed(2)} m/s (${(avgSpeed * 3.6).toFixed(0)} km/h)`);
    console.log(`Min turn radius: ${minRadius.toFixed(2)} m`);
    console.log(`Max required centripetal accel: ${maxAccel.toFixed(2)} m/s²`);
    console.log(`Drone max accel: ${DRONE_MAX_ACCEL} m/s²`);
    console.log(`Feasibility margin: ${((DRONE_MAX_ACCEL - maxAccel) / DRONE_MAX_ACCEL * 100).toFixed(1)}%`);

    if (maxAccel > DRONE_MAX_ACCEL) {
        console.log(`\n⚠ Trajectory exceeds drone capability!`);
        const speedReduction = Math.sqrt(DRONE_MAX_ACCEL / maxAccel);
        console.log(`  Required speed reduction: ${((1 - speedReduction) * 100).toFixed(1)}%`);
        console.log(`  Or increase min turn radius to: ${(maxSpeed * maxSpeed / DRONE_MAX_ACCEL).toFixed(1)} m`);
    } else if (maxAccel > DRONE_MAX_ACCEL * 0.8) {
        console.log(`\n⚠ Marginally feasible (>80% of max accel)`);
    } else {
        console.log(`\n✓ Trajectory is feasible`);
    }
}

// Run analysis
console.log('TRAJECTORY FEASIBILITY ANALYSIS');
console.log('================================');
console.log(`Drone max acceleration: ${DRONE_MAX_ACCEL} m/s²`);
console.log(`Required min turn radius at 70 km/h: ${(19.4 * 19.4 / DRONE_MAX_ACCEL).toFixed(1)} m`);

analyzeTrajectory('Figure-8 (Racing Scale)', 'figure8');
analyzeTrajectory('Split-S (Racing Scale)', 'splits');
analyzeTrajectory('Dive (Stacked Gates)', 'dive');
analyzeTrajectory('Crazy (Chaos Mode)', 'crazy');

console.log('\n' + '='.repeat(80));
console.log('ANALYSIS COMPLETE');
console.log('='.repeat(80));
