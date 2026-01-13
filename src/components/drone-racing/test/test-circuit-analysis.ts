/**
 * Circuit Analysis Test
 *
 * Analyzes the trajectory through the racing circuit to understand
 * speed, curvature, and acceleration at each point - particularly
 * around the stacked gates (power loop).
 *
 * Run: npx tsx src/components/drone-racing/test/test-circuit-analysis.ts
 */

import { createTrajectory } from '../trajectory/index';
import { Trajectory } from '../trajectory/Trajectory';

// ANSI colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const MAGENTA = '\x1b[35m';
const RESET = '\x1b[0m';

interface PointAnalysis {
    phase: number;
    time: number;
    position: { x: number; y: number; z: number };
    speed: number;
    curvature: number;
    targetSpeed: number;
    accelMag: number;
    segment: string;
}

/**
 * Analyze the full circuit
 */
function analyzeCircuit(speed: number, maxCentripetalAccel: number): PointAnalysis[] {
    // Create trajectory with given parameters
    const traj = createTrajectory(speed, 4.0);

    // Override parameters
    (traj as any).speed = speed;
    (traj as any).maxCentripetalAccel = maxCentripetalAccel;
    (traj as any)._speedTable = null;  // Force rebuild

    const period = traj.getPeriod();
    const samples = 100;
    const points: PointAnalysis[] = [];

    for (let i = 0; i < samples; i++) {
        const phase = i / samples;
        const t = phase * period;
        const wp = traj.getWaypoint(t);

        const speed = Math.sqrt(
            wp.velocity.x ** 2 +
            wp.velocity.y ** 2 +
            wp.velocity.z ** 2
        );

        const accelMag = Math.sqrt(
            wp.acceleration.x ** 2 +
            wp.acceleration.y ** 2 +
            wp.acceleration.z ** 2
        );

        const curvature = (traj as any).getCurvatureAtPhase(phase);
        const targetSpeed = (traj as any).getSpeedAtPhase(phase);

        // Determine segment based on position
        let segment = 'unknown';
        const y = wp.position.y;
        const z = wp.position.z;

        if (z < 15) {
            segment = 'Start/Finish';
        } else if (z >= 15 && z < 35 && y > 8) {
            segment = '>>> CLIMB (Power Loop)';
        } else if (z >= 25 && z < 35 && y <= 8) {
            segment = '>>> DIVE (Power Loop)';
        } else if (z >= 35 && wp.position.x > 10) {
            segment = 'Far turn';
        } else if (z < 35 && z > 15 && wp.position.x > 20) {
            segment = 'Return path';
        } else if (z < 0) {
            segment = 'Final turn';
        }

        points.push({
            phase,
            time: t,
            position: { ...wp.position },
            speed,
            curvature: isFinite(curvature) ? curvature : 999,
            targetSpeed,
            accelMag,
            segment,
        });
    }

    return points;
}

/**
 * Print circuit analysis
 */
function printAnalysis(points: PointAnalysis[], config: string): void {
    console.log(`\n${CYAN}${'═'.repeat(90)}${RESET}`);
    console.log(`${CYAN}  Circuit Analysis: ${config}${RESET}`);
    console.log(`${CYAN}${'═'.repeat(90)}${RESET}\n`);

    console.log(
        'Phase'.padEnd(7) +
        'Position (x,y,z)'.padEnd(22) +
        'Speed'.padStart(8) +
        'Target'.padStart(8) +
        'Curv'.padStart(8) +
        'Accel'.padStart(8) +
        '  Segment'
    );
    console.log('─'.repeat(90));

    let prevSegment = '';

    for (const p of points) {
        // Highlight segment changes
        if (p.segment !== prevSegment && p.segment.includes('Power Loop')) {
            console.log(`${YELLOW}${'─'.repeat(90)}${RESET}`);
        }

        const posStr = `(${p.position.x.toFixed(0)}, ${p.position.y.toFixed(0)}, ${p.position.z.toFixed(0)})`;
        const speedKmh = p.speed * 3.6;
        const targetKmh = p.targetSpeed * 3.6;

        // Color speed based on whether it matches target
        const speedDiff = Math.abs(p.speed - p.targetSpeed) / p.targetSpeed;
        const speedColor = speedDiff < 0.05 ? GREEN : speedDiff < 0.2 ? YELLOW : RED;

        // Color curvature
        const curvColor = p.curvature > 1 ? RED : p.curvature > 0.5 ? YELLOW : '';

        // Color segment
        const segColor = p.segment.includes('Power Loop') ? MAGENTA : '';

        console.log(
            `${p.phase.toFixed(2)}`.padEnd(7) +
            posStr.padEnd(22) +
            `${speedColor}${speedKmh.toFixed(0)}${RESET}`.padStart(8 + (speedColor ? 9 : 0)) +
            `${targetKmh.toFixed(0)}`.padStart(8) +
            `${curvColor}${p.curvature.toFixed(2)}${RESET}`.padStart(8 + (curvColor ? 9 : 0)) +
            `${p.accelMag.toFixed(0)}`.padStart(8) +
            `  ${segColor}${p.segment}${RESET}`
        );

        prevSegment = p.segment;
    }

    // Summary statistics
    console.log(`\n${'─'.repeat(90)}`);

    const powerLoopPoints = points.filter(p => p.segment.includes('Power Loop'));
    const otherPoints = points.filter(p => !p.segment.includes('Power Loop'));

    if (powerLoopPoints.length > 0) {
        const plMinSpeed = Math.min(...powerLoopPoints.map(p => p.speed)) * 3.6;
        const plMaxSpeed = Math.max(...powerLoopPoints.map(p => p.speed)) * 3.6;
        const plMaxCurv = Math.max(...powerLoopPoints.map(p => p.curvature));
        const plMaxAccel = Math.max(...powerLoopPoints.map(p => p.accelMag));

        console.log(`\n${MAGENTA}Power Loop Stats:${RESET}`);
        console.log(`  Speed range: ${plMinSpeed.toFixed(0)} - ${plMaxSpeed.toFixed(0)} km/h`);
        console.log(`  Max curvature: ${plMaxCurv.toFixed(2)} (R = ${(1/plMaxCurv).toFixed(1)} m)`);
        console.log(`  Max acceleration: ${plMaxAccel.toFixed(0)} m/s²`);
    }

    if (otherPoints.length > 0) {
        const otherMinSpeed = Math.min(...otherPoints.map(p => p.speed)) * 3.6;
        const otherMaxSpeed = Math.max(...otherPoints.map(p => p.speed)) * 3.6;

        console.log(`\n${CYAN}Other Segments:${RESET}`);
        console.log(`  Speed range: ${otherMinSpeed.toFixed(0)} - ${otherMaxSpeed.toFixed(0)} km/h`);
    }
}

/**
 * Test different configurations
 */
function testConfigurations(): void {
    console.log(`\n${CYAN}${'═'.repeat(90)}${RESET}`);
    console.log(`${CYAN}  Testing Different Speed Configurations${RESET}`);
    console.log(`${CYAN}${'═'.repeat(90)}${RESET}\n`);

    // Configurations: [speed, maxCentripetalAccel, description]
    const configs: [number, number, string][] = [
        [18, 15, 'Original (65 km/h, 15 m/s²)'],
        [25, 15, '90 km/h, 15 m/s² centripetal'],
        [25, 20, '90 km/h, 20 m/s² centripetal'],
        [25, 10, '90 km/h, 10 m/s² centripetal (more slowdown)'],
        [25, 8,  '90 km/h, 8 m/s² centripetal (aggressive slowdown)'],
        [30, 15, '108 km/h, 15 m/s² centripetal'],
        [30, 10, '108 km/h, 10 m/s² centripetal'],
    ];

    console.log(
        'Config'.padEnd(45) +
        'Max'.padStart(8) +
        'Min'.padStart(8) +
        'PL Min'.padStart(8) +
        'PL Curv'.padStart(10)
    );
    console.log('─'.repeat(85));

    for (const [speed, accel, desc] of configs) {
        const points = analyzeCircuit(speed, accel);

        const maxSpeed = Math.max(...points.map(p => p.speed)) * 3.6;
        const minSpeed = Math.min(...points.map(p => p.speed)) * 3.6;

        const powerLoopPoints = points.filter(p => p.segment.includes('Power Loop'));
        const plMinSpeed = powerLoopPoints.length > 0
            ? Math.min(...powerLoopPoints.map(p => p.speed)) * 3.6
            : 0;
        const plMaxCurv = powerLoopPoints.length > 0
            ? Math.max(...powerLoopPoints.map(p => p.curvature))
            : 0;

        console.log(
            desc.padEnd(45) +
            `${maxSpeed.toFixed(0)} km/h`.padStart(8) +
            `${minSpeed.toFixed(0)} km/h`.padStart(8) +
            `${plMinSpeed.toFixed(0)} km/h`.padStart(8) +
            `${plMaxCurv.toFixed(2)}`.padStart(10)
        );
    }
}

/**
 * Detailed analysis of specific config
 */
function detailedAnalysis(): void {
    // Current config
    const points = analyzeCircuit(25, 20);
    printAnalysis(points, 'Current: 25 m/s, 20 m/s² centripetal');

    // More aggressive slowdown
    const points2 = analyzeCircuit(25, 10);
    printAnalysis(points2, 'More slowdown: 25 m/s, 10 m/s² centripetal');
}

// Main
async function main(): Promise<void> {
    testConfigurations();
    detailedAnalysis();

    console.log(`\n${YELLOW}Recommendations:${RESET}`);
    console.log('─'.repeat(60));
    console.log('1. Lower maxCentripetalAccel = more slowdown in corners');
    console.log('2. Power loop needs very low speed due to tight radius');
    console.log('3. Try maxCentripetalAccel = 8-10 m/s² for power loop');
    console.log('');
}

main().catch(console.error);
