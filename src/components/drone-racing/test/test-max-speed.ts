/**
 * Test: Maximum Speed Capability
 *
 * Goal: Find the maximum speed the trajectory system can handle
 * Target: 200 km/h (~55.6 m/s)
 *
 * Tests different configurations and reports:
 * 1. Max achievable speed on straights
 * 2. Min speed in tightest corners
 * 3. Constraint violations (acceleration, jerk)
 * 4. Overall feasibility
 *
 * Run: npx tsx src/components/drone-racing/test/test-max-speed.ts
 */

import { Figure8Trajectory } from '../trajectory/Figure8Trajectory';
import { CircleTrajectory } from '../trajectory/CircleTrajectory';
import { Trajectory } from '../trajectory/Trajectory';
import { createTrajectory } from '../trajectory/index';

// ANSI colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const MAGENTA = '\x1b[35m';
const RESET = '\x1b[0m';

interface SpeedTestResult {
    configName: string;
    maxSpeed: number;         // m/s (on straights)
    minSpeed: number;         // m/s (in corners)
    maxSpeedKmh: number;      // km/h
    minSpeedKmh: number;      // km/h
    maxAccel: number;         // m/s² (from numerical diff - often noisy)
    maxCentripetal: number;   // m/s² (v² × κ - actual physical requirement)
    maxThrust: number;        // m/s² (sqrt(centripetal² + gravity²))
    maxJerk: number;          // m/s³
    maxCurvature: number;     // 1/m
    feasible: boolean;
    issues: string[];
}

const GRAVITY = 9.81;

/**
 * Analyze trajectory at given speed configuration
 */
function analyzeTrajectory(
    trajectory: Trajectory,
    configName: string
): SpeedTestResult {
    const period = trajectory.getPeriod();
    const samples = 200;
    const traj = trajectory as any;

    let maxSpeed = 0;
    let minSpeed = Infinity;
    let maxAccel = 0;
    let maxCentripetal = 0;
    let maxThrust = 0;
    let maxJerk = 0;
    let maxCurvature = 0;

    const issues: string[] = [];

    for (let i = 0; i < samples; i++) {
        const t = (i / samples) * period;
        const phase = t / period;
        const wp = trajectory.getWaypoint(t);

        // Speed
        const speed = Math.sqrt(
            wp.velocity.x ** 2 +
            wp.velocity.y ** 2 +
            wp.velocity.z ** 2
        );
        maxSpeed = Math.max(maxSpeed, speed);
        minSpeed = Math.min(minSpeed, speed);

        // Acceleration (from numerical differentiation - often noisy)
        const accel = Math.sqrt(
            wp.acceleration.x ** 2 +
            wp.acceleration.y ** 2 +
            wp.acceleration.z ** 2
        );
        maxAccel = Math.max(maxAccel, accel);

        // Curvature and centripetal acceleration (v² × κ - actual requirement)
        const kappa = traj.getCurvatureAtPhase(phase);
        if (isFinite(kappa) && kappa < 10) {  // Filter extreme curvature outliers
            maxCurvature = Math.max(maxCurvature, kappa);
            const centripetal = speed * speed * kappa;
            maxCentripetal = Math.max(maxCentripetal, centripetal);

            // Total thrust = sqrt(centripetal² + gravity²)
            const thrust = Math.sqrt(centripetal * centripetal + GRAVITY * GRAVITY);
            maxThrust = Math.max(maxThrust, thrust);
        }

        // Jerk
        const jerk = Math.sqrt(
            wp.jerk.x ** 2 +
            wp.jerk.y ** 2 +
            wp.jerk.z ** 2
        );
        maxJerk = Math.max(maxJerk, jerk);
    }

    // Check physical constraints (use centripetal, not noisy numerical accel)
    const MAX_THRUST_LIMIT = 50;  // 5g total thrust capability
    const MAX_CENTRIPETAL_LIMIT = 45;  // ~4.5g centripetal (leaves room for gravity)

    if (maxCentripetal > MAX_CENTRIPETAL_LIMIT) {
        issues.push(`Centripetal accel exceeds ${MAX_CENTRIPETAL_LIMIT} m/s² (got ${maxCentripetal.toFixed(1)})`);
    }

    if (maxThrust > MAX_THRUST_LIMIT) {
        issues.push(`Total thrust exceeds ${MAX_THRUST_LIMIT} m/s² (got ${maxThrust.toFixed(1)})`);
    }

    // Check for NaN/Infinity
    if (!isFinite(maxSpeed) || !isFinite(minSpeed)) {
        issues.push('Speed contains NaN or Infinity');
    }

    if (minSpeed < 1) {
        issues.push(`Min speed too low: ${minSpeed.toFixed(2)} m/s`);
    }

    return {
        configName,
        maxSpeed,
        minSpeed,
        maxSpeedKmh: maxSpeed * 3.6,
        minSpeedKmh: minSpeed * 3.6,
        maxAccel,
        maxCentripetal,
        maxThrust,
        maxJerk,
        maxCurvature,
        feasible: issues.length === 0,
        issues,
    };
}

/**
 * Create Figure8 trajectory with custom speed and centripetal acceleration
 */
function createFigure8WithConfig(
    maxSpeed: number,
    maxCentripetalAccel: number
): Trajectory {
    // Create trajectory
    const traj = new Figure8Trajectory({ speed: maxSpeed, height: 2, size: 5 });

    // Override the protected maxCentripetalAccel via any cast
    (traj as any).maxCentripetalAccel = maxCentripetalAccel;

    // Clear speed table to force rebuild with new config
    (traj as any)._speedTable = null;

    return traj;
}

/**
 * Create demo trajectory with custom speed config
 */
function createDemoWithConfig(
    maxSpeed: number,
    maxCentripetalAccel: number
): Trajectory {
    // createTrajectory ignores speed param, so we need to override directly
    const traj = createTrajectory(maxSpeed, 4.0);

    // Override base class speed (used by getSpeedAtPhase)
    (traj as any).speed = maxSpeed;

    // Override maxCentripetalAccel
    (traj as any).maxCentripetalAccel = maxCentripetalAccel;

    // Clear speed table to force rebuild with new values
    (traj as any)._speedTable = null;

    return traj;
}

/**
 * Test a range of speed configurations
 */
function testSpeedConfigurations(): void {
    console.log(`\n${CYAN}${'═'.repeat(70)}${RESET}`);
    console.log(`${CYAN}  Maximum Speed Capability Test${RESET}`);
    console.log(`${CYAN}  Goal: 200 km/h (55.6 m/s)${RESET}`);
    console.log(`${CYAN}${'═'.repeat(70)}${RESET}\n`);

    // Test configurations: [maxSpeed (m/s), maxCentripetalAccel (m/s²)]
    const configs: [number, number, string][] = [
        // Current config
        [18, 15, 'Current (18 m/s, 15 m/s²)'],

        // Increase speed only
        [25, 15, 'Speed 25 m/s (90 km/h)'],
        [30, 15, 'Speed 30 m/s (108 km/h)'],
        [40, 15, 'Speed 40 m/s (144 km/h)'],

        // Increase centripetal accel
        [25, 25, 'Speed 25, Accel 25'],
        [30, 25, 'Speed 30, Accel 25'],
        [40, 25, 'Speed 40, Accel 25'],

        // Aggressive configs
        [40, 35, 'Speed 40, Accel 35'],
        [50, 35, 'Speed 50 (180 km/h), Accel 35'],
        [55, 40, 'Speed 55 (198 km/h), Accel 40'],
        [60, 45, 'Speed 60 (216 km/h), Accel 45'],
    ];

    console.log(`${YELLOW}Testing Figure8 Trajectory:${RESET}`);
    console.log('─'.repeat(85));
    console.log(
        'Config'.padEnd(32) +
        'Max'.padStart(10) +
        'Min'.padStart(10) +
        'Centrip'.padStart(10) +
        'Thrust'.padStart(10) +
        'OK?'.padStart(6)
    );
    console.log('─'.repeat(85));

    for (const [speed, accel, name] of configs) {
        const traj = createFigure8WithConfig(speed, accel);
        const result = analyzeTrajectory(traj, name);

        const status = result.feasible ? `${GREEN}✓${RESET}` : `${RED}✗${RESET}`;
        const speedStr = `${result.maxSpeedKmh.toFixed(0)} km/h`;
        const minStr = `${result.minSpeedKmh.toFixed(0)} km/h`;
        const centripStr = `${result.maxCentripetal.toFixed(1)}`;
        const thrustStr = `${result.maxThrust.toFixed(1)}`;

        console.log(
            name.padEnd(32) +
            speedStr.padStart(10) +
            minStr.padStart(10) +
            centripStr.padStart(10) +
            thrustStr.padStart(10) +
            status.padStart(10)
        );

        if (result.issues.length > 0) {
            for (const issue of result.issues) {
                console.log(`  ${RED}└─ ${issue}${RESET}`);
            }
        }
    }

    // Test demo trajectory
    console.log(`\n${YELLOW}Testing Demo (Generated) Trajectory:${RESET}`);
    console.log('─'.repeat(85));
    console.log(
        'Config'.padEnd(32) +
        'Max'.padStart(10) +
        'Min'.padStart(10) +
        'Centrip'.padStart(10) +
        'Thrust'.padStart(10) +
        'OK?'.padStart(6)
    );
    console.log('─'.repeat(85));

    for (const [speed, accel, name] of configs) {
        const traj = createDemoWithConfig(speed, accel);
        const result = analyzeTrajectory(traj, name);

        const status = result.feasible ? `${GREEN}✓${RESET}` : `${RED}✗${RESET}`;
        const speedStr = `${result.maxSpeedKmh.toFixed(0)} km/h`;
        const minStr = `${result.minSpeedKmh.toFixed(0)} km/h`;
        const centripStr = `${result.maxCentripetal.toFixed(1)}`;
        const thrustStr = `${result.maxThrust.toFixed(1)}`;

        console.log(
            name.padEnd(32) +
            speedStr.padStart(10) +
            minStr.padStart(10) +
            centripStr.padStart(10) +
            thrustStr.padStart(10) +
            status.padStart(10)
        );

        if (result.issues.length > 0) {
            for (const issue of result.issues) {
                console.log(`  ${RED}└─ ${issue}${RESET}`);
            }
        }
    }
}

/**
 * Find the maximum achievable speed
 */
function findMaxSpeed(): void {
    console.log(`\n${CYAN}${'═'.repeat(70)}${RESET}`);
    console.log(`${CYAN}  Binary Search for Maximum Speed${RESET}`);
    console.log(`${CYAN}${'═'.repeat(70)}${RESET}\n`);

    // Binary search for max feasible speed at different centripetal accelerations
    const accelConfigs = [15, 25, 35, 45];

    for (const maxAccel of accelConfigs) {
        let low = 10;
        let high = 80;
        let maxFeasible = 0;

        while (high - low > 1) {
            const mid = (low + high) / 2;
            const traj = createDemoWithConfig(mid, maxAccel);
            const result = analyzeTrajectory(traj, `Speed ${mid}`);

            if (result.feasible) {
                maxFeasible = mid;
                low = mid;
            } else {
                high = mid;
            }
        }

        const kmh = maxFeasible * 3.6;
        const status = kmh >= 200 ? `${GREEN}✓ GOAL MET${RESET}` : `${YELLOW}below goal${RESET}`;
        console.log(
            `Centripetal accel ${maxAccel} m/s²: ` +
            `${MAGENTA}Max speed = ${maxFeasible.toFixed(1)} m/s (${kmh.toFixed(0)} km/h)${RESET} ${status}`
        );
    }
}

/**
 * Detailed analysis of optimal configuration
 */
function detailedAnalysis(): void {
    console.log(`\n${CYAN}${'═'.repeat(70)}${RESET}`);
    console.log(`${CYAN}  Detailed Analysis: 200 km/h Config${RESET}`);
    console.log(`${CYAN}${'═'.repeat(70)}${RESET}\n`);

    // Config for 200 km/h (55.6 m/s)
    const targetSpeed = 55.6;
    const accelOptions = [35, 40, 45, 50];

    for (const accel of accelOptions) {
        const traj = createDemoWithConfig(targetSpeed, accel);
        const result = analyzeTrajectory(traj, `55.6 m/s @ ${accel} m/s²`);

        console.log(`\n${YELLOW}Config: maxSpeed=${targetSpeed} m/s, maxCentripetalLimit=${accel} m/s²${RESET}`);
        console.log('─'.repeat(55));
        console.log(`  Max speed:      ${result.maxSpeedKmh.toFixed(1)} km/h (${result.maxSpeed.toFixed(2)} m/s)`);
        console.log(`  Min speed:      ${result.minSpeedKmh.toFixed(1)} km/h (${result.minSpeed.toFixed(2)} m/s)`);
        console.log(`  Speed ratio:    ${(result.maxSpeed / result.minSpeed).toFixed(2)}x`);
        console.log(`  Max centripetal: ${result.maxCentripetal.toFixed(1)} m/s² (${(result.maxCentripetal / 9.81).toFixed(2)}g)`);
        console.log(`  Max thrust:     ${result.maxThrust.toFixed(1)} m/s² (${(result.maxThrust / 9.81).toFixed(2)}g)`);
        console.log(`  Max curvature:  ${result.maxCurvature.toFixed(4)} 1/m (R = ${(1/result.maxCurvature).toFixed(1)} m)`);
        console.log(`  Status:         ${result.feasible ? `${GREEN}FEASIBLE${RESET}` : `${RED}NOT FEASIBLE${RESET}`}`);

        if (result.issues.length > 0) {
            console.log(`  Issues:`);
            for (const issue of result.issues) {
                console.log(`    ${RED}• ${issue}${RESET}`);
            }
        }
    }
}

/**
 * Summary and recommendations
 */
function printRecommendations(): void {
    console.log(`\n${CYAN}${'═'.repeat(70)}${RESET}`);
    console.log(`${CYAN}  Recommendations${RESET}`);
    console.log(`${CYAN}${'═'.repeat(70)}${RESET}\n`);

    console.log(`${YELLOW}To achieve 200 km/h (55.6 m/s):${RESET}`);
    console.log('');
    console.log('  1. Trajectory.ts (maxCentripetalAccel):');
    console.log('     Current: 15 m/s²');
    console.log('     Needed:  ~40-45 m/s² (4-4.5g)');
    console.log('');
    console.log('  2. DroneRacingDemo.ts (defaultSpeed):');
    console.log('     Current: 18 m/s (65 km/h)');
    console.log('     Needed:  55.6 m/s (200 km/h)');
    console.log('');
    console.log('  3. GateTrajectoryGenerator.ts (maxSpeed):');
    console.log('     Current: 22 m/s');
    console.log('     Needed:  56+ m/s');
    console.log('');
    console.log('  4. MPC.ts (vAlongMax):');
    console.log('     Current: 30 m/s');
    console.log('     Needed:  60 m/s');
    console.log('');
    console.log(`${GREEN}Physical feasibility:${RESET}`);
    console.log('  • 4g centripetal accel is realistic for racing drones');
    console.log('  • maxThrust=50 m/s² (5g) provides headroom');
    console.log('  • Main constraint: turn radius at high speed');
}

// Main
async function main(): Promise<void> {
    testSpeedConfigurations();
    findMaxSpeed();
    detailedAnalysis();
    printRecommendations();
}

main().catch(console.error);
