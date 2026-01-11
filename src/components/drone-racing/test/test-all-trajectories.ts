/**
 * Trajectory Performance Tests
 *
 * Tests MPC tracking performance on all trajectory types:
 * 1. Circle (baseline)
 * 2. Figure-8 (lemniscate)
 * 3. Hairpin (tight turns)
 * 4. Snake (serpentine)
 * 5. RaceTrack (multi-segment)
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-all-trajectories.ts
 */

import { MPC } from '../control/MPC';
import { DroneDynamics } from '../core/DroneDynamics';
import { DroneState, Waypoint } from '../types';
import {
    Trajectory,
    CircleTrajectory,
    Figure8Trajectory,
    HairpinTrajectory,
    SnakeTrajectory,
    RaceTrackTrajectory,
} from '../trajectory';

const simDt = 0.02;  // 50 Hz simulation

// Position error
function posError(state: DroneState, ref: Waypoint): number {
    return Math.sqrt(
        (state.position.x - ref.position.x) ** 2 +
        (state.position.y - ref.position.y) ** 2 +
        (state.position.z - ref.position.z) ** 2
    );
}

// Format position
function fmtPos(p: { x: number; y: number; z: number }): string {
    return `(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})`;
}

interface TestResult {
    name: string;
    avgError: number;
    maxError: number;
    finalError: number;
    period: number;
    passed: boolean;
    worstPhase: number;  // Phase (0-1) where max error occurred
}

/**
 * Test a trajectory with MPC
 */
function testTrajectory(trajectory: Trajectory, verbose: boolean = false): TestResult {
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const name = trajectory.getName();
    const period = trajectory.getPeriod();

    // Initialize at trajectory start
    const initialState = trajectory.getInitialState();
    const startWp = trajectory.getWaypoint(0);

    dynamics.setPosition(initialState.position.x, initialState.position.y, initialState.position.z);
    dynamics.setVelocity(startWp.velocity.x, startWp.velocity.y, startWp.velocity.z);
    dynamics.setHeading(initialState.heading);

    const numSteps = Math.ceil(period / simDt);
    const errors: { phase: number; error: number }[] = [];
    let simTime = 0;

    if (verbose) {
        console.log(`\n--- ${name} ---`);
        console.log(`Period: ${period.toFixed(2)}s, Steps: ${numSteps}`);
        console.log('phase  | time   | pos                  | ref                  | err');
        console.log('-'.repeat(80));
    }

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = trajectory.getWaypoint(simTime);
        const err = posError(state, ref);
        const phase = simTime / period;

        errors.push({ phase, error: err });

        if (verbose && step % Math.floor(numSteps / 10) === 0) {
            console.log(
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${simTime.toFixed(2).padStart(5)}s | ` +
                `${fmtPos(state.position).padEnd(20)} | ` +
                `${fmtPos(ref.position).padEnd(20)} | ` +
                `${err.toFixed(3)}m`
            );
        }

        const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgError = errors.reduce((a, b) => a + b.error, 0) / errors.length;
    const maxEntry = errors.reduce((a, b) => b.error > a.error ? b : a, errors[0]);
    const maxError = maxEntry.error;
    const worstPhase = maxEntry.phase;
    const finalError = errors[errors.length - 1].error;

    // Pass if avg error < 1m for all trajectories
    const passed = avgError < 1.0;

    return { name, avgError, maxError, finalError, period, passed, worstPhase };
}

/**
 * Detailed debug for a specific trajectory
 */
function debugTrajectory(trajectory: Trajectory): void {
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const name = trajectory.getName();
    const period = trajectory.getPeriod();

    // Initialize
    const initialState = trajectory.getInitialState();
    const startWp = trajectory.getWaypoint(0);

    dynamics.setPosition(initialState.position.x, initialState.position.y, initialState.position.z);
    dynamics.setVelocity(startWp.velocity.x, startWp.velocity.y, startWp.velocity.z);
    dynamics.setHeading(initialState.heading);

    console.log(`\n${'='.repeat(80)}`);
    console.log(`DETAILED DEBUG: ${name}`);
    console.log(`${'='.repeat(80)}`);
    console.log(`Period: ${period.toFixed(2)}s`);
    console.log(`Initial pos: ${fmtPos(initialState.position)}, heading: ${(initialState.heading * 180 / Math.PI).toFixed(1)}°`);

    // Sample waypoints to understand the trajectory
    console.log(`\nTrajectory shape (sample waypoints):`);
    console.log('phase | pos                  | vel                  | heading | headRate');
    console.log('-'.repeat(90));

    for (let i = 0; i <= 10; i++) {
        const phase = i / 10;
        const t = phase * period;
        const wp = trajectory.getWaypoint(t);
        const speed = Math.sqrt(wp.velocity.x ** 2 + wp.velocity.y ** 2 + wp.velocity.z ** 2);
        console.log(
            `${(phase * 100).toFixed(0).padStart(4)}% | ` +
            `${fmtPos(wp.position).padEnd(20)} | ` +
            `v=${speed.toFixed(1).padStart(4)} ${fmtPos(wp.velocity).padEnd(14)} | ` +
            `${(wp.heading * 180 / Math.PI).toFixed(0).padStart(6)}° | ` +
            `${(wp.headingRate * 180 / Math.PI).toFixed(1).padStart(6)}°/s`
        );
    }

    // Now run simulation with detailed output
    console.log(`\nSimulation (every 10%):`);
    console.log('phase | pos                  | ref                  | err    | thrust | rates');
    console.log('-'.repeat(100));

    const numSteps = Math.ceil(period / simDt);
    let simTime = 0;
    let maxErr = 0;
    let maxErrPhase = 0;

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = trajectory.getWaypoint(simTime);
        const err = posError(state, ref);
        const phase = simTime / period;

        if (err > maxErr) {
            maxErr = err;
            maxErrPhase = phase;
        }

        const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);

        if (step % Math.floor(numSteps / 10) === 0) {
            console.log(
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${fmtPos(state.position).padEnd(20)} | ` +
                `${fmtPos(ref.position).padEnd(20)} | ` +
                `${err.toFixed(3).padStart(5)}m | ` +
                `${cmd.thrust.toFixed(1).padStart(5)} | ` +
                `(${cmd.rollRate.toFixed(1)}, ${cmd.pitchRate.toFixed(1)}, ${cmd.yawRate.toFixed(1)})`
            );
        }

        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    console.log(`\nMax error: ${maxErr.toFixed(3)}m at ${(maxErrPhase * 100).toFixed(0)}% of trajectory`);
}

// =====================================================
// MAIN TEST SUITE
// =====================================================

console.log('='.repeat(80));
console.log('TRAJECTORY PERFORMANCE TESTS');
console.log('='.repeat(80));
console.log(`Simulation dt: ${simDt}s (${1/simDt} Hz)`);

// Create all trajectories using default parameters (which should be tuned for good tracking)
const height = 4.0;

const trajectories: Trajectory[] = [
    new CircleTrajectory({ height, radius: 15.0, speed: 12.0 }),
    new Figure8Trajectory({ height }),  // Use defaults (speed: 8, size: 20)
    new HairpinTrajectory({ height, turnRadius: 8.0, straightLength: 25.0, speed: 10.0 }),
    new SnakeTrajectory({ height }),    // Use defaults (speed: 10, with smooth turnarounds)
    new RaceTrackTrajectory({ height, gateSpacing: 20.0, turnRadius: 10.0, speed: 10.0 }),
];

// Run tests
const results: TestResult[] = [];

for (const traj of trajectories) {
    const result = testTrajectory(traj, false);
    results.push(result);
}

// Print summary
console.log('\n' + '='.repeat(80));
console.log('SUMMARY');
console.log('='.repeat(80));
console.log('Trajectory      | Period  | Avg Err | Max Err | Worst@  | Status');
console.log('-'.repeat(80));

for (const r of results) {
    console.log(
        `${r.name.padEnd(15)} | ` +
        `${r.period.toFixed(1).padStart(5)}s | ` +
        `${r.avgError.toFixed(3).padStart(6)}m | ` +
        `${r.maxError.toFixed(3).padStart(6)}m | ` +
        `${(r.worstPhase * 100).toFixed(0).padStart(4)}% | ` +
        `${r.passed ? 'PASS' : 'FAIL'}`
    );
}

const allPassed = results.every(r => r.passed);
console.log('\n' + '-'.repeat(80));
console.log(`Overall: ${allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED'}`);

// Debug failing trajectories
const failing = results.filter(r => !r.passed || r.avgError > 0.5);
if (failing.length > 0) {
    console.log('\n' + '='.repeat(80));
    console.log('DEBUGGING PROBLEMATIC TRAJECTORIES');
    console.log('='.repeat(80));

    for (const f of failing) {
        const traj = trajectories.find(t => t.getName() === f.name)!;
        debugTrajectory(traj);
    }
}

// Also debug Figure-8 and Snake specifically (user mentioned issues)
const figure8 = trajectories.find(t => t.getName() === 'Figure-8')!;
const snake = trajectories.find(t => t.getName() === 'Snake')!;

console.log('\n' + '='.repeat(80));
console.log('REQUESTED DEBUG: Figure-8 and Snake');
console.log('='.repeat(80));

debugTrajectory(figure8);
debugTrajectory(snake);

console.log('\n' + '='.repeat(80));
console.log('TESTS COMPLETE');
console.log('='.repeat(80));
