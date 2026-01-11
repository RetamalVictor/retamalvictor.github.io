/**
 * Full MPC Test
 *
 * Tests the complete MPC controller:
 * 1. Hover stabilization
 * 2. Position tracking
 * 3. Circular trajectory following
 * 4. Performance metrics
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-mpc-full.ts
 */

import { MPC, DEFAULT_MPC_CONFIG } from '../control/MPC';
import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { DroneState, Waypoint, ControlCommand } from '../types';

// Helper to create waypoint
function createWaypoint(
    px: number, py: number, pz: number,
    vx: number, vy: number, vz: number,
    ax: number, ay: number, az: number,
    heading: number, headingRate: number,
    time: number
): Waypoint {
    return {
        position: { x: px, y: py, z: pz },
        velocity: { x: vx, y: vy, z: vz },
        acceleration: { x: ax, y: ay, z: az },
        jerk: { x: 0, y: 0, z: 0 },
        heading,
        headingRate,
        time,
    };
}

// Format state
function fmtPos(s: DroneState): string {
    return `(${s.position.x.toFixed(3)}, ${s.position.y.toFixed(3)}, ${s.position.z.toFixed(3)})`;
}

function fmtVel(s: DroneState): string {
    return `(${s.velocity.x.toFixed(3)}, ${s.velocity.y.toFixed(3)}, ${s.velocity.z.toFixed(3)})`;
}

function fmtCmd(c: ControlCommand): string {
    return `T=${c.thrust.toFixed(2)} rates=(${c.rollRate.toFixed(3)}, ${c.pitchRate.toFixed(3)}, ${c.yawRate.toFixed(3)})`;
}

// Compute position error
function posError(s: DroneState, ref: Waypoint): number {
    return Math.sqrt(
        (s.position.x - ref.position.x) ** 2 +
        (s.position.y - ref.position.y) ** 2 +
        (s.position.z - ref.position.z) ** 2
    );
}

console.log('='.repeat(70));
console.log('FULL MPC TEST');
console.log('='.repeat(70));

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const simDt = 0.02;  // 50 Hz simulation

console.log(`\nMPC Config: ${JSON.stringify(DEFAULT_MPC_CONFIG, null, 2)}`);
console.log(`\nDynamics: ${JSON.stringify(DEFAULT_DYNAMICS_PARAMS)}`);

// =====================================================
// TEST 1: Hover Stabilization
// =====================================================
console.log('\n' + '='.repeat(70));
console.log('TEST 1: Hover Stabilization');
console.log('='.repeat(70));
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    // Start slightly perturbed from target
    dynamics.setPosition(0.5, 2.5, -0.3);

    // Target: hover at origin, height 2
    const getReference = (t: number): Waypoint => createWaypoint(
        0, 2, 0,    // position
        0, 0, 0,    // velocity
        0, 0, 0,    // acceleration
        0, 0, t     // heading, headingRate, time
    );

    console.log(`\nInitial state: pos=${fmtPos(dynamics.getState())}`);
    console.log(`Target: pos=(0, 2, 0)`);
    console.log(`\nSimulating 2 seconds...`);

    const errors: number[] = [];
    let simTime = 0;

    for (let step = 0; step < 100; step++) {
        const state = dynamics.getState();
        const ref = getReference(simTime);
        const err = posError(state, ref);
        errors.push(err);

        // Log every 0.5s
        if (step % 25 === 0) {
            const cmd = mpc.computeControl(state, getReference, simTime);
            console.log(`t=${simTime.toFixed(2)}s: pos=${fmtPos(state)} err=${err.toFixed(4)}m cmd: ${fmtCmd(cmd)}`);
        }

        const cmd = mpc.computeControl(state, getReference, simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const finalErr = errors[errors.length - 1];
    const maxErr = Math.max(...errors);
    console.log(`\nResults:`);
    console.log(`  Final error: ${finalErr.toFixed(4)} m`);
    console.log(`  Max error: ${maxErr.toFixed(4)} m`);
    console.log(`  PASS: ${finalErr < 0.1 ? 'YES' : 'NO'}`);
}

// =====================================================
// TEST 2: Step Response
// =====================================================
console.log('\n' + '='.repeat(70));
console.log('TEST 2: Step Response (Position Change)');
console.log('='.repeat(70));
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    dynamics.setPosition(0, 2, 0);

    // Step change: move to (2, 2, 2)
    const getReference = (t: number): Waypoint => createWaypoint(
        2, 2, 2,    // position
        0, 0, 0,    // velocity (want to stop there)
        0, 0, 0,    // acceleration
        0, 0, t
    );

    console.log(`\nInitial: pos=(0, 2, 0)`);
    console.log(`Target: pos=(2, 2, 2)`);
    console.log(`\nSimulating 3 seconds...`);

    let simTime = 0;
    const trajectory: { t: number; x: number; y: number; z: number; err: number }[] = [];

    for (let step = 0; step < 150; step++) {
        const state = dynamics.getState();
        const ref = getReference(simTime);
        const err = posError(state, ref);

        trajectory.push({
            t: simTime,
            x: state.position.x,
            y: state.position.y,
            z: state.position.z,
            err,
        });

        if (step % 30 === 0) {
            const cmd = mpc.computeControl(state, getReference, simTime);
            console.log(`t=${simTime.toFixed(2)}s: pos=${fmtPos(state)} err=${err.toFixed(3)}m`);
        }

        const cmd = mpc.computeControl(state, getReference, simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const finalErr = trajectory[trajectory.length - 1].err;
    console.log(`\nFinal error: ${finalErr.toFixed(4)} m`);
    console.log(`PASS: ${finalErr < 0.2 ? 'YES' : 'NO'}`);
}

// =====================================================
// TEST 3: Circular Trajectory
// =====================================================
console.log('\n' + '='.repeat(70));
console.log('TEST 3: Circular Trajectory Following');
console.log('='.repeat(70));
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const radius = 5;
    const speed = 3;  // Match reference: aggressive trajectory
    const height = 2;
    const omega = speed / radius;
    const centripetalAccel = speed * speed / radius;

    // Circular trajectory
    // Use continuous heading that doesn't wrap (heading = -angle for CCW motion)
    const getReference = (t: number): Waypoint => {
        const angle = omega * t;
        // Heading is tangent to circle: for CCW motion (positive angle), heading decreases
        // Use -angle to avoid atan2 discontinuities
        const heading = -angle;
        return createWaypoint(
            radius * Math.cos(angle),  // x
            height,                     // y
            radius * Math.sin(angle),   // z
            -speed * Math.sin(angle),   // vx
            0,                          // vy
            speed * Math.cos(angle),    // vz
            -centripetalAccel * Math.cos(angle),  // ax
            0,                                     // ay
            -centripetalAccel * Math.sin(angle),  // az
            heading,                               // heading (continuous, no atan2 discontinuity)
            -omega,                                // headingRate (negative for CW rotation in XZ plane)
            t
        );
    };

    // Start on the circle with correct velocity
    const start = getReference(0);
    dynamics.setPosition(start.position.x, start.position.y, start.position.z);
    dynamics.setVelocity(start.velocity.x, start.velocity.y, start.velocity.z);
    dynamics.setHeading(start.heading);

    console.log(`\nCircle: radius=${radius}m, speed=${speed}m/s, height=${height}m`);
    console.log(`Angular velocity: ${omega.toFixed(3)} rad/s`);
    console.log(`Centripetal accel: ${centripetalAccel.toFixed(3)} m/s²`);
    console.log(`Period: ${(2 * Math.PI / omega).toFixed(2)}s`);
    console.log(`\nSimulating 1 lap...`);

    let simTime = 0;
    const period = 2 * Math.PI / omega;
    const numSteps = Math.ceil(period / simDt);
    const errors: number[] = [];

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = getReference(simTime);
        const err = posError(state, ref);
        errors.push(err);

        if (step % Math.floor(numSteps / 8) === 0) {
            console.log(`t=${simTime.toFixed(2)}s: pos=${fmtPos(state)} ref=(${ref.position.x.toFixed(2)}, ${ref.position.y.toFixed(2)}, ${ref.position.z.toFixed(2)}) err=${err.toFixed(3)}m`);
        }

        const cmd = mpc.computeControl(state, getReference, simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgErr = errors.reduce((a, b) => a + b, 0) / errors.length;
    const maxErr = Math.max(...errors);
    const finalErr = errors[errors.length - 1];

    console.log(`\nResults after 1 lap:`);
    console.log(`  Average error: ${avgErr.toFixed(4)} m`);
    console.log(`  Max error: ${maxErr.toFixed(4)} m`);
    console.log(`  Final error: ${finalErr.toFixed(4)} m`);
    console.log(`  PASS: ${avgErr < 0.5 ? 'YES' : 'NO'}`);
}

// =====================================================
// TEST 4: Performance Analysis
// =====================================================
console.log('\n' + '='.repeat(70));
console.log('TEST 4: MPC Computation Time');
console.log('='.repeat(70));
{
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const getReference = (t: number): Waypoint => createWaypoint(
        Math.sin(t) * 3, 2, Math.cos(t) * 3,
        Math.cos(t) * 3, 0, -Math.sin(t) * 3,
        -Math.sin(t) * 3, 0, -Math.cos(t) * 3,
        t, 1, t
    );

    const times: number[] = [];
    let simTime = 0;

    // Warmup
    for (let i = 0; i < 10; i++) {
        mpc.computeControl(dynamics.getState(), getReference, simTime);
        dynamics.step({ thrust: g, rollRate: 0, pitchRate: 0, yawRate: 0, timestamp: 0 }, simDt);
        simTime += simDt;
    }

    // Measure
    for (let i = 0; i < 50; i++) {
        const start = performance.now();
        const cmd = mpc.computeControl(dynamics.getState(), getReference, simTime);
        const elapsed = performance.now() - start;
        times.push(elapsed);

        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    const maxTime = Math.max(...times);
    const minTime = Math.min(...times);

    console.log(`\nMPC computation time (50 iterations):`);
    console.log(`  Average: ${avgTime.toFixed(2)} ms`);
    console.log(`  Min: ${minTime.toFixed(2)} ms`);
    console.log(`  Max: ${maxTime.toFixed(2)} ms`);
    console.log(`  Target: ${(simDt * 1000).toFixed(1)} ms (real-time at 50Hz)`);
    console.log(`  PASS: ${avgTime < simDt * 1000 ? 'YES (real-time capable)' : 'NO (too slow)'}`);
}

console.log('\n' + '='.repeat(70));
console.log('ALL MPC TESTS COMPLETE');
console.log('='.repeat(70));
