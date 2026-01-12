/**
 * Trajectory Generator Tests
 *
 * Tests the gate-based trajectory generation:
 * 1. SmoothLineSegment - jerk-limited straight lines
 * 2. ArcSegment - circular arcs for turns
 * 3. GateTrajectoryGenerator - automatic trajectory from gates
 * 4. MPC tracking on generated trajectories
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-trajectory-generator.ts
 */

import { MPC } from '../control/MPC';
import { DroneDynamics } from '../core/DroneDynamics';
import { DroneState, Waypoint, GateWaypoint } from '../types';
import { SmoothLineSegment } from '../trajectory/segments/SmoothLineSegment';
import { ArcSegment } from '../trajectory/segments/ArcSegment';
import { GateTrajectoryGenerator } from '../trajectory/GateTrajectoryGenerator';

const simDt = 0.02;  // 50 Hz simulation

// =====================================================
// HELPER FUNCTIONS
// =====================================================

function posError(state: DroneState, ref: Waypoint): number {
    return Math.sqrt(
        (state.position.x - ref.position.x) ** 2 +
        (state.position.y - ref.position.y) ** 2 +
        (state.position.z - ref.position.z) ** 2
    );
}

function fmtPos(p: { x: number; y: number; z: number }): string {
    return `(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})`;
}

function fmtVec(v: { x: number; y: number; z: number }): string {
    return `(${v.x.toFixed(3)}, ${v.y.toFixed(3)}, ${v.z.toFixed(3)})`;
}

function vecMag(v: { x: number; y: number; z: number }): number {
    return Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// =====================================================
// TEST 1: SmoothLineSegment
// =====================================================

function testSmoothLineSegment(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 1: SmoothLineSegment');
    console.log('='.repeat(80));

    const p0 = { x: 0, y: 5, z: 0 };
    const p1 = { x: 20, y: 5, z: 0 };
    const v0 = 5.0;   // 5 m/s start
    const v1 = 15.0;  // 15 m/s end

    const segment = new SmoothLineSegment(p0, p1, v0, v1);

    console.log(`\nSegment: ${fmtPos(p0)} -> ${fmtPos(p1)}`);
    console.log(`Velocity: ${v0} m/s -> ${v1} m/s`);
    console.log(`Length: ${segment.getLength().toFixed(2)} m`);
    console.log(`Duration: ${segment.getDuration().toFixed(3)} s`);

    // Expected duration: T = 2*L / (v0 + v1) = 2*20 / (5+15) = 2.0s
    const expectedDuration = 2 * 20 / (v0 + v1);
    console.log(`Expected duration: ${expectedDuration.toFixed(3)} s`);

    let passed = true;

    // Check duration
    if (Math.abs(segment.getDuration() - expectedDuration) > 0.001) {
        console.log(`FAIL: Duration mismatch`);
        passed = false;
    }

    // Check start and end positions
    const startPos = segment.getPosition(0);
    const endPos = segment.getPosition(1);

    console.log(`\nStart position: ${fmtPos(startPos)} (expected ${fmtPos(p0)})`);
    console.log(`End position: ${fmtPos(endPos)} (expected ${fmtPos(p1)})`);

    if (Math.abs(startPos.x - p0.x) > 0.01 || Math.abs(startPos.z - p0.z) > 0.01) {
        console.log(`FAIL: Start position mismatch`);
        passed = false;
    }

    if (Math.abs(endPos.x - p1.x) > 0.01 || Math.abs(endPos.z - p1.z) > 0.01) {
        console.log(`FAIL: End position mismatch`);
        passed = false;
    }

    // Check start and end velocities
    const startVel = segment.getVelocity(0);
    const endVel = segment.getVelocity(1);
    const startSpeed = vecMag(startVel);
    const endSpeed = vecMag(endVel);

    console.log(`\nStart speed: ${startSpeed.toFixed(2)} m/s (expected ${v0})`);
    console.log(`End speed: ${endSpeed.toFixed(2)} m/s (expected ${v1})`);

    if (Math.abs(startSpeed - v0) > 0.1) {
        console.log(`FAIL: Start speed mismatch`);
        passed = false;
    }

    if (Math.abs(endSpeed - v1) > 0.1) {
        console.log(`FAIL: End speed mismatch`);
        passed = false;
    }

    // Sample along segment
    console.log(`\nSampling along segment:`);
    console.log('t     | pos                  | vel                  | speed  | accel');
    console.log('-'.repeat(80));

    for (let i = 0; i <= 10; i++) {
        const t = i / 10;
        const pos = segment.getPosition(t);
        const vel = segment.getVelocity(t);
        const acc = segment.getAcceleration(t);
        const speed = vecMag(vel);
        const accMag = vecMag(acc);

        console.log(
            `${t.toFixed(1).padStart(4)} | ` +
            `${fmtPos(pos).padEnd(20)} | ` +
            `${fmtVec(vel).padEnd(20)} | ` +
            `${speed.toFixed(2).padStart(5)} | ` +
            `${accMag.toFixed(2)}`
        );
    }

    // Verify monotonic speed increase
    let prevSpeed = 0;
    let monotonic = true;
    for (let i = 0; i <= 20; i++) {
        const t = i / 20;
        const vel = segment.getVelocity(t);
        const speed = vecMag(vel);
        if (speed < prevSpeed - 0.01) {
            monotonic = false;
            console.log(`FAIL: Speed not monotonic at t=${t.toFixed(2)}: ${speed.toFixed(2)} < ${prevSpeed.toFixed(2)}`);
        }
        prevSpeed = speed;
    }

    if (!monotonic) {
        passed = false;
    }

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'}`);
    return passed;
}

// =====================================================
// TEST 2: ArcSegment
// =====================================================

function testArcSegment(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 2: ArcSegment');
    console.log('='.repeat(80));

    // Quarter circle arc
    const center = { x: 0, y: 5, z: 0 };
    const radius = 10;
    const startAngle = 0;        // Start at (10, 5, 0)
    const endAngle = Math.PI / 2;  // End at (0, 5, 10)
    const v0 = 10.0;
    const v1 = 10.0;  // Constant speed

    const arc = new ArcSegment({
        center,
        radius,
        startAngle,
        endAngle,
        v0,
        v1,
        height: 5,
    });

    // Arc length = r * angle = 10 * π/2 ≈ 15.7m
    const expectedLength = radius * Math.abs(endAngle - startAngle);
    const expectedDuration = 2 * expectedLength / (v0 + v1);

    console.log(`\nArc: center=${fmtPos(center)}, r=${radius}m`);
    console.log(`Angles: ${(startAngle * 180 / Math.PI).toFixed(1)}° -> ${(endAngle * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Velocity: ${v0} m/s -> ${v1} m/s`);
    console.log(`Length: ${arc.getLength().toFixed(2)} m (expected ${expectedLength.toFixed(2)} m)`);
    console.log(`Duration: ${arc.getDuration().toFixed(3)} s (expected ${expectedDuration.toFixed(3)} s)`);

    let passed = true;

    // Check length
    if (Math.abs(arc.getLength() - expectedLength) > 0.1) {
        console.log(`FAIL: Length mismatch`);
        passed = false;
    }

    // Check start and end positions
    const startPos = arc.getPosition(0);
    const endPos = arc.getPosition(1);

    const expectedStart = { x: center.x + radius, y: 5, z: center.z };
    const expectedEnd = { x: center.x, y: 5, z: center.z + radius };

    console.log(`\nStart position: ${fmtPos(startPos)} (expected ${fmtPos(expectedStart)})`);
    console.log(`End position: ${fmtPos(endPos)} (expected ${fmtPos(expectedEnd)})`);

    if (Math.abs(startPos.x - expectedStart.x) > 0.1 || Math.abs(startPos.z - expectedStart.z) > 0.1) {
        console.log(`FAIL: Start position mismatch`);
        passed = false;
    }

    if (Math.abs(endPos.x - expectedEnd.x) > 0.1 || Math.abs(endPos.z - expectedEnd.z) > 0.1) {
        console.log(`FAIL: End position mismatch`);
        passed = false;
    }

    // Sample along arc
    console.log(`\nSampling along arc:`);
    console.log('t     | pos                  | vel                  | speed  | dist from center');
    console.log('-'.repeat(90));

    for (let i = 0; i <= 10; i++) {
        const t = i / 10;
        const pos = arc.getPosition(t);
        const vel = arc.getVelocity(t);
        const speed = vecMag(vel);

        // Distance from center should be constant (radius)
        const distFromCenter = Math.sqrt(
            (pos.x - center.x) ** 2 + (pos.z - center.z) ** 2
        );

        console.log(
            `${t.toFixed(1).padStart(4)} | ` +
            `${fmtPos(pos).padEnd(20)} | ` +
            `${fmtVec(vel).padEnd(20)} | ` +
            `${speed.toFixed(2).padStart(5)} | ` +
            `${distFromCenter.toFixed(2)} m`
        );

        // Check distance from center is approximately radius
        if (Math.abs(distFromCenter - radius) > 0.5) {
            console.log(`  WARNING: Distance from center deviates from radius`);
        }
    }

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'}`);
    return passed;
}

// =====================================================
// TEST 3: GateTrajectoryGenerator - Triangle Course
// =====================================================

function testTriangleTrajectory(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 3: GateTrajectoryGenerator - Triangle Course');
    console.log('='.repeat(80));

    const generator = new GateTrajectoryGenerator();

    // Create a simple triangular course
    const gates: GateWaypoint[] = [
        {
            position: { x: 0, y: 5, z: 0 },
            entranceDir: { x: 0, y: 0, z: 1 },  // Enter heading +Z
        },
        {
            position: { x: 20, y: 5, z: 25 },
            entranceDir: { x: 0.707, y: 0, z: 0.707 },  // Enter at 45°
        },
        {
            position: { x: -20, y: 5, z: 25 },
            entranceDir: { x: -0.707, y: 0, z: 0.707 },  // Enter at -45°
        },
    ];

    console.log('\nGates:');
    for (let i = 0; i < gates.length; i++) {
        const g = gates[i];
        console.log(`  Gate ${i}: pos=${fmtPos(g.position)}, dir=${fmtVec(g.entranceDir)}`);
    }

    const trajectory = generator.generate(gates);

    console.log(`\nGenerated trajectory:`);
    console.log(`  Period: ${trajectory.getPeriod().toFixed(2)} s`);
    console.log(`  Segments: ${trajectory.getSegments().length}`);

    // Sample trajectory
    console.log(`\nSampling trajectory:`);
    console.log('t     | phase | pos                  | vel                  | speed  | heading');
    console.log('-'.repeat(95));

    const period = trajectory.getPeriod();
    let passed = true;

    for (let i = 0; i <= 20; i++) {
        const t = (i / 20) * period;
        const phase = i / 20;
        const wp = trajectory.getWaypoint(t);
        const speed = vecMag(wp.velocity);
        const headingDeg = (wp.heading * 180 / Math.PI);

        if (i % 2 === 0) {
            console.log(
                `${t.toFixed(2).padStart(5)} | ` +
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${fmtPos(wp.position).padEnd(20)} | ` +
                `${fmtVec(wp.velocity).padEnd(20)} | ` +
                `${speed.toFixed(2).padStart(5)} | ` +
                `${headingDeg.toFixed(0)}°`
            );
        }

        // Check speed is reasonable (> 0 and < 30 m/s)
        if (speed < 0.1 || speed > 30) {
            console.log(`  WARNING: Unusual speed at t=${t.toFixed(2)}: ${speed.toFixed(2)} m/s`);
            passed = false;
        }
    }

    // Check that trajectory is closed (end ≈ start)
    const startWp = trajectory.getWaypoint(0);
    const endWp = trajectory.getWaypoint(period - 0.01);
    const closureDist = Math.sqrt(
        (endWp.position.x - startWp.position.x) ** 2 +
        (endWp.position.y - startWp.position.y) ** 2 +
        (endWp.position.z - startWp.position.z) ** 2
    );

    console.log(`\nClosure check:`);
    console.log(`  Start: ${fmtPos(startWp.position)}`);
    console.log(`  End:   ${fmtPos(endWp.position)}`);
    console.log(`  Distance: ${closureDist.toFixed(2)} m`);

    if (closureDist > 5.0) {
        console.log(`  WARNING: Trajectory may not be properly closed`);
    }

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'}`);
    return passed;
}

// =====================================================
// TEST 4: MPC Tracking on Generated Trajectory
// =====================================================

function testMPCTracking(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 4: MPC Tracking on Generated Trajectory');
    console.log('='.repeat(80));

    const generator = new GateTrajectoryGenerator();

    // Simple triangle course
    const gates: GateWaypoint[] = [
        {
            position: { x: 0, y: 5, z: 0 },
            entranceDir: { x: 0, y: 0, z: 1 },
        },
        {
            position: { x: 25, y: 5, z: 30 },
            entranceDir: { x: 0.866, y: 0, z: 0.5 },
        },
        {
            position: { x: -25, y: 5, z: 30 },
            entranceDir: { x: -0.866, y: 0, z: 0.5 },
        },
    ];

    const trajectory = generator.generate(gates);
    const period = trajectory.getPeriod();

    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    // Initialize at trajectory start
    const initialState = trajectory.getInitialState();
    const startWp = trajectory.getWaypoint(0);

    dynamics.setPosition(initialState.position.x, initialState.position.y, initialState.position.z);
    dynamics.setVelocity(startWp.velocity.x, startWp.velocity.y, startWp.velocity.z);
    dynamics.setHeading(initialState.heading);

    console.log(`\nTrajectory period: ${period.toFixed(2)} s`);
    console.log(`Initial position: ${fmtPos(initialState.position)}`);
    console.log(`Initial velocity: ${fmtVec(startWp.velocity)}`);

    const numSteps = Math.ceil(period / simDt);
    const errors: number[] = [];
    let simTime = 0;

    console.log(`\nRunning simulation (${numSteps} steps)...`);
    console.log('phase | pos                  | ref                  | error  | thrust');
    console.log('-'.repeat(85));

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = trajectory.getWaypoint(simTime);
        const err = posError(state, ref);
        const phase = simTime / period;

        errors.push(err);

        if (step % Math.floor(numSteps / 10) === 0) {
            const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);
            console.log(
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${fmtPos(state.position).padEnd(20)} | ` +
                `${fmtPos(ref.position).padEnd(20)} | ` +
                `${err.toFixed(3).padStart(5)}m | ` +
                `${cmd.thrust.toFixed(1)}`
            );
        }

        const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
    const maxError = Math.max(...errors);

    console.log(`\nResults:`);
    console.log(`  Average error: ${avgError.toFixed(3)} m`);
    console.log(`  Max error: ${maxError.toFixed(3)} m`);

    // Pass criteria: avg error < 1.5m (relaxed due to sharp turns between gates)
    const passed = avgError < 1.5;

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'} (threshold: avg < 1.5m)`);
    return passed;
}

// =====================================================
// TEST 5: Square Course with Sharp Turns
// =====================================================

function testSquareTrajectory(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 5: Square Course with Sharp Turns');
    console.log('='.repeat(80));

    const generator = new GateTrajectoryGenerator();

    // Square course - 90° turns
    const size = 20;
    const gates: GateWaypoint[] = [
        {
            position: { x: 0, y: 5, z: 0 },
            entranceDir: { x: 0, y: 0, z: 1 },  // Going +Z
        },
        {
            position: { x: 0, y: 5, z: size },
            entranceDir: { x: 1, y: 0, z: 0 },  // Turn to +X
        },
        {
            position: { x: size, y: 5, z: size },
            entranceDir: { x: 0, y: 0, z: -1 },  // Turn to -Z
        },
        {
            position: { x: size, y: 5, z: 0 },
            entranceDir: { x: -1, y: 0, z: 0 },  // Turn to -X
        },
    ];

    console.log('\nGates (square course):');
    for (let i = 0; i < gates.length; i++) {
        const g = gates[i];
        console.log(`  Gate ${i}: pos=${fmtPos(g.position)}, dir=${fmtVec(g.entranceDir)}`);
    }

    const trajectory = generator.generate(gates);
    const period = trajectory.getPeriod();

    console.log(`\nGenerated trajectory:`);
    console.log(`  Period: ${period.toFixed(2)} s`);
    console.log(`  Segments: ${trajectory.getSegments().length}`);

    // MPC tracking test
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const initialState = trajectory.getInitialState();
    const startWp = trajectory.getWaypoint(0);

    dynamics.setPosition(initialState.position.x, initialState.position.y, initialState.position.z);
    dynamics.setVelocity(startWp.velocity.x, startWp.velocity.y, startWp.velocity.z);
    dynamics.setHeading(initialState.heading);

    const numSteps = Math.ceil(period / simDt);
    const errors: number[] = [];
    let simTime = 0;

    console.log(`\nRunning simulation...`);

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = trajectory.getWaypoint(simTime);
        const err = posError(state, ref);

        errors.push(err);

        const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
    const maxError = Math.max(...errors);

    console.log(`\nResults:`);
    console.log(`  Average error: ${avgError.toFixed(3)} m`);
    console.log(`  Max error: ${maxError.toFixed(3)} m`);

    const passed = avgError < 1.5;  // Allow slightly higher for sharp turns

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'} (threshold: avg < 1.5m)`);
    return passed;
}

// =====================================================
// TEST 6: Vertical Dive Gates (Stacked)
// =====================================================

function testVerticalDiveGates(): boolean {
    console.log('\n' + '='.repeat(80));
    console.log('TEST 6: Vertical Dive Gates (Immediately Stacked)');
    console.log('='.repeat(80));

    const generator = new GateTrajectoryGenerator();

    // Two gates stacked vertically - same X/Z, different Y
    const highHeight = 12;
    const lowHeight = 4;  // Higher to prevent underground trajectory

    const gates: GateWaypoint[] = [
        // High gate - enter going forward (+Z)
        {
            position: { x: 0, y: highHeight, z: 0 },
            entranceDir: { x: 0, y: 0, z: 1 },
        },
        // Low gate - SAME X/Z, directly below
        // Enter diving at 60° angle (down + forward to avoid underground exit)
        {
            position: { x: 0, y: lowHeight, z: 0 },
            entranceDir: { x: 0, y: -0.866, z: 0.5 },  // 60° dive angle
        },
        // Recovery gate - forward motion to return
        {
            position: { x: 0, y: 8, z: -30 },
            entranceDir: { x: 0, y: 0, z: -1 },
        },
    ];

    console.log('\nGates (vertically stacked):');
    for (let i = 0; i < gates.length; i++) {
        const g = gates[i];
        console.log(`  Gate ${i}: pos=${fmtPos(g.position)}, dir=${fmtVec(g.entranceDir)}`);
    }

    const trajectory = generator.generate(gates);
    const period = trajectory.getPeriod();

    console.log(`\nGenerated trajectory:`);
    console.log(`  Period: ${period.toFixed(2)} s`);
    console.log(`  Segments: ${trajectory.getSegments().length}`);

    // Sample trajectory - focus on vertical motion
    console.log(`\nSampling trajectory (focus on Y motion):`);
    console.log('t     | phase | pos                  | vel                  | vy     ');
    console.log('-'.repeat(85));

    let hasVerticalMotion = false;
    let minY = Infinity;
    let maxY = -Infinity;

    for (let i = 0; i <= 20; i++) {
        const t = (i / 20) * period;
        const phase = i / 20;
        const wp = trajectory.getWaypoint(t);

        minY = Math.min(minY, wp.position.y);
        maxY = Math.max(maxY, wp.position.y);

        if (Math.abs(wp.velocity.y) > 2) {
            hasVerticalMotion = true;
        }

        if (i % 2 === 0) {
            console.log(
                `${t.toFixed(2).padStart(5)} | ` +
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${fmtPos(wp.position).padEnd(20)} | ` +
                `${fmtVec(wp.velocity).padEnd(20)} | ` +
                `${wp.velocity.y.toFixed(2).padStart(6)}`
            );
        }
    }

    console.log(`\nY range: ${minY.toFixed(2)} to ${maxY.toFixed(2)} (span: ${(maxY - minY).toFixed(2)}m)`);
    console.log(`Has significant vertical motion (vy > 2 m/s): ${hasVerticalMotion}`);

    // MPC tracking test
    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const initialState = trajectory.getInitialState();
    const startWp = trajectory.getWaypoint(0);

    dynamics.setPosition(initialState.position.x, initialState.position.y, initialState.position.z);
    dynamics.setVelocity(startWp.velocity.x, startWp.velocity.y, startWp.velocity.z);
    dynamics.setHeading(initialState.heading);

    const numSteps = Math.ceil(period / simDt);
    const errors: number[] = [];
    let simTime = 0;

    console.log(`\nRunning MPC simulation...`);
    console.log('phase | pos                  | ref                  | error  | vy_drone | vy_ref');
    console.log('-'.repeat(95));

    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = trajectory.getWaypoint(simTime);
        const err = posError(state, ref);
        const phase = simTime / period;

        errors.push(err);

        if (step % Math.floor(numSteps / 10) === 0) {
            console.log(
                `${(phase * 100).toFixed(0).padStart(4)}% | ` +
                `${fmtPos(state.position).padEnd(20)} | ` +
                `${fmtPos(ref.position).padEnd(20)} | ` +
                `${err.toFixed(3).padStart(5)}m | ` +
                `${state.velocity.y.toFixed(2).padStart(7)} | ` +
                `${ref.velocity.y.toFixed(2).padStart(6)}`
            );
        }

        const cmd = mpc.computeControl(state, (t) => trajectory.getWaypoint(t), simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgError = errors.reduce((a, b) => a + b, 0) / errors.length;
    const maxError = Math.max(...errors);

    console.log(`\nResults:`);
    console.log(`  Average error: ${avgError.toFixed(3)} m`);
    console.log(`  Max error: ${maxError.toFixed(3)} m`);
    console.log(`  Y span achieved: ${(maxY - minY).toFixed(2)} m`);

    // Pass if avg error < 2m (vertical maneuvers are harder)
    const passed = avgError < 2.0 && hasVerticalMotion;

    console.log(`\nResult: ${passed ? 'PASS' : 'FAIL'} (threshold: avg < 2.0m, needs vertical motion)`);
    return passed;
}

// =====================================================
// MAIN TEST SUITE
// =====================================================

console.log('='.repeat(80));
console.log('TRAJECTORY GENERATOR TESTS');
console.log('='.repeat(80));
console.log(`Simulation dt: ${simDt}s (${1/simDt} Hz)`);

const results: { name: string; passed: boolean }[] = [];

// Run all tests
results.push({ name: 'SmoothLineSegment', passed: testSmoothLineSegment() });
results.push({ name: 'ArcSegment', passed: testArcSegment() });
results.push({ name: 'Triangle Course', passed: testTriangleTrajectory() });
results.push({ name: 'MPC Tracking', passed: testMPCTracking() });
results.push({ name: 'Square Course', passed: testSquareTrajectory() });
results.push({ name: 'Vertical Dive', passed: testVerticalDiveGates() });

// Summary
console.log('\n' + '='.repeat(80));
console.log('TEST SUMMARY');
console.log('='.repeat(80));

for (const r of results) {
    console.log(`  ${r.name.padEnd(20)}: ${r.passed ? 'PASS' : 'FAIL'}`);
}

const allPassed = results.every(r => r.passed);
console.log('\n' + '-'.repeat(80));
console.log(`Overall: ${allPassed ? 'ALL TESTS PASSED' : 'SOME TESTS FAILED'}`);
console.log('='.repeat(80));
