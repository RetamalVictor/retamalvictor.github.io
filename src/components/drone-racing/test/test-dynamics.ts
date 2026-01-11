/**
 * Dynamics Model Test
 *
 * Tests the DroneDynamics class to verify:
 * 1. Hover equilibrium (thrust = gravity)
 * 2. Vertical motion (thrust > gravity)
 * 3. Attitude response (roll/pitch commands)
 * 4. Drag effects at velocity
 *
 * Run with: npx ts-node src/components/drone-racing/test/test-dynamics.ts
 */

import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { ControlCommand } from '../types';

// Helper to create control command
function cmd(thrust: number, rollRate = 0, pitchRate = 0, yawRate = 0): ControlCommand {
    return { thrust, rollRate, pitchRate, yawRate, timestamp: 0 };
}

// Helper to format vector
function vec3(v: { x: number; y: number; z: number }): string {
    return `(${v.x.toFixed(4)}, ${v.y.toFixed(4)}, ${v.z.toFixed(4)})`;
}

// Helper to format quaternion
function quat(q: { w: number; x: number; y: number; z: number }): string {
    return `[w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}]`;
}

console.log('='.repeat(60));
console.log('DRONE DYNAMICS TEST');
console.log('='.repeat(60));
console.log(`\nParameters: ${JSON.stringify(DEFAULT_DYNAMICS_PARAMS, null, 2)}\n`);

// Test 1: Hover Equilibrium
console.log('\n--- TEST 1: Hover Equilibrium ---');
console.log('Expected: Drone stays at same position with thrust = gravity');
{
    const dynamics = new DroneDynamics();
    dynamics.setPosition(0, 2, 0);

    const hoverCmd = cmd(DEFAULT_DYNAMICS_PARAMS.gravity);  // thrust = g
    const dt = 0.02;  // 50 Hz

    console.log(`Initial state:`);
    let state = dynamics.getState();
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);

    // Simulate 1 second
    for (let i = 0; i < 50; i++) {
        dynamics.step(hoverCmd, dt);
    }

    state = dynamics.getState();
    console.log(`After 1s hover:`);
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);

    const posError = Math.abs(state.position.y - 2);
    const velError = Math.sqrt(state.velocity.x ** 2 + state.velocity.y ** 2 + state.velocity.z ** 2);
    console.log(`  Position error: ${posError.toFixed(6)} m`);
    console.log(`  Velocity magnitude: ${velError.toFixed(6)} m/s`);
    console.log(`  PASS: ${posError < 0.01 && velError < 0.01 ? 'YES' : 'NO'}`);
}

// Test 2: Vertical Motion
console.log('\n--- TEST 2: Vertical Motion ---');
console.log('Expected: Drone accelerates upward with thrust > gravity');
{
    const dynamics = new DroneDynamics();
    dynamics.setPosition(0, 2, 0);

    const upCmd = cmd(DEFAULT_DYNAMICS_PARAMS.gravity + 2);  // +2 m/s² acceleration
    const dt = 0.02;

    console.log(`Command: thrust = ${upCmd.thrust.toFixed(2)} m/s² (gravity + 2)`);

    // Simulate 1 second
    for (let i = 0; i < 50; i++) {
        dynamics.step(upCmd, dt);
    }

    const state = dynamics.getState();
    console.log(`After 1s:`);
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);

    // Expected: y ≈ 2 + 0.5*a*t² = 2 + 0.5*2*1 = 3 (without drag)
    // With drag, slightly less
    const expectedY = 2 + 0.5 * 2 * 1;  // ~3m
    console.log(`  Expected Y (no drag): ~${expectedY.toFixed(2)} m`);
    console.log(`  PASS: ${state.position.y > 2.5 && state.velocity.y > 1 ? 'YES' : 'NO'}`);
}

// Test 3: Pitch Response (Forward Motion)
console.log('\n--- TEST 3: Pitch Response ---');
console.log('Expected: Positive pitch rate → forward acceleration (+Z)');
{
    const dynamics = new DroneDynamics();
    dynamics.setPosition(0, 2, 0);

    // Apply pitch rate to tilt forward
    const pitchCmd = cmd(DEFAULT_DYNAMICS_PARAMS.gravity * 1.1, 0, 0.5, 0);  // slight extra thrust + pitch
    const dt = 0.02;

    console.log(`Command: thrust=${pitchCmd.thrust.toFixed(2)}, pitchRate=${pitchCmd.pitchRate} rad/s`);

    // Simulate 2 seconds
    for (let i = 0; i < 100; i++) {
        dynamics.step(pitchCmd, dt);
    }

    const state = dynamics.getState();
    console.log(`After 2s:`);
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);
    console.log(`  Orientation: ${quat(state.orientation)}`);

    // Should have moved forward (+Z) and possibly dropped slightly
    console.log(`  Forward motion (Z): ${state.position.z > 0 ? 'YES' : 'NO'}`);
}

// Test 4: Roll Response (Lateral Motion)
console.log('\n--- TEST 4: Roll Response ---');
console.log('Expected: Positive roll rate → lateral acceleration (-X in body frame)');
{
    const dynamics = new DroneDynamics();
    dynamics.setPosition(0, 2, 0);

    // Apply roll rate
    const rollCmd = cmd(DEFAULT_DYNAMICS_PARAMS.gravity * 1.1, 0.5, 0, 0);
    const dt = 0.02;

    console.log(`Command: thrust=${rollCmd.thrust.toFixed(2)}, rollRate=${rollCmd.rollRate} rad/s`);

    // Simulate 2 seconds
    for (let i = 0; i < 100; i++) {
        dynamics.step(rollCmd, dt);
    }

    const state = dynamics.getState();
    console.log(`After 2s:`);
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);
    console.log(`  Orientation: ${quat(state.orientation)}`);

    // Should have lateral motion
    console.log(`  Lateral motion: ${Math.abs(state.position.x) > 0.1 ? 'YES' : 'NO'}`);
}

// Test 5: Drag Effects
console.log('\n--- TEST 5: Drag Effects ---');
console.log('Expected: Terminal velocity reached when thrust matches drag');
{
    const dynamics = new DroneDynamics();
    dynamics.setPosition(0, 10, 0);  // Start high
    dynamics.setVelocity(0, -5, 0);  // Falling

    // No thrust - free fall with drag
    const freefallCmd = cmd(0);
    const dt = 0.02;

    console.log(`Initial: height=10m, velocity=-5 m/s (falling)`);
    console.log(`Command: thrust=0 (free fall)`);

    let lastVy = -5;
    let terminalReached = false;

    // Simulate 5 seconds
    for (let i = 0; i < 250; i++) {
        dynamics.step(freefallCmd, dt);
        const state = dynamics.getState();

        // Check for terminal velocity (velocity stops changing)
        if (Math.abs(state.velocity.y - lastVy) < 0.001 && !terminalReached) {
            console.log(`  Terminal velocity reached at t=${(i * dt).toFixed(2)}s: vy=${state.velocity.y.toFixed(3)} m/s`);
            terminalReached = true;
        }
        lastVy = state.velocity.y;

        // Stop if hit ground
        if (state.position.y <= 0) {
            console.log(`  Hit ground at t=${(i * dt).toFixed(2)}s`);
            break;
        }
    }

    const state = dynamics.getState();
    console.log(`Final state:`);
    console.log(`  Position: ${vec3(state.position)}`);
    console.log(`  Velocity: ${vec3(state.velocity)}`);

    // Terminal velocity = g / drag = 9.81 / 0.3 ≈ 32.7 m/s
    const expectedTerminal = DEFAULT_DYNAMICS_PARAMS.gravity / DEFAULT_DYNAMICS_PARAMS.linearDrag;
    console.log(`  Expected terminal velocity: ~${expectedTerminal.toFixed(1)} m/s`);
}

// Test 6: Circular Trajectory Following (manual)
console.log('\n--- TEST 6: Circular Trajectory Simulation ---');
console.log('Simulating what it would take to fly a circle');
{
    const dynamics = new DroneDynamics();
    const radius = 5;
    const speed = 3;
    const omega = speed / radius;  // angular velocity
    const centripetalAccel = speed * speed / radius;

    // Start at (radius, height, 0) moving in +Z direction
    dynamics.setPosition(radius, 2, 0);
    dynamics.setVelocity(0, 0, speed);

    console.log(`Circle: radius=${radius}m, speed=${speed}m/s`);
    console.log(`Angular velocity: ${omega.toFixed(3)} rad/s`);
    console.log(`Centripetal acceleration: ${centripetalAccel.toFixed(3)} m/s²`);

    // Required roll angle to produce centripetal acceleration
    const rollAngle = Math.atan2(centripetalAccel, DEFAULT_DYNAMICS_PARAMS.gravity);
    console.log(`Required roll angle: ${(rollAngle * 180 / Math.PI).toFixed(1)}°`);

    // Required thrust to maintain altitude while rolled
    const requiredThrust = DEFAULT_DYNAMICS_PARAMS.gravity / Math.cos(rollAngle);
    console.log(`Required thrust: ${requiredThrust.toFixed(2)} m/s²`);
}

console.log('\n' + '='.repeat(60));
console.log('DYNAMICS TESTS COMPLETE');
console.log('='.repeat(60));
