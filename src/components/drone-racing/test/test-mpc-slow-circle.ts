/**
 * Test MPC with slower circular trajectory
 * If speed is lower, bank angle is smaller, yaw changes slower
 */

import { MPC } from '../control/MPC';
import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { Waypoint } from '../types';

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

const simDt = 0.02;

function testCircle(radius: number, speed: number, label: string) {
    console.log(`\n--- ${label}: radius=${radius}m, speed=${speed}m/s ---`);

    const mpc = new MPC();
    const dynamics = new DroneDynamics();

    const height = 2;
    const omega = speed / radius;
    const centripetalAccel = speed * speed / radius;
    const requiredRoll = Math.atan(centripetalAccel / DEFAULT_DYNAMICS_PARAMS.gravity);

    console.log(`Angular velocity: ${omega.toFixed(3)} rad/s (${(omega * 180 / Math.PI).toFixed(1)}°/s)`);
    console.log(`Centripetal accel: ${centripetalAccel.toFixed(2)} m/s²`);
    console.log(`Required bank angle: ${(requiredRoll * 180 / Math.PI).toFixed(1)}°`);

    const getReference = (t: number): Waypoint => {
        const angle = omega * t;
        const heading = -angle;
        return createWaypoint(
            radius * Math.cos(angle), height, radius * Math.sin(angle),
            -speed * Math.sin(angle), 0, speed * Math.cos(angle),
            -centripetalAccel * Math.cos(angle), 0, -centripetalAccel * Math.sin(angle),
            heading, -omega, t
        );
    };

    const start = getReference(0);
    dynamics.setPosition(start.position.x, start.position.y, start.position.z);
    dynamics.setVelocity(start.velocity.x, start.velocity.y, start.velocity.z);
    dynamics.setHeading(start.heading);

    const period = 2 * Math.PI / omega;
    const numSteps = Math.ceil(period / simDt);
    const errors: number[] = [];

    let simTime = 0;
    for (let step = 0; step < numSteps; step++) {
        const state = dynamics.getState();
        const ref = getReference(simTime);
        const err = Math.sqrt(
            (state.position.x - ref.position.x) ** 2 +
            (state.position.y - ref.position.y) ** 2 +
            (state.position.z - ref.position.z) ** 2
        );
        errors.push(err);

        if (step % Math.floor(numSteps / 8) === 0) {
            console.log(`t=${simTime.toFixed(2)}s: pos=(${state.position.x.toFixed(2)}, ${state.position.y.toFixed(2)}, ${state.position.z.toFixed(2)}) err=${err.toFixed(3)}m`);
        }

        const cmd = mpc.computeControl(state, getReference, simTime);
        dynamics.step(cmd, simDt);
        simTime += simDt;
    }

    const avgErr = errors.reduce((a, b) => a + b, 0) / errors.length;
    const maxErr = Math.max(...errors);
    console.log(`\nAverage error: ${avgErr.toFixed(4)}m, Max error: ${maxErr.toFixed(4)}m`);
    console.log(`PASS: ${avgErr < 0.5 && maxErr < 1.0 ? 'YES' : 'NO'}`);
    return { avgErr, maxErr };
}

console.log('='.repeat(70));
console.log('MPC CIRCULAR TRAJECTORY - VARYING SPEED');
console.log('='.repeat(70));

// Original test case
testCircle(5, 3, "Original");

// Slower speeds
testCircle(5, 2, "Slow (2 m/s)");
testCircle(5, 1.5, "Very slow (1.5 m/s)");

// Smaller radius (tighter turn, more bank)
testCircle(3, 2, "Tight (r=3, v=2)");

console.log('\n' + '='.repeat(70));
