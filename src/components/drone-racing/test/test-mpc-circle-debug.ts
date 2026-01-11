/**
 * Debug test for circular trajectory
 * Investigates why MPC fails on the far side of the circle
 */

import { MPC, DEFAULT_MPC_CONFIG } from '../control/MPC';
import { DroneDynamics, DEFAULT_DYNAMICS_PARAMS } from '../core/DroneDynamics';
import { DroneState, Waypoint } from '../types';

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const simDt = 0.02;

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

console.log('='.repeat(70));
console.log('CIRCULAR TRAJECTORY DEBUG');
console.log('='.repeat(70));

const mpc = new MPC();
const dynamics = new DroneDynamics();

const radius = 5;
const speed = 3;
const height = 2;
const omega = speed / radius;
const centripetalAccel = speed * speed / radius;

const getReference = (t: number): Waypoint => {
    const angle = omega * t;
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
        heading,                               // heading
        -omega,                                // headingRate
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
console.log(`\nFocusing on t=2.0s to t=4.5s (problematic region)\n`);

let simTime = 0;

// Skip to t=2.0s
while (simTime < 2.0) {
    const state = dynamics.getState();
    const cmd = mpc.computeControl(state, getReference, simTime);
    dynamics.step(cmd, simDt);
    simTime += simDt;
}

console.log('time   | angle  | heading | pos_x   | pos_y   | pos_z   | ref_y | vel_y   | thrust | roll_r  | pitch_r | err');
console.log('-'.repeat(120));

// Detailed logging from t=2.0s to t=4.5s
for (let step = 0; step < 125; step++) {
    const state = dynamics.getState();
    const ref = getReference(simTime);
    const angle = omega * simTime;

    const err = Math.sqrt(
        (state.position.x - ref.position.x) ** 2 +
        (state.position.y - ref.position.y) ** 2 +
        (state.position.z - ref.position.z) ** 2
    );

    const cmd = mpc.computeControl(state, getReference, simTime);

    // Log every 5 steps (0.1s)
    if (step % 5 === 0) {
        console.log(
            `${simTime.toFixed(2).padStart(5)}s | ` +
            `${(angle * 180 / Math.PI).toFixed(0).padStart(5)}° | ` +
            `${(ref.heading * 180 / Math.PI).toFixed(0).padStart(6)}° | ` +
            `${state.position.x.toFixed(2).padStart(6)} | ` +
            `${state.position.y.toFixed(2).padStart(6)} | ` +
            `${state.position.z.toFixed(2).padStart(6)} | ` +
            `${ref.position.y.toFixed(2).padStart(5)} | ` +
            `${state.velocity.y.toFixed(2).padStart(6)} | ` +
            `${cmd.thrust.toFixed(2).padStart(5)} | ` +
            `${cmd.rollRate.toFixed(2).padStart(6)} | ` +
            `${cmd.pitchRate.toFixed(2).padStart(6)} | ` +
            `${err.toFixed(3)}`
        );
    }

    dynamics.step(cmd, simDt);
    simTime += simDt;
}

console.log('\n' + '='.repeat(70));

// Also check what reference attitudes are being computed
console.log('\nReference attitude analysis at problematic angles:');
console.log('-'.repeat(70));

const testAngles = [90, 120, 150, 180, 210, 240, 270];
for (const angleDeg of testAngles) {
    const t = angleDeg * Math.PI / 180 / omega;
    const ref = getReference(t);

    // Manually compute what attitude should be
    const ax = ref.acceleration.x;
    const ay = ref.acceleration.y + g;  // Compensate gravity
    const az = ref.acceleration.z;

    const yaw = ref.heading;
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);

    const axBody = ax * cy - az * sy;
    const ayBody = ay;
    const azBody = ax * sy + az * cy;

    const roll = -Math.atan2(axBody, ayBody);
    const pitch = Math.atan2(azBody, ayBody);

    const thrustMag = Math.sqrt(ax * ax + ay * ay + az * az);

    console.log(
        `${angleDeg}°: heading=${(yaw * 180 / Math.PI).toFixed(0)}°, ` +
        `accel=(${ax.toFixed(2)}, ${ay.toFixed(2)}, ${az.toFixed(2)}), ` +
        `roll=${(roll * 180 / Math.PI).toFixed(1)}°, ` +
        `pitch=${(pitch * 180 / Math.PI).toFixed(1)}°, ` +
        `thrust=${thrustMag.toFixed(2)}`
    );
}

console.log('\n' + '='.repeat(70));
