/**
 * Debug MPC circular trajectory failure
 * Focus on what happens around t=3.9s and t=6.5s
 */

import { MPC } from '../control/MPC';
import { MPCModel } from '../control/MPCModel';
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

const g = DEFAULT_DYNAMICS_PARAMS.gravity;
const simDt = 0.02;

console.log('='.repeat(70));
console.log('MPC CIRCULAR TRAJECTORY DEBUG');
console.log('='.repeat(70));

const mpc = new MPC();
const model = new MPCModel();
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
        radius * Math.cos(angle), height, radius * Math.sin(angle),
        -speed * Math.sin(angle), 0, speed * Math.cos(angle),
        -centripetalAccel * Math.cos(angle), 0, -centripetalAccel * Math.sin(angle),
        heading, -omega, t
    );
};

// Initialize at start of trajectory
const start = getReference(0);
dynamics.setPosition(start.position.x, start.position.y, start.position.z);
dynamics.setVelocity(start.velocity.x, start.velocity.y, start.velocity.z);
dynamics.setHeading(start.heading);

console.log(`Circle: radius=${radius}m, speed=${speed}m/s`);
console.log(`Angular velocity: ${omega.toFixed(3)} rad/s`);

const period = 2 * Math.PI / omega;
const numSteps = Math.ceil(period / simDt);

// Key times to debug
const debugTimes = [4.5, 5.0, 5.2, 5.5, 6.0, 6.5];

let simTime = 0;

for (let step = 0; step < numSteps; step++) {
    const state = dynamics.getState();
    const ref = getReference(simTime);

    // Check if this is a debug time
    const isDebugTime = debugTimes.some(t => Math.abs(simTime - t) < simDt / 2);

    if (isDebugTime) {
        console.log('\n' + '='.repeat(70));
        console.log(`TIME: t=${simTime.toFixed(2)}s`);
        console.log('='.repeat(70));

        // Reference state
        let heading = ref.heading;
        while (heading > Math.PI) heading -= 2 * Math.PI;
        while (heading < -Math.PI) heading += 2 * Math.PI;

        console.log(`\nReference:`);
        console.log(`  Position: (${ref.position.x.toFixed(3)}, ${ref.position.y.toFixed(3)}, ${ref.position.z.toFixed(3)})`);
        console.log(`  Velocity: (${ref.velocity.x.toFixed(3)}, ${ref.velocity.y.toFixed(3)}, ${ref.velocity.z.toFixed(3)})`);
        console.log(`  Accel:    (${ref.acceleration.x.toFixed(3)}, ${ref.acceleration.y.toFixed(3)}, ${ref.acceleration.z.toFixed(3)})`);
        console.log(`  Heading:  ${(heading * 180 / Math.PI).toFixed(1)}° (raw: ${(ref.heading * 180 / Math.PI).toFixed(1)}°)`);

        // Actual state
        const mpcState = model.fromDroneState(state.position, state.velocity, state.orientation);
        console.log(`\nActual:`);
        console.log(`  Position: (${state.position.x.toFixed(3)}, ${state.position.y.toFixed(3)}, ${state.position.z.toFixed(3)})`);
        console.log(`  Velocity: (${state.velocity.x.toFixed(3)}, ${state.velocity.y.toFixed(3)}, ${state.velocity.z.toFixed(3)})`);
        console.log(`  Euler (YXZ): roll=${(mpcState.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(mpcState.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(mpcState.yaw * 180 / Math.PI).toFixed(1)}°`);
        console.log(`  Quaternion: w=${state.orientation.w.toFixed(4)}, x=${state.orientation.x.toFixed(4)}, y=${state.orientation.y.toFixed(4)}, z=${state.orientation.z.toFixed(4)}`);

        // Compute reference attitude
        const ax = ref.acceleration.x;
        const ay = ref.acceleration.y + g;
        const az = ref.acceleration.z;
        const thrustMag = Math.sqrt(ax * ax + ay * ay + az * az);
        const thrustDir = { x: ax / thrustMag, y: ay / thrustMag, z: az / thrustMag };
        console.log(`\nReference thrust direction: (${thrustDir.x.toFixed(3)}, ${thrustDir.y.toFixed(3)}, ${thrustDir.z.toFixed(3)})`);

        // Compute actual thrust direction from Euler angles
        const { roll, pitch, yaw } = mpcState;
        const sr = Math.sin(roll), cr = Math.cos(roll);
        const sp = Math.sin(pitch), cp = Math.cos(pitch);
        const sy = Math.sin(yaw), cy = Math.cos(yaw);
        const actualThrustDir = {
            x: -sr * cy + cr * sp * sy,
            y: cr * cp,
            z: cr * sp * cy + sr * sy,
        };
        console.log(`Actual thrust direction:    (${actualThrustDir.x.toFixed(3)}, ${actualThrustDir.y.toFixed(3)}, ${actualThrustDir.z.toFixed(3)})`);

        // Position error
        const posErr = Math.sqrt(
            (state.position.x - ref.position.x) ** 2 +
            (state.position.y - ref.position.y) ** 2 +
            (state.position.z - ref.position.z) ** 2
        );
        console.log(`\nPosition error: ${posErr.toFixed(3)}m`);

        // Get MPC command
        const cmd = mpc.computeControl(state, getReference, simTime);
        console.log(`\nMPC Command:`);
        console.log(`  Thrust:    ${cmd.thrust.toFixed(3)} m/s² (min=2, hover=${g.toFixed(2)}, max=20)`);
        console.log(`  RollRate:  ${cmd.rollRate.toFixed(3)} rad/s (max=±10)`);
        console.log(`  PitchRate: ${cmd.pitchRate.toFixed(3)} rad/s (max=±10)`);
        console.log(`  YawRate:   ${cmd.yawRate.toFixed(3)} rad/s (max=±3)`);

        // Check if hitting limits
        const atMinThrust = cmd.thrust <= 2.01;
        const atMaxThrust = cmd.thrust >= 19.99;
        const atMaxRollRate = Math.abs(cmd.rollRate) >= 9.99;
        const atMaxPitchRate = Math.abs(cmd.pitchRate) >= 9.99;
        if (atMinThrust || atMaxThrust || atMaxRollRate || atMaxPitchRate) {
            console.log(`  ⚠️  At limits: ${atMinThrust ? 'MIN_THRUST ' : ''}${atMaxThrust ? 'MAX_THRUST ' : ''}${atMaxRollRate ? 'MAX_ROLL_RATE ' : ''}${atMaxPitchRate ? 'MAX_PITCH_RATE' : ''}`);
        }

        dynamics.step(cmd, simDt);
    } else {
        const cmd = mpc.computeControl(state, getReference, simTime);
        dynamics.step(cmd, simDt);
    }

    simTime += simDt;
}

console.log('\n' + '='.repeat(70));
