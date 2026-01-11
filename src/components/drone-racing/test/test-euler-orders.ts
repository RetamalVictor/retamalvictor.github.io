/**
 * Test different Euler orders to find which matches DroneDynamics
 */

import { DroneDynamics } from '../core/DroneDynamics';

console.log('='.repeat(70));
console.log('EULER ORDER TEST');
console.log('='.repeat(70));

// Euler extraction functions for different orders

function extractYXZ(q: { w: number; x: number; y: number; z: number }) {
    const { w, x, y, z } = q;

    // Roll (Z)
    const sinRoll = 2 * (w * z + x * y);
    const cosRoll = 1 - 2 * (y * y + z * z);
    const roll = Math.atan2(sinRoll, cosRoll);

    // Pitch (X)
    const sinPitch = 2 * (w * x - z * y);
    const pitch = Math.abs(sinPitch) >= 1
        ? Math.sign(sinPitch) * Math.PI / 2
        : Math.asin(sinPitch);

    // Yaw (Y)
    const sinYaw = 2 * (w * y + z * x);
    const cosYaw = 1 - 2 * (x * x + y * y);
    const yaw = Math.atan2(sinYaw, cosYaw);

    return { roll, pitch, yaw };
}

function extractZXY(q: { w: number; x: number; y: number; z: number }) {
    const { w, x, y, z } = q;

    // Roll (Z) - first rotation
    const sinRoll = 2 * (w * z - x * y);
    const cosRoll = 1 - 2 * (x * x + z * z);
    const roll = Math.atan2(sinRoll, cosRoll);

    // Pitch (X) - second rotation
    const sinPitch = 2 * (w * x + y * z);
    const pitch = Math.abs(sinPitch) >= 1
        ? Math.sign(sinPitch) * Math.PI / 2
        : Math.asin(sinPitch);

    // Yaw (Y) - third rotation
    const sinYaw = 2 * (w * y - z * x);
    const cosYaw = 1 - 2 * (x * x + y * y);
    const yaw = Math.atan2(sinYaw, cosYaw);

    return { roll, pitch, yaw };
}

function extractXYZ(q: { w: number; x: number; y: number; z: number }) {
    const { w, x, y, z } = q;

    // Roll (X) - first rotation... this doesn't match our axis convention
    // Let's map to our roll=Z, pitch=X, yaw=Y convention
    // For XYZ: first X (pitch), then Y (yaw), then Z (roll)

    const sinYaw = 2 * (w * y + z * x);
    const cosYaw = 1 - 2 * (x * x + y * y);
    const yaw = Math.atan2(sinYaw, cosYaw);

    const sinRoll = 2 * (w * z + x * y);
    const cosRoll = w * w + x * x - y * y - z * z;
    const roll = Math.atan2(sinRoll, cosRoll);

    const sinPitch = 2 * (w * x - z * y);
    const pitch = Math.abs(sinPitch) >= 1
        ? Math.sign(sinPitch) * Math.PI / 2
        : Math.asin(sinPitch);

    return { roll, pitch, yaw };
}

// Alternative: extract directly from rotation matrix
function extractFromRotationMatrix(q: { w: number; x: number; y: number; z: number }) {
    const { w, x, y, z } = q;

    // Build rotation matrix from quaternion
    const r00 = 1 - 2 * (y * y + z * z);
    const r01 = 2 * (x * y - w * z);
    const r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z);
    const r11 = 1 - 2 * (x * x + z * z);
    const r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y);
    const r21 = 2 * (y * z + w * x);
    const r22 = 1 - 2 * (x * x + y * y);

    // For DroneDynamics, thrust is in +Y direction
    // After rotation, thrust_world = R * [0, 1, 0]^T = [r01, r11, r21]
    // For hover, we want thrust_world = [0, 1, 0], so r01=0, r11=1, r21=0

    // Roll tilts thrust in X: positive roll -> thrust in -X
    // Pitch tilts thrust in Z: positive pitch -> thrust in +Z

    // From the rotation matrix, for small angles:
    // r01 ≈ -sin(roll) * cos(pitch)
    // r21 ≈ sin(pitch) * cos(roll)
    // r11 ≈ cos(roll) * cos(pitch)

    // Extract roll from r01 and r11
    const roll = Math.atan2(-r01, r11);

    // Extract pitch from r21 and sqrt(r01² + r11²)
    const pitch = Math.atan2(r21, Math.sqrt(r01 * r01 + r11 * r11));

    // Extract yaw from r00 and r20
    const yaw = Math.atan2(r20, r00);

    return { roll, pitch, yaw };
}

// Test cases
console.log('\n--- Test 1: Pure yaw = -90° ---');
{
    const dynamics = new DroneDynamics();
    dynamics.setHeading(-Math.PI / 2);  // -90°
    const q = dynamics.getState().orientation;
    console.log(`Quaternion: w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}`);

    const yxz = extractYXZ(q);
    const zxy = extractZXY(q);
    const rotMat = extractFromRotationMatrix(q);

    console.log(`\nYXZ: roll=${(yxz.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(yxz.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(yxz.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`ZXY: roll=${(zxy.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(zxy.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(zxy.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`RotMat: roll=${(rotMat.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rotMat.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(rotMat.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Expected: roll=0°, pitch=0°, yaw=-90°`);
}

console.log('\n--- Test 2: Pure roll = 30° (with yaw = 0°) ---');
{
    const dynamics = new DroneDynamics();
    // Apply roll rate for 0.3s
    for (let i = 0; i < 15; i++) {
        dynamics.step({ thrust: 9.81, rollRate: 2, pitchRate: 0, yawRate: 0, timestamp: 0 }, 0.02);
    }
    const q = dynamics.getState().orientation;
    console.log(`Quaternion: w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}`);

    const yxz = extractYXZ(q);
    const zxy = extractZXY(q);
    const rotMat = extractFromRotationMatrix(q);

    console.log(`\nYXZ: roll=${(yxz.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(yxz.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(yxz.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`ZXY: roll=${(zxy.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(zxy.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(zxy.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`RotMat: roll=${(rotMat.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rotMat.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(rotMat.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Expected: roll≈30°, pitch=0°, yaw=0°`);
}

console.log('\n--- Test 3: Pure pitch = 30° (with yaw = 0°) ---');
{
    const dynamics = new DroneDynamics();
    // Apply pitch rate for 0.3s
    for (let i = 0; i < 15; i++) {
        dynamics.step({ thrust: 9.81, rollRate: 0, pitchRate: 2, yawRate: 0, timestamp: 0 }, 0.02);
    }
    const q = dynamics.getState().orientation;
    console.log(`Quaternion: w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}`);

    const yxz = extractYXZ(q);
    const zxy = extractZXY(q);
    const rotMat = extractFromRotationMatrix(q);

    console.log(`\nYXZ: roll=${(yxz.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(yxz.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(yxz.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`ZXY: roll=${(zxy.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(zxy.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(zxy.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`RotMat: roll=${(rotMat.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rotMat.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(rotMat.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Expected: roll=0°, pitch≈30°, yaw=0°`);
}

console.log('\n--- Test 4: Yaw=-90° + small roll = 10° ---');
{
    const dynamics = new DroneDynamics();
    dynamics.setHeading(-Math.PI / 2);  // -90°
    // Apply roll rate to get ~10° roll
    for (let i = 0; i < 5; i++) {
        dynamics.step({ thrust: 9.81, rollRate: 2, pitchRate: 0, yawRate: 0, timestamp: 0 }, 0.02);
    }
    const q = dynamics.getState().orientation;
    console.log(`Quaternion: w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}`);

    const yxz = extractYXZ(q);
    const zxy = extractZXY(q);
    const rotMat = extractFromRotationMatrix(q);

    console.log(`\nYXZ: roll=${(yxz.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(yxz.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(yxz.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`ZXY: roll=${(zxy.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(zxy.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(zxy.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`RotMat: roll=${(rotMat.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rotMat.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(rotMat.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Expected: roll≈10°, pitch=0°, yaw≈-90°`);
}

console.log('\n--- Test 5: Actual quaternion from circular trajectory at t=2.55s ---');
{
    // From previous test output
    const q = { w: 0.7091, x: 0.0409, y: -0.7020, z: 0.0525 };
    console.log(`Quaternion: w=${q.w.toFixed(4)}, x=${q.x.toFixed(4)}, y=${q.y.toFixed(4)}, z=${q.z.toFixed(4)}`);

    const yxz = extractYXZ(q);
    const zxy = extractZXY(q);
    const rotMat = extractFromRotationMatrix(q);

    console.log(`\nYXZ: roll=${(yxz.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(yxz.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(yxz.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`ZXY: roll=${(zxy.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(zxy.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(zxy.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`RotMat: roll=${(rotMat.roll * 180 / Math.PI).toFixed(1)}°, pitch=${(rotMat.pitch * 180 / Math.PI).toFixed(1)}°, yaw=${(rotMat.yaw * 180 / Math.PI).toFixed(1)}°`);
    console.log(`Expected: roll≈small, pitch≈small, yaw≈-90°`);
}

console.log('\n' + '='.repeat(70));
