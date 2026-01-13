/**
 * Gate Passing Test
 *
 * Checks if the trajectory passes through all gates.
 * Samples the trajectory at high resolution and checks gate crossings.
 *
 * Run: npx tsx src/components/drone-racing/test/test-gate-passing.ts
 */

import { createTrajectory } from '../trajectory/index';

// ANSI colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const RESET = '\x1b[0m';

interface GateResult {
    index: number;
    position: { x: number; y: number; z: number };
    passed: boolean;
    closestDistance: number;
    closestPos: { x: number; y: number; z: number };
    phaseAtClosest: number;
    withinXY: boolean;
    withinZ: boolean;
}

/**
 * Run trajectory gate check
 */
function checkGatePassing(): void {
    console.log(`\n${CYAN}${'═'.repeat(70)}${RESET}`);
    console.log(`${CYAN}  Trajectory Gate Passing Check${RESET}`);
    console.log(`${CYAN}${'═'.repeat(70)}${RESET}\n`);

    // Create trajectory
    const trajectory = createTrajectory(25, 4);
    const period = trajectory.getPeriod();
    const gatePositions = trajectory.getGatePositions();

    console.log(`Trajectory period: ${period.toFixed(2)}s`);
    console.log(`Number of gates: ${gatePositions.length}\n`);

    // Gate tolerance (matching DroneRacingDemo.ts)
    const gateHalfWidth = 2.0;
    const gateHalfHeight = 2.0;
    const zTolerance = 1.5;  // How close in Z to count as "at gate plane"

    // Initialize results
    const results: GateResult[] = gatePositions.map((g, i) => ({
        index: i,
        position: { ...g.position },
        passed: false,
        closestDistance: Infinity,
        closestPos: { x: 0, y: 0, z: 0 },
        phaseAtClosest: 0,
        withinXY: false,
        withinZ: false,
    }));

    // Sample trajectory at high resolution
    const samples = 2000;
    let prevX: number[] = gatePositions.map(() => -Infinity);
    let prevZ: number[] = gatePositions.map(() => -Infinity);

    for (let i = 0; i <= samples; i++) {
        const phase = i / samples;
        const t = phase * period;
        const wp = trajectory.getWaypoint(t);
        const pos = wp.position;

        // Check each gate
        for (let gi = 0; gi < gatePositions.length; gi++) {
            const gate = gatePositions[gi];
            const result = results[gi];

            // Distance to gate center
            const dx = pos.x - gate.position.x;
            const dy = pos.y - gate.position.y;
            const dz = pos.z - gate.position.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

            // Update closest approach
            if (dist < result.closestDistance) {
                result.closestDistance = dist;
                result.closestPos = { ...pos };
                result.phaseAtClosest = phase;
                result.withinXY = Math.abs(dx) < gateHalfWidth && Math.abs(dy) < gateHalfHeight;
                result.withinZ = Math.abs(dz) < zTolerance;
            }

            // Check for plane crossing in correct direction
            const heading = gate.heading;
            const facesX = Math.abs(Math.sin(heading)) > 0.7;
            const cosH = Math.cos(heading);
            const sinH = Math.sin(heading);

            let crossed = false;
            let withinBounds = false;

            if (facesX) {
                const expectPlusX = sinH > 0.5;
                const crossedX = expectPlusX
                    ? (prevX[gi] < gate.position.x && pos.x >= gate.position.x)
                    : (prevX[gi] > gate.position.x && pos.x <= gate.position.x);
                crossed = crossedX;
                withinBounds = Math.abs(dz) < gateHalfWidth && Math.abs(dy) < gateHalfHeight;
            } else {
                const expectPlusZ = cosH > 0.5;
                const crossedZ = expectPlusZ
                    ? (prevZ[gi] < gate.position.z && pos.z >= gate.position.z)
                    : (prevZ[gi] > gate.position.z && pos.z <= gate.position.z);
                crossed = crossedZ;
                withinBounds = Math.abs(dx) < gateHalfWidth && Math.abs(dy) < gateHalfHeight;
            }

            if (crossed && withinBounds) {
                result.passed = true;
            }

            prevX[gi] = pos.x;
            prevZ[gi] = pos.z;
        }
    }

    // Print results
    console.log(`${CYAN}Gate Results:${RESET}\n`);
    console.log('Gate  Position          Status    Closest   At Phase  Within XY  Within Z');
    console.log('─'.repeat(75));

    let allPassed = true;
    for (const r of results) {
        const status = r.passed ? `${GREEN}PASS${RESET}` : `${RED}MISS${RESET}`;
        const posStr = `(${r.position.x.toFixed(0)}, ${r.position.y.toFixed(0)}, ${r.position.z.toFixed(0)})`.padEnd(16);
        const distStr = `${r.closestDistance.toFixed(2)}m`.padStart(8);
        const phaseStr = r.phaseAtClosest.toFixed(3).padStart(8);
        const xyStr = r.withinXY ? `${GREEN}Yes${RESET}` : `${RED}No${RESET}`;
        const zStr = r.withinZ ? `${GREEN}Yes${RESET}` : `${RED}No${RESET}`;

        console.log(`  ${r.index + 1}   ${posStr}  ${status}    ${distStr}  ${phaseStr}     ${xyStr}        ${zStr}`);

        if (!r.passed) allPassed = false;
    }

    // Summary
    console.log(`\n${'─'.repeat(75)}`);
    if (allPassed) {
        console.log(`\n${GREEN}✓ All gates passed by trajectory!${RESET}\n`);
    } else {
        console.log(`\n${RED}✗ Some gates missed by trajectory!${RESET}\n`);

        // Detailed analysis of missed gates
        console.log(`${YELLOW}Analysis of missed gates:${RESET}\n`);
        for (const r of results) {
            if (!r.passed) {
                const dx = Math.abs(r.closestPos.x - r.position.x);
                const dy = Math.abs(r.closestPos.y - r.position.y);
                const dz = Math.abs(r.closestPos.z - r.position.z);

                console.log(`  Gate ${r.index + 1} at (${r.position.x}, ${r.position.y}, ${r.position.z}):`);
                console.log(`    Closest approach: (${r.closestPos.x.toFixed(1)}, ${r.closestPos.y.toFixed(1)}, ${r.closestPos.z.toFixed(1)})`);
                console.log(`    Distance: ${r.closestDistance.toFixed(2)}m at phase ${r.phaseAtClosest.toFixed(3)}`);
                console.log(`    Offsets: dx=${dx.toFixed(2)}m, dy=${dy.toFixed(2)}m, dz=${dz.toFixed(2)}m`);

                if (dx >= gateHalfWidth) {
                    console.log(`    ${RED}→ Too far in X (need dx < ${gateHalfWidth}m)${RESET}`);
                }
                if (dy >= gateHalfHeight) {
                    console.log(`    ${RED}→ Too far in Y (need dy < ${gateHalfHeight}m)${RESET}`);
                }
                if (!r.withinZ) {
                    console.log(`    ${RED}→ Doesn't cross gate Z plane closely${RESET}`);
                }
                console.log('');
            }
        }
    }

    // Show trajectory path around each gate
    console.log(`\n${CYAN}Trajectory near each gate:${RESET}\n`);
    for (const r of results) {
        const phase = r.phaseAtClosest;
        const samples = [-0.02, -0.01, 0, 0.01, 0.02];

        console.log(`Gate ${r.index + 1} (${r.position.x}, ${r.position.y}, ${r.position.z}):`);
        for (const dp of samples) {
            const p = Math.max(0, Math.min(1, phase + dp));
            const wp = trajectory.getWaypoint(p * period);
            const speed = Math.sqrt(wp.velocity.x**2 + wp.velocity.y**2 + wp.velocity.z**2) * 3.6;
            console.log(`  phase ${p.toFixed(3)}: (${wp.position.x.toFixed(1)}, ${wp.position.y.toFixed(1)}, ${wp.position.z.toFixed(1)}) @ ${speed.toFixed(0)} km/h`);
        }
        console.log('');
    }
}

// Run
checkGatePassing();
