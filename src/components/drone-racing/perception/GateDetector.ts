import * as THREE from 'three';
import { PinholeCamera } from '../../ibvs/Camera';
import { RaceGate } from '../core/RaceGate';
import { GateDetection, Point2D, RacingConfig, DEFAULT_CONFIG } from '../types';

/**
 * Gate Detection Simulator
 *
 * Simulates the gate detection pipeline from the real system:
 * 1. Project gate corners from 3D to 2D image space
 * 2. Add realistic detection noise
 * 3. Check visibility (in frame, not occluded)
 * 4. Compute detection confidence
 *
 * In the real system, this is done by YOLO + CNN keypoint regression.
 * Here we directly project known gate corners with added noise.
 */
export class GateDetector {
    private camera: PinholeCamera;
    private noiseLevel: number;
    private config: RacingConfig;

    // Detection state
    private lastDetections: GateDetection[] = [];

    constructor(camera: PinholeCamera, config: Partial<RacingConfig> = {}) {
        this.camera = camera;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.noiseLevel = this.config.detectionNoise;
    }

    /**
     * Detect gates visible from current camera pose
     *
     * @param gates - Array of gates to detect
     * @param cameraPose - Camera position and orientation
     * @returns Array of gate detections with 2D keypoints
     */
    public detectGates(
        gates: RaceGate[],
        cameraPose: { position: THREE.Vector3; rotation: THREE.Euler }
    ): GateDetection[] {
        // Update camera pose
        this.camera.setPose(
            cameraPose.position.x,
            cameraPose.position.y,
            cameraPose.position.z,
            cameraPose.rotation.x,
            cameraPose.rotation.y,
            cameraPose.rotation.z
        );

        const detections: GateDetection[] = [];

        for (const gate of gates) {
            const detection = this.detectSingleGate(gate);
            if (detection) {
                detections.push(detection);
            }
        }

        this.lastDetections = detections;
        return detections;
    }

    /**
     * Detect a single gate
     */
    private detectSingleGate(gate: RaceGate): GateDetection | null {
        // Get gate corners in world frame
        const worldCorners = gate.getWorldCorners();

        // Project corners to image space
        const { imagePoints, visible } = this.camera.projectPoints(worldCorners);

        // Check if all corners are visible
        const allVisible = visible.every(v => v);
        if (!allVisible) {
            return null;
        }

        // Extract corner points
        const corners: [Point2D, Point2D, Point2D, Point2D] = [
            { u: imagePoints[0], v: imagePoints[1] },  // TL
            { u: imagePoints[2], v: imagePoints[3] },  // TR
            { u: imagePoints[4], v: imagePoints[5] },  // BR
            { u: imagePoints[6], v: imagePoints[7] },  // BL
        ];

        // Add detection noise
        const noisyCorners = this.addNoise(corners);

        // Compute confidence based on:
        // 1. Gate size in image (larger = more confident)
        // 2. Distance from image center
        // 3. Viewing angle
        const confidence = this.computeConfidence(noisyCorners, gate);

        // Skip low-confidence detections
        if (confidence < 0.3) {
            return null;
        }

        return {
            gateId: gate.id,
            keypoints: {
                corners: noisyCorners,
                confidence,
            },
            reprojectionError: 0,  // Will be computed by PnP solver
            visible: true,
        };
    }

    /**
     * Add Gaussian noise to corner detections
     */
    private addNoise(corners: [Point2D, Point2D, Point2D, Point2D]): [Point2D, Point2D, Point2D, Point2D] {
        return corners.map(corner => ({
            u: corner.u + this.gaussianNoise() * this.noiseLevel,
            v: corner.v + this.gaussianNoise() * this.noiseLevel,
        })) as [Point2D, Point2D, Point2D, Point2D];
    }

    /**
     * Generate Gaussian noise using Box-Muller transform
     */
    private gaussianNoise(): number {
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /**
     * Compute detection confidence
     */
    private computeConfidence(corners: [Point2D, Point2D, Point2D, Point2D], _gate: RaceGate): number {
        // 1. Gate size in image (area of quadrilateral)
        const area = this.computeQuadArea(corners);
        const maxArea = this.camera.width * this.camera.height * 0.5;
        const sizeScore = Math.min(1, Math.sqrt(area / maxArea) * 2);

        // 2. Distance from image center
        const centerU = this.camera.width / 2;
        const centerV = this.camera.height / 2;
        const gateCenterU = (corners[0].u + corners[1].u + corners[2].u + corners[3].u) / 4;
        const gateCenterV = (corners[0].v + corners[1].v + corners[2].v + corners[3].v) / 4;
        const distFromCenter = Math.sqrt(
            Math.pow(gateCenterU - centerU, 2) + Math.pow(gateCenterV - centerV, 2)
        );
        const maxDist = Math.sqrt(centerU * centerU + centerV * centerV);
        const centerScore = 1 - (distFromCenter / maxDist) * 0.5;

        // 3. Check if gate is roughly square (not too skewed)
        const aspectScore = this.computeAspectScore(corners);

        // Combine scores
        return sizeScore * centerScore * aspectScore;
    }

    /**
     * Compute area of quadrilateral using shoelace formula
     */
    private computeQuadArea(corners: [Point2D, Point2D, Point2D, Point2D]): number {
        let area = 0;
        for (let i = 0; i < 4; i++) {
            const j = (i + 1) % 4;
            area += corners[i].u * corners[j].v;
            area -= corners[j].u * corners[i].v;
        }
        return Math.abs(area) / 2;
    }

    /**
     * Compute how square-like the detection is
     */
    private computeAspectScore(corners: [Point2D, Point2D, Point2D, Point2D]): number {
        // Compute diagonal lengths
        const diag1 = Math.sqrt(
            Math.pow(corners[2].u - corners[0].u, 2) +
            Math.pow(corners[2].v - corners[0].v, 2)
        );
        const diag2 = Math.sqrt(
            Math.pow(corners[3].u - corners[1].u, 2) +
            Math.pow(corners[3].v - corners[1].v, 2)
        );

        // Perfect square has equal diagonals
        const ratio = Math.min(diag1, diag2) / Math.max(diag1, diag2);
        return ratio;
    }

    /**
     * Get last detections
     */
    public getLastDetections(): GateDetection[] {
        return this.lastDetections;
    }

    /**
     * Get camera reference
     */
    public getCamera(): PinholeCamera {
        return this.camera;
    }

    /**
     * Set noise level
     */
    public setNoiseLevel(noise: number): void {
        this.noiseLevel = noise;
    }
}
