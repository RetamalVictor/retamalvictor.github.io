import * as THREE from 'three';

export interface CameraConfig {
    focalLength: number;      // in mm
    sensorWidth: number;      // in pixels
    sensorHeight: number;     // in pixels
    sensorSizeMM?: number;    // physical sensor width in mm (default 36mm)
}

/**
 * Pinhole camera model for visual servoing
 * Matches the Python simulation coordinate system:
 * - Camera at origin looks toward +Z world axis
 * - X right, Y down in image, Z forward (depth)
 */
export class PinholeCamera {
    // Intrinsic parameters
    public readonly focalLengthPx: number;
    public readonly cx: number;
    public readonly cy: number;
    public readonly width: number;
    public readonly height: number;

    // Extrinsic parameters - camera position in world frame
    private position: THREE.Vector3;
    private rotationMatrix: THREE.Matrix3;

    constructor(config: CameraConfig) {
        this.width = config.sensorWidth;
        this.height = config.sensorHeight;

        // Principal point at image center
        this.cx = this.width / 2;
        this.cy = this.height / 2;

        // Convert focal length from mm to pixels
        const sensorSizeMM = config.sensorSizeMM || 36;
        this.focalLengthPx = config.focalLength * (this.width / sensorSizeMM);

        // Initialize at origin, looking along +Z
        this.position = new THREE.Vector3(0, 0, 0);
        this.rotationMatrix = new THREE.Matrix3().identity();
    }

    /**
     * Set camera pose in world coordinates
     * Position is where the camera is located
     * Rotation is applied to transform world points to camera frame
     */
    public setPose(x: number, y: number, z: number, pitch: number, yaw: number, roll: number): void {
        this.position.set(x, y, z);

        // Build rotation matrix from Euler angles (in radians)
        // Using Rodrigues-style rotation: R = Rz * Ry * Rx
        const cx = Math.cos(pitch), sx = Math.sin(pitch);
        const cy = Math.cos(yaw), sy = Math.sin(yaw);
        const cz = Math.cos(roll), sz = Math.sin(roll);

        // Combined rotation matrix
        this.rotationMatrix.set(
            cy * cz, -cy * sz, sy,
            sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy,
            -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy
        );
    }

    /**
     * Get current camera position
     */
    public getPosition(): THREE.Vector3 {
        return this.position.clone();
    }

    /**
     * Transform points from world frame to camera frame
     * World frame: X right, Y up, Z forward
     * Camera frame: X right, Y down, Z forward (standard CV convention)
     */
    private transformToCamera(worldPoints: THREE.Vector3[]): THREE.Vector3[] {
        const cameraPoints: THREE.Vector3[] = [];

        for (const wp of worldPoints) {
            // Translate: point relative to camera position
            const relative = new THREE.Vector3(
                wp.x - this.position.x,
                wp.y - this.position.y,
                wp.z - this.position.z
            );

            // Apply rotation matrix (world to camera)
            // Note: We invert Y to convert from world (Y up) to camera (Y down)
            const camPoint = new THREE.Vector3();
            camPoint.x = this.rotationMatrix.elements[0] * relative.x +
                         this.rotationMatrix.elements[3] * relative.y +
                         this.rotationMatrix.elements[6] * relative.z;
            // Invert Y: world Y up → camera Y down
            camPoint.y = -(this.rotationMatrix.elements[1] * relative.x +
                          this.rotationMatrix.elements[4] * relative.y +
                          this.rotationMatrix.elements[7] * relative.z);
            camPoint.z = this.rotationMatrix.elements[2] * relative.x +
                         this.rotationMatrix.elements[5] * relative.y +
                         this.rotationMatrix.elements[8] * relative.z;

            cameraPoints.push(camPoint);
        }

        return cameraPoints;
    }

    /**
     * Project 3D world points to 2D image coordinates
     */
    public projectPoints(worldPoints: THREE.Vector3[]): {
        imagePoints: Float32Array;
        cameraPoints: THREE.Vector3[];
        visible: boolean[];
    } {
        const cameraPoints = this.transformToCamera(worldPoints);
        const imagePoints = new Float32Array(worldPoints.length * 2);
        const visible: boolean[] = [];

        for (let i = 0; i < cameraPoints.length; i++) {
            const p = cameraPoints[i];
            const Z = p.z;

            // Point is visible if in front of camera (Z > 0)
            if (Z > 0.01) {
                // Pinhole projection (standard: camera Y down, image V down)
                // u = f * (X / Z) + cx
                // v = f * (Y / Z) + cy
                const u = this.focalLengthPx * (p.x / Z) + this.cx;
                const v = this.focalLengthPx * (p.y / Z) + this.cy;

                imagePoints[i * 2] = u;
                imagePoints[i * 2 + 1] = v;

                // Check bounds
                visible.push(u >= 0 && u < this.width && v >= 0 && v < this.height);
            } else {
                imagePoints[i * 2] = -1;
                imagePoints[i * 2 + 1] = -1;
                visible.push(false);
            }
        }

        return { imagePoints, cameraPoints, visible };
    }

    /**
     * Get depths (Z in camera frame) for points
     */
    public getDepths(worldPoints: THREE.Vector3[]): Float32Array {
        const cameraPoints = this.transformToCamera(worldPoints);
        return new Float32Array(cameraPoints.map(p => p.z));
    }
}
