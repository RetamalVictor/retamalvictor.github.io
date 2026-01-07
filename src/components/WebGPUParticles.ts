import * as THREE from 'three/webgpu';
import {
    Fn,
    uniform,
    float,
    vec3,
    vec4,
    storage,
    instanceIndex,
    length,
    mix,
    smoothstep,
    max,
    sin,
    cos,
    fract,
    floor
} from 'three/tsl';

interface ParticleConfig {
    containerId: string;
    particleCount?: number;
    backgroundColor?: number;
}

export class WebGPUParticles {
    private container: HTMLElement;
    private renderer!: THREE.WebGPURenderer;
    private scene!: THREE.Scene;
    private camera!: THREE.PerspectiveCamera;
    private animationId: number | null = null;

    // Particle system
    private particleCount: number;
    private positionBuffer!: THREE.StorageInstancedBufferAttribute;
    private velocityBuffer!: THREE.StorageInstancedBufferAttribute;
    private positions: any;
    private velocities: any;
    private computeUpdate: any;
    private computeReset: any;  // GPU-side reset shader

    // Interaction uniforms
    private mousePosition = uniform(vec3(0, 0, 0));
    private mouseStrength = uniform(0);
    private timeUniform = uniform(0);
    private resetSeed = uniform(0);  // Seed for GPU reset randomization

    // Config
    private backgroundColor: number;

    constructor(config: ParticleConfig) {
        this.container = document.getElementById(config.containerId)!;
        this.particleCount = config.particleCount || 50000;
        this.backgroundColor = config.backgroundColor || 0x12121a;

        if (!this.container) {
            throw new Error(`Container with id ${config.containerId} not found`);
        }

        this.init();
    }

    private async init(): Promise<void> {
        // Clear container
        this.container.innerHTML = '';

        try {
            await this.setupRenderer();
            this.setupScene();
            this.setupCamera();
            this.createParticleSystem();
            this.setupInteraction();
            this.setupResize();
            this.animate();
        } catch (error) {
            console.warn('WebGPU initialization failed, showing fallback:', error);
            this.showFallback();
        }
    }

    private async setupRenderer(): Promise<void> {
        this.renderer = new THREE.WebGPURenderer({
            antialias: false, // Disabled for performance - particles don't need AA
            alpha: false,     // Not using transparency for background
            powerPreference: 'high-performance'
        });

        await this.renderer.init();

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5)); // Cap lower for performance
        this.renderer.setClearColor(this.backgroundColor, 1);

        this.container.appendChild(this.renderer.domElement);
    }

    private setupScene(): void {
        this.scene = new THREE.Scene();
    }

    private setupCamera(): void {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
        this.camera.position.set(0, 0, 15);
        this.camera.lookAt(0, 0, 0);
    }

    private createParticleSystem(): void {
        const count = this.particleCount;

        // Initialize with zeros - GPU will set initial values
        const initPositions = new Float32Array(count * 3);
        const initVelocities = new Float32Array(count * 3);

        // Create storage buffers
        this.positionBuffer = new THREE.StorageInstancedBufferAttribute(initPositions, 3);
        this.velocityBuffer = new THREE.StorageInstancedBufferAttribute(initVelocities, 3);

        // Create TSL storage references
        this.positions = storage(this.positionBuffer, 'vec3', count);
        this.velocities = storage(this.velocityBuffer, 'vec3', count);

        // ============================================================
        // GPU-SIDE RESET COMPUTE SHADER
        // ============================================================
        this.computeReset = Fn(() => {
            const position = this.positions.element(instanceIndex);
            const velocity = this.velocities.element(instanceIndex);

            // Better hash for uncorrelated random values
            // Use different prime multipliers for each random value
            const idx = float(instanceIndex);
            const seed = this.resetSeed;

            // Generate 3 independent random values using varied hash inputs
            const rand1 = fract(sin(idx.mul(12.9898).add(seed.mul(78.233))).mul(43758.5453));
            const rand2 = fract(sin(idx.mul(39.3468).add(seed.mul(11.135))).mul(28461.2731));
            const rand3 = fract(sin(idx.mul(73.1568).add(seed.mul(44.847))).mul(23164.7532));

            // Spherical distribution
            const theta = rand1.mul(6.28318);  // 2 * PI
            const phi = (rand2.mul(2.0).sub(1.0)).acos();  // acos(2*rand - 1)
            const r = rand3.sqrt().mul(8.0);  // sqrt for uniform volume distribution

            // Convert to Cartesian
            const sinPhi = sin(phi);
            const cosPhi = cos(phi);
            const sinTheta = sin(theta);
            const cosTheta = cos(theta);

            const newPos = vec3(
                r.mul(sinPhi).mul(cosTheta),
                r.mul(sinPhi).mul(sinTheta),
                r.mul(cosPhi)
            );

            // Small random velocities with different hash
            const velRand1 = fract(sin(idx.mul(91.2345).add(seed.mul(23.456))).mul(54321.9876));
            const velRand2 = fract(sin(idx.mul(47.8912).add(seed.mul(67.891))).mul(12345.6789));
            const velRand3 = fract(sin(idx.mul(23.4567).add(seed.mul(89.012))).mul(98765.4321));
            const newVel = vec3(
                velRand1.sub(0.5).mul(0.02),
                velRand2.sub(0.5).mul(0.02),
                velRand3.sub(0.5).mul(0.02)
            );

            position.assign(newPos);
            velocity.assign(newVel);
        })().compute(count);

        // ============================================================
        // MAIN UPDATE COMPUTE SHADER (with hash-based noise)
        // ============================================================
        const maxSpeed = float(0.3);

        this.computeUpdate = Fn(() => {
            const position = this.positions.element(instanceIndex);
            const velocity = this.velocities.element(instanceIndex);

            const pos = position.toVar();
            const vel = velocity.toVar();

            // === OPTIMIZED HASH-BASED NOISE ===
            // Create time-varying seed for organic movement
            const idx = float(instanceIndex);
            const t = floor(this.timeUniform.mul(3.0));  // Changes every ~0.33s

            // 3 independent noise values using varied hash
            const noiseX = fract(sin(idx.mul(12.9898).add(t.mul(78.233))).mul(43758.5453));
            const noiseY = fract(sin(idx.mul(39.346).add(t.mul(11.135))).mul(28461.273));
            const noiseZ = fract(sin(idx.mul(73.156).add(t.mul(44.847))).mul(23164.753));

            const noise = vec3(noiseX, noiseY, noiseZ).sub(0.5).mul(0.04);
            vel.addAssign(noise);

            // === MOUSE ATTRACTION ===
            const toMouse = this.mousePosition.sub(pos);
            const mouseDist = length(toMouse);
            const mouseDir = toMouse.div(max(mouseDist, float(0.1)));

            // Attract to mouse, strength falls off with distance
            const attraction = mouseDir.mul(this.mouseStrength.mul(0.15).div(mouseDist.add(1.0)));
            vel.addAssign(attraction);

            // === SOFT REPULSION FROM MOUSE CENTER ===
            const repelDist = float(1.5);
            const repelStrength = max(repelDist.sub(mouseDist), float(0)).mul(0.1);
            vel.addAssign(mouseDir.negate().mul(this.mouseStrength.mul(repelStrength)));

            // === DAMPING ===
            vel.mulAssign(0.95);

            // Limit speed
            const speed = length(vel);
            const limitedVel = vel.div(max(speed, maxSpeed)).mul(maxSpeed.min(speed));
            vel.assign(limitedVel);

            // Update position
            pos.addAssign(vel);

            // Soft boundary
            const boundary = float(10);
            const boundaryDist = length(pos);
            const boundaryForce = pos.normalize().mul(
                max(boundaryDist.sub(boundary), float(0)).mul(-0.15)
            );
            pos.addAssign(boundaryForce);

            // Write back
            position.assign(pos);
            velocity.assign(vel);
        })().compute(count);

        // ============================================================
        // OPTIMIZED MATERIAL (cached attribute reads)
        // ============================================================
        const material = new THREE.SpriteNodeMaterial({
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        // Single attribute reference for position (avoids redundant reads)
        const posAttr = this.positions.toAttribute();
        const velAttr = this.velocities.toAttribute();

        material.positionNode = posAttr;

        // Use attributes directly for color calculation
        const dist = length(posAttr);
        const speed = length(velAttr);

        // Cyan to purple gradient based on distance, brighter with speed
        const cyan = vec3(0, 0.83, 1);
        const purple = vec3(0.66, 0.33, 0.97);
        const t = smoothstep(float(0), float(8), dist);
        const baseColor = mix(cyan, purple, t);
        const brightness = speed.mul(50).add(0.3).min(1);

        material.colorNode = vec4(baseColor.mul(brightness), brightness.mul(0.6));

        // Particle size
        const baseSize = float(0.08);
        const sizeVariation = speed.mul(2).add(1);
        material.scaleNode = baseSize.mul(sizeVariation);

        // Create instanced mesh
        const geometry = new THREE.PlaneGeometry(1, 1);
        const particles = new THREE.InstancedMesh(geometry, material, count);

        this.scene.add(particles);

        // Run initial reset on GPU
        this.renderer.compute(this.computeReset);
    }

    private setupInteraction(): void {
        let isMouseInContainer = false;

        this.container.addEventListener('mouseenter', () => {
            isMouseInContainer = true;
        });

        this.container.addEventListener('mouseleave', () => {
            isMouseInContainer = false;
            this.mouseStrength.value = 0;
        });

        this.container.addEventListener('mousemove', (event) => {
            if (!isMouseInContainer) return;

            const rect = this.container.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            // Convert to 3D position - match camera view
            const vector = new THREE.Vector3(x * 12, y * 8, 0);

            this.mousePosition.value.set(vector.x, vector.y, vector.z);
            this.mouseStrength.value = 2;  // Stronger initial value
        });

        // Touch support
        this.container.addEventListener('touchmove', (event) => {
            event.preventDefault();
            const touch = event.touches[0];
            const rect = this.container.getBoundingClientRect();
            const x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;

            const vector = new THREE.Vector3(x * 12, y * 8, 0);
            this.mousePosition.value.set(vector.x, vector.y, vector.z);
            this.mouseStrength.value = 2;
        }, { passive: false });

        this.container.addEventListener('touchend', () => {
            this.mouseStrength.value = 0;
        });
    }

    private setupResize(): void {
        const resizeObserver = new ResizeObserver(() => {
            this.handleResize();
        });
        resizeObserver.observe(this.container);
    }

    private handleResize(): void {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    private animate = (): void => {
        this.animationId = requestAnimationFrame(this.animate);

        // Update time
        this.timeUniform.value += 0.016;

        // Decay mouse strength slowly to maintain attraction
        if (this.mouseStrength.value > 0.01) {
            this.mouseStrength.value *= 0.98;
        } else {
            this.mouseStrength.value = 0;
        }

        // Run compute shader and render
        this.renderer.compute(this.computeUpdate);
        this.renderer.render(this.scene, this.camera);
    };

    private showFallback(): void {
        // Show a simple CSS animation fallback
        this.container.innerHTML = `
            <div class="w-full h-full flex items-center justify-center relative overflow-hidden">
                <div class="absolute inset-0 bg-gradient-to-br from-dark-surface to-dark-bg"></div>
                <div class="relative text-center text-gray-500">
                    <div class="w-16 h-16 mx-auto mb-4 rounded-full bg-accent-cyan/20 animate-pulse"></div>
                    <p class="text-sm">Interactive 3D</p>
                </div>
                <div class="particle-field absolute inset-0 pointer-events-none">
                    ${Array.from({length: 50}, () => `
                        <div class="absolute w-1 h-1 bg-accent-cyan/30 rounded-full animate-float"
                             style="left: ${Math.random() * 100}%; top: ${Math.random() * 100}%;
                                    animation-delay: ${Math.random() * 5}s;
                                    animation-duration: ${3 + Math.random() * 4}s;"></div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    public reset(): void {
        // GPU-side reset - instant even with 100k+ particles
        // Change seed to get different random distribution each reset
        this.resetSeed.value = Math.random() * 1000;

        // Run reset compute shader on GPU
        this.renderer.compute(this.computeReset);

        // Reset time
        this.timeUniform.value = 0;
    }

    public destroy(): void {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }

        if (this.renderer) {
            this.renderer.dispose();
        }

        if (this.container && this.renderer?.domElement?.parentNode === this.container) {
            this.container.removeChild(this.renderer.domElement);
        }
    }
}
