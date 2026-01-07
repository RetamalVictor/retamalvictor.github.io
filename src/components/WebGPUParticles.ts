import * as THREE from 'three/webgpu';
import {
    Fn,
    If,
    uniform,
    float,
    vec3,
    vec4,
    storage,
    instanceIndex,
    sin,
    cos,
    length,
    mix,
    smoothstep
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

    // Interaction
    private mousePosition = uniform(vec3(0, 0, 0));
    private mouseStrength = uniform(0);
    private timeUniform = uniform(0);

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
            antialias: true,
            alpha: true
        });

        await this.renderer.init();

        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
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

        // Initialize particles in a sphere distribution
        const initPositions = new Float32Array(count * 3);
        const initVelocities = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            // Spherical distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = Math.pow(Math.random(), 0.5) * 8;

            initPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
            initPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            initPositions[i * 3 + 2] = r * Math.cos(phi);

            // Small random velocities
            initVelocities[i * 3] = (Math.random() - 0.5) * 0.02;
            initVelocities[i * 3 + 1] = (Math.random() - 0.5) * 0.02;
            initVelocities[i * 3 + 2] = (Math.random() - 0.5) * 0.02;
        }

        // Create storage buffers
        this.positionBuffer = new THREE.StorageInstancedBufferAttribute(initPositions, 3);
        this.velocityBuffer = new THREE.StorageInstancedBufferAttribute(initVelocities, 3);

        // Create TSL storage references
        this.positions = storage(this.positionBuffer, 'vec3', count);
        this.velocities = storage(this.velocityBuffer, 'vec3', count);

        // Compute shader for particle physics
        this.computeUpdate = Fn(() => {
            const position = this.positions.element(instanceIndex);
            const velocity = this.velocities.element(instanceIndex);

            // Get current position
            const pos = position.toVar();
            const vel = velocity.toVar();

            // Orbital motion around center
            const toCenter = pos.mul(-1).normalize();
            const tangent = vec3(
                toCenter.z.mul(-1),
                float(0),
                toCenter.x
            ).normalize();

            // Add swirl force
            const dist = length(pos);
            const orbitForce = tangent.mul(0.0003).div(dist.add(0.5));
            vel.addAssign(orbitForce);

            // Mouse interaction
            const toMouse = this.mousePosition.sub(pos);
            const mouseDist = length(toMouse);
            const mouseForce = toMouse.normalize().mul(
                this.mouseStrength.mul(0.05).div(mouseDist.add(1))
            );
            vel.addAssign(mouseForce);

            // Gentle pull toward center (keeps particles from flying away)
            const centerPull = pos.mul(-0.0001);
            vel.addAssign(centerPull);

            // Add some noise/turbulence based on time
            const noiseOffset = vec3(
                sin(this.timeUniform.add(float(instanceIndex).mul(0.001))).mul(0.0005),
                cos(this.timeUniform.mul(1.3).add(float(instanceIndex).mul(0.0013))).mul(0.0005),
                sin(this.timeUniform.mul(0.7).add(float(instanceIndex).mul(0.0017))).mul(0.0005)
            );
            vel.addAssign(noiseOffset);

            // Damping
            vel.mulAssign(0.995);

            // Update position
            pos.addAssign(vel);

            // Soft boundary (push back if too far)
            const maxDist = float(10);
            If(dist.greaterThan(maxDist), () => {
                pos.assign(pos.normalize().mul(maxDist));
                vel.mulAssign(0.5);
            });

            // Write back
            position.assign(pos);
            velocity.assign(vel);
        })().compute(count);

        // Create particle material with color gradient based on distance
        const material = new THREE.SpriteNodeMaterial({
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        // Position from compute buffer
        material.positionNode = this.positions.toAttribute();

        // Color based on distance from center and velocity
        const pos = this.positions.element(instanceIndex);
        const vel = this.velocities.element(instanceIndex);
        const dist = length(pos);
        const speed = length(vel);

        // Cyan to purple gradient based on distance, brighter with speed
        const cyan = vec3(0, 0.83, 1);      // #00d4ff
        const purple = vec3(0.66, 0.33, 0.97); // #a855f7
        const t = smoothstep(float(0), float(8), dist);
        const baseColor = mix(cyan, purple, t);
        const brightness = speed.mul(50).add(0.3).min(1);

        material.colorNode = vec4(baseColor.mul(brightness), brightness.mul(0.6));

        // Particle size - smaller when further, larger when faster
        const baseSize = float(0.08);
        const sizeVariation = speed.mul(2).add(1);
        material.scaleNode = baseSize.mul(sizeVariation);

        // Create instanced mesh
        const geometry = new THREE.PlaneGeometry(1, 1);
        const particles = new THREE.InstancedMesh(geometry, material, count);

        this.scene.add(particles);
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

            // Convert to 3D position
            const vector = new THREE.Vector3(x * 10, y * 6, 0);

            this.mousePosition.value.set(vector.x, vector.y, vector.z);
            this.mouseStrength.value = 1;
        });

        // Touch support
        this.container.addEventListener('touchmove', (event) => {
            event.preventDefault();
            const touch = event.touches[0];
            const rect = this.container.getBoundingClientRect();
            const x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;

            const vector = new THREE.Vector3(x * 10, y * 6, 0);
            this.mousePosition.value.set(vector.x, vector.y, vector.z);
            this.mouseStrength.value = 1;
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

        // Slowly rotate camera for subtle motion
        const t = this.timeUniform.value * 0.1;
        this.camera.position.x = Math.sin(t) * 2;
        this.camera.position.y = Math.cos(t * 0.7) * 1;
        this.camera.lookAt(0, 0, 0);

        // Decay mouse strength
        if (this.mouseStrength.value > 0) {
            this.mouseStrength.value *= 0.98;
        }

        // Run compute shader
        this.renderer.compute(this.computeUpdate);

        // Render
        this.renderer.renderAsync(this.scene, this.camera);
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
