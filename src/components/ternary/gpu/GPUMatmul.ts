/**
 * GPUMatmul - WebGPU-accelerated ternary matrix multiplication
 *
 * Provides GPU acceleration for the ternary matmul operation, which is
 * the main computational bottleneck in transformer inference.
 */

// Import shader as raw string (Vite handles this with ?raw)
import shaderSource from './shaders/ternary_matmul.wgsl?raw';

/** Ternary layer weight data (matches TransformerCPUEngine) */
export interface TernaryLayer {
    weightsPacked: Uint8Array;
    scales: Float32Array;
    outFeatures: number;
    inFeatures: number;
}

/** GPU buffer allocation for a single layer */
interface LayerBuffers {
    weightsPacked: GPUBuffer;
    scales: GPUBuffer;
    outFeatures: number;
    inFeatures: number;
    kBytes: number;
}

// Uniforms structure: M (seqLen), N (outFeatures), K (inFeatures), K_bytes
// Passed as Uint32Array to GPU

export class GPUMatmul {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private bindGroupLayout: GPUBindGroupLayout;

    // Pre-uploaded weight buffers per layer
    private layerBuffers: Map<string, LayerBuffers> = new Map();

    // Reusable buffers (sized for largest layer)
    private inputBuffer: GPUBuffer | null = null;
    private outputBuffer: GPUBuffer | null = null;
    private stagingBuffer: GPUBuffer | null = null;
    private uniformBuffer: GPUBuffer;

    // Track max sizes for buffer allocation
    private maxInputSize = 0;
    private maxOutputSize = 0;

    // Batching state
    private commandEncoder: GPUCommandEncoder | null = null;
    private batchResults: Map<string, { buffer: GPUBuffer; size: number }> = new Map();
    private batchTempBuffers: GPUBuffer[] = [];  // Track temp buffers for cleanup

    private constructor(device: GPUDevice, pipeline: GPUComputePipeline, bindGroupLayout: GPUBindGroupLayout) {
        this.device = device;
        this.pipeline = pipeline;
        this.bindGroupLayout = bindGroupLayout;

        // Create uniform buffer (16 bytes for 4 u32s)
        this.uniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * Create a GPUMatmul instance. Returns null if WebGPU is unavailable.
     */
    static async create(): Promise<GPUMatmul | null> {
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            return null;
        }

        try {
            const adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance',
            });

            if (!adapter) {
                return null;
            }

            // Request device with limits capped to what adapter supports (for mobile compatibility)
            const limits = adapter.limits;
            console.log(`[GPUMatmul] Adapter limits: storage=${Math.round(limits.maxStorageBufferBindingSize / 1024 / 1024)}MB, buffer=${Math.round(limits.maxBufferSize / 1024 / 1024)}MB`);

            const desiredStorageSize = 128 * 1024 * 1024; // 128MB
            const desiredBufferSize = 256 * 1024 * 1024; // 256MB

            console.log('[GPUMatmul] Requesting device...');
            const device = await adapter.requestDevice({
                requiredLimits: {
                    maxStorageBufferBindingSize: Math.min(desiredStorageSize, limits.maxStorageBufferBindingSize),
                    maxBufferSize: Math.min(desiredBufferSize, limits.maxBufferSize),
                },
            });
            console.log('[GPUMatmul] Device created successfully');

            // Handle device loss
            device.lost.then((info) => {
                console.error('[GPUMatmul] Device lost:', info.message);
            });

            // Compile shader
            console.log('[GPUMatmul] Compiling shader...');
            const shaderModule = device.createShaderModule({
                code: shaderSource,
            });

            // Check for compilation errors
            const compilationInfo = await shaderModule.getCompilationInfo();
            for (const message of compilationInfo.messages) {
                console.log(`[GPUMatmul] Shader ${message.type}:`, message.message);
                if (message.type === 'error') {
                    console.error('[GPUMatmul] Shader compilation failed');
                    return null;
                }
            }
            console.log('[GPUMatmul] Shader compiled successfully');

            // Create bind group layout
            const bindGroupLayout = device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ],
            });

            // Create pipeline
            const pipeline = device.createComputePipeline({
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout],
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: 'main',
                },
            });

            console.log('[GPUMatmul] WebGPU initialized successfully');
            return new GPUMatmul(device, pipeline, bindGroupLayout);
        } catch (error) {
            console.error('[GPUMatmul] Failed to initialize WebGPU');
            console.error('[GPUMatmul] Error:', error instanceof Error ? error.message : error);
            if (error instanceof Error && error.stack) {
                console.error('[GPUMatmul] Stack:', error.stack);
            }
            return null;
        }
    }

    /**
     * Upload a layer's weights to GPU. Call once per layer at init time.
     *
     * IMPORTANT: Weights must be row-aligned to 4 bytes for u32 access in shader.
     * We repack weights here if kBytes isn't a multiple of 4.
     */
    uploadLayer(name: string, layer: TernaryLayer): void {
        const { weightsPacked, scales, outFeatures, inFeatures } = layer;
        const kBytes = Math.ceil(inFeatures / 4);

        // Align kBytes to multiple of 4 for u32 access in shader
        const kBytesAligned = Math.ceil(kBytes / 4) * 4;

        let weightsData: Uint8Array;
        if (kBytes === kBytesAligned) {
            // Already aligned, use as-is
            weightsData = weightsPacked;
        } else {
            // Repack weights with row padding for alignment
            weightsData = new Uint8Array(outFeatures * kBytesAligned);
            for (let row = 0; row < outFeatures; row++) {
                const srcOffset = row * kBytes;
                const dstOffset = row * kBytesAligned;
                weightsData.set(
                    weightsPacked.subarray(srcOffset, srcOffset + kBytes),
                    dstOffset
                );
                // Padding bytes are already 0 (Uint8Array default)
            }
        }

        // Create weight buffer
        const weightsBuffer = this.device.createBuffer({
            size: weightsData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(weightsBuffer, 0, weightsData as BufferSource);

        // Create scales buffer
        const scalesBuffer = this.device.createBuffer({
            size: scales.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(scalesBuffer, 0, scales as BufferSource);

        this.layerBuffers.set(name, {
            weightsPacked: weightsBuffer,
            scales: scalesBuffer,
            outFeatures,
            inFeatures,
            kBytes: kBytesAligned,  // Store aligned value for shader
        });

        // Track max sizes for dynamic buffer allocation
        this.maxInputSize = Math.max(this.maxInputSize, inFeatures);
        this.maxOutputSize = Math.max(this.maxOutputSize, outFeatures);
    }

    /**
     * Ensure activation buffers are large enough for the given dimensions.
     */
    private ensureBuffers(seqLen: number): void {
        const requiredInputSize = seqLen * this.maxInputSize * 4;
        const requiredOutputSize = seqLen * this.maxOutputSize * 4;

        // Input buffer
        if (!this.inputBuffer || this.inputBuffer.size < requiredInputSize) {
            this.inputBuffer?.destroy();
            this.inputBuffer = this.device.createBuffer({
                size: requiredInputSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }

        // Output buffer
        if (!this.outputBuffer || this.outputBuffer.size < requiredOutputSize) {
            this.outputBuffer?.destroy();
            this.outputBuffer = this.device.createBuffer({
                size: requiredOutputSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
        }

        // Staging buffer for readback
        if (!this.stagingBuffer || this.stagingBuffer.size < requiredOutputSize) {
            this.stagingBuffer?.destroy();
            this.stagingBuffer = this.device.createBuffer({
                size: requiredOutputSize,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
        }
    }

    /**
     * Execute a single matmul and return result immediately.
     * For better performance with multiple matmuls, use the batch API.
     */
    async matmul(
        input: Float32Array,
        layerName: string,
        seqLen: number
    ): Promise<Float32Array> {
        const layer = this.layerBuffers.get(layerName);
        if (!layer) {
            throw new Error(`[GPUMatmul] Layer not found: ${layerName}`);
        }

        this.ensureBuffers(seqLen);

        // Upload input
        this.device.queue.writeBuffer(this.inputBuffer!, 0, input as BufferSource);

        // Update uniforms
        const uniforms = new Uint32Array([
            seqLen,           // M
            layer.outFeatures, // N
            layer.inFeatures,  // K
            layer.kBytes,      // K_bytes
        ]);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms as BufferSource);

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.inputBuffer! } },
                { binding: 1, resource: { buffer: layer.weightsPacked } },
                { binding: 2, resource: { buffer: layer.scales } },
                { binding: 3, resource: { buffer: this.outputBuffer! } },
                { binding: 4, resource: { buffer: this.uniformBuffer } },
            ],
        });

        // Dispatch
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);

        // Workgroup size is 64x4, so dispatch ceil(N/64) x ceil(M/4)
        const workgroupsX = Math.ceil(layer.outFeatures / 64);
        const workgroupsY = Math.ceil(seqLen / 4);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        pass.end();

        // Copy to staging buffer
        const outputSize = seqLen * layer.outFeatures * 4;
        encoder.copyBufferToBuffer(this.outputBuffer!, 0, this.stagingBuffer!, 0, outputSize);

        // Submit and wait
        this.device.queue.submit([encoder.finish()]);

        // Read back results
        await this.stagingBuffer!.mapAsync(GPUMapMode.READ);
        const resultData = new Float32Array(this.stagingBuffer!.getMappedRange(0, outputSize).slice(0));
        this.stagingBuffer!.unmap();

        return resultData;
    }

    /**
     * Begin recording a batch of matmuls.
     * Reduces GPU dispatch overhead by combining multiple operations.
     */
    beginBatch(): void {
        if (this.commandEncoder) {
            console.warn('[GPUMatmul] Batch already in progress');
            return;
        }
        this.commandEncoder = this.device.createCommandEncoder();
        this.batchResults.clear();
        this.batchTempBuffers = [];  // Clear temp buffer tracking
    }

    /**
     * Queue a matmul operation to the current batch.
     * Must be called between beginBatch() and endBatch().
     *
     * Note: For simplicity, each queued matmul gets its own output buffer.
     * This allows results to be read back independently.
     */
    queueMatmul(
        input: Float32Array,
        layerName: string,
        seqLen: number
    ): void {
        if (!this.commandEncoder) {
            throw new Error('[GPUMatmul] No batch in progress. Call beginBatch() first.');
        }

        const layer = this.layerBuffers.get(layerName);
        if (!layer) {
            throw new Error(`[GPUMatmul] Layer not found: ${layerName}`);
        }

        const inputSize = seqLen * layer.inFeatures * 4;
        const outputSize = seqLen * layer.outFeatures * 4;

        // Create per-operation buffers
        const inputBuffer = this.device.createBuffer({
            size: inputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(inputBuffer, 0, input as BufferSource);
        this.batchTempBuffers.push(inputBuffer);  // Track for cleanup

        const outputBuffer = this.device.createBuffer({
            size: outputSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        // outputBuffer tracked in batchResults, will be cleaned up there

        // Create uniform buffer for this operation
        const uniformBuffer = this.device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const uniforms = new Uint32Array([
            seqLen,
            layer.outFeatures,
            layer.inFeatures,
            layer.kBytes,
        ]);
        this.device.queue.writeBuffer(uniformBuffer, 0, uniforms as BufferSource);
        this.batchTempBuffers.push(uniformBuffer);  // Track for cleanup

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: inputBuffer } },
                { binding: 1, resource: { buffer: layer.weightsPacked } },
                { binding: 2, resource: { buffer: layer.scales } },
                { binding: 3, resource: { buffer: outputBuffer } },
                { binding: 4, resource: { buffer: uniformBuffer } },
            ],
        });

        // Record dispatch
        const pass = this.commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        const workgroupsX = Math.ceil(layer.outFeatures / 64);
        const workgroupsY = Math.ceil(seqLen / 4);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        pass.end();

        // Store for readback
        this.batchResults.set(layerName, { buffer: outputBuffer, size: outputSize });
    }

    /**
     * Submit all queued matmuls and return results.
     */
    async endBatch(): Promise<Map<string, Float32Array>> {
        if (!this.commandEncoder) {
            throw new Error('[GPUMatmul] No batch in progress');
        }

        // Create staging buffers and copy commands
        const stagingBuffers: Map<string, { staging: GPUBuffer; size: number }> = new Map();

        for (const [name, { buffer, size }] of this.batchResults) {
            const staging = this.device.createBuffer({
                size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            this.commandEncoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
            stagingBuffers.set(name, { staging, size });
        }

        // Submit
        this.device.queue.submit([this.commandEncoder.finish()]);
        this.commandEncoder = null;

        // Read back all results
        const results: Map<string, Float32Array> = new Map();

        for (const [name, { staging, size }] of stagingBuffers) {
            await staging.mapAsync(GPUMapMode.READ);
            const data = new Float32Array(staging.getMappedRange(0, size).slice(0));
            staging.unmap();
            staging.destroy();
            results.set(name, data);
        }

        // Cleanup batch output buffers
        for (const { buffer } of this.batchResults.values()) {
            buffer.destroy();
        }
        this.batchResults.clear();

        // Cleanup temporary buffers (input, uniform)
        for (const buffer of this.batchTempBuffers) {
            buffer.destroy();
        }
        this.batchTempBuffers = [];

        return results;
    }

    /**
     * Check if GPU acceleration is available and initialized.
     */
    isAvailable(): boolean {
        return true;
    }

    /**
     * Get memory usage statistics.
     */
    getMemoryStats(): { layerBuffersKB: number; activationBuffersKB: number } {
        let layerTotal = 0;
        for (const layer of this.layerBuffers.values()) {
            layerTotal += layer.weightsPacked.size + layer.scales.size;
        }

        const activationTotal =
            (this.inputBuffer?.size ?? 0) +
            (this.outputBuffer?.size ?? 0) +
            (this.stagingBuffer?.size ?? 0);

        return {
            layerBuffersKB: Math.round(layerTotal / 1024),
            activationBuffersKB: Math.round(activationTotal / 1024),
        };
    }

    /**
     * Release all GPU resources.
     */
    destroy(): void {
        for (const layer of this.layerBuffers.values()) {
            layer.weightsPacked.destroy();
            layer.scales.destroy();
        }
        this.layerBuffers.clear();

        this.inputBuffer?.destroy();
        this.outputBuffer?.destroy();
        this.stagingBuffer?.destroy();
        this.uniformBuffer.destroy();

        this.device.destroy();
    }
}
