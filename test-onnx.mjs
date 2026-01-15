#!/usr/bin/env node
/**
 * Test ONNX Runtime Web loading in Node.js
 * Run: node test-onnx.mjs
 */

import * as ort from 'onnxruntime-web';

const MODEL_PATH = './public/assets/models/depth/depth_pretrained.onnx';

async function testOnnx() {
    console.log('Testing ONNX Runtime Web...\n');

    // Configure for Node.js
    ort.env.wasm.numThreads = 1;

    console.log('1. Creating inference session...');
    try {
        const session = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log('   ✓ Session created successfully\n');

        // Get model info
        console.log('2. Model info:');
        console.log(`   Inputs: ${session.inputNames.join(', ')}`);
        console.log(`   Outputs: ${session.outputNames.join(', ')}\n`);

        // Create dummy input
        console.log('3. Running inference with dummy input...');
        const inputSize = 384;
        const dummyData = new Float32Array(3 * inputSize * inputSize).fill(0.5);
        const inputTensor = new ort.Tensor('float32', dummyData, [1, 3, inputSize, inputSize]);

        const startTime = performance.now();
        const results = await session.run({ input: inputTensor });
        const latency = performance.now() - startTime;

        const outputTensor = results['depth'];
        console.log(`   ✓ Inference completed in ${latency.toFixed(1)}ms`);
        console.log(`   Output shape: [${outputTensor.dims.join(', ')}]`);
        console.log(`   Output type: ${outputTensor.type}\n`);

        // Check output values
        const data = outputTensor.data;
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < data.length; i++) {
            if (data[i] < min) min = data[i];
            if (data[i] > max) max = data[i];
        }
        console.log(`   Depth range: [${min.toFixed(4)}, ${max.toFixed(4)}]\n`);

        console.log('✓ All tests passed!');
        return true;

    } catch (error) {
        console.error('   ✗ Error:', error.message);
        console.error('\nFull error:', error);
        return false;
    }
}

testOnnx().then(success => {
    process.exit(success ? 0 : 1);
});
