// Softmax Shader for Logits
// Computes softmax over the vocabulary dimension
//
// For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//
// This is a simple single-pass implementation suitable for small vocab sizes (<=256)

struct Params {
    vocabSize: u32,
    temperature: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;           // [vocab_size]
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;      // [vocab_size]
@group(0) @binding(2) var<uniform> params: Params;

// Workgroup shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let vocab_size = params.vocabSize;
    let temp = params.temperature;

    // Initialize shared memory
    shared_max[tid] = -1e10;
    shared_sum[tid] = 0.0;

    // Each thread handles one element (vocab_size <= 256)
    if (tid < vocab_size) {
        shared_max[tid] = logits[tid] / temp;
    }

    workgroupBarrier();

    // Parallel reduction to find max (simple for small vocab)
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Compute exp(x - max) and partial sum
    if (tid < vocab_size) {
        let exp_val = exp(logits[tid] / temp - max_val);
        shared_sum[tid] = exp_val;
        probs[tid] = exp_val;  // Temporarily store exp values
    }

    workgroupBarrier();

    // Parallel reduction to find sum
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    let sum_val = shared_sum[0];
    workgroupBarrier();

    // Normalize
    if (tid < vocab_size) {
        probs[tid] = probs[tid] / sum_val;
    }
}
