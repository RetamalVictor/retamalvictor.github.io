// Ternary Matrix Multiplication Shader
// Computes Y = X @ (W_tern * scale) where W_tern ∈ {-1, 0, +1}
//
// Weight encoding (2 bits per weight):
//   00 = 0
//   01 = +1
//   10 = -1
//   11 = reserved (treated as 0)
//
// Layout: Weights stored as [K_packed, N] (transposed) for coalesced access
// where K_packed = ceil(K / 16) (16 weights per u32)

struct Params {
    B: u32,          // Batch size
    K: u32,          // Input features (unpacked)
    N: u32,          // Output features
    K_packed: u32,   // K / 16 (packed dimension in u32s)
}

@group(0) @binding(0) var<storage, read> packed_weights: array<u32>;   // [K_packed, N]
@group(0) @binding(1) var<storage, read> scales: array<f32>;           // [N]
@group(0) @binding(2) var<storage, read> input_data: array<f32>;       // [B, K]
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;// [B, N]
@group(0) @binding(4) var<uniform> params: Params;

// Branchless ternary decode: converts 2-bit code to {-1, 0, +1}
fn decode_ternary(code: u32) -> i32 {
    // code 00 -> 0, code 01 -> +1, code 10 -> -1, code 11 -> 0
    let is_pos = select(0, 1, code == 1u);
    let is_neg = select(0, 1, code == 2u);
    return i32(is_pos) - i32(is_neg);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;  // Batch index
    let n = gid.y;  // Output feature index

    // Bounds check
    if (b >= params.B || n >= params.N) {
        return;
    }

    var acc: f32 = 0.0;

    // Process 16 weights per u32
    for (var k_packed: u32 = 0u; k_packed < params.K_packed; k_packed = k_packed + 1u) {
        // Load packed weights (transposed layout: [K_packed, N])
        let packed = packed_weights[k_packed * params.N + n];

        // Unpack and accumulate 16 weights
        for (var i: u32 = 0u; i < 16u; i = i + 1u) {
            let k = k_packed * 16u + i;

            // Skip if beyond actual K dimension
            if (k >= params.K) {
                break;
            }

            // Extract 2-bit code for this weight
            let code = (packed >> (i * 2u)) & 3u;
            let w = decode_ternary(code);

            // Load input and accumulate
            let x = input_data[b * params.K + k];
            acc = acc + x * f32(w);
        }
    }

    // Apply per-channel scale
    let scale = scales[n];
    output_data[b * params.N + n] = acc * scale;
}
