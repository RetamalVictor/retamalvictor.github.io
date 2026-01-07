// Embedding Lookup Shader
// Looks up token embeddings and flattens for MLP input
//
// Input: token indices [context_length]
// Output: flattened embeddings [context_length * embed_dim]

struct Params {
    contextLength: u32,
    embedDim: u32,
    vocabSize: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> embedding_table: array<f32>;  // [vocab_size, embed_dim]
@group(0) @binding(1) var<storage, read> token_indices: array<u32>;    // [context_length]
@group(0) @binding(2) var<storage, read_write> output_data: array<f32>;// [context_length * embed_dim]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total_elements = params.contextLength * params.embedDim;

    if (idx >= total_elements) {
        return;
    }

    // Compute which token and which embedding dimension
    let pos = idx / params.embedDim;       // Token position in context
    let dim = idx % params.embedDim;       // Embedding dimension

    // Get token index (clamped to vocab size)
    let token_idx = min(token_indices[pos], params.vocabSize - 1u);

    // Look up embedding
    let embed_value = embedding_table[token_idx * params.embedDim + dim];

    // Write to output (flattened)
    output_data[idx] = embed_value;
}
