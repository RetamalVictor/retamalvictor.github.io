/**
 * Loading and parsing the exported policy weights.
 *
 * The blob is a flat float32 array; the manifest says where each tensor starts.
 * Parsing is separated from fetching so the parity test can run under node with
 * bytes read from disk.
 */

import type { DemoManifest, Model, ModelManifest, Tensor } from './types';

/**
 * Build a model from a manifest entry and the bytes of its blob.
 *
 * `Float32Array` reads in the platform's byte order. Every platform that runs a
 * browser is little-endian, which is what the exporter writes, so a view is
 * safe and avoids copying a quarter of a megabyte.
 */
export function parseModel(manifest: ModelManifest, buffer: ArrayBuffer): Model {
    const expected = manifest.parameters * 4;
    if (buffer.byteLength !== expected) {
        throw new Error(
            `${manifest.file}: expected ${expected} bytes for ${manifest.parameters} ` +
            `parameters, got ${buffer.byteLength}`
        );
    }

    const all = new Float32Array(buffer);
    const tensors: Record<string, Tensor> = {};
    for (const entry of manifest.tensors) {
        const size = entry.shape.reduce((a, b) => a * b, 1);
        if (size !== entry.count) {
            throw new Error(`${entry.name}: shape ${entry.shape} does not match count ${entry.count}`);
        }
        tensors[entry.name] = {
            shape: entry.shape,
            data: all.subarray(entry.offset, entry.offset + entry.count),
        };
    }

    return {
        arch: manifest.arch,
        tensors,
        channels: manifest.channels,
        encoderDim: manifest.encoder_dim,
        nodeDim: manifest.node_dim,
        filterNumber: manifest.filter_number,
        parameters: manifest.parameters,
        trainedAgents: manifest.trained_agents,
        trainedBoard: manifest.trained_board,
    };
}

export interface LoadedAssets {
    manifest: DemoManifest;
    models: Record<string, Model>;
}

/**
 * Fetch the manifest and every model it lists.
 *
 * The blobs are independent, so they are fetched concurrently; together they
 * are around 340 KB.
 */
export async function loadAssets(basePath: string): Promise<LoadedAssets> {
    const base = basePath.replace(/\/$/, '');
    const manifest = await fetchJson<DemoManifest>(`${base}/config.json`);

    const names = Object.keys(manifest.models);
    const buffers = await Promise.all(
        names.map(name => fetchBuffer(`${base}/${manifest.models[name].file}`))
    );

    const models: Record<string, Model> = {};
    names.forEach((name, index) => {
        models[name] = parseModel(manifest.models[name], buffers[index]);
    });

    return { manifest, models };
}

async function fetchJson<T>(url: string): Promise<T> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    return response.json() as Promise<T>;
}

async function fetchBuffer(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    return response.arrayBuffer();
}
