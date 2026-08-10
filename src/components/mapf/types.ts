/**
 * Shared types for the MAPF demo.
 *
 * The manifest types mirror what `scripts/export_web_model.py` writes in the
 * MAPF-GNN repository. If you change one, change the other.
 */

/** One tensor's slice of a weights blob. */
export interface TensorEntry {
    name: string;
    shape: number[];
    offset: number;
    count: number;
}

export type Architecture = 'gcn' | 'baseline';

export interface ModelManifest {
    file: string;
    arch: Architecture;
    source: string;
    trained_agents: number;
    trained_board: number;
    channels: number[];
    encoder_dim: number;
    node_dim: number | null;
    filter_number: number | null;
    parameters: number;
    tensors: TensorEntry[];
}

export interface DemoManifest {
    note: string;
    dtype: string;
    /** Field of view, [height, width]. */
    fov: [number, number];
    pad: number;
    sensing_range: number;
    max_neighbours: number;
    actions: { names: string[]; dx: number[]; dy: number[] };
    defaults: {
        temperature: number;
        horizon_factor: number;
        agents: number;
        board: number;
        obstacle_fraction: number;
    };
    models: Record<string, ModelManifest>;
}

/** A weight tensor as a view into the loaded blob. */
export interface Tensor {
    shape: number[];
    data: Float32Array;
}

/** A parsed model: weights plus the shape metadata the forward pass needs. */
export interface Model {
    arch: Architecture;
    tensors: Record<string, Tensor>;
    /** Channel widths including the 2 input channels, e.g. [2, 16, 16, 16]. */
    channels: number[];
    encoderDim: number;
    /** Null for the baseline, which has no graph filter. */
    nodeDim: number | null;
    filterNumber: number | null;
    parameters: number;
    trainedAgents: number;
    trainedBoard: number;
}

/** Everything needed to rebuild an instance exactly. */
export interface Instance {
    board: number;
    obstacles: Array<[number, number]>;
    starts: Array<[number, number]>;
    goals: Array<[number, number]>;
}

export interface EnvConfig {
    board: number;
    sensingRange: number;
    /** Field-of-view padding; the patch is `pad * 2 - 1` on a side. */
    pad: number;
    maxNeighbours: number;
}

/** Board cell contents. Matches the Python environment's encoding. */
export const CELL_FREE = 0;
export const CELL_AGENT = 1;
export const CELL_OBSTACLE = 2;

/** The value the goal marker is stamped with in field-of-view channel 1. */
export const GOAL_MARKER = 3;
