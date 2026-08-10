/**
 * Multi-Agent Path Finding demo.
 *
 * A graph neural network trained by imitating an optimal centralised planner on
 * ten robots, running client-side on hundreds. Two policies race the same
 * instance so the value of communication is visible rather than asserted.
 *
 * Weights come from the MAPF-GNN repository via `scripts/export_web_model.py`,
 * and the port is pinned to PyTorch by `test/parity.ts` (`npm run test:mapf`).
 */

export { MapfDemo } from './MapfDemo';
export type { MapfDemoConfig } from './MapfDemo';
export { MapfEnv, boardForAgents, sampleInstance } from './env';
export { Policy } from './policy';
export { loadAssets, parseModel } from './weights';
