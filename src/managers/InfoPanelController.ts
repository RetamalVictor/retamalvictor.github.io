import type { DemoType } from './DemoManager.js';

/**
 * Info panel content structure
 */
export interface InfoPanelContent {
    title: string;
    content: string;
}

/**
 * Info panel content for each demo type
 */
export const INFO_PANEL_CONTENT: Record<DemoType, InfoPanelContent> = {
    'ibvs': {
        title: 'Visual Servoing Demo',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo shows <strong class="text-white">Image-Based Visual Servoing (IBVS)</strong>
                    controlling a quadrotor to track a target using only camera feedback.
                    The drone tilts to move, demonstrating underactuated dynamics.
                </p>
            </div>

            <!-- IBVS Controller -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">IBVS Controller</h3>
                <p class="text-gray-400 mb-3">
                    The controller minimizes feature error in image space:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">e</span> = <span class="text-accent-purple">s*</span> - <span class="text-white">s</span>
                        <span class="text-gray-500 ml-2">// feature error</span>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">v</span> = <span class="text-yellow-400">λ</span> · <span class="text-white">L</span><sup>+</sup> · <span class="text-accent-cyan">e</span>
                        <span class="text-gray-500 ml-2">// control law</span>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <strong class="text-gray-400">L</strong> is the interaction matrix relating feature motion to camera velocity.
                    <strong class="text-gray-400">L<sup>+</sup></strong> is its pseudo-inverse.
                </p>
            </div>

            <!-- Interaction Matrix -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Interaction Matrix</h3>
                <p class="text-gray-400 mb-3">
                    For each image point (x, y) at depth Z:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs overflow-x-auto">
                    <div class="text-gray-300 whitespace-nowrap">
                        L = [<span class="text-accent-cyan">-f/Z</span>, 0, <span class="text-accent-cyan">x/Z</span>, <span class="text-yellow-400">xy/f</span>, <span class="text-yellow-400">-(f+x²/f)</span>, <span class="text-yellow-400">y</span>]
                    </div>
                    <div class="text-gray-300 whitespace-nowrap mt-1">
                        &nbsp;&nbsp;&nbsp;&nbsp;[0, <span class="text-accent-cyan">-f/Z</span>, <span class="text-accent-cyan">y/Z</span>, <span class="text-yellow-400">f+y²/f</span>, <span class="text-yellow-400">-xy/f</span>, <span class="text-yellow-400">-x</span>]
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <span class="text-accent-cyan">Cyan</span>: translation terms,
                    <span class="text-yellow-400">Yellow</span>: rotation terms
                </p>
            </div>

            <!-- Quadrotor Dynamics (Differential Flatness) -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Differential Flatness Control</h3>
                <p class="text-gray-400 mb-3">
                    Direct computation of thrust & attitude from desired acceleration:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-500">// 1. Velocity tracking → acceleration</div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">a</span><sub>des</sub> = k<sub>v</sub>·(v<sub>des</sub> - v)
                    </div>
                    <div class="text-gray-500 mt-2">// 2. Flatness: thrust vector</div>
                    <div class="text-gray-300">
                        <span class="text-yellow-400">F</span> = m·(a<sub>des</sub> + g)
                    </div>
                    <div class="text-gray-500 mt-2">// 3. Attitude from thrust direction</div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">θ</span><sub>pitch</sub> = atan2(F<sub>z</sub>, F<sub>y</sub>)
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">θ</span><sub>roll</sub> = atan2(-F<sub>x</sub>, F<sub>y</sub>)
                    </div>
                </div>
            </div>

            <!-- Camera Model -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Pinhole Camera</h3>
                <p class="text-gray-400 mb-3">
                    Projects 3D world points to 2D image coordinates:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">u</span> = f·(X/Z) + c<sub>x</sub>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">v</span> = f·(Y/Z) + c<sub>y</sub>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    <strong class="text-gray-400">f</strong>: focal length,
                    <strong class="text-gray-400">(c<sub>x</sub>, c<sub>y</sub>)</strong>: principal point
                </p>
            </div>

            <!-- Parameters -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Parameters</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">IBVS gain (λ)</span>
                        <span class="text-white font-mono">1.5</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Velocity gain (k<sub>v</sub>)</span>
                        <span class="text-white font-mono">2.0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Attitude gain (k<sub>p</sub>)</span>
                        <span class="text-white font-mono">25.0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Max tilt</span>
                        <span class="text-white font-mono">60°</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Desired distance</span>
                        <span class="text-white font-mono">2.0m</span>
                    </div>
                </div>
            </div>

            <!-- References -->
            <div class="info-section border-t border-dark-border pt-4">
                <h3 class="text-accent-purple font-medium mb-2">References</h3>
                <ul class="text-gray-500 text-xs space-y-1">
                    <li>Chaumette & Hutchinson, "Visual Servo Control" (2006)</li>
                    <li>Corke, "Robotics, Vision and Control" (2017)</li>
                </ul>
            </div>
        `
    },
    'ternary': {
        title: 'Ternary Language Model',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo shows a <strong class="text-white">Ternary Quantized Language Model</strong>
                    running entirely in the browser. Weights are compressed to {-1, 0, +1}, enabling
                    efficient inference with minimal memory footprint.
                </p>
            </div>

            <!-- Architecture -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Architecture</h3>
                <p class="text-gray-400 mb-3">
                    Transformer-based model with ternary weights:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">W</span> ∈ {-1, 0, +1}<sup>d×d</sup>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">y</span> = W · x <span class="text-gray-500">// No multiplications!</span>
                    </div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    Multiplications become additions/subtractions, enabling fast CPU inference.
                </p>
            </div>

            <!-- Quantization -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Quantization</h3>
                <p class="text-gray-400 mb-3">
                    Ternary quantization reduces memory by ~16x:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">FP32 weights</span>
                        <span class="text-white font-mono">32 bits/param</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Ternary weights</span>
                        <span class="text-white font-mono">2 bits/param</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Compression</span>
                        <span class="text-accent-cyan font-mono">16x</span>
                    </div>
                </div>
            </div>

            <!-- Model Details -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Model Details</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Parameters</span>
                        <span class="text-white font-mono">~1M</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Layers</span>
                        <span class="text-white font-mono">4</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Hidden dim</span>
                        <span class="text-white font-mono">256</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Vocab size</span>
                        <span class="text-white font-mono">65</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Training data</span>
                        <span class="text-white font-mono">Shakespeare</span>
                    </div>
                </div>
            </div>

            <!-- References -->
            <div class="info-section border-t border-dark-border pt-4">
                <h3 class="text-accent-purple font-medium mb-2">References</h3>
                <ul class="text-gray-500 text-xs space-y-1">
                    <li>Ma et al., "The Era of 1-bit LLMs" (2024)</li>
                    <li>Karpathy, "nanoGPT" (2023)</li>
                </ul>
            </div>
        `
    },
    'drone-racing': {
        title: 'Drone Racing MPC',
        content: `
            <!-- Overview -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Overview</h3>
                <p class="text-gray-400 leading-relaxed">
                    This demo implements <strong class="text-white">Model Predictive Control (MPC)</strong> for autonomous drone racing
                    using <strong class="text-white">Sequential Quadratic Programming (SQP)</strong>. The controller predicts 500ms into the future
                    and optimizes thrust and angular rate commands in real-time.
                </p>
                <p class="text-gray-500 text-xs mt-2">
                    Coordinate system: <strong class="text-gray-400">Y-up</strong> (Three.js convention)
                </p>
            </div>

            <!-- System Model -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">1. System Model</h3>
                <p class="text-gray-400 mb-2">
                    <strong class="text-white">State Vector</strong> (14-dimensional):
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">x</span> = [p<sub>x</sub>, p<sub>y</sub>, p<sub>z</sub>, <span class="text-gray-500">// position</span>
                    </div>
                    <div class="text-gray-300 pl-4">
                        v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>, <span class="text-gray-500">// velocity</span>
                    </div>
                    <div class="text-gray-300 pl-4">
                        q<sub>w</sub>, q<sub>x</sub>, q<sub>y</sub>, q<sub>z</sub>, <span class="text-gray-500">// quaternion</span>
                    </div>
                    <div class="text-gray-300 pl-4">
                        T, ω<sub>r</sub>, ω<sub>p</sub>, ω<sub>y</sub>] <span class="text-gray-500">// actuators</span>
                    </div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Input Vector</strong> (4-dimensional):
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs">
                    <div class="text-gray-300">
                        <span class="text-yellow-400">u</span> = [T<sub>cmd</sub>, ω<sub>r,cmd</sub>, ω<sub>p,cmd</sub>, ω<sub>y,cmd</sub>]
                    </div>
                </div>
            </div>

            <!-- Continuous Dynamics -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">2. Continuous Dynamics</h3>
                <p class="text-gray-400 mb-2">
                    <strong class="text-white">Position:</strong> ṗ = v
                </p>
                <p class="text-gray-400 mb-2">
                    <strong class="text-white">Velocity</strong> (with drag):
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs">
                    <div class="text-gray-300">
                        v̇ = (1/m)·<span class="text-accent-cyan">R(q)</span>·[0, T, 0]ᵀ - [0, g, 0]ᵀ - c<sub>d</sub>·v
                    </div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Quaternion kinematics:</strong>
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">q̇ = ½ · q ⊗ [0, ω<sub>body</sub>]</div>
                    <div class="text-gray-500 mt-2">// Full quaternion product expansion:</div>
                    <div class="text-gray-300">q̇<sub>w</sub> = -½(q<sub>x</sub>ω<sub>x</sub> + q<sub>y</sub>ω<sub>y</sub> + q<sub>z</sub>ω<sub>z</sub>)</div>
                    <div class="text-gray-300">q̇<sub>x</sub> = ½(q<sub>w</sub>ω<sub>x</sub> + q<sub>y</sub>ω<sub>z</sub> - q<sub>z</sub>ω<sub>y</sub>)</div>
                    <div class="text-gray-300">q̇<sub>y</sub> = ½(q<sub>w</sub>ω<sub>y</sub> + q<sub>z</sub>ω<sub>x</sub> - q<sub>x</sub>ω<sub>z</sub>)</div>
                    <div class="text-gray-300">q̇<sub>z</sub> = ½(q<sub>w</sub>ω<sub>z</sub> + q<sub>x</sub>ω<sub>y</sub> - q<sub>y</sub>ω<sub>x</sub>)</div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Actuator dynamics</strong> (first-order):
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs">
                    <div class="text-gray-300">Ṫ = (T<sub>cmd</sub> - T) / <span class="text-yellow-400">τ<sub>T</sub></span></div>
                    <div class="text-gray-300">ω̇ = (ω<sub>cmd</sub> - ω) / <span class="text-yellow-400">τ<sub>ω</sub></span></div>
                    <div class="text-gray-500 mt-1">τ<sub>T</sub> = 40ms, τ<sub>ω</sub> = 30ms</div>
                </div>
            </div>

            <!-- MPC Formulation -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">3. MPC Formulation</h3>
                <p class="text-gray-400 mb-2">
                    Optimal control problem solved at each timestep:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">min</span> Σ<sub>k=0</sub><sup>N-1</sup> ‖x<sub>k</sub> - x<sub>k</sub><sup>ref</sup>‖<sub>Q</sub>² + ‖u<sub>k</sub> - u<sub>k</sub><sup>ref</sup>‖<sub>R</sub>²
                    </div>
                    <div class="text-gray-500 text-xs">+ terminal cost ‖x<sub>N</sub> - x<sub>N</sub><sup>ref</sup>‖<sub>Q<sub>f</sub></sub>²</div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    u<sup>ref</sup> from feedforward (hover thrust + trajectory acceleration)
                </p>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Subject to:</strong>
                </p>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="text-gray-300">• x<sub>k+1</sub> = F(x<sub>k</sub>, u<sub>k</sub>, Δt) <span class="text-gray-500">// discrete dynamics</span></div>
                    <div class="text-gray-300">• x<sub>0</sub> = x<sub>current</sub> <span class="text-gray-500">// initial condition</span></div>
                    <div class="text-gray-300">• u<sub>min</sub> ≤ u<sub>k</sub> ≤ u<sub>max</sub> <span class="text-gray-500">// box constraints</span></div>
                </div>
            </div>

            <!-- Configuration -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">4. MPC Configuration</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Horizon</span>
                        <span class="text-white font-mono">N=10 × 50ms = 500ms</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Position weight Q<sub>p</sub></span>
                        <span class="text-white font-mono">400</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Velocity weight Q<sub>v</sub></span>
                        <span class="text-white font-mono">0.1</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Attitude weight Q<sub>θ</sub></span>
                        <span class="text-white font-mono">5</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Input weight R</span>
                        <span class="text-white font-mono">0.1 · I<sub>4</sub></span>
                    </div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Input Bounds:</strong>
                </p>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Thrust</span>
                        <span class="text-white font-mono">[0, 50] m/s² <span class="text-gray-500">(5:1 T/W)</span></span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Roll/Pitch rate</span>
                        <span class="text-white font-mono">±20 rad/s</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Yaw rate</span>
                        <span class="text-white font-mono">±10 rad/s</span>
                    </div>
                </div>
            </div>

            <!-- SQP -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">5. Sequential Quadratic Programming</h3>
                <p class="text-gray-400 mb-2">
                    Linearize discrete dynamics at operating point (x̄<sub>k</sub>, ū<sub>k</sub>):
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">x<sub>k+1</sub> ≈ <span class="text-accent-cyan">A</span><sub>k</sub>x<sub>k</sub> + <span class="text-yellow-400">B</span><sub>k</sub>u<sub>k</sub> + c<sub>k</sub></div>
                    <div class="text-gray-500 mt-2">// Jacobians via finite differences (ε = 10⁻⁶)</div>
                    <div class="text-gray-300 mt-1"><span class="text-accent-cyan">A</span> = ∂F/∂x <span class="text-gray-500">(14×14)</span></div>
                    <div class="text-gray-300"><span class="text-yellow-400">B</span> = ∂F/∂u <span class="text-gray-500">(14×4)</span></div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    <strong class="text-white">Condensed QP formulation</strong> eliminates states:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs">
                    <div class="text-gray-300">x<sub>k</sub> = Φ<sub>k</sub>x<sub>0</sub> + Ψ<sub>k</sub>ΔU + d<sub>k</sub></div>
                    <div class="text-gray-500 mt-1">ΔU ∈ ℝ<sup>N·n<sub>u</sub></sup> = ℝ<sup>40</sup></div>
                </div>
            </div>

            <!-- QP Solver -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">6. QP Solver</h3>
                <p class="text-gray-400 mb-2">
                    <strong class="text-white">Projected Gradient with Nesterov Acceleration:</strong>
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">min</span> ½ΔUᵀHΔU + gᵀΔU
                    </div>
                    <div class="text-gray-300">s.t. lb ≤ ΔU ≤ ub</div>
                </div>
                <p class="text-gray-400 mt-3 mb-2">
                    Algorithm per iteration:
                </p>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-1">
                    <div class="text-gray-300">1. ∇f = H·y + g</div>
                    <div class="text-gray-300">2. x̃ = y - α·∇f</div>
                    <div class="text-gray-300">3. x<sup>k+1</sup> = Π<sub>[lb,ub]</sub>(x̃)</div>
                    <div class="text-gray-300">4. β = (t<sub>k</sub>-1)/t<sub>k+1</sub> <span class="text-gray-500">// momentum</span></div>
                    <div class="text-gray-300">5. y = x<sup>k+1</sup> + β(x<sup>k+1</sup> - x<sup>k</sup>)</div>
                </div>
                <p class="text-gray-500 text-xs mt-2">
                    Step size α = 1/λ<sub>max</sub> via Gershgorin circles. QP dim: 40×40.
                </p>
            </div>

            <!-- Pipeline -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">7. Control Pipeline</h3>
                <div class="bg-dark-bg rounded-lg p-3 font-mono text-xs space-y-2">
                    <div class="text-accent-cyan">// ~60Hz control loop</div>
                    <div class="text-gray-300">1. Sample trajectory waypoints</div>
                    <div class="text-gray-300">2. Compute feedforward (accel → quat)</div>
                    <div class="text-gray-300">3. Initialize trajectory rollout</div>
                    <div class="text-gray-300">4. <span class="text-yellow-400">SQP iteration:</span></div>
                    <div class="text-gray-300 pl-4">a. Linearize dynamics</div>
                    <div class="text-gray-300 pl-4">b. Build condensed QP (Ψ matrices)</div>
                    <div class="text-gray-300 pl-4">c. Solve via projected gradient</div>
                    <div class="text-gray-300 pl-4">d. Update trajectory</div>
                    <div class="text-gray-300">5. Extract u<sub>0</sub> = u<sub>nom</sub> + Δu<sub>0</sub></div>
                    <div class="text-gray-300">6. Apply to drone dynamics</div>
                </div>
            </div>

            <!-- Improvement Opportunities -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">8. Potential Improvements</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-2">
                    <div class="flex justify-between items-start">
                        <span class="text-yellow-400">Numerical Jacobians</span>
                        <span class="text-gray-400 text-right">56 evals/linearization</span>
                    </div>
                    <div class="text-gray-500 text-xs">→ Analytical Jacobians: 5-10× speedup</div>

                    <div class="flex justify-between items-start mt-2">
                        <span class="text-yellow-400">Projected Gradient</span>
                        <span class="text-gray-400 text-right">~50 iterations</span>
                    </div>
                    <div class="text-gray-500 text-xs">→ OSQP/qpOASES: 5-20 iterations</div>

                    <div class="flex justify-between items-start mt-2">
                        <span class="text-yellow-400">Warm-start disabled</span>
                        <span class="text-gray-400 text-right">circular issues</span>
                    </div>
                    <div class="text-gray-500 text-xs">→ Fix for faster convergence</div>
                </div>
            </div>

            <!-- Controls -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Controls</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Mouse drag</span>
                        <span class="text-white">Orbit camera</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Scroll</span>
                        <span class="text-white">Zoom in/out</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Overview/Follow</span>
                        <span class="text-white">Camera modes</span>
                    </div>
                </div>
            </div>

            <!-- Code References -->
            <div class="info-section">
                <h3 class="text-accent-purple font-medium mb-2">Code References</h3>
                <div class="bg-dark-bg rounded-lg p-3 text-xs space-y-1">
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">MPC.ts</span>
                        <span class="text-gray-500"> - SQP solver, QP formulation</span>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">MPCModel.ts</span>
                        <span class="text-gray-500"> - dynamics, linearization</span>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">QPSolver.ts</span>
                        <span class="text-gray-500"> - projected gradient</span>
                    </div>
                    <div class="text-gray-300">
                        <span class="text-accent-cyan">DroneDynamics.ts</span>
                        <span class="text-gray-500"> - simulation model</span>
                    </div>
                </div>
            </div>

        `
    }
};

/**
 * Controls the info panel (slide-in panel with demo explanations)
 */
export class InfoPanelController {
    private panel: HTMLElement | null = null;
    private overlay: HTMLElement | null = null;
    private isOpen: boolean = false;
    private boundHandleEscape: (e: KeyboardEvent) => void;

    constructor() {
        this.boundHandleEscape = this.handleEscape.bind(this);
    }

    /**
     * Initialize the panel with DOM elements
     */
    public initialize(): void {
        this.panel = document.getElementById('info-panel');
        this.overlay = document.getElementById('info-panel-overlay');

        const toggleBtn = document.getElementById('info-panel-toggle');
        const closeBtn = document.getElementById('info-panel-close');

        if (!this.panel || !this.overlay || !toggleBtn) {
            console.warn('InfoPanelController: required elements not found');
            return;
        }

        toggleBtn.addEventListener('click', () => this.open());
        closeBtn?.addEventListener('click', () => this.close());
        this.overlay.addEventListener('click', () => this.close());

        document.addEventListener('keydown', this.boundHandleEscape);
    }

    /**
     * Handle escape key to close panel
     */
    private handleEscape(e: KeyboardEvent): void {
        if (e.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    /**
     * Open the info panel
     */
    public open(): void {
        if (!this.panel || !this.overlay) return;

        this.panel.classList.remove('translate-x-full');
        this.overlay.classList.remove('opacity-0', 'pointer-events-none');
        this.overlay.classList.add('opacity-100', 'pointer-events-auto');
        this.isOpen = true;
    }

    /**
     * Close the info panel
     */
    public close(): void {
        if (!this.panel || !this.overlay) return;

        this.panel.classList.add('translate-x-full');
        this.overlay.classList.add('opacity-0', 'pointer-events-none');
        this.overlay.classList.remove('opacity-100', 'pointer-events-auto');
        this.isOpen = false;
    }

    /**
     * Toggle the panel open/closed
     */
    public toggle(): void {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    /**
     * Update panel content for a specific demo type
     */
    public setContent(demoType: DemoType): void {
        const title = document.getElementById('info-panel-title');
        const content = document.getElementById('info-panel-content');
        if (!title || !content) return;

        const panelData = INFO_PANEL_CONTENT[demoType];
        title.textContent = panelData.title;
        content.innerHTML = panelData.content;
    }

    /**
     * Check if panel is currently open
     */
    public isCurrentlyOpen(): boolean {
        return this.isOpen;
    }

    /**
     * Cleanup event listeners
     */
    public destroy(): void {
        document.removeEventListener('keydown', this.boundHandleEscape);
    }
}
