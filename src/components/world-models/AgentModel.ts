/**
 * The trained world model and policy, running in the browser.
 *
 * Two loops share these weights.
 *
 *   controlled:  e = enc(o); h = GRU(h, z, a); z = post(h, e);
 *                a = max_action * tanh(mu_pi([h, z]))
 *   dreamed:     h = GRU(h, z, a); z = prior(h); o_hat = dec([h, z])
 *
 * The controlled loop is what drives the agent in the real environment:
 * it sees pixels, never the simulator's state. The dreamed loop never
 * sees a frame at all. Numerics are checked against JAX in
 * test/agent-parity.ts, because a hand-written ConvTranspose is wrong
 * until proven otherwise.
 */

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const silu = (x: number) => x * sigmoid(x);
const softplus = (x: number) => (x > 20 ? x : Math.log1p(Math.exp(x)));

export interface AgentConfig {
    hidden: number;
    latent_dim: number;
    action_dim: number;
    obs_size: number;
    obs_channels: number;
    min_sigma: number;
    embed_dim: number;
    encoder_ladder: number[];
    decoder_ladder: number[];
    max_action: number;
    actor_min_sigma: number;
}

export interface AgentManifest {
    config: AgentConfig;
    tensors: Array<{ name: string; shape: number[]; offset: number }>;
}

export interface AgentState {
    h: Float32Array;
    z: Float32Array;
    a: Float32Array;
}

function dense(x: Float32Array, W: Float32Array, b: Float32Array | null,
               inDim: number, outDim: number): Float32Array {
    const y = new Float32Array(outDim);
    if (b) y.set(b);
    for (let i = 0; i < inDim; i++) {
        const xi = x[i];
        if (xi === 0) continue;
        const row = i * outDim;
        for (let o = 0; o < outDim; o++) y[o] += xi * W[row + o];
    }
    return y;
}

function applySilu(v: Float32Array): Float32Array {
    for (let i = 0; i < v.length; i++) v[i] = silu(v[i]);
    return v;
}

/** Flax GRUCell: hr and hz carry no bias, hn does. */
function gruCell(p: Record<string, Float32Array>, h: Float32Array,
                 x: Float32Array, hidden: number, inDim: number): Float32Array {
    const ir = dense(x, p['core/ir/kernel'], p['core/ir/bias'], inDim, hidden);
    const iz = dense(x, p['core/iz/kernel'], p['core/iz/bias'], inDim, hidden);
    const inn = dense(x, p['core/in/kernel'], p['core/in/bias'], inDim, hidden);
    const hr = dense(h, p['core/hr/kernel'], null, hidden, hidden);
    const hz = dense(h, p['core/hz/kernel'], null, hidden, hidden);
    const hn = dense(h, p['core/hn/kernel'], p['core/hn/bias'], hidden, hidden);
    const out = new Float32Array(hidden);
    for (let i = 0; i < hidden; i++) {
        const r = sigmoid(ir[i] + hr[i]);
        const z = sigmoid(iz[i] + hz[i]);
        const n = Math.tanh(inn[i] + r * hn[i]);
        out[i] = (1 - z) * n + z * h[i];
    }
    return out;
}

/**
 * Flax Conv, kernel 4x4, stride 2, padding SAME. For an even input the
 * SAME rule pads 1 before and 2 after on each axis (total k - s = 2).
 */
function conv2Stride2(input: Float32Array, H: number, W: number, cin: number,
                      K: Float32Array, bias: Float32Array, cout: number) {
    const outH = H >> 1, outW = W >> 1;
    const out = new Float32Array(outH * outW * cout);
    const kStrideY = 4 * cin * cout, kStrideX = cin * cout;
    for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
            const base = (oy * outW + ox) * cout;
            out.set(bias, base);
            for (let ky = 0; ky < 4; ky++) {
                const iy = oy * 2 + ky - 1;
                if (iy < 0 || iy >= H) continue;
                for (let kx = 0; kx < 4; kx++) {
                    const ix = ox * 2 + kx - 1;
                    if (ix < 0 || ix >= W) continue;
                    const inBase = (iy * W + ix) * cin;
                    const kBase = ky * kStrideY + kx * kStrideX;
                    for (let c = 0; c < cin; c++) {
                        const v = input[inBase + c];
                        if (v === 0) continue;
                        const kRow = kBase + c * cout;
                        for (let o = 0; o < cout; o++) out[base + o] += v * K[kRow + o];
                    }
                }
            }
        }
    }
    return out;
}

/** Flax ConvTranspose, kernel 4x4, stride 2, SAME, kernel unflipped. */
function convTranspose2x(input: Float32Array, H: number, W: number, cin: number,
                         K: Float32Array, bias: Float32Array, cout: number) {
    const outH = H * 2, outW = W * 2;
    const out = new Float32Array(outH * outW * cout);
    for (let i = 0; i < outH * outW; i++) out.set(bias, i * cout);
    const kStrideY = 4 * cin * cout, kStrideX = cin * cout;

    for (let y = 0; y < outH; y++) {
        for (let ky = (y & 1) ? 1 : 0; ky < 4; ky += 2) {
            const sy = y + ky - 2;
            if (sy & 1) continue;
            const iy = sy >> 1;
            if (iy < 0 || iy >= H) continue;
            for (let x = 0; x < outW; x++) {
                const outBase = (y * outW + x) * cout;
                for (let kx = (x & 1) ? 1 : 0; kx < 4; kx += 2) {
                    const sx = x + kx - 2;
                    if (sx & 1) continue;
                    const ix = sx >> 1;
                    if (ix < 0 || ix >= W) continue;
                    const inBase = (iy * W + ix) * cin;
                    const kBase = ky * kStrideY + kx * kStrideX;
                    for (let c = 0; c < cin; c++) {
                        const v = input[inBase + c];
                        if (v === 0) continue;
                        const kRow = kBase + c * cout;
                        for (let o = 0; o < cout; o++) out[outBase + o] += v * K[kRow + o];
                    }
                }
            }
        }
    }
    return out;
}

export class AgentModel {
    public cfg: AgentConfig;
    private p: Record<string, Float32Array> = {};

    constructor(manifest: AgentManifest, buffer: ArrayBuffer) {
        this.cfg = manifest.config;
        const f32 = new Float32Array(buffer);
        for (const t of manifest.tensors) {
            const n = t.shape.reduce((a, b) => a * b, 1);
            this.p[t.name] = f32.subarray(t.offset / 4, t.offset / 4 + n);
        }
    }

    static async load(baseUrl: string): Promise<AgentModel> {
        const manifest = await (await fetch(`${baseUrl}/manifest.json`)).json();
        const buffer = await (await fetch(`${baseUrl}/weights.bin`)).arrayBuffer();
        return new AgentModel(manifest, buffer);
    }

    initialState(): AgentState {
        return {
            h: new Float32Array(this.cfg.hidden),
            z: new Float32Array(this.cfg.latent_dim),
            a: new Float32Array(this.cfg.action_dim),
        };
    }

    /** conv trunk over an (obs_size, obs_size, C) frame -> embedding */
    encode(frame: Float32Array): Float32Array {
        const { obs_size, obs_channels, encoder_ladder, embed_dim } = this.cfg;
        let v = frame;
        let size = obs_size;
        let cin = obs_channels;
        encoder_ladder.forEach((cout, i) => {
            v = conv2Stride2(v, size, size, cin,
                this.p[`encoder/Conv_${i}/kernel`],
                this.p[`encoder/Conv_${i}/bias`], cout);
            applySilu(v);
            size >>= 1;
            cin = cout;
        });
        return applySilu(dense(v, this.p['encoder/Dense_0/kernel'],
            this.p['encoder/Dense_0/bias'], size * size * cin, embed_dim));
    }

    coreStep(h: Float32Array, z: Float32Array, a: Float32Array): Float32Array {
        const { hidden, latent_dim: L, action_dim: A } = this.cfg;
        const x = new Float32Array(L + A);
        x.set(z, 0);
        x.set(a, L);
        return gruCell(this.p, h, x, hidden, L + A);
    }

    private gaussianHead(prefix: string, input: Float32Array, inDim: number) {
        const { hidden, latent_dim: L, min_sigma } = this.cfg;
        const g = applySilu(dense(input, this.p[`${prefix}/Dense_0/kernel`],
            this.p[`${prefix}/Dense_0/bias`], inDim, hidden));
        const mu = dense(g, this.p[`${prefix}/Dense_1/kernel`],
            this.p[`${prefix}/Dense_1/bias`], hidden, L);
        const raw = dense(g, this.p[`${prefix}/Dense_2/kernel`],
            this.p[`${prefix}/Dense_2/bias`], hidden, L);
        const sigma = new Float32Array(L);
        for (let i = 0; i < L; i++) sigma[i] = softplus(raw[i]) + min_sigma;
        return { mu, sigma };
    }

    /** posterior q(z | h, enc(o)): perception, the frame is allowed in */
    posterior(h: Float32Array, e: Float32Array) {
        const s = new Float32Array(h.length + e.length);
        s.set(h, 0);
        s.set(e, h.length);
        return this.gaussianHead('post_head', s, s.length);
    }

    /** prior p(z | h): the model's physics, no frame */
    prior(h: Float32Array) {
        return this.gaussianHead('prior_head', h, h.length);
    }

    /** decoder([h, z]) -> (obs_size, obs_size, C), linear output */
    decode(h: Float32Array, z: Float32Array): Float32Array {
        const { embed_dim, decoder_ladder, obs_channels } = this.cfg;
        const s = new Float32Array(h.length + z.length);
        s.set(h, 0);
        s.set(z, h.length);

        let v = applySilu(dense(s, this.p['decoder/Dense_0/kernel'],
            this.p['decoder/Dense_0/bias'], s.length, embed_dim));
        const top = decoder_ladder[decoder_ladder.length - 1];
        v = applySilu(dense(v, this.p['decoder/Dense_1/kernel'],
            this.p['decoder/Dense_1/bias'], embed_dim, 4 * 4 * top));

        const up = decoder_ladder.slice(0, -1).reverse();
        let size = 4, cin = top;
        up.forEach((cout, i) => {
            v = convTranspose2x(v, size, size, cin,
                this.p[`decoder/ConvTranspose_${i}/kernel`],
                this.p[`decoder/ConvTranspose_${i}/bias`], cout);
            applySilu(v);
            size *= 2;
            cin = cout;
        });
        return convTranspose2x(v, size, size, cin,
            this.p[`decoder/ConvTranspose_${up.length}/kernel`],
            this.p[`decoder/ConvTranspose_${up.length}/bias`], obs_channels);
    }

    /** Deterministic policy: max_action * tanh(mu). No sampling at play time. */
    act(h: Float32Array, z: Float32Array): Float32Array {
        const { hidden, latent_dim: L, action_dim: A, max_action } = this.cfg;
        const s = new Float32Array(hidden + L);
        s.set(h, 0);
        s.set(z, hidden);
        let x = applySilu(dense(s, this.p['actor/Dense_0/kernel'],
            this.p['actor/Dense_0/bias'], hidden + L, 128));
        x = applySilu(dense(x, this.p['actor/Dense_1/kernel'],
            this.p['actor/Dense_1/bias'], 128, 128));
        const mu = dense(x, this.p['actor/Dense_2/kernel'],
            this.p['actor/Dense_2/bias'], 128, A);
        const a = new Float32Array(A);
        for (let i = 0; i < A; i++) a[i] = max_action * Math.tanh(mu[i]);
        return a;
    }

    /** One controlled step: look at a real frame, then decide. */
    observeAndAct(state: AgentState, frame: Float32Array): AgentState {
        const e = this.encode(frame);
        const h = this.coreStep(state.h, state.z, state.a);
        const { mu } = this.posterior(h, e);
        const a = this.act(h, mu);
        return { h, z: mu, a };
    }

    /**
     * One dreamed step: no frame exists, the prior invents the next state.
     *
     * Pass `action` to make this a prediction of what a known action will
     * do, which is the only way to compare a dream against reality without
     * confounding model error with a different policy. Leave it out and the
     * actor acts inside the dream instead, which is what training does.
     */
    imagine(state: AgentState, action?: Float32Array):
            { state: AgentState; frame: Float32Array } {
        const h = this.coreStep(state.h, state.z, state.a);
        const { mu } = this.prior(h);
        const a = action ?? this.act(h, mu);
        return { state: { h, z: mu, a }, frame: this.decode(h, mu) };
    }
}
