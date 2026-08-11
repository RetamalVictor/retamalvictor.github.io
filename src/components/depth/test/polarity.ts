/**
 * Do the two depth models agree about which way is near?
 *
 * MiDaS predicts inverse depth, so a close surface is a large number.
 * FastDepth predicts metric depth, so a close surface is a small one. Their raw
 * outputs should therefore be *negatively* correlated on the same photograph,
 * and DepthEngine negates the light model to put both on MiDaS's convention.
 *
 * Getting this wrong is invisible in code review and obvious on screen — the
 * colour map comes out inside-out, near rendered as far — so it is worth an
 * actual measurement rather than an appeal to what the architectures ought to
 * do. Nothing else in the pipeline can catch it: everything downstream
 * normalises between min and max, which is happy either way round.
 *
 * Runs in a page, because that is where onnxruntime-web lives. With the dev
 * server up, paste this into the browser console:
 *
 *     (await import('/src/components/depth/test/polarity.ts')).runPolarityCheck()
 *
 * Last run: raw correlation -0.72, corrected +0.72 — the two models disagree
 * about direction exactly as expected, and the flag puts them back in step.
 */

import * as ort from 'onnxruntime-web/webgpu';

const MODEL_DIR = '/assets/models/depth';
const TEST_IMAGE = '/images/learning/one_leg_dog.png';
const GRID = 64;

interface Variant {
    file: string;
    label: string;
    inputSize: number;
    inputName: string;
    outputName: string;
    normalizeMean: number[];
    normalizeStd: number[];
    invertDepth: boolean;
}

async function loadImage(src: string): Promise<HTMLImageElement> {
    const image = new Image();
    image.src = src;
    await image.decode();
    return image;
}

/** Run one model over the image and return its raw prediction plus its size. */
async function predict(image: HTMLImageElement, variant: Variant) {
    const canvas = document.createElement('canvas');
    canvas.width = variant.inputSize;
    canvas.height = variant.inputSize;
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    ctx.drawImage(image, 0, 0, variant.inputSize, variant.inputSize);
    const { data } = ctx.getImageData(0, 0, variant.inputSize, variant.inputSize);

    const area = variant.inputSize * variant.inputSize;
    const buffer = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        const p = i * 4;
        for (let c = 0; c < 3; c++) {
            buffer[c * area + i] = (data[p + c] / 255 - variant.normalizeMean[c]) / variant.normalizeStd[c];
        }
    }

    const session = await ort.InferenceSession.create(`${MODEL_DIR}/${variant.file}`, {
        executionProviders: ['wasm'],
    });
    const feeds: Record<string, ort.Tensor> = {};
    feeds[variant.inputName] = new ort.Tensor('float32', buffer, [1, 3, variant.inputSize, variant.inputSize]);
    const output = await session.run(feeds);
    const tensor = output[variant.outputName];
    const values = Float32Array.from(tensor.data as Float32Array);
    const side = tensor.dims[tensor.dims.length - 1];
    await session.release();
    return { values, side };
}

/** Nearest-neighbour resample onto a common grid so the two can be compared. */
function resample(values: Float32Array, side: number): Float32Array {
    const out = new Float32Array(GRID * GRID);
    for (let y = 0; y < GRID; y++) {
        const sy = Math.min(side - 1, Math.floor((y / GRID) * side));
        for (let x = 0; x < GRID; x++) {
            const sx = Math.min(side - 1, Math.floor((x / GRID) * side));
            out[y * GRID + x] = values[sy * side + sx];
        }
    }
    return out;
}

function correlation(a: Float32Array, b: Float32Array): number {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;

    let cov = 0, va = 0, vb = 0;
    for (let i = 0; i < n; i++) {
        const da = a[i] - ma;
        const db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / (Math.sqrt(va * vb) || 1);
}

export async function runPolarityCheck() {
    ort.env.wasm.numThreads = 1;

    const manifest = await (await fetch(`${MODEL_DIR}/config.json`)).json();
    const full: Variant = manifest.variants.full;
    const light: Variant = manifest.variants.light;

    const image = await loadImage(TEST_IMAGE);
    const a = await predict(image, full);
    const b = await predict(image, light);

    const raw = correlation(resample(a.values, a.side), resample(b.values, b.side));

    // What the engine actually feeds the renderer, once each variant's
    // invertDepth flag has been applied.
    const signed = (values: Float32Array, variant: Variant) =>
        variant.invertDepth ? values.map(v => -v) : values;
    const corrected = correlation(
        resample(signed(a.values, full), a.side),
        resample(signed(b.values, light), b.side)
    );

    return {
        full: `${full.label} (${full.file}, ${a.side}x${a.side})`,
        light: `${light.label} (${light.file}, ${b.side}x${b.side})`,
        rawCorrelation: Number(raw.toFixed(3)),
        correctedCorrelation: Number(corrected.toFixed(3)),
        // After correction both should mean "large is near", so they must agree.
        agree: corrected > 0,
        verdict: corrected > 0
            ? 'invertDepth is correct: both models now report near as large'
            : 'invertDepth is WRONG: the light model renders near as far',
    };
}
