/**
 * Headless verification + benchmark runner (Node).
 *
 * Same logic as the browser harness, but runs without a DOM so it can be
 * scripted in CI. Loads the raw .wasm directly — no wasm-bindgen glue.
 */

import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import * as ref from '../web/js/reference.js';

const WASM_PATH = new URL('../target/wasm32-unknown-unknown/release/wasm.wasm', import.meta.url);

async function instantiate() {
  const bytes = await readFile(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  return instance.exports;
}

function makeFixtures(w, h) {
  const rgba = new Uint8Array(w * h * 4);
  const luma = new Uint8Array(w * h);
  let s = 0x12345678;
  const next = () => {
    s = (Math.imul(s, 1664525) + 1013904223) | 0;
    return (s >>> 24) & 0xff;
  };
  for (let i = 0; i < w * h; i++) {
    rgba[i * 4] = next();
    rgba[i * 4 + 1] = next();
    rgba[i * 4 + 2] = next();
    rgba[i * 4 + 3] = 255;
    luma[i] = next();
  }
  return { rgba, luma };
}

const median = (xs) => {
  const a = [...xs].sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
};
const p95 = (xs) => {
  const a = [...xs].sort((p, q) => p - q);
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.95))];
};

function timeIt(fn, warmup = 5, runs = 20) {
  for (let i = 0; i < warmup; i++) fn();
  const samples = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  return { median: median(samples), p95: p95(samples) };
}

function firstMismatch(a, b) {
  if (a.length !== b.length) return `length ${a.length} vs ${b.length}`;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return `idx ${i}: js=${a[i]} rust=${b[i]}`;
  }
  return null;
}

/** Always re-view memory: wasm heap growth detaches old ArrayBuffers. */
const view = (mem, ptr, len) => new Uint8Array(mem.buffer, ptr, len);

async function main() {
  const ex = await instantiate();
  const mem = ex.memory;

  const rows = [];
  const fails = [];

  for (const { label, w, h } of [
    { label: '512x512', w: 512, h: 512 },
    { label: '1920x1080', w: 1920, h: 1080 },
  ]) {
    const px = w * h;
    const { rgba, luma } = makeFixtures(w, h);

    const pRgba = ex.alloc(px * 4);
    const pLumaOut = ex.alloc(px);
    const pLumaIn = ex.alloc(px);
    const pDst = ex.alloc(px);
    const pScratch = ex.alloc(px);

    if (!pRgba || !pLumaOut || !pLumaIn || !pDst || !pScratch) {
      throw new Error('wasm alloc returned null');
    }

    view(mem, pRgba, px * 4).set(rgba);

    const jsLuma = new Uint8Array(px);
    const jsOut = new Uint8Array(px);
    const jsScratch = new Uint8Array(px);

    // ---- correctness ----
    const checks = [];

    {
      const expect = ref.grayscale(rgba, new Uint8Array(px));
      ex.grayscale(pRgba, px * 4, pLumaOut, px);
      checks.push(['grayscale', firstMismatch(expect, view(mem, pLumaOut, px))]);
    }
    {
      const expect = ref.threshold(luma, new Uint8Array(px), 128);
      view(mem, pLumaIn, px).set(luma);
      ex.threshold_out(pLumaIn, pDst, px, 128);
      checks.push(['threshold', firstMismatch(expect, view(mem, pDst, px))]);
    }
    {
      const expect = ref.sobel(luma, new Uint8Array(px), w, h);
      view(mem, pLumaIn, px).set(luma);
      ex.sobel(pLumaIn, pDst, w, h);
      checks.push(['sobel', firstMismatch(expect, view(mem, pDst, px))]);
    }
    {
      const expect = ref.boxBlur(luma, new Uint8Array(px), new Uint8Array(px), w, h, 8);
      view(mem, pLumaIn, px).set(luma);
      ex.box_blur(pLumaIn, pDst, pScratch, w, h, 8);
      checks.push(['boxBlur', firstMismatch(expect, view(mem, pDst, px))]);
    }

    for (const [kernel, diff] of checks) {
      const ok = diff === null;
      if (!ok) fails.push(`${label} ${kernel}: ${diff}`);
      console.log(`${ok ? '✅' : '❌'} ${label} ${kernel}: ${ok ? 'byte-identical' : diff}`);
    }

    // ---- speed ----
    const mpx = px / 1e6;

    const cases = [
      ['grayscale', () => ref.grayscale(rgba, jsOut), () => ex.grayscale(pRgba, px * 4, pLumaOut, px)],
      ['threshold', () => ref.threshold(luma, jsOut, 128), () => ex.threshold_out(pLumaIn, pDst, px, 128)],
      ['sobel', () => ref.sobel(luma, jsOut, w, h), () => ex.sobel(pLumaIn, pDst, w, h)],
      ['boxBlur', () => ref.boxBlur(luma, jsOut, jsScratch, w, h, 8), () => ex.box_blur(pLumaIn, pDst, pScratch, w, h, 8)],
    ];

    for (const [kernel, jsFn, wasmFn] of cases) {
      const js = timeIt(jsFn);
      const wsm = timeIt(wasmFn);
      const speedup = js.median / wsm.median;
      rows.push({ label, kernel, jsMedian: js.median, wasmMedian: wsm.median, speedup, mpx });
      console.log(
        `   ${label} ${kernel}: JS ${js.median.toFixed(2)}ms → WASM ${wsm.median.toFixed(2)}ms ` +
          `(${speedup.toFixed(2)}x)  [${(mpx / (wsm.median / 1000)).toFixed(0)} MP/s]`
      );
    }

    ex.dealloc(pRgba, px * 4);
    ex.dealloc(pLumaOut, px);
    ex.dealloc(pLumaIn, px);
    ex.dealloc(pDst, px);
    ex.dealloc(pScratch, px);
  }

  return { rows, fails };
}

main()
  .then(({ fails }) => {
    if (fails.length) {
      console.error(`\n${fails.length} correctness failure(s).`);
      process.exit(1);
    }
    console.log('\nAll kernels byte-identical to the JS reference.');
  })
  .catch((e) => {
    console.error(e);
    process.exit(1);
  });
