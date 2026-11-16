/**
 * Benchmark + differential-correctness harness.
 *
 * Two questions, answered separately:
 *   1. CORRECTNESS — does Rust/WASM produce byte-identical output to the JS
 *      reference? Any mismatch is reported with the first differing index.
 *   2. SPEED — median and p95 of N runs, after warmup, for both sides on the
 *      same input. Reported as ms and megapixels/second.
 *
 * Measurement discipline:
 *   - warmup runs discarded (JIT + wasm compile)
 *   - results collected, then median (not mean — outlier-resistant)
 *   - GC pauses leak into numbers; noted in the README as a known limitation
 */

import * as ref from './reference.js';

const KERNELS = ['grayscale', 'threshold', 'sobel', 'boxBlur'];

function makeFixtures(w, h) {
  const rgba = new Uint8Array(w * h * 4);
  const luma = new Uint8Array(w * h);
  // Deterministic LCGB so runs are comparable across sessions.
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

function median(xs) {
  const a = [...xs].sort((p, q) => p - q);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

function p95(xs) {
  const a = [...xs].sort((p, q) => p - q);
  return a[Math.min(a.length - 1, Math.floor(a.length * 0.95))];
}

function timeIt(fn, { warmup = 5, runs = 20 } = {}) {
  for (let i = 0; i < warmup; i++) fn();
  const samples = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    samples.push(performance.now() - t0);
  }
  return { median: median(samples), p95: p95(samples), samples };
}

function firstMismatch(a, b) {
  if (a.length !== b.length) return { idx: -1, reason: `length ${a.length} vs ${b.length}` };
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return { idx: i, js: a[i], rust: b[i] };
  }
  return null;
}

/**
 * Views over wasm linear memory. Re-created whenever the module grows the heap,
 * because growing invalidates the old ArrayBuffer.
 */
class WasmMemory {
  constructor(wasm) {
    this.wasm = wasm;
    this.views = new Map();
  }
  _buffer() {
    return this.wasm.memory.buffer;
  }
  u8(ptr, len) {
    // Always rebuild: cheap, and immune to heap growth invalidation.
    return new Uint8Array(this._buffer(), ptr, len);
  }
  alloc(len) {
    return this.wasm.alloc(len);
  }
  free(ptr, len) {
    this.wasm.dealloc(ptr, len);
  }
}

export async function runBenchmark(wasmModule, log = console.log) {
  const results = [];

  const sizes = [
    { label: '512x512', w: 512, h: 512 },
    { label: '1920x1080', w: 1920, h: 1080 },
  ];

  const mem = new WasmMemory(wasmModule);

  for (const { label, w, h } of sizes) {
    const px = w * h;
    const { rgba, luma } = makeFixtures(w, h);

    // --- one-time wasm allocations, reused across all runs ---
    const pRgbaIn = mem.alloc(px * 4);
    const pLumaOut = mem.alloc(px);
    const pLumaIn = mem.alloc(px);
    const pDst = mem.alloc(px);
    const pScratch = mem.alloc(px);

    // Seed the wasm-side luma input once.
    mem.u8(pRgbaIn, px * 4).set(rgba);
    wasmModule.grayscale(pRgbaIn, px * 4, pLumaOut, px);
    mem.u8(pLumaIn, px).set(mem.u8(pLumaOut, px));

    // JS scratch buffers, allocated once.
    const jsLuma = new Uint8Array(px);
    const jsOut = new Uint8Array(px);
    const jsScratch = new Uint8Array(px);

    // ---------- correctness ----------
    const checks = [];

    {
      const a = ref.grayscale(rgba, new Uint8Array(px));
      wasmModule.grayscale(pRgbaIn, px * 4, pLumaOut, px);
      const b = mem.u8(pLumaOut, px).slice();
      checks.push({ kernel: 'grayscale', diff: firstMismatch(a, b) });
    }
    {
      const a = ref.threshold(luma, new Uint8Array(px), 128);
      mem.u8(pLumaIn, px).set(luma);
      wasmModule.threshold_out(pLumaIn, pDst, px, 128);
      const b = mem.u8(pDst, px).slice();
      checks.push({ kernel: 'threshold', diff: firstMismatch(a, b) });
    }
    {
      const a = ref.sobel(luma, new Uint8Array(px), w, h);
      mem.u8(pLumaIn, px).set(luma);
      wasmModule.sobel(pLumaIn, px, w, h);
      // sobel writes via dst pointer — re-read from pDst after call
      // (the wasm fn signature takes dst pointer)
      const b = mem.u8(pDst, px).slice();
      checks.push({ kernel: 'sobel', diff: firstMismatch(a, b) });
    }
    {
      const a = ref.boxBlur(luma, new Uint8Array(px), new Uint8Array(px), w, h, 8);
      mem.u8(pLumaIn, px).set(luma);
      wasmModule.box_blur(pLumaIn, pDst, pScratch, w, h, 8);
      const b = mem.u8(pDst, px).slice();
      checks.push({ kernel: 'boxBlur', diff: firstMismatch(a, b) });
    }

    for (const c of checks) {
      results.push({
        size: label,
        kernel: c.kernel,
        kind: 'correctness',
        ok: c.diff === null,
        detail: c.diff,
      });
      log(
        c.diff === null
          ? `✅ ${label} ${c.kernel}: byte-identical`
          : `❌ ${label} ${c.kernel}: mismatch ${JSON.stringify(c.diff)}`
      );
    }

    // ---------- speed ----------
    const mpx = px / 1e6;

    const cases = [
      {
        kernel: 'grayscale',
        js: () => ref.grayscale(rgba, jsOut),
        wasm: () => wasmModule.grayscale(pRgbaIn, px * 4, pLumaOut, px),
      },
      {
        kernel: 'threshold',
        js: () => ref.threshold(luma, jsOut, 128),
        wasm: () => wasmModule.threshold_out(pLumaIn, pDst, px, 128),
      },
      {
        kernel: 'sobel',
        js: () => ref.sobel(luma, jsOut, w, h),
        wasm: () => wasmModule.sobel(pLumaIn, pDst, w, h),
      },
      {
        kernel: 'boxBlur',
        js: () => ref.boxBlur(luma, jsOut, jsScratch, w, h, 8),
        wasm: () => wasmModule.box_blur(pLumaIn, pDst, pScratch, w, h, 8),
      },
    ];

    for (const c of cases) {
      const jsStats = timeIt(c.js);
      const wasmStats = timeIt(c.wasm);

      const row = {
        size: label,
        kernel: c.kernel,
        kind: 'speed',
        mpx,
        jsMedian: jsStats.median,
        jsP95: jsStats.p95,
        wasmMedian: wasmStats.median,
        wasmP95: wasmStats.p95,
        speedup: jsStats.median / wasmStats.median,
        jsThroughput: mpx / (jsStats.median / 1000),
        wasmThroughput: mpx / (wasmStats.median / 1000),
      };
      results.push(row);
      log(
        `${label} ${c.kernel}: JS ${row.jsMedian.toFixed(2)}ms → Rust/WASM ` +
          `${row.wasmMedian.toFixed(2)}ms (${row.speedup.toFixed(2)}x)`
      );
    }

    mem.free(pRgbaIn, px * 4);
    mem.free(pLumaOut, px);
    mem.free(pLumaIn, px);
    mem.free(pDst, px);
    mem.free(pScratch, px);
  }

  return results;
}

export { makeFixtures, median, p95, KERNELS };
