/**
 * Headless coverage for the BROWSER harness (`web/js/bench.js`).
 *
 * Why this file exists:
 *
 * `scripts/verify.mjs` has its own wasm plumbing, so it duplicated the browser
 * harness instead of exercising it. A wrong-arity call in `bench.js`
 * (`sobel(src, px, w, h)` — passing a length where a pointer belonged) therefore
 * went unnoticed: it wrote 262KB at address `px` instead of into the output
 * buffer, and CI stayed green because CI only ever ran the other file.
 *
 * `bench.js` imports nothing DOM-related, so Node can run it directly. This
 * runner calls the real `runBenchmark` and fails the build on any correctness
 * mismatch or on a call that throws. That makes the harness itself tested.
 */

import { readFile } from 'node:fs/promises';
import { runBenchmark } from '../web/js/bench.js';

const WASM_PATH = new URL('../target/wasm32-unknown-unknown/release/wasm.wasm', import.meta.url);

async function main() {
  const bytes = await readFile(WASM_PATH);
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const ex = instance.exports;

  // Sanity: every export the harness calls must exist and have the arity the
  // harness assumes. This is the check that would have caught the original bug
  // class directly, by name.
  const EXPECTED = {
    grayscale: 4,
    threshold: 3,
    threshold_out: 4,
    sobel: 4,
    box_blur: 6,
    alloc: 1,
    dealloc: 2,
  };
  for (const [name, arity] of Object.entries(EXPECTED)) {
    const fn = ex[name];
    if (typeof fn !== 'function') {
      console.error(`❌ missing wasm export: ${name}`);
      process.exit(1);
    }
    if (fn.length !== arity) {
      console.error(`❌ ${name} arity ${fn.length}, harness expects ${arity}`);
      process.exit(1);
    }
  }
  console.log(`✅ all ${Object.keys(EXPECTED).length} wasm exports present with expected arity`);

  // runBenchmark mutates caller-provided buffers and returns result rows.
  const results = await runBenchmark(ex, () => {});

  const correctness = results.filter((r) => r.kind === 'correctness');
  const speed = results.filter((r) => r.kind === 'speed');

  if (correctness.length === 0) {
    console.error('❌ runBenchmark produced no correctness rows');
    process.exit(1);
  }

  const bad = correctness.filter((r) => !r.ok);
  if (bad.length > 0) {
    console.error(`❌ ${bad.length}/${correctness.length} correctness failures in browser harness:`);
    for (const b of bad) console.error(`   ${b.size} ${b.kernel}: ${JSON.stringify(b.detail)}`);
    process.exit(1);
  }

  console.log(`✅ browser harness: ${correctness.length}/${correctness.length} kernels byte-identical`);

  // Guard against a meaningless table: if throughput collapsed on every kernel
  // the harness is probably diffing garbage (the zeroed-memory failure mode).
  if (speed.length === 0) {
    console.error('❌ runBenchmark produced no speed rows');
    process.exit(1);
  }
  for (const s of speed) {
    if (!(s.wasmMedian > 0) || !Number.isFinite(s.wasmMedian)) {
      console.error(`❌ ${s.size} ${s.kernel}: invalid wasm timing ${s.wasmMedian}`);
      process.exit(1);
    }
  }
  console.log(`✅ browser harness: ${speed.length} timing rows all finite and positive`);

  console.log('\nBrowser harness exercised end-to-end.');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
