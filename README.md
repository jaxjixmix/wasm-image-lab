# wasm-image-lab

Differential-tested, honestly-benchmarked Rust → WebAssembly image kernels.

The question this repo answers: **when you rewrite a hot JS routine in Rust and
compile it to WASM, how much do you actually win, and can you prove the output
is still correct?**

Most "rewrite it in Rust" benchmarks are unusable because they (a) never check
correctness, (b) handicap the JS side, or (c) hide glue-code overhead. This one
is built to avoid all three.

## The claim under test

> For memory-bound integer image kernels, Rust/WASM beats idiomatic JS by a
> measurable factor — but the factor depends entirely on the kernel's shape, and
> is nowhere near the 10x folklore.

## What's measured

| Kernel | Shape | Work per pixel |
|---|---|---|
| `grayscale` | pointwise, 4-byte read → 1-byte write | 3 mults + adds |
| `threshold` | pointwise, 1-byte read → 1-byte write | 1 compare |
| `sobel` | 3×3 stencil | 8 reads, 12 adds |
| `box_blur` (r=8) | separable, sliding window | O(1) per pixel regardless of radius |

Sizes: 512×512 and 1920×1080. Median of 20 runs after 5 warmup runs.

## Results (Apple M-series, Node v24 / V8, Rust 1.100, `--release`, lto=true)

### Correctness — 8/8 byte-identical

Every kernel produces output identical to the JS reference, checked byte by
byte with a first-mismatch reporter. This is the part most benchmarks skip.

### Speed

**512×512 (0.26 MP)**

| Kernel | JS median | WASM median | Speedup | WASM throughput |
|---|---|---|---|---|
| grayscale | 0.37 ms | 0.20 ms | **1.85x** | 1328 MP/s |
| threshold | 0.96 ms | 0.08 ms | **12.1x** | 3281 MP/s |
| sobel | 3.78 ms | 1.09 ms | **3.46x** | 240 MP/s |
| box_blur r=8 | 1.39 ms | 0.66 ms | **2.12x** | 399 MP/s |

**1920×1080 (2.07 MP)**

| Kernel | JS median | WASM median | Speedup | WASM throughput |
|---|---|---|---|---|
| grayscale | 2.89 ms | 1.56 ms | **1.85x** | 1328 MP/s |
| threshold | 8.24 ms | 0.64 ms | **12.8x** | 3223 MP/s |
| sobel | 31.80 ms | 8.73 ms | **3.64x** | 237 MP/s |
| box_blur r=8 | 11.15 ms | 5.17 ms | **2.16x** | 401 MP/s |

Note the throughput columns are flat across image sizes — that means the
measurement is dominated by per-pixel work, not per-call overhead. Good sign.

## The interesting result: `threshold` at 12x

That number looked wrong to me, so I tested it. I rewrote the JS loop five ways
(hoisted length, branchless, `while`, cached length) and V8 stayed pinned at
**~273 MP/s** for all of them. The wasm version does 3,223 MP/s — twelve times
faster on a kernel that is a single compare and store.

Interpretation: this is not a Rust-vs-JS algorithmic win. It is V8's
typed-array store path being much slower than a straight `u8` store in compiled
wasm for this access pattern. The lesson for a Rust-rewrite decision is
important and general:

> A 12x win on `threshold` does **not** mean "rewrite everything in Rust."
> It means one specific V8 code path is weak, and profiling found it.

`grayscale`, by contrast, is only 1.85x — it reads 4 bytes per output byte, so
it is closer to memory-bandwidth-bound, where V8 and wasm converge. That is the
more representative number for real image work.

## Why the JS side is not sabotaged

`web/js/reference.js` is written the way a competent frontend dev would write
it: typed arrays, hoisted loop bounds, no pointless allocation in the loop. It
uses the **same algorithm** as the Rust side (separable blur with a sliding
window running sum, not a naive 2-D convolution). If I had compared
sliding-window Rust against naive JS, the speedup would be meaningless.

Deliberately *excluded* from the JS baseline, and left as follow-up work:
`OffscreenCanvas` GPU filters, Web Workers, and SIMD. Those are the real
competitors to WASM and the honest next experiment.

## Design decisions worth reading

1. **No `wasm-bindgen`.** The module is a plain `cdylib` exporting `memory` and
   C-ABI functions taking `(ptr, len)`. JS makes typed-array views directly over
   linear memory. Fewer moving parts, and the benchmark measures the kernel
   instead of generated glue.
2. **No allocation in the hot path.** Buffers are allocated once, reused across
   all runs, freed at the end. Both sides do this.
3. **Integer math only.** Rec. 601 weights in fixed point (`(299r+587g+114b)/1000`),
   and `|Gx|+|Gy|` for Sobel magnitude instead of `sqrt(Gx²+Gy²)`. Float rounding
   differs between JS `f64` and Rust, which would make byte-exact differential
   testing impossible. Documented tradeoff: the Sobel magnitude has a small
   angular error versus the Euclidean norm.
4. **Clamped edges, defined explicitly.** No reliance on library defaults, so the
   JS reference can reproduce edge behaviour exactly.
5. **`threshold_inplace` exists because the borrow checker said so.** Rust
   forbids passing one buffer as both `&[u8]` and `&mut [u8]`. The API bent; the
   safety rule did not. Small but instructive example of ownership shaping design.
6. **One clippy lint is deliberately allowed, with evidence.** Clippy prefers an
   `as_chunks` iterator in `grayscale`. I tried it, and it cost ~15% throughput
   (1122 MP/s vs 1328 MP/s at 1920x1080) because LLVM can already prove the
   index-loop offsets don't alias. The `#[allow]` carries that measurement in a
   comment. Lint hygiene is not worth a 15% regression in a crate whose whole
   purpose is speed.

## Structure

```
crates/kernels/     pure Rust, no wasm, no JS — unit-tested + criterion benches
crates/wasm/        thin C-ABI surface, cdylib, zero glue
web/js/reference.js JS reference implementations
web/js/bench.js     browser benchmark + differential harness
web/index.html      results table + visual side-by-side
scripts/verify.mjs  headless Node verification (CI-able)
```

## Run it

```bash
cargo test -p kernels                       # 8 native unit tests
cargo bench -p kernels                      # criterion, native baseline
cargo build -p wasm --release --target wasm32-unknown-unknown
node scripts/verify.mjs                     # headless correctness + speed
npx http-server web -p 8080 -c-1            # then open localhost:8080
```

CI (`.github/workflows/verify.yml`) runs the unit tests, the differential
verification, and enforces `rustfmt` + `clippy -D warnings` on every push. The
correctness check gates the build, because without it the benchmark numbers are
meaningless.

## Known limitations

- Single-threaded; no Workers or `SharedArrayBuffer` yet.
- GC pauses leak into the JS timings; medians reduce but do not eliminate this.
- Tested on one machine, one V8 version. Numbers are indicative, not universal.
- Sobel uses `|Gx|+|Gy|`, not the Euclidean magnitude — deliberately, for
  bit-exactness.

## Follow-up experiments (the actual research programme)

1. **SIMD** — `wasm32` SIMD128 for all four kernels. Does it push Sobel past 237 MP/s?
2. **Workers** — JS baseline with N workers vs single-threaded WASM. Which wins, and at what N?
3. **GPU** — `OffscreenCanvas` filter pipeline vs WASM blur. Suspect GPU wins big on blur, loses on small images.
4. **Memory copies** — measure the cost of the JS↔wasm copy exactly, by timing a no-op kernel.
5. **Threshold mystery** — dig into V8's typed-array store path. Is it bounds-checking, or something else?

Each is a bounded experiment with a number at the end.
