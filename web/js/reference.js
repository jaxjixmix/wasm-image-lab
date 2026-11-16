/**
 * JS reference implementations.
 *
 * Deliberately written the way a competent frontend developer would write them:
 * straightforward loops over typed arrays. NOT deliberately sabotaged — if I
 * handicaps JS the benchmark is worthless.
 *
 * Optimisations that ARE included because they are idiomatic and obvious:
 * - typed arrays, not plain arrays
 * - `|0` coercion avoided; rely on V8's Smi fast path
 * - hoisted length lookups in the inner loops
 *
 * Deliberately excluded, to be tested separately later:
 * - SIMD via WebAssembly
 * - `OffscreenCanvas` / `filter: blur()` GPU paths
 * - Workers
 */

export function grayscale(rgba, out) {
  const px = rgba.length >>> 2;
  for (let i = 0; i < px; i++) {
    const b = i << 2;
    // (299r + 587g + 114b) / 1000, integer-only to match Rust bit-exactly
    out[i] = ((rgba[b] * 299 + rgba[b + 1] * 587 + rgba[b + 2] * 114) / 1000) | 0;
  }
  return out;
}

export function threshold(luma, out, t) {
  const n = luma.length;
  for (let i = 0; i < n; i++) {
    out[i] = luma[i] >= t ? 255 : 0;
  }
  return out;
}

export function sobel(src, out, w, h) {
  const at = (x, y) => {
    const xx = x < w ? x : w - 1;
    const yy = y < h ? y : h - 1;
    return src[yy * w + xx];
  };

  for (let y = 0; y < h; y++) {
    const y0 = y > 0 ? y - 1 : 0;
    const y2 = y + 1 < h ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const x0 = x > 0 ? x - 1 : 0;
      const x2 = x + 1 < w ? x + 1 : w - 1;

      const tl = at(x0, y0), tc = at(x, y0), tr = at(x2, y0);
      const ml = at(x0, y), mr = at(x2, y);
      const bl = at(x0, y2), bc = at(x, y2), br = at(x2, y2);

      const gx = tr + 2 * mr + br - (tl + 2 * ml + bl);
      const gy = bl + 2 * bc + br - (tl + 2 * tc + tr);
      const mag = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
      out[y * w + x] = mag > 255 ? 255 : mag;
    }
  }
  return out;
}

/**
 * Separable box blur with a sliding-window running sum — the same algorithmic
 * approach as the Rust version, so the comparison is implementation quality,
 * not algorithm choice. This is the honest way to run the experiment.
 */
export function boxBlur(src, out, scratch, w, h, r) {
  if (r === 0) {
    out.set(src.subarray(0, w * h));
    return out;
  }
  const win = 2 * r + 1;

  // Pass 1: horizontal -> scratch
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let sum = src[row] * r;

    for (let x = 0; x <= r; x++) {
      sum += src[row + (x < w ? x : w - 1)];
    }

    for (let x = 0; x < w; x++) {
      scratch[row + x] = (sum / win) | 0;

      const outX = x >= r ? x - r : 0;
      sum -= src[row + outX];

      const inRaw = x + r + 1;
      sum += src[row + (inRaw < w ? inRaw : w - 1)];
    }
  }

  // Pass 2: vertical scratch -> out
  for (let x = 0; x < w; x++) {
    let sum = scratch[x] * r;

    for (let y = 0; y <= r; y++) {
      sum += scratch[(y < h ? y : h - 1) * w + x];
    }

    for (let y = 0; y < h; y++) {
      out[y * w + x] = (sum / win) | 0;

      const outY = y >= r ? y - r : 0;
      sum -= scratch[outY * w + x];

      const inRaw = y + r + 1;
      sum += scratch[(inRaw < h ? inRaw : h - 1) * w + x];
    }
  }
  return out;
}
