//! Separable box blur.
//!
//! Separable = horizontal pass then vertical pass, which turns an O(r^2)
//! convolution into O(2r). This is the classic place where a naive JS
//! implementation (nested loops over a 2-D neighbourhood) loses badly to a
//! cache-aware, allocation-free implementation.
//!
//! Hot-path rules enforced here:
//! - no allocation (scratch buffer is supplied by the caller)
//! - no bounds-check-free `unsafe`; the loop bounds are structured so LLVM can
//!   eliminate redundant checks on its own
//! - running-sum sliding window, so cost is independent of radius

/// Box blur over a single-channel u8 image, clamped edges.
///
/// `scratch` must be at least `w * h` bytes; it is used as an intermediate
/// buffer and its contents are not preserved.
pub fn box_blur(src: &[u8], out: &mut [u8], scratch: &mut [u8], w: usize, h: usize, r: usize) {
    assert!(w > 0 && h > 0);
    assert!(src.len() >= w * h, "src too small");
    assert!(out.len() >= w * h, "out too small");
    assert!(scratch.len() >= w * h, "scratch too small");

    if r == 0 {
        out[..w * h].copy_from_slice(&src[..w * h]);
        return;
    }

    let win = (2 * r + 1) as u32;

    // --- Pass 1: horizontal -> scratch ---
    for y in 0..h {
        let row = y * w;
        // Seed the running sum with the clamped left edge repeated.
        let mut sum: u32 = 0;
        let left = src[row] as u32;
        sum += left * (r as u32);

        for x in 0..=r {
            let xx = if x < w { x } else { w - 1 };
            sum += src[row + xx] as u32;
        }

        for x in 0..w {
            scratch[row + x] = (sum / win) as u8;

            // Slide: drop the pixel leaving the window, add the entering one.
            let out_x = x.saturating_sub(r);
            sum -= src[row + out_x] as u32;

            let in_raw = x + r + 1;
            let in_x = if in_raw < w { in_raw } else { w - 1 };
            sum += src[row + in_x] as u32;
        }
    }

    // --- Pass 2: vertical scratch -> out ---
    for x in 0..w {
        let mut sum: u32 = 0;
        let top = scratch[x] as u32;
        sum += top * (r as u32);

        for y in 0..=r {
            let yy = if y < h { y } else { h - 1 };
            sum += scratch[yy * w + x] as u32;
        }

        for y in 0..h {
            out[y * w + x] = (sum / win) as u8;

            let out_y = y.saturating_sub(r);
            sum -= scratch[out_y * w + x] as u32;

            let in_raw = y + r + 1;
            let in_y = if in_raw < h { in_raw } else { h - 1 };
            sum += scratch[in_y * w + x] as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radius_zero_is_identity() {
        let src = [10u8, 20, 30, 40];
        let mut out = [0u8; 4];
        let mut scratch = [0u8; 4];
        box_blur(&src, &mut out, &mut scratch, 2, 2, 0);
        assert_eq!(out, src);
    }

    #[test]
    fn uniform_image_is_preserved() {
        let w = 8;
        let h = 8;
        let src = vec![77u8; w * h];
        let mut out = vec![0u8; w * h];
        let mut scratch = vec![0u8; w * h];
        box_blur(&src, &mut out, &mut scratch, w, h, 3);
        // Integer division may shave a little, but a uniform field must stay
        // within 1 of the original everywhere.
        for v in &out {
            assert!((*v as i32 - 77).abs() <= 1, "got {v}");
        }
    }

    #[test]
    fn impulse_spreads_symmetrically() {
        let w = 9;
        let h = 9;
        let mut src = vec![0u8; w * h];
        src[4 * w + 4] = 255;
        let mut out = vec![0u8; w * h];
        let mut scratch = vec![0u8; w * h];
        box_blur(&src, &mut out, &mut scratch, w, h, 1);

        // Centre is the mean of a 3x3 with one lit pixel -> 255/9 = 28
        assert_eq!(out[4 * w + 4], 28);
        // Symmetry around the centre must hold.
        assert_eq!(out[3 * w + 3], out[5 * w + 5]);
        assert_eq!(out[3 * w + 4], out[5 * w + 4]);
    }
}
