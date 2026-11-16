//! Sobel edge detection (single-channel in, u8 magnitude out).

/// Sobel gradient magnitude, clamped edges.
///
/// Gx and Gy are computed with integer kernels, then the magnitude is
/// approximated as `|Gx| + |Gy|` clamped to 255 rather than `sqrt(Gx^2+Gy^2)`.
/// That choice keeps the JS reference bit-exact and avoids float divergence,
/// at the cost of a small angular error — documented in the README.
pub fn sobel(src: &[u8], out: &mut [u8], w: usize, h: usize) {
    assert!(w > 0 && h > 0);
    assert!(src.len() >= w * h, "src too small");
    assert!(out.len() >= w * h, "out too small");

    let at = |x: usize, y: usize| -> i32 {
        let xx = x.min(w - 1);
        let yy = y.min(h - 1);
        src[yy * w + xx] as i32
    };

    for y in 0..h {
        // Clamped row indices, computed once per row instead of per pixel.
        let y0 = y.saturating_sub(1);
        let y2 = (y + 1).min(h - 1);

        for x in 0..w {
            let x0 = x.saturating_sub(1);
            let x2 = (x + 1).min(w - 1);

            let tl = at(x0, y0);
            let tc = at(x, y0);
            let tr = at(x2, y0);
            let ml = at(x0, y);
            let mr = at(x2, y);
            let bl = at(x0, y2);
            let bc = at(x, y2);
            let br = at(x2, y2);

            let gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
            let gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);

            let mag = gx.abs() + gy.abs();
            out[y * w + x] = if mag > 255 { 255 } else { mag as u8 };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_image_has_no_edges() {
        let w = 6;
        let h = 6;
        let src = vec![42u8; w * h];
        let mut out = vec![9u8; w * h];
        sobel(&src, &mut out, w, h);
        assert!(out.iter().all(|v| *v == 0));
    }

    #[test]
    fn vertical_step_produces_response() {
        // Left half 0, right half 255 -> strong vertical edge.
        let w = 8;
        let h = 8;
        let mut src = vec![0u8; w * h];
        for y in 0..h {
            for x in 4..w {
                src[y * w + x] = 255;
            }
        }
        let mut out = vec![0u8; w * h];
        sobel(&src, &mut out, w, h);

        // Interior columns adjacent to the step must be strongly lit.
        assert!(out[4 * w + 3] > 200, "left side of edge");
        assert!(out[4 * w + 4] > 200, "right side of edge");
        // Far from the step there should be nothing.
        assert_eq!(out[4 * w + 0], 0);
    }
}
