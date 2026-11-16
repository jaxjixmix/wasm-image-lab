//! Colour-space and pointwise kernels.

/// RGBA8 -> luma (single channel).
///
/// Uses Rec. 601 weights in fixed point: `(299R + 587G + 114B) / 1000`.
/// Integer-only so the JS reference can reproduce it bit-exactly.
#[inline]
pub fn grayscale(rgba: &[u8], out: &mut [u8]) {
    assert_eq!(rgba.len() % 4, 0, "input must be RGBA8");
    let px = rgba.len() / 4;
    assert!(out.len() >= px, "output too small");

    for i in 0..px {
        let b = i * 4;
        let r = rgba[b] as u32;
        let g = rgba[b + 1] as u32;
        let bl = rgba[b + 2] as u32;
        out[i] = ((r * 299 + g * 587 + bl * 114) / 1000) as u8;
    }
}

/// Binary threshold on a luma buffer.
#[inline]
pub fn threshold(luma: &[u8], out: &mut [u8], t: u8) {
    assert!(out.len() >= luma.len(), "output too small");
    for i in 0..luma.len() {
        out[i] = if luma[i] >= t { 255 } else { 0 };
    }
}

/// Binary threshold, in place.
///
/// Separate from [`threshold`] because Rust forbids passing one buffer as both
/// `&[u8]` and `&mut [u8]` — the borrow checker is the point of the language, so
/// the API bends rather than the safety rule.
#[inline]
pub fn threshold_inplace(buf: &mut [u8], t: u8) {
    for v in buf.iter_mut() {
        *v = if *v >= t { 255 } else { 0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grayscale_known_values() {
        // Pure red -> 0.299 * 255 = 76.245 -> 76
        let src = [255u8, 0, 0, 255];
        let mut out = [0u8; 1];
        grayscale(&src, &mut out);
        assert_eq!(out[0], 76);

        // Pure white -> 255
        let src = [255u8, 255, 255, 255];
        let mut out = [0u8; 1];
        grayscale(&src, &mut out);
        assert_eq!(out[0], 255);
    }

    #[test]
    fn threshold_inplace_matches_allocating_version() {
        let src = [0u8, 127, 128, 255, 64];
        let mut a = src;
        let mut b = [0u8; 5];
        threshold_inplace(&mut a, 128);
        threshold(&src, &mut b, 128);
        assert_eq!(a, b);
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        let src = [0u8, 127, 128, 255];
        let mut out = [0u8; 4];
        threshold(&src, &mut out, 128);
        assert_eq!(out, [0, 0, 255, 255]);
    }
}
