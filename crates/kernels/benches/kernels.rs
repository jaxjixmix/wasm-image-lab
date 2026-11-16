//! Criterion benchmarks for the native Rust side.
//!
//! These give the "what does Rust cost on this machine, with no JS in the way"
//! baseline. The WASM numbers in the harness are compared against these to
//! quantify the wasm boundary + runtime overhead separately from raw compute.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use kernels::{box_blur, grayscale, sobel, threshold};

fn fixture(w: usize, h: usize) -> Vec<u8> {
    // Deterministic pseudo-random RGBA: cheap, reproducible, no dependency.
    let mut v = Vec::with_capacity(w * h * 4);
    let mut s: u32 = 0x1234_5678;
    for _ in 0..w * h {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s >> 24) as u8);
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s >> 24) as u8);
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s >> 24) as u8);
        v.push(255);
    }
    v
}

fn benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("image_kernels");

    for &(w, h) in &[(512usize, 512usize), (1920, 1080)] {
        let px = (w * h) as u64;
        let rgba = fixture(w, h);
        let mut luma = vec![0u8; w * h];
        let mut out_luma = vec![0u8; w * h];
        let mut scratch = vec![0u8; w * h];
        let mut out_blur = vec![0u8; w * h];

        group.throughput(Throughput::Elements(px));

        group.bench_with_input(BenchmarkId::new("grayscale", px), &px, |b, _| {
            b.iter(|| grayscale(black_box(&rgba), black_box(&mut luma)));
        });

        grayscale(&rgba, &mut luma);

        group.bench_with_input(BenchmarkId::new("threshold", px), &px, |b, _| {
            b.iter(|| threshold(black_box(&luma), black_box(&mut out_luma), 128));
        });

        group.bench_with_input(BenchmarkId::new("sobel", px), &px, |b, _| {
            b.iter(|| sobel(black_box(&luma), black_box(&mut out_luma), w, h));
        });

        group.bench_with_input(BenchmarkId::new("box_blur_r8", px), &px, |b, _| {
            b.iter(|| {
                box_blur(
                    black_box(&luma),
                    black_box(&mut out_blur),
                    black_box(&mut scratch),
                    w,
                    h,
                    8,
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches_group, benches);
criterion_main!(benches_group);
