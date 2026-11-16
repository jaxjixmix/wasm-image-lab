//! Raw wasm boundary — no `wasm-bindgen`, no generated JS.
//!
//! Why no bindgen:
//!  - a `cdylib` built for `wasm32-unknown-unknown` exports its memory, so JS
//!    can create typed-array views directly over linear memory;
//!  - `i32` pointers and lengths are the whole ABI, so there is nothing to
//!    generate;
//!  - fewer moving parts means the measured time is the kernel, not glue.
//!
//! All functions are `extern "C"` + `#[no_mangle]` so the linker keeps them.

use std::alloc::{alloc as raw_alloc, dealloc as raw_dealloc, Layout};

/// Bytes reserved per allocation header. We stash the length immediately before
/// the returned pointer so `dealloc` needs only `(ptr, len)` — actually we store
/// nothing and require the caller to pass `len`, which keeps this zero-overhead.
const ALIGN: usize = 4;

/// Allocate `len` bytes and return a pointer into wasm linear memory.
///
/// Returns 0 on failure. The caller owns the block and must call `dealloc`
/// with the same `len`.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    match Layout::from_size_align(len, ALIGN) {
        Ok(layout) => unsafe { raw_alloc(layout) },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a block from `alloc`. Length must match the original request exactly.
///
/// # Safety
/// `ptr` must come from `alloc(len)` and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    if let Ok(layout) = Layout::from_size_align(len, ALIGN) {
        raw_dealloc(ptr, layout);
    }
}

/// RGBA8 -> luma. `src_len` is bytes, `dst_len` is pixels.
///
/// # Safety
/// Both pointers must be valid for the stated lengths within linear memory.
#[no_mangle]
pub unsafe extern "C" fn grayscale(
    src_ptr: *const u8,
    src_len: usize,
    dst_ptr: *mut u8,
    dst_len: usize,
) {
    let src = std::slice::from_raw_parts(src_ptr, src_len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, dst_len);
    kernels::grayscale(src, dst);
}

/// In-place luma threshold.
///
/// # Safety
/// `ptr` must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn threshold(ptr: *mut u8, len: usize, t: u8) {
    let buf = std::slice::from_raw_parts_mut(ptr, len);
    kernels::threshold_inplace(buf, t);
}

/// Out-of-place luma threshold.
///
/// Exists so the benchmark can compare like with like: the JS reference reads
/// one buffer and writes another, so the wasm side must do the same. Comparing
/// in-place wasm against out-of-place JS measures memory traffic, not Rust.
///
/// # Safety
/// Both pointers must be valid for `len` bytes.
#[no_mangle]
pub unsafe extern "C" fn threshold_out(src_ptr: *const u8, dst_ptr: *mut u8, len: usize, t: u8) {
    let src = std::slice::from_raw_parts(src_ptr, len);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, len);
    kernels::threshold(src, dst, t);
}

/// Luma -> Sobel edge magnitude.
///
/// # Safety
/// Pointers must be valid for `w * h` bytes each.
#[no_mangle]
pub unsafe extern "C" fn sobel(src_ptr: *const u8, dst_ptr: *mut u8, w: usize, h: usize) {
    let n = w * h;
    let src = std::slice::from_raw_parts(src_ptr, n);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, n);
    kernels::sobel(src, dst, w, h);
}

/// Luma -> separable box blur. `scratch_ptr` must hold `w * h` bytes and is
/// caller-owned so it can be reused across calls.
///
/// # Safety
/// All pointers must be valid for at least `w * h` bytes.
#[no_mangle]
pub unsafe extern "C" fn box_blur(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    scratch_ptr: *mut u8,
    w: usize,
    h: usize,
    r: usize,
) {
    let n = w * h;
    let src = std::slice::from_raw_parts(src_ptr, n);
    let dst = std::slice::from_raw_parts_mut(dst_ptr, n);
    let scratch = std::slice::from_raw_parts_mut(scratch_ptr, n);
    kernels::box_blur(src, dst, scratch, w, h, r);
}
