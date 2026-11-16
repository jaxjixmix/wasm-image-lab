//! Pure-Rust image kernels.
//!
//! Design notes that matter for the research question:
//!
//! 1. **No allocation in the hot path.** Every kernel takes caller-provided
//!    output (and scratch) buffers. This means the benchmark measures arithmetic,
//!    not `malloc`, on both sides of the comparison.
//! 2. **Integer math only.** Float rounding differs between JS (`f64`) and Rust
//!    (`f32`/`f64`), which would make byte-for-byte differential testing
//!    impossible. All kernels here are exact integer arithmetic so a mismatch is
//!    a real bug, not a rounding artifact.
//! 3. **Clamped edges.** Edge behaviour is defined explicitly instead of relying
//!    on library defaults, so the JS reference can match it exactly.

pub mod blur;
pub mod color;
pub mod edge;

pub use blur::box_blur;
pub use color::{grayscale, threshold, threshold_inplace};
pub use edge::sobel;
