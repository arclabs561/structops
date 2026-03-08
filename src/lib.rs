//! Differentiable operators for discrete structure.
//!
//! This crate is intended to hold “structured operators” that show up across the stack:
//! dynamic programming relaxations, structured attention, and sparse structured inference.
//!
//! Public invariants (must not change):
//! - APIs are backend-agnostic (slice-based, `Vec<f64>` outputs).
//! - Numeric code is deterministic (no RNG in core ops).
//! - Parameters that control smoothing (e.g. \(\gamma\)) are explicit and validated.

pub mod soft_dtw;
pub mod soft_shortest_path;
