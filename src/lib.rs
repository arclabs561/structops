#![doc = include_str!("../README.md")]
//! Differentiable operators for discrete structure.
//!
//! This crate provides dynamic programming relaxations, structured attention,
//! and relaxed control flow.
//!
//! Public invariants (must not change):
//! - APIs are backend-agnostic (slice-based; `soft_dtw` and `soft_shortest_path`
//!   use `f64`, `control` uses `f32`).
//! - Numeric code is deterministic (no RNG in core ops).
//! - Parameters that control smoothing (e.g. \(\gamma\)) are explicit and validated.

pub mod control;
pub mod soft_dtw;
pub mod soft_shortest_path;
