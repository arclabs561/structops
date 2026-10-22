# Changelog

All notable changes to this project are documented here. Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `soft_dtw_cost_grad`, `soft_dtw_grad` and `soft_dtw_divergence_grad`: Soft-DTW
  value plus gradient via the backward recursion of Cuturi & Blondel (2017).

### Fixed

- Preserve Soft-DTW partition mass and expected-alignment gradients for tiny
  positive temperatures and large cost offsets; avoid intermediate subtraction
  overflow when opposite-sign finite costs have a finite scaled difference.
- Attribute the Soft-DTW divergence to Blondel, Mensch & Vert (2021).
- README no longer claims end-to-end training for operators without
  gradients, and states that `control` uses `f32`.

### Changed

- Clarify that the debiased Soft-DTW divergence is self-zeroing in exact
  arithmetic, while nonnegativity is not guaranteed for arbitrary cost
  matrices; describe the gradient as expected alignment.

## [0.2.0] - 2026-06-11

### Added

- Relaxed (soft) control-flow operators.
- `soft_path_on_graph` example.
- Soft-DTW invariant tests (gamma to 0 vs DTW, diagonal bias).

### Changed

- Refined the soft control-flow API.

### Fixed

- Corrected error semantics and added NaN validation.

### Removed

- Dead code.

[0.2.0]: https://github.com/arclabs561/structops/releases/tag/v0.2.0
