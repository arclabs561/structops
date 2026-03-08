# structops

Differentiable operators for discrete structure.

This crate is a small collection of “structured operators” that show up across the stack:
dynamic programming relaxations, structured attention, and structured inference primitives.

## Quickstart

```toml
[dependencies]
structops = "0.1.0"
```

## What’s here

- `soft_dtw`: Soft-DTW (Cuturi & Blondel 2017) and the common debiased Soft‑DTW divergence.
- `soft_shortest_path`: Soft shortest path on a DAG (Mensch & Blondel 2018 framing), including
  edge marginals (a DP-shaped “attention” distribution over edges).

## Public invariants (must not change)

- **Backend-agnostic core**: slice-based APIs and `Vec<f64>` internals (no tensor backend types in public APIs).
- **Determinism**: no RNG in core operators.
- **Explicit smoothing parameters**: e.g. $\gamma$ must be passed and validated.

## Examples

```bash
# soft shortest path = “DP attention” on a tiny DAG
cargo run -p structops --example soft_path_attention

# Soft-DTW used for ordered (sequence-aware) sentence alignment
cargo run -p structops --example sentence_alignment_soft_dtw

# Soft-DTW for shift detection (sanity check / visualization)
cargo run -p structops --example soft_dtw_shift_scan

# User journey alignment to a canonical “golden path”
cargo run -p structops --example user_journey_alignment
```

## References

- M. Cuturi, M. Blondel. “Soft-DTW: a Differentiable Loss Function for Time-Series.” ICML 2017.
- A. Mensch, M. Blondel. “Differentiable Dynamic Programming for Structured Prediction and Attention.” ICML 2018.
