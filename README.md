# structops

Differentiable operators for discrete structure.

structops implements continuous relaxations of discrete operations so they can
sit inside a gradient-trained model. A hard `min`, an `if`/`else` branch, or a
dynamic-programming recurrence has a zero or undefined gradient, which blocks
backpropagation through any model that needs to align two sequences, route over
a graph, or branch on a learned condition. Each operator replaces the hard
operation with a temperature- or smoothing-parameterized version: it recovers
the discrete semantics in the zero-temperature limit and is differentiable
everywhere else. Soft-DTW and soft shortest path return their gradients; the
`control` operators have no gradient helpers, so differentiate them in your own
autodiff. Inputs and outputs are plain slices and `Vec`, so the operators drop
into any backend.

## Quickstart

```toml
[dependencies]
structops = "0.2"
```

```rust
use structops::soft_shortest_path::{soft_shortest_path_edge_marginals, Edge};

// Two paths from source 0 to sink 3: 0->1->3 (cheap) and 0->2->3 (costly).
let edges = [
    Edge { from: 0, to: 1, cost: 1.0 },
    Edge { from: 1, to: 3, cost: 1.0 },
    Edge { from: 0, to: 2, cost: 3.0 },
    Edge { from: 2, to: 3, cost: 3.0 },
];

// `gamma` smooths min into log-sum-exp. `p[k]` is the marginal probability
// that edge k lies on a path, equal to d(value)/d(cost_k): a differentiable
// attention distribution over edges.
let (value, p) = soft_shortest_path_edge_marginals(4, &edges, 0.5).unwrap();
assert!(p[0] > p[2]); // the cheaper path carries more weight
```

The same graph and `gamma` in `cargo run --release --example soft_path_attention`:

```text
soft shortest-path value = 1.9998322968135522
edge 0 (0->1, cost=1): marginal p(e in path) = 0.9996646498695337
edge 1 (1->3, cost=1): marginal p(e in path) = 0.9996646498695337
edge 2 (0->2, cost=3): marginal p(e in path) = 0.00033535013046647805
edge 3 (2->3, cost=3): marginal p(e in path) = 0.00033535013046647805
```

## What's here

- `soft_dtw`: Soft-DTW (Cuturi & Blondel 2017) and the debiased Soft-DTW
  divergence (Blondel, Mensch & Vert 2021). `soft_dtw` / `soft_dtw_cost` for the
  value, `soft_dtw_divergence` / `soft_dtw_divergence_cost` for the debiased
  variant (self-zeroing in exact arithmetic), and `soft_dtw_cost_grad` / `soft_dtw_grad` /
  `soft_dtw_divergence_grad` for value plus gradient (Cuturi & Blondel's
  expected alignment). Nonnegativity is not guaranteed for arbitrary supplied
  cost matrices. Use it to score how well two sequences align.
- `soft_shortest_path`: soft shortest path on a DAG (Mensch & Blondel 2018
  framing). `soft_shortest_path_edge_marginals` returns the value and the
  per-edge marginals; those marginals are the gradient w.r.t. edge cost, i.e. a
  DP-shaped attention distribution over edges.
- `control`: relaxed control flow (Petersen et al. 2021). Soft comparisons
  (`soft_lt`, `soft_gt`, `soft_eq`) and soft conditionals (`soft_if`,
  `soft_if_scalar`) approach their hard counterparts as `beta` grows or
  `temperature` goes to 0. The bounded soft loops (`soft_while`, `soft_for`)
  blend each step by a caller-supplied continuation probability or weight, and
  become hard loops when those are 0 or 1.

## Public invariants (must not change)

- **Backend-agnostic core**: slice-based APIs (no tensor backend types in public APIs). `soft_dtw` and `soft_shortest_path` use `f64`; `control` uses `f32`.
- **Determinism**: no RNG in core operators.
- **Explicit smoothing parameters**: e.g. `gamma` must be passed and validated.

## Examples

See [examples/README.md](examples/README.md) for runnable examples with
captured output.

## References

- M. Cuturi, M. Blondel. "Soft-DTW: a Differentiable Loss Function for Time-Series." ICML 2017.
- M. Blondel, A. Mensch, J.-P. Vert. "Differentiable Divergences Between Time Series." AISTATS 2021.
- A. Mensch, M. Blondel. "Differentiable Dynamic Programming for Structured Prediction and Attention." ICML 2018.
- F. Petersen, C. Borgelt, H. Kuehne, O. Deussen. "Learning with Algorithmic Supervision via Continuous Relaxations." NeurIPS 2021.

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.
