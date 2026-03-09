//! Differentiable dynamic programming on a DAG via softmin (Mensch & Blondel 2018 framing).
//!
//! We model a family of paths from a source `s=0` to a sink `t=n-1` in a DAG with
//! nodes ordered topologically (we assume edges satisfy `from < to`).
//!
//! Each edge \(e=(u\to v)\) has a cost \(c_e\). The **soft shortest-path value**
//! (a log-sum-exp relaxation of min) is:
//! \[
//! V_\gamma = -\gamma \log \sum_{\pi \in \mathcal{P}(s\to t)} \exp\left(-\frac{C(\pi)}{\gamma}\right),
//! \quad C(\pi)=\sum_{e\in\pi} c_e.
//! \]
//!
//! Key property (this is the point of the paper/approach):
//! \[
//! \frac{\partial V_\gamma}{\partial c_e} = \mathbb{P}_\gamma(e \in \pi),
//! \]
//! i.e. the gradient w.r.t. edge costs is the marginal probability that edge `e`
//! is used under the Gibbs distribution over paths.
//!
//! We expose those marginals explicitly via a forward-backward pass.

/// Errors for soft shortest-path operators.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum Error {
    /// Smoothing parameter \(\gamma\) must be positive and finite.
    #[error("gamma must be positive and finite, got {0}")]
    InvalidGamma(f64),
    /// Graph must have at least 2 nodes (source and sink).
    #[error("graph must have at least 2 nodes, got {0}")]
    TooFewNodes(usize),
    /// Edge endpoint out of bounds.
    #[error("edge endpoint out of bounds: edge {edge_idx} has ({from}->{to}) for n={n}")]
    EdgeOutOfBounds {
        /// Index of the offending edge in the provided slice.
        edge_idx: usize,
        /// Source endpoint of that edge.
        from: usize,
        /// Destination endpoint of that edge.
        to: usize,
        /// Number of nodes in the graph.
        n: usize,
    },
    /// Non-finite edge cost.
    #[error("non-finite edge cost at edge {edge_idx}: cost={cost}")]
    NonFiniteCost {
        /// Index of the offending edge in the provided slice.
        edge_idx: usize,
        /// The non-finite cost value.
        cost: f64,
    },
    /// DAG/topological invariant violated.
    #[error("expected DAG/topological order with from < to; edge {edge_idx} has ({from}->{to})")]
    NotDagOrder {
        /// Index of the offending edge in the provided slice.
        edge_idx: usize,
        /// Source endpoint of that edge.
        from: usize,
        /// Destination endpoint of that edge.
        to: usize,
    },
    /// No path exists from source to sink.
    #[error("no path exists from source to sink")]
    NoPath,
}

/// Convenience result type for this module.
pub type Result<T> = std::result::Result<T, Error>;

/// Directed edge in a DAG.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Source node index.
    pub from: usize,
    /// Destination node index.
    pub to: usize,
    /// Edge cost (finite `f64`).
    pub cost: f64,
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    let mut m = f64::NEG_INFINITY;
    for &x in xs {
        if x > m {
            m = x;
        }
    }
    if !m.is_finite() {
        return f64::NEG_INFINITY;
    }
    let mut s = 0.0;
    for &x in xs {
        s += (x - m).exp();
    }
    m + s.ln()
}

fn softmin_gamma(gamma: f64, candidates: &[f64], scratch: &mut Vec<f64>) -> f64 {
    // softmin(a_i) = -γ log Σ exp(-a_i/γ)
    scratch.clear();
    scratch.reserve(candidates.len());
    for &a in candidates {
        scratch.push(-a / gamma);
    }
    let lse = log_sum_exp(scratch);
    if lse == f64::NEG_INFINITY {
        f64::INFINITY
    } else {
        -gamma * lse
    }
}

fn validate(n: usize, edges: &[Edge]) -> Result<()> {
    if n < 2 {
        return Err(Error::TooFewNodes(n));
    }
    for (k, e) in edges.iter().enumerate() {
        if e.from >= n || e.to >= n {
            return Err(Error::EdgeOutOfBounds {
                edge_idx: k,
                from: e.from,
                to: e.to,
                n,
            });
        }
        if e.from >= e.to {
            return Err(Error::NotDagOrder {
                edge_idx: k,
                from: e.from,
                to: e.to,
            });
        }
        if !e.cost.is_finite() {
            return Err(Error::NonFiniteCost {
                edge_idx: k,
                cost: e.cost,
            });
        }
    }
    Ok(())
}

/// Compute edge marginals \(p_e = \mathbb{P}_\gamma(e \in \pi)\) for paths from 0 to n-1.
///
/// Returns `(value, edge_marginals)` where `edge_marginals.len() == edges.len()`.
pub fn soft_shortest_path_edge_marginals(
    n: usize,
    edges: &[Edge],
    gamma: f64,
) -> Result<(f64, Vec<f64>)> {
    if gamma <= 0.0 || !gamma.is_finite() {
        return Err(Error::InvalidGamma(gamma));
    }
    validate(n, edges)?;

    let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, e) in edges.iter().enumerate() {
        incoming[e.to].push(k);
        outgoing[e.from].push(k);
    }

    // Forward potentials (soft shortest from source to each node)
    let mut fwd = vec![f64::INFINITY; n];
    fwd[0] = 0.0;
    let mut scratch = Vec::new();
    let mut cands = Vec::new();
    for v in 1..n {
        cands.clear();
        for &ek in &incoming[v] {
            let e = edges[ek];
            let a = fwd[e.from];
            if a.is_finite() {
                cands.push(a + e.cost);
            }
        }
        fwd[v] = if cands.is_empty() {
            f64::INFINITY
        } else {
            softmin_gamma(gamma, &cands, &mut scratch)
        };
    }

    let value = fwd[n - 1];
    if !value.is_finite() {
        return Err(Error::NoPath);
    }

    // Backward potentials (soft shortest from each node to sink)
    let mut bwd = vec![f64::INFINITY; n];
    bwd[n - 1] = 0.0;
    for u_rev in 1..n {
        let u = n - 1 - u_rev;
        cands.clear();
        for &ek in &outgoing[u] {
            let e = edges[ek];
            let a = bwd[e.to];
            if a.is_finite() {
                cands.push(e.cost + a);
            }
        }
        bwd[u] = if cands.is_empty() {
            f64::INFINITY
        } else {
            softmin_gamma(gamma, &cands, &mut scratch)
        };
    }

    // Edge marginals:
    // p_e = exp(-(fwd[u] + c_e + bwd[v] - value)/gamma)
    let mut p = vec![0.0; edges.len()];
    for (k, e) in edges.iter().enumerate() {
        let a = fwd[e.from];
        let b = bwd[e.to];
        if a.is_finite() && b.is_finite() {
            let z = -((a + e.cost + b - value) / gamma);
            // prevent overflow in exp for extremely negative (shouldn’t happen much)
            p[k] = if z < -745.0 { 0.0 } else { z.exp() };
        } else {
            p[k] = 0.0;
        }
    }

    Ok((value, p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn diamond_graph_matches_softmax_over_path_costs() {
        // Two paths: 0-1-3 with cost a, and 0-2-3 with cost b.
        let n = 4;
        let a = 1.0 + 2.0;
        let b = 3.0 + 4.0;
        let edges = [
            Edge {
                from: 0,
                to: 1,
                cost: 1.0,
            },
            Edge {
                from: 1,
                to: 3,
                cost: 2.0,
            },
            Edge {
                from: 0,
                to: 2,
                cost: 3.0,
            },
            Edge {
                from: 2,
                to: 3,
                cost: 4.0,
            },
        ];
        let gamma = 0.5;
        let (v, p) = soft_shortest_path_edge_marginals(n, &edges, gamma).unwrap();

        // Path probabilities under Gibbs:
        let pa = (-a / gamma).exp();
        let pb = (-b / gamma).exp();
        let z = pa + pb;
        let p_path_a = pa / z;
        let p_path_b = pb / z;

        // Edge marginals should equal path probabilities for edges on each path.
        assert!(
            (p[0] - p_path_a).abs() < 1e-9,
            "p0={} pa={}",
            p[0],
            p_path_a
        );
        assert!(
            (p[1] - p_path_a).abs() < 1e-9,
            "p1={} pa={}",
            p[1],
            p_path_a
        );
        assert!(
            (p[2] - p_path_b).abs() < 1e-9,
            "p2={} pb={}",
            p[2],
            p_path_b
        );
        assert!(
            (p[3] - p_path_b).abs() < 1e-9,
            "p3={} pb={}",
            p[3],
            p_path_b
        );

        // Value equals softmin over path costs.
        let v_expected = -gamma * (pa + pb).ln();
        assert!(
            (v - v_expected).abs() < 1e-9,
            "v={} v_expected={}",
            v,
            v_expected
        );
    }

    proptest! {
        #[test]
        fn edge_marginals_are_probabilities_on_diamond(
            c01 in 0.0f64..10.0,
            c13 in 0.0f64..10.0,
            c02 in 0.0f64..10.0,
            c23 in 0.0f64..10.0,
            gamma in 0.05f64..5.0
        ) {
            let n = 4;
            let edges = [
                Edge { from: 0, to: 1, cost: c01 },
                Edge { from: 1, to: 3, cost: c13 },
                Edge { from: 0, to: 2, cost: c02 },
                Edge { from: 2, to: 3, cost: c23 },
            ];
            let (_v, p) = soft_shortest_path_edge_marginals(n, &edges, gamma).unwrap();
            for &pe in &p {
                prop_assert!((-1e-12..=1.0 + 1e-12).contains(&pe));
            }
            // Outgoing from source should sum to 1 on this graph.
            let s = p[0] + p[2];
            prop_assert!((s - 1.0).abs() < 1e-10, "s={}", s);
        }
    }
}
