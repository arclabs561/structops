//! Soft-DTW (Cuturi & Blondel 2017): a differentiable relaxation of Dynamic Time Warping.
//!
//! We implement the forward dynamic program for:
//! \[
//! \operatorname{softDTW}_\gamma(x,y)
//! = R_{n,m},\quad
//! R_{i,j} = d(x_i,y_j) + \operatorname{softmin}_\gamma(R_{i-1,j}, R_{i,j-1}, R_{i-1,j-1})
//! \]
//! with boundary conditions \(R_{0,0}=0\) and \(R_{i,0}=R_{0,j}=+\infty\).
//!
//! Here \(d(a,b)=(a-b)^2\) and
//! \[
//! \operatorname{softmin}_\gamma(a,b,c) = -\gamma \log\left(e^{-a/\gamma}+e^{-b/\gamma}+e^{-c/\gamma}\right).
//! \]
//!
//! Notes:
//! - `soft_dtw` is not a metric in general.
//! - `soft_dtw(x,y,γ)` is a smooth relaxation of DTW, but it can be biased (and
//!   is not minimized at `x == y` for many settings).
//! - The **Soft-DTW divergence** (Blondel, Mensch & Vert 2021, "Differentiable
//!   Divergences Between Time Series", AISTATS) is a debiased substitute:
//!   \(\operatorname{sdtw}_\gamma(x,y)-\tfrac12\operatorname{sdtw}_\gamma(x,x)-\tfrac12\operatorname{sdtw}_\gamma(y,y)\).
//!   Self-comparison cancels in exact arithmetic. Nonnegativity depends on
//!   the ground cost; it is not guaranteed for arbitrary precomputed matrices.
//! - Gradients: [`soft_dtw_cost_grad`] computes the expected alignment of Cuturi &
//!   Blondel (2017) by reverse accumulation; [`soft_dtw_grad`] and
//!   [`soft_dtw_divergence_grad`] chain it through squared scalar distance.

/// Errors for Soft-DTW operators.
#[derive(thiserror::Error, Debug, Clone, PartialEq)]
pub enum Error {
    /// Smoothing parameter \(\gamma\) must be positive and finite.
    #[error("gamma must be positive and finite, got {0}")]
    InvalidGamma(f64),
    /// Inputs must be non-empty sequences.
    #[error("inputs must be non-empty")]
    EmptyInput,
    /// Non-finite value in input sequence.
    #[error("non-finite value in input at index {0}")]
    NonFiniteInput(usize),
    /// Non-finite cost in cost matrix.
    #[error("non-finite cost at index {0}")]
    NonFiniteCost(usize),
    /// Cost matrix shape mismatch.
    #[error("cost matrix has length {len}, expected {n}*{m}={expected}")]
    InvalidCostShape {
        /// The provided `cost` slice length.
        len: usize,
        /// Expected row count.
        n: usize,
        /// Expected column count.
        m: usize,
        /// `n*m`, included explicitly for readability.
        expected: usize,
    },
}

/// Convenience result type for this module.
pub type Result<T> = std::result::Result<T, Error>;

// Keep the hard-path cost and the dimensionless partition residual separate:
// Z = exp(-baseline / gamma) * exp(log_mass). Folding log_mass into the
// baseline would lose path multiplicities at large offsets or tiny gamma.
#[derive(Clone, Copy)]
struct Partition {
    baseline: f64,
    log_mass: f64,
}

fn forward(cost: &[f64], n: usize, m: usize, gamma: f64) -> (f64, Vec<[f64; 3]>) {
    let w = m + 1;
    let mut states = vec![None; (n + 1) * w];
    states[0] = Some(Partition {
        baseline: 0.0,
        log_mass: 0.0,
    });
    let mut weights = vec![[0.0; 3]; n * m];
    for i in 1..=n {
        for j in 1..=m {
            let predecessors = [
                states[(i - 1) * w + j],
                states[i * w + j - 1],
                states[(i - 1) * w + j - 1],
            ];
            let baseline = predecessors
                .iter()
                .flatten()
                .map(|p| p.baseline)
                .fold(f64::INFINITY, f64::min);
            let logits = predecessors.map(|p| match p {
                Some(p) => {
                    let difference = p.baseline - baseline;
                    // Opposite-sign finite costs can overflow on subtraction
                    // even when their temperature-scaled difference is finite.
                    // Dividing first is safe here: the quotients have opposite
                    // signs, so this fallback cannot introduce inf - inf.
                    let scaled = if difference.is_infinite()
                        && p.baseline.is_finite()
                        && baseline.is_finite()
                    {
                        p.baseline / gamma - baseline / gamma
                    } else {
                        difference / gamma
                    };
                    p.log_mass - scaled
                }
                None => f64::NEG_INFINITY,
            });
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let numerators = logits.map(|x| (x - max).exp());
            let sum: f64 = numerators.iter().sum();
            weights[(i - 1) * m + j - 1] = numerators.map(|x| x / sum);
            states[i * w + j] = Some(Partition {
                baseline: baseline + cost[(i - 1) * m + j - 1],
                log_mass: max + sum.ln(),
            });
        }
    }
    let last = states[n * w + m].unwrap();
    (last.baseline - gamma * last.log_mass, weights)
}

/// Soft-DTW value for two 1D sequences.
pub fn soft_dtw(x: &[f64], y: &[f64], gamma: f64) -> Result<f64> {
    if gamma <= 0.0 || !gamma.is_finite() {
        return Err(Error::InvalidGamma(gamma));
    }
    if x.is_empty() || y.is_empty() {
        return Err(Error::EmptyInput);
    }
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::NonFiniteInput(i));
        }
    }
    for (i, &v) in y.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::NonFiniteInput(i));
        }
    }
    let n = x.len();
    let m = y.len();
    let mut cost = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            cost[i * m + j] = (x[i] - y[j]).powi(2);
        }
    }
    soft_dtw_cost(&cost, n, m, gamma)
}

/// Soft-DTW value given a precomputed cost matrix `cost` (row-major).
///
/// This is the more general form used in practice when the elements are not scalars
/// (e.g. sentence embeddings) and you want \(d(x_i,y_j)\) to be an arbitrary distance.
///
/// `cost` must have length `n * m`, storing `cost[i*m + j] = d(x_i, y_j)` for
/// 0-based indices `i in 0..n`, `j in 0..m`.
///
/// Partition residuals are retained separately to preserve path multiplicities
/// when a large common cost or tiny temperature rounds away the entropy term.
/// Accumulated hard-path costs still use `f64`: their rounding and overflow can
/// limit accuracy even when every individual input cost is finite.
pub fn soft_dtw_cost(cost: &[f64], n: usize, m: usize, gamma: f64) -> Result<f64> {
    validate_cost(cost, n, m, gamma)?;
    Ok(forward(cost, n, m, gamma).0)
}

fn validate_cost(cost: &[f64], n: usize, m: usize, gamma: f64) -> Result<()> {
    if gamma <= 0.0 || !gamma.is_finite() {
        return Err(Error::InvalidGamma(gamma));
    }
    if n == 0 || m == 0 {
        return Err(Error::EmptyInput);
    }
    if cost.len() != n * m {
        return Err(Error::InvalidCostShape {
            len: cost.len(),
            n,
            m,
            expected: n * m,
        });
    }
    for (i, &c) in cost.iter().enumerate() {
        if !c.is_finite() {
            return Err(Error::NonFiniteCost(i));
        }
    }

    Ok(())
}

/// Soft-DTW value and its gradient with respect to the cost matrix.
///
/// Returns `(value, e)` where `e[i*m + j]` is \(\partial R_{n,m} / \partial
/// \text{cost}_{i,j}\): the expected alignment matrix of Cuturi & Blondel
/// (2017), computed by reverse accumulation of local Gibbs probabilities. Each entry is in
/// `[0, 1]`; as \(\gamma \to 0\) it approaches the average alignment over
/// all minimum-cost paths (the hard DTW alignment when that path is unique).
///
/// Chain `e` through your own distance to get gradients with respect to the
/// sequence elements; [`soft_dtw_grad`] does this for squared scalar distance.
pub fn soft_dtw_cost_grad(cost: &[f64], n: usize, m: usize, gamma: f64) -> Result<(f64, Vec<f64>)> {
    // Validates inputs exactly as the forward pass does.
    validate_cost(cost, n, m, gamma)?;

    let (value, weights) = forward(cost, n, m, gamma);
    // Reverse adjoints through the retained local Gibbs probabilities. Do not
    // reconstruct probabilities from rounded differences of cumulative costs.
    let w = m + 1;
    let mut adjoint = vec![0.0; (n + 1) * w];
    adjoint[n * w + m] = 1.0;
    let mut grad = vec![0.0; n * m];
    for i in (1..=n).rev() {
        for j in (1..=m).rev() {
            let index = (i - 1) * m + j - 1;
            let g = adjoint[i * w + j];
            grad[index] = g;
            let [up, left, diagonal] = weights[index];
            adjoint[(i - 1) * w + j] += g * up;
            adjoint[i * w + j - 1] += g * left;
            adjoint[(i - 1) * w + j - 1] += g * diagonal;
        }
    }
    Ok((value, grad))
}

fn squared_cost(x: &[f64], y: &[f64]) -> Vec<f64> {
    let m = y.len();
    let mut cost = vec![0.0f64; x.len() * m];
    for (i, &xi) in x.iter().enumerate() {
        for (j, &yj) in y.iter().enumerate() {
            cost[i * m + j] = (xi - yj).powi(2);
        }
    }
    cost
}

fn check_finite(x: &[f64]) -> Result<()> {
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::NonFiniteInput(i));
        }
    }
    Ok(())
}

/// Soft-DTW value for two 1D sequences and its gradient with respect to `x`.
///
/// Uses \(d(a,b)=(a-b)^2\), as [`soft_dtw`] does, so
/// \(\partial / \partial x_i = \sum_j E_{i,j}\, 2(x_i - y_j)\) with `E` from
/// [`soft_dtw_cost_grad`].
pub fn soft_dtw_grad(x: &[f64], y: &[f64], gamma: f64) -> Result<(f64, Vec<f64>)> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::EmptyInput);
    }
    check_finite(x)?;
    check_finite(y)?;
    let (n, m) = (x.len(), y.len());
    let (value, e) = soft_dtw_cost_grad(&squared_cost(x, y), n, m, gamma)?;
    let grad = (0..n)
        .map(|i| (0..m).map(|j| e[i * m + j] * 2.0 * (x[i] - y[j])).sum())
        .collect();
    Ok((value, grad))
}

/// Soft-DTW divergence and its gradient with respect to `x`.
///
/// The self term `sdtw(x, x)` depends on `x` through both arguments, so its
/// gradient is \(\sum_j (E^{xx}_{i,j} + E^{xx}_{j,i}) \cdot 2(x_i - x_j)\).
pub fn soft_dtw_divergence_grad(x: &[f64], y: &[f64], gamma: f64) -> Result<(f64, Vec<f64>)> {
    let (xy, grad_xy) = soft_dtw_grad(x, y, gamma)?;
    let n = x.len();
    let (xx, e_xx) = soft_dtw_cost_grad(&squared_cost(x, x), n, n, gamma)?;
    let yy = soft_dtw(y, y, gamma)?;
    let grad = (0..n)
        .map(|i| {
            let self_term: f64 = (0..n)
                .map(|j| (e_xx[i * n + j] + e_xx[j * n + i]) * 2.0 * (x[i] - x[j]))
                .sum();
            grad_xy[i] - 0.5 * self_term
        })
        .collect();
    Ok((xy - 0.5 * xx - 0.5 * yy, grad))
}

/// Debiased Soft-DTW value, zero on identical inputs in exact arithmetic.
pub fn soft_dtw_divergence(x: &[f64], y: &[f64], gamma: f64) -> Result<f64> {
    let xy = soft_dtw(x, y, gamma)?;
    let xx = soft_dtw(x, x, gamma)?;
    let yy = soft_dtw(y, y, gamma)?;
    Ok(xy - 0.5 * xx - 0.5 * yy)
}

/// Soft-DTW divergence given precomputed cost matrices:
/// - `cost_xy` shape `n×m`
/// - `cost_xx` shape `n×n`
/// - `cost_yy` shape `m×m`
///
/// This mirrors the scalar `soft_dtw_divergence` for supplied costs. Arbitrary
/// matrices do not guarantee nonnegativity; the result is not clamped.
pub fn soft_dtw_divergence_cost(
    cost_xy: &[f64],
    cost_xx: &[f64],
    cost_yy: &[f64],
    n: usize,
    m: usize,
    gamma: f64,
) -> Result<f64> {
    let xy = soft_dtw_cost(cost_xy, n, m, gamma)?;
    let xx = soft_dtw_cost(cost_xx, n, n, gamma)?;
    let yy = soft_dtw_cost(cost_yy, m, m, gamma)?;
    Ok(xy - 0.5 * xx - 0.5 * yy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn identical_sequences_have_zero_divergence() {
        let x = [1.0, 2.0, 3.0];
        let d = soft_dtw_divergence(&x, &x, 1.0).unwrap();
        assert!(d.abs() < 1e-12, "d={}", d);
    }

    #[test]
    fn divergence_is_symmetric() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.5, 2.0];
        let a = soft_dtw_divergence(&x, &y, 0.5).unwrap();
        let b = soft_dtw_divergence(&y, &x, 0.5).unwrap();
        assert!((a - b).abs() < 1e-12, "a={} b={}", a, b);
    }

    proptest! {
        #[test]
        fn divergence_is_nonnegative_for_small_random_inputs(
            x in prop::collection::vec(-3.0f64..3.0, 1..20),
            y in prop::collection::vec(-3.0f64..3.0, 1..20),
            gamma in 0.05f64..5.0
        ) {
            let d = soft_dtw_divergence(&x, &y, gamma).unwrap();
            prop_assert!(d >= -1e-9, "d={}", d);
        }
    }

    #[test]
    fn cost_matrix_version_matches_scalar_version_for_squared_distance() {
        let x: [f64; 3] = [1.0, -2.0, 0.5];
        let y: [f64; 2] = [1.2, -1.5];
        let gamma = 0.7;

        let n = x.len();
        let m = y.len();
        let mut cost_xy = vec![0.0f64; n * m];
        for i in 0..n {
            for j in 0..m {
                cost_xy[i * m + j] = (x[i] - y[j]).powi(2);
            }
        }

        let v_scalar = soft_dtw(&x, &y, gamma).unwrap();
        let v_cost = soft_dtw_cost(&cost_xy, n, m, gamma).unwrap();

        assert!(
            (v_scalar - v_cost).abs() < 1e-12,
            "scalar={} cost={}",
            v_scalar,
            v_cost
        );
    }

    fn dtw_squared(x: &[f64], y: &[f64]) -> f64 {
        // Classic DTW DP with squared distance and min-plus semiring.
        // Returns the minimal path cost.
        let n = x.len();
        let m = y.len();
        assert!(n > 0 && m > 0);
        let w = m + 1;
        let mut r = vec![f64::INFINITY; (n + 1) * (m + 1)];
        r[0] = 0.0;
        for i in 1..=n {
            for j in 1..=m {
                let d = (x[i - 1] - y[j - 1]).powi(2);
                let a = r[(i - 1) * w + j];
                let b = r[i * w + (j - 1)];
                let c = r[(i - 1) * w + (j - 1)];
                r[i * w + j] = d + a.min(b).min(c);
            }
        }
        r[n * w + m]
    }

    #[test]
    fn soft_dtw_bounds_dtw_with_gamma_ln3_slack() {
        // Lemma: softmin_γ(a,b,c) ∈ [min(a,b,c) - γ ln 3, min(a,b,c)].
        // Over an (n+m)-step DTW path, the total slack is O((n+m) γ ln 3).
        let x = [0.2, -0.1, 0.5, 0.0];
        let y = [0.1, 0.4, -0.2];
        let gamma = 1e-3;

        let dtw = dtw_squared(&x, &y);
        let s = soft_dtw(&x, &y, gamma).unwrap();

        let slack = ((x.len() + y.len()) as f64) * gamma * 3.0_f64.ln();
        assert!(
            s <= dtw + 1e-12,
            "expected soft_dtw <= dtw (s={} dtw={})",
            s,
            dtw
        );
        assert!(
            dtw - s <= slack + 1e-9,
            "expected dtw - soft_dtw <= O((n+m)γln3): dtw={} s={} slack={}",
            dtw,
            s,
            slack
        );
    }

    #[test]
    fn soft_dtw_can_be_negative_on_diagonal_but_divergence_is_zero() {
        // This encodes the “entropic bias” intuition from the Soft-DTW literature:
        // the relaxed objective can be negative even with nonnegative costs, because
        // soft-min aggregates over many warping paths.
        //
        // The divergence construction cancels that bias by subtracting the self-terms.
        let x = [0.0, 1.0, 2.0, 3.0];
        let gamma = 5.0;

        let xx = soft_dtw(&x, &x, gamma).unwrap();
        assert!(xx.is_finite());
        assert!(
            xx < 0.0,
            "expected soft_dtw(x,x,gamma) < 0 for large gamma, got {}",
            xx
        );

        let d = soft_dtw_divergence(&x, &x, gamma).unwrap();
        assert!(d.abs() < 1e-10, "expected divergence(x,x)=0, got {}", d);
    }

    fn central_difference(f: impl Fn(f64) -> f64, at: f64) -> f64 {
        let h = 1e-6;
        (f(at + h) - f(at - h)) / (2.0 * h)
    }

    #[test]
    fn cost_gradient_matches_finite_differences_of_the_value() {
        let (n, m, gamma) = (4usize, 3usize, 0.7);
        let cost = vec![
            0.1, 1.0, 0.3, //
            0.4, 0.2, 0.9, //
            1.2, 0.7, 0.4, //
            0.3, 0.6, 0.8, //
        ];
        let (value, grad) = soft_dtw_cost_grad(&cost, n, m, gamma).unwrap();
        assert!((value - soft_dtw_cost(&cost, n, m, gamma).unwrap()).abs() < 1e-12);
        for k in 0..n * m {
            let fd = central_difference(
                |c| {
                    let mut perturbed = cost.clone();
                    perturbed[k] = c;
                    soft_dtw_cost(&perturbed, n, m, gamma).unwrap()
                },
                cost[k],
            );
            assert!(
                (grad[k] - fd).abs() < 1e-6,
                "entry {k}: grad={} fd={fd}",
                grad[k]
            );
            assert!(
                (0.0..=1.0 + 1e-12).contains(&grad[k]),
                "entry {k} = {}",
                grad[k]
            );
        }
        // Every warping path visits both corners.
        assert!((grad[0] - 1.0).abs() < 1e-12);
        assert!((grad[n * m - 1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cost_gradient_approaches_the_hard_alignment_as_gamma_shrinks() {
        // Diagonal is the unique cheapest path.
        let cost = vec![
            0.0, 5.0, 5.0, //
            5.0, 0.0, 5.0, //
            5.0, 5.0, 0.0, //
        ];
        let (_, grad) = soft_dtw_cost_grad(&cost, 3, 3, 1e-3).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (grad[i * 3 + j] - want).abs() < 1e-9,
                    "({i},{j}) = {}",
                    grad[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn sequence_gradients_match_finite_differences() {
        let x = [0.2, -0.1, 0.5, 0.0];
        let y = [0.1, 0.4, -0.2];
        let gamma = 0.5;

        let (value, grad) = soft_dtw_grad(&x, &y, gamma).unwrap();
        assert!((value - soft_dtw(&x, &y, gamma).unwrap()).abs() < 1e-12);
        let (div, div_grad) = soft_dtw_divergence_grad(&x, &y, gamma).unwrap();
        assert!((div - soft_dtw_divergence(&x, &y, gamma).unwrap()).abs() < 1e-12);

        for k in 0..x.len() {
            let at = |v: f64| {
                let mut p = x;
                p[k] = v;
                p
            };
            let fd = central_difference(|v| soft_dtw(&at(v), &y, gamma).unwrap(), x[k]);
            assert!(
                (grad[k] - fd).abs() < 1e-6,
                "sdtw x[{k}]: {} vs {fd}",
                grad[k]
            );
            let fd_div =
                central_difference(|v| soft_dtw_divergence(&at(v), &y, gamma).unwrap(), x[k]);
            assert!(
                (div_grad[k] - fd_div).abs() < 1e-6,
                "divergence x[{k}]: {} vs {fd_div}",
                div_grad[k]
            );
        }
    }

    #[test]
    fn soft_dtw_cost_is_monotone_in_costs() {
        // Correct invariant: the soft-min DP is monotone.
        // If C' >= C elementwise, then softDTW_γ(C') >= softDTW_γ(C).
        let n = 4usize;
        let m = 3usize;
        let gamma = 0.8;

        let cost_xy = vec![
            0.1, 1.0, 0.3, //
            0.4, 0.2, 0.9, //
            1.2, 0.7, 0.4, //
            0.3, 0.6, 0.8, //
        ];
        let mut cost_xy2 = cost_xy.clone();

        // Increase a few entries (elementwise nonnegative delta).
        cost_xy2[0] += 0.5;
        cost_xy2[4] += 0.2;
        cost_xy2[11] += 1.0;

        let s1 = soft_dtw_cost(&cost_xy, n, m, gamma).unwrap();
        let s2 = soft_dtw_cost(&cost_xy2, n, m, gamma).unwrap();

        assert!(
            s2 + 1e-12 >= s1,
            "expected monotonicity: softDTW(C')={} >= softDTW(C)={}",
            s2,
            s1
        );
    }
}
