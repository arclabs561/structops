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
//! - The **Soft-DTW divergence** is a common debiased substitute:
//!   \(\operatorname{sdtw}_\gamma(x,y)-\tfrac12\operatorname{sdtw}_\gamma(x,x)-\tfrac12\operatorname{sdtw}_\gamma(y,y)\).
//!   This is typically nonnegative and is zero on identical inputs (under the
//!   usual squared-distance setting).

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

fn softmin3(gamma: f64, a: f64, b: f64, c: f64) -> f64 {
    // log-sum-exp stabilization: compute
    // -γ log(exp(-a/γ)+exp(-b/γ)+exp(-c/γ))
    let xa = -a / gamma;
    let xb = -b / gamma;
    let xc = -c / gamma;
    let m = xa.max(xb).max(xc);
    if !m.is_finite() {
        // If all are +∞ in original space, return +∞.
        return f64::INFINITY;
    }
    let s = (xa - m).exp() + (xb - m).exp() + (xc - m).exp();
    -gamma * (m + s.ln())
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
pub fn soft_dtw_cost(cost: &[f64], n: usize, m: usize, gamma: f64) -> Result<f64> {
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

    let w = m + 1;
    let mut r = vec![f64::INFINITY; (n + 1) * (m + 1)];
    r[0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let d = cost[(i - 1) * m + (j - 1)];
            let a = r[(i - 1) * w + j];
            let b = r[i * w + (j - 1)];
            let c = r[(i - 1) * w + (j - 1)];
            r[i * w + j] = d + softmin3(gamma, a, b, c);
        }
    }
    Ok(r[n * w + m])
}

/// Soft-DTW divergence (commonly used because it is nonnegative and zero on identical inputs).
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
/// This mirrors the scalar `soft_dtw_divergence` but works for arbitrary distances.
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
