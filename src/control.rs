//! Relaxed control flow operators (Petersen et al., NeurIPS 2021).
//!
//! Discrete control flow (if/else, while, for) is non-differentiable because
//! branching is a step function.  These operators replace hard branching with
//! temperature-parameterized soft interpolation, making the computation graph
//! smooth and amenable to gradient-based optimization.
//!
//! Convention: `temperature` (or `beta`) > 0.  As `temperature -> 0` the soft
//! operators recover the hard discrete semantics.

// ---------------------------------------------------------------------------
// Soft comparisons
// ---------------------------------------------------------------------------

/// Soft less-than: `sigmoid(beta * (b - a))`.
///
/// Approaches 1 when `a < b`, 0 when `a > b`.  `beta` controls sharpness.
///
/// # Panics
///
/// Panics if `beta` is not positive and finite.
pub fn soft_lt(a: f32, b: f32, beta: f32) -> f32 {
    assert_positive_finite(beta, "beta");
    sigmoid(beta * (b - a))
}

/// Soft greater-than: `sigmoid(beta * (a - b))`.
///
/// Complement of [`soft_lt`]: `soft_gt(a, b, beta) == soft_lt(b, a, beta)`.
///
/// # Panics
///
/// Panics if `beta` is not positive and finite.
pub fn soft_gt(a: f32, b: f32, beta: f32) -> f32 {
    assert_positive_finite(beta, "beta");
    sigmoid(beta * (a - b))
}

/// Soft equality: `exp(-beta * (a - b)^2)`.
///
/// Peaks at 1 when `a == b`, decays as a Gaussian with width controlled by
/// `beta`.  Symmetric: `soft_eq(a, b, beta) == soft_eq(b, a, beta)`.
///
/// # Panics
///
/// Panics if `beta` is not positive and finite.
pub fn soft_eq(a: f32, b: f32, beta: f32) -> f32 {
    assert_positive_finite(beta, "beta");
    (-beta * (a - b).powi(2)).exp()
}

// ---------------------------------------------------------------------------
// Soft conditional
// ---------------------------------------------------------------------------

/// Soft conditional over vectors.
///
/// Interpolates element-wise between `then_val` and `else_val`:
///
/// ```text
/// w = sigmoid(condition / temperature)
/// result[i] = w * then_val[i] + (1 - w) * else_val[i]
/// ```
///
/// As `temperature -> 0`, this approaches a hard if/else on `sign(condition)`.
///
/// # Panics
///
/// Panics if `then_val.len() != else_val.len()` or `temperature` is not
/// positive and finite.
pub fn soft_if(condition: f32, then_val: &[f32], else_val: &[f32], temperature: f32) -> Vec<f32> {
    assert_eq!(
        then_val.len(),
        else_val.len(),
        "then_val and else_val must have the same length"
    );
    assert_positive_finite(temperature, "temperature");
    let w = sigmoid(condition / temperature);
    then_val
        .iter()
        .zip(else_val.iter())
        .map(|(&t, &e)| w * t + (1.0 - w) * e)
        .collect()
}

/// Soft conditional for scalars.
///
/// `sigmoid(condition / temperature) * then_val + (1 - sigmoid(condition / temperature)) * else_val`
///
/// # Panics
///
/// Panics if `temperature` is not positive and finite.
pub fn soft_if_scalar(condition: f32, then_val: f32, else_val: f32, temperature: f32) -> f32 {
    assert_positive_finite(temperature, "temperature");
    let w = sigmoid(condition / temperature);
    w * then_val + (1.0 - w) * else_val
}

// ---------------------------------------------------------------------------
// Soft loops
// ---------------------------------------------------------------------------

/// Soft while loop with bounded iteration.
///
/// At each step `t`:
/// 1. Evaluate `condition(&state)` to get a continuation probability `p`.
/// 2. Compute `next = body(&state)`.
/// 3. Blend: `state = p * next + (1 - p) * state`.
///
/// After `max_iter` steps the accumulated state is returned.  The blending
/// ensures that once the condition drifts toward 0 the state "freezes,"
/// approximating a hard while loop's early exit.
pub fn soft_while<F, C>(init: &[f32], body: F, condition: C, max_iter: usize) -> Vec<f32>
where
    F: Fn(&[f32]) -> Vec<f32>,
    C: Fn(&[f32]) -> f32,
{
    let mut state = init.to_vec();
    for _ in 0..max_iter {
        let p = condition(&state);
        let next = body(&state);
        for (s, &n) in state.iter_mut().zip(next.iter()) {
            *s = p * n + (1.0 - p) * *s;
        }
    }
    state
}

/// Soft for loop with per-iteration weights.
///
/// Runs `body(state, i)` for `i in 0..n`, blending each iteration's output
/// into the running state according to `weights[i]`:
///
/// ```text
/// state = weights[i] * body(state, i) + (1 - weights[i]) * state
/// ```
///
/// Uniform weights `[1/n; n]` produce the average over all iteration outputs.
/// Binary weights `{0, 1}` select which iterations execute (hard masking).
///
/// # Panics
///
/// Panics if `weights.len() != n`.
pub fn soft_for<F>(init: &[f32], body: F, n: usize, weights: &[f32]) -> Vec<f32>
where
    F: Fn(&[f32], usize) -> Vec<f32>,
{
    assert_eq!(weights.len(), n, "weights.len() must equal n");
    let mut state = init.to_vec();
    for (i, &w) in weights.iter().enumerate() {
        let next = body(&state, i);
        for (s, &nv) in state.iter_mut().zip(next.iter()) {
            *s = w * nv + (1.0 - w) * *s;
        }
    }
    state
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_positive_finite(value: f32, name: &str) {
    assert!(
        value.is_finite() && value > 0.0,
        "{name} must be finite and > 0"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn central_difference(mut f: impl FnMut(f32) -> f32, x: f32) -> f32 {
        let h = 1e-3;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn invalid_smoothing_parameters_always_panic() {
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(std::panic::catch_unwind(|| soft_lt(0.0, 1.0, invalid)).is_err());
            assert!(std::panic::catch_unwind(|| soft_gt(0.0, 1.0, invalid)).is_err());
            assert!(std::panic::catch_unwind(|| soft_eq(0.0, 1.0, invalid)).is_err());
            assert!(std::panic::catch_unwind(|| soft_if(1.0, &[1.0], &[0.0], invalid)).is_err());
            assert!(std::panic::catch_unwind(|| soft_if_scalar(1.0, 1.0, 0.0, invalid)).is_err());
        }
    }

    #[test]
    fn soft_comparisons_match_closed_forms() {
        let (a, b, beta) = (1.25f32, 2.75f32, 0.7f32);
        let logistic = 1.0 / (1.0 + (-beta * (b - a)).exp());
        assert!((soft_lt(a, b, beta) - logistic).abs() < 1e-7);
        assert!((soft_gt(a, b, beta) - (1.0 - logistic)).abs() < 1e-7);

        let gaussian = (-beta * (a - b).powi(2)).exp();
        assert!((soft_eq(a, b, beta) - gaussian).abs() < 1e-7);
    }

    #[test]
    fn soft_if_matches_two_way_softmax_and_temperature_limits() {
        let condition = 0.8f32;
        let temperature = 0.4f32;
        let then_val = 7.0f32;
        let else_val = -2.0f32;
        let then_logit = condition / temperature;
        let max_logit = then_logit.max(0.0);
        let then_exp = (then_logit - max_logit).exp();
        let else_exp = (-max_logit).exp();
        let weight = then_exp / (then_exp + else_exp);
        let expected = weight * then_val + (1.0 - weight) * else_val;
        assert!(
            (soft_if_scalar(condition, then_val, else_val, temperature) - expected).abs() < 1e-6
        );

        let midpoint = soft_if_scalar(condition, then_val, else_val, 1e8);
        assert!((midpoint - (then_val + else_val) * 0.5).abs() < 1e-5);
        assert!((soft_if_scalar(condition, then_val, else_val, 1e-6) - then_val).abs() < 1e-6);
        assert!((soft_if_scalar(-condition, then_val, else_val, 1e-6) - else_val).abs() < 1e-6);
    }

    #[test]
    fn soft_operator_gradients_match_finite_differences() {
        let (a, b, beta) = (0.3f32, 1.1f32, 0.8f32);
        let lt = soft_lt(a, b, beta);
        let expected_da = -beta * lt * (1.0 - lt);
        let actual_da = central_difference(|x| soft_lt(x, b, beta), a);
        assert!(
            (actual_da - expected_da).abs() < 1e-4,
            "soft_lt derivative: finite_difference={actual_da}, analytic={expected_da}"
        );

        let eq = soft_eq(a, b, beta);
        let expected_eq_da = -2.0 * beta * (a - b) * eq;
        let actual_eq_da = central_difference(|x| soft_eq(x, b, beta), a);
        assert!(
            (actual_eq_da - expected_eq_da).abs() < 1e-4,
            "soft_eq derivative: finite_difference={actual_eq_da}, analytic={expected_eq_da}"
        );

        let (condition, then_val, else_val, temperature) = (0.2f32, 3.0f32, -1.0f32, 0.9f32);
        let weight = 1.0 / (1.0 + (-condition / temperature).exp());
        let expected_condition = (then_val - else_val) * weight * (1.0 - weight) / temperature;
        let actual_condition = central_difference(
            |x| soft_if_scalar(x, then_val, else_val, temperature),
            condition,
        );
        assert!(
            (actual_condition - expected_condition).abs() < 1e-4,
            "soft_if derivative: finite_difference={actual_condition}, analytic={expected_condition}"
        );
    }

    // -- soft_if -----------------------------------------------------------

    #[test]
    fn soft_if_extreme_temperature_selects_then() {
        // Large positive condition with tiny temperature -> hard "then"
        let then_val = vec![10.0, 20.0];
        let else_val = vec![1.0, 2.0];
        let result = soft_if(1.0, &then_val, &else_val, 1e-6);
        for (r, &t) in result.iter().zip(then_val.iter()) {
            assert!((r - t).abs() < 1e-4, "r={} t={}", r, t);
        }
    }

    #[test]
    fn soft_if_extreme_temperature_selects_else() {
        // Large negative condition with tiny temperature -> hard "else"
        let then_val = vec![10.0, 20.0];
        let else_val = vec![1.0, 2.0];
        let result = soft_if(-1.0, &then_val, &else_val, 1e-6);
        for (r, &e) in result.iter().zip(else_val.iter()) {
            assert!((r - e).abs() < 1e-4, "r={} e={}", r, e);
        }
    }

    // -- soft_lt / soft_gt -------------------------------------------------

    #[test]
    fn soft_lt_and_soft_gt_are_complementary() {
        let a = 2.3;
        let b = 4.7;
        let beta = 5.0;
        let sum = soft_lt(a, b, beta) + soft_gt(a, b, beta);
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "soft_lt + soft_gt = {} (expected ~1.0)",
            sum
        );
    }

    #[test]
    fn soft_lt_high_beta_approaches_hard() {
        let beta = 100.0;
        assert!(soft_lt(1.0, 5.0, beta) > 0.999);
        assert!(soft_lt(5.0, 1.0, beta) < 0.001);
    }

    // -- soft_eq -----------------------------------------------------------

    #[test]
    fn soft_eq_peaks_at_equality() {
        assert!((soft_eq(3.0, 3.0, 10.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn soft_eq_decays_away_from_equality() {
        let v = soft_eq(0.0, 5.0, 1.0);
        assert!(v < 0.01, "soft_eq(0,5,1) = {} (expected near 0)", v);
    }

    // -- soft_while --------------------------------------------------------

    #[test]
    fn soft_while_condition_always_false_returns_init() {
        let init = vec![1.0, 2.0, 3.0];
        let result = soft_while(
            &init,
            |s| s.iter().map(|x| x + 1.0).collect(), // body that increments
            |_| 0.0,                                 // never continue
            100,
        );
        for (r, &i) in result.iter().zip(init.iter()) {
            assert!(
                (r - i).abs() < 1e-9,
                "expected init unchanged, got r={} i={}",
                r,
                i
            );
        }
    }

    #[test]
    fn soft_while_condition_always_true_converges() {
        // body: state *= 0.5 (contraction). condition always 1.0.
        // Fixed point is [0, 0, 0].
        let init = vec![8.0, 4.0, 2.0];
        let result = soft_while(&init, |s| s.iter().map(|x| x * 0.5).collect(), |_| 1.0, 200);
        for &r in &result {
            assert!(r.abs() < 1e-6, "expected convergence to 0, got {}", r);
        }
    }

    // -- soft_for ----------------------------------------------------------

    #[test]
    fn soft_for_uniform_weights_averages() {
        // With uniform weights 1/n and body that replaces state with [i as f32],
        // the result is the weighted blend.
        let n = 4;
        let weights = vec![1.0 / n as f32; n];
        let init = vec![0.0];
        let result = soft_for(&init, |_state, i| vec![i as f32], n, &weights);
        // This is a sequential blend, not a simple average.
        // state_0 = 0.0
        // state_1 = 0.25 * 0.0 + 0.75 * 0.0 = 0.0
        // state_2 = 0.25 * 1.0 + 0.75 * 0.0 = 0.25
        // state_3 = 0.25 * 2.0 + 0.75 * 0.25 = 0.6875
        // state_4 = 0.25 * 3.0 + 0.75 * 0.6875 = 1.265625
        let expected = 1.265625;
        assert!(
            (result[0] - expected).abs() < 1e-6,
            "result={} expected={}",
            result[0],
            expected
        );
    }

    #[test]
    fn soft_for_zero_weights_returns_init() {
        let init = vec![42.0, -7.0];
        let weights = vec![0.0; 5];
        let result = soft_for(&init, |_state, _i| vec![999.0, 999.0], 5, &weights);
        for (r, &i) in result.iter().zip(init.iter()) {
            assert!((r - i).abs() < 1e-9);
        }
    }

    // -- proptests ----------------------------------------------------------

    proptest! {
        #[test]
        fn soft_if_output_bounded_between_branches(
            condition in -10.0f32..10.0,
            a in -100.0f32..100.0,
            b in -100.0f32..100.0,
            c in -100.0f32..100.0,
            d in -100.0f32..100.0,
            temperature in 0.01f32..10.0
        ) {
            let then_val = vec![a, b];
            let else_val = vec![c, d];
            let result = soft_if(condition, &then_val, &else_val, temperature);
            for i in 0..2 {
                let lo = then_val[i].min(else_val[i]);
                let hi = then_val[i].max(else_val[i]);
                prop_assert!(
                    result[i] >= lo - 1e-6 && result[i] <= hi + 1e-6,
                    "result[{}]={} not in [{}, {}]", i, result[i], lo, hi
                );
            }
        }

        #[test]
        fn soft_lt_monotonically_decreasing_in_a(
            a1 in -10.0f32..10.0,
            delta in 0.0f32..5.0,
            b in -10.0f32..10.0,
            beta in 0.1f32..20.0
        ) {
            let a2 = a1 + delta;
            prop_assert!(
                soft_lt(a2, b, beta) <= soft_lt(a1, b, beta) + 1e-6,
                "soft_lt not monotonically decreasing: a1={} a2={} b={} beta={}",
                a1, a2, b, beta
            );
        }

        #[test]
        fn soft_eq_is_symmetric(
            a in -10.0f32..10.0,
            b in -10.0f32..10.0,
            beta in 0.1f32..20.0
        ) {
            let ab = soft_eq(a, b, beta);
            let ba = soft_eq(b, a, beta);
            prop_assert!(
                (ab - ba).abs() < 1e-6,
                "soft_eq not symmetric: soft_eq({},{},{})={} vs {}",
                a, b, beta, ab, ba
            );
        }

        #[test]
        fn soft_lt_plus_soft_gt_sums_to_one(
            a in -10.0f32..10.0,
            b in -10.0f32..10.0,
            beta in 0.1f32..20.0
        ) {
            let sum = soft_lt(a, b, beta) + soft_gt(a, b, beta);
            prop_assert!(
                (sum - 1.0).abs() < 1e-5,
                "sum={} for a={} b={} beta={}",
                sum, a, b, beta
            );
        }
    }
}
