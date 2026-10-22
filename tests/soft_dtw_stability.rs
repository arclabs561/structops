use structops::soft_dtw::{
    soft_dtw_cost, soft_dtw_cost_grad, soft_dtw_divergence_grad, soft_dtw_grad,
};

#[test]
fn unique_path_survives_tiny_positive_temperature() {
    for gamma in [1.0, 1e-308, 1e-320, f64::from_bits(1)] {
        // A 1x2 grid has exactly one path, independent of temperature.
        assert_eq!(soft_dtw_cost(&[1.0, 2.0], 1, 2, gamma).unwrap(), 3.0);
        let (value, grad) = soft_dtw_cost_grad(&[1.0, 2.0], 1, 2, gamma).unwrap();
        assert_eq!(value, 3.0);
        assert_eq!(grad, [1.0, 1.0]);
    }
}

#[test]
fn three_equal_paths_have_normalized_gradients_despite_rounding() {
    for (cost, gamma) in [([1e16, 0.0, 0.0, 0.0], 1.0), ([0.0; 4], f64::from_bits(1))] {
        // Three equal-cost paths: each endpoint is mandatory, while each
        // off-diagonal cell occurs in exactly one path. The rounded value
        // cannot retain the log(3) correction in both of these cases.
        let (_, grad) = soft_dtw_cost_grad(&cost, 2, 2, gamma).unwrap();
        for (actual, expected) in grad.iter().zip([1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-14, "{grad:?}");
        }
    }
}

#[test]
fn sequence_and_divergence_gradients_survive_tiny_temperature() {
    let gamma = 1e-320;
    let (value, gradient) = soft_dtw_grad(&[0.0], &[1.0, 2.0], gamma).unwrap();
    assert_eq!(value, 5.0);
    assert_eq!(gradient, [-6.0]);
    let (divergence, gradient) = soft_dtw_divergence_grad(&[0.0], &[1.0, 2.0], gamma).unwrap();
    assert_eq!(divergence, 5.0);
    assert_eq!(gradient, [-6.0]);
}

#[test]
fn five_equal_paths_preserve_prefix_multiplicity_at_large_offset() {
    // A 2x3 grid has five paths, all with total cost B. Counting visits
    // across those paths gives this oracle without a floating-point DP.
    let (value, grad) = soft_dtw_cost_grad(&[1e100, 0.0, 0.0, 0.0, 0.0, 0.0], 2, 3, 1.0).unwrap();
    assert_eq!(value, 1e100);
    for (actual, expected) in grad.iter().zip([1.0, 0.6, 0.2, 0.2, 0.6, 1.0]) {
        assert!((actual - expected).abs() < 1e-14, "{grad:?}");
    }
}

#[test]
fn gradients_match_independent_enumerated_paths() {
    fn paths(i: usize, j: usize, path: Vec<usize>, out: &mut Vec<Vec<usize>>) {
        let mut path = path;
        path.push(i * 3 + j);
        if i == 2 && j == 2 {
            out.push(path);
            return;
        }
        for (di, dj) in [(1, 0), (0, 1), (1, 1)] {
            if i + di < 3 && j + dj < 3 {
                paths(i + di, j + dj, path.clone(), out);
            }
        }
    }
    let mut all = Vec::new();
    paths(0, 0, Vec::new(), &mut all);
    assert_eq!(all.len(), 13);
    let costs = [0.3, -0.2, 0.7, 0.9, 0.1, -0.4, 0.2, 0.6, 0.5];
    for gamma in [0.1, 0.7, 3.0] {
        let masses: Vec<f64> = all
            .iter()
            .map(|p| {
                let total: f64 = p.iter().map(|&k| costs[k]).sum();
                (-total / gamma).exp()
            })
            .collect();
        let partition: f64 = masses.iter().sum();
        let (value, grad) = soft_dtw_cost_grad(&costs, 3, 3, gamma).unwrap();
        assert!((value + gamma * partition.ln()).abs() < 1e-13);
        for (k, actual) in grad.iter().enumerate() {
            let expected: f64 = all
                .iter()
                .zip(&masses)
                .filter(|(p, _)| p.contains(&k))
                .map(|(_, mass)| mass / partition)
                .sum();
            assert!((actual - expected).abs() < 1e-13);
        }
        assert!((grad[0] - 1.0).abs() < 1e-14);
        assert_eq!(grad[8], 1.0);
        let length: f64 = grad.iter().sum();
        assert!((3.0..=5.0).contains(&length));
    }
}

#[test]
fn opposite_large_costs_keep_all_three_path_masses() {
    // The three path costs divided by gamma are exactly -1, 0, and 1.
    // Enumerate their Gibbs masses directly, without subtracting large costs.
    let partition = 1.0_f64.exp() + 1.0 + (-1.0_f64).exp();
    let expected = [
        1.0,
        (-1.0_f64).exp() / partition,
        1.0_f64.exp() / partition,
        1.0,
    ];
    let (value, grad) = soft_dtw_cost_grad(&[0.0, 1e308, -1e308, 0.0], 2, 2, 1e308).unwrap();
    assert!(value.is_finite());
    assert!((value / 1e308 + partition.ln()).abs() < 1e-14);
    for (actual, expected) in grad.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-14, "{grad:?}");
    }
}
