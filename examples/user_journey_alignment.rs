//! User Journey Alignment Demo (Soft-DTW)
//!
//! Demonstrates using Soft-DTW to align noisy user sessions to a canonical "Golden Path".
//!
//! # The Scenario
//!
//! - **Golden Path**: `Landing -> Pricing -> Sign Up`
//! - **User A (Focused)**: `Landing -> Pricing -> Sign Up` (Perfect)
//! - **User B (Lost)**: `Landing -> Blog -> Pricing -> Blog -> Pricing -> Sign Up` (Noisy)
//! - **User C (Bounce)**: `Landing -> Blog -> Exit` (Incomplete)
//!
//! # Why Soft-DTW?
//!
//! Standard DTW is hard (min). Soft-DTW is differentiable and smooth.
//! This allows us to use the alignment score as a continuous feature for clustering
//! or churn prediction models.

use structops::soft_dtw::soft_dtw_divergence_cost;

// Simple one-hot state encoding
#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Landing,
    Pricing,
    SignUp,
    Blog,
    Exit,
}

impl State {
    fn name(&self) -> &'static str {
        match self {
            State::Landing => "Landing",
            State::Pricing => "Pricing",
            State::SignUp => "SignUp",
            State::Blog => "Blog",
            State::Exit => "Exit",
        }
    }
}

fn state_cost(a: State, b: State) -> f64 {
    // A tiny, explicit metric over categorical states.
    // Equal states have 0 cost; mismatches have 1 cost, except "Exit" which is
    // “far” from everything else (we don't want exiting to look like a mild detour).
    if a == b {
        return 0.0;
    }
    if a == State::Exit || b == State::Exit {
        return 2.0;
    }
    1.0
}

fn cost_matrix_xy(x: &[State], y: &[State]) -> Vec<f64> {
    let n = x.len();
    let m = y.len();
    let mut cost = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            cost[i * m + j] = state_cost(x[i], y[j]);
        }
    }
    cost
}

fn sdtw_divergence_states(x: &[State], y: &[State], gamma: f64) -> f64 {
    let n = x.len();
    let m = y.len();
    let cost_xy = cost_matrix_xy(x, y);
    let cost_xx = cost_matrix_xy(x, x);
    let cost_yy = cost_matrix_xy(y, y);
    soft_dtw_divergence_cost(&cost_xy, &cost_xx, &cost_yy, n, m, gamma).unwrap()
}

fn print_seq(name: &str, seq: &[State]) {
    let s: Vec<&str> = seq.iter().map(|s| s.name()).collect();
    println!("{:<15}: {}", name, s.join(" -> "));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let golden_path = [State::Landing, State::Pricing, State::SignUp];

    let user_a = [State::Landing, State::Pricing, State::SignUp];
    let user_b = [
        State::Landing,
        State::Blog,
        State::Pricing,
        State::Blog,
        State::Pricing,
        State::SignUp,
    ];
    let user_c = [State::Landing, State::Blog, State::Exit];

    let gamma = 1.0;

    println!("User Journey Alignment (Soft-DTW, gamma={})", gamma);
    println!("Note: Using an explicit categorical cost (0=same, 1=mismatch, 2=Exit mismatch).");
    println!();

    print_seq("Golden Path", &golden_path);
    println!();

    let score_a = sdtw_divergence_states(&golden_path, &user_a, gamma);
    print_seq("User A (Ideal)", &user_a);
    println!("   Score: {:.4} (Perfect match)", score_a);

    let score_b = sdtw_divergence_states(&golden_path, &user_b, gamma);
    print_seq("User B (Noisy)", &user_b);
    println!("   Score: {:.4} (High alignment despite noise)", score_b);

    let score_c = sdtw_divergence_states(&golden_path, &user_c, gamma);
    print_seq("User C (Bounce)", &user_c);
    println!("   Score: {:.4} (Poor alignment)", score_c);

    println!();
    println!("Interpretation:");
    println!("User B has a good score despite extra steps (warping handles insertions).");
    println!("User C has a bad score because they never reached the goal state.");

    Ok(())
}
