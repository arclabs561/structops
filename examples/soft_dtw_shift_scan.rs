//! Relatable demo: find the best shift between two sequences with Soft-DTW.
//!
//! We create a base signal `x`, then a shifted copy `y`. We scan integer shifts and
//! compute Soft-DTW divergence; the minimum should occur near the true shift.

fn shift_circular(seq: &[f64], shift: isize) -> Vec<f64> {
    let n = seq.len() as isize;
    (0..n)
        .map(|i| {
            let j = (i - shift).rem_euclid(n);
            seq[j as usize]
        })
        .collect()
}

fn main() {
    let x: Vec<f64> = (0..24)
        .map(|i| {
            let t = i as f64 / 24.0;
            (2.0 * std::f64::consts::PI * t).sin()
        })
        .collect();

    let true_shift: isize = 3;
    let y = shift_circular(&x, true_shift);

    let gamma = 0.5;
    println!("gamma={gamma}  true_shift={true_shift}");
    println!("shift  soft_dtw_divergence");

    let mut best = (0isize, f64::INFINITY);
    for s in -8..=8 {
        let ys = shift_circular(&y, s);
        let d = structops::soft_dtw::soft_dtw_divergence(&x, &ys, gamma).unwrap();
        println!("{s:>5}  {d:.6}");
        if d < best.1 {
            best = (s, d);
        }
    }
    println!();
    println!("best_shift={}  best_div={:.6}", best.0, best.1);
}
