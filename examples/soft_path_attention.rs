//! End-to-end example: “DP = attention” on a tiny DAG.
//!
//! The `soft_shortest_path_edge_marginals` function returns edge marginals under a
//! Gibbs distribution over paths. These marginals are exactly the gradients of the
//! soft shortest-path value w.r.t. edge costs (Mensch & Blondel 2018 framing).

use structops::soft_shortest_path::{soft_shortest_path_edge_marginals, Edge};

fn main() {
    // A tiny DAG with two alternative paths from 0 to 3:
    // 0->1->3 (cheap) and 0->2->3 (expensive).
    //
    // The edge marginals returned by the DP are exactly the “soft attention”
    // weights over edges induced by a Gibbs distribution over paths.
    let edges = [
        Edge {
            from: 0,
            to: 1,
            cost: 1.0,
        },
        Edge {
            from: 1,
            to: 3,
            cost: 1.0,
        },
        Edge {
            from: 0,
            to: 2,
            cost: 3.0,
        },
        Edge {
            from: 2,
            to: 3,
            cost: 3.0,
        },
    ];

    let gamma = 0.5;
    let (value, p) = soft_shortest_path_edge_marginals(4, &edges, gamma).unwrap();

    println!("soft shortest-path value = {value}");
    for (k, pe) in p.iter().enumerate() {
        let e = edges[k];
        println!(
            "edge {} ({}->{}, cost={}): marginal p(e in path) = {}",
            k, e.from, e.to, e.cost, pe
        );
    }
}
