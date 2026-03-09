//! Soft shortest path on a graphops-constructed graph.
//!
//! Demonstrates the connection between graphops (graph structure) and structops
//! (differentiable path computation):
//!
//! 1. Build a small weighted directed graph using graphops::AdjacencyMatrix.
//! 2. Extract edges via the Graph + WeightedGraph trait methods.
//! 3. Convert to structops::Edge format and compute soft shortest-path marginals.
//! 4. Show how the temperature parameter gamma controls path concentration:
//!    - small gamma: marginals concentrate on the single shortest path (hard argmin).
//!    - large gamma: marginals spread across all feasible paths (uniform).

use graphops::{AdjacencyMatrix, Graph, WeightedGraph};
use structops::soft_shortest_path::{soft_shortest_path_edge_marginals, Edge};

/// Build a 6-node directed transportation network as an adjacency matrix.
///
/// Layout (costs on edges):
///
/// ```text
///          1
///        / | \
///      2   5   3
///     /    |    \
///    0     |     5
///     \    |    /
///      4   1   6
///        \ | /
///          3
///          |  2
///          4
/// ```
///
/// Node 0 = origin, node 5 = destination.
/// Two main corridors: 0->1->3->5 and 0->2->3->5, with cross-links.
fn build_network() -> Vec<Vec<f64>> {
    let n = 6;
    let mut adj = vec![vec![0.0; n]; n];

    // From node 0
    adj[0][1] = 2.0; // 0 -> 1
    adj[0][2] = 4.0; // 0 -> 2

    // From node 1
    adj[1][3] = 5.0; // 1 -> 3
    adj[1][4] = 1.0; // 1 -> 4 (shortcut)

    // From node 2
    adj[2][3] = 6.0; // 2 -> 3
    adj[2][4] = 3.0; // 2 -> 4

    // From node 3
    adj[3][5] = 2.0; // 3 -> 5

    // From node 4
    adj[4][5] = 1.0; // 4 -> 5

    adj
}

/// Extract directed edges from an adjacency matrix using graphops trait methods.
fn extract_edges(adj: &[Vec<f64>]) -> Vec<Edge> {
    let graph = AdjacencyMatrix(adj);
    let n = graph.node_count();
    let mut edges = Vec::new();
    for u in 0..n {
        let neighbors = graph.neighbors(u);
        for v in neighbors {
            // Only forward edges (u < v) -- required by structops DAG invariant.
            if u < v {
                let cost = graph.edge_weight(u, v);
                edges.push(Edge {
                    from: u,
                    to: v,
                    cost,
                });
            }
        }
    }
    edges
}

/// Format an edge label for display.
fn edge_label(e: &Edge) -> String {
    format!("{}->{} (cost={:.1})", e.from, e.to, e.cost)
}

fn main() {
    let adj = build_network();
    let n = adj.len();
    let edges = extract_edges(&adj);

    println!("Graph: {n} nodes, {} edges", edges.len());
    println!();
    for (i, e) in edges.iter().enumerate() {
        println!("  edge {i}: {}", edge_label(e));
    }
    println!();

    // Enumerate paths manually for reference.
    // Source=0, sink=5. Feasible s-t paths:
    //   0->1->3->5  cost = 2+5+2 = 9
    //   0->1->4->5  cost = 2+1+1 = 4  (shortest)
    //   0->2->3->5  cost = 4+6+2 = 12
    //   0->2->4->5  cost = 4+3+1 = 8
    println!("Feasible paths (source=0, sink=5):");
    println!("  0->1->3->5  cost = 9");
    println!("  0->1->4->5  cost = 4  <-- shortest");
    println!("  0->2->3->5  cost = 12");
    println!("  0->2->4->5  cost = 8");
    println!();

    // Sweep gamma from small (concentrating) to large (spreading).
    let gammas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 50.0];

    // Header
    print!("{:>22}", "edge \\ gamma");
    for &g in &gammas {
        print!("  {:>7}", format!("{g}"));
    }
    println!();
    print!("{:>22}", "");
    for _ in &gammas {
        print!("  -------");
    }
    println!();

    // Collect all results first.
    let results: Vec<(f64, Vec<f64>)> = gammas
        .iter()
        .map(|&g| {
            soft_shortest_path_edge_marginals(n, &edges, g).expect("computation should succeed")
        })
        .collect();

    // Print per-edge marginals across temperatures.
    for (i, e) in edges.iter().enumerate() {
        print!("{:>22}", edge_label(e));
        for (value, marginals) in &results {
            let _ = value; // used below
            print!("  {:>7.4}", marginals[i]);
        }
        println!();
    }
    println!();

    // Print soft value at each temperature.
    print!("{:>22}", "soft value");
    for (value, _) in &results {
        print!("  {:>7.2}", value);
    }
    println!();
    println!();

    // Interpretation.
    println!("Interpretation:");
    println!("  gamma -> 0:   marginals concentrate on the shortest path (0->1->4->5, cost=4).");
    println!("                Edges 0->1, 1->4, 4->5 each get marginal ~1.0; all others ~0.0.");
    println!("  gamma -> inf: marginals spread across all paths. Every edge that appears in");
    println!("                at least one s-t path gets positive mass, approaching the uniform");
    println!("                distribution over paths.");
    println!();

    // Verify: at very low gamma, the shortest-path edges should dominate.
    let (_, p_cold) = &results[0]; // gamma = 0.01
    let shortest_path_edges: Vec<usize> = edges
        .iter()
        .enumerate()
        .filter(|(_, e)| {
            // edges on 0->1->4->5
            (e.from == 0 && e.to == 1) || (e.from == 1 && e.to == 4) || (e.from == 4 && e.to == 5)
        })
        .map(|(i, _)| i)
        .collect();

    let off_path: f64 = p_cold
        .iter()
        .enumerate()
        .filter(|(i, _)| !shortest_path_edges.contains(i))
        .map(|(_, &p)| p)
        .sum();

    println!("Sanity check at gamma=0.01: off-shortest-path mass = {off_path:.6} (should be ~0).");
}
