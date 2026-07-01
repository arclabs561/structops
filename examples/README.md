# structops examples

Each example is runnable from the repo root. Output excerpts below are real,
captured from release runs.

## Which example should I run?

| I want to... | Example |
|---|---|
| See edge marginals as path attention | `soft_path_attention` |
| Find an integer shift with Soft-DTW | `soft_dtw_shift_scan` |
| Score noisy user journeys against a canonical path | `user_journey_alignment` |
| Align noisy sentence sequences | `sentence_alignment_soft_dtw` |
| Run soft shortest path over a graphops graph | `soft_path_on_graph` |

## Soft Shortest Path

### `soft_path_attention`: what do edge marginals look like?

Computes soft shortest-path edge marginals on a tiny two-path DAG. Cheap edges
carry nearly all mass at low temperature.

```bash
cargo run --release --example soft_path_attention
```

```text
soft shortest-path value = 1.9998322968135522
edge 0 (0->1, cost=1): marginal p(e in path) = 0.9996646498695337
edge 1 (1->3, cost=1): marginal p(e in path) = 0.9996646498695337
edge 2 (0->2, cost=3): marginal p(e in path) = 0.00033535013046647805
edge 3 (2->3, cost=3): marginal p(e in path) = 0.00033535013046647805
```

### `soft_path_on_graph`: how does temperature spread path mass?

Builds a weighted DAG with `graphops`, converts it to `structops::Edge`, and
sweeps the smoothing parameter.

```bash
cargo run --release --example soft_path_on_graph
```

```text
Feasible paths (source=0, sink=5):
  0->1->3->5  cost = 9
  0->1->4->5  cost = 4  <-- shortest
  0->2->3->5  cost = 12
  0->2->4->5  cost = 8

          edge \ gamma     0.01      0.1      0.5        1        2        5       50
       0->1 (cost=2.0)   1.0000   1.0000   0.9997   0.9818   0.8757   0.6775   0.5176
       1->4 (cost=1.0)   1.0000   1.0000   0.9996   0.9752   0.8092   0.4953   0.2717
       4->5 (cost=1.0)   1.0000   1.0000   1.0000   0.9931   0.9188   0.7178   0.5226

Sanity check at gamma=0.01: off-shortest-path mass = 0.000000 (should be ~0).
```

## Soft-DTW

### `soft_dtw_shift_scan`: can Soft-DTW recover a circular shift?

Builds a sinusoid, shifts it by three samples, then scans inverse shifts with
Soft-DTW divergence.

```bash
cargo run --release --example soft_dtw_shift_scan
```

```text
gamma=0.5  original_shift=3
shift  soft_dtw_divergence
   -5  1.198787
   -4  0.296093
   -3  0.000000
   -2  0.282719
   -1  1.088817
    0  2.314612

best_inverse_shift=-3  best_div=0.000000
```

### `user_journey_alignment`: how does Soft-DTW score noisy paths?

Compares user sessions with a canonical path using an explicit categorical
cost matrix. Extra steps are less severe than missing the goal.

```bash
cargo run --release --example user_journey_alignment
```

```text
Canonical path : Landing -> Pricing -> SignUp

User A (Ideal) : Landing -> Pricing -> SignUp
   Score: 0.0000 (exact match)
User B (Noisy) : Landing -> Blog -> Pricing -> Blog -> Pricing -> SignUp
   Score: 1.9827 (High alignment despite noise)
User C (Bounce): Landing -> Blog -> Exit
   Score: 2.8463 (Poor alignment)
```

### `sentence_alignment_soft_dtw`: can ordered text alignment tolerate boilerplate?

Embeds sentences with signed character n-gram hashing, builds a cosine-distance
cost matrix, and computes a sequence-aware Soft-DTW value.

```bash
cargo run --release --example sentence_alignment_soft_dtw
```

```text
Reference sentences (3):
  0: Quarterly earnings showed steady growth in all sectors.
  1: Revenue was up 12% year-over-year.
  2: Guidance remains unchanged.

Noisy sentences (5):
  0: CONFIDENTIAL - INTERNAL MEMO.
  1: Qarterly earnigns showd stdy grwth across sectrs.
  2: Revnue +12 percent YoY.
  3: MENU HOME CONTACT.
  4: Guidance remains unchngd.

Soft-DTW value (gamma=0.5): 1.569354

Greedy best sentence matches (by min cost):
  ref[0] -> noisy[1]  dist=0.360
  ref[2] -> noisy[4]  dist=0.245
```
