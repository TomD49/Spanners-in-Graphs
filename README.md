# Greedy $t$-Spanner Experiments

This repository provides a **clean Python implementation** of **greedy $t$-spanners** and a suite of experiments to explore **sparse subgraph design** and the theoretical bounds on size, weight, and stretch.

## 🌐 Background

A **$t$-spanner** of a weighted graph $G$ is a sparse subgraph $H$ such that  

dist_H(u,v) ≤ t · dist_G(u,v)   for all (u,v) in G

The greedy construction offers **provable guarantees** on size and weight while remaining conceptually simple and practical.

## ✨ Features

* **Greedy $r$-Spanner** (`greedy_spanner`) – builds a sparse subgraph while preserving distances within a factor $r$.  
* **Random Graph Generators** – create undirected random-weighted  graphs for testing:
  * Complete graphs
  * Random graphs
  * High-girth random graphs (including a BFS-based cycle-avoidance method)

## 📊 Metrics & Evaluation

Each experiment prints a single CSV-style line with the following metrics:

* **$t$** – The stretch parameter controlling how well distances are preserved.  
  The algorithm internally sets $r = 2t + 1$. Larger $t$ values typically allow sparser spanners.

* **$\lvert E(G) \rvert$** – Number of edges in the input graph $G$, indicating the density of the original graph.

* **$\lvert E(H) \rvert$** – Number of edges in the resulting spanner $H$, showing how sparse the spanner is compared to $G$.

* **$w(G)$** – Total weight of all edges in the input graph $G$, serving as a baseline for weight comparisons.

* **$w(H)$** – Total weight of all edges in the spanner $H$, a key measure of how “light” the spanner is.

* **$w(\mathrm{MST}(G))$** – Weight of the minimum spanning tree of $G$, which acts as a natural lower bound for any connected subgraph.

* **`max_stretch`** – Maximum observed stretch max_{u,v} [ dist_H(u,v) / dist_G(u,v) ]

indicating how well the spanner preserves pairwise distances relative to the target $r$.

* **`edge_bound_pass`** – Boolean (`True`/`False`) indicating whether the size bound |E(H)| < n · ⌈ n^(1/t) ⌉ is satisfied.

* **`weight_bound_pass`** – Boolean indicating whether the weight bound w(H) < w(MST(G)) · (1 + n / (2t)) is satisfied.

* **`edge_bound_value`** – The computed upper limit n · ⌈ n^(1/t) ⌉. used in the size-bound check.

* **`weight_bound_value`** – The computed upper limit w(MST(G)) · (1 + n / (2t)). used in the weight-bound check.

* **`build_seconds`** – Time (in seconds) required to generate the random input graph $G$.

* **`spanner_seconds`** – Time (in seconds) required to run the greedy spanner algorithm on $G$.

These metrics reveal how **sparse and light** the resulting spanner is, whether it meets theoretical guarantees, and how much computational effort each stage requires.

## 🚀 Quick Start

### 1️⃣ Clone and Install
```bash
git clone https://github.com/<your-user>/greedy-spanner-experiments.git
cd greedy-spanner-experiments
pip install -r requirements.txt   # installs networkx, numpy
```

### 2️⃣ Run Experiments
```bash
python main.py
```

The script prints CSV-style output with columns:
```
t, run_idx, |E(G)|, |E(H)|, w(G), w(H), w(MST),
max_stretch, edge_bound_pass, weight_bound_pass,
edge_bound_value, weight_bound_value,
build_seconds, spanner_seconds
```

### 3️⃣ Adjust Parameters
Open **`main.py`** and edit the `CONFIG` dictionary to:

* Switch graph types (`"complete"`, `"er"`, `"high_girth"`)
* Change vertex count, weight range, or stretch factors
* Control the number of runs per setting

## 📂 Project Structure
```
spanner.py   # Core algorithms and graph generators
main.py      # Experiment runner and bound checks
requirements.txt 
```

Enjoy experimenting with **greedy $t$-spanners** and exploring the trade-offs between sparsity, weight, and distance preservation!
