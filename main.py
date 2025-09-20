# main.py
from __future__ import annotations
import time
import math
from typing import List, Tuple

import networkx as nx

from spanner import (
    greedy_spanner,
    mst_weight,
    graph_weight,
    compute_stretch,
    random_complete_weighted_graph,
    random_graph,
    high_girth_random_graph_bfs,
)

# =======================
# Global experiment config
# =======================
CONFIG = {
    "graph_type": "random",                      # "complete" | "random" | "high_girth"
    "n": 1000,                                   # number of vertices
    "p": 0.1,                                    # used only when graph_type == "random"
    "weight_range": (0.00001, 1.0),              # (w_low, w_high)
    "t": [1.00001, 1.25, 2.5, 5.0, 10.0],        # stretch factors to test
    "runs_per_t": 10,                            # number of graph experiments for each stretch factor t
}


def build_graph(graph_type: str, n: int, t: float, p: float,
                w_low: float, w_high: float) -> nx.Graph:
    """
    Build a graph according to graph_type and parameters.
    - "complete": uses random_complete_weighted_graph
    - "random":       uses random_graph with edge prob p
    - "high_girth": uses high_girth_random_graph(n, t)  (t affects p internally)
    """
    if graph_type == "complete":
        return random_complete_weighted_graph(n, w_low=w_low, w_high=w_high)
    elif graph_type == "random":
        return random_graph(n, p=p, w_low=w_low, w_high=w_high)
    elif graph_type == "high_girth":
        return high_girth_random_graph_bfs(n, t=t, w_low=w_low, w_high=w_high)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


def check_bounds(n: int, t: float, H: nx.Graph, wH: float, wMST: float) -> Tuple[bool, bool, float, float]:
    """
    Check the two bounds requested:
    1) size(H) = |E(H)| < n * (n^(1/t))
    2) weight(H) < weight(MST(G)) * (1 + n/(2*t))
    Returns [ok_edges, ok_weight, edge_bound_value, weight_bound_value]
    """
    size_H = H.number_of_edges()
    edge_bound = n * math.ceil(n ** (1.0 / t))
    ok_edges = size_H < edge_bound

    weight_bound_mult = 1.0 + n / (2.0 * t)
    weight_bound = wMST * weight_bound_mult
    ok_weight = wH < weight_bound

    return ok_edges, ok_weight, edge_bound, weight_bound


def run_experiments():
    cfg = CONFIG
    graph_type = cfg["graph_type"]
    n = cfg["n"]
    p = cfg["p"]
    w_low, w_high = cfg["weight_range"]
    t_values: List[float] = cfg["t"]
    runs_per_t: int = cfg["runs_per_t"]

    print("# Greedy t-spanner experiment")
    print(f"# graph_type={graph_type}, n={n}, p={p if graph_type=='random' else 'N/A'}, "
          f"weight_range=({w_low},{w_high}), runs_per_t={runs_per_t}")
    print("# Columns: t, run_idx, |E(G)|, |E(H)|, w(G), w(H), w(MST), max_stretch, "
          "edge_bound_pass, weight_bound_pass, edge_bound_value, weight_bound_value, "
          "build_seconds, spanner_seconds")

    for t in t_values:
        for run_idx in range(runs_per_t):
            # Build graph
            t0 = time.time()
            G = build_graph(graph_type, n, t, p, w_low, w_high)
            r = 2*t +1  
            build_sec = time.time() - t0
            wG = graph_weight(G)

            # Compute MST weight once per G
            wMST = mst_weight(G)

            # Spanner
            t1 = time.time()
            H = greedy_spanner(G, r=r)
            spanner_sec = time.time() - t1

            # Metrics
            m = G.number_of_edges()
            size_H = H.number_of_edges()
            wH = graph_weight(H)
            stretch = compute_stretch(G, H)

            # Bounds
            ok_edges, ok_weight, edge_bound_val, weight_bound_val = check_bounds(n, t, H, wH, wMST)

            # Print a single CSV-like line
            print(f"{t:.6f}, {run_idx}, {m}, {size_H}, {wG:.6f}, {wH:.6f}, {wMST:.6f}, "
                  f"{stretch:.6f}, "
                  f"{ok_edges}, {ok_weight}, "
                  f"{edge_bound_val:.6f}, {weight_bound_val:.6f}, "
                  f"{build_sec:.3f}, {spanner_sec:.3f}")


if __name__ == "__main__":
    run_experiments()
