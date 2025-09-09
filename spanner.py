# spanner.py
# Greedy t-spanner & utilities
from __future__ import annotations
from typing import Optional, Dict
import math
import random
import networkx as nx
import numpy as np


# =======================
# Core: Greedy t-Spanner
# =======================

def greedy_spanner(G: nx.Graph, t: float) -> nx.Graph:
    """
    Build a greedy t-spanner from a weighted, connected graph G.
    For each edge (u,v) in nondecreasing weight order, add it iff
    the current spanner has dist(u,v) > t * w(u,v) (or no path).

    :param G: undirected weighted graph (edge attr 'weight' must exist)
    :param t: stretch factor target (t >= 1.0)
    :return: H, a t-spanner subgraph of G
    """
    if t < 1.0:
        raise ValueError("t must be >= 1.0")
    H : nx.Graph = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Sort edges by weight ascending
    edges_sorted = sorted(G.edges(data=True), key=lambda e: e[2].get("weight", 1.0))

    for u, v, data in edges_sorted:
        w = data.get("weight", 1.0)
        if not nx.has_path(H, u, v):
            H.add_edge(u, v, weight=w)
        else:
            # Shortest path in current H
            d = nx.shortest_path_length(H, u, v, weight="weight")
            if d > t * w:
                H.add_edge(u, v, weight=w)
    return H


# =======================
# Metrics / Evaluation
# =======================

def mst_weight(G: nx.Graph) -> float:
    """Return the total weight of a minimum spanning tree of G (Kruskal/Prim)."""
    T = nx.minimum_spanning_tree(G, weight="weight")
    weights = [d.get("weight", 1.0) for _, _, d in T.edges(data=True)]
    total_weight = sum(weights)
    return total_weight


def graph_weight(G: nx.Graph) -> float:
    """Sum of edge weights in G."""
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    total_weight = sum(weights)
    return total_weight


def compute_stretch(G: nx.Graph, H: nx.Graph,
                    sample_pairs: int = 500,
                    rng: Optional[random.Random] = None) -> float:
    """
    Compute the maximum observed stretch max_{u,v} dist_H(u,v)/dist_G(u,v).
    Assumes G is connected and H spans the same vertex set.
    uses all-pairs Dijkstra (O(n*(m+n log n))).
    Returns: max stretch observed (>=1.0)
    """
    if rng is None:
        rng = random.Random(0)

    nodes = list(G.nodes())
    # Precompute all-pairs shortest path lengths
    distG: Dict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    distH: Dict = dict(nx.all_pairs_dijkstra_path_length(H, weight="weight"))
    max_ratio = 1.0
    for u in nodes:
        for v in nodes:
            if u == v:
                continue
            duv = distG[u].get(v, math.inf)
            huv = distH[u].get(v, math.inf)
            if duv == 0 or duv == math.inf:
                continue
            ratio = huv / duv
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio
    


# =======================
# Graph Generators
# =======================

def random_complete_weighted_graph(n: int,
                                   w_low: float = 0.0,
                                   w_high: float = 1.0) -> nx.Graph:
    """Complete graph on n nodes with uniform edge weights in [w_low, w_high]."""
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            w = random.uniform(w_low, w_high)
            G.add_edge(i, j, weight=w)
    return G


def er_random_graph(n: int, p: float,
                    w_low: float = 0.0,
                    w_high: float = 1.0) -> nx.Graph:
    """
    Erdos–Rényi G(n,p) (undirected, simple). If disconnected, lightly augment to connect.
    Random uniform weights on edges.
    """
    # create random graph
    G = nx.gnp_random_graph(n, p, directed=False)

    # if not connected, add edges to connect components
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for a, b in zip(comps, comps[1:]):
            u = a[0]; v = b[0]
            G.add_edge(u, v)

    # add random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = random.uniform(w_low, w_high)

    return G

def high_girth_random_graph(n: int, t: float,
                    w_low: float = 0.0,
                    w_high: float = 1.0) -> nx.Graph:
    """
    high girth graph. If disconnected, lightly augment to connect.
    Random uniform weights on edges.
    """
    # create random graph
    p = (n ** (1 / t)) / n
    G = nx.gnp_random_graph(n, p, directed=False)

    # if not connected, add edges to connect components
    if not nx.is_connected(G):
        comps = [list(c) for c in nx.connected_components(G)]
        for a, b in zip(comps, comps[1:]):
            u = a[0]; v = b[0]
            G.add_edge(u, v)

    # add random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = random.uniform(w_low, w_high)

    return G


# =======================
# Random points utility
# =======================

def random_points(n: int, d: int = 2) -> "np.ndarray":
    """
    Uniform points in [0,1]^d. Requires numpy.
    """
    return np.random.random((n, d))
