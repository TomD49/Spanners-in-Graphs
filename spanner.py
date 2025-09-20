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

def greedy_spanner(G: nx.Graph, r: float) -> nx.Graph:
    """
    Build a greedy r-spanner from a weighted, connected graph G.
    For each edge (u,v) in nondecreasing weight order, add it iff
    the current spanner has dist(u,v) > r * w(u,v) (or no path).

    :param G: undirected weighted graph (edge attr 'weight' must exist)
    :param t: stretch factor target (r >= 1.0)
    :return: H, a r-spanner subgraph of G
    """
    if r < 1.0:
        raise ValueError("r must be >= 1.0")
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
            if d > r * w:
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


def random_graph(n: int, p: float,
                    w_low: float = 0.0,
                    w_high: float = 1.0) -> nx.Graph:
    """
    random G(n,p) (undirected, simple). If disconnected, lightly augment to connect.
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


def high_girth_random_graph_bfs(n: int,
                                       t: float,
                                       w_low: float = 0.0,
                                       w_high: float = 1.0) -> nx.Graph:
    """
    Generates an undirected random graph with edge probability p = (n**(1/t))/n,
    assigns edge weights uniformly in the range [w_low, w_high],
    and builds a final subgraph with no cycles shorter than t
    by adding edges only if they do not close a short cycle.
    The check uses BFS via `shortest_path_length` with a cutoff.

    Idea: if in the current subgraph the distance d(u,v) < t-1,
        then adding the edge (u,v) would create a cycle of length d(u,v) + 1 < t ⇒ skip it.
        Otherwise, add the edge and assign a random weight.
    """

    # Calculate edge probability p to achieve desired girth
    p = (n ** (1.0 / t)) / n
    p = min(max(p, 0.0), 1.0)

    # Create the initial random graph G0 to sample edges from 
    G0 = nx.gnp_random_graph(n, p, directed=False)

    # construct H- add edges from G0 only if they do not close a cycle shorter than t
    H : nx.Graph = nx.Graph()
    H.add_nodes_from(range(n))

    #shuffle edges to get a random order
    edges = list(G0.edges())
    random.shuffle(edges)

    # If there is a path of length <= floor(t-1), adding (u,v) would form a cycle of length < t
    cutoff = math.floor(t - 1)

    for u, v in edges:
        if H.has_edge(u, v):
            continue  # already have this edge
        can_add = True
        # Check if there's a path of length <= cutoff between u and v in H
        dists = nx.single_source_shortest_path_length(H, source=u, cutoff=cutoff)
        if v in dists:        # v reachable within <= cutoff ⇒ would create short cycle
            can_add = False

        if can_add:
            H.add_edge(u, v, weight=random.uniform(w_low, w_high))

    # If H is not connected, add edges to connect components
    # This never creates a cycle because there is no existing path between the components.
    if not nx.is_connected(H):
        comps = [list(c) for c in nx.connected_components(H)]
        for a, b in zip(comps, comps[1:]):
            u = a[0]
            v = b[0]
            H.add_edge(u, v, weight=random.uniform(w_low, w_high))
    
    return H

# =======================
# Random points utility
# =======================

def random_points(n: int, d: int = 2) -> "np.ndarray":
    """
    Uniform points in [0,1]^d. Requires numpy.
    """
    return np.random.random((n, d))
