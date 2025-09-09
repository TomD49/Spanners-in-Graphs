# experiments.py
# Run experiments for greedy spanner on various graph families.
from __future__ import annotations
import argparse
import math
import sys
from typing import List, Tuple, Optional
import csv

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from spanner import (
    greedy_spanner,
    mst_weight,
    graph_weight,
    compute_stretch,
    random_complete_weighted_graph,
    er_random_graph,
    random_points,
)


def run_single(G: nx.Graph, t_values: List[float], all_pairs: bool, name: str) -> List[dict]:
    """
    For a fixed input graph G, run greedy spanner for each t in t_values and collect metrics.
    Returns list of dict rows.
    """
    rows = []
    n = G.number_of_nodes()
    m = G.number_of_edges()
    w_mst = mst_weight(G)

    for t in t_values:
        H = greedy_spanner(G, t=t)
        size = H.number_of_edges()
        wH = graph_weight(H)
        stretch = compute_stretch(G, H, all_pairs=all_pairs)

        rows.append({
            "graph": name,
            "n": n,
            "m": m,
            "t": t,
            "spanner_edges": size,
            "spanner_weight": wH,
            "mst_weight": w_mst,
            "weight_ratio": wH / w_mst,
            "max_stretch": stretch
        })
    return rows


def scenario_complete_random(n: int, t_values: List[float], all_pairs: bool):
    G = random_complete_weighted_graph(n)
    return run_single(G, t_values, all_pairs, name=f"complete_random(n={n})")


def scenario_er(n: int, p: float, t_values: List[float], all_pairs: bool):
    G = er_random_graph(n, p)
    return run_single(G, t_values, all_pairs, name=f"er(n={n},p={p})")



def save_csv(rows: List[dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] Saved CSV: {path}")


def maybe_plot(rows: List[dict], out_png: Optional[str]):
    if plt is None or not rows:
        return
    # Plot: t vs weight_ratio and t vs spanner_edges for each scenario
    by_graph: dict = {}
    for r in rows:
        by_graph.setdefault(r["graph"], []).append(r)
    fig_count = 0
    for name, lst in by_graph.items():
        lst = sorted(lst, key=lambda x: x["t"])
        ts = [x["t"] for x in lst]
        wr = [x["weight_ratio"] for x in lst]
        m_edges = [x["spanner_edges"] for x in lst]
        stretch = [x["max_stretch"] for x in lst]

        plt.figure()
        plt.plot(ts, wr, marker="o", label="weight_ratio (W(H)/W(MST))")
        plt.plot(ts, m_edges, marker="o", label="#edges in spanner")
        plt.plot(ts, stretch, marker="o", label="max_stretch")
        plt.xlabel("t (stretch target)")
        plt.title(name)
        plt.legend()
        plt.grid(True)
        fig_count += 1
    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        print(f"[OK] Saved plot(s) to {out_png}")
    else:
        plt.show()

"""
def main():
    parser = argparse.ArgumentParser(description="Greedy t-spanner experiments")
    sub = parser.add_subparsers(dest="scenario")
    parser.set_defaults(scenario="er")

    # complete random
    p_complete = sub.add_parser("complete", help="Complete graph with random weights")
    p_complete.add_argument("--n", type=int, default=100)

    # ER
    p_er = sub.add_parser("er", help="Erdos-Renyi G(n,p)")
    p_er.add_argument("--n", type=int, default=40)
    p_er.add_argument("--p", type=float, default=0.15)

    # common args
    for sp in [p_complete, p_er]:
        sp.add_argument("--t", type= float, nargs="+", default=[1.5, 2.0, 3.0])
        sp.add_argument("--all-pairs", action="store_true",
                        help="Use all-pairs distances (slower but exact); default: sample pairs")
        sp.add_argument("--csv", type=str, default="", help="Save results CSV to path")
        sp.add_argument("--plot", type=str, default="", help="Save plots to PNG (or show if empty)")

    args = parser.parse_args()

    if args.scenario == "complete":
        rows = scenario_complete_random(args.n, args.t, args.all_pairs)
    elif args.scenario == "er":
        rows = scenario_er(args.n, args.p, args.t, args.all_pairs)
    else:
        print("Unknown scenario", file=sys.stderr)
        sys.exit(2)

    # Print table
    if rows:
        header = list(rows[0].keys())
        print("\t".join(header))
        for r in rows:
            print("\t".join(str(r[h]) for h in header))

    # Save CSV
    if getattr(args, "csv", ""):
        save_csv(rows, args.csv)

    # Plots
    maybe_plot(rows, args.plot)
"""

def main():
    parser = argparse.ArgumentParser(description="Greedy t-spanner experiments")
    sub = parser.add_subparsers(dest="scenario")

    # Parent with common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--t", type=float, nargs="+", default=[1.5, 2.0, 3.0])
    common.add_argument("--all-pairs", action="store_true",
                        help="Use all-pairs distances (slower but exact); default: sample pairs")
    common.add_argument("--csv", type=str, default="", help="Save results CSV to path")
    common.add_argument("--plot", type=str, default="", help="Save plots to PNG (or show if empty)")

    # complete
    p_complete = sub.add_parser("complete", parents=[common],
                                help="Complete graph with random weights")
    p_complete.add_argument("--n", type=int, default=100)

    # er
    p_er = sub.add_parser("er", parents=[common], help="Erdos-Renyi G(n,p)")
    p_er.add_argument("--n", type=int, default=40)
    p_er.add_argument("--p", type=float, default=0.15)

    # default to "er" if no scenario given
    argv = sys.argv[1:]
    if not argv or argv[0] not in {"complete", "er"}:
        argv = ["er"] + argv

    args = parser.parse_args(argv)

    # הרצה
    if args.scenario == "complete":
        rows = scenario_complete_random(args.n, args.t, args.all_pairs)
    elif args.scenario == "er":
        rows = scenario_er(args.n, args.p, args.t, args.all_pairs)
    else:
        print("Unknown scenario", file=sys.stderr)
        sys.exit(2)

    # chart
    if rows:
        header = list(rows[0].keys())
        print("\t".join(header))
        for r in rows:
            print("\t".join(str(r[h]) for h in header))

    # CSV
    if getattr(args, "csv", ""):
        save_csv(rows, args.csv)

    # graphical output
    maybe_plot(rows, args.plot)




if __name__ == "__main__":
    main()
