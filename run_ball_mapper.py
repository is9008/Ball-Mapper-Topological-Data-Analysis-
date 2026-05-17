"""
Ball Mapper —> Python implementation
Algorithm 1–4 from Dłotko (2019), arXiv:1901.07410

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle
from matplotlib.colors import Normalize



OUTPUT_DIR = r"C:\Users\ismai\OneDrive - FH Dortmund\Desktop\Master\SS26\InformatikSeminar\Code\images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use("dark_background")
RNG = np.random.default_rng(42)



# ==================================================================
# ALGORITHM 1 — Greedy ε-net
# ==================================================================

def greedy_epsilon_net(X: np.ndarray, eps: float):
    n = len(X)
    covered = np.zeros(n, dtype=bool)
    cover_vector = [[] for _ in range(n)]
    centers = []

    while not covered.all():
        p = int(np.argmax(~covered))
        centers.append(p)
        dists = np.linalg.norm(X - X[p], axis=1)
        in_ball = np.where(dists <= eps)[0]
        for x in in_ball:
            cover_vector[x].append(p)
            covered[x] = True

    return centers, cover_vector


# ==================================================================
# ALGORITHM 2 — MaxMin ε-net
# ==================================================================

def maxmin_epsilon_net(X: np.ndarray, eps: float):
    n = len(X)
    C = [0]                                
    dist_to_C = np.linalg.norm(X - X[0], axis=1)

    while True:
        p = int(np.argmax(dist_to_C))
        d = dist_to_C[p]
        if d <= eps:
            break
        C.append(p)
        new_dists = np.linalg.norm(X - X[p], axis=1)
        dist_to_C = np.minimum(dist_to_C, new_dists)

    cover_vector = [[] for _ in range(n)]
    for c in C:
        in_ball = np.where(np.linalg.norm(X - X[c], axis=1) <= eps)[0]
        for x in in_ball:
            cover_vector[x].append(c)

    return C, cover_vector


# ==================================================================
# ALGO 3  Ball Mapper Graph
# ==================================================================

def build_bm_graph(centers: list, cover_vector: list) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(centers)

    for cv in cover_vector:
        for i in range(len(cv)):
            for j in range(i + 1, len(cv)):
                a, b = cv[i], cv[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    counts = {c: 0 for c in centers}
    for cv in cover_vector:
        for c in cv:
            if c in counts:
                counts[c] += 1
    nx.set_node_attributes(G, counts, "count")

    return G


# ==================================================================
# ALGO 4 — Multiscale Ball Mapper
# ==================================================================

def multiscale_bm(X: np.ndarray, epsilons: list, algo: int = 1):
    eps1 = epsilons[0]
    if algo == 1:
        centers, _ = greedy_epsilon_net(X, eps1)
    else:
        centers, _ = maxmin_epsilon_net(X, eps1)

    results = []
    for eps in epsilons:
        cover_vector = [[] for _ in range(len(X))]
        for c in centers:
            in_ball = np.where(np.linalg.norm(X - X[c], axis=1) <= eps)[0]
            for x in in_ball:
                cover_vector[x].append(c)
        G = build_bm_graph(centers, cover_vector)
        results.append({"eps": eps, "graph": G, "cover_vector": cover_vector})

    return results


# ==================================================================
# POINT CLOUD gen
# ==================================================================

def make_circle(n=200, r=1.0, noise=0.08):
    angles = RNG.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(angles), np.sin(angles)]) * r
    X += RNG.normal(0, noise, X.shape)
    return X

def make_yjunction(n=200, noise=0.03):
    third = n // 3
    stem  = np.column_stack([RNG.normal(0, noise, third),
                             np.linspace(0, 0.6, third)])
    left  = np.column_stack([np.linspace(0, -0.5, third) + RNG.normal(0, noise, third),
                             np.linspace(0.6, 1.1, third) + RNG.normal(0, noise, third)])
    right = np.column_stack([np.linspace(0, 0.5, n - 2*third) + RNG.normal(0, noise, n - 2*third),
                             np.linspace(0.6, 1.1, n - 2*third) + RNG.normal(0, noise, n - 2*third)])
    return np.vstack([stem, left, right])

def make_torus(n=300, R=1.0, r=0.35):
    theta = RNG.uniform(0, 2 * np.pi, n)
    phi   = RNG.uniform(0, 2 * np.pi, n)
    X = np.column_stack([
        (R + r * np.cos(phi)) * np.cos(theta),
        (R + r * np.cos(phi)) * np.sin(theta),
    ])
    return X


# ==================================================================
# drawing HELPERS
# ==================================================================

ACCENT   = "#4af"
BG       = "#0a0a0f"

def _draw_bm(ax, X, G, eps, title, show_balls=True, show_points=True):
    ax.set_facecolor(BG)
    ax.set_title(title, color="#aac8f0", fontsize=9, pad=6, fontfamily="monospace")
    ax.set_aspect("equal")
    ax.axis("off")

    counts = np.array([G.nodes[n].get("count", 0) for n in G.nodes()])
    max_count = counts.max() if counts.size else 1
    cmap = plt.get_cmap("cool")
    norm = Normalize(vmin=0, vmax=max_count)

    if show_points and len(X):
        ax.scatter(X[:, 0], X[:, 1], s=2, color="white", alpha=0.15, zorder=1)

    if show_balls:
        for n in G.nodes():
            c = Circle(X[n], eps, color=ACCENT, alpha=0.06, zorder=2)
            ax.add_patch(c)

    weights = np.array([d.get("weight", 1) for _, _, d in G.edges(data=True)])
    max_w   = weights.max() if weights.size else 1
    for (u, v, d) in G.edges(data=True):
        w = d.get("weight", 1)
        ax.plot([X[u][0], X[v][0]], [X[u][1], X[v][1]],
                color="white", alpha=0.12 + 0.4 * (w / max_w),
                linewidth=0.6 + 1.4 * (w / max_w), zorder=3)

    node_list = list(G.nodes())
    node_counts = np.array([G.nodes[n].get("count", 0) for n in node_list])
    node_colors = cmap(norm(node_counts))
    node_sizes  = 30 + 120 * (node_counts / max_count)
    xs = np.array([X[n][0] for n in node_list])
    ys = np.array([X[n][1] for n in node_list])
    ax.scatter(xs, ys, s=node_sizes, c=node_colors, zorder=4, edgecolors="white",
               linewidths=0.5, alpha=0.95)

    return norm, cmap


# ==================================================================
# FIGURES 
# ==================================================================

def figure_alg1_vs_alg2():
    X = make_circle(n=150, r=1.0, noise=0.06)
    eps = 0.30

    c1, cv1 = greedy_epsilon_net(X, eps)
    G1 = build_bm_graph(c1, cv1)

    c2, cv2 = maxmin_epsilon_net(X, eps)
    G2 = build_bm_graph(c2, cv2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithms 1 & 2 — Cover strategies  |  circle  |  ε = 0.30",
                 color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    _draw_bm(axes[0], X, G1, eps, f"Algorithm 1 · Greedy ε-net\n|centers|={len(c1)}")
    _draw_bm(axes[1], X, G2, eps, f"Algorithm 2 · Max-min ε-net\n|centers|={len(c2)}")

    for ax in axes:
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig1_alg1_vs_alg2.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def figure_alg3_datasets():
    datasets = [
        (make_circle(200),    0.28,  "circle",    (-1.3,1.3), (-1.3,1.3)),
        (make_yjunction(200), 0.12,  "y-junction",(-0.7,0.7), (-0.15,1.25)),
        (make_torus(300),     0.28,  "torus",     (-1.6,1.6), (-1.6,1.6)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithm 3 — Ball Mapper Graph", color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    for ax, (X, eps, name, xlim, ylim) in zip(axes, datasets):
        centers, cv = greedy_epsilon_net(X, eps)
        G = build_bm_graph(centers, cv)
        _draw_bm(ax, X, G, eps, f"{name}  ·  ε={eps}\n|V|={G.number_of_nodes()}")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig2_alg3_datasets.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def figure_alg4_multiscale():
    X = make_torus(n=300)
    epsilons = [0.18, 0.30, 0.48, 0.70]
    scales = multiscale_bm(X, epsilons, algo=1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithm 4 — Multi-scale Ball Mapper", color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    for ax, s in zip(axes, scales):
        G = s["graph"]
        _draw_bm(ax, X, G, s["eps"], f"ε = {s['eps']:.2f}\n|V|={G.number_of_nodes()}")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig3_alg4_multiscale.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


def figure_dimension_estimation():
    dims = [2, 3, 4]
    epsilons = np.linspace(0.10, 0.50, 10)
    n_pts = 300
    n_trials = 3   

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_title("Dimension estimation via average BM neighbour count", color="#9de", fontsize=11)
    ax.set_xlabel("ε", color="#aac8f0")
    ax.set_ylabel("avg neighbours per node", color="#aac8f0")

    cmap_d = plt.get_cmap("plasma")
    colors = [cmap_d(i / (len(dims) - 1)) for i in range(len(dims))]

    for d, col in zip(dims, colors):
        avg_neighbours = []
        for eps in epsilons:
            trial_avgs = []
            for _ in range(n_trials):
                X = RNG.uniform(0, 1, (n_pts, d))
                centers, cv = greedy_epsilon_net(X, eps)
                G = build_bm_graph(centers, cv)
                if G.number_of_nodes() > 0:
                    trial_avgs.append(np.mean([G.degree(n) for n in G.nodes()]))
            avg_neighbours.append(np.mean(trial_avgs) if trial_avgs else 0)
        ax.plot(epsilons, avg_neighbours, color=col, linewidth=2, label=f"d = {d}")

    ax.legend()
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "fig4_dimension_estimation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    print("Starting generation ")
    figure_alg1_vs_alg2()
    figure_alg3_datasets()
    figure_alg4_multiscale()
    figure_dimension_estimation()
    print("\nall figure generated successfully")