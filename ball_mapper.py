"""
Ball Mapper — Python implementation
Algorithms 1–4 from Dłotko (2019), arXiv:1901.07410

Algorithms:
  1. Greedy ε-net
  2. Max-min ε-net
  3. Ball Mapper Graph construction
  4. Multi-scale Ball Mapper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.style.use("dark_background")
RNG = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════
# ALGORITHM 1 — Greedy ε-net
# ══════════════════════════════════════════════════════════════════

def greedy_epsilon_net(X: np.ndarray, eps: float):
    """
    Algorithm 1: Greedy ε-net cover.

    Parameters
    ----------
    X   : (n, d) array of points
    eps : ball radius

    Returns
    -------
    centers      : list of indices into X that are ball centers
    cover_vector : list of lists — cover_vector[i] = centers covering X[i]
    """
    n = len(X)
    covered = np.zeros(n, dtype=bool)
    cover_vector = [[] for _ in range(n)]
    centers = []

    while not covered.all():
        # Pick first uncovered point
        p = int(np.argmax(~covered))
        centers.append(p)
        # Find all points within eps of p
        dists = np.linalg.norm(X - X[p], axis=1)
        in_ball = np.where(dists <= eps)[0]
        for x in in_ball:
            cover_vector[x].append(p)
            covered[x] = True

    return centers, cover_vector


# ══════════════════════════════════════════════════════════════════
# ALGORITHM 2 — Max-min ε-net
# ══════════════════════════════════════════════════════════════════

def maxmin_epsilon_net(X: np.ndarray, eps: float):
    """
    Algorithm 2: Max-min ε-net cover.

    Selects centers by always picking the point farthest from
    existing centers — produces a more evenly spread cover.

    Parameters
    ----------
    X   : (n, d) array of points
    eps : ball radius

    Returns
    -------
    centers      : list of indices
    cover_vector : list of lists
    """
    n = len(X)
    C = [0]                              # start with arbitrary point
    dist_to_C = np.linalg.norm(X - X[0], axis=1)

    while True:
        p = int(np.argmax(dist_to_C))
        d = dist_to_C[p]
        if d <= eps:
            break
        C.append(p)
        # Update min-distance to C for every point
        new_dists = np.linalg.norm(X - X[p], axis=1)
        dist_to_C = np.minimum(dist_to_C, new_dists)

    cover_vector = [[] for _ in range(n)]
    for c in C:
        in_ball = np.where(np.linalg.norm(X - X[c], axis=1) <= eps)[0]
        for x in in_ball:
            cover_vector[x].append(c)

    return C, cover_vector


# ══════════════════════════════════════════════════════════════════
# ALGORITHM 3 — Ball Mapper Graph
# ══════════════════════════════════════════════════════════════════

def build_bm_graph(centers: list, cover_vector: list) -> nx.Graph:
    """
    Algorithm 3: Construct Ball Mapper graph from a cover vector.

    Vertices  = ball centers
    Edges     = pairs of centers that share at least one covered point
    Edge weight = number of shared points

    Parameters
    ----------
    centers      : list of center indices
    cover_vector : list of lists — cover_vector[i] = centers covering point i

    Returns
    -------
    G : NetworkX Graph with node attribute 'count' and edge attribute 'weight'
    """
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

    # Store coverage count per node
    counts = {c: 0 for c in centers}
    for cv in cover_vector:
        for c in cv:
            if c in counts:
                counts[c] += 1
    nx.set_node_attributes(G, counts, "count")

    return G


# ══════════════════════════════════════════════════════════════════
# ALGORITHM 4 — Multi-scale Ball Mapper
# ══════════════════════════════════════════════════════════════════

def multiscale_bm(X: np.ndarray, epsilons: list, algo: int = 1):
    """
    Algorithm 4: Multi-scale Ball Mapper.

    Fixes ball centers using the smallest ε, then builds BM graphs
    for each ε in the sequence using the same centers.

    Parameters
    ----------
    X        : (n, d) point cloud
    epsilons : sorted list of radii [ε1 ≤ ε2 ≤ … ≤ εn]
    algo     : 1 = greedy (Alg.1), 2 = max-min (Alg.2)

    Returns
    -------
    results : list of dicts with keys 'eps', 'graph', 'cover_vector'
    """
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


# ══════════════════════════════════════════════════════════════════
# POINT CLOUD GENERATORS
# ══════════════════════════════════════════════════════════════════

def make_circle(n=200, r=1.0, noise=0.08):
    angles = RNG.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(angles), np.sin(angles)]) * r
    X += RNG.normal(0, noise, X.shape)
    return X

def make_uniform(n=200):
    return RNG.uniform(0, 1, (n, 2))

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


# ══════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════

ACCENT   = "#4af"
BG       = "#0a0a0f"
NODEBASE = "#1a6fff"

def _draw_bm(ax, X, G, eps, title, show_balls=True, show_points=True):
    ax.set_facecolor(BG)
    ax.set_title(title, color="#aac8f0", fontsize=9, pad=6, fontfamily="monospace")
    ax.set_aspect("equal")
    ax.axis("off")

    counts = np.array([G.nodes[n].get("count", 0) for n in G.nodes()])
    max_count = counts.max() if counts.size else 1
    cmap = plt.get_cmap("cool")
    norm = Normalize(vmin=0, vmax=max_count)

    pos = {n: X[n] for n in G.nodes()}

    # Point cloud
    if show_points and len(X):
        ax.scatter(X[:, 0], X[:, 1], s=2, color="white", alpha=0.15, zorder=1)

    # Balls
    if show_balls:
        for n in G.nodes():
            c = Circle(X[n], eps, color=ACCENT, alpha=0.06, zorder=2)
            ax.add_patch(c)

    # Edges
    weights = np.array([d.get("weight", 1) for _, _, d in G.edges(data=True)])
    max_w   = weights.max() if weights.size else 1
    for (u, v, d) in G.edges(data=True):
        w = d.get("weight", 1)
        ax.plot([X[u][0], X[v][0]], [X[u][1], X[v][1]],
                color="white", alpha=0.12 + 0.4 * (w / max_w),
                linewidth=0.6 + 1.4 * (w / max_w), zorder=3)

    # Nodes
    node_list = list(G.nodes())
    node_counts = np.array([G.nodes[n].get("count", 0) for n in node_list])
    node_colors = cmap(norm(node_counts))
    node_sizes  = 30 + 120 * (node_counts / max_count)
    xs = np.array([X[n][0] for n in node_list])
    ys = np.array([X[n][1] for n in node_list])
    ax.scatter(xs, ys, s=node_sizes, c=node_colors, zorder=4, edgecolors="white",
               linewidths=0.5, alpha=0.95)

    return norm, cmap


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Algorithms 1 & 2 side-by-side on circle data
# ══════════════════════════════════════════════════════════════════

def figure_alg1_vs_alg2():
    X = make_circle(n=150, r=1.0, noise=0.06)
    eps = 0.30

    c1, cv1 = greedy_epsilon_net(X, eps)
    G1 = build_bm_graph(c1, cv1)

    c2, cv2 = maxmin_epsilon_net(X, eps)
    G2 = build_bm_graph(c2, cv2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithms 1 & 2 — Cover strategies  |  circle point cloud  |  ε = 0.30",
                 color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    _draw_bm(axes[0], X, G1, eps,
             f"Algorithm 1 · Greedy ε-net\n|centers|={len(c1)}  |edges|={G1.number_of_edges()}")
    _draw_bm(axes[1], X, G2, eps,
             f"Algorithm 2 · Max-min ε-net\n|centers|={len(c2)}  |edges|={G2.number_of_edges()}")

    for ax in axes:
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)

    plt.tight_layout()
    plt.savefig("/home/claude/fig1_alg1_vs_alg2.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("Saved fig1_alg1_vs_alg2.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Algorithm 3: BM Graph on three datasets
# ══════════════════════════════════════════════════════════════════

def figure_alg3_datasets():
    datasets = [
        (make_circle(200),    0.28,  "circle",    (-1.3,1.3), (-1.3,1.3)),
        (make_yjunction(200), 0.12,  "y-junction",(-0.7,0.7), (-0.15,1.25)),
        (make_torus(300),     0.28,  "torus",     (-1.6,1.6), (-1.6,1.6)),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithm 3 — Ball Mapper Graph  |  three point clouds",
                 color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    for ax, (X, eps, name, xlim, ylim) in zip(axes, datasets):
        centers, cv = greedy_epsilon_net(X, eps)
        G = build_bm_graph(centers, cv)
        _draw_bm(ax, X, G, eps,
                 f"{name}  ·  ε={eps}\n|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

    plt.tight_layout()
    plt.savefig("/home/claude/fig2_alg3_datasets.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("Saved fig2_alg3_datasets.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Algorithm 4: Multi-scale BM on torus
# ══════════════════════════════════════════════════════════════════

def figure_alg4_multiscale():
    X = make_torus(n=300)
    epsilons = [0.18, 0.30, 0.48, 0.70]
    scales = multiscale_bm(X, epsilons, algo=1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Algorithm 4 — Multi-scale Ball Mapper  |  torus  |  fixed centers from ε₁=0.18",
                 color="#9de", fontsize=11, fontfamily="monospace", y=1.01)

    for ax, s in zip(axes, scales):
        G = s["graph"]
        _draw_bm(ax, X, G, s["eps"],
                 f"ε = {s['eps']:.2f}\n|V|={G.number_of_nodes()}  |E|={G.number_of_edges()}")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)

    plt.tight_layout()
    plt.savefig("/home/claude/fig3_alg4_multiscale.png", dpi=150, bbox_inches="tight",
                facecolor=BG)
    print("Saved fig3_alg4_multiscale.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Dimension estimation via avg neighbour count
# ══════════════════════════════════════════════════════════════════

def figure_dimension_estimation():
    """
    Recreates Figure 4 from the paper: average number of BM graph
    neighbours vs ε for point clouds sampled from [0,1]^d.
    """
    dims = [2, 3, 4, 5, 6]
    epsilons = np.linspace(0.05, 0.55, 20)
    n_pts = 500
    n_trials = 5   # paper uses 1000; we use 5 for speed

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_title("Dimension estimation via average BM neighbour count",
                 color="#9de", fontsize=11, fontfamily="monospace")
    ax.set_xlabel("ε", color="#aac8f0", fontfamily="monospace")
    ax.set_ylabel("avg neighbours per node", color="#aac8f0", fontfamily="monospace")
    ax.tick_params(colors="#567")

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
        ax.plot(epsilons, avg_neighbours, color=col, linewidth=2,
                label=f"d = {d}")

    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax.spines[:].set_color("#2a3a5a")
    plt.tight_layout()
    plt.savefig("/home/claude/fig4_dimension_estimation.png", dpi=150,
                bbox_inches="tight", facecolor=BG)
    print("Saved fig4_dimension_estimation.png")


# ══════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Figure 1: Algorithms 1 & 2 comparison...")
    figure_alg1_vs_alg2()

    print("Generating Figure 2: Algorithm 3 on multiple datasets...")
    figure_alg3_datasets()

    print("Generating Figure 3: Algorithm 4 multi-scale...")
    figure_alg4_multiscale()

    print("Generating Figure 4: Dimension estimation...")
    figure_dimension_estimation()

    print("\nAll figures saved.")
