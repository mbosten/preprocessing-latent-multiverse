from __future__ import annotations

import os

import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
from gudhi.representations import Landscape
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle


# sample data in the shape of a circle
# sample data in the shape of a circle
def sample_noisy_circle(
    n: int = 60,
    radius: float = 1.0,
    noise: float = 0.03,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, size=n)
    X = np.c_[radius * np.cos(angles), radius * np.sin(angles)]
    X += rng.normal(scale=noise, size=X.shape)
    return X


# build example alpha complex with gudhi
# build example alpha complex with gudhi
def build_alpha_simplex_tree(X: np.ndarray, max_dimension: int = 2) -> gd.SimplexTree:
    ac = gd.AlphaComplex(points=X)
    st = ac.create_simplex_tree()
    st.prune_above_dimension(max_dimension)
    return st


def compute_persistence(st: gd.SimplexTree, coeff_field: int = 2):

    persistence_pairs = st.persistence(
        homology_coeff_field=coeff_field, min_persistence=0.0
    )
    intervals = {
        0: st.persistence_intervals_in_dimension(0),
        1: st.persistence_intervals_in_dimension(1),
    }
    return persistence_pairs, intervals


def pick_alphas_from_h1(intervals: dict[int, np.ndarray], fallback=(0.005, 0.03, 0.12)):

    H1 = intervals.get(1, np.empty((0, 2)))
    if H1.size == 0:
        return fallback

    pers = H1[:, 1] - H1[:, 0]
    idx = int(np.argmax(pers))
    birth, death = float(H1[idx, 0]), float(H1[idx, 1])

    if not np.isfinite(death) or death <= birth:
        return fallback

    a_low = max(1e-12, 0.85 * birth)
    a_mid = 0.5 * (birth + death)
    a_high = 1.10 * death
    return a_low, a_mid, a_high


# Extract simplices present at alpha (for plotting specific scales)
# Extract simplices present at alpha (for plotting specific scales)
def simplices_at_alpha(st: gd.SimplexTree, alpha: float):

    edges, tris = [], []
    for simplex, filt in st.get_filtration():
        if filt > alpha:
            break
            break
        if len(simplex) == 2:
            i, j = simplex
            edges.append((i, j) if i < j else (j, i))
        elif len(simplex) == 3:
            i, j, k = sorted(simplex)
            tris.append((i, j, k))

    return sorted(set(edges)), sorted(set(tris))


def _finite_diagram(dgm: np.ndarray) -> np.ndarray:
    dgm = np.asarray(dgm, dtype=float)
    if dgm.ndim != 2 or dgm.shape[1] != 2 or dgm.size == 0:
        return np.empty((0, 2), dtype=float)
    return dgm[np.isfinite(dgm[:, 1])]


# Figure 1 - alpha complex panel
# Figure 1 - alpha complex panel
def plot_alpha_panel(
    ax,
    X: np.ndarray,
    st: gd.SimplexTree,
    alpha: float,
    title: str = "",
    draw_balls: bool = True,
    draw_triangles: bool = True,
):
    ax.scatter(X[:, 0], X[:, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    edges, tris = simplices_at_alpha(st, alpha)

    # draw 2-simplices first
    if draw_triangles and tris:
        polys = [X[list(t), :] for t in tris]
        ax.add_collection(PolyCollection(polys, alpha=0.18))

    # edges
    for i, j in edges:
        ax.plot(
            [X[i, 0], X[j, 0]],
            [X[i, 1], X[j, 1]],
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )

    # ball visualization
    # ball visualization
    if draw_balls:
        r = float(np.sqrt(max(alpha, 0.0)))
        for k in range(X.shape[0]):
            ax.add_patch(Circle((X[k, 0], X[k, 1]), r, fill=False, alpha=0.15))


# Figure 2 - barcode
# Figure 2 - barcode
def plot_barcode(
    ax, intervals: dict[int, np.ndarray], dims=(0, 1), cap_inf: float | None = None
):

    dim_to_color = {d: f"C{i}" for i, d in enumerate(dims)}

    finite_deaths = []
    for d in dims:
        D = intervals.get(d, np.empty((0, 2)))
        if D.size:
            D = np.asarray(D, dtype=float)
            finite = np.isfinite(D[:, 1])
            finite_deaths.extend(D[finite, 1].tolist())

    xmax = (max(finite_deaths) * 1.05) if finite_deaths else 1.0
    if cap_inf is not None:
        xmax = max(xmax, float(cap_inf))

    y = 0
    yticks, yticklabels = [], []

    for d in dims:
        D = intervals.get(d, np.empty((0, 2)))
        if D.size == 0:
            continue
        D = np.asarray(D, dtype=float)
        D = D[np.lexsort((D[:, 1], D[:, 0]))]  # sort by birth then death

        # Label row for each dimension
        yticks.append(y)
        yticklabels.append(f"H{d}")
        y += 1

        color = dim_to_color[d]
        D = D[-15:, :]  # plot at most 15 bars per dimension

        for birth, death in D:
            if np.isfinite(death):
                de_plot = float(death)
            else:
                de_plot = float(cap_inf) if cap_inf is not None else float(xmax)

            ax.hlines(y, float(birth), de_plot, colors=color, linewidth=3.0)
            y += 1

        y += 1  # gap between dimensions

    ax.set_xlabel("filtration value (alpha)", fontsize=16)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.5, y + 0.5)
    ax.set_xlim(0.0, xmax)
    ax.tick_params(axis="both", labelsize=14)

    # simple legend
    # simple legend
    handles = [
        plt.Line2D([0], [0], color=dim_to_color[d], lw=2.0, label=f"H{d}")
        for d in dims
        if intervals.get(d, np.empty((0, 2))).size > 0
    ]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=14)


# Figure 3 - persistence diagram
# Figure 3 - persistence diagram
def plot_persistence_diagram_gudhi(
    ax,
    persistence_pairs=None,
    intervals_by_dim=None,
    dims=(0, 1),
    show_diagonal=True,
):

    if persistence_pairs is None and intervals_by_dim is None:
        raise ValueError("Provide either persistence_pairs or intervals_by_dim.")

    # Collect points per dimension
    pts = {d: [] for d in dims}

    if intervals_by_dim is not None:
        for d in dims:
            D = intervals_by_dim.get(d, None)
            if D is None:
                continue
            D = np.asarray(D, dtype=float)
            if D.ndim == 2 and D.shape[1] == 2 and D.size > 0:
                for b, de in D:
                    pts[d].append((float(b), float(de)))
    else:
        for dim, (b, de) in persistence_pairs:
            if dim in pts:
                pts[dim].append((float(b), float(de)))

    # Flatten for axis limits
    # Flatten for axis limits
    all_births = []
    all_deaths = []
    for d in dims:
        if len(pts[d]) == 0:
            continue
        arr = np.array(pts[d], dtype=float)
        finite = np.isfinite(arr[:, 1])
        arr = arr[finite]
        if arr.size:
            all_births.extend(arr[:, 0].tolist())
            all_deaths.extend(arr[:, 1].tolist())

    if len(all_births) == 0:
        ax.text(0.5, 0.5, "No finite-death features to plot.", ha="center", va="center")
        ax.set_axis_off()
        return

    xmin = float(min(all_births))
    xmax = float(max(all_births + all_deaths))
    pad = 0.05 * (xmax - xmin + 1e-12)
    lo, hi = xmin - pad, xmax + pad

    if show_diagonal:
        ax.plot([lo, hi], [lo, hi])

    for d in dims:
        if len(pts[d]) == 0:
            continue
        arr = np.array(pts[d], dtype=float)

        # Drop inifite deaths
        # Drop inifite deaths
        finite = np.isfinite(arr[:, 1])
        arr = arr[finite]
        if arr.size == 0:
            continue

        ax.scatter(arr[:, 0], arr[:, 1], label=f"H{d}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("birth", fontsize=16)
    ax.set_ylabel("death", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)


# landscape plot function
# landscape plot function
def plot_landscape_gudhi(
    ax,
    dgm: np.ndarray,
    title: str,
    k_levels: int = 3,
    resolution: int = 600,
    sample_range=None,
):

    dgm_f = _finite_diagram(dgm)
    if dgm_f.size == 0:
        ax.text(0.5, 0.5, "No finite-death features to plot.", ha="center", va="center")
        ax.set_axis_off()
        return

    if sample_range is None:
        x_min = float(np.min(dgm_f[:, 0]))
        x_max = float(np.max(dgm_f[:, 1]))
        pad = 0.05 * (x_max - x_min + 1e-12)
        start, stop = x_min - pad, x_max + pad
    else:
        start, stop = map(float, sample_range)

    L = Landscape(
        num_landscapes=k_levels, resolution=resolution, sample_range=[start, stop]
    )
    vec = L.fit_transform([dgm_f])[0]  # shape: (k_levels * resolution,)
    vals = vec.reshape(k_levels, resolution)  # (k_levels, resolution)

    xs = np.linspace(start, stop, resolution)
    for k in range(k_levels):
        ax.plot(xs, vals[k], label=f"$\\lambda_{{{k+1}}}$")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("filtration value (alpha)", fontsize=16)
    ax.set_ylabel("landscape value", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="best", fontsize=14)


# Figure 4 - landscape
# Figure 4 - landscape
def plot_both_landscapes(
    intervals: dict,
    k_levels: int = 3,
    resolution: int = 800,
    shared_x_range: bool = False,
):

    h0 = intervals.get(0, np.empty((0, 2)))
    h1 = intervals.get(1, np.empty((0, 2)))

    h0f = _finite_diagram(h0)
    h1f = _finite_diagram(h1)

    sample_range = None
    if shared_x_range:
        all_f = (
            np.vstack([h0f, h1f])
            if (h0f.size and h1f.size)
            else (h0f if h0f.size else h1f)
        )
        if all_f.size:
            x_min = float(np.min(all_f[:, 0]))
            x_max = float(np.max(all_f[:, 1]))
            pad = 0.05 * (x_max - x_min + 1e-12)
            sample_range = (x_min - pad, x_max + pad)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)

    plot_landscape_gudhi(
        axes[0],
        h0,
        title="a) Persistence landscape (H0)",
        k_levels=k_levels + 2,
        resolution=resolution,
        sample_range=sample_range,
    )

    plot_landscape_gudhi(
        axes[1],
        h1,
        title="b) Persistence landscape (H1)",
        k_levels=k_levels,
        resolution=resolution,
        sample_range=sample_range,
    )

    return fig, axes


def plot_example_figures(
    out_dir: os.PathLike | str = "data/figures",
):
    # 1) Data
    X = sample_noisy_circle(n=60, radius=1.0, noise=0.03, seed=7)

    # 2) Alpha complex up to triangles
    st = build_alpha_simplex_tree(X, max_dimension=2)

    # 3) Persistence
    persistence_pairs, intervals = compute_persistence(st, coeff_field=2)

    # 4) three alpha values
    # 4) three alpha values
    a_low, a_mid, a_high = pick_alphas_from_h1(intervals)

    # 4.5) output directory
    # 4.5) output directory
    os.makedirs(out_dir, exist_ok=True)

    # Figure 1 - panel
    # Figure 1 - panel
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    plot_alpha_panel(
        axes[0],
        X,
        st,
        a_low,
        title=f"r≈{np.sqrt(a_low):.3f} ($\\alpha$={a_low:.4g})",
        draw_balls=True,
        draw_triangles=True,
    )
    plot_alpha_panel(
        axes[1],
        X,
        st,
        a_mid,
        title=f"r≈{np.sqrt(a_mid):.3f} ($\\alpha$={a_mid:.4g})",
        draw_balls=True,
        draw_triangles=True,
    )
    plot_alpha_panel(
        axes[2],
        X,
        st,
        a_high,
        title=f"r≈{np.sqrt(a_high):.3f} ($\\alpha$={a_high:.4g})",
        draw_balls=True,
        draw_triangles=True,
    )

    fig1.savefig(
        os.path.join(out_dir, "example_filtration_panel.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Figure 2 - barcode
    fig2 = plt.figure(figsize=(7, 4), constrained_layout=True)
    ax2 = fig2.add_subplot(1, 1, 1)
    plot_barcode(ax2, intervals, dims=(0, 1), cap_inf=None)

    fig2.savefig(
        os.path.join(out_dir, "example_barcode.png"), dpi=300, bbox_inches="tight"
    )

    # Figure 3 - persistence diagram
    # Figure 3 - persistence diagram
    fig3 = plt.figure(figsize=(6, 6), constrained_layout=True)
    ax3 = fig3.add_subplot(1, 1, 1)
    plot_persistence_diagram_gudhi(
        ax3,
        intervals_by_dim=intervals,
        dims=(0, 1),
    )
    fig3.savefig(
        os.path.join(out_dir, "example_persistence_diagram.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Figure 4 - persistence landscape
    # Figure 4 - persistence landscape
    fig4, axes4 = plot_both_landscapes(
        intervals, k_levels=3, resolution=1000, shared_x_range=False
    )
    fig4.savefig(
        os.path.join(out_dir, "example_landscapes.png"), dpi=300, bbox_inches="tight"
    )

    plt.show()
