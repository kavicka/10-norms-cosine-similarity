"""Turn results.json (+ results_bge_m3.json) into the plots and the GIF used in the README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

R = json.loads(Path("results.json").read_text())
B = json.loads(Path("results_bge_m3.json").read_text())
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
DIM_COLORS = ["#cbd5e1", "#93c5fd", "#3b82f6", "#1e3a8a"]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def norms_and_dims() -> None:
    n = R["norms"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    for (k, lab), col in zip(
        [("L1", "L1: |x| + |y| = 1"), ("L2", "L2: √(x² + y²) = 1"), ("Linf", "L∞: max(|x|, |y|) = 1")],
        [ORANGE, BLUE, GREEN],
    ):
        pts = np.array(n["balls"][k])
        a1.plot(pts[0], pts[1], color=col, lw=2, label=lab)
    a1.set(aspect="equal", xlim=(-1.4, 1.4), ylim=(-2.0, 1.4), title="All vectors of length 1, by norm")
    a1.legend(fontsize=8, loc="lower center")
    a1.grid(alpha=0.3)

    rows = [r for r in R["high_dim"] if r["d"] in (2, 10, 100, 1000)]
    for r, col in zip(rows, DIM_COLORS):
        e = np.array(r["cos_edges"])
        h = np.array(r["cos_hist"], float)
        a2.plot((e[:-1] + e[1:]) / 2, h / h.sum(), color=col, lw=2, label=f"d = {r['d']}: std {r['cos_std']:.3f}")
    a2.set(
        xlabel="cosine between two random vectors",
        ylabel="share of pairs",
        title="In high dimensions random vectors are nearly orthogonal",
    )
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)

    hd = R["high_dim"]
    a3.loglog(
        [r["d"] for r in hd], [r["cos_std"] for r in hd], "o", color=BLUE, ms=7, label="std of cosine (simulated)"
    )
    a3.loglog([r["d"] for r in hd], [r["cos_std_theory"] for r in hd], "-", color=GRAY, lw=2, label="1/√d")
    a3.loglog(
        [r["d"] for r in hd], [r["norm_cv"] for r in hd], "s", color=ORANGE, ms=6, label="spread of ‖x‖ / mean ‖x‖"
    )
    a3.set(xlabel="dimension d", title="Both shrink like 1/√d")
    a3.legend(fontsize=8)
    a3.grid(alpha=0.3, which="both")
    save(fig, "norms_high_dim.png")


def retrieval() -> None:
    r = R["retrieval"]
    labels = [f"doc {i}: {d[:38]}…" if len(d) > 40 else f"doc {i}: {d}" for i, d in enumerate(r["docs"])]
    y = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    dot, cos = np.array(r["dot"]), np.array(r["cos"])
    a1.barh(y, dot, color=[RED if i == int(np.argmax(dot)) else GRAY for i in y])
    a1.set(xlabel="dot product with the query (raw counts)", title="Raw dot product: the long document wins")
    a2.barh(y, cos, color=[BLUE if i == int(np.argmax(cos)) else GRAY for i in y])
    a2.set(xlabel="cosine similarity", title="Cosine: the short, on-topic document wins")
    a1.set_yticks(y, labels, fontsize=8)
    a1.invert_yaxis()
    for a in (a1, a2):
        a.grid(alpha=0.3, axis="x")
    fig.suptitle(f"Query: „{r['query']}“ (character 3–5-grams)", fontsize=11)
    save(fig, "retrieval.png")


def bge() -> None:
    M = np.array(B["dot"])
    short = ["SK: krupobitie, strecha (1)", "SK: krupobitie, strecha (2)", "SK: halušky", "EN: hail, roof"]
    fig, a = plt.subplots(figsize=(6.4, 5))
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            a.text(
                j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="white" if M[i, j] > 0.6 else DARK, fontsize=10
            )
    a.set_xticks(range(4), short, rotation=25, ha="right", fontsize=8)
    a.set_yticks(range(4), short, fontsize=8)
    a.set_title(f"bge-m3 (Ollama): all norms = {B['norms'][0]:.4f}, so dot = cosine", fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046)
    save(fig, "bge_m3.png")


def gif() -> None:
    """Fixed a; b first grows at a fixed angle (dot grows, cosine stays), then turns at a fixed length."""
    a = np.array([np.cos(np.radians(20)), np.sin(np.radians(20))]) * 2
    frames = [(L, 70.0) for L in np.linspace(0.5, 3, 20)] + [(3.0, ang) for ang in np.linspace(70, 200, 30)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax.set(xlim=(-3.3, 3.3), ylim=(-3.3, 3.3), aspect="equal")
    ax.grid(alpha=0.3)
    ax.quiver(0, 0, *a, angles="xy", scale_units="xy", scale=1, color=BLUE, width=0.012)
    ax.text(*(a * 1.08), "a", color=BLUE, fontsize=12)
    qb = ax.quiver(0, 0, 1, 1, angles="xy", scale_units="xy", scale=1, color=RED, width=0.012)
    lb = ax.text(0, 0, "b", color=RED, fontsize=12)
    bars = bx.bar(["a · b", "cos θ", "a·b of unit vectors"], [0, 0, 0], color=[GRAY, BLUE, GREEN])
    bx.axhline(0, color=DARK, lw=0.8)
    bx.set(ylim=(-6.5, 6.5), title="dot product vs cosine")
    txt = bx.text(1, 4.6, "", ha="center", fontsize=10)

    def frame(i):
        L, ang = frames[min(i, len(frames) - 1)]
        b = L * np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
        qb.set_UVC(*b)
        lb.set_position(b * 1.08)
        dot = float(a @ b)
        cos = dot / (np.linalg.norm(a) * np.linalg.norm(b))
        for bar, v in zip(bars, [dot, cos, cos]):
            bar.set_height(v)
        phase = "b gets longer, angle fixed" if i < 20 else "b turns, length fixed"
        ax.set_title(phase, fontsize=10)
        txt.set_text(f"‖b‖ = {L:.2f}, θ = {ang - 20:.0f}°\na·b = {dot:.2f}, cos = {cos:.2f}")
        return [qb, lb, *bars, txt]

    fig.tight_layout()
    FuncAnimation(fig, frame, frames=len(frames) + 8).save(
        "assets/cosine_vs_dot.gif", writer=PillowWriter(fps=8), dpi=75
    )


if __name__ == "__main__":
    norms_and_dims()
    retrieval()
    bge()
    gif()
