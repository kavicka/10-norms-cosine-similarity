"""Step 10 · Norms, cosine similarity and vector spaces: when is cosine the same as the dot product?

Four experiments, NumPy + scikit-learn, fixed seed (every number in the README comes from results.json):
1. Norms: L1, L2 and L∞ of one vector, and what their "unit circles" look like.
2. High dimensions: random vectors' lengths concentrate around √d and random pairs become nearly orthogonal.
3. The identities: a·b = ‖a‖‖b‖ cos θ; after L2-normalisation dot = cosine and ‖a − b‖² = 2 − 2 cos θ,
   so dot, cosine and Euclidean distance rank neighbours identically.
4. Retrieval on a small Slovak corpus: ranking by raw dot product favours a long document; cosine does not.

Run:  uv run python cosine.py   → results.json
      uv run python plot.py     → assets/*.png, assets/cosine_vs_dot.gif
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

rng = np.random.default_rng(0)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


# --- 1. Norms ---------------------------------------------------------------------------------
def norms() -> dict:
    v = np.array([3.0, -4.0])
    t = np.linspace(0, 2 * np.pi, 400)
    d = np.stack([np.cos(t), np.sin(t)], 1)  # directions; scale each one to norm 1 under every p
    balls = {}
    for name, p in [("L1", 1), ("L2", 2), ("Linf", np.inf)]:
        r = 1 / np.linalg.norm(d, ord=p, axis=1)
        balls[name] = (d * r[:, None]).T.tolist()
    return {
        "v": v.tolist(),
        "L1": float(np.abs(v).sum()),
        "L2": float(np.linalg.norm(v)),
        "Linf": float(np.abs(v).max()),
        "balls": balls,
    }


# --- 2. High dimensions -----------------------------------------------------------------------------
def high_dim(dims=(2, 10, 100, 1000, 10000), n: int = 2000) -> list[dict]:
    rows = []
    for d in dims:
        X = rng.normal(size=(n, d))
        norm = np.linalg.norm(X, axis=1)
        a, b = l2_normalize(X[: n // 2]), l2_normalize(X[n // 2 :])
        cos = (a * b).sum(1)
        hist, edges = np.histogram(cos, bins=40, range=(-1, 1))
        rows.append(
            {
                "d": d,
                "norm_mean_over_sqrt_d": float(norm.mean() / np.sqrt(d)),
                "norm_cv": float(norm.std() / norm.mean()),
                "cos_std": float(cos.std()),
                "cos_std_theory": float(1 / np.sqrt(d)),
                "share_abs_cos_below_0.1": float((np.abs(cos) < 0.1).mean()),
                "cos_hist": hist.tolist(),
                "cos_edges": edges.tolist(),
            }
        )
    return rows


# --- 3. Identities ---------------------------------------------------------------------------------------
def identities(d: int = 384, docs: int = 1000, queries: int = 200) -> dict:
    """384 = the size of a small sentence-embedding model. Vectors get random lengths on purpose."""
    D = rng.normal(size=(docs, d)) * rng.uniform(0.2, 5, (docs, 1))
    Q = rng.normal(size=(queries, d))
    dot = Q @ D.T
    cos = l2_normalize(Q) @ l2_normalize(D).T
    law = np.abs(dot - np.linalg.norm(Q, axis=1)[:, None] * np.linalg.norm(D, axis=1)[None] * cos).max()
    Dn, Qn = l2_normalize(D), l2_normalize(Q)
    dot_n = Qn @ Dn.T
    dist2 = ((Qn[:, None, :] - Dn[None]) ** 2).sum(-1)

    def top10(S, largest=True):
        return np.argsort(-S if largest else S, axis=1)[:, :10]

    same = lambda a, b: float((a == b).all(1).mean())
    t_cos = top10(cos)
    overlap_raw = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top10(dot), t_cos)]))
    return {
        "d": d,
        "docs": docs,
        "queries": queries,
        "max_abs_dot_minus_norms_times_cos": float(law),
        "max_abs_dot_normalized_minus_cos": float(np.abs(dot_n - cos).max()),
        "max_abs_dist2_minus_2_minus_2cos": float(np.abs(dist2 - (2 - 2 * cos)).max()),
        "top10_identical_cos_vs_dot_normalized": same(t_cos, top10(dot_n)),
        "top10_identical_cos_vs_euclid_normalized": same(t_cos, top10(dist2, largest=False)),
        "top10_identical_cos_vs_raw_dot": same(t_cos, top10(dot)),
        "top10_overlap_cos_vs_raw_dot": overlap_raw,
        "doc_norm_range": [float(np.linalg.norm(D, axis=1).min()), float(np.linalg.norm(D, axis=1).max())],
        "mean_norm_of_raw_dot_top10": float(np.linalg.norm(D[top10(dot)], axis=-1).mean()),
        "mean_norm_all_docs": float(np.linalg.norm(D, axis=1).mean()),
    }


# --- 4. Retrieval ---------------------------------------------------------------------------------------
DOCS = [
    "Krupobitie poškodilo strechu nášho rodinného domu.",
    "Víchrica strhla strechu garáže a poškodila plot.",
    "Voda z práčky vytopila kúpeľňu aj chodbu.",
    "Krupobitie poškodilo auto zaparkované pred domom.",
    "Pri požiari kuchyne zhorela linka a zadymila sa obývačka.",
    # a long, generic policy text: it repeats many query-like word pieces, but is about nothing in particular
    (
        "Všeobecné poistné podmienky poistenia domu. Poistenie domu kryje poškodenie domu, poškodenie strechy, "
        "poškodenie okien, poškodenie fasády a poškodenie plotu. Poistenie strechy domu kryje poškodenie strechy "
        "pri víchrici aj poškodenie strechy pri krupobití. Poškodenie domu nahláste do troch dní. Poškodenie strechy "
        "doložte fotografiou strechy. Poistenie domu sa nevzťahuje na poškodenie domu spôsobené zanedbaním údržby domu."
    ),
]
QUERY = "krupobitie poškodilo strechu domu"


def retrieval() -> dict:
    # character 3–5-grams inside words: Slovak inflection (strecha / strechu / strechy) still shares pieces
    vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    D = vec.fit_transform(DOCS).toarray().astype(float)
    q = vec.transform([QUERY]).toarray().astype(float)[0]
    dot = D @ q
    cos = dot / (np.linalg.norm(D, axis=1) * np.linalg.norm(q))
    return {
        "query": QUERY,
        "docs": DOCS,
        "n_features": int(D.shape[1]),
        "doc_lengths_chars": [len(d) for d in DOCS],
        "doc_norms": np.linalg.norm(D, axis=1).tolist(),
        "dot": dot.tolist(),
        "cos": cos.tolist(),
        "rank_by_dot": np.argsort(-dot).tolist(),
        "rank_by_cos": np.argsort(-cos).tolist(),
    }


def main() -> None:
    res = {"seed": 0, "norms": norms(), "high_dim": high_dim(), "identities": identities(), "retrieval": retrieval()}
    n = res["norms"]
    print(f"v = {n['v']}: L1 {n['L1']}  L2 {n['L2']}  Linf {n['Linf']}")
    for r in res["high_dim"]:
        print(
            f"d={r['d']:6d}: |x|/√d {r['norm_mean_over_sqrt_d']:.3f}  CV {r['norm_cv']:.3f}  cos std {r['cos_std']:.4f} "
            f"(1/√d {r['cos_std_theory']:.4f})  |cos|<0.1: {r['share_abs_cos_below_0.1']:.1%}"
        )
    print(json.dumps({k: v for k, v in res["identities"].items()}, indent=1))
    r = res["retrieval"]
    for i in range(len(DOCS)):
        print(
            f"doc {i} ({r['doc_lengths_chars'][i]:3d} chars, ‖d‖ {r['doc_norms'][i]:6.1f}): dot {r['dot'][i]:6.1f}  cos {r['cos'][i]:.3f}"
        )
    print("rank by dot:", r["rank_by_dot"], " rank by cos:", r["rank_by_cos"])
    Path("results.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
