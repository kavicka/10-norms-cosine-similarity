# Tutorial · 10 · Norms, cosine similarity & when cosine = dot product

🇸🇰 [Slovenská verzia](TUTORIAL.sk.md) · 📄 [README (results)](README.md)

This tutorial walks through [`cosine.py`](cosine.py) and [`bge_m3_check.py`](bge_m3_check.py) step by step: what each experiment computes, why it is designed that way, and what you should take into the retrieval projects that follow.

**Goal:** answer *"when is cosine similarity the same as the dot product?"* (when both vectors have length 1), and understand why it matters which one a search uses.

**You need:** project 03 (dot product, cosine). NumPy and scikit-learn; for the last step a local Ollama with `ollama pull bge-m3`.

---

## Step 1 · Three ways to measure length (`norms`)

```python
v = np.array([3.0, -4.0])
L1   = np.abs(v).sum()        # 7   "taxi distance"
L2   = np.linalg.norm(v)      # 5   straight-line length
Linf = np.abs(v).max()        # 4   largest coordinate
```

**What a norm is:** a rule that gives every vector a length. Different rules, different lengths for the same vector.

**How the "unit circles" are drawn:** take 400 directions around a circle and scale each so its norm is exactly 1 (`d / ‖d‖_p`). For L1 that traces a diamond, for L2 a circle, for L∞ a square (`assets/norms_high_dim.png`).

**Why it matters:** embeddings almost always use **L2**, and "normalise a vector" means divide by its L2 norm so it has length 1.

---

## Step 2 · High dimensions behave strangely (`high_dim`)

```python
for d in (2, 10, 100, 1000, 10000):
    X = rng.normal(size=(2000, d))
    norm = np.linalg.norm(X, axis=1)                    # lengths
    a, b = l2_normalize(X[:1000]), l2_normalize(X[1000:])
    cos = (a * b).sum(1)                                # cosine of 1000 random pairs
```

**What it measures and what you see:**
- **Lengths concentrate:** every random vector has length ≈ √d, and the spread shrinks (2.2 % at d = 1000). So in high dimensions, random vectors all have nearly the same length.
- **Random directions are nearly orthogonal:** the standard deviation of the cosine is ≈ 1/√d (0.0317 vs 0.0316 at d = 1000), and 99.9 % of random pairs have |cos| < 0.1.

**Why it matters for retrieval:**
- In a 1024-d embedding space, "unrelated" sits near cos 0, so a moderate cosine already carries signal.
- Big length differences between real embeddings are therefore not chance; they come from the model or the text. A raw dot product would let them dominate the ranking (Step 4).

**Why `(a * b).sum(1)` and not `a @ b.T`:** we want the cosine of 1000 *pairs* (row i with row i), not all 1000 × 1000 combinations.

---

## Step 3 · The identities, checked numerically (`identities`)

```
a · b = ‖a‖ ‖b‖ cos θ         →   cos θ = a·b / (‖a‖ ‖b‖)
if ‖a‖ = ‖b‖ = 1:   a·b = cos θ   and   ‖a − b‖² = 2 − 2 cos θ
```

The second line comes from expanding `‖a − b‖² = ‖a‖² + ‖b‖² − 2 a·b` with both norms equal to 1.

**So on unit vectors**, the largest dot product = the largest cosine = the smallest Euclidean distance. Three metrics, one ranking.

```python
D = rng.normal(size=(1000, 384)) * rng.uniform(0.2, 5, (1000, 1))   # random lengths on purpose
Q = rng.normal(size=(200, 384))
dot = Q @ D.T
cos = l2_normalize(Q) @ l2_normalize(D).T
```

**Why random lengths on purpose:** to show what happens when embeddings are *not* normalised. 384 dimensions is the size of a small sentence-embedding model.

**What is compared:**
- the identity `a·b = ‖a‖‖b‖cos θ`: max error 6.8e-13 (float rounding),
- normalised dot vs cosine: difference 0.0,
- normalised distance² vs `2 − 2cos`: 1.3e-15,
- **top-10 neighbours** for each of the 200 queries: cosine vs normalised dot and vs normalised L2 are identical for 100 % of queries; cosine vs **raw** dot for **0 %** (only 39 % overlap).

**Why the raw dot fails:** its top-10 documents have a mean norm of 84.9 vs 50.4 overall. It rewards *length*, not only direction.

---

## Step 4 · See the effect on Slovak text (`retrieval`)

```python
vec = CountVectorizer(analyzer="char_wb", ngram_range=(3, 5))
D = vec.fit_transform(DOCS).toarray()
q = vec.transform([QUERY]).toarray()[0]
dot = D @ q
cos = dot / (np.linalg.norm(D, axis=1) * np.linalg.norm(q))
```

**The corpus:** five short insurance claims plus one long, generic policy text that repeats words like *poškodenie*, *strecha*, *dom*. Query: „krupobitie poškodilo strechu domu“.

**Why character 3–5-grams (`char_wb`):** Slovak inflects words (*strecha / strechu / strechy*). Whole-word counts would treat them as different words; character pieces like `str`, `stre`, `rech` are shared, so the match survives. `char_wb` takes n-grams only inside word boundaries.

**Result:**
- **Raw dot:** the long policy text wins (296 vs 76), simply because it has more of everything (norm 72 vs ~11).
- **Cosine:** divides the length out and ranks the actual hail-on-a-roof claim first (0.782 vs 0.464).

**Why it matters later:** this is exactly why TF-IDF in scikit-learn L2-normalises by default, and why BM25 (project 14 and beyond) has a document-length penalty.

---

## Step 5 · Check a real embedding model ([`bge_m3_check.py`](bge_m3_check.py))

```python
body = json.dumps({"model": "bge-m3", "input": TEXTS}).encode()
req = urllib.request.Request("http://localhost:11434/api/embed", body, {"Content-Type": "application/json"})
emb = json.loads(urllib.request.urlopen(req).read())["embeddings"]
norms = [math.sqrt(sum(x * x for x in e)) for e in emb]
```

**Why standard library only:** one HTTP request to the local Ollama server, no extra dependency.

**Why these four texts:** two Slovak hail claims (should be close), one English hail claim (tests multilingual matching), and a halušky recipe (unrelated).

**Result:**
- 1024 dimensions, **every norm = 1.0000**. bge-m3 normalises its output, so dot = cosine and a fast inner-product index is safe.
- SK–SK hail 0.81, SK–EN hail 0.66–0.77, recipe 0.34–0.39.

**Note the floor:** "unrelated" is ~0.35, not 0 like random vectors in Step 2. Real embeddings share a common direction (they are not isotropic). So judge scores **relative to each other**, never with an absolute cut-off chosen by feel.

---

## Step 6 · Save and plot

```bash
uv run python cosine.py        # → results.json
uv run python bge_m3_check.py  # → results_bge_m3.json (Mac with Ollama)
uv run python plot.py          # → assets/*.png, cosine_vs_dot.gif
```

The GIF shows a query vector `a` and a document `b`: making `b` longer pushes the dot product up while the cosine stays still; turning `b` moves both.

**Why no Racing Lab chapter:** the question "which document is closest to a query?" belongs to text, not cars.

---

## Try it yourself

1. Replace `CountVectorizer` with `TfidfVectorizer` (which L2-normalises by default). Does the raw dot product now rank like cosine?
2. In `identities`, set all lengths to 1 before computing `dot`. Check that the 0 % becomes 100 %.
3. Add a fifth text to `bge_m3_check.py` that is about hail on a car. Where does it land between the roof claims and the recipe?

## Recap

| Step | Idea |
|---|---|
| 1 | Norm = length; embeddings use L2; normalising = divide by L2 norm |
| 2 | High-d: lengths concentrate at √d, random pairs have cos ≈ 0 |
| 3 | On unit vectors dot = cosine and distance gives the same ranking |
| 4 | Without normalisation the dot product rewards long documents |
| 5 | Check your model: bge-m3 is already normalised; real "unrelated" ≈ 0.35, not 0 |
