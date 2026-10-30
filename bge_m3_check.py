"""Real embeddings: are bge-m3 vectors from Ollama already normalised (so dot product = cosine)?

Needs a local Ollama with `ollama pull bge-m3`. Standard library only.
Run:  uv run python bge_m3_check.py   → results_bge_m3.json
"""

import json
import math
import urllib.request
from pathlib import Path

TEXTS = [
    "Poistná udalosť: krupobitie poškodilo strechu rodinného domu.",
    "Krupobitie zničilo strechu domu, žiadam o náhradu škody.",
    "Recept na bryndzové halušky so slaninou.",
    "Hail damaged the roof of my house and I want to file a claim.",
]


def main() -> None:
    body = json.dumps({"model": "bge-m3", "input": TEXTS}).encode()
    req = urllib.request.Request("http://localhost:11434/api/embed", body, {"Content-Type": "application/json"})
    emb = json.loads(urllib.request.urlopen(req).read())["embeddings"]
    norms = [math.sqrt(sum(x * x for x in e)) for e in emb]
    dots = [[round(sum(a * b for a, b in zip(e, f)), 4) for f in emb] for e in emb]
    res = {
        "model": "bge-m3 via Ollama",
        "texts": TEXTS,
        "dim": len(emb[0]),
        "norms": [round(n, 4) for n in norms],
        "dot": dots,
    }
    print(json.dumps(res, indent=1, ensure_ascii=False))
    Path("results_bge_m3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
