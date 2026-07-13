"""V4 — Can a DeepMoji probe replace the word lexicon?

V2 showed the sentence-level numbers were bottlenecked by
``word_emotion_lexicon.csv``, which maps *miserable* to **anger**. The obvious
replacement is already a dependency: DeepMoji is MIT-licensed, trained on 1.2B
tweets, and V3 showed it decodes the core axes.

So: train a 64->5 linear probe on GoEmotions (Apache-2.0), and compare it against
the bag-of-words lexicon on **held-out EmoBank**.

The answer is a **split decision**, which is why the probe is not simply swapped
in:

===============  ==================  ==================
axis             bag-of-words        DeepMoji probe
===============  ==================  ==================
valence          +0.331              **+0.466**
arousal          **+0.129**          +0.028
===============  ==================  ==================

The probe is much better at valence and **worse at arousal** — near zero. Emoji
usage evidently carries hedonic tone far more than activation, which is not
surprising in hindsight but was not predicted.

Neither path is good enough to be the default, and shipping the probe as a
straight upgrade would be a misrepresentation. Arousal from text remains an open
problem here.

Run:
    python scripts/validate/v4_deepmoji_vs_lexicon.py \\
        --goemotions train.tsv --emotions emotions.txt --emobank emobank.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z']+")

LABEL_MAP = {
    "anger": "anger", "annoyance": "annoyance", "fear": "fear",
    "nervousness": "apprehension", "sadness": "sadness", "grief": "grief",
    "joy": "joy", "excitement": "ecstasy", "admiration": "admiration",
    "disgust": "disgust", "surprise": "surprise", "curiosity": "interest",
    "confusion": "distraction", "embarrassment": "shame",
    "disappointment": "pensiveness", "approval": "acceptance",
    "caring": "trust", "desire": "anticipation",
}


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def encode(model, texts, batch=512):
    return np.vstack(
        [np.asarray(model.encode(texts[i:i + batch]), dtype=float)
         for i in range(0, len(texts), batch)]
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goemotions", required=True, type=Path)
    ap.add_argument("--emotions", required=True, type=Path)
    ap.add_argument("--emobank", required=True, type=Path)
    ap.add_argument("--per-label", type=int, default=400)
    args = ap.parse_args()

    import pandas as pd
    from deepmoji_onnx import DeepMojiONNX

    from emotion_algebra.affect import mixture
    from emotion_algebra.lexicons import LEXICON, load_lexicon
    from emotion_algebra.prototypes import PROTOTYPES, prototype

    # --- train the probe on GoEmotions ---
    names = args.emotions.read_text().split()
    go = pd.read_csv(args.goemotions, sep="\t", header=None,
                     names=["text", "labels", "id"])
    go = go[~go["labels"].astype(str).str.contains(",")]
    go["label"] = go["labels"].astype(int).map(lambda i: names[i])
    go = go[go["label"].isin(LABEL_MAP)]
    go = pd.concat([g.sample(min(len(g), args.per_label), random_state=0)
                    for _, g in go.groupby("label")])

    model = DeepMojiONNX.from_pretrained()
    X = encode(model, [str(t) for t in go["text"]])
    protos = [prototype(LABEL_MAP[l]) for l in go["label"]]
    Y = np.array([[p.positivity, p.negativity, p.potency, p.arousal,
                   p.unpredictability] for p in protos])

    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + np.eye(X1.shape[1])
    A[-1, -1] -= 1.0
    W = np.linalg.solve(A, X1.T @ Y)

    # --- evaluate both paths on held-out EmoBank ---
    eb = pd.read_csv(args.emobank)
    keep = [(str(t).strip(), v, a) for t, v, a in zip(eb["text"], eb["V"], eb["A"])
            if isinstance(t, str) and str(t).strip()]
    texts = [k[0] for k in keep]
    gold_v = np.array([(k[1] - 3.0) / 2.0 for k in keep])
    gold_a = np.array([(k[2] - 3.0) / 2.0 for k in keep])

    Xe = encode(model, texts)
    Pe = np.hstack([Xe, np.ones((len(Xe), 1))]) @ W
    probe_v = np.clip(Pe[:, 0], 0, 1) - np.clip(Pe[:, 1], 0, 1)
    probe_a = np.clip(Pe[:, 3], 0, 1)

    load_lexicon()

    def bow(text):
        hits = [prototype(label)
                for tok in _TOKEN.findall(text.lower())
                if (label := LEXICON.get(tok, {}).get("emotion")) in PROTOTYPES]
        return mixture(hits) if hits else None

    lex_idx, lex_v, lex_a = [], [], []
    for i, t in enumerate(texts):
        s = bow(t)
        if s is not None:
            lex_idx.append(i)
            lex_v.append(s.valence)
            lex_a.append(s.arousal)
    lex_idx = np.array(lex_idx)

    print("V4 — DeepMoji probe vs the word lexicon, on held-out EmoBank\n")
    print(f"    probe trained on {len(go)} GoEmotions comments ({W.size} floats)")
    print(f"    evaluated on {len(texts)} EmoBank sentences "
          f"({len(lex_idx)} have lexicon coverage)\n")

    print(f"{'axis':<10}{'bag-of-words':>16}{'DeepMoji probe':>18}")
    print("-" * 44)
    print(f"{'valence':<10}{pearson(gold_v[lex_idx], lex_v):>16.3f}"
          f"{pearson(gold_v, probe_v):>18.3f}")
    print(f"{'arousal':<10}{pearson(gold_a[lex_idx], lex_a):>16.3f}"
          f"{pearson(gold_a, probe_a):>18.3f}")

    print("\nA SPLIT DECISION, and the reason the probe is not simply swapped in:")
    print("the probe is much better at valence and WORSE at arousal — near zero.")
    print("Emoji usage carries hedonic tone far more than activation. Neither path")
    print("is good enough to be the default; arousal from text is unsolved here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
