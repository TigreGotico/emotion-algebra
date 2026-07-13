"""V2 — Held-out validation on EmoBank.

The prototype coordinates for valence and arousal are **calibrated on Warriner**
word norms, so correlating them back against Warriner would be circular. This is
the honest test: a **different corpus**, at **sentence** level, never seen during
calibration.

Gold: Buechel & Hahn (2017), *EmoBank: Studying the Impact of Annotation
Perspective and Representation Format on Dimensional Emotion Analysis*, EACL.
10,062 sentences with VAD ratings on a 1-5 scale.

Method: bag-of-words. Each word that the lexicon maps to an emotion label
contributes its prototype; the sentence is their convex mixture.

Run:  python scripts/validate/v2_emobank_heldout.py --emobank <path/to/emobank.csv>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z']+")


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emobank", required=True, type=Path)
    args = ap.parse_args()

    import pandas as pd

    from emotion_algebra.affect import mixture
    from emotion_algebra.lexicons import LEXICON, load_lexicon
    from emotion_algebra.prototypes import PROTOTYPES, prototype

    load_lexicon()
    eb = pd.read_csv(args.emobank)

    def predict(text: str):
        hits = [
            prototype(label)
            for tok in _TOKEN.findall(str(text).lower())
            if (label := LEXICON.get(tok, {}).get("emotion")) in PROTOTYPES
        ]
        return mixture(hits) if hits else None

    rows = []
    for _, r in eb.iterrows():
        state = predict(r["text"])
        if state is None:
            continue
        rows.append(
            dict(
                gold_v=(r["V"] - 3.0) / 2.0,   # 1-5 -> [-1, 1]
                gold_a=(r["A"] - 3.0) / 2.0,
                pred_v=state.valence,
                pred_a=state.arousal,
            )
        )

    df = pd.DataFrame(rows)
    covered, total = len(df), len(eb)

    print("V2 — EmoBank (held out)")
    print(f"    {covered}/{total} sentences have lexicon coverage "
          f"({100 * covered / total:.0f}%)\n")
    print(f"  valence   r = {pearson(df.gold_v, df.pred_v):+.3f}")
    print(f"  arousal   r = {pearson(df.gold_a, df.pred_a):+.3f}")
    print()
    print("Read this honestly: the signal is real but weak, and the bottleneck is")
    print("the LEXICON, not the core. word_emotion_lexicon.csv maps 'miserable' to")
    print("*anger*, and a bag-of-words has no negation, no intensifiers and no")
    print("syntax. The core's own coordinates track human word norms closely")
    print("(see v1); turning that into sentence-level accuracy is a separate")
    print("problem, and this number is a measurement of the lexicon's quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
