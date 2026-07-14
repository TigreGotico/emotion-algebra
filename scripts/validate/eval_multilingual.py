"""Does the model survive the crossing into Portuguese and Arabic?

The probe was fitted on English and has never seen a word of either language. The
bet is that ``paraphrase-multilingual-MiniLM-L12-v2``, distilled so a sentence and
its translation land in the same place, carries the English mapping across for
free.

That is a **claim**, and this is where it is put at risk.

The crux
--------
This library exists for one distinction. Anger and fear are both unpleasant and
both highly aroused; valence and arousal cannot separate them, and the whole
design rests on a third axis — **potency**, appraised coping — that can. The angry
customer escalates. The frightened one leaves without a word.

So the question for a new language is not "does the probe produce numbers" (it
always will) but:

**1. Does it still separate anger from fear — and does it do it on POTENCY?**

If Arabic anger and Arabic fear land on top of each other, or separate on some
other axis, then the model does not transfer and this library should say so
rather than ship a language it cannot read.

**2. The permutation control.** Shuffle the labels and refit. If "separation"
survives randomised labels, the result above is an artefact of the method and
means nothing.

**3. English cross-check.** The same multilingual probe against DeepMoji on
English. English is the only language with real gold behind it, so if the
multilingual encoder is far worse *there*, that bounds how much to believe its
Portuguese and Arabic numbers — they cannot be better than the pipeline that
produced them.

The gold, and what is wrong with it
-----------------------------------
**XED** (Öhman et al. 2020, COLING; CC-BY-4.0) — OpenSubtitles lines with
Plutchik labels. For English and Finnish these were annotated by humans. For every
other language, including both of ours, they are **projected** across sentence
alignments — so the Portuguese and Arabic labels are machine-derived, not human
judgements, and the Portuguese is **Brazilian**.

This is weak gold and it is reported as weak gold. It is also the best that
exists: there is no human-rated Arabic affect corpus with an open licence, and no
Arabic VAD lexicon at all. A weak test that can fail is worth more than a strong
claim that was never tested.

Run:
    python scripts/validate/eval_multilingual.py --xed data/ --emobank data/emobank.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from emotion_algebra.multilingual import affect_from_texts, encode
from emotion_algebra.prototypes import prototype

#: XED's label key, from its README: ascending alphabetical order.
XED_LABELS = {
    1: "anger", 2: "anticipation", 3: "disgust", 4: "fear",
    5: "joy", 6: "sadness", 7: "surprise", 8: "trust",
}

AXES = ("valence", "potency", "arousal", "unpredictability")


def load_xed(path: Path) -> Dict[str, List[str]]:
    """Single-label lines only, grouped by emotion.

    A line tagged both 'anger' and 'joy' tells us nothing clean about either.
    """
    out: Dict[str, List[str]] = {}
    with path.open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 2:
                continue
            labels = [int(x) for x in row[1].split(",") if x.strip().isdigit()]
            if len(labels) == 1 and labels[0] in XED_LABELS:
                out.setdefault(XED_LABELS[labels[0]], []).append(row[0])
    return out


def crux(states, is_anger: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Which axis separates anger from fear here, and by how much?

    Reported as the standardised difference between the two groups' means
    (Cohen's d) on each axis — a scale-free number, so the axes can be compared
    with each other and across languages.
    """
    columns = {
        "valence": np.array([s.valence for s in states]),
        "potency": np.array([s.potency for s in states]),
        "arousal": np.array([s.arousal for s in states]),
        "unpredictability": np.array([s.unpredictability for s in states]),
    }
    effects = {}
    for axis, values in columns.items():
        a, f = values[is_anger], values[~is_anger]
        pooled = np.sqrt((a.var() + f.var()) / 2)
        effects[axis] = float((a.mean() - f.mean()) / pooled) if pooled > 1e-9 else 0.0
    winner = max(effects, key=lambda k: abs(effects[k]))
    return winner, effects


def separability(X: np.ndarray, y: np.ndarray, rng, lam=1.0) -> Tuple[float, float, float]:
    """Held-out accuracy of a linear anger/fear classifier, and its null.

    Returns ``(accuracy, majority_baseline, permutation_control)``.
    """
    idx = rng.permutation(len(X))
    split = len(idx) // 2
    tr, te = idx[:split], idx[split:]

    def fit(features, target):
        A = features.T @ features + lam * np.eye(features.shape[1])
        return np.linalg.solve(A, features.T @ target)

    X1 = np.hstack([X, np.ones((len(X), 1))])
    w = fit(X1[tr], y[tr].astype(float))
    acc = float(((X1[te] @ w > 0.5) == y[te]).mean())
    base = float(max(y[te].mean(), 1 - y[te].mean()))

    shuffled = y[tr].astype(float).copy()
    rng.shuffle(shuffled)
    w_null = fit(X1[tr], shuffled)
    null = float(((X1[te] @ w_null > 0.5) == y[te]).mean())
    return acc, base, null


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate(lang: str, xed_dir: Path, cap: int, rng) -> dict:
    path = xed_dir / f"xed-{lang}.tsv"
    if not path.exists():
        print(f"  {lang}: no {path.name}, skipped")
        return {}

    by_label = load_xed(path)
    anger = by_label.get("anger", [])[:cap]
    fear = by_label.get("fear", [])[:cap]
    if len(anger) < 50 or len(fear) < 50:
        print(f"  {lang}: too few examples (anger={len(anger)}, fear={len(fear)})")
        return {}

    texts = anger + fear
    is_anger = np.array([True] * len(anger) + [False] * len(fear))

    states = affect_from_texts(texts, lang=lang)
    winner, effects = crux(states, is_anger)

    acc, base, null = separability(encode(texts), is_anger, rng)

    print(f"\n  {lang.upper()}  (anger={len(anger)}, fear={len(fear)})")
    print(f"    separable at all?   held-out accuracy {acc:.3f}  "
          f"(majority {base:.3f}, shuffled-label control {null:.3f})")
    print(f"    on WHICH axis?      (Cohen's d, anger minus fear)")
    for axis in AXES:
        mark = "   <-- the claim" if axis == "potency" else ""
        print(f"      {axis:<18} d = {effects[axis]:+.3f}{mark}")

    if winner == "potency" and effects["potency"] > 0:
        print(f"    -> POTENCY separates them, and anger is the high-potency side.")
        print(f"       The distinction the library is built on survives the crossing.")
    elif abs(effects["potency"]) < 0.2:
        print(f"    -> potency does NOT separate them here (|d| < 0.2). The model")
        print(f"       does not transfer to {lang}, and must be documented as such.")
    else:
        print(f"    -> {winner} dominates, not potency. Transfer is partial: report it.")

    return {
        "lang": lang, "n_anger": len(anger), "n_fear": len(fear),
        "accuracy": acc, "baseline": base, "control": null,
        "effects": effects, "winner": winner,
    }


def english_cross_check(emobank: Path, rng) -> None:
    """The multilingual encoder against DeepMoji, on the one language with gold.

    English is the only language where real human valence and arousal ratings
    exist. If the multilingual encoder is much worse here, that is a ceiling on
    how much its Portuguese and Arabic numbers can be believed — they cannot be
    better than the pipeline that produced them.
    """
    import pandas as pd

    df = pd.read_csv(emobank)
    df = df[df["split"] == "test"]
    texts = df["text"].astype(str).tolist()
    gold_v = ((df["V"].astype(float) - 3.0) / 2.0).to_numpy()
    gold_a = ((df["A"].astype(float) - 1.0) / 4.0).to_numpy()

    print("\n3. English cross-check — EmoBank held-out, human ratings")
    print(f"   ({len(texts)} sentences)\n")
    print(f"   {'encoder':<28} {'valence r':>10} {'arousal r':>10}")

    ml = affect_from_texts(texts, lang="en")
    print(f"   {'multilingual-MiniLM':<28} "
          f"{pearson([s.valence for s in ml], gold_v):>+10.3f} "
          f"{pearson([s.arousal for s in ml], gold_a):>+10.3f}")

    try:
        from emotion_algebra.neural import affect_from_texts as deepmoji_affect
        dm = deepmoji_affect(texts, lang="en")
        print(f"   {'DeepMoji (shipped, English)':<28} "
              f"{pearson([s.valence for s in dm], gold_v):>+10.3f} "
              f"{pearson([s.arousal for s in dm], gold_a):>+10.3f}")
    except Exception as e:                      # pragma: no cover
        print(f"   DeepMoji unavailable ({e})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xed", required=True, type=Path,
                    help="directory holding xed-{en,pt,ar}.tsv")
    ap.add_argument("--emobank", type=Path)
    ap.add_argument("--cap", type=int, default=600, help="max lines per label")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("Does the model survive the crossing?")
    print("=" * 72)
    print("\nThe probe was fitted on ENGLISH and has never seen Portuguese or")
    print("Arabic. Gold is XED (CC-BY-4.0) — and for pt and ar the labels are")
    print("PROJECTED across subtitle alignments, not human-annotated. Weak gold,")
    print("reported as weak gold. It is also the best that exists.")

    print("\n1 & 2. The crux: anger vs fear, and the axis that separates them")

    results = [r for lang in ("en", "pt", "ar")
               if (r := evaluate(lang, args.xed, args.cap, rng))]

    if args.emobank:
        english_cross_check(args.emobank, rng)

    print("\n" + "=" * 72)
    print("Summary — potency effect size (Cohen's d, anger vs fear)\n")
    print(f"  {'lang':<6} {'n':>6} {'accuracy':>10} {'control':>9} "
          f"{'potency d':>11}  separating axis")
    for r in results:
        print(f"  {r['lang']:<6} {r['n_anger'] + r['n_fear']:>6} "
              f"{r['accuracy']:>10.3f} {r['control']:>9.3f} "
              f"{r['effects']['potency']:>+11.3f}  {r['winner']}")
    print("\nA language whose potency separation collapses does not get shipped")
    print("quietly. It gets shipped with this table next to it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
