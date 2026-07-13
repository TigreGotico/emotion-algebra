"""V1 — Do the core's axes predict human VAD ratings?

Gold: Warriner, Kuperman & Brysbaert (2013), *Norms of valence, arousal, and
dominance for 13,915 English lemmas*, Behavior Research Methods 45:1191-1207.
Human ratings on a 1-9 scale, ~20 raters per word.

**The experiment.** Our lexicon maps words to emotion labels. So:

    word --> emotion label --> MODEL --> (valence, arousal, dominance)

Run that with two models, against the same gold:

* **old** — the Hourglass core: label -> FloatEmotion -> ``to_pad()``, whose
  Dominance is a *fitted regression* over three hand-tuned weights, because the
  Sensitivity axis had collapsed anger and fear together and destroyed the
  dimension.
* **new** — the affect core: label -> prototype -> (valence, arousal, potency),
  where potency is a real axis.

Same inputs, same gold, two models. The central claim of the new core is that
**potency is a real dimension and the old dominance was a reconstruction of a
dimension that had been thrown away.** If that claim is true, dominance
correlation should improve markedly. If it does not, the claim is wrong and this
script will say so.

Run:  python scripts/validate/v1_warriner_vad.py --warriner <path/to/csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    """Rank correlation, without a scipy dependency."""
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return pearson(ra, rb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--warriner", required=True, type=Path,
                    help="Ratings_Warriner_et_al.csv")
    args = ap.parse_args()

    import pandas as pd

    from emotion_algebra.emotions import get_emotion
    from emotion_algebra.float_emotion import FloatEmotion
    from emotion_algebra.lexicons import LEXICON, load_lexicon
    from emotion_algebra.pad import to_pad as hourglass_to_pad
    from emotion_algebra.prototypes import PROTOTYPES, prototype

    gold = pd.read_csv(args.warriner)
    gold = gold[["Word", "V.Mean.Sum", "A.Mean.Sum", "D.Mean.Sum"]].dropna()
    # 1-9 -> valence/dominance in [-1,1], arousal in [0,1]
    gold["v"] = (gold["V.Mean.Sum"] - 5.0) / 4.0
    gold["a"] = (gold["A.Mean.Sum"] - 1.0) / 8.0
    gold["d"] = (gold["D.Mean.Sum"] - 5.0) / 4.0
    gold = gold.set_index(gold["Word"].str.lower())

    load_lexicon()

    rows = []
    for word, entry in LEXICON.items():
        label = (entry.get("emotion") or "").strip().lower()
        if not label or label not in PROTOTYPES:
            continue
        if word not in gold.index:
            continue
        g = gold.loc[word]
        if isinstance(g, pd.DataFrame):      # duplicate lemma
            g = g.iloc[0]

        emo = get_emotion(label)
        if emo is None:
            continue

        # --- old: Hourglass core, dominance by fitted regression
        old = hourglass_to_pad(FloatEmotion.from_emotion(emo))

        # --- new: affect core, potency is a real axis
        p = prototype(label)

        rows.append(
            dict(
                word=word, label=label,
                gold_v=float(g["v"]), gold_a=float(g["a"]), gold_d=float(g["d"]),
                old_v=old.pleasure, old_a=old.arousal, old_d=old.dominance,
                new_v=p.valence, new_a=p.arousal, new_d=p.potency,
            )
        )

    if not rows:
        print("no overlap between the lexicon and Warriner", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    n = len(df)
    labels = df["label"].nunique()

    print(f"V1 — Warriner et al. (2013) human VAD norms")
    print(f"    gold: 13,915 lemmas | overlap with our lexicon: {n} words "
          f"across {labels} emotion labels\n")

    header = f"{'axis':<12}{'old (Hourglass)':>18}{'new (affect core)':>20}{'delta':>10}"
    print(header)
    print("-" * len(header))

    results = {}
    for axis, gold_col in (("valence", "gold_v"), ("arousal", "gold_a"),
                           ("dominance", "gold_d")):
        old_key = {"valence": "old_v", "arousal": "old_a", "dominance": "old_d"}[axis]
        new_key = {"valence": "new_v", "arousal": "new_a", "dominance": "new_d"}[axis]
        r_old = pearson(df[gold_col], df[old_key])
        r_new = pearson(df[gold_col], df[new_key])
        results[axis] = (r_old, r_new)
        flag = "  <--" if axis == "dominance" else ""
        print(f"{axis:<12}{r_old:>18.3f}{r_new:>20.3f}{r_new - r_old:>+10.3f}{flag}")

    print("\n(Pearson r against human means. Dominance is the axis the whole")
    print(" redesign turns on: the old model had to reconstruct it by regression")
    print(" because collapsing anger and fear onto one axis had destroyed it.)\n")

    print("Spearman (rank), same comparison:")
    for axis, gold_col in (("valence", "gold_v"), ("arousal", "gold_a"),
                           ("dominance", "gold_d")):
        old_key = {"valence": "old_v", "arousal": "old_a", "dominance": "old_d"}[axis]
        new_key = {"valence": "new_v", "arousal": "new_a", "dominance": "new_d"}[axis]
        print(f"  {axis:<12}old {spearman(df[gold_col], df[old_key]):+.3f}"
              f"   new {spearman(df[gold_col], df[new_key]):+.3f}")

    # The crux, isolated: the anger and fear families only.
    print("\nThe crux — anger-family vs fear-family words only:")
    fam = {
        "anger": ("annoyance", "anger", "rage"),
        "fear": ("apprehension", "fear", "terror"),
    }
    for family, names in fam.items():
        sub = df[df["label"].isin(names)]
        if sub.empty:
            print(f"  {family}: no words")
            continue
        print(f"  {family:<6} n={len(sub):<5} "
              f"human dominance {sub['gold_d'].mean():+.3f} | "
              f"old {sub['old_d'].mean():+.3f} | new {sub['new_d'].mean():+.3f}")
    print("\n  Humans rate anger words as MORE dominant than fear words. A model")
    print("  that puts them at opposite ends of one axis can only get this right")
    print("  by accident.")

    r_old_d, r_new_d = results["dominance"]
    if r_new_d > r_old_d:
        print(f"\nVERDICT: potency predicts human dominance better than the old "
              f"fitted reconstruction (r {r_old_d:.3f} -> {r_new_d:.3f}).")
    else:
        print(f"\nVERDICT: the new potency axis does NOT beat the old fitted "
              f"dominance (r {r_old_d:.3f} -> {r_new_d:.3f}). The central claim "
              f"of the redesign is not supported by this benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
