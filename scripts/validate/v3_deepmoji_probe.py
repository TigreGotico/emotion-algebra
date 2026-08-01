"""V3 — Is potency a real axis, or an artefact of the appraisal literature?

This is the experiment that can refute the core.

Everything else in this library's benchmark suite leans on *rating scales* —
people asked to score words on a questionnaire. That is exactly the sort of
evidence that can encode a theory rather than test it. So here is an independent
line of evidence with none of that baggage:

* **DeepMoji** (Felbo et al. 2017, EMNLP) was trained on **1.2 billion tweets** to
  predict which emoji a message carried. It has never heard of Plutchik, of
  Scherer, or of coping potential. Its 64-emoji output is a representation of how
  people *actually express* emotion, learned at scale, with no theory imposed.
* **GoEmotions** (Demszky et al. 2020, ACL) is 43,410 Reddit comments with human
  emotion labels — real text, not word lists.

The question
------------
Anger and fear are both negative and both highly aroused. Valence and arousal
cannot tell them apart; the whole redesign rests on the claim that a third axis —
**potency / appraised coping** — does.

If that axis is real, then a theory-free model of human expression should be able
to *find* it. So:

1. Encode GoEmotions comments with DeepMoji.
2. Ask whether **anger and fear are linearly separable** in that space at all.
3. Ask what that separating direction *is*: does it align with our **potency**
   axis, or is it just valence or arousal wearing a hat?
4. Cross-validate a linear probe from DeepMoji -> each core axis, and report
   held-out R^2 per axis.

**If potency is not decodable from DeepMoji above chance, the axis is an
artefact of the appraisal literature and this library is wrong.** That is a real
risk, and it is the point of running it.

Run:
    python scripts/validate/v3_deepmoji_probe.py \\
        --goemotions train.tsv --emotions emotions.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: GoEmotions label -> our prototype. Only unambiguous mappings; the rest are
#: dropped rather than forced.
LABEL_MAP = {
    "anger": "anger",
    "annoyance": "annoyance",
    "fear": "fear",
    "nervousness": "apprehension",
    "sadness": "sadness",
    "grief": "grief",
    "joy": "joy",
    "excitement": "ecstasy",
    "admiration": "admiration",
    "disgust": "disgust",
    "surprise": "surprise",
    "curiosity": "interest",
    "confusion": "distraction",
    "embarrassment": "shame",
    "disappointment": "pensiveness",
    "boredom": "boredom",
    "approval": "acceptance",
    "caring": "trust",
    "desire": "anticipation",
}

AXES = ("valence", "arousal", "potency", "unpredictability")


def ridge(X, y, lam=1.0):
    """Closed-form ridge regression with an intercept."""
    X1 = np.hstack([X, np.ones((len(X), 1))])
    A = X1.T @ X1 + lam * np.eye(X1.shape[1])
    A[-1, -1] -= lam                       # do not penalise the intercept
    return np.linalg.solve(A, X1.T @ y)


def predict(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goemotions", required=True, type=Path, help="train.tsv")
    ap.add_argument("--emotions", required=True, type=Path, help="emotions.txt")
    ap.add_argument("--per-label", type=int, default=300,
                    help="max comments per label (keeps DeepMoji runtime sane)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    from deepmoji_onnx import DeepMojiONNX

    from emotion_algebra.prototypes import prototype

    names = args.emotions.read_text().split()
    df = pd.read_csv(args.goemotions, sep="\t", header=None,
                     names=["text", "labels", "id"])

    # Single-label examples only: a comment tagged both 'anger' and 'joy' tells
    # us nothing clean about either.
    df = df[~df["labels"].astype(str).str.contains(",")]
    df["label"] = df["labels"].astype(int).map(lambda i: names[i])
    df = df[df["label"].isin(LABEL_MAP)]

    rng = np.random.default_rng(args.seed)
    df = pd.concat(
        [
            g.sample(min(len(g), args.per_label), random_state=args.seed)
            for _, g in df.groupby("label")
        ]
    ).reset_index(drop=True)

    print(f"V3 — DeepMoji probe on GoEmotions")
    print(f"    {len(df)} single-label comments across "
          f"{df['label'].nunique()} labels\n")

    model = DeepMojiONNX.from_pretrained()
    texts = df["text"].astype(str).tolist()
    X = np.asarray(model.encode(texts), dtype=float)   # (n, 64) emoji distribution
    labels = df["label"].tolist()

    # Targets: the core coordinates of each comment's emotion.
    protos = [prototype(LABEL_MAP[l]) for l in labels]
    Y = np.array(
        [[p.valence, p.arousal, p.potency, p.unpredictability] for p in protos]
    )

    # ---------------------------------------------------------------
    # 1. Are anger and fear separable in DeepMoji space AT ALL?
    # ---------------------------------------------------------------
    anger_mask = np.array([l in ("anger", "annoyance") for l in labels])
    fear_mask = np.array([l in ("fear", "nervousness") for l in labels])

    print("1. Anger vs fear — both negative, both aroused. Separable?")
    if anger_mask.sum() > 20 and fear_mask.sum() > 20:
        idx = np.where(anger_mask | fear_mask)[0]
        rng.shuffle(idx)
        split = len(idx) // 2
        tr, te = idx[:split], idx[split:]
        target = anger_mask.astype(float)          # 1 = anger, 0 = fear

        w = ridge(X[tr], target[tr], lam=1.0)
        pred = predict(w, X[te]) > 0.5
        acc = float((pred == anger_mask[te]).mean())
        base = float(max(anger_mask[te].mean(), 1 - anger_mask[te].mean()))
        print(f"   held-out accuracy {acc:.3f}  (majority baseline {base:.3f}, "
              f"n={len(te)})")

        # Permutation control: shuffle the labels and refit. If the pipeline can
        # "separate" randomised labels, the result above means nothing.
        shuffled = target[tr].copy()
        rng.shuffle(shuffled)
        w_null = ridge(X[tr], shuffled, lam=1.0)
        acc_null = float(((predict(w_null, X[te]) > 0.5) == anger_mask[te]).mean())
        print(f"   permutation control (labels shuffled): {acc_null:.3f}")

        if acc > base + 0.05:
            print("   -> DeepMoji separates them. A theory-free model of how people")
            print("      actually write DOES encode the anger/fear distinction.")
        else:
            print("   -> NOT separable above baseline. The distinction this library")
            print("      is built on does not show up in expression data.")
    else:
        print("   too few examples")

    # ---------------------------------------------------------------
    # 2. Which core axis does the separating direction look like?
    # ---------------------------------------------------------------
    print("\n2. What IS that direction? Correlate DeepMoji's anger-vs-fear axis")
    print("   with each core axis, over all labels:")
    if anger_mask.sum() > 20 and fear_mask.sum() > 20:
        direction = X[anger_mask].mean(0) - X[fear_mask].mean(0)
        proj = X @ direction
        for i, axis in enumerate(AXES):
            r = float(np.corrcoef(proj, Y[:, i])[0, 1])
            mark = "  <-- the claim" if axis == "potency" else ""
            print(f"     {axis:<18} r = {r:+.3f}{mark}")

    # ---------------------------------------------------------------
    # 3. Cross-validated linear probe: is each axis decodable?
    # ---------------------------------------------------------------
    print("\n3. Can each core axis be DECODED from DeepMoji? (5-fold CV, held-out R^2)")
    n = len(X)
    order = rng.permutation(n)
    folds = np.array_split(order, 5)

    for i, axis in enumerate(AXES):
        scores = []
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            w = ridge(X[tr], Y[tr, i], lam=1.0)
            scores.append(r2(Y[te, i], predict(w, X[te])))
        mean, sd = float(np.mean(scores)), float(np.std(scores))
        mark = "  <-- the claim" if axis == "potency" else ""
        print(f"     {axis:<18} R^2 = {mean:+.3f} +/- {sd:.3f}{mark}")

    print("\nRead this honestly: the targets are PROTOTYPE coordinates, so a high")
    print("R^2 means 'DeepMoji can recover the label's coordinates', not 'the")
    print("coordinates are correct'. The load-bearing result is (1) and (2): that")
    print("anger and fear separate in theory-free expression data, and that the")
    print("direction separating them tracks POTENCY rather than valence or arousal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
