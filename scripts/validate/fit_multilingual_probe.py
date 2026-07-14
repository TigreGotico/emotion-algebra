"""Fit the multilingual probe — on English, and only on English.

The probe that reads Portuguese and Arabic is fitted on English gold and never
sees a word of either. That is deliberate, and it is the only honest option
available:

* **There is no human-rated Arabic valence/arousal/dominance lexicon.** The
  Arabic entries in the NRC lexicon family are machine translations of English
  sentiment scores — English raters' judgements in Arabic clothing. Fitting on
  them would launder an English opinion into an Arabic-looking number.
* The only openly available Portuguese affective norms are **Brazilian**, not
  European.

What makes it work anyway is the encoder. ``paraphrase-multilingual-MiniLM-L12-v2``
is *distilled so that a sentence and its translation land in the same place*. If
that alignment holds, a direction fitted in the English region of the space is
the same direction in the Portuguese and Arabic regions of it — and the mapping
crosses for free.

**If.** That is a claim, not an assumption, and it is not tested here. This
script only fits. ``eval_multilingual.py`` is where it is put at risk, against
Arabic and Portuguese gold the probe has never seen.

Targets, as for the English probe:

* **valence, potency, unpredictability** — GoEmotions (Demszky et al. 2020),
  43,410 human-labelled comments, mapped to prototype coordinates.
* **arousal** — EmoBank (Buechel & Hahn 2017) human arousal ratings. Prototype
  arousal is too coarse to fit against, and arousal is the axis the typographic
  cues carry, so it gets real gold.

Run:
    python scripts/validate/fit_multilingual_probe.py \\
        --goemotions data/goemotions.tsv --emotions data/emotions.txt \\
        --emobank data/emobank.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from emotion_algebra.lang import CHANNELS
from emotion_algebra.multilingual import N_EMBED, PROBE_PATH, design_matrix, encode
from emotion_algebra.neural import PROBE_AXES
from emotion_algebra.prototypes import prototype

#: GoEmotions label -> our prototype. Only unambiguous mappings; the rest are
#: dropped rather than forced. Identical to the English probe's map, so the two
#: are fitted against the same target and can be compared.
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

N_TYPO = len(CHANNELS)

#: The typographic rows are wired to AROUSAL and nothing else.
#:
#: Emoji and embeddings tell you how someone feels; punctuation tells you how
#: loudly. Letting the typographic cues touch valence would let "!!!" be read as
#: enthusiasm, which is how "this is unacceptable!!!" ends up scored as mildly
#: positive. The constraint is structural, not fitted.
AROUSAL = PROBE_AXES.index("arousal")


def ridge(X, y, lam=1.0):
    """Closed-form ridge regression with an unpenalised intercept."""
    A = X.T @ X + lam * np.eye(X.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, X.T @ y)


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_goemotions(tsv: Path, emotions: Path, per_label: int, seed: int):
    import pandas as pd

    names = emotions.read_text().split()
    df = pd.read_csv(tsv, sep="\t", header=None, names=["text", "labels", "id"])
    # Single-label only: a comment tagged both 'anger' and 'joy' tells us nothing
    # clean about either.
    df = df[~df["labels"].astype(str).str.contains(",")]
    df["label"] = df["labels"].astype(int).map(lambda i: names[i])
    df = df[df["label"].isin(LABEL_MAP)]
    df = pd.concat([
        g.sample(min(len(g), per_label), random_state=seed)
        for _, g in df.groupby("label")
    ]).reset_index(drop=True)
    return df["text"].astype(str).tolist(), df["label"].tolist()


def load_emobank(csv: Path, split: str, cap: int = 0, seed: int = 0):
    import pandas as pd

    df = pd.read_csv(csv)
    df = df[df["split"] == split]
    if cap and len(df) > cap:
        df = df.sample(cap, random_state=seed)
    # EmoBank is rated 1-5, centred at 3. Rescale arousal to the core's [0, 1].
    return df["text"].astype(str).tolist(), ((df["A"].astype(float) - 1.0) / 4.0).to_numpy()


def encode_verbose(texts, label: str) -> np.ndarray:
    print(f"    encoding {len(texts)} {label} sentences...", flush=True)
    return encode(texts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--goemotions", required=True, type=Path)
    ap.add_argument("--emotions", required=True, type=Path)
    ap.add_argument("--emobank", required=True, type=Path)
    ap.add_argument("--per-label", type=int, default=300,
                    help="max comments per label; the encoder is the cost here")
    ap.add_argument("--emobank-cap", type=int, default=0,
                    help="subsample EmoBank train to this many sentences (0 = all)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=PROBE_PATH)
    args = ap.parse_args()

    print(__doc__.split("Run:")[0])

    # ----------------------------------------------------------------- valence,
    # potency, unpredictability: GoEmotions -> prototype coordinates.
    texts, labels = load_goemotions(
        args.goemotions, args.emotions, args.per_label, args.seed
    )
    print(f"GoEmotions: {len(texts)} single-label comments, "
          f"{len(set(labels))} labels")

    X = design_matrix(encode_verbose(texts, "GoEmotions"), texts, lang="en")
    protos = [prototype(LABEL_MAP[l]) for l in labels]
    Y = np.array([
        [p.positivity, p.negativity, p.potency, p.arousal, p.unpredictability]
        for p in protos
    ])

    weights = np.zeros((N_EMBED + N_TYPO + 1, len(PROBE_AXES)))
    for i, axis in enumerate(PROBE_AXES):
        if i == AROUSAL:
            continue                      # fitted on real gold below
        weights[:, i] = ridge(X, Y[:, i], lam=args.lam)

    # The typographic rows must not touch anything but arousal.
    typo_rows = slice(N_EMBED, N_EMBED + N_TYPO)
    for i in range(len(PROBE_AXES)):
        if i != AROUSAL:
            weights[typo_rows, i] = 0.0

    # ------------------------------------------------------------------ arousal:
    # EmoBank human ratings. This is the axis typography carries, so it is worth
    # fitting on gold rather than on prototype coordinates.
    tr_texts, tr_a = load_emobank(args.emobank, "train", args.emobank_cap, args.seed)
    te_texts, te_a = load_emobank(args.emobank, "test")
    print(f"EmoBank:    {len(tr_texts)} train / {len(te_texts)} test sentences")

    Xa = design_matrix(encode_verbose(tr_texts, "EmoBank train"), tr_texts, lang="en")
    weights[:, AROUSAL] = ridge(Xa, tr_a, lam=args.lam)

    # ------------------------------------------------------------------- report
    Xte = design_matrix(encode_verbose(te_texts, "EmoBank test"), te_texts, lang="en")
    held_out_arousal = pearson(Xte @ weights[:, AROUSAL], te_a)

    # Valence too, on the same held-out EmoBank split — a corpus the valence
    # columns were never fitted on.
    import pandas as pd
    eb = pd.read_csv(args.emobank)
    te_v = ((eb[eb["split"] == "test"]["V"].astype(float) - 3.0) / 2.0).to_numpy()
    pred_v = (Xte @ weights[:, PROBE_AXES.index("positivity")]
              - Xte @ weights[:, PROBE_AXES.index("negativity")])
    held_out_valence = pearson(pred_v, te_v)

    print("Held out on EmoBank (never seen in fitting):")
    print(f"    valence   r = {held_out_valence:+.3f}")
    print(f"    arousal   r = {held_out_arousal:+.3f}")
    print("\nThese are ENGLISH numbers. What the probe does to Portuguese and")
    print("Arabic is not established by them, and is measured separately in")
    print("eval_multilingual.py — which is where this can still fail.\n")

    args.out.write_text(json.dumps({
        "axes": list(PROBE_AXES),
        "shape": list(weights.shape),
        "features": (
            f"{N_EMBED} {ENCODER_NOTE} + {N_TYPO} typographic cues (read with the "
            f"language's own orthography; see emotion_algebra.lang) + intercept"
        ),
        "note": (
            "FITTED ON ENGLISH ONLY. Valence/potency/unpredictability on "
            "GoEmotions label prototypes; arousal on EmoBank human ratings "
            "(train split). Applied to Portuguese and Arabic ZERO-SHOT, relying "
            "on the encoder's cross-lingual alignment. Typographic rows are "
            "non-zero only for arousal: embeddings carry valence, typography "
            "carries activation."
        ),
        "held_out_english": {
            "valence": round(held_out_valence, 3),
            "arousal": round(held_out_arousal, 3),
        },
        "weights": weights.tolist(),
    }, indent=1))
    print(f"wrote {args.out}")
    return 0


ENCODER_NOTE = (
    "paraphrase-multilingual-MiniLM-L12-v2 embedding dims (encoder sees the text "
    "with emphasis stripped)"
)


if __name__ == "__main__":
    raise SystemExit(main())
