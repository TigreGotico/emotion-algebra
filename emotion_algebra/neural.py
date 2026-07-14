"""Text to affect, via DeepMoji.

The word-lexicon path (:mod:`emotion_algebra.text`) is a bag of words, and it
shows: it has no syntax, no word senses, and its labels are noisy.  On the
sentence *"I'm scared I've broken something"* it fires on **broken -> anger** and
reports an angry, approach-motivated user.  That is the opposite of the truth,
and it is the exact failure this module exists to fix.

DeepMoji (Felbo et al. 2017, *Using millions of emoji occurrences to learn
any-domain representations for detecting sentiment, emotion and sarcasm*, EMNLP)
was trained on **1.2 billion tweets** to predict which emoji a message carried.
It has no theory of emotion at all — which is precisely why it is worth listening
to.  A 64-emoji distribution goes in; a linear probe maps it onto the affect core.

The probe is 325 floats, fitted on GoEmotions (Demszky et al. 2020, 43,410
human-labelled comments), and ships with the package.

What it is good at
------------------
Measured on **EmoBank** (Buechel & Hahn 2017), held out — a different corpus, at
sentence level, never seen during fitting:

=================  ==============
axis               held-out r
=================  ==============
valence            **+0.42**
arousal            **+0.35**
=================  ==============

Arousal is the interesting one.  A probe built on the emoji distribution alone
scores **+0.16**; the eight typographic cues *on their own* score **+0.32**, and
together they reach +0.35.

**Emoji tell you how someone feels.  Punctuation tells you how loudly.**  Emoji
encode hedonic tone; exclamation marks, capitals and stretched vowels encode
activation.  A probe with only the first is deaf in one ear — which is why this
one has both, and why the typographic rows are wired to arousal and nothing else.

One subtlety: DeepMoji reads ``!`` as *excitement*, and excitement looks positive
to it.  Left alone, both "this is wonderful!!!" and "this is unacceptable!!!"
drift toward **neutral valence** — nonsense in one direction and dangerously
wrong in the other.  So the encoder is shown the text with emphasis stripped,
while the typographic cues see it intact.  Emphasis changes how *loud* something
is; it does not change whether it is good or bad.

    >>> affect_from_text("this is wonderful!!!")    # doctest: +SKIP
    AffectState(..., arousal=0.95, ...)             # valence unchanged at +0.56

Honest ceiling: EmoBank's arousal ratings have low inter-annotator agreement, so
+0.35 from 72 linear features is decent, not solved.  A fine-tuned transformer
would do better and cost far more.

**Potency is the reason to use it.**  It is what separates an angry user (who will
escalate) from a frightened one (who will quietly leave), and the probe recovers
it where a bag of words cannot:

    >>> affect_from_text("This is the third time your app has lost my work.")  # doctest: +SKIP
    AffectState(..., potency=+0.16, ...)
    >>> affect_from_text("I'm scared I've broken something.")                  # doctest: +SKIP
    AffectState(..., potency=-0.43, ...)

Near-identical valence; opposite potency. That distinction is the whole point of
the core, and it survives contact with real text.

Requires ``deepmoji-onnx``, which is a core dependency — the model weights (~90 MB)
are downloaded on first use and cached.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import numpy as np

from emotion_algebra.affect import AffectState

#: Where the fitted probe lives.
PROBE_PATH = Path(__file__).parent / "deepmoji_probe.json"

#: Axis order of the probe's output columns.
PROBE_AXES = ("positivity", "negativity", "potency", "arousal", "unpredictability")

#: Number of DeepMoji emoji probabilities.
N_EMOJI = 64

#: Number of typographic cues appended to them.
N_TYPO = 8

_REPEATED = re.compile(r"([a-z])\1{2,}")     # "sooo", "ahhh"
_ELLIPSIS = re.compile(r"[.]{2,}")
_EMPHASIS = re.compile(r"[!?]+")
_INTENSIFIERS = ("very", "extremely", "so ", "really", "totally")


def typographic_features(text: str) -> List[float]:
    """Eight surface cues that carry **arousal**.

    Emoji tell you how someone *feels*; punctuation tells you how *loudly*.
    Exclamation marks, capitals and stretched vowels are the written equivalent
    of raising your voice, and they turn out to predict arousal roughly twice as
    well as the entire 64-dimensional emoji distribution does.

    That asymmetry is the whole reason this function exists. Emoji carry valence;
    typography carries activation. A model with only one of them is deaf in one
    ear.
    """
    text = str(text)
    words = text.split() or [""]
    n = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]

    return [
        text.count("!") / n,                                  # exclamation density
        min(text.count("!"), 5) / 5,                          # "!!!" saturation
        min(text.count("?"), 5) / 5,
        sum(c.isupper() for c in letters) / max(len(letters), 1),   # SHOUTING
        len(_REPEATED.findall(text.lower())) / n,             # "sooo", "ahhh"
        min(len(words), 50) / 50,                             # length
        len(_ELLIPSIS.findall(text)) / n,                     # "..." — *low* arousal
        sum(text.lower().count(w) for w in _INTENSIFIERS) / n,
    ]


@lru_cache(maxsize=1)
def _probe() -> np.ndarray:
    """The (73, 5) probe: 64 emoji + 8 typographic features + intercept -> 5 axes."""
    data = json.loads(PROBE_PATH.read_text())
    return np.array(data["weights"], dtype=float)


@lru_cache(maxsize=1)
def _model():
    """The DeepMoji encoder, loaded once."""
    from deepmoji_onnx import DeepMojiONNX

    return DeepMojiONNX.from_pretrained()


def affect_from_features(features, texts: Sequence[str] = None) -> List[AffectState]:
    """Map DeepMoji features onto the core.

    This is the part of the pipeline this library owns: the probe and the
    projection. It needs no model, no network and no download — which is why it,
    rather than :func:`affect_from_text`, is what the tests exercise.

    Parameters
    ----------
    features:
        Either an ``(n, 64)`` array of raw DeepMoji emoji probabilities — in
        which case *texts* is required, so the typographic cues can be computed —
        or a complete ``(n, 72)`` array with those cues already appended.
    texts:
        The source strings. Required when *features* is ``(n, 64)``.

    Raises
    ------
    ValueError
        If the shapes do not line up.
    """
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {features.shape}")

    if features.shape[1] == N_EMOJI:
        if texts is None:
            raise ValueError(
                "raw (n, 64) emoji features need `texts` too, so the typographic "
                "cues can be computed — they carry the arousal signal"
            )
        if len(texts) != len(features):
            raise ValueError("features and texts must be the same length")
        typo = np.array([typographic_features(t) for t in texts], dtype=float)
        features = np.hstack([features, typo])
    elif features.shape[1] != N_EMOJI + N_TYPO:
        raise ValueError(
            f"expected (n, {N_EMOJI}) or (n, {N_EMOJI + N_TYPO}) features, "
            f"got {features.shape}"
        )

    design = np.hstack([features, np.ones((len(features), 1))])
    scores = design @ _probe()

    return [
        AffectState(
            positivity=float(np.clip(row[0], 0.0, 1.0)),
            negativity=float(np.clip(row[1], 0.0, 1.0)),
            potency=float(np.clip(row[2], -1.0, 1.0)),
            arousal=float(np.clip(row[3], 0.0, 1.0)),
            unpredictability=float(np.clip(row[4], 0.0, 1.0)),
        )
        for row in scores
    ]


def affect_from_texts(texts: Sequence[str]) -> List[AffectState]:
    """Map each string to an :class:`~emotion_algebra.affect.AffectState`.

    Batched — prefer this over calling :func:`affect_from_text` in a loop, since
    the DeepMoji forward pass dominates the cost.

    Downloads the DeepMoji weights (~90 MB) on first use and caches them.

    Raises
    ------
    ValueError
        If *texts* is empty, or any entry is not a string.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("no texts given")
    if not all(isinstance(t, str) for t in texts):
        raise ValueError("every text must be a str")

    from emotion_algebra.homeostasis import SET_POINT

    # DeepMoji's tokenizer rejects empty input, so hold a place for blanks and
    # return the resting state for them rather than failing the whole batch.
    filled = [t if t.strip() else "." for t in texts]

    # The encoder sees the text stripped of emphasis; the typographic cues see it
    # intact. Both halves of that matter.
    #
    # DeepMoji reads "!" as *excitement*, and excitement looks positive to it —
    # so "this is wonderful!!!" and "this is unacceptable!!!" BOTH drift toward
    # neutral valence, which is nonsense in one direction and wrong in the other.
    # Capitals do the same. Strip them for the encoder and the valence estimate
    # holds; keep them for the typographic features, where they belong, because
    # emphasis is exactly what carries AROUSAL.
    #
    # Emoji tell you how someone feels. Punctuation tells you how loudly.
    encoder_input = [
        _EMPHASIS.sub(" ", t.lower()).strip() or "." for t in filled
    ]
    emoji = np.asarray(_model().encode(encoder_input), dtype=float)
    states = affect_from_features(emoji, texts=filled)

    return [
        SET_POINT if not text.strip() else state
        for text, state in zip(texts, states)
    ]


def affect_from_text(text: str) -> AffectState:
    """Map one string to an :class:`~emotion_algebra.affect.AffectState`.

    See the module docstring for what this is good at (valence, potency) and what
    it is not (arousal — near-zero correlation; do not trust it).

    Parameters
    ----------
    text:
        Any string. Empty or whitespace-only input returns the resting set point.

    Returns
    -------
    AffectState
    """
    return affect_from_texts([text])[0]
