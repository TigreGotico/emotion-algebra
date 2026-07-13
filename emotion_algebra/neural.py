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

What it is good at, and what it is not
--------------------------------------
Measured on **EmoBank** (Buechel & Hahn 2017), held out — a different corpus, at
sentence level, never seen during fitting:

=================  ==============  ==================
axis               word lexicon    DeepMoji probe
=================  ==============  ==================
valence            +0.331          **+0.466**
arousal            **+0.129**      +0.028
=================  ==============  ==================

**Valence is much better.  Arousal is worse — near zero.**  Emoji usage carries
hedonic tone far more than activation, which is not obvious in advance but is
plainly true.  *Arousal from text is an unsolved problem here, and neither path
solves it.*  Do not trust :attr:`~emotion_algebra.affect.AffectState.arousal`
from this function.

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
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import numpy as np

from emotion_algebra.affect import AffectState

#: Where the fitted probe lives.
PROBE_PATH = Path(__file__).parent / "deepmoji_probe.json"

#: Axis order of the probe's output columns.
PROBE_AXES = ("positivity", "negativity", "potency", "arousal", "unpredictability")


@lru_cache(maxsize=1)
def _probe() -> np.ndarray:
    """The (65, 5) probe matrix: 64 emoji features + intercept -> 5 core axes."""
    data = json.loads(PROBE_PATH.read_text())
    return np.array(data["weights"], dtype=float)


@lru_cache(maxsize=1)
def _model():
    """The DeepMoji encoder, loaded once."""
    from deepmoji_onnx import DeepMojiONNX

    return DeepMojiONNX.from_pretrained()


def affect_from_texts(texts: Sequence[str]) -> List[AffectState]:
    """Map each string to an :class:`~emotion_algebra.affect.AffectState`.

    Batched — prefer this over calling :func:`affect_from_text` in a loop, since
    the DeepMoji forward pass dominates the cost.

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

    # DeepMoji's tokenizer rejects empty input, so hold a place for blanks and
    # return the resting state for them rather than failing the whole batch.
    filled = [t if t.strip() else "." for t in texts]

    features = np.asarray(_model().encode(filled), dtype=float)
    design = np.hstack([features, np.ones((len(features), 1))])
    scores = design @ _probe()

    states = []
    for text, row in zip(texts, scores):
        if not text.strip():
            from emotion_algebra.homeostasis import SET_POINT

            states.append(SET_POINT)
            continue
        values = dict(zip(PROBE_AXES, row))
        states.append(
            AffectState(
                positivity=float(np.clip(values["positivity"], 0.0, 1.0)),
                negativity=float(np.clip(values["negativity"], 0.0, 1.0)),
                potency=float(np.clip(values["potency"], -1.0, 1.0)),
                arousal=float(np.clip(values["arousal"], 0.0, 1.0)),
                unpredictability=float(
                    np.clip(values["unpredictability"], 0.0, 1.0)
                ),
            )
        )
    return states


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
