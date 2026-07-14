"""Category readout — emotion names as a distribution, never as a basis.

The honest answer to "what emotion is this?" is not a name. Lindquist et al.
(2012) and Siegel et al. (2018) found no consistent category-specific signature
for discrete emotions, in neuroimaging or in autonomic physiology; and Cowen &
Keltner (2017) found categories bridged by *continuous gradients* rather than
separated by boundaries. So the readout is a **distribution over labels**.

    >>> from emotion_algebra.prototypes import prototype
    >>> from emotion_algebra.readout import label, dominant
    >>> dominant(prototype("rage"))
    'rage'
    >>> d = label(prototype("rage"))
    >>> d["rage"] > d["terror"]        # but terror is not zero — they are neighbours
    True

:func:`dominant` is the argmax convenience for callers who need one name. It is
a *summary* of the distribution, and it discards exactly the information that
makes the distribution honest — the runners-up.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from emotion_algebra.affect import AffectState
from emotion_algebra.prototypes import PROTOTYPES

#: Softness of the readout. Lower is sharper (more category-like); higher is
#: smoother (more gradient-like). Calibrated so that a prototype reads as itself
#: with clear margin while its true neighbours stay visibly non-zero — the
#: gradient structure Cowen & Keltner describe.
TEMPERATURE: float = 0.25


def label(
    state: AffectState,
    temperature: float = TEMPERATURE,
    top_k: Optional[int] = None,
    lang: str = "en",
) -> Dict[str, float]:
    """Return ``P(label | state)`` over the named prototypes.

    Weights fall off with distance from each prototype in the core, softmaxed.
    They are non-negative and sum to 1.

    Parameters
    ----------
    state:
        The :class:`~emotion_algebra.affect.AffectState` to name.
    lang:
        The language to answer in. This renames the labels; it does **not** move
        the prototypes, and the probabilities are identical in every language.
        See :mod:`emotion_algebra.names` for why that is a claim worth being
        careful about rather than an implementation detail.
    temperature:
        Softness. Must be positive.
    top_k:
        Keep only the *k* most probable labels (renormalized). ``None`` keeps all.

    Returns
    -------
    dict
        Label to probability, ordered most probable first.

    Raises
    ------
    ValueError
        If *temperature* is not positive, or *top_k* is not positive.
    """
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    if top_k is not None and int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}")

    table = _table(lang)
    names = list(table)
    dists = np.array([state.distance(table[n]) for n in names], dtype=float)

    # Softmax over negative distance. Subtracting the min is the standard
    # overflow guard and leaves the distribution unchanged.
    logits = -dists / float(temperature)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()

    ranked = sorted(zip(names, weights), key=lambda kv: (-kv[1], kv[0]))
    if top_k is not None:
        ranked = ranked[: int(top_k)]
        total = sum(w for _, w in ranked)
        ranked = [(n, w / total) for n, w in ranked]
    return {n: float(w) for n, w in ranked}


def dominant(state: AffectState, lang: str = "en") -> str:
    """The single most probable label — the argmax of :func:`label`.

    A convenience, and a lossy one: it throws away the runners-up, which are
    where the gradient structure lives. Prefer :func:`label` when the answer
    matters.

    *lang* changes the **name** returned, never the geometry that chose it. The
    nearest prototype to a state is the same point whatever it is called.
    """
    table = _table(lang)
    dists = [(state.distance(table[n]), n) for n in table]
    return min(dists)[1]


def _table(lang: str) -> dict:
    """The prototypes, keyed by their names in *lang*.

    The coordinates are canonical and language-neutral. Only the keys move — see
    :mod:`emotion_algebra.names` for why that distinction is load-bearing rather
    than pedantic.
    """
    if lang == "en":
        return PROTOTYPES

    from emotion_algebra.names import names_in

    return {
        localized: PROTOTYPES[name]
        for localized, name in names_in(lang).items()
    }


def entropy(state: AffectState, temperature: float = TEMPERATURE) -> float:
    """Shannon entropy of the readout, in bits.

    High entropy means the state sits *between* named categories — which is a
    real thing for a state to do, not a failure to classify it. A model that
    always returns one confident name is hiding this.
    """
    probs = np.array(list(label(state, temperature=temperature).values()))
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())
