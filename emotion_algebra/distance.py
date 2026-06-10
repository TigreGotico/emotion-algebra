"""Emotion distance and nearest-neighbour utilities — v1.3.

Distances are computed in Cambria's 4-axis signed integer space using
Euclidean distance over the ``as_array`` property.
"""
from __future__ import annotations

from typing import Iterable, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.plutchik import Emotion

#: Radius of the neutral zone around the origin.  The nearest non-neutral
#: candidates (the basic emotions) sit on the intensity-1 shell, so any vector
#: with norm below a quarter of that shell is overwhelmingly inert rather than
#: a faint emotion, and is reported as :class:`~emotion_algebra.plutchik.Neutrality`.
NEUTRAL_RADIUS = 0.25

#: Quantization applied to distances/alignments before comparison, so that
#: float rounding noise never decides a winner — only the explicit tie-break does.
_TIE_DECIMALS = 9


def emotion_distance(a: "EmotionBase", b: "EmotionBase") -> float:
    """Euclidean distance between two emotions in the 4-axis Hourglass space.

    Parameters
    ----------
    a, b:
        Any :class:`~emotion_algebra.base.EmotionBase` instances.

    Returns
    -------
    float
        Non-negative; 0.0 means identical vectors.

    Examples
    --------
    >>> emotion_distance(get_emotion("rage"), get_emotion("anger"))
    1.0
    >>> emotion_distance(get_emotion("joy"), get_emotion("sadness"))
    4.0
    """
    va = a.as_array.astype(float)
    vb = b.as_array.astype(float)
    return float(np.linalg.norm(va - vb))


def _nearest(vec: np.ndarray, candidates: "Iterable[EmotionBase]") -> "EmotionBase":
    """Return the candidate nearest to *vec* with a deterministic tie-break.

    Candidates are ranked by:

    1. Euclidean distance to *vec* (smaller wins),
    2. emotional-flow alignment — the dot product of the candidate vector with
       *vec* (larger wins), so among equidistant names the one pointing *with*
       the query beats one pointing across or against it,
    3. lexicographic name (stable, order-independent last resort).

    Distances and alignments are quantized to ``_TIE_DECIMALS`` decimals so the
    winner never depends on float epsilon or candidate iteration order.
    """
    best = None
    best_key = None
    for cand in candidates:
        cvec = cand.as_array.astype(float)
        d = round(float(np.linalg.norm(cvec - vec)), _TIE_DECIMALS)
        alignment = round(float(np.dot(cvec, vec)), _TIE_DECIMALS)
        key = (d, -alignment, cand.name)
        if best_key is None or key < best_key:
            best_key = key
            best = cand
    return best


def closest_emotion(vector, include_feelings: bool = True) -> "EmotionBase":
    """Return the named emotional state nearest to *vector*.

    Parameters
    ----------
    vector:
        Any 4-element array-like ``[sensitivity, attention, pleasantness, aptitude]``.
    include_feelings:
        When ``True`` (default) the candidate set is the 24 named emotions
        **plus** the named composite feelings (Plutchik dyads such as
        ``optimism = anticipation + joy``), so mixed-axis states resolve to an
        honest nearest name instead of an arbitrary single-axis emotion.
        When ``False`` only the 24 basis emotions are considered.

    Returns
    -------
    EmotionBase
        :class:`~emotion_algebra.plutchik.Neutrality` when *vector* lies within
        :data:`NEUTRAL_RADIUS` of the origin; otherwise the
        :class:`~emotion_algebra.plutchik.Emotion` or
        :class:`~emotion_algebra.feelings.Feeling` with the smallest Euclidean
        distance to *vector*.  Exact-distance ties are broken deterministically:
        higher dot-product alignment with the query first, then lexicographic
        name — never float epsilon or iteration order.
    """
    from emotion_algebra.emotions import EMOTIONS
    from emotion_algebra.plutchik import Neutrality

    vec = np.array(vector, dtype=float)
    if float(np.linalg.norm(vec)) < NEUTRAL_RADIUS:
        return Neutrality()

    candidates: List["EmotionBase"] = list(EMOTIONS.values())
    if include_feelings:
        from emotion_algebra.feelings import FEELINGS
        candidates += list(FEELINGS.values())
    return _nearest(vec, candidates)


def emotion_clusters(threshold: float = 1.5) -> List[List["Emotion"]]:
    """Group all named emotions into proximity clusters.

    Parameters
    ----------
    threshold:
        Maximum Euclidean distance for two emotions to share a cluster.

    Returns
    -------
    list of lists
        Each inner list is a group of emotions within *threshold* of the seed.
    """
    from emotion_algebra.emotions import EMOTIONS
    emotions = list(EMOTIONS.values())
    visited: set = set()
    clusters: List[List] = []
    for seed in emotions:
        if seed.name in visited:
            continue
        cluster = [seed]
        visited.add(seed.name)
        for other in emotions:
            if other.name not in visited and emotion_distance(seed, other) <= threshold:
                cluster.append(other)
                visited.add(other.name)
        clusters.append(cluster)
    return clusters
