"""Emotion distance and nearest-neighbour utilities — v1.3.

Distances are computed in Cambria's 4-axis signed integer space using
Euclidean distance over the ``as_array`` property.
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from emotion_algebra.base import EmotionBase
    from emotion_algebra.plutchik import Emotion


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


def closest_emotion(vector) -> "Emotion":
    """Return the named :class:`~emotion_algebra.plutchik.Emotion` nearest to *vector*.

    Parameters
    ----------
    vector:
        Any 4-element array-like ``[sensitivity, attention, pleasantness, aptitude]``.

    Returns
    -------
    Emotion
        The named emotion with the smallest Euclidean distance to *vector*.
    """
    from emotion_algebra.emotions import EMOTIONS
    vec = np.array(vector, dtype=float)
    best = None
    best_dist = float("inf")
    for emo in EMOTIONS.values():
        d = float(np.linalg.norm(emo.as_array.astype(float) - vec))
        if d < best_dist:
            best_dist = d
            best = emo
    return best


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
