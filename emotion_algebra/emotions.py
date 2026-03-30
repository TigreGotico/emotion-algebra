from __future__ import annotations

import random
from typing import Optional, TYPE_CHECKING
from types import MappingProxyType

from emotion_algebra.plutchik import DIMENSIONS

if TYPE_CHECKING:
    from emotion_algebra.plutchik import Emotion, EmotionalDimension


def _get_emotion_map():
    bucket = {}

    # get the basic emotions from each dimension
    for dimension_name in DIMENSIONS:
        dimension = DIMENSIONS[dimension_name]
        bucket[dimension.basic_emotion.name] = dimension.basic_emotion
        bucket[dimension.basic_opposite.name] = dimension.basic_opposite
        bucket[dimension.mild_emotion.name] = dimension.mild_emotion
        bucket[dimension.mild_opposite.name] = dimension.mild_opposite
        bucket[dimension.intense_emotion.name] = dimension.intense_emotion
        bucket[dimension.intense_opposite.name] = dimension.intense_opposite
    return bucket


EMOTIONS: MappingProxyType = MappingProxyType(_get_emotion_map())

EMOTION_NAMES = [EMOTIONS[e].name for e in EMOTIONS]

POSITIVE_EMOTIONS = [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].valence > 0]

NEGATIVE_EMOTIONS = [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].valence < 0]

DIMENSION_TO_EMOTION_MAP = {
    "sensitivity": [EMOTIONS[e] for e in EMOTIONS if
                    EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "sensitivity"],
    "attention": [EMOTIONS[e] for e in EMOTIONS if
                  EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "attention"],
    "pleasantness": [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].dimension and
                     EMOTIONS[e].dimension.name == "pleasantness"],
    "aptitude": [EMOTIONS[e] for e in EMOTIONS if
                 EMOTIONS[e].dimension and EMOTIONS[e].dimension.name == "aptitude"]
}

KIND_TO_EMOTION_MAP = {
    "related to object properties":
        [EMOTIONS[e] for e in EMOTIONS if
         EMOTIONS[e].kind == "related to object properties"],
    'future appraisal': [EMOTIONS[e] for e in EMOTIONS if
                         EMOTIONS[e].kind == "future appraisal"],
    'event related': [EMOTIONS[e] for e in EMOTIONS if
                      EMOTIONS[e].kind == "event related"],
    'self appraisal': [EMOTIONS[e] for e in EMOTIONS if
                       EMOTIONS[e].kind == "self appraisal"],
    'social': [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].kind == "social"],
    'cathected': [EMOTIONS[e] for e in EMOTIONS if EMOTIONS[e].kind == "cathected"]
}


def random_emotion() -> "Emotion":
    """Return a random :class:`~emotion_algebra.plutchik.Emotion` from the 24 primary emotions."""
    return EMOTIONS.get(random.choice(list(EMOTIONS.keys())))


def get_emotion(emotion_name: str) -> Optional["Emotion"]:
    """Return the :class:`~emotion_algebra.plutchik.Emotion` for *emotion_name*, or ``None``."""
    return EMOTIONS.get(emotion_name)


def get_dimension(dimension_name: str) -> Optional["EmotionalDimension"]:
    """Return the :class:`~emotion_algebra.plutchik.EmotionalDimension` for *dimension_name*, or ``None``."""
    return DIMENSIONS.get(dimension_name)


def emotion_to_dimension(emotion_name):
    """Return the :class:`EmotionalDimension` for a named emotion, or ``None``."""
    emotion = get_emotion(emotion_name)
    if emotion is not None:
        return emotion.dimension
    return None
