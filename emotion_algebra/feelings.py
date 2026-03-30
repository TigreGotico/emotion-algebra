"""Feelings — composite emotional states composed of two primary :class:`Emotion` objects.

A :class:`Feeling` is created when two emotions from different dimensions are combined
with ``+``: ``joy + trust == Feeling("love")``.
"""
from __future__ import annotations

import random
from copy import copy, deepcopy
from types import MappingProxyType
from typing import List, Optional, Union

from emotion_algebra.base import EmotionBase
from emotion_algebra.plutchik import Emotion, Neutrality


FEELING_NAMES = {'acknowledgement': ['serenity', 'acceptance'],
             'acquiescence': ['acceptance', 'apprehension'],
             'aggressiveness': ['anger', 'anticipation'],
             'anxiety': ['anticipation', 'fear'],
             'awe': ['fear', 'surprise'],
             'bemusement': ['interest', 'serenity'],
             'contempt': ['disgust', 'anger'],
             'curiosity': ['trust', 'surprise'],
             'cynicism': ['disgust', 'anticipation'],
             'delight': ['joy', 'surprise'],
             'despair': ['fear', 'sadness'],
             'devotion': ['ecstasy', 'admiration'],
             'disapproval': ['surprise', 'sadness'],
             'disfavor': ['annoyance', 'interest'],
             'dismay': ['distraction', 'pensiveness'],
             'dominance': ['anger', 'trust'],
             'domination': ['rage', 'vigilance'],
             'envy': ['sadness', 'anger'],
             'fatalism': ['vigilance', 'fear'],
             'guilt': ['joy', 'fear'],
             'hatred': ['loathing', 'rage'],
             'hope': ['anticipation', 'trust'],
             'horror': ['amazement', 'grief'],
             'impatience': ['boredom', 'annoyance'],
             'listlessness': ['pensiveness', 'boredom'],
             'love': ['joy', 'trust'],
             'morbidness': ['disgust', 'joy'],
             'optimism': ['anticipation', 'joy'],
             'outrage': ['surprise', 'anger'],
             'pessimism': ['sadness', 'anticipation'],
             'petrification': ['terror', 'amazement'],
             'pride': ['anger', 'joy'],
             'remorse': ['sadness', 'disgust'],
             'sentimentality': ['trust', 'sadness'],
             'shame': ['grief', 'loathing'],
             'submission': ['trust', 'fear'],
             'subservience': ['admiration', 'terror'],
             'unbelief': ['surprise', 'disgust'],
             'wariness': ['apprehension', 'distraction'],
             'zeal': ['vigilance', 'ecstasy']}
OPPOSITE_FEELINGS_NAMES = {
    "optimism": "disapproval",
    "hope": "unbelief",
    "anxiety": "outrage",
    "love": "remorse",
    "guilt": "envy",
    "delight": "pessimism",
    "submission": "contempt",
    "curiosity": "cynicism",
    "sentimentality": "morbidness",
    "despair": "pride",
    "shame": "dominance",
    "bemusement": "dismay",
    "zeal": "horror",
    "acknowledgement": "listlessness",
    "devotion": "shame",
    "acquiescence": "impatience",
    "subservience": "hatred",
    "wariness": "disfavor",
    "petrification": "domination"
}


def get_feeling_from_emotions(emotion1: str, emotion2: str) -> Optional[str]:
    """Return the feeling name for a pair of emotion names, or ``None`` if not found."""
    for feel in FEELING_NAMES:
        emos = FEELING_NAMES[feel]
        if emotion1 in emos and emotion2 in emos:
            return feel
    return None


class Feeling(EmotionBase):
    """A composite emotional state composed of two or more :class:`Emotion` objects.

    Feelings are named dyads from Plutchik's Wheel (e.g. joy+trust → love).
    When no named dyad matches, a descriptive secondary name is generated.

    Parameters
    ----------
    name:
        Optional explicit name override.
    """

    def __init__(self, name: str = "") -> None:
        self._name: str = name
        self.emotions: List[Emotion] = []

    @property
    def name(self) -> str:
        """Named dyad if recognized, otherwise falls back to :attr:`secondary_name`."""
        if self._name:
            return self._name
        if len(self.emotions) == 2:
            name = get_feeling_from_emotions(self.emotions[0].name, self.emotions[1].name)
            if name:
                return name
        return self.secondary_name

    @property
    def secondary_name(self) -> str:
        """Descriptive fallback name listing all component emotions."""
        if len(self.emotions):
            name = "mix of "
            for emo in self.emotions:
                name += emo.name + " and "
            return name[:-5]

        return "neutrality"

    @property
    def base_feeling(self):
        sensitivity, attention, pleasantness, aptitude = self.emotion_vector

        return sensitivity + attention + pleasantness + aptitude

    @property
    def emotion_vector(self):
        sensitivity = Neutrality()
        attention = Neutrality()
        pleasantness = Neutrality()
        aptitude = Neutrality()

        for e in self.emotions:
            if "sensitivity" in e:
                sensitivity = sensitivity + e
            if "attention" in e:
                attention = attention + e
            if "aptitude" in e:
                aptitude = aptitude + e
            if "pleasantness" in e:
                pleasantness = pleasantness + e

        return [sensitivity, attention, pleasantness, aptitude]

    @property
    def opposite_feeling(self):
        f = Feeling()
        for e in self.emotions:
            f = f - e
        return f

    @property
    def dimensions(self):
        return [emo.dimension for emo in self.emotions]

    @property
    def emotional_flow(self) -> int:
        """Signed net intensity: sum of component flows (spec §2.5).

        Previously used ``np.linalg.norm``, which is always ≥ 0 and made the
        signed branches of ``type`` permanently dead.  Now matches
        ``CompositeEmotion.emotional_flow`` semantics.
        """
        return sum(e.emotional_flow for e in self.emotions)

    @property
    def valence(self) -> int:
        """Pleasantness-axis sum across all component emotions (spec §2.5).

        Only emotions on the Pleasantness axis contribute to hedonic tone.
        """
        return sum(e.valence for e in self.emotions)

    @property
    def arousal(self) -> int:
        """Peak activation across component emotions: ``max(|e.emotional_flow|)``."""
        if not self.emotions:
            return 0
        return max(e.arousal for e in self.emotions)

    @property
    def type(self) -> str:
        """Russell (1980) Circumplex classification using feeling valence and arousal."""
        from emotion_algebra.plutchik import _circumplex_type
        return _circumplex_type(self.valence, self.arousal)

    def __str__(self):
        return self.secondary_name

    def __repr__(self):
        return "FeelingObject:" + self.name

    def __len__(self):
        return len([e for e in self.emotions if e.emotional_flow != 0])

    def __bool__(self) -> bool:
        return len(self) > 0

    def __neg__(self):
        # get opposite emotion
        return self.opposite_feeling

    def __pos__(self):
        if self.emotional_flow:
            return copy(self)
        return copy(self.opposite_feeling)

    def __abs__(self):
        return Neutrality()

    def __add__(self, other):
        if isinstance(other, Feeling):
            feel = deepcopy(self)
            for emo in other.emotions:
                feel = feel + emo
            if len(feel) == 1:
                return feel.emotions[0]
            return feel

        if isinstance(other, str):
            other = Emotion.string_to_emotion(other)
            if isinstance(other, str):
                return self.name + other

        if isinstance(other, Neutrality):
            feel = deepcopy(self)
            return feel

        if isinstance(other, Emotion):
            feel = deepcopy(self)
            feel.emotions.append(other)
            return feel

        # upgrade emotions
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo + other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, str):
            other = Emotion.string_to_emotion(other)

        if isinstance(other, Feeling):
            feel = deepcopy(self)
            for emo in other.emotions:
                result = feel - emo
                if isinstance(result, Feeling):
                    feel = result
                else:
                    return result  # reduced to a single Emotion
            if isinstance(feel, Feeling) and len(feel) == 1:
                return feel.emotions[0]
            return feel

        if isinstance(other, str):
            return self.name + other

        if isinstance(other, Neutrality):
            feel = deepcopy(self)
            return feel

        if isinstance(other, Emotion):
            feel = deepcopy(self)
            if other in feel.emotions:
                feel.emotions.remove(other)
            else:
                feel = feel + other.opposite_emotion
            if len(feel) == 1:
                return feel.emotions[0]
            return feel

        # upgrade emotions
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo - other
            return feel
        except:
            return NotImplemented

    def __mul__(self, other):
        """Intensify all component emotions by *other* steps up their axis.

        Equivalent to applying ``+ other`` to each component's intensity.
        Uses addition rather than multiplication because ``Emotion`` does not
        define ``__mul__(int)`` (multiplication of an emotion by an integer has
        no Plutchik-defined semantics; only intensity stepping does).
        """
        if isinstance(other, str):
            other = Emotion.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return deepcopy(self)
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo + other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except Exception:
            return NotImplemented

    def __truediv__(self, other):
        """Divide all component emotion flows by *other* (int or float)."""
        if isinstance(other, Neutrality):
            return deepcopy(self)
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo / other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except Exception:
            return NotImplemented

    def __floordiv__(self, other):
        """Floor-divide all component emotion flows by *other* (int)."""
        if isinstance(other, Neutrality):
            return deepcopy(self)
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo // other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except Exception:
            return NotImplemented

    def __lshift__(self, other):
        """Decrease all component emotions by *other* intensity steps."""
        if isinstance(other, Neutrality):
            return deepcopy(self)
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo << other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except Exception:
            return NotImplemented

    def __rshift__(self, other):
        """Increase all component emotions by *other* intensity steps."""
        if isinstance(other, Neutrality):
            return deepcopy(self)
        try:
            other = int(other)
            feel = deepcopy(self)
            for idx, emo in enumerate(feel.emotions):
                feel.emotions[idx] = emo >> other
            feel.emotions = [e for e in feel.emotions if isinstance(e, Emotion)]
            if len(feel.emotions) == 1:
                return feel.emotions[0]
            return feel
        except Exception:
            return NotImplemented

    def __eq__(self, other):
        # compare emotion vectors
        if isinstance(other, Emotion) or isinstance(other, Feeling):
            if other.emotion_vector == self.emotion_vector:
                return True
            return False
        return self._name == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, item):
        if isinstance(item, Neutrality):
            return True
        if isinstance(item, Emotion):
            return item.base_emotion in self.emotion_vector
        return NotImplemented

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "type": "feeling",
            "name": self._name,
            "emotions": [e.to_dict() for e in self.emotions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Feeling":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        from emotion_algebra.feelings import get_feeling
        name = data.get("name", "")
        f = get_feeling(name)
        if f is not None:
            return f
        feeling = cls(name)
        from emotion_algebra.emotions import get_emotion
        feeling.emotions = [
            get_emotion(e["name"]) for e in data.get("emotions", [])
            if get_emotion(e["name"]) is not None
        ]
        return feeling


def _get_feeling_emotions():
    bucket = {}
    from emotion_algebra.emotions import EMOTIONS
    for feeling in FEELING_NAMES:
        emotions = FEELING_NAMES[feeling]
        bucket[feeling] = []
        for e in emotions:
            bucket[feeling].append(EMOTIONS[e])
    return bucket


FEELINGS_TO_EMOTION_MAP = _get_feeling_emotions()


def _get_feelings():
    bucket = {}
    for feeling in FEELINGS_TO_EMOTION_MAP:
        f = Feeling()
        #f.name = feeling.lower()
        for emotion in FEELINGS_TO_EMOTION_MAP[feeling]:
            f.emotions.append(emotion)
            from emotion_algebra.emotions import DIMENSIONS
            d = DIMENSIONS.get(emotion.name)

            if isinstance(d, list):
                for dimension in d:
                    f.dimensions.append(dimension)
            elif d:
                f.dimensions.append(d)
        bucket[feeling.lower()] = f
    return bucket


FEELINGS: MappingProxyType = MappingProxyType(_get_feelings())


def get_feeling(name: str) -> Optional["Feeling"]:
    """Return the :class:`Feeling` for *name*, or ``None`` if not found."""
    return FEELINGS.get(name)


def random_feeling() -> "Feeling":
    """Return a random :class:`Feeling` from the named dyads."""
    return FEELINGS[random.choice(list(FEELINGS.keys()))]


