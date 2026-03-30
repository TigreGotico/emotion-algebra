"""Feelings — composite emotional states composed of two primary :class:`Emotion` objects.

A :class:`Feeling` is created when two emotions from different dimensions are combined
with ``+``: ``joy + trust == Feeling("love")``.
"""
from __future__ import annotations

import random
from copy import copy, deepcopy
from typing import List, Optional, Union

import numpy as np

from emotion_data.plutchik import Emotion, Neutrality


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
    "acknowledgment": "listlessness",
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


class Feeling(object):
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
    def secondary_name(self):
        if len(self.emotions):
            name = "mix of "
            print(self.emotions)
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

    @staticmethod
    def string_to_emotion(string=""):
        from emotion_data.emotions import EMOTIONS
        if string in EMOTIONS:
            return copy(EMOTIONS[string])
        if string in FEELINGS:
            return FEELINGS[string]
        return string

    @property
    def emotional_flow(self):
        # scalar product
        flows = [e.emotional_flow for e in self.emotion_vector]
        flow_vector = np.array(flows)
        if self.valence > 0:
            return np.linalg.norm(flow_vector)
        return np.linalg.norm(flow_vector) * -1

    @property
    def valence(self) -> int:
        """Aggregate valence: ``1`` positive, ``-1`` negative, ``0`` neutral."""
        flows = sum([e.emotional_flow for e in self.emotion_vector])
        if flows < 0:
            return -1
        if flows > 0:
            return 1
        return 0

    def __int__(self):
        return int(self.emotional_flow)

    def __float__(self):
        return self.emotional_flow

    def __str__(self):
        return self.secondary_name

    def __repr__(self):
        return "FeelingObject:" + self.name

    def __len__(self):
        return len([e for e in self.emotions if e.emotional_flow != 0])

    def __bool__(self):
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
            other = self.string_to_emotion(other)
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
            other = self.string_to_emotion(other)

        if isinstance(other, Feeling):
            feel = deepcopy(self)
            for emo in other.emotions:
                feel = feel - emo
            if len(feel) == 1:
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

    def __eq__(self, other):
        # compare emotion vectors
        if isinstance(other, Emotion) or isinstance(other, Feeling):
            if other.emotion_vector == self.emotion_vector:
                return True
            return False
        return self._name == other

    def __lt__(self, other):
        return self.emotional_flow < int(other)

    def __le__(self, other):
        return self.emotional_flow <= int(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return self.emotional_flow > int(other)

    def __ge__(self, other):
        return self.emotional_flow >= int(other)

    def __contains__(self, item):
        if isinstance(item, Neutrality):
            return True
        if isinstance(item, Emotion):
            return item.base_emotion in self.emotion_vector
        return NotImplemented

    """
    def __mul__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return copy(self)
        # TODO composite emotion
        if isinstance(other, Emotion):
            return NotImplemented
        # create composite or mutiply flow
        try:
            other = int(other)
            flow = self.emotional_flow * other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return copy(self)
        if isinstance(other, Emotion):
            return NotImplemented

        try:
            other = int(other)
            flow = self.emotional_flow / other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return copy(self)
        if isinstance(other, Emotion):
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow // other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __lshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return copy(self)
        if isinstance(other, Emotion):
            if other.dimension == self.dimension:
                flow = other.emotional_flow - self.emotional_flow
                return self.emotion_from_flow(flow)
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow - other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, str):
            other = self.string_to_emotion(other)
        if isinstance(other, Neutrality):
            return copy(self)
        if isinstance(other, Emotion):
            if other.dimension == self.dimension:
                flow = other.emotional_flow + self.emotional_flow
                return self.emotion_from_flow(flow)
            return NotImplemented
        try:
            other = int(other)
            flow = self.emotional_flow + other
            return self.emotion_from_flow(flow)
        except:
            return NotImplemented

    # TODO
    # (+=, -=, *=, @=, /=, //=, %=, **=, <<=, >>=, &=, ^=, |=).
    #    object.__iadd__(self, other)
    # object.__isub__(self, other)
    # object.__imul__(self, other)
    # object.__imatmul__(self, other)¶
    # object.__itruediv__(self, other)
    # object.__ifloordiv__(self, other)
    # object.__imod__(self, other)
    # object.__ipow__(self, other[, modulo])
    # object.__ilshift__(self, other)
    # object.__irshift__(self, other)
    # object.__iand__(self, other)
    # object.__ixor__(self, other)
    # object.__ior__(self, other)

    
    """


def _get_feeling_emotions():
    bucket = {}
    from emotion_data.emotions import EMOTIONS
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
            from emotion_data.emotions import DIMENSIONS
            d = DIMENSIONS.get(emotion.name)

            if isinstance(d, list):
                for dimension in d:
                    f.dimensions.append(dimension)
            elif d:
                f.dimensions.append(d)
        bucket[feeling.lower()] = f
    return bucket


FEELINGS = _get_feelings()


def get_feeling(name):
    return FEELINGS.get(name)


def random_feeling():
    return FEELINGS[random.choice(list(FEELINGS.keys()))]


if __name__ == "__main__":
    from pprint import pprint

    pprint(FEELINGS_TO_EMOTION_MAP)
    pprint(FEELINGS)