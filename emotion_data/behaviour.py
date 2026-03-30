"""Plutchik's behavioural reactions — maps emotions to adaptive behaviours.

Classes
-------
Behaviour
    An adaptive survival behaviour (e.g. protection, destruction).
BehavioralReaction
    A cognitive-appraisal reaction that maps a trigger to an emotion and behaviour.
"""
from __future__ import annotations

from typing import List, Optional

from emotion_data.emotions import EMOTIONS

BEHAVIOUR_NAMES = {
    "protection": {
        "purpose": "Withdrawal, retreat",
        "activated_by": ["fear", "terror"]
    },
    "destruction": {
            "purpose": "Elimination of barrier to the satisfaction of needs",
            "activated_by": ["anger", "rage"]
        },
    "incorporation": {
            "purpose": "Ingesting nourishment",
            "activated_by": ["acceptance"]
        },
    "rejection": {
            "purpose": "Riddance response to harmful material",
            "activated_by": ["disgust"]
        },
    "reproduction": {
            "purpose": "Approach, contract, genetic exchanges",
            "activated_by": ["joy", "pleasure"]
        },
    "reintegration": {
            "purpose": "Reaction to loss of nutrient product",
            "activated_by": ["sadness", "grief"]
        },
    "exploration": {
            "purpose": "Investigating an environment",
            "activated_by": ["curiosity", "play"]
        },
    "orientation": {
            "purpose": "Reaction to contact with unfamiliar object",
            "activated_by": ["surprise"]
        }
}

REACTION_NAMES = {
    "retain or repeat": {
        "function": "gain resources",
        "cognite appraisal": "possess",
        "trigger": "gain of value",
        "base_emotion": "serenity",
        "behaviour": "incorporation"
    },
    "groom": {
        "function": "mutual support",
        "cognite appraisal": "friend",
        "trigger": "member of one's group",
        "base_emotion": "acceptance",
        "behaviour": "reproduction"
    },
    "escape": {
        "function": "safety",
        "cognite appraisal": "danger",
        "trigger": "threat",
        "base_emotion": "apprehension",
        "behaviour": "protection"
    },
    "stop": {
        "function": "gain time",
        "cognite appraisal": "orient self",
        "trigger": "unexpected event",
        "base_emotion": "distraction",
        "behaviour": "orientation"
    },
    "cry": {
        "function": "reattach to lost object",
        "cognite appraisal": "abandonment",
        "trigger": "loss of value",
        "base_emotion": "pensiveness",
        "behaviour": "reintegration"
    },
    "vomit": {
        "function": "eject poison",
        "cognite appraisal": "poison",
        "trigger": "unpalatable object",
        "base_emotion": "boredom",
        "behaviour": "rejection"
    },
    "attack": {
        "function": "destroy obstacle",
        "cognite appraisal": "enemy",
        "trigger": "obstacle",
        "base_emotion": "annoyance",
        "behaviour": "destruction"
    },
    "map": {
        "function": "knowledge of territory",
        "cognite appraisal": "examine",
        "trigger": "new territory",
        "base_emotion": "interest",
        "behaviour": "exploration"
    }
}


class Behaviour(object):
    """An adaptive survival behaviour triggered by one or more emotions.

    Parameters
    ----------
    name:
        Behaviour identifier (e.g. ``"protection"``, ``"destruction"``).
    purpose:
        Plain-language description of the adaptive purpose.
    """

    def __init__(self, name: str, purpose: str = "") -> None:
        self.name: str = name
        self.purpose: str = purpose
        self.activated_by: list = []

    def __repr__(self):
        return "BehaviourObject:" + self.name


def _get_behaviours():
    bucket = {}
    for behaviour in BEHAVIOUR_NAMES:
        data = BEHAVIOUR_NAMES[behaviour]
        b = Behaviour(behaviour)
        b.purpose = data["purpose"]
        for emo in data["activated_by"]:
            e = EMOTIONS.get(emo)
            if e is not None:
                b.activated_by.append(e)
        bucket[behaviour] = b
    return bucket


BEHAVIOURS = _get_behaviours()


class BehavioralReaction(object):
    """A cognitive-appraisal reaction linking a trigger to an emotion and behaviour.

    Parameters
    ----------
    name:
        Reaction identifier (e.g. ``"escape"``, ``"attack"``).
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.function: str = ""
        self.cognite_appraisal: str = ""
        self.trigger: str = ""
        self.base_emotion: Optional[object] = None  # Emotion object
        self.behaviour: Optional[Behaviour] = None

    def from_data(self, data: Optional[dict] = None) -> None:
        data = data or {}
        self.name = data.get("name") or self.name
        self.function = data.get("function", "")
        self.cognite_appraisal = data.get("cognite appraisal", "")
        self.trigger = data.get("trigger", "")
        self.base_emotion = EMOTIONS.get(data.get("base_emotion", ""))
        behaviour_key = data.get("behaviour")
        self.behaviour = BEHAVIOURS[behaviour_key] if behaviour_key else self.behaviour

    def __repr__(self):
        return "BehavioralReactionObject:" + self.name


def _get_reactions():
    bucket = {}
    bucket2 = {}
    for reaction in REACTION_NAMES:
        data = REACTION_NAMES[reaction]

        r = BehavioralReaction(reaction)
        r.from_data(data)
        bucket[r.name] = r
        bucket2[r.name] = r.base_emotion

    return bucket, bucket2


REACTIONS, REACTION_TO_EMOTION_MAP = _get_reactions()


if __name__ == "__main__":
    from pprint import pprint

    pprint(BEHAVIOURS)
    pprint(REACTIONS)
    pprint(REACTION_TO_EMOTION_MAP)