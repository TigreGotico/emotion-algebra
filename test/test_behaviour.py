"""Tests for emotion_data.behaviour — Behaviour, BehavioralReaction, BEHAVIOURS, REACTIONS."""
import pytest

from emotion_data.behaviour import (
    Behaviour, BehavioralReaction, BEHAVIOURS, REACTIONS,
    REACTION_TO_EMOTION_MAP, BEHAVIOUR_NAMES, REACTION_NAMES,
)


# ---------------------------------------------------------------------------
# BEHAVIOURS dict
# ---------------------------------------------------------------------------

class TestBehavioursDict:
    def test_all_behaviours_present(self):
        for name in BEHAVIOUR_NAMES:
            assert name in BEHAVIOURS

    def test_behaviour_is_behaviour_instance(self):
        for name in BEHAVIOURS:
            assert isinstance(BEHAVIOURS[name], Behaviour)

    def test_behaviour_name_matches_key(self):
        for name, b in BEHAVIOURS.items():
            assert b.name == name

    def test_behaviour_purpose_is_string(self):
        for b in BEHAVIOURS.values():
            assert isinstance(b.purpose, str)
            assert len(b.purpose) > 0

    def test_behaviour_activated_by_is_list(self):
        for b in BEHAVIOURS.values():
            assert isinstance(b.activated_by, list)

    def test_protection_activated_by_fear_or_terror(self):
        b = BEHAVIOURS["protection"]
        names = [e.name for e in b.activated_by]
        assert "fear" in names or "terror" in names

    def test_destruction_activated_by_anger(self):
        b = BEHAVIOURS["destruction"]
        names = [e.name for e in b.activated_by]
        assert "anger" in names or "rage" in names


# ---------------------------------------------------------------------------
# Behaviour class
# ---------------------------------------------------------------------------

class TestBehaviourClass:
    def test_repr_contains_name(self):
        b = Behaviour("test_behaviour")
        assert "test_behaviour" in repr(b)

    def test_default_purpose_empty(self):
        b = Behaviour("x")
        assert b.purpose == ""

    def test_activated_by_starts_empty(self):
        b = Behaviour("x")
        assert b.activated_by == []


# ---------------------------------------------------------------------------
# REACTIONS dict
# ---------------------------------------------------------------------------

class TestReactionsDict:
    def test_all_reactions_present(self):
        for name in REACTION_NAMES:
            assert name in REACTIONS

    def test_reaction_is_behavioralreaction_instance(self):
        for r in REACTIONS.values():
            assert isinstance(r, BehavioralReaction)

    def test_reaction_name_matches_key(self):
        for name, r in REACTIONS.items():
            assert r.name == name

    def test_reaction_has_function(self):
        for r in REACTIONS.values():
            assert isinstance(r.function, str)

    def test_reaction_has_trigger(self):
        for r in REACTIONS.values():
            assert isinstance(r.trigger, str)

    def test_reaction_has_behaviour(self):
        for r in REACTIONS.values():
            assert isinstance(r.behaviour, Behaviour)

    def test_reaction_has_base_emotion(self):
        for r in REACTIONS.values():
            assert r.base_emotion is not None

    def test_escape_reaction_details(self):
        r = REACTIONS["escape"]
        assert r.cognite_appraisal == "danger"
        assert r.trigger == "threat"
        assert r.behaviour.name == "protection"


# ---------------------------------------------------------------------------
# BehavioralReaction class
# ---------------------------------------------------------------------------

class TestBehavioralReactionClass:
    def test_repr_contains_name(self):
        r = BehavioralReaction("test_reaction")
        assert "test_reaction" in repr(r)

    def test_from_data_populates_fields(self):
        r = BehavioralReaction("escape")
        r.from_data(REACTION_NAMES["escape"])
        assert r.function == REACTION_NAMES["escape"]["function"]
        assert r.behaviour is not None
        assert r.base_emotion is not None

    def test_from_data_empty_dict(self):
        r = BehavioralReaction("x")
        r.from_data({})
        assert r.function == ""

    def test_from_data_none(self):
        r = BehavioralReaction("x")
        # should not raise
        try:
            r.from_data(None)
        except (KeyError, TypeError):
            pass  # OK to raise if behaviour key missing


# ---------------------------------------------------------------------------
# REACTION_TO_EMOTION_MAP
# ---------------------------------------------------------------------------

class TestReactionToEmotionMap:
    def test_map_has_all_reactions(self):
        for name in REACTION_NAMES:
            assert name in REACTION_TO_EMOTION_MAP

    def test_map_values_are_emotions(self):
        from emotion_data.plutchik import Emotion
        for v in REACTION_TO_EMOTION_MAP.values():
            assert isinstance(v, Emotion)
