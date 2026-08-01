"""Tests for the dual-lineage name taxonomy.

Plutchik dyads (Feelings) and Hourglass compounds (CompositeEmotions) both claim
names like "love" and "optimism".  These lock the sanctioned resolution so it
never depends on import order again.
"""
import pytest

from emotion_algebra.composite_emotions import CompositeEmotion
from emotion_algebra.feelings import Feeling
from emotion_algebra.taxonomy import (
    AMBIGUOUS_NAMES,
    collision_table,
    describe_collision,
    is_ambiguous,
    resolve,
)


class TestCollisions:
    def test_there_are_collisions(self):
        assert AMBIGUOUS_NAMES

    def test_is_ambiguous_agrees_with_the_set(self):
        for name in AMBIGUOUS_NAMES:
            assert is_ambiguous(name)

    def test_unambiguous_names_are_not_flagged(self):
        assert not is_ambiguous("joy")
        assert not is_ambiguous("definitely-not-an-emotion")

    def test_is_ambiguous_is_case_insensitive(self):
        name = sorted(AMBIGUOUS_NAMES)[0]
        assert is_ambiguous(name.upper())

    def test_collision_table_covers_every_collision(self):
        table = collision_table()
        assert {row["name"] for row in table} == set(AMBIGUOUS_NAMES)

    def test_describe_collision_returns_both_lineages(self):
        name = sorted(AMBIGUOUS_NAMES)[0]
        info = describe_collision(name)
        assert info["name"] == name
        assert info["feeling"] is not None
        assert info["composite"] is not None

    def test_describe_collision_is_none_for_clean_names(self):
        assert describe_collision("joy") is None


class TestResolve:
    @pytest.mark.parametrize("name", sorted(AMBIGUOUS_NAMES))
    def test_default_prefers_the_feeling(self, name):
        assert isinstance(resolve(name), Feeling)

    @pytest.mark.parametrize("name", sorted(AMBIGUOUS_NAMES))
    def test_composite_preference_is_honoured(self, name):
        assert isinstance(resolve(name, prefer="composite"), CompositeEmotion)

    def test_resolves_unambiguous_names_regardless_of_preference(self):
        assert resolve("joy").name == "joy"
        assert resolve("joy", prefer="composite").name == "joy"

    def test_unknown_name_returns_none(self):
        assert resolve("definitely-not-an-emotion") is None

    def test_bad_preference_rejected(self):
        with pytest.raises(ValueError):
            resolve("love", prefer="whatever")

    def test_resolution_is_deterministic(self):
        for name in sorted(AMBIGUOUS_NAMES):
            assert resolve(name).name == resolve(name).name
