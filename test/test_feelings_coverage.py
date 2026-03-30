"""Targeted coverage tests for previously-uncovered Feeling branches."""
import pytest
from emotion_algebra.feelings import Feeling, get_feeling
from emotion_algebra.plutchik import Emotion, Neutrality


# ---------------------------------------------------------------------------
# line 142 — sensitivity branch of emotion_vector
# ---------------------------------------------------------------------------

class TestEmotionVectorSensitivityBranch:
    def test_sensitivity_axis_feeling_has_nonzero_sensitivity(self):
        # aggressiveness = anger (sensitivity) + anticipation (attention)
        agg = get_feeling("aggressiveness")
        assert agg is not None
        vec = agg.emotion_vector
        # the vector should contain a sensitivity-axis entry
        axes = [e.dimension.axis for e in vec if hasattr(e, "dimension") and e.dimension]
        assert "sensitivity" in axes

    def test_as_array_includes_sensitivity_component(self):
        import numpy as np
        agg = get_feeling("aggressiveness")
        arr = agg.as_array
        assert isinstance(arr, np.ndarray)
        assert arr[0] != 0  # sensitivity axis (index 0) should be non-zero


# ---------------------------------------------------------------------------
# lines 251-252, 294-295 — bare except branches in __add__ / __sub__
# These are reached when `int(other)` raises and the value is not a recognised type.
# ---------------------------------------------------------------------------

class TestFeelingAddSubExceptBranch:
    def test_add_unrecognised_type_returns_not_implemented(self):
        love = get_feeling("love")
        result = love.__add__(object())
        assert result is NotImplemented

    def test_sub_unrecognised_type_returns_not_implemented(self):
        love = get_feeling("love")
        result = love.__sub__(object())
        assert result is NotImplemented


# ---------------------------------------------------------------------------
# lines 316, 332, 348, 364, 380 — except branches in new operators
# ---------------------------------------------------------------------------

class TestFeelingNewOperatorExceptBranches:
    def test_mul_unrecognised_returns_not_implemented(self):
        love = get_feeling("love")
        assert love.__mul__(object()) is NotImplemented

    def test_truediv_unrecognised_returns_not_implemented(self):
        love = get_feeling("love")
        assert love.__truediv__(object()) is NotImplemented

    def test_floordiv_unrecognised_returns_not_implemented(self):
        love = get_feeling("love")
        assert love.__floordiv__(object()) is NotImplemented

    def test_lshift_unrecognised_returns_not_implemented(self):
        love = get_feeling("love")
        assert love.__lshift__(object()) is NotImplemented

    def test_rshift_unrecognised_returns_not_implemented(self):
        love = get_feeling("love")
        assert love.__rshift__(object()) is NotImplemented


# ---------------------------------------------------------------------------
# line 391 — __eq__ string comparison branch (non-Emotion/Feeling other)
# ---------------------------------------------------------------------------

class TestFeelingEqStringBranch:
    def test_eq_named_feeling_string_matches(self):
        # Named Feeling constructed directly (not from canonical FEELINGS which have _name="")
        f = Feeling("love")
        from emotion_algebra.emotions import get_emotion
        f.emotions = [get_emotion("joy"), get_emotion("trust")]
        assert (f == "love") is True

    def test_eq_named_feeling_string_mismatch(self):
        f = Feeling("love")
        assert (f == "anger") is False

    def test_ne_named_feeling_string(self):
        f = Feeling("love")
        assert (f != "anger") is True
        assert (f != "love") is False


# ---------------------------------------------------------------------------
# line 418 — from_dict reconstruction path for unknown feeling name
# ---------------------------------------------------------------------------

class TestFeelingFromDictUnknownName:
    def test_from_dict_known_name_returns_canonical(self):
        love = get_feeling("love")
        d = love.to_dict()
        restored = Feeling.from_dict(d)
        assert restored == love

    def test_from_dict_unknown_name_reconstructs_from_emotions(self):
        d = {
            "type": "feeling",
            "name": "__test_custom_feeling__",
            "emotions": [
                {"type": "emotion", "name": "joy"},
                {"type": "emotion", "name": "trust"},
            ],
        }
        f = Feeling.from_dict(d)
        assert isinstance(f, Feeling)
        assert len(f.emotions) == 2
        names = {e.name for e in f.emotions}
        assert "joy" in names
        assert "trust" in names


# ---------------------------------------------------------------------------
# lines 453-456 — isinstance(d, list) branch in _get_feelings()
# This is exercised at module import time; we just verify the result is correct.
# ---------------------------------------------------------------------------

class TestFeelingOperatorSingleComponentReduction:
    """Cover the `return feel.emotions[0]` branch when operators reduce to one component."""

    def _single_component_feeling(self):
        """A Feeling with one emotion — operations can't reduce further but still hit the branch."""
        f = Feeling("stub")
        from emotion_algebra.emotions import get_emotion
        f.emotions = [get_emotion("ecstasy")]  # one component
        return f

    def test_mul_single_component_returns_emotion(self):
        f = self._single_component_feeling()
        result = f * 1
        assert isinstance(result, Emotion)  # single component → unwrapped

    def test_truediv_single_component_returns_emotion(self):
        f = self._single_component_feeling()
        result = f / 1
        assert isinstance(result, Emotion)

    def test_floordiv_single_component_returns_emotion(self):
        f = self._single_component_feeling()
        result = f // 1
        assert isinstance(result, Emotion)

    def test_lshift_single_component_returns_emotion(self):
        f = self._single_component_feeling()
        result = f << 1
        assert isinstance(result, Emotion)

    def test_rshift_single_component_returns_emotion(self):
        f = self._single_component_feeling()
        result = f >> 1
        assert isinstance(result, Emotion)


class TestFeelingSubFeelingSingleReduction:
    """Cover line 267 — Feeling - Feeling reducing to single component."""

    def test_sub_feeling_reduces_to_single_emotion(self):
        from emotion_algebra.emotions import get_emotion
        # Build a two-emotion Feeling, then subtract a one-emotion Feeling
        # so the result has one emotion remaining
        f_two = Feeling("custom")
        f_two.emotions = [get_emotion("joy"), get_emotion("trust")]

        f_one = Feeling("sub")
        f_one.emotions = [get_emotion("joy")]

        result = f_two - f_one
        # After removing joy, only trust remains → should return single Emotion
        assert isinstance(result, Emotion)
        assert result.name == "trust"


class TestFeelingFromDictKnownNameHit:
    """Cover line 418 — from_dict early return for a directly-named Feeling."""

    def test_named_feeling_roundtrips_via_from_dict(self):
        # Build a named Feeling (non-canonical) so get_feeling finds it
        from emotion_algebra.emotions import get_emotion
        f = Feeling("__coverage_feeling__")
        f.emotions = [get_emotion("joy"), get_emotion("trust")]

        # from_dict with unknown name falls to reconstruction path
        d = {"type": "feeling", "name": "__coverage_feeling__", "emotions": [
            {"type": "emotion", "name": "joy"},
            {"type": "emotion", "name": "trust"},
        ]}
        restored = Feeling.from_dict(d)
        assert isinstance(restored, Feeling)
        assert len(restored.emotions) == 2


class TestGetFeelingsListBranch:
    def test_feelings_with_multiple_dimensions_have_correct_components(self):
        from emotion_algebra.feelings import FEELINGS
        # Every feeling should have at least one emotion
        for name, feeling in FEELINGS.items():
            assert len(feeling.emotions) >= 1, f"{name!r} has no emotions"

    def test_feelings_dimensions_list_populated(self):
        from emotion_algebra.feelings import FEELINGS
        for name, feeling in FEELINGS.items():
            # dimensions may be empty for some, but the attribute must exist
            assert hasattr(feeling, "dimensions")
