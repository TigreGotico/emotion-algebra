"""Tests for emotion_algebra.needs — need-to-emotion mapping."""
import pytest

from emotion_algebra.needs import (
    need_deficit_to_emotion,
    need_deficit_to_float_emotion,
    NEED_DEFICIT_EMOTIONS,
    MAXNEEF_DEFICIT_EMOTIONS,
    MURRAY_DEFICIT_EMOTIONS,
    MaxNeefNeed,
    MurrayNeed,
    CIADrive,
    DRIVE_AXES,
    NEED_DRIVES,
)
from emotion_algebra.emotions import get_emotion
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.base import EmotionBase


class TestStrEnumBackportStr:
    """On Python < 3.11, ``StrEnum`` is a hand-rolled backport
    (``class StrEnum(str, Enum)``). Plain ``Enum`` formats ``str(member)``
    as ``"ClassName.MEMBER"`` unless ``__str__`` is overridden — unlike the
    real 3.11+ ``enum.StrEnum``, which returns the value. The module builds
    ``NEED_DEFICIT_EMOTIONS`` via ``str(k)`` for every enum key, so this must
    hold on every supported Python version or every lookup silently misses.
    """

    def test_maxneef_str_equals_value(self):
        for member in MaxNeefNeed:
            assert str(member) == member.value

    def test_murray_str_equals_value(self):
        for member in MurrayNeed:
            assert str(member) == member.value

    def test_cia_drive_str_equals_value(self):
        for member in CIADrive:
            assert str(member) == member.value


class TestNeedDeficitToEmotion:
    def test_all_maxneef_needs_have_emotions(self):
        for need in MAXNEEF_DEFICIT_EMOTIONS:
            emo = need_deficit_to_emotion(need)
            assert emo is not None, f"No emotion for Max-Neef need {need!r}"
            assert isinstance(emo, EmotionBase)

    def test_all_murray_needs_have_emotions(self):
        for need in MURRAY_DEFICIT_EMOTIONS:
            emo = need_deficit_to_emotion(need)
            assert emo is not None, f"No emotion for Murray need {need!r}"

    def test_protection_deficit_is_fear(self):
        assert need_deficit_to_emotion("protection").name == "fear"

    def test_subsistence_deficit_is_sadness(self):
        assert need_deficit_to_emotion("subsistence").name == "sadness"

    def test_freedom_deficit_is_anger(self):
        assert need_deficit_to_emotion("freedom").name == "anger"

    def test_creation_deficit_is_boredom(self):
        assert need_deficit_to_emotion("creation").name == "boredom"

    def test_harm_avoidance_deficit_is_fear(self):
        assert need_deficit_to_emotion("harm_avoidance").name == "fear"

    def test_unknown_need_returns_none(self):
        assert need_deficit_to_emotion("nonexistent") is None


class TestNeedDeficitToFloatEmotion:
    def test_returns_float_emotion(self):
        fe = need_deficit_to_float_emotion("protection")
        assert isinstance(fe, FloatEmotion)

    def test_protection_has_negative_sensitivity(self):
        """Fear → negative sensitivity axis."""
        fe = need_deficit_to_float_emotion("protection")
        assert float(fe.as_array[0]) < 0

    def test_subsistence_has_negative_pleasantness(self):
        """Sadness → negative pleasantness axis."""
        fe = need_deficit_to_float_emotion("subsistence")
        assert float(fe.as_array[2]) < 0

    def test_freedom_has_positive_sensitivity(self):
        """Anger → positive sensitivity axis."""
        fe = need_deficit_to_float_emotion("freedom")
        assert float(fe.as_array[0]) > 0

    def test_unknown_need_returns_none(self):
        assert need_deficit_to_float_emotion("nonexistent") is None

    def test_all_needs_produce_nonzero_vector(self):
        import numpy as np
        for need in NEED_DEFICIT_EMOTIONS:
            fe = need_deficit_to_float_emotion(need)
            assert fe is not None
            assert np.any(fe.as_array != 0), f"Need {need!r} produced zero vector"


class TestCIACoherence:
    """The needs table must obey the CIA buckets its own docstring documents.

    This is the test that keeps the docstring honest: every deficit emotion has
    to land on an Hourglass axis its need's meta-drive is actually allowed to
    move.  Without it the table and the prose drift apart silently.
    """

    def test_every_need_is_filed_under_a_drive(self):
        for need in list(MAXNEEF_DEFICIT_EMOTIONS) + list(MURRAY_DEFICIT_EMOTIONS):
            assert str(need) in NEED_DRIVES, f"{need} has no CIA drive"

    def test_no_orphan_drive_entries(self):
        known = {str(n) for n in MAXNEEF_DEFICIT_EMOTIONS} | {
            str(n) for n in MURRAY_DEFICIT_EMOTIONS
        }
        assert set(NEED_DRIVES) <= known

    def test_every_drive_has_axes(self):
        for drive in CIADrive:
            assert DRIVE_AXES[drive]

    @pytest.mark.parametrize(
        "need,emotion_name",
        sorted(
            ((str(k), v) for k, v in MAXNEEF_DEFICIT_EMOTIONS.items()),
            key=lambda kv: kv[0],
        )
        + sorted(
            ((str(k), v) for k, v in MURRAY_DEFICIT_EMOTIONS.items()),
            key=lambda kv: kv[0],
        ),
    )
    def test_deficit_emotion_lies_on_its_drives_axis(self, need, emotion_name):
        emotion = get_emotion(emotion_name)
        assert emotion is not None, f"{need} maps to unknown emotion {emotion_name!r}"
        allowed = DRIVE_AXES[NEED_DRIVES[need]]
        axis = emotion.dimension.name if emotion.dimension else None
        assert axis in allowed, (
            f"{need} is a {NEED_DRIVES[need]} need, so its deficit emotion must sit "
            f"on one of {sorted(allowed)}; {emotion_name!r} sits on {axis!r}"
        )
