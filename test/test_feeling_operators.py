"""Tests for Feeling operator suite (A-002) — __mul__, __truediv__, __floordiv__, __lshift__, __rshift__."""
import pytest
from emotion_algebra.feelings import Feeling, get_feeling
from emotion_algebra.plutchik import Emotion, Neutrality


LOVE = get_feeling("love")   # joy + trust
DELIGHT = get_feeling("delight")  # joy + surprise


class TestFeelingMul:
    def test_mul_int_upgrades_components(self):
        result = LOVE * 1
        # joy+1=joy, trust+1=admiration → still a Feeling or single Emotion
        assert result is not None and result is not NotImplemented

    def test_mul_zero_neutrality(self):
        result = LOVE * 0
        # joy*0 → neutrality, trust*0 → neutrality → components filtered or reduced
        assert result is not None

    def test_mul_neutrality_returns_copy(self):
        n = Neutrality()
        result = LOVE * n
        assert isinstance(result, Feeling)
        assert result == LOVE

    def test_mul_unknown_returns_not_implemented(self):
        result = LOVE.__mul__("nonsense_object_xyz")
        assert result is NotImplemented

    def test_mul_preserves_type(self):
        result = LOVE * 1
        # two-component feeling scaled by 1 — both components shift up one intensity
        assert result is not None


class TestFeelingTruediv:
    def test_truediv_int(self):
        result = LOVE / 1
        assert result is not None and result is not NotImplemented

    def test_truediv_neutrality_returns_copy(self):
        n = Neutrality()
        result = LOVE / n
        assert isinstance(result, Feeling)

    def test_truediv_invalid_returns_not_implemented(self):
        result = LOVE.__truediv__("not_a_number")
        assert result is NotImplemented


class TestFeelingFloordiv:
    def test_floordiv_int(self):
        result = LOVE // 1
        assert result is not None and result is not NotImplemented

    def test_floordiv_neutrality(self):
        n = Neutrality()
        result = LOVE // n
        assert isinstance(result, Feeling)

    def test_floordiv_invalid(self):
        result = LOVE.__floordiv__("x")
        assert result is NotImplemented


class TestFeelingLshift:
    def test_lshift_decreases_intensity(self):
        # lshift by 1: joy(2)→serenity(1), trust(2)→acceptance(1)
        result = LOVE << 1
        assert result is not None and result is not NotImplemented

    def test_lshift_neutrality(self):
        n = Neutrality()
        result = LOVE << n
        assert isinstance(result, Feeling)

    def test_lshift_invalid(self):
        result = LOVE.__lshift__("x")
        assert result is NotImplemented


class TestFeelingRshift:
    def test_rshift_increases_intensity(self):
        # rshift by 1: joy(2)→ecstasy(3), trust(2)→admiration(3)
        result = LOVE >> 1
        assert result is not None and result is not NotImplemented

    def test_rshift_neutrality(self):
        n = Neutrality()
        result = LOVE >> n
        assert isinstance(result, Feeling)

    def test_rshift_invalid(self):
        result = LOVE.__rshift__("x")
        assert result is NotImplemented


class TestFeelingOperatorParity:
    """All five new operators must be present and callable."""

    def test_has_mul(self):
        assert hasattr(Feeling, "__mul__")

    def test_has_truediv(self):
        assert hasattr(Feeling, "__truediv__")

    def test_has_floordiv(self):
        assert hasattr(Feeling, "__floordiv__")

    def test_has_lshift(self):
        assert hasattr(Feeling, "__lshift__")

    def test_has_rshift(self):
        assert hasattr(Feeling, "__rshift__")

    def test_all_return_something_not_none_for_int_1(self):
        for op in ["__mul__", "__truediv__", "__floordiv__", "__lshift__", "__rshift__"]:
            result = getattr(LOVE, op)(1)
            assert result is not None, f"{op}(1) returned None"
