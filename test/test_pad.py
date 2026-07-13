"""Tests for the PAD/VAD interop layer."""
import pytest

from emotion_algebra.emotions import get_emotion
from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.pad import PAD, from_pad, pad_distance, to_pad


class TestPAD:
    def test_valence_aliases_pleasure(self):
        p = PAD(0.5, 0.2, -0.1)
        assert p.valence == p.pleasure == 0.5

    def test_is_a_tuple(self):
        assert tuple(PAD(0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3)


class TestToPad:
    @pytest.mark.parametrize("name", ["ecstasy", "joy", "serenity"])
    def test_joy_family_is_pleasant(self, name):
        assert to_pad(get_emotion(name)).pleasure > 0

    @pytest.mark.parametrize("name", ["grief", "loathing", "terror"])
    def test_negative_family_is_unpleasant(self, name):
        assert to_pad(get_emotion(name)).pleasure < 0

    def test_anger_is_dominant_fear_is_submissive(self):
        # The reason PAD needs a third axis at all: anger and fear are close in
        # pleasure and arousal and separate almost entirely on dominance.
        assert to_pad(get_emotion("rage")).dominance > 0
        assert to_pad(get_emotion("terror")).dominance < 0

    def test_all_components_in_range(self):
        from emotion_algebra.emotions import EMOTIONS

        for emotion in EMOTIONS.values():
            p = to_pad(emotion)
            assert -1.0 <= p.pleasure <= 1.0
            assert 0.0 <= p.arousal <= 1.0
            assert -1.0 <= p.dominance <= 1.0

    def test_arousal_grows_with_intensity(self):
        assert to_pad(get_emotion("ecstasy")).arousal > to_pad(get_emotion("serenity")).arousal

    def test_neutral_is_the_origin(self):
        p = to_pad(FloatEmotion(0, 0, 0, 0))
        assert p == PAD(0.0, 0.0, 0.0)


class TestFromPad:
    def test_returns_a_float_emotion(self):
        assert isinstance(from_pad(0.5, 0.5, 0.5), FloatEmotion)

    def test_origin_lifts_to_neutral(self):
        assert list(from_pad(0.0, 0.0, 0.0).as_array) == [0.0, 0.0, 0.0, 0.0]

    def test_dominance_drives_sensitivity(self):
        assert from_pad(0.0, 0.5, 0.9).sensitivity > 0
        assert from_pad(0.0, 0.5, -0.9).sensitivity < 0

    @pytest.mark.parametrize(
        "args", [(-1.5, 0, 0), (0, -0.1, 0), (0, 1.1, 0), (0, 0, 2.0), (1.5, 0, 0)]
    )
    def test_out_of_range_rejected(self, args):
        with pytest.raises(ValueError):
            from_pad(*args)

    def test_is_an_exact_right_inverse_on_pleasure_and_dominance(self):
        # Inside the reachable region the lift inverts to_pad exactly on both
        # pleasure and dominance — not approximately.
        for pleasure, arousal, dominance in [
            (0.6, 0.7, 0.4),
            (0.2, 0.3, -0.1),
            (0.0, 0.0, 0.0),
            (-0.3, 0.4, 0.2),
        ]:
            back = to_pad(from_pad(pleasure, arousal, dominance))
            assert back.pleasure == pytest.approx(pleasure, abs=1e-9)
            assert back.dominance == pytest.approx(dominance, abs=1e-9)

    def test_arousal_is_a_floor_not_an_equality(self):
        # to_pad reads arousal as a max, which is not invertible; from_pad
        # therefore treats it as an activation floor.
        for pleasure, arousal, dominance in [(0.9, 0.1, 0.1), (-0.8, 0.2, -0.3)]:
            assert to_pad(from_pad(pleasure, arousal, dominance)).arousal >= arousal

    def test_unreachable_targets_saturate_rather_than_raise(self):
        # "maximally pleasant, no arousal" has no pre-image in the bounded
        # Hourglass cube; it must clamp, not explode.
        fe = from_pad(1.0, 0.0, 0.0)
        assert all(-3.0 <= v <= 3.0 for v in fe.as_array)


class TestPadDistance:
    def test_is_zero_to_itself(self):
        assert pad_distance(get_emotion("joy"), get_emotion("joy")) == pytest.approx(0.0)

    def test_is_symmetric(self):
        a, b = get_emotion("joy"), get_emotion("grief")
        assert pad_distance(a, b) == pytest.approx(pad_distance(b, a))

    def test_separates_anger_from_fear(self):
        # They are close in Hourglass space but far apart in PAD — that is the
        # whole point of the dominance axis.
        assert pad_distance(get_emotion("rage"), get_emotion("terror")) > 0.5
