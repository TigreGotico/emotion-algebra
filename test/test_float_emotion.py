"""Tests for emotion_algebra.float_emotion — FloatEmotion."""
import numpy as np
import pytest

from emotion_algebra.float_emotion import FloatEmotion
from emotion_algebra.base import EmotionBase
from emotion_algebra.emotions import get_emotion
from emotion_algebra.plutchik import Neutrality


@pytest.fixture
def fe_anger():
    """FloatEmotion at sensitivity=2.0 (≈ anger)."""
    return FloatEmotion(sensitivity=2.0)

@pytest.fixture
def fe_joy():
    """FloatEmotion at pleasantness=2.0 (≈ joy)."""
    return FloatEmotion(pleasantness=2.0)


class TestFloatEmotionConstruction:
    def test_default_is_zero(self):
        fe = FloatEmotion()
        assert np.allclose(fe.as_array, 0)

    def test_set_axis_values(self):
        fe = FloatEmotion(sensitivity=1.5, pleasantness=-0.5)
        assert fe.as_array[0] == pytest.approx(1.5)
        assert fe.as_array[2] == pytest.approx(-0.5)

    def test_explicit_name(self):
        fe = FloatEmotion(name="custom")
        assert fe.name == "custom"

    def test_derived_name_has_tilde(self, fe_anger):
        assert fe_anger.name.startswith("~")

    def test_derived_name_closest_emotion(self, fe_anger):
        assert "anger" in fe_anger.name or "annoyance" in fe_anger.name or "rage" in fe_anger.name


class TestFloatEmotionProperties:
    def test_valence_is_pleasantness(self):
        fe = FloatEmotion(sensitivity=3.0, pleasantness=1.5)
        assert fe.valence == pytest.approx(1.5)

    def test_arousal_is_max_abs(self):
        fe = FloatEmotion(sensitivity=3.0, pleasantness=-1.5)
        assert fe.arousal == pytest.approx(3.0)

    def test_emotional_flow_is_sum(self):
        fe = FloatEmotion(sensitivity=1.0, attention=2.0, pleasantness=-1.0, aptitude=0.5)
        assert fe.emotional_flow == pytest.approx(2.5)

    def test_type_is_string(self, fe_joy):
        assert isinstance(fe_joy.type, str)

    def test_type_excited_positive(self, fe_joy):
        assert fe_joy.type == "excited positive"

    def test_as_array_shape(self, fe_anger):
        assert fe_anger.as_array.shape == (4,)

    def test_as_array_is_copy(self, fe_anger):
        arr = fe_anger.as_array
        arr[0] = 999.0
        assert fe_anger.as_array[0] != 999.0

    def test_as_matrix_shape(self, fe_anger):
        assert fe_anger.as_matrix.shape == (2, 2)

    def test_emotion_vector_length(self, fe_anger):
        assert len(fe_anger.emotion_vector) == 4

    def test_is_emotion_base(self, fe_anger):
        assert isinstance(fe_anger, EmotionBase)


class TestFloatEmotionArithmetic:
    def test_add_float_emotion(self, fe_anger, fe_joy):
        result = fe_anger + fe_joy
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(2.0)
        assert result.as_array[2] == pytest.approx(2.0)

    def test_add_int(self, fe_anger):
        result = fe_anger + 1
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(3.0)

    def test_add_emotion_base(self, fe_anger):
        joy = get_emotion("joy")
        result = fe_anger + joy
        assert isinstance(result, FloatEmotion)

    def test_sub_float_emotion(self, fe_anger, fe_joy):
        result = fe_anger - fe_joy
        assert isinstance(result, FloatEmotion)

    def test_sub_int(self, fe_anger):
        result = fe_anger - 1.0
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(1.0)

    def test_mul_scalar(self, fe_anger):
        result = fe_anger * 2.0
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(4.0)

    def test_truediv_scalar(self, fe_anger):
        result = fe_anger / 2.0
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(1.0)

    def test_truediv_zero_returns_notimplemented(self, fe_anger):
        assert fe_anger.__truediv__(0) is NotImplemented

    def test_neg(self, fe_anger):
        result = -fe_anger
        assert isinstance(result, FloatEmotion)
        assert result.as_array[0] == pytest.approx(-2.0)

    def test_abs_returns_zero(self, fe_anger):
        result = abs(fe_anger)
        assert isinstance(result, FloatEmotion)
        assert np.allclose(result.as_array, 0)

    def test_add_unknown_returns_notimplemented(self, fe_anger):
        assert fe_anger.__add__("not_valid") is NotImplemented

    def test_mul_unknown_returns_notimplemented(self, fe_anger):
        assert fe_anger.__mul__(fe_anger) is NotImplemented


class TestFloatEmotionConversions:
    def test_int_conversion(self, fe_anger):
        assert isinstance(int(fe_anger), int)

    def test_float_conversion(self, fe_anger):
        assert isinstance(float(fe_anger), float)

    def test_repr(self, fe_anger):
        assert "FloatEmotion" in repr(fe_anger)

    def test_str_is_name(self, fe_anger):
        assert str(fe_anger) == fe_anger.name


class TestFloatEmotionEquality:
    def test_eq_same_vector(self):
        fe1 = FloatEmotion(2.0, 0.0, 0.0, 0.0)
        fe2 = FloatEmotion(2.0, 0.0, 0.0, 0.0)
        assert fe1 == fe2

    def test_ne_different_vector(self, fe_anger, fe_joy):
        assert fe_anger != fe_joy

    def test_eq_non_float_emotion_returns_notimplemented(self, fe_anger):
        assert fe_anger.__eq__("anger") is NotImplemented


class TestFloatEmotionSerialization:
    def test_to_dict_has_type_key(self, fe_anger):
        d = fe_anger.to_dict()
        assert d["type"] == "float_emotion"

    def test_to_dict_has_vector(self, fe_anger):
        d = fe_anger.to_dict()
        assert "vector" in d
        assert len(d["vector"]) == 4

    def test_from_dict_roundtrip(self):
        fe = FloatEmotion(1.5, -0.5, 2.3, 0.1, name="test")
        fe2 = FloatEmotion.from_dict(fe.to_dict())
        assert np.allclose(fe.as_array, fe2.as_array)
        assert fe2._name == "test"


class TestFloatEmotionAlternativeConstructors:
    def test_from_emotion_joy(self):
        joy = get_emotion("joy")
        fe = FloatEmotion.from_emotion(joy)
        assert isinstance(fe, FloatEmotion)
        assert np.allclose(fe.as_array, joy.as_array.astype(float))

    def test_from_embedding_no_projection(self):
        vec = np.array([0.8, 0.1, 0.5, -0.2])
        fe = FloatEmotion.from_embedding(vec)
        assert isinstance(fe, FloatEmotion)
        # Should be normalised to [-3, 3]
        assert np.max(np.abs(fe.as_array)) <= 3.0 + 1e-9

    def test_from_embedding_short_vector(self):
        vec = np.array([1.0, 0.5])
        fe = FloatEmotion.from_embedding(vec)
        assert fe.as_array.shape == (4,)

    def test_from_embedding_with_projection_matrix(self):
        proj = np.eye(4, 8)  # 4×8 projection matrix
        vec = np.ones(8)
        fe = FloatEmotion.from_embedding(vec, projection_matrix=proj)
        assert isinstance(fe, FloatEmotion)
        assert fe.as_array.shape == (4,)

    def test_from_embedding_zero_vector(self):
        vec = np.zeros(4)
        fe = FloatEmotion.from_embedding(vec)
        assert np.allclose(fe.as_array, 0)
