"""Tests for emotion_algebra.__main__ — CLI entry point."""
import subprocess
import sys
import pytest


def run(*args):
    """Run the CLI and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "emotion_algebra"] + list(args),
        capture_output=True, text=True,
    )
    return result.returncode, result.stdout, result.stderr


class TestCLIInfo:
    def test_info_anger(self):
        rc, out, _ = run("info", "anger")
        assert rc == 0
        assert "anger" in out
        assert "Emotion" in out
        assert "sensitivity" in out

    def test_info_love(self):
        rc, out, _ = run("info", "love")
        assert rc == 0
        assert "love" in out
        assert "Feeling" in out

    def test_info_sensitivity_dimension(self):
        rc, out, _ = run("info", "sensitivity")
        assert rc == 0
        assert "sensitivity" in out
        assert "dimension" in out

    def test_info_unknown_exits_1(self):
        rc, _, err = run("info", "not_a_real_emotion_xyz")
        assert rc == 1
        assert "Unknown" in err

    def test_info_composite_emotion(self):
        rc, out, _ = run("info", "aggressiveness")
        assert rc == 0
        assert "aggressiveness" in out


class TestCLIExpressions:
    def test_add_same_axis(self):
        rc, out, _ = run("rage", "-", "1")
        assert rc == 0
        assert "anger" in out

    def test_sub_int(self):
        rc, out, _ = run("anger", "+", "1")
        assert rc == 0
        assert "rage" in out

    def test_cross_axis_add(self):
        rc, out, _ = run("joy", "+", "trust")
        assert rc == 0
        assert "→" in out

    def test_result_includes_type(self):
        rc, out, _ = run("joy", "+", "1")
        assert rc == 0
        assert "type=" in out

    def test_unknown_left_operand_exits_1(self):
        rc, _, err = run("not_an_emotion", "+", "anger")
        assert rc == 1

    def test_unknown_right_operand_exits_1(self):
        rc, _, err = run("anger", "+", "not_an_emotion")
        assert rc == 1

    def test_single_token_shows_info(self):
        rc, out, _ = run("anger")
        assert rc == 0
        assert "anger" in out


class TestCLISerialization:
    def test_emotion_to_dict(self):
        from emotion_algebra.emotions import get_emotion
        anger = get_emotion("anger")
        d = anger.to_dict()
        assert d["type"] == "emotion"
        assert d["name"] == "anger"

    def test_emotion_from_dict(self):
        from emotion_algebra.plutchik import Emotion
        anger = Emotion.from_dict({"type": "emotion", "name": "anger"})
        assert anger.name == "anger"

    def test_feeling_to_dict(self):
        from emotion_algebra.feelings import get_feeling
        love = get_feeling("love")
        d = love.to_dict()
        assert d["type"] == "feeling"
        assert "emotions" in d

    def test_feeling_from_dict(self):
        from emotion_algebra.feelings import Feeling, get_feeling
        love = get_feeling("love")
        d = love.to_dict()
        love2 = Feeling.from_dict(d)
        assert love2.name == "love"

    def test_composite_to_dict(self):
        from emotion_algebra.composite_emotions import COMPOSITE_EMOTIONS
        agg = COMPOSITE_EMOTIONS["aggressiveness"]
        d = agg.to_dict()
        assert d["type"] == "composite"
        assert "components" in d

    def test_composite_from_dict(self):
        from emotion_algebra.composite_emotions import CompositeEmotion, COMPOSITE_EMOTIONS
        agg = COMPOSITE_EMOTIONS["aggressiveness"]
        d = agg.to_dict()
        agg2 = CompositeEmotion.from_dict(d)
        assert agg2.name == "aggressiveness"


# ---------------------------------------------------------------------------
# Direct function tests (subprocess doesn't count toward coverage)
# ---------------------------------------------------------------------------

class TestCLIFunctions:
    def test_show_info_anger(self, capsys):
        from emotion_algebra.__main__ import _show_info
        _show_info("anger")
        out = capsys.readouterr().out
        assert "anger" in out
        assert "Emotion" in out

    def test_show_info_love(self, capsys):
        from emotion_algebra.__main__ import _show_info
        _show_info("love")
        out = capsys.readouterr().out
        assert "Feeling" in out

    def test_show_info_sensitivity(self, capsys):
        from emotion_algebra.__main__ import _show_info
        _show_info("sensitivity")
        out = capsys.readouterr().out
        assert "dimension" in out

    def test_show_info_aggressiveness(self, capsys):
        from emotion_algebra.__main__ import _show_info
        _show_info("aggressiveness")
        out = capsys.readouterr().out
        assert "aggressiveness" in out

    def test_show_info_unknown_exits(self):
        from emotion_algebra.__main__ import _show_info
        with pytest.raises(SystemExit) as exc:
            _show_info("xyzzy_not_real")
        assert exc.value.code == 1

    def test_eval_expr_add(self, capsys):
        from emotion_algebra.__main__ import _eval_expr
        _eval_expr(["anger", "+", "1"])
        out = capsys.readouterr().out
        assert "rage" in out

    def test_eval_expr_sub(self, capsys):
        from emotion_algebra.__main__ import _eval_expr
        _eval_expr(["rage", "-", "1"])
        out = capsys.readouterr().out
        assert "anger" in out

    def test_eval_expr_cross_axis(self, capsys):
        from emotion_algebra.__main__ import _eval_expr
        _eval_expr(["joy", "+", "trust"])
        out = capsys.readouterr().out
        assert "→" in out

    def test_eval_expr_single_token_shows_info(self, capsys):
        from emotion_algebra.__main__ import _eval_expr
        _eval_expr(["anger"])
        out = capsys.readouterr().out
        assert "anger" in out

    def test_eval_expr_unknown_left_exits(self):
        from emotion_algebra.__main__ import _eval_expr
        with pytest.raises(SystemExit):
            _eval_expr(["not_real", "+", "anger"])

    def test_eval_expr_unknown_right_exits(self):
        from emotion_algebra.__main__ import _eval_expr
        with pytest.raises(SystemExit):
            _eval_expr(["anger", "+", "not_real"])

    def test_print_result_none(self, capsys):
        from emotion_algebra.__main__ import _print_result
        _print_result(None, "x + y")
        out = capsys.readouterr().out
        assert "no result" in out

    def test_print_result_notimplemented(self, capsys):
        from emotion_algebra.__main__ import _print_result
        _print_result(NotImplemented, "x + y")
        out = capsys.readouterr().out
        assert "no result" in out

    def test_main_info(self, capsys, monkeypatch):
        import sys
        from emotion_algebra.__main__ import main
        monkeypatch.setattr(sys, "argv", ["emotion_algebra", "info", "anger"])
        main()
        out = capsys.readouterr().out
        assert "anger" in out

    def test_main_expr(self, capsys, monkeypatch):
        import sys
        from emotion_algebra.__main__ import main
        monkeypatch.setattr(sys, "argv", ["emotion_algebra", "rage", "-", "1"])
        main()
        out = capsys.readouterr().out
        assert "anger" in out
