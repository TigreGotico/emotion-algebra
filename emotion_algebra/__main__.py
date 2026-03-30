"""CLI entry point — ``python -m emotion_algebra`` — v1.4.

Usage
-----
::

    # Show info about a named emotion, feeling, or dimension
    python -m emotion_algebra info anger
    python -m emotion_algebra info love
    python -m emotion_algebra info sensitivity

    # Evaluate a binary expression
    python -m emotion_algebra "joy + trust"
    python -m emotion_algebra rage - 1
    python -m emotion_algebra anger + fear

    # Interactive REPL with emotion_algebra pre-imported
    python -m emotion_algebra
"""
from __future__ import annotations

import re
import sys


# ---------------------------------------------------------------------------
# Info display
# ---------------------------------------------------------------------------

def _show_info(name: str) -> None:
    from emotion_algebra.emotions import get_emotion, get_dimension
    from emotion_algebra.feelings import get_feeling

    obj = get_emotion(name) or get_feeling(name) or get_dimension(name)
    if obj is None:
        print(f"Unknown: {name!r}", file=sys.stderr)
        sys.exit(1)

    from emotion_algebra.plutchik import Emotion, EmotionalDimension
    from emotion_algebra.feelings import Feeling
    from emotion_algebra.composite_emotions import CompositeEmotion

    print(f"\n{obj.name}")
    print("─" * max(len(obj.name), 1))

    if isinstance(obj, EmotionalDimension):
        print(f"  type      : dimension")
        print(f"  axis      : {obj.axis}")
        print(f"  valence   : {obj.valence:+d}")
        print(f"  kind      : {obj.kind}")
        print(f"  emotions  : {obj.basic_opposite.name} ←0→ {obj.basic_emotion.name}")
    elif isinstance(obj, CompositeEmotion):
        print(f"  type      : CompositeEmotion")
        print(f"  flow      : {obj.emotional_flow:+d}")
        print(f"  valence   : {obj.valence:+d}")
        print(f"  arousal   : {obj.arousal}")
        print(f"  affect    : {obj.type}")
        if obj.components:
            print(f"  components: {', '.join(c.name for c in obj.components)}")
    elif isinstance(obj, Emotion):
        print(f"  type      : Emotion")
        if obj._dimension:
            print(f"  axis      : {obj._dimension.axis}")
        print(f"  flow      : {obj.emotional_flow:+d}")
        print(f"  valence   : {obj.valence:+d}")
        print(f"  arousal   : {obj.arousal}")
        print(f"  affect    : {obj.type}")
        print(f"  opposite  : {obj.opposite_emotion.name}")
        if obj.base_emotion.name != obj.name:
            print(f"  base      : {obj.base_emotion.name}")
        if obj.intensity_offset:
            print(f"  intensity : {obj.intensity} (offset {obj.intensity_offset:+d})")
    elif isinstance(obj, Feeling):
        print(f"  type      : Feeling")
        print(f"  valence   : {obj.valence:+d}")
        print(f"  arousal   : {obj.arousal}")
        print(f"  affect    : {obj.type}")
        print(f"  components: {', '.join(e.name for e in obj.emotions)}")
    print()


# ---------------------------------------------------------------------------
# Expression evaluation
# ---------------------------------------------------------------------------

_OP_RE = re.compile(r"\s*(//|>>|<<|[+\-*])\s*")

_OPS = {
    "+":  lambda a, b: a + b,
    "-":  lambda a, b: a - b,
    "*":  lambda a, b: a * b,
    "//": lambda a, b: a // b,
    ">>": lambda a, b: a >> b,
    "<<": lambda a, b: a << b,
}


def _resolve(token: str):
    """Resolve a token to an emotion, or an int, or fail."""
    from emotion_algebra.emotions import get_emotion
    try:
        return int(token)
    except ValueError:
        pass
    emo = get_emotion(token)
    if emo is None:
        print(f"Unknown emotion or operand: {token!r}", file=sys.stderr)
        sys.exit(1)
    return emo


def _eval_expr(tokens: list) -> None:
    expr = " ".join(tokens)
    parts = _OP_RE.split(expr, maxsplit=1)

    if len(parts) == 1:
        _show_info(parts[0].strip())
        return

    left_str, op, right_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
    left = _resolve(left_str)
    right = _resolve(right_str)

    if op not in _OPS:
        print(f"Unsupported operator: {op!r}", file=sys.stderr)
        sys.exit(1)

    result = _OPS[op](left, right)
    _print_result(result, expr)


def _print_result(result, expr: str) -> None:
    from emotion_algebra.base import EmotionBase
    if result is None or result is NotImplemented:
        print(f"{expr}  →  (no result)")
        return
    name = getattr(result, "name", str(result))
    rtype = type(result).__name__
    extra = ""
    if isinstance(result, EmotionBase):
        extra = (
            f"  (valence={result.valence:+}, "
            f"arousal={result.arousal}, "
            f"type={result.type!r})"
        )
    print(f"{expr}  →  {name}  [{rtype}]{extra}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def _repl() -> None:
    import code
    from emotion_algebra.emotions import EMOTIONS, get_emotion, get_dimension
    from emotion_algebra.feelings import FEELINGS, get_feeling
    from emotion_algebra.plutchik import Neutrality

    banner = (
        "emotion-algebra interactive console\n"
        "  All 24 emotions available by name (joy, anger, fear, …)\n"
        "  Try: joy + trust   |   rage - 1   |   anger * fear\n"
        "  Type exit() or Ctrl-D to quit.\n"
    )
    local_ns: dict = {
        name: emo for name, emo in EMOTIONS.items()
    }
    local_ns.update({
        "EMOTIONS": EMOTIONS,
        "FEELINGS": FEELINGS,
        "get_emotion": get_emotion,
        "get_feeling": get_feeling,
        "get_dimension": get_dimension,
        "Neutrality": Neutrality,
    })
    code.interact(banner=banner, local=local_ns, exitmsg="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]

    if not args:
        _repl()
        return

    if args[0] == "info" and len(args) >= 2:
        _show_info(" ".join(args[1:]))
        return

    # Emoji shortcut: single token where every non-whitespace char is a mapped emoji
    if len(args) == 1:
        from emotion_algebra.emoji import score_emojis, EMOJI_EMOTION_MAP
        candidate = args[0]
        chars = [ch for ch in candidate if ch.strip()]
        if chars and all(ch in EMOJI_EMOTION_MAP for ch in chars):
            state = score_emojis(candidate)
            result = state.dominant()
            _print_result(result, candidate)
            return

    _eval_expr(args)


if __name__ == "__main__":
    main()
