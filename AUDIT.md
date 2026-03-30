# AUDIT.md — emotion_data

Evidence-based register of technical debt, known issues, and security observations.
Updated: 2026-03-30 (Phase 4).

---

## Open Issues

### A-001 — `Feeling.emotional_flow` used by `__int__`/`__float__` but not spec-derived
**File**: `emotion_data/feelings.py:194–199`
**Severity**: Low
**Detail**: `emotional_flow` on `Feeling` is now the signed sum of component flows, consistent
with `CompositeEmotion`. However `__int__` and `__float__` delegate to it, making `int(love)` = 4
(joy +2 + trust +2). This is arithmetically consistent but psychologically opaque. No caller in
the codebase relies on it; it should be documented as "net activation scalar, not a hedonic score".
**Recommendation**: Add a docstring to `__int__`/`__float__` clarifying the interpretation.

### ~~A-002~~ — `CompositeEmotion` inherits `Emotion` (RESOLVED 2026-03-30)
**File**: `emotion_data/composite_emotions.py`
**Resolution**: `EmotionBase` ABC introduced in `emotion_data/base.py`. `CompositeEmotion`,
`Emotion`, and `Feeling` now all independently implement `EmotionBase`. The erroneous
single-axis inherited methods (`is_primary`, `is_secondary`, `is_tertiary`, `intensity_offset`,
`emotion_from_flow`, `parent_emotion`) are no longer present on `CompositeEmotion`.

### A-003 — `emotions.py:80–87` dead `__main__` print block
**File**: `emotion_data/emotions.py:79–87`
**Severity**: Negligible
**Detail**: `if __name__ == "__main__"` block with `print` / `pprint` calls — a legacy debug
script. Not harmful but untested (excluded from coverage at lines 80–87).
**Recommendation**: Remove or convert to a proper `python -m emotion_data.emotions` CLI entry
point in `pyproject.toml`.

### A-004 — `__init__.py` EmotionAnalyzer methods at 82% coverage
**File**: `emotion_data/__init__.py:20,35,40,45,55,60–61,66–67`
**Severity**: Low
**Detail**: The `EmotionAnalyzer` deepmoji/tagging paths (lines 55–67) are unreachable in the
test environment because the optional deps are not installed. The lazy-import guard is correct but
the code paths have zero test coverage.
**Recommendation**: Mock the optional imports in tests to cover the fallback error messages, or
add integration tests under a `[test-all]` extra.

### A-005 — `composite_emotions.py` `__truediv__` / `__floordiv__` return `NotImplemented` for Emotion operands
**File**: `emotion_data/composite_emotions.py:398–420`
**Severity**: Low
**Detail**: `CompositeEmotion / Emotion` and `CompositeEmotion // Emotion` reach a `pass` branch
and return `None` implicitly (lines 399, 414). Callers get `None` instead of `NotImplemented`,
which suppresses the reflected operator and gives a confusing result.
**Recommendation**: Replace `pass` with `return NotImplemented` in both branches.

### A-006 — `Feeling` operator suite not symmetric with `Emotion`
**File**: `emotion_data/feelings.py`
**Severity**: Low
**Detail**: `Feeling` has no `__lshift__`, `__rshift__`, `__floordiv__`, `__truediv__`,
`__mul__`, or `__neg__` that return `Feeling`. Cross-feeling arithmetic produces `CompositeEmotion`
or `Emotion` in ways that are not documented and not tested (coverage miss at lines 295–326).
**Recommendation**: Document the expected return type for each operator on `Feeling`, add tests
for the currently uncovered branches.

### A-007 — `lexicons.py` pandas dependency at 86% coverage
**File**: `emotion_data/lexicons.py:19,25,31,56–58`
**Severity**: Low
**Detail**: CSV-loading paths and `get_word_emotion` error paths uncovered. The pandas dependency
requires the `[lexicon]` extra, so these lines are skipped in a bare install.
**Recommendation**: Same approach as A-004 — mock pandas in tests or add an integration suite.

---

## Resolved Issues (closed in Phases 2–4)

| ID | Description | Fixed |
|----|-------------|-------|
| R-001 | `Emotion.__bool__` returned `numpy.bool_` — TypeError in 3.11+ | Phase 2 |
| R-002 | `_get_behaviours()` silently dropped fear/terror via `if e:` | Phase 2 |
| R-003 | `CompositeEmotion.emotional_flow` was `np.linalg.norm` (always ≥ 0) | Phase 3 |
| R-004 | `copy()` aliased mutable list state in all operator methods | Phase 3 |
| R-005 | `get_emotion` import shadow in `__init__.py` | Phase 3 |
| R-006 | `np.matrix` deprecated in `as_matrix` | Phase 3 |
| R-007 | `Emotion.valence` conflated arousal with hedonics (all axes ±1) | Phase 4 |
| R-008 | `Emotion.type` / `CompositeEmotion.type` used undocumented heuristics | Phase 4 |
| R-009 | `EmotionalDimension.valence` wrong (sensitivity=−1, attention=+1) | Phase 4 |
| R-010 | Cross-axis `+` returned `Feeling` instead of `CompositeEmotion` | Phase 4 |
| R-011 | `Feeling.emotional_flow` used `np.linalg.norm` (always ≥ 0) | Phase 4 |
| R-012 | `Feeling.__float__` returned `numpy.float64` — DeprecationWarning | Phase 4 |

---

## Security

No network I/O, no file-system writes, no exec/eval in the core library.
The optional `tag.py` (ParallelDots API) and `deepmoji.py` (torch model loading) are
not audited here — they are excluded from coverage and not part of the algebraic core.
