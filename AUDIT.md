# AUDIT.md — emotion-algebra

Evidence-based register of technical debt, known issues, and security observations.
Updated: 2026-03-30 (v1.0.0 pre-release audit).

---

## Open Issues

### A-001 — `Emotion.kind` and `CompositeEmotion.kind` are eye-balled, not spec-derived
**Files**: `emotion_algebra/plutchik.py:93,252`, `emotion_algebra/composite_emotions.py:469–518`
**Severity**: Low
**Detail**: Both `kind` properties contain `# TODO science this` comments. Category assignments
(`"event related"`, `"social"`, `"cathected"`, etc.) are manually curated from Plutchik's prose, not
derived from Hourglass axes. Two entries (`"gloat"`, `"frivolity"`) are explicitly unclassified.
**Recommendation**: Accept for v1.0; document in API reference as approximate.

### A-002 — `Feeling` operator suite has coverage gaps
**File**: `emotion_algebra/feelings.py:142,251–252,267,294–295,303,341–344`
**Severity**: Low
**Detail**: `Feeling` has no `__lshift__`, `__rshift__`, `__floordiv__`, `__truediv__`, or `__mul__`.
The sensitivity branch of `emotion_vector` (line 142) is unreachable in current test data.
**Recommendation**: Acceptable for v1.0. Document unsupported operators.

### A-003 — `EmotionAnalyzer` optional-dep paths are untestable without extras
**File**: `emotion_algebra/__init__.py:21,36,41,46,56,61–62,67–68`
**Severity**: Low
**Detail**: `tag_emotions`/`tag_emojis` require `torch`+`torchMoji`; lexicon methods require `pandas`.
Lazy-import guard is correct; zero coverage of these paths in the default test run.
**Recommendation**: Add `test_optional.py` under a `[test-all]` extra, skipped by default.

### A-004 — `CompositeEmotion.__eq__` same-dimension branch is dead code
**File**: `emotion_algebra/composite_emotions.py:425,432`
**Severity**: Negligible
**Detail**: `CompositeEmotion.__eq__(Emotion)` checks `if other._dimension == self.dimension`. A
`CompositeEmotion` holds a `CompositeDimension`, never a plain `EmotionalDimension`, so the branch
can never be `True`. Always falls through to name comparison.
**Recommendation**: Remove the branch; simplify `__eq__`/`__ne__` to name comparison only.

### A-005 — `emotions.py` type map comment is stale
**File**: `emotion_algebra/emotions.py:55`
**Severity**: Negligible
**Detail**: `# TODO emotion type map` was never implemented.
**Recommendation**: Remove the comment or implement `TYPE_TO_EMOTION_MAP` grouped by Russell type.

### A-006 — `EmotionBase.__int__`/`__float__` semantics are undocumented
**File**: `emotion_algebra/base.py`
**Severity**: Low
**Detail**: `int(love)` returns 4 (joy flow 2 + trust flow 2) — net activation scalar, not hedonic
score. No docstring explains this.
**Recommendation**: Add a clarifying docstring to `EmotionBase.__int__`.

---

## Resolved Issues

| ID | Description | Fixed |
|----|-------------|-------|
| R-001 | `Emotion.__bool__` returned `numpy.bool_` — TypeError in 3.11+ | Phase 2 |
| R-002 | `_get_behaviours()` silently dropped negative-flow emotions via `if e:` | Phase 2 |
| R-003 | `CompositeEmotion.emotional_flow` was `np.linalg.norm` (always ≥ 0) | Phase 3 |
| R-004 | `copy()` aliased mutable list state in all operator methods | Phase 3 |
| R-005 | `get_emotion` import shadow in `__init__.py` | Phase 3 |
| R-006 | `np.matrix` deprecated in `as_matrix` | Phase 3 |
| R-007 | `Emotion.valence` conflated arousal with hedonics | Phase 4 |
| R-008 | `Emotion.type`/`CompositeEmotion.type` used undocumented heuristics | Phase 4 |
| R-009 | `EmotionalDimension.valence` wrong (sensitivity=−1, attention=+1) | Phase 4 |
| R-010 | Cross-axis `+` returned `Feeling` instead of `CompositeEmotion` | Phase 4 |
| R-011 | `Feeling.emotional_flow` used `np.linalg.norm` (always ≥ 0) | Phase 4 |
| R-012 | `Feeling.__float__` returned `numpy.float64` — DeprecationWarning | Phase 4 |
| R-013 | `CompositeEmotion` inherited `Emotion` — single-axis methods on multi-axis object | Phase 4 |
| R-014 | `CompositeEmotion.__truediv__`/`__floordiv__` returned `None` for Emotion operands | Phase 4 |
| R-015 | `emotions.py` `__main__` debug print block (dead code) | Phase 4 |
| R-016 | Package renamed `emotion_data` → `emotion-algebra` | v1.0.0 |
| R-017 | License corrected from MIT to Apache-2.0 | v1.0.0 |

---

## Security

No network I/O, no file-system writes, no `exec`/`eval` in the core library.
The optional `tag.py` (ParallelDots API) performs outbound HTTPS to `api.paralleldots.com`.
The optional `deepmoji.py` loads a torch model from a local path or via torchMoji download.
Neither is loaded unless `EmotionAnalyzer.tag_emotions()` or `tag_emojis()` is explicitly called.
