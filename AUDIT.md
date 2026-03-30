# AUDIT.md — emotion-algebra

Evidence-based register of technical debt, known issues, and security observations.
Updated: 2026-03-30 (v1.0.1).

---

## Open Issues

### A-002 — `emotion_vector` sensitivity branch unreachable
**File**: `emotion_algebra/feelings.py:142`
**Severity**: Low
**Detail**: The `sensitivity` branch of `emotion_vector` is unreachable in current test data.
All five missing operators (`__mul__`, `__truediv__`, `__floordiv__`, `__lshift__`, `__rshift__`)
have been implemented; operator parity with `Emotion` is restored.

### A-003 — `EmotionAnalyzer` optional-dep paths are untestable without extras
**File**: `emotion_algebra/__init__.py:21,36,41,46,56,61–62,67–68`
**Severity**: Low
**Detail**: `tag_emotions`/`tag_emojis` require `torch`+`torchMoji`; lexicon methods require `pandas`.
Lazy-import guard is correct; zero coverage of these paths in the default test run.
**Recommendation**: Add `test_optional.py` under a `[test-all]` extra, skipped by default.

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
| R-018 | `CompositeEmotion.__eq__`/`__ne__` dead same-dimension branch removed | v1.0.1 |
| R-019 | Stale `# TODO` comments removed from `emotions.py`, `plutchik.py`, `composite_emotions.py` | v1.0.1 |
| R-020 | `EmotionBase.__int__` docstring added — clarifies net activation scalar semantics | v1.0.1 |
| R-021 | `Emotion.kind` inline comment updated — explicit approximate disclaimer | v1.0.1 |
| R-022 | A-002 partial: `Feeling` operators `__mul__`/`__truediv__`/`__floordiv__`/`__lshift__`/`__rshift__` added | v1.7 |
| R-023 | `score_mixed` / `from_mixed` — unified word+emoji scoring in one pass | v1.7 |
| R-024 | `register_emoji` / `unregister_emoji` — runtime emoji map extensibility | v1.7 |
| R-025 | CLI emoji support — single emoji args dispatched via `score_emojis` | v1.7 |

---

## Security

No network I/O, no file-system writes, no `exec`/`eval` in the core library.
The optional `[transformers]` extra loads a HuggingFace model via the `transformers` pipeline;
model weights are downloaded to the HF cache on first use.
