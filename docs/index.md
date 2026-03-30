# emotion-algebra — documentation index

Python library implementing **emotion algebra** — emotions as first-class mathematical objects, grounded in Cambria's Hourglass of Emotions and Plutchik's Wheel.

## Pages

| Page | Contents |
|------|----------|
| [Taxonomy](taxonomy.md) | All 24 named emotions, 4 axes, intensity levels |
| [Algebra](algebra.md) | Operators: intensity arithmetic, negation, composition |
| [Valence, Arousal & Type](valence_arousal.md) | Scientific properties; Russell Circumplex |
| [Feelings](feelings.md) | Named dyads, Feeling vs CompositeEmotion |
| [State & Timeline](state.md) | EmotionalState accumulation, decay, EmotionTimeline |
| [Text & Emoji](text_emoji.md) | Lexicon pipeline, emoji map, DeepMojiAdapter, mixed scoring |
| [CLI](cli.md) | Command-line interface reference |
| [API Reference](api_reference.md) | All public classes, properties, functions |
| [Maintainers Guide](MAINTAINERS_GUIDE.md) | Release, CI/CD, contribution workflow |

## Model provenance

| Model | Authors | Role in this library |
|-------|---------|----------------------|
| **Hourglass of Emotions** (2012) | Cambria, Livingstone, Hussain | 4-axis PASA integer representation; `emotional_flow` |
| **Plutchik's Wheel** (1980) | Robert Plutchik | 8 primaries, named dyads, opposite pairs |
| **Russell's Circumplex** (1980) | James Russell | `type` classification via valence × arousal quadrants |
| **Scherer CPM** (2001) | Klaus Scherer | Cognitive appraisal → primary emotion mapping |
| **Felbo et al.** (2017) | DeepMoji team | Emoji label set for `DeepMojiAdapter`; ONNX model via `deepmoji-onnx` |
| **NRC EmoLex** (2013) | Mohammad & Turney | Plutchik emotion labels in canonical lexicon |
| **SenticNet 6** (2022) | Cambria et al. | Hourglass float axes in canonical lexicon |
| **AFINN-111** (2011) | Nielsen | Integer sentiment scores in canonical lexicon |

## Key design decisions

- `Emotion.valence` = Pleasantness axis only — `anger.valence == 0` (arousal ⊥ hedonics, Posner 2005)
- `Emotion.arousal` = `|emotional_flow|` — axis-independent activation intensity
- Cross-axis `+` returns `CompositeEmotion`; `Feeling` is a separate named construct
- All global registries (`EMOTIONS`, `FEELINGS`, `EMOJI_EMOTION_MAP`) are `MappingProxyType` — immutable at import time
- Runtime extensibility via `register_emoji()` / `unregister_emoji()`

## Key classes

| Class | Module | Role |
|-------|--------|------|
| `Emotion` | `plutchik.py:Emotion` | Core algebra carrier — single-axis lattice point |
| `Neutrality` | `plutchik.py:Neutrality` | Identity element (`e + Neutrality() == e`) |
| `EmotionalDimension` | `plutchik.py:EmotionalDimension` | One Hourglass axis |
| `CompositeEmotion` | `composite_emotions.py:CompositeEmotion` | Multi-axis blend (cross-axis `+` result) |
| `Feeling` | `feelings.py:Feeling` | Named cultural dyad (Plutchik) |
| `FloatEmotion` | `float_emotion.py:FloatEmotion` | Continuous-valued point in Hourglass space |
| `EmotionalState` | `state.py:EmotionalState` | Mutable 4-axis float accumulator with decay |
| `EmotionTimeline` | `state.py:EmotionTimeline` | Sequence of EmotionalState snapshots |
| `Appraisal` | `appraisal.py:Appraisal` | Scherer CPM appraisal dimensions |
| `DeepMojiAdapter` | `emoji.py:DeepMojiAdapter` | Bridge from emoji probability distributions |
| `DeepMojiONNXAdapter` | `deepmoji.py:DeepMojiONNXAdapter` | Neural text→emoji→emotion (canonical engine) |
| `EmotionAnalyzer` | `__init__.py:EmotionAnalyzer` | High-level facade over all sub-modules |
