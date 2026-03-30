#!/usr/bin/env python
"""Build the canonical emotion-algebra lexicon.

Sources merged (in priority order per field):
  1. Existing word_emotion_lexicon.csv  — color, orientation, subjectivity
  2. NRC EmoLex (via nrclex)            — Plutchik emotion label, pos/neg sentiment
  3. SenticNet 6 (via senticnet)        — Hourglass float axes
  4. AFINN-111 (downloaded, cached)     — integer sentiment score

Output: emotion_algebra/word_emotion_lexicon.csv
Schema:
  word            — lowercase token
  emotion         — Plutchik primary (anger/fear/joy/sadness/trust/disgust/surprise/anticipation) or ""
  pleasantness    — SenticNet introspection axis [-1,1] or ""
  attention       — SenticNet temper axis [-1,1] or ""
  aptitude        — SenticNet attitude axis [-1,1] or ""
  sensitivity     — SenticNet sensitivity axis [-1,1] or ""
  sentiment       — positive/negative/neutral or ""
  afinn_score     — integer or ""
  color           — color string or ""
  orientation     — orientation tag or ""
  subjectivity    — subjectivity tag or ""
"""
from __future__ import annotations

import csv
import os
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
OUTPUT = REPO / "emotion_algebra" / "word_emotion_lexicon.csv"
EXISTING = OUTPUT  # same file — we overwrite

PLUTCHIK_EMOTIONS = {
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
}

# ---------------------------------------------------------------------------
# 1. Load existing CSV
# ---------------------------------------------------------------------------
print("Loading existing lexicon …")
existing: dict[str, dict] = {}
with open(EXISTING, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        word = row["word"].strip().lower()
        if not word:
            continue
        existing[word] = {
            "emotion":      row.get("emotion", "").strip(),
            "sentiment":    row.get("sentiment", "").strip(),
            "color":        row.get("color", "").strip(),
            "orientation":  row.get("orientation", "").strip(),
            "subjectivity": row.get("subjectivity", "").strip(),
        }
print(f"  {len(existing):,} words in existing CSV")

# ---------------------------------------------------------------------------
# 2. Load NRC EmoLex via nrclex
# ---------------------------------------------------------------------------
print("Loading NRC EmoLex …")
nrc: dict[str, dict] = {}
try:
    import os as _os
    _cwd = _os.getcwd()
    _os.chdir("/tmp")  # avoid nrclex CWD-as-lexicon-path bug
    from nrclex import NRCLex
    _instance = NRCLex("placeholder")
    _lexicon = _instance.__lexicon__
    _os.chdir(_cwd)
    for word, labels in _lexicon.items():
        word = word.strip().lower()
        emotions = [l for l in labels if l in PLUTCHIK_EMOTIONS]
        sentiment = (
            "positive" if "positive" in labels
            else "negative" if "negative" in labels
            else ""
        )
        nrc[word] = {
            "emotions": emotions,
            "sentiment": sentiment,
        }
    print(f"  {len(nrc):,} words in NRC EmoLex")
except ImportError:
    print("  nrclex not available — skipping")

# ---------------------------------------------------------------------------
# 3. Load SenticNet
# ---------------------------------------------------------------------------
print("Loading SenticNet …")
senticnet: dict[str, dict] = {}
try:
    from senticnet.senticnet import SenticNet
    sn = SenticNet()
    for concept, data in sn.data.items():
        if not isinstance(data, list) or len(data) < 7:
            continue
        # SenticNet includes multi-word concepts joined by underscores —
        # we include them but also split single-word entries.
        word = concept.strip().lower().replace("_", " ")
        try:
            # data is a list: [introspection, temper, attitude, sensitivity,
            #                   moodtag1, moodtag2, polarity_label, polarity_value, sem...]
            senticnet[word] = {
                "pleasantness": round(float(data[0] or 0), 4),
                "attention":    round(float(data[1] or 0), 4),
                "aptitude":     round(float(data[2] or 0), 4),
                "sensitivity":  round(float(data[3] or 0), 4),
                "sentiment":    str(data[6]).strip() if len(data) > 6 else "",
            }
        except (ValueError, TypeError, IndexError):
            pass
    print(f"  {len(senticnet):,} concepts in SenticNet")
except ImportError:
    print("  senticnet not available — skipping")

# ---------------------------------------------------------------------------
# 4. Download AFINN-111
# ---------------------------------------------------------------------------
AFINN_URL = "https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt"
AFINN_CACHE = Path(os.path.expanduser("~/.local/share/emotion-algebra/lexicons/AFINN-111.txt"))
print("Loading AFINN-111 …")
afinn: dict[str, int] = {}
try:
    if not AFINN_CACHE.exists():
        AFINN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {AFINN_URL} …")
        urllib.request.urlretrieve(AFINN_URL, AFINN_CACHE)
    with open(AFINN_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "\t" in line:
                w, score = line.rsplit("\t", 1)
                afinn[w.strip().lower()] = int(score)
    print(f"  {len(afinn):,} words in AFINN-111")
except Exception as e:
    print(f"  AFINN unavailable: {e}")

# ---------------------------------------------------------------------------
# 5. Merge into canonical rows
# ---------------------------------------------------------------------------
print("Merging …")

# Collect union of all single-word keys
all_words: set[str] = set()
all_words.update(existing.keys())
all_words.update(nrc.keys())
all_words.update(afinn.keys())
# SenticNet: only add single-word entries (no spaces) to avoid pollution
all_words.update(w for w in senticnet.keys() if " " not in w)

COLUMNS = [
    "word", "emotion",
    "pleasantness", "attention", "aptitude", "sensitivity",
    "sentiment", "afinn_score",
    "color", "orientation", "subjectivity",
]

rows: list[dict] = []
for word in sorted(all_words):
    if not word or not word.replace("-", "").replace("'", "").isalpha():
        # skip numeric tokens and pure-punctuation entries
        continue

    ex  = existing.get(word, {})
    nc  = nrc.get(word, {})
    sn  = senticnet.get(word, {})
    af  = afinn.get(word)

    # --- emotion: NRC primary plutchik > existing ---
    nrc_emotions = nc.get("emotions", [])
    emotion = ""
    if nrc_emotions:
        # NRC can give multiple — pick the first (they are equal weight)
        emotion = nrc_emotions[0]
    elif ex.get("emotion") and ex["emotion"] in PLUTCHIK_EMOTIONS:
        emotion = ex["emotion"]

    # --- hourglass axes ---
    pleasantness = sn.get("pleasantness", "")
    attention    = sn.get("attention", "")
    aptitude     = sn.get("aptitude", "")
    sensitivity  = sn.get("sensitivity", "")

    # --- sentiment: existing > NRC > SenticNet > AFINN sign ---
    sentiment = ex.get("sentiment", "")
    if not sentiment:
        sentiment = nc.get("sentiment", "")
    if not sentiment:
        sentiment = sn.get("sentiment", "")
    if not sentiment and af is not None:
        sentiment = "positive" if af > 0 else "negative" if af < 0 else ""

    # Normalise to canonical strings
    if sentiment not in ("positive", "negative", "neutral"):
        sentiment = ""

    rows.append({
        "word":         word,
        "emotion":      emotion,
        "pleasantness": pleasantness if pleasantness != "" else "",
        "attention":    attention    if attention    != "" else "",
        "aptitude":     aptitude     if aptitude     != "" else "",
        "sensitivity":  sensitivity  if sensitivity  != "" else "",
        "sentiment":    sentiment,
        "afinn_score":  "" if af is None else af,
        "color":        ex.get("color", ""),
        "orientation":  ex.get("orientation", ""),
        "subjectivity": ex.get("subjectivity", ""),
    })

print(f"  {len(rows):,} canonical entries")

# ---------------------------------------------------------------------------
# 6. Write output
# ---------------------------------------------------------------------------
print(f"Writing {OUTPUT} …")
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print("Done.")
