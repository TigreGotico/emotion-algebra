"""02_intensity_arithmetic.py — Moving along an axis with +, -, *, /, //, <<, >>."""
from copy import copy
from emotion_algebra.emotions import EMOTIONS
from emotion_algebra.plutchik import Neutrality

anger = copy(EMOTIONS["anger"])       # Sensitivity +2
annoyance = copy(EMOTIONS["annoyance"])  # Sensitivity +1
rage = copy(EMOTIONS["rage"])         # Sensitivity +3

# --- Upgrading / downgrading ------------------------------------------------
print("=== Intensity ladder: annoyance → anger → rage ===")
print(f"  annoyance + 1 = {annoyance + 1}")   # anger
print(f"  anger     + 1 = {anger + 1}")        # rage
print(f"  anger     - 1 = {anger - 1}")        # annoyance
print(f"  rage      - 1 = {rage - 1}")         # anger
print(f"  annoyance - 1 = {annoyance - 1}")    # Neutrality

# --- Division / floor division ----------------------------------------------
print("\n=== Division ===")
print(f"  rage / 3  = {rage / 3}")    # annoyance (3/3=1)
print(f"  rage // 2 = {rage // 2}")   # annoyance (3//2=1)

# --- Shift operators --------------------------------------------------------
print("\n=== Shift operators ===")
print(f"  anger << 1 = {anger << 1}")  # annoyance (flow 2-1=1)
print(f"  anger >> 1 = {anger >> 1}")  # rage      (flow 2+1=3)

# --- Hyper-intensity --------------------------------------------------------
print("\n=== Hyper-intensity ===")
mega = rage + 1
extreme = rage + 2
hyper = rage + 3
print(f"  rage + 1 = {mega}   (intensity_offset={mega.intensity_offset})")
print(f"  rage + 2 = {extreme}  (intensity_offset={extreme.intensity_offset})")
print(f"  rage + 3 = {hyper}   (intensity_offset={hyper.intensity_offset})")

# --- Neutrality as arithmetic zero ------------------------------------------
print("\n=== Neutrality identity ===")
n = Neutrality()
print(f"  anger + Neutrality() = {anger + n}")  # anger
print(f"  bool(anger)          = {bool(anger)}")  # True
print(f"  bool(Neutrality())   = {bool(n)}")      # False
