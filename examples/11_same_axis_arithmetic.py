"""11_same_axis_arithmetic.py — Same-axis addition: flows sum within an axis."""
from copy import copy
from emotion_algebra.emotions import EMOTIONS

# --- Adding two same-axis emotions ------------------------------------------
print("=== Same-axis addition (Sensitivity) ===")
annoyance = copy(EMOTIONS["annoyance"])   # flow +1
anger     = copy(EMOTIONS["anger"])       # flow +2
apprehension = copy(EMOTIONS["apprehension"])  # flow -1

print(f"  annoyance + annoyance = {annoyance + annoyance}")   # flow 2 → anger
print(f"  annoyance + anger     = {annoyance + anger}")       # flow 3 → rage
print(f"  anger + anger         = {anger + anger}")           # flow 4 → mega rage
print(f"  annoyance + apprehension = {annoyance + apprehension}")  # flow 0 → neutrality

# --- Pleasantness axis -------------------------------------------------------
print("\n=== Same-axis addition (Pleasantness) ===")
serenity    = copy(EMOTIONS["serenity"])      # flow +1
joy         = copy(EMOTIONS["joy"])           # flow +2
pensiveness = copy(EMOTIONS["pensiveness"])   # flow -1
sadness     = copy(EMOTIONS["sadness"])       # flow -2

print(f"  serenity + serenity   = {serenity + serenity}")       # joy
print(f"  joy + serenity        = {joy + serenity}")            # ecstasy
print(f"  serenity + pensiveness= {serenity + pensiveness}")    # neutrality
print(f"  sadness + sadness     = {sadness + sadness}")         # grief (flow -4 → mega grief)

# --- Subtraction (adds the opposite) ----------------------------------------
print("\n=== Subtraction ===")
print(f"  anger - 1         = {anger - 1}")         # annoyance
print(f"  anger - anger     = {anger - anger}")     # neutrality (anger + (-anger) = anger + fear = composite!)
print(f"  joy   - serenity  = {joy - serenity}")    # serenity + (-serenity) wait...
# Note: emotion - emotion adds the negation of the other
fear = copy(EMOTIONS["fear"])
print(f"  anger - fear      = {anger - fear}")   # anger + (-fear) = anger + anger = rage
