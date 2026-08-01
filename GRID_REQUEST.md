# Data request — GRID emotion coordinates

**Not sent. For the maintainer to send, or not.**

The only route to *fitting* `potency` and `unpredictability` (currently the two
guessed axes, 52 unvalidated numbers) is the GRID per-emotion coordinate table.
Fontaine, Scherer, Roesch & Ellsworth (2007) state in footnote 3 that it is not
published in the article and "can be requested from the first author".

---

**To:** grid@unige.ch (Swiss Center for Affective Sciences); Prof. Johnny Fontaine
**Subject:** Request for the 24-term GRID dimension coordinates (Fontaine et al. 2007, fn. 3)

Dear Professor Fontaine,

I am building **emotion-algebra**, an open-source (Apache-2.0) Python library for
computational affect, and I have based its core representation on the four
dimensions your 2007 *Psychological Science* paper established — valence, potency,
arousal and unpredictability.

Footnote 3 of that paper notes that the profiles of the emotion words on the four
dimensions can be requested from you. I would be very grateful for that table: the
factor scores of the 24 emotion terms on the four dimensions.

My reason for asking is specific. The library currently ships coordinates in which
valence and arousal are taken from human norms (Warriner et al. 2013), but
**potency and unpredictability are reasoned from the appraisal literature rather
than fitted** — because no public dataset provides them. We register every constant
with its provenance and state plainly which are guessed; these two axes are the
largest remaining gap, and I would rather fit them than defend them.

If the data can be shared, I would of course cite the GRID work prominently and
would be happy to contribute the resulting fit back in whatever form is useful to
you.

With thanks and best regards,

Casimiro Ferreira
TigreGotico
https://github.com/TigreGotico/emotion-algebra

---

## If it arrives

Fitting is a clean follow-up:

1. Map the 24 GRID terms onto our prototype names (most are direct; some need a
   documented substitution, as `calibrate_prototypes.py` already does for Warriner).
2. Rescale the factor scores onto our axis bounds.
3. Re-register `prototypes.potency` and `prototypes.unpredictability` as `FITTED`,
   naming the script.
4. Re-run `scripts/robustness.py` — it already tells us in advance how much would
   change: every load-bearing claim survives ±50% perturbation of these axes at
   **100%**, so the fit would sharpen the numbers without overturning the science.
