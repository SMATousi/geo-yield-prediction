# Distribution shift between held-out years (LOYO) and regions (LORO)

Generated 2026-10-07 by `compute_shift.py` (scikit-learn, project env) and `plot_shift.py` (matplotlib).

## Data

- **Unit:** one point per field-season (≈300 pixels per field, monthly cache).
- **Features:**
  - the 12 S2 band means over dated slots;
  - NDVI peak, mean and peak timing;
  - season length;
  - seasonal temp_mean and precipitation sums, and mean temp_max;
  - terrain and soil means.

## Files

- `<PAIR>.png`:
  - t-SNE of the standardized field-season features, coloured by harvest year, by region (provinces for Argentina, farms elsewhere) and by field mean yield;
  - yield box plots by year and by region;
  - each held-out group's covariate shift against its label shift.
- `summary_loyo.png`, `summary_loro.png`: every held-out year / region of all pairs.
- `shift_scores.md`: the numbers.

## Scores

- **Covariate shift:** cross-validated AUROC of a logistic regression telling the group from the rest.
  - `all inputs`: near 1 almost everywhere, because weather, season dates and soil/terrain are (near-)constant within a year or region.
  - `S2 only` (band means + NDVI peak/mean): the more meaningful score; used in the plots.
- **Label shift:** the group's mean field yield minus the rest's, in SDs of field yields.

## Reading t-SNE

Only overlap versus separation is meaningful. Cluster sizes and between-cluster distances are not.
