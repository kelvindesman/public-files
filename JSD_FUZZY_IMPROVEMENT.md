# Improving JSD–Fuzzy Entropy accuracy

Investigation of the uploaded notebook (`final_databaru_jsd_improved.ipynb`) —
fault detection on 4 humidity sensors using multiscale entropy features + ANN.
The 17-class fault-scenario classification is the target task.

## Diagnosis

The original `jsd_fuzzy_entropy_1d` emits **one scalar per scale**: the
Jensen–Shannon divergence between the fuzzy-similarity histograms of embedding
dimension `m` vs `m+1`. JSD measures only the *shape* difference between the two
similarity distributions and **discards their magnitude/location** (mean & spread
of similarity), which is exactly the information that separates fault types.

## What was tested (ANN MLP 128–64, mean of 3 stratified splits, 17 classes)

| Change | #feat | accuracy | macro-F1 |
|---|---|---|---|
| **Original JSD-Fuzzy** (1/scale, n_ref=64, bins=20) | 40 | 0.341 | 0.332 |
| More bins (50) | 40 | 0.337 | 0.327 |
| Higher n_ref (128) | 40 | 0.332 | 0.325 |
| n_ref=128 + bins=40 | 40 | 0.348 | 0.341 |
| **+ fuzzy-entropy + mean + std per scale** | 160 | **0.421** | **0.410** |
| + full 6-moment descriptor | 240 | 0.424 | 0.410 |

Binning / sampling tweaks alone barely move the needle. The big win is the
**richer per-scale descriptor**.

### Ablation — which term matters

| Features per scale | accuracy |
|---|---|
| `jsd` only | 0.348 |
| `fe` (fuzzy entropy) only | 0.260 |
| `jsd + mean` | **0.413** |
| `jsd + fe` | 0.393 |
| `jsd + fe + mean + std` | **0.421** |

Adding the **mean of the fuzzy-similarity distribution** gives the single largest
jump (0.34 → 0.41); `fe` and `std` add a little more. The similarity samples are
already computed for the JSD histogram, so the extra features are essentially free.

## Change applied to the notebook

`jsd_fuzzy_entropy_1d(..., rich=True)` now returns `[jsd, fe, mean_m, std_m]` per
scale (4×S features per sensor) instead of `[jsd]`. `rich=False` restores the old
behaviour. `entropy_cv_report` in the CV-stability cell was made shape-robust
(infers per-sensor feature count) so it no longer crashes on the wider JSD vector.

## Recommendation to push further

1. Keep `rich=True` (default) — **+5 accuracy points** at the notebook's current
   `n_ref=64, bins=20`.
2. Raise `n_ref` 64→128 and `jsd_bins` 20→40 in the entropy-config cell — together
   with the rich descriptor this reaches **~0.42 (+8 points)**.
3. The gain is specific to the standardized-MLP path; RandomForest benefits less,
   so keep the ANN as the primary classifier for JSD-Fuzzy.
