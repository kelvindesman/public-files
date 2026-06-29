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

## Follow-up fixes (feature cache + config)

The first pass shipped the `rich=True` code but the **feature cache silently
defeated it**. In the entropy-compute cell the cache file was keyed by method
name only (`F_JSD_Fuzzy.npy`), so an older `rich=False` cache (24 features =
6 scales × 4 sensors × 1) was reloaded and the new rich features were **never
recomputed**. EDM/JSD even loaded a 6-scale cache while CMSE/FME computed fresh
at 10 scales, making the cross-method comparison unfair. Two bugs in the same
cell compounded it: the per-method `np.save` sat *outside* the loop (only the
last method was cached) and a trailing untagged `np.save("cache/F_EDM_Fuzzy.npy")`
re-wrote stale files.

Fixes applied to the notebook:

1. **Config-aware cache key** — cache filename now embeds a `CFG_TAG`
   (`S{scales}_m{m}_r{r}_nref{n_ref}_bins{jsd_bins}_rich{0|1}`). Any config or
   `rich` change invalidates the cache automatically, so stale features can no
   longer be reused.
2. **Cache write moved inside the loop** so every method is persisted, and the
   untagged `F_EDM_Fuzzy.npy` write was removed (replaced by `F_default.npy`).
3. **Config bumped to the best row** — `n_ref=128`, `jsd_bins=40` in the
   entropy-config cell (was 64 / 20).
4. **Stale outputs cleared** across all code cells so the notebook is in a clean
   `Run All` state — the previous file had outputs only from a partial 2026-06-27
   run, which is why most cells (ANN accuracy, classification report, comparison
   tables, plots, exports) showed no output.

> Re-run note: delete any pre-existing `cache/` directory (or the change in
> `CFG_TAG` already bypasses the old files) and run the notebook top-to-bottom in
> a single kernel so `X_feat`, `y`, `W_s`, `gs` are all in scope.
