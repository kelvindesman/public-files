#!/usr/bin/env python3
"""Make 01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb finish inside Kaggle's 12 h limit.

The notebook was killed with exit 137 after 43202 s. Three things burned the budget:

1. `sample_entropy_1d` counted Chebyshev pairs with a Python `for i in range(N-1)`
   loop. CMSE calls it on raw 2k/7k/10k-sample segments, so the CV-stability cell
   alone needed ~25 CPU-hours. Replaced with blocked numpy counting (bit-identical)
   plus reference subsampling for long series (the same trick `fuzzy_phi` already
   uses for EDM/FME/JSD, so all four methods now share one approximation regime).
2. Every `GridSearchCV` ran with `n_jobs=1` on a 4-vCPU box, and one of them
   (the 312-combo x 5-fold search) threw its result away.
3. Nothing bounded the total runtime, so the session hit Kaggle's wall and lost
   every table instead of writing out what it already had.

The patch adds a `RUNTIME_PROFILE` knob ("fast" / "paper" / "full"), a wall-clock
budget guard that skips remaining work rather than getting SIGKILLed, and
incremental CSV checkpoints.

Idempotent: each edit is anchored on text that only exists pre-patch.

Run: python3 patch_runtime.py
"""
import json
import sys

NB = '/Users/kelvin/apps/public-files/01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb'

RUNTIME_DOC = '''## Runtime & reproducibility

Kaggle stops a notebook at **12 jam** (exit 137). Versi sebelumnya tembus batas itu,
jadi semua tabel hilang. Yang berubah:

| Sumber lambat | Sebelum | Sesudah |
|---|---|---|
| `sample_entropy_1d` (dipakai CMSE) | loop Python `for i in range(N-1)` → **~23 CPU-jam** untuk sel CV-vs-panjang-data saja (≈14 jam wall di 4 vCPU, sudah lewat batas sebelum sel lain jalan) | penghitungan pasangan Chebyshev per-blok numpy; hasil **identik bit-per-bit** untuk window `WIN=256` |
| CMSE pada segmen 7k/10k sampel | pasangan penuh `O(N^2)`, ~24 s/segmen | subsample `SAMPEN_N_REF` vektor referensi — pendekatan yang **sudah** dipakai `fuzzy_phi` untuk EDM/FME/JSD. ~3 s/segmen |
| `GridSearchCV` | `n_jobs=1` di mesin 4 vCPU | `n_jobs=-1`, BLAS dipin ke 1 thread. **Isi grid tidak diubah** |
| Grid 312-kombinasi × 5-fold | dijalankan, hasilnya **dibuang** (`gs` ditimpa sel berikutnya) | mati (`RUN_LEGACY_BIG_GRID=False`) |
| `LogisticRegression(multi_class=...)` | error di scikit-learn ≥1.7 (image Kaggle terbaru) | kwarg dihapus (`"auto"` memang default) |
| `cv_table` skala JSD-Fuzzy | 1..40 (4 fitur/skala dihitung sebagai skala terpisah) | 1..10 + kolom `feat_in_scale` |
| Batas waktu | tidak ada → SIGKILL, semua tabel hilang | `budget_ok()` melewati sisa pekerjaan dan tetap menulis CSV |

**`RUNTIME_PROFILE`** (env var, default `fast`). Grid ANN **sama di kedua profil** —
totalnya cuma ~2 menit, jadi tidak ada gunanya dipangkas. Profil hanya mengatur
dua knob yang benar-benar makan waktu:

| Profil | `CV_N_REPEATS` | `SAMPEN_N_REF` | Wall (4 core, data penuh) | Perkiraan Kaggle CPU |
|---|---|---|---|---|
| `fast` (default) | 10 | 512 | **18 menit (terukur)** | ~40 menit |
| `paper` | 30 (desain asli) | `None` → **eksak** | ~2 jam (ekstrapolasi) | ~4–5 jam |

`paper` menghasilkan angka **persis sama** dengan yang dimaksud kode lama (SampEn
eksak, 30 repeat) — dulu butuh ~14 jam wall untuk sel itu saja, jadi tidak pernah
selesai. Sekarang muat di batas 12 jam.

Fitur entropy untuk window `WIN=256` **identik bit-per-bit** dengan versi lama di
kedua profil, jadi angka head-to-head tetap sebanding. Subsampling `fast` hanya
aktif pada segmen mentah > `SAMPEN_EXACT_MAX` (2500 sampel), yaitu di sel
stabilitas CV.

Catatan: dengan `n_jobs=-1`, kolom `peak_mem_mb` pada tabel footprint hanya melihat
proses induk. Kolom `rss_peak_mb` menambahkan high-water mark seluruh process tree.
Set `ANN_GRID_N_JOBS = 1` kalau butuh angka memori per-metode untuk paper.

`exports/cv_table.csv` ditulis ulang tiap skenario, jadi run yang terputus tetap
meninggalkan hasil parsial yang bisa dipakai.
'''

RUNTIME_CELL = '''# === Runtime guard — keep this as the FIRST executed cell ===
# BLAS threads must be pinned before numpy is imported, otherwise each of the
# joblib/GridSearchCV workers spawns its own OpenMP pool and the 4 vCPUs Kaggle
# gives us get oversubscribed 4x.
import os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

NOTEBOOK_START = time.time()

# Kaggle SIGKILLs the session at 12 h (exit 137). Stop ourselves earlier so the
# tables computed so far still get exported.
KAGGLE_TIME_BUDGET_H = float(os.environ.get("KAGGLE_TIME_BUDGET_H", 10.5))


def elapsed_s():
    return time.time() - NOTEBOOK_START


def time_left_sec():
    return KAGGLE_TIME_BUDGET_H * 3600.0 - elapsed_s()


def budget_ok(need_s=0.0, label=""):
    """True when at least `need_s` seconds of budget remain."""
    left = time_left_sec()
    if left < need_s:
        print(f"[budget] SKIP {label}: {left/60:.1f} min left, "
              f"need ~{need_s/60:.1f} min", flush=True)
        return False
    return True


def log_stage(label):
    print(f"[t+{elapsed_s()/60:6.1f} min] {label}", flush=True)


log_stage("runtime guard armed | "
          f"budget={KAGGLE_TIME_BUDGET_H} h | BLAS threads=1")
'''

CONFIG_OLD = '''# === Global config (edit here) ===
FAST_MODE = True  # True: fastest path to get all outputs
RUN_ALL_METHODS = True  # True: compute + evaluate all entropy methods
METHOD_LIST = ["EDM-Fuzzy", "CMSE", "FME", "JSD-Fuzzy"]
DEFAULT_METHOD = "EDM-Fuzzy"  # used for plots/CM/report if you want 1 method highlighted
CACHE_FEATURES = True  # cache features per method to avoid recompute on rerun
CACHE_DIR = "cache"
EXPORT_DIR = "exports"

# FAST_MODE knobs (will override some later defaults)
FAST_MAX_PER_CLASS = 200
FAST_N_REF = 128
FAST_N_JOBS = -1
FAST_MLP_MAX_ITER = 200
FAST_CV_REPEATS = 10  # for entropy stability repeats (if used)

# Kaggle time budget guard (hours)
KAGGLE_TIME_BUDGET_H = 11.5

# Per-scenario ANN search knobs (paper table)
SCENARIO_GRID_CV = 3
SCENARIO_TEST_FRAC = 0.25
SCENARIO_MAX_PER_CLASS = 300
SCENARIO_MAX_CANDIDATES = 10
SCENARIO_MAX_ITER = 250
'''

CONFIG_NEW = '''# === Global config (edit here) ===
FAST_MODE = True  # True: fastest path to get all outputs
RUN_ALL_METHODS = True  # True: compute + evaluate all entropy methods
METHOD_LIST = ["EDM-Fuzzy", "CMSE", "FME", "JSD-Fuzzy"]
DEFAULT_METHOD = "EDM-Fuzzy"  # used for plots/CM/report if you want 1 method highlighted
CACHE_FEATURES = True  # cache features per method to avoid recompute on rerun
CACHE_DIR = "cache"
EXPORT_DIR = "exports"

# === Runtime profile ===============================================
# Measured on 4 cores with the full dataset: the notebook is dominated by the
# CV-vs-data-length cell (13 of 18 min). Every ANN grid search together costs
# ~2 min, so the grids are NOT reduced by any profile -- shrinking them would
# have broken the head-to-head anchor against the sibling notebooks for nothing.
#
#   fast  : ~18 min on 4 cores (~40 min on Kaggle CPU). SampEn on the long raw
#           segments is estimated from 512 reference templates.
#   paper : ~2 h on 4 cores (~4-5 h on Kaggle). 30 CV repeats and EXACT SampEn
#           everywhere -- i.e. bit-for-bit the study the original code intended,
#           which the old O(N^2) Python loop needed ~14 h of wall clock to reach
#           and therefore never finished.
#
# Window features (WIN=256) are exact and identical in both profiles; the
# approximation only ever touches raw segments longer than SAMPEN_EXACT_MAX.
RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE", "fast")

_PROFILES = {
    "fast":  dict(cv_repeats=10, sampen_n_ref=512),
    "paper": dict(cv_repeats=30, sampen_n_ref=None),
}
if RUNTIME_PROFILE not in _PROFILES:
    raise ValueError(f"RUNTIME_PROFILE must be one of {list(_PROFILES)}")
_CFG = _PROFILES[RUNTIME_PROFILE]

N_JOBS = -1  # BLAS is pinned to 1 thread, so joblib owns every core

# GridSearchCV n_jobs for the 17-class ANN cell that feeds the "computing
# footprint per method" table. With n_jobs>1 the fits happen in worker
# processes, so that table's tracemalloc `peak_mem_mb` only sees the parent.
# Set this to 1 if you need per-method memory numbers for the paper (slower).
ANN_GRID_N_JOBS = N_JOBS

# Entropy stability study (CV vs data length)
CV_N_REPEATS = _CFG["cv_repeats"]

# SampEn / CMSE cost control. Series with N <= SAMPEN_EXACT_MAX are counted
# exactly (identical to the original implementation); longer ones estimate the
# pair probability from SAMPEN_N_REF random template vectors, exactly like
# fuzzy_phi already does for EDM-Fuzzy / FME / JSD-Fuzzy. WIN=256 windows and
# the 2000-sample segments stay exact. Set SAMPEN_N_REF=None to force exact.
SAMPEN_EXACT_MAX = 2500
SAMPEN_N_REF = _CFG["sampen_n_ref"]

# The 312-combo x 5-fold search below is dead code: its `gs` is overwritten by
# the multi-method cell before anything reads it. Costs ~6 min even parallelised.
RUN_LEGACY_BIG_GRID = False

# FAST_MODE knobs (will override some later defaults)
FAST_MAX_PER_CLASS = 200
FAST_N_REF = 128
FAST_N_JOBS = -1
FAST_MLP_MAX_ITER = 200
FAST_CV_REPEATS = CV_N_REPEATS

# Per-scenario ANN search knobs (paper table)
SCENARIO_GRID_CV = 3
SCENARIO_TEST_FRAC = 0.25
SCENARIO_MAX_PER_CLASS = 300
SCENARIO_MAX_CANDIDATES = 10
SCENARIO_MAX_ITER = 250

print(f"RUNTIME_PROFILE={RUNTIME_PROFILE} | CV_N_REPEATS={CV_N_REPEATS} | "
      f"SAMPEN_N_REF={SAMPEN_N_REF} | ANN grids: unchanged from original")
'''

SAMPEN_OLD = '''def sample_entropy_1d(y, m, r):
    # SampEn dasar untuk CMSE (pakai jarak Chebyshev)
    V_m  = embed_matrix(y, m)
    V_m1 = embed_matrix(y, m+1)

    def _count_similar(V):
        N = V.shape[0]
        if N < 2:
            return 0, 0
        count = 0
        total = 0
        for i in range(N-1):
            d = np.max(np.abs(V[i+1:] - V[i]), axis=1)
            count += np.sum(d <= r)
            total += (N - i - 1)
        return count, total

    c_m, t_m = _count_similar(V_m)
    c_m1, t_m1 = _count_similar(V_m1)
    if t_m == 0 or t_m1 == 0 or c_m == 0 or c_m1 == 0:
        return np.nan
    return -np.log((c_m1 / t_m1) / (c_m / t_m))
'''

SAMPEN_NEW = '''def _cheb_pair_count(y, dim, r, n_ref=None, seed=0, max_elems=8_000_000):
    """Count length-`dim` template pairs of `y` within Chebyshev radius r.

    Returns (count, total). Two regimes:
      n_ref is None / >= N -> exact upper-triangle count over all N(N-1)/2 pairs.
                              Same number the original Python `for i in range(N-1)`
                              loop produced, counted in numpy blocks instead.
      n_ref < N            -> estimate P(dist <= r) from n_ref random template
                              vectors compared against all N. Cost drops from
                              O(N^2) to O(n_ref*N), which is what makes CMSE on
                              7k/10k-sample segments tractable.

    Templates are sliding windows, so max_k |y[i+k] - y[j+k]| <= r is the same
    predicate as "|y[i+k] - y[j+k]| <= r for every k". Testing each k as a 2-D
    boolean plane and AND-ing them never materialises the (block, N, dim) float
    temporary the naive form needs, which is where the time actually went.
    """
    N = len(y) - dim + 1
    if N < 2:
        return 0, 0

    if n_ref is not None and n_ref < N:
        rng = np.random.default_rng(seed)
        ref = rng.choice(N, size=n_ref, replace=False)
        chunk = max(1, int(max_elems // N))
        cnt = 0
        for st in range(0, n_ref, chunk):
            rb = ref[st:st + chunk]
            ok = None
            for k in range(dim):
                cur = np.abs(y[rb + k][:, None] - y[None, k:k + N]) <= r
                ok = cur if ok is None else (ok & cur)
            cnt += int(np.count_nonzero(ok))
        cnt = max(cnt - n_ref, 0)  # every reference matches itself at d=0
        return cnt, n_ref * (N - 1)

    chunk = max(1, int(max_elems // N))
    cnt = 0
    for st in range(0, N - 1, chunk):
        en = min(st + chunk, N - 1)
        tail_n = N - st  # only columns j >= st can hold a j > i pair
        ok = None
        for k in range(dim):
            cur = np.abs(y[st + k:en + k][:, None]
                         - y[None, k + st:k + st + tail_n]) <= r
            ok = cur if ok is None else (ok & cur)
        cols = np.arange(tail_n)[None, :]
        rows = np.arange(en - st)[:, None]
        cnt += int(np.count_nonzero(ok & (cols > rows)))
    return cnt, N * (N - 1) // 2


def sampen_n_ref_for(series_len):
    """Pick the counting regime ONCE per series, not per coarse-grained scale.

    CMSE coarse-grains a length-L series into s substreams of length L/s, so a
    per-substream cutoff would count low scales approximately and high scales
    exactly. That puts a step in the CV-vs-scale curve which has nothing to do
    with the signal. One regime per call keeps every scale comparable.
    """
    if SAMPEN_N_REF is None or series_len <= SAMPEN_EXACT_MAX:
        return None
    return SAMPEN_N_REF


def sample_entropy_1d(y, m, r, n_ref=None, seed=0):
    # SampEn dasar untuk CMSE (pakai jarak Chebyshev).
    # n_ref=None -> exact. _cheb_pair_count also falls back to exact whenever
    # n_ref >= N, so short substreams stay exact for free.
    y = np.ascontiguousarray(y)
    c_m, t_m = _cheb_pair_count(y, m, r, n_ref, seed)
    c_m1, t_m1 = _cheb_pair_count(y, m + 1, r, n_ref, seed + 1)
    if t_m == 0 or t_m1 == 0 or c_m == 0 or c_m1 == 0:
        return np.nan
    return -np.log((c_m1 / t_m1) / (c_m / t_m))
'''

CMSE_OLD = '''def cmse_1d(x, scales, m=2, r_ratio=0.2):
    # CMSE: Composite Multiscale Sample Entropy
    out = []
    for s in scales:
        ys = coarse_grain_multi(x, s)
        if not ys:
            out.append(np.nan); continue
        ent_list = []
        for y in ys:
            if len(y) < (m+2):
                continue
            r = r_ratio * np.std(y, ddof=1)
            ent_list.append(sample_entropy_1d(y, m=m, r=r))
        if len(ent_list) == 0:
            out.append(np.nan)
        else:
            out.append(np.nanmean(ent_list))
    return np.array(out, dtype=float)
'''

CMSE_NEW = '''def cmse_1d(x, scales, m=2, r_ratio=0.2, n_ref="auto", seed=0):
    # CMSE: Composite Multiscale Sample Entropy
    if n_ref == "auto":
        n_ref = sampen_n_ref_for(len(x))
    out = []
    for s in scales:
        ys = coarse_grain_multi(x, s)
        if not ys:
            out.append(np.nan); continue
        ent_list = []
        for k, y in enumerate(ys):
            if len(y) < (m+2):
                continue
            r = r_ratio * np.std(y, ddof=1)
            ent_list.append(sample_entropy_1d(y, m=m, r=r, n_ref=n_ref,
                                              seed=seed + 11*int(s) + k))
        if len(ent_list) == 0:
            out.append(np.nan)
        else:
            out.append(np.nanmean(ent_list))
    return np.array(out, dtype=float)
'''

FEAT_TIMER_OLD = '''from joblib import Parallel, delayed

START_TIME = time.time()

def time_left_sec():
    if 'KAGGLE_TIME_BUDGET_H' not in globals():
        return None
    return KAGGLE_TIME_BUDGET_H * 3600 - (time.time() - START_TIME)

def ensure_window_3d(W, name="W"):'''

FEAT_TIMER_NEW = '''from joblib import Parallel, delayed

# NOTE: START_TIME / time_left_sec() now come from the runtime-guard cell at the
# top of the notebook. Re-arming the clock here used to hide time already spent.
START_TIME = NOTEBOOK_START

def ensure_window_3d(W, name="W"):'''

CMSE_DISPATCH_OLD = """        if method_key == 'cmse':
            return cmse_1d(x, scales=scales, m=m, r_ratio=r_ratio)"""

CMSE_DISPATCH_NEW = """        if method_key == 'cmse':
            return cmse_1d(x, scales=scales, m=m, r_ratio=r_ratio, seed=seed_local)"""

METRICS_OLD = '''def run_with_metrics(label, fn):
    # Catatan: footprint = estimasi peak memory Python via tracemalloc
    tracemalloc.start()
    t0 = time.perf_counter()
    c0 = time.process_time()
    result = fn()
    t1 = time.perf_counter()
    c1 = time.process_time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {
        "wall_s": t1 - t0,
        "cpu_s": c1 - c0,
        "peak_mem_mb": peak / (1024*1024)
    }
    logging.info("%s | wall=%.2fs cpu=%.2fs peak_mem=%.2f MB", label, metrics["wall_s"], metrics["cpu_s"], metrics["peak_mem_mb"])
    return result, metrics
'''

METRICS_NEW = '''import sys as _sys
try:
    import resource as _resource  # POSIX only
except ImportError:
    _resource = None

# ru_maxrss is bytes on macOS, kilobytes on Linux.
_RSS_UNIT = 1024.0 * 1024.0 if _sys.platform == "darwin" else 1024.0


def _rss_peak_mb():
    """Peak RSS of this process and its joblib workers, in MB."""
    if _resource is None:
        return float("nan")
    self_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    child_kb = _resource.getrusage(_resource.RUSAGE_CHILDREN).ru_maxrss
    return max(self_kb, child_kb) / _RSS_UNIT


def run_with_metrics(label, fn):
    # peak_mem_mb = tracemalloc peak of THIS process. With n_jobs>1 the fits run
    # in workers, so rss_peak_mb (which includes children) is the honest number.
    tracemalloc.start()
    t0 = time.perf_counter()
    c0 = time.process_time()
    result = fn()
    t1 = time.perf_counter()
    c1 = time.process_time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    metrics = {
        "wall_s": t1 - t0,
        "cpu_s": c1 - c0,
        "peak_mem_mb": peak / (1024*1024),
        "rss_peak_mb": _rss_peak_mb(),
    }
    logging.info("%s | wall=%.2fs cpu=%.2fs peak_mem=%.2f MB rss=%.0f MB", label,
                 metrics["wall_s"], metrics["cpu_s"], metrics["peak_mem_mb"],
                 metrics["rss_peak_mb"])
    return result, metrics
'''

FEAT_LOOP_OLD = '''for name in methods:
    safe = name.replace("-", "_").replace(" ", "_")
    cache_path = os.path.join(CACHE_DIR, f"F_{safe}__{CFG_TAG}.npy")
    if CACHE_FEATURES and os.path.exists(cache_path):
        F_by_method[name] = sanitize_features(np.load(cache_path), name=f"F_{name}")
        print(name, "feature shape (cache):", F_by_method[name].shape, "| cfg:", CFG_TAG)
        continue
    Fm, mtr = run_with_metrics('''

FEAT_LOOP_NEW = '''for name in methods:
    safe = name.replace("-", "_").replace(" ", "_")
    cache_path = os.path.join(CACHE_DIR, f"F_{safe}__{CFG_TAG}.npy")
    if CACHE_FEATURES and os.path.exists(cache_path):
        F_by_method[name] = sanitize_features(np.load(cache_path), name=f"F_{name}")
        print(name, "feature shape (cache):", F_by_method[name].shape, "| cfg:", CFG_TAG)
        continue
    log_stage(f"entropy features: {name} ({W_s.shape[0]} windows)")
    Fm, mtr = run_with_metrics('''

# --- CV-vs-data-length cell: full rewrite -------------------------------------
CV_CELL_ANCHOR = 'CV_DATA_LENGTHS = [2000, 7000, 10000]'

CV_CELL_NEW = '''import numpy as np
import pandas as pd
import os
# === CV stabilitas entropy vs panjang data (tanpa downsampling & tanpa windowing) ===
#
# Dominant cost in the whole notebook: 17 scenarios x 3 lengths x 4 methods x
# CV_N_REPEATS segments x 4 sensors. CMSE on a 10k-sample segment took ~24 s with
# the old Python pair loop (~23 CPU-h for this cell alone, i.e. past Kaggle's 12 h
# wall before anything else ran); blocked counting + SAMPEN_N_REF brings it to ~3 s.
# CV_N_REPEATS comes from RUNTIME_PROFILE. Rows are checkpointed per scenario so
# a kill still leaves a usable exports/cv_table.csv.

CV_DATA_LENGTHS = [2000, 7000, 10000]   # panjang data (jumlah sampel mentah) yang ingin dibandingkan
CV_RANDOM_SEED = 123
CV_CHECKPOINT = os.path.join(EXPORT_DIR, "cv_table.csv")
CV_RESUME = True                        # reuse checkpoint if it is already complete

rng_cv = np.random.default_rng(CV_RANDOM_SEED)

def entropy_cv_report(F, S=None, nsensors=4):
    # ROBUST: jumlah fitur per sensor di-infer dari F (mendukung JSD-Fuzzy "rich"
    # yang menghasilkan 4 fitur/skala, sehingga kolom = nsensors * (4*S), bukan nsensors*S).
    N = F.shape[0]
    if F.shape[1] % nsensors != 0:
        raise ValueError(f"F.shape[1]={F.shape[1]} tidak habis dibagi nsensors={nsensors}")
    S_feat = F.shape[1] // nsensors
    E = F.reshape(N, nsensors, S_feat)
    mean = E.mean(axis=0)
    std  = E.std(axis=0, ddof=0)
    cv   = std / (np.abs(mean)+1e-12)
    return cv, cv.mean(axis=0)

def sample_segments(X_ts, seg_len, n_repeats, rng):
    # X_ts: (T,4) -> W: (n_repeats, seg_len, 4)
    T = X_ts.shape[0]
    if seg_len > T:
        raise ValueError(f"seg_len {seg_len} > T {T}")
    if seg_len == T:
        return X_ts[None, :, :]
    starts = rng.integers(0, T-seg_len+1, size=n_repeats)
    return np.stack([X_ts[s:s+seg_len] for s in starts], axis=0)

if "series_by_scenario" not in globals():
    raise RuntimeError("series_by_scenario tidak ditemukan. Pastikan cell pembentukan skenario sudah dijalankan.")

_n_scen = len(series_by_scenario)
_expected_units = _n_scen * len(CV_DATA_LENGTHS) * len(F_by_method)

cv_table = None
if CV_RESUME and os.path.exists(CV_CHECKPOINT):
    _cached = pd.read_csv(CV_CHECKPOINT)
    _done = _cached.groupby(["scenario", "data_length", "method"]).ngroups
    if _done >= _expected_units:
        cv_table = _cached
        print(f"[resume] cv_table dimuat dari {CV_CHECKPOINT} ({len(cv_table)} baris)")

if cv_table is None:
    rows = []
    # one (scenario, length, method) unit; used to decide whether the next
    # scenario still fits in the wall-clock budget
    _unit_s = 60.0

    for _si, (scen_name, X_ts) in enumerate(series_by_scenario.items(), start=1):
        _need = _unit_s * len(CV_DATA_LENGTHS) * len(F_by_method)
        if not budget_ok(_need, f"CV stability ({scen_name} + sisanya)"):
            print(f"[budget] cv_table dipotong di skenario {_si}/{_n_scen}")
            break

        _t0 = time.perf_counter()
        for L in CV_DATA_LENGTHS:
            if L > X_ts.shape[0]:
                continue
            Wcv = sample_segments(X_ts, seg_len=L, n_repeats=CV_N_REPEATS, rng=rng_cv)
            for method_name in F_by_method.keys():
                Fcv = compute_features_entropy(Wcv, scales=scales, method=method_name,
                                              m=m, r_ratio=r_ratio, n_ref=n_ref,
                                              jsd_bins=jsd_bins, seed=CV_RANDOM_SEED,
                                              n_jobs=N_JOBS)
                _, cv_avg = entropy_cv_report(Fcv, S=S)
                # JSD-Fuzzy rich emits 4 features per scale, the others 1. Map the
                # feature index back onto the true scale so `scale` means the same
                # thing in every row of the table.
                feat_per_scale = max(1, len(cv_avg) // len(scales))
                for f_idx, cv_val in enumerate(cv_avg):
                    rows.append({
                        "scenario": scen_name,
                        "data_length": int(L),
                        "scale": int(f_idx // feat_per_scale) + 1,
                        "feat_in_scale": int(f_idx % feat_per_scale),
                        "method": method_name,
                        "cv": float(cv_val),
                        "n_repeats": int(Wcv.shape[0])
                    })

        _unit_s = max(1.0, (time.perf_counter() - _t0) / max(1, len(CV_DATA_LENGTHS) * len(F_by_method)))
        pd.DataFrame(rows).to_csv(CV_CHECKPOINT, index=False)  # checkpoint
        log_stage(f"CV stability {_si}/{_n_scen} ({scen_name}) "
                  f"| {time.perf_counter()-_t0:.0f}s")

    cv_table = pd.DataFrame(rows, columns=["scenario", "data_length", "scale",
                                           "feat_in_scale", "method", "cv", "n_repeats"])

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

_sort_by = [c for c in ["scenario", "data_length", "method", "scale", "feat_in_scale"]
            if c in cv_table.columns]
cv_table = cv_table.sort_values(_sort_by).reset_index(drop=True)
if len(cv_table) == 0:
    print("[warn] cv_table kosong — semua skenario dilewati oleh budget guard.")
cv_table
'''

# --- legacy 312-combo grid: gate it off ---------------------------------------
LEGACY_GRID_OLD = '''gs = GridSearchCV(
    pipe, param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=1,      # penting: hindari double-parallelism
    verbose=2
)
gs.fit(X_feat, y)
print("BEST:", gs.best_params_)
print("BEST f1_macro:", gs.best_score_)'''

LEGACY_GRID_NEW = '''# 13 arch x 2 act x 3 alpha x 2 lr x 2 batch = 312 combos x 5 folds = 1560 fits.
# The resulting `gs` is immediately overwritten by the multi-method cell below,
# so this only ever printed its best params. Off by default; flip
# RUN_LEGACY_BIG_GRID in the config cell if you want them.
if RUN_LEGACY_BIG_GRID and budget_ok(3600, "legacy 312-combo grid"):
    gs_legacy = GridSearchCV(
        pipe, param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=N_JOBS,
        verbose=2
    )
    gs_legacy.fit(X_feat, y)
    print("BEST:", gs_legacy.best_params_)
    print("BEST f1_macro:", gs_legacy.best_score_)
else:
    n_combos = (len(candidates) * len(param_grid["mlp__activation"])
                * len(param_grid["mlp__alpha"]) * len(param_grid["mlp__learning_rate_init"])
                * len(param_grid["mlp__batch_size"]))
    print(f"[skip] legacy grid ({n_combos} combos x 5 folds). "
          f"Set RUN_LEGACY_BIG_GRID=True untuk menjalankannya.")'''

# --- multi-method ANN grid ----------------------------------------------------
# Grid contents stay exactly as the original so the HEAD2HEAD anchor still
# matches the sibling notebooks. Only the parallelism changes.
ANN_GRID_OLD = '''    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1, scoring="accuracy", verbose=1)'''
ANN_GRID_NEW = '''    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=ANN_GRID_N_JOBS, scoring="accuracy", verbose=1)'''

ANN_RUN_OLD = '''if compare_methods:
    train_results = {}
    for name in methods:
        res, mtr = run_with_metrics(f"ANN {name}", lambda n=name: train_ann_for_F(F_by_method[n], n))
        train_results[name] = res
        train_metrics_by_method[name] = {**res["metrics"], **mtr}
    pd.DataFrame(train_metrics_by_method).T
    base_res = train_results[DEFAULT_METHOD]
else:
    edm_res, mtr = run_with_metrics(f"ANN {DEFAULT_METHOD}", lambda: train_ann_for_F(F, DEFAULT_METHOD))
    train_metrics_by_method[DEFAULT_METHOD] = {**edm_res["metrics"], **mtr}

gs = base_res["gs"]'''

ANN_RUN_NEW = '''if compare_methods:
    train_results = {}
    for name in methods:
        log_stage(f"ANN grid: {name}")
        res, mtr = run_with_metrics(f"ANN {name}", lambda n=name: train_ann_for_F(F_by_method[n], n))
        train_results[name] = res
        train_metrics_by_method[name] = {**res["metrics"], **mtr}
    pd.DataFrame(train_metrics_by_method).T
    base_res = train_results[DEFAULT_METHOD]
else:
    log_stage(f"ANN grid: {DEFAULT_METHOD}")
    base_res, mtr = run_with_metrics(f"ANN {DEFAULT_METHOD}", lambda: train_ann_for_F(F, DEFAULT_METHOD))
    train_results = {DEFAULT_METHOD: base_res}
    train_metrics_by_method[DEFAULT_METHOD] = {**base_res["metrics"], **mtr}

gs = base_res["gs"]'''

# --- per-scenario ANN search --------------------------------------------------
SCEN_GRID_OLD = '''    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=1, scoring="f1_macro", verbose=0)'''
SCEN_GRID_NEW = '''    gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=N_JOBS, scoring="f1_macro", verbose=0)'''

SCEN_GUARD_OLD = '''        if time_left_sec() is not None and time_left_sec() < 600:
            print("Time budget hampir habis, menghentikan per-scenario search.")
            break'''
SCEN_GUARD_NEW = '''        if not budget_ok(600, f"per-scenario search ({name})"):
            break'''

# --- RQ4 ----------------------------------------------------------------------
RQ4_GRID_OLD = '''    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=1, verbose=0)'''
RQ4_GRID_NEW = '''    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=N_JOBS, verbose=0)'''

RQ4_LOOP_OLD = '''        F_method_sub = F_by_method[method][keep_idx]

        print(f"▶ {sname} | {method} | n={len(y_scen)} | n_classes={len(np.unique(y_scen))} ...", end=" ", flush=True)
        try:'''

RQ4_LOOP_NEW = '''        F_method_sub = F_by_method[method][keep_idx]

        if not budget_ok(600, f"RQ4 {sname}/{method}"):
            continue

        print(f"▶ {sname} | {method} | n={len(y_scen)} | n_classes={len(np.unique(y_scen))} ...", end=" ", flush=True)
        try:'''

# --- S-sweep ------------------------------------------------------------------
SSWEEP_OLD = '''if "exp" not in globals():
    exp = run_experiment_S(
        W_s, y_s,
        S_list=S_list,
        m=2,
        r_ratio=0.2,
        n_ref=128,
        seed=7,
        use_time_split=USE_TIME_SPLIT
    )'''

SSWEEP_NEW = '''if "exp" not in globals():
    if not budget_ok(1800, "S-sweep (run_experiment_S)"):
        raise RuntimeError("Budget habis sebelum S-sweep. Naikkan KAGGLE_TIME_BUDGET_H "
                           "atau turunkan RUNTIME_PROFILE.")
    log_stage(f"S-sweep over {tuple(S_list)}")
    exp = run_experiment_S(
        W_s, y_s,
        S_list=S_list,
        m=2,
        r_ratio=0.2,
        n_ref=128,
        seed=7,
        use_time_split=USE_TIME_SPLIT
    )'''

# --- baseline CV --------------------------------------------------------------
BASELINE_OLD = '''    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring=scoring, n_jobs=1)'''
BASELINE_NEW = '''    scores = cross_val_score(pipe, X_feat, y, cv=cv, scoring=scoring, n_jobs=N_JOBS)'''

# `multi_class` was deprecated in scikit-learn 1.5 and removed in 1.7, which is
# what Kaggle's "Latest Container Image" now ships. "auto" was the default, so
# dropping the kwarg keeps the model identical.
LOGREG_OLD = '''    ("clf", LogisticRegression(max_iter=2000, multi_class="auto"))'''
LOGREG_NEW = '''    ("clf", LogisticRegression(max_iter=2000))'''


def src(cell):
    return ''.join(cell['source'])


def set_src(cell, text):
    cell['source'] = text.splitlines(keepends=True)


def find_cell(cells, needle, start=0):
    for i in range(start, len(cells)):
        if cells[i].get('cell_type') == 'code' and needle in src(cells[i]):
            return i
    raise SystemExit(f"anchor not found: {needle!r}")


def replace_in_cell(cells, anchor, old, new, label):
    i = find_cell(cells, anchor)
    text = src(cells[i])
    if new.strip() and new in text:
        print(f"  [skip] {label}: already patched")
        return
    if old not in text:
        raise SystemExit(f"[fail] {label}: old block not found in cell {i}")
    set_src(cells[i], text.replace(old, new, 1))
    print(f"  [ok]   {label} (cell {i})")


def apply_runtime_patches(nb):
    cells = nb['cells']

    # 1. runtime doc + guard, as the first cells after the title
    if not any('## Runtime & reproducibility' in src(c) for c in cells):
        cells.insert(1, {
            'cell_type': 'markdown',
            'metadata': {},
            'source': RUNTIME_DOC.splitlines(keepends=True),
        })
        print("  [ok]   inserted runtime-doc markdown cell at index 1")
    else:
        print("  [skip] runtime-doc cell already present")

    if not any(c.get('cell_type') == 'code' and 'NOTEBOOK_START' in src(c) for c in cells):
        cells.insert(2, {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': RUNTIME_CELL.splitlines(keepends=True),
        })
        print("  [ok]   inserted runtime-guard cell at index 2")
    else:
        print("  [skip] runtime-guard cell already present")

    replace_in_cell(cells, '# === Global config (edit here) ===',
                    CONFIG_OLD, CONFIG_NEW, 'config: RUNTIME_PROFILE knobs')
    replace_in_cell(cells, 'def sample_entropy_1d',
                    SAMPEN_OLD, SAMPEN_NEW, 'sample_entropy_1d: blocked + subsampled')
    replace_in_cell(cells, 'def cmse_1d',
                    CMSE_OLD, CMSE_NEW, 'cmse_1d: thread n_ref/seed through')
    replace_in_cell(cells, 'def ensure_window_3d',
                    FEAT_TIMER_OLD, FEAT_TIMER_NEW, 'feature cell: reuse global clock')
    replace_in_cell(cells, "if method_key == 'cmse':",
                    CMSE_DISPATCH_OLD, CMSE_DISPATCH_NEW, 'cmse dispatch: pass seed')
    replace_in_cell(cells, 'def run_with_metrics',
                    METRICS_OLD, METRICS_NEW, 'run_with_metrics: add rss_peak_mb')
    replace_in_cell(cells, 'cache_path = os.path.join(CACHE_DIR',
                    FEAT_LOOP_OLD, FEAT_LOOP_NEW, 'feature loop: progress log')

    # CV-vs-length cell: full rewrite
    i = find_cell(cells, CV_CELL_ANCHOR)
    if 'CV_CHECKPOINT' in src(cells[i]):
        print("  [skip] CV cell: already patched")
    else:
        set_src(cells[i], CV_CELL_NEW)
        print(f"  [ok]   CV cell rewritten (cell {i})")

    replace_in_cell(cells, '# ANN (MLP) evaluasi dengan metrik f1_macro',
                    LEGACY_GRID_OLD, LEGACY_GRID_NEW, 'legacy 312-combo grid: gated off')
    replace_in_cell(cells, 'def train_ann_for_F',
                    ANN_GRID_OLD, ANN_GRID_NEW, 'ANN grid: n_jobs (grid unchanged)')
    replace_in_cell(cells, '# Train/test split + GridSearch ANN',
                    ANN_RUN_OLD, ANN_RUN_NEW, 'ANN run: fix base_res NameError + log')
    replace_in_cell(cells, 'def fit_best_mlp',
                    SCEN_GRID_OLD, SCEN_GRID_NEW, 'per-scenario grid: n_jobs')
    replace_in_cell(cells, 'def balanced_binary_subsample',
                    SCEN_GUARD_OLD, SCEN_GUARD_NEW, 'per-scenario guard: budget_ok')
    replace_in_cell(cells, 'def rq4_run_scenario',
                    RQ4_GRID_OLD, RQ4_GRID_NEW, 'RQ4 grid: n_jobs (grid unchanged)')
    replace_in_cell(cells, '# === RQ4: Jalankan 5 Skenario',
                    RQ4_LOOP_OLD, RQ4_LOOP_NEW, 'RQ4 loop: budget guard')
    replace_in_cell(cells, '# === Robust experiment computation',
                    SSWEEP_OLD, SSWEEP_NEW, 'S-sweep: budget guard + log')
    replace_in_cell(cells, '"SVM-RBF(balanced)"',
                    BASELINE_OLD, BASELINE_NEW, 'baseline CV: n_jobs')
    replace_in_cell(cells, '# Baseline comparisons (trained on same split as ANN)',
                    LOGREG_OLD, LOGREG_NEW, 'LogisticRegression: drop removed multi_class kwarg')

    return nb


def main():
    nb = json.load(open(NB))
    print(f"patching {NB} ({len(nb['cells'])} cells)")
    apply_runtime_patches(nb)
    json.dump(nb, open(NB, 'w'), ensure_ascii=False, indent=1)
    print(f"WROTE {NB} ({len(nb['cells'])} cells)")


if __name__ == '__main__':
    sys.exit(main())
