#!/usr/bin/env python3
"""Build JSD_Fuzzy_TwoStage_SensorID.ipynb.

Fixes the gap flagged by Bu Luh (WhatsApp 2026-07-17):
  1) "sensor mana yang ada fault" -> belum ada jawaban yang di-chain dari hasil
     klasifikasi 17-kelas (JSD_Fuzzy_Broker_PerSensor.ipynb menjawabnya tapi
     lewat task multi-label terpisah, bukan lanjutan dari 17-kelas).
  2) "akurasi tetap per klasifikasi... 17 kelas di EDM dan 17 kelas di JSD,
     setelah itu identifikasi sensor yang salah" -> Stage 1 (17-kelas, sama
     seperti JSD_Fuzzy_ANN_Paper.ipynb) tetap dipertahankan utuh, lalu Stage 2
     (baru) mengidentifikasi sensor yang fault, dari split test yang SAMA.

Root cause kenapa belum bisa di-chain: `inject_faults_multisensor` di notebook
utama menyuntik fault ke SEMUA 4 sensor sekaligus untuk tiap skenario, jadi
label 17-kelas tidak membawa informasi sensor mana yang kena. Notebook ini
mengubah injeksi jadi sensor-selective (subset sensor acak per repeat), tanpa
mengubah definisi/nama 17 skenario, sehingga tersedia ground-truth per-sensor
yang genuin untuk Stage 2.
"""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# Two-Stage Fault Detection: 17-Kelas -> Identifikasi Sensor

Menjawab 2 pesan WhatsApp Bu Luh (2026-07-17):

1. *"sensor mana yang ada fault belum ya vin?"*
2. *"Akurasinya tetap dibuat per klasifikasi ya vin, setelah itu identifikasi
   sensor yang salah. jadi tetap ada 17 kelas di EDM dan 17 kelas di JSD"*

**Desain 2 stage (chained, dari split test yang sama):**
- **Stage 1** — klasifikasi 17 kelas (normal + 16 skenario fault kombinasi),
  identik dengan `JSD_Fuzzy_ANN_Paper.ipynb`: akurasi + classification report
  per kelas, dijalankan terpisah untuk **EDM-Fuzzy** dan **JSD-Fuzzy**.
- **Stage 2** (baru) — untuk window yang diklasifikasi fault oleh Stage 1,
  jalankan `MultiOutputClassifier` (4 label biner `[S1,S2,S3,S4]`) untuk
  menunjuk **sensor mana** yang benar-benar membawa fault tersebut.

**Kenapa belum bisa di-chain sebelumnya:** `inject_faults_multisensor` di
notebook utama menyuntik fault ke ke-4 sensor sekaligus untuk tiap skenario ->
label 17-kelas tidak membawa info "sensor mana". Fix di notebook ini:
injeksi jadi **sensor-selective** — tiap skenario diulang untuk beberapa
subset sensor acak (ukuran 1..4), sensor di luar subset tetap bersih. Nama
dan definisi 16 skenario (drift/spike/bias/hardware + kombinasinya) **tidak
diubah**, jadi Stage 1 tetap 17 kelas seperti sebelumnya.
"""))

cells.append(code("""# === Runtime guard — keep this as the FIRST executed cell ===
import os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

NOTEBOOK_START = time.time()
KAGGLE_TIME_BUDGET_H = float(os.environ.get("KAGGLE_TIME_BUDGET_H", 10.5))

def elapsed_s(): return time.time() - NOTEBOOK_START
def time_left_sec(): return KAGGLE_TIME_BUDGET_H * 3600.0 - elapsed_s()

def budget_ok(need_s=0.0, label=""):
    left = time_left_sec()
    if left < need_s:
        print(f"[budget] SKIP {label}: {left/60:.1f} min left, need ~{need_s/60:.1f} min", flush=True)
        return False
    return True

def log_stage(label):
    print(f"[t+{elapsed_s()/60:6.1f} min] {label}", flush=True)

log_stage(f"runtime guard armed | budget={KAGGLE_TIME_BUDGET_H} h | BLAS threads=1")
"""))

cells.append(code("""# === Global config ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, logging, time as _time, tracemalloc
from pathlib import Path
from IPython.display import FileLink, display

RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE", "fast")  # fast | paper
METHOD_LIST = ["EDM-Fuzzy", "JSD-Fuzzy"]   # Bu Luh: tetap 17 kelas di EDM dan di JSD
DEFAULT_METHOD = "EDM-Fuzzy"
N_JOBS = -1
ANN_GRID_N_JOBS = N_JOBS
EXPORT_DIR = "exports"
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

def export_df(df, name, index=False):
    p = Path(EXPORT_DIR) / f"{name}.csv"
    df.to_csv(p, index=index)
    display(FileLink(str(p)))
    return str(p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_with_metrics(label, fn):
    tracemalloc.start()
    t0 = _time.perf_counter(); c0 = _time.process_time()
    result = fn()
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    metrics = {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024)}
    logging.info("%s | wall=%.2fs cpu=%.2fs peak_mem=%.2f MB", label, metrics["wall_s"], metrics["cpu_s"], metrics["peak_mem_mb"])
    return result, metrics

print(f"RUNTIME_PROFILE={RUNTIME_PROFILE} | METHOD_LIST={METHOD_LIST}")
"""))

cells.append(code("""# === Speed / sampling toggles (Kaggle-friendly defaults) ===
DS = 4
WIN = 256
STRIDE = 128
MAX_PER_CLASS = 200       # max window per 17-kelas SETELAH balanced subsample
RANDOM_SEED = 42
# spike fault ticks every tau=int(1/p)=66 sample -> ratio-per-window caps at
# ~1/66=0.0152 regardless of WIN once WIN>=tau. thr must stay below that or
# the "spike" scenario silently produces zero windows (verified locally).
FAULT_RATIO_THR = 0.01

# Stage 2 (sensor-selective injection): tiap skenario diulang untuk beberapa
# ukuran subset sensor supaya ground-truth per-sensor bervariasi (1..4 sensor
# kena fault), bukan selalu ke-4 sensor sekaligus.
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]
SENSORS = ["S1", "S2", "S3", "S4"]

S = 10; m = 2; r_ratio = 0.2; n_ref = 128; jsd_bins = 40
scales = np.arange(1, S + 1)

print("config:", dict(DS=DS, WIN=WIN, STRIDE=STRIDE, MAX_PER_CLASS=MAX_PER_CLASS,
                       SENSOR_SUBSET_SIZES=SENSOR_SUBSET_SIZES, S=S))
"""))

cells.append(md("# Load Data (4 Sensor) — broker output = satu tabel gabungan"))

cells.append(code("""import requests
from io import StringIO

def load_default_data():
    url = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/tabel_sensor4_generated.csv"
    r = requests.get(url); r.raise_for_status()
    return pd.read_csv(StringIO(r.text))

df = load_default_data()
cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
X_df = pd.DataFrame(df[cols].to_numpy(dtype=float), columns=cols).ffill().bfill()
X_df = X_df.fillna(X_df.median(numeric_only=True))
if X_df.isna().any().any():
    raise ValueError("Error-nya jelas: X masih ada NaN setelah imputasi.")
X = X_df.to_numpy()
X_ds = X[::DS]
print("Columns:", cols, "| Shape X:", X.shape, "-> downsampled X_ds:", X_ds.shape)
"""))

cells.append(md("""# Fault Injection — Sensor-Selective (17 skenario tetap sama)

Simulator fault dan `SCENARIOS` (16 kombinasi + normal = 17 kelas) **identik**
dengan `JSD_Fuzzy_ANN_Paper.ipynb`. Yang berubah hanya `inject_faults_multisensor`:
sekarang menerima `sensor_subset` — hanya sensor dalam subset yang disuntik
fault, sensor lain tetap bersih. Tiap skenario diulang untuk beberapa ukuran
subset (`SENSOR_SUBSET_SIZES`) supaya window punya keragaman "berapa & sensor
mana" yang kena, dan `window_fault_label_per_sensor` merekam ground-truth
per-sensor per-window untuk Stage 2.
"""))

cells.append(code("""# --- Fault simulators (identik notebook utama) ---
def simulate_drift_fault(x, intensity=0.02, seed=None):
    drift = np.arange(len(x)) * intensity
    return x + drift, np.abs(drift) > 1e-6

def simulate_spike_fault(x, intensity=0.08, p=0.015, seed=None):
    tau = max(1, int(1.0 / p)) if p > 0 else len(x)
    spikes = (np.arange(len(x)) % tau == 0).astype(float) * (intensity * np.nanstd(x))
    return x + spikes, spikes != 0

def simulate_bias_fault(x, bias=0.08, seed=None):
    return x + bias, np.ones(len(x), bool)

def simulate_hardware_fault(x, stuck_prob=0.08, loss_prob=0.05, seed=None):
    rng = np.random.default_rng(seed)
    n = len(x); rv = rng.random(n); idx = rng.integers(n, size=n)
    m1 = rv < stuck_prob; y = x.copy(); y[m1] = x[idx[m1]]
    m2 = rv < loss_prob; y[m2] = np.nan
    return y, (m1 | m2)

def simulate_multiple_faults(x, faults, seed=None):
    y = x.copy(); m_ = np.zeros(len(x), bool)
    for f, kw in faults:
        y, mi = f(y, **kw, seed=seed); m_ |= mi
    return y, m_

def simulate_choose_one(x, options, seed=None):
    rng = np.random.default_rng(seed)
    f, kw = options[rng.integers(len(options))]
    return f(x, **kw, seed=seed)

# 17 skenario (1 normal + 16 fault) -- IDENTIK JSD_Fuzzy_ANN_Paper.ipynb
SCENARIOS = {
    "faulty": [(simulate_choose_one, {"options": [
        (simulate_drift_fault, {"intensity": 0.02}),
        (simulate_spike_fault, {"intensity": 0.08, "p": 0.015}),
        (simulate_bias_fault, {"bias": 0.08}),
        (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05}),
    ]})],
    "drift": [(simulate_drift_fault, {"intensity": 0.02})],
    "spike": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015})],
    "bias": [(simulate_bias_fault, {"bias": 0.08})],
    "hardware": [(simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "bias+malfunc": [(simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+bias": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08})],
    "drift+malfunc": [(simulate_drift_fault, {"intensity": 0.02}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+bias": [(simulate_drift_fault, {"intensity": 0.02}), (simulate_bias_fault, {"bias": 0.08})],
    "drift+spike": [(simulate_drift_fault, {"intensity": 0.02}), (simulate_spike_fault, {"intensity": 0.08, "p": 0.015})],
    "spike+bias+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+bias+malfunc": [(simulate_drift_fault, {"intensity": 0.02}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+drift+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_drift_fault, {"intensity": 0.02}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+spike+bias": [(simulate_drift_fault, {"intensity": 0.02}), (simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08})],
    "spike+bias+malfunc+drift": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05}), (simulate_drift_fault, {"intensity": 0.02})],
}
print("Skenario:", len(SCENARIOS), "+ normal =", len(SCENARIOS) + 1, "kelas")
"""))

cells.append(code("""from numpy.lib.stride_tricks import sliding_window_view

def make_windows(X, win, stride):
    Xn = np.asarray(X, dtype=np.float32); N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    view = sliding_window_view(Xn, window_shape=win, axis=0)
    starts = np.arange(0, N - win + 1, stride, dtype=int)
    return view[starts], starts

def inject_faults_multisensor(X, scenario_faults, sensor_subset, seed=0):
    # X: (T,4) -> Y: (T,4), M: (T,4). Hanya sensor di `sensor_subset` disuntik.
    rng = np.random.default_rng(seed)
    Y = X.copy(); M = np.zeros_like(Y, dtype=bool)
    for s in sensor_subset:
        y, m_ = simulate_multiple_faults(Y[:, s], scenario_faults, seed=int(rng.integers(1e9)))
        Y[:, s] = y; M[:, s] = m_
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M

def window_fault_label(mask, win, stride, thr=0.02):
    # sliding_window_view(mask, window_shape=win, axis=0) on a (T,4) array
    # returns (Nwin, 4, win) -- channel axis stays put, win axis is appended
    # at the END (NOT (Nwin, win, 4)). Mean over axis=2 (time), not axis=1.
    T = len(mask)
    if win > T:
        return np.zeros(0, dtype=bool), np.array([], dtype=int)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]  # (Nwin, 4, win)
    ratio = Wm.mean(axis=2)  # (Nwin, 4)
    return (ratio > thr).any(axis=1), np.arange(0, T - win + 1, stride, dtype=int)

def window_fault_label_per_sensor(mask, win, stride, thr=0.02):
    T = len(mask)
    if win > T:
        return np.zeros((0, mask.shape[1]), dtype=bool)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]  # (Nwin, 4, win)
    return (Wm.mean(axis=2) > thr)  # (Nwin, 4)

rng_master = np.random.default_rng(RANDOM_SEED)

datasets, labels, sens_labels = [], [], []
scenario_names = ["normal"] + list(SCENARIOS.keys())

# normal: tidak ada fault -> ground-truth sensor semua False
W0, _ = make_windows(X_ds, WIN, STRIDE)
datasets.append(W0); labels.append(np.zeros(len(W0), dtype=int))
sens_labels.append(np.zeros((len(W0), 4), dtype=bool))

for k, (name, faults) in enumerate(SCENARIOS.items(), start=1):
    n_scenario = 0
    for subset_size in SENSOR_SUBSET_SIZES:
        subset = rng_master.choice(4, size=subset_size, replace=False)
        seed_k = int(rng_master.integers(1e9))
        Y, M = inject_faults_multisensor(X_ds, faults, subset, seed=seed_k)
        is_fault_win, _ = window_fault_label(M, WIN, STRIDE, thr=FAULT_RATIO_THR)
        sens_win = window_fault_label_per_sensor(M, WIN, STRIDE, thr=FAULT_RATIO_THR)
        Wk, _ = make_windows(Y, WIN, STRIDE)
        Wk = Wk[is_fault_win]; sens_win = sens_win[is_fault_win]
        datasets.append(Wk); labels.append(np.full(len(Wk), k, dtype=int)); sens_labels.append(sens_win)
        n_scenario += len(Wk)
    print(f"{name:28s} subset_sizes={SENSOR_SUBSET_SIZES} windows_total={n_scenario}")

W_all = np.concatenate(datasets, axis=0)
y_all = np.concatenate(labels, axis=0)
Ysens_all = np.concatenate(sens_labels, axis=0).astype(int)  # (N,4) ground-truth sensor fault

if W_all.ndim == 3 and W_all.shape[1] == 4 and W_all.shape[2] == WIN:
    W_all = W_all.transpose(0, 2, 1)

print("Total windows:", W_all.shape, "| 17-class counts:", dict(zip(*np.unique(y_all, return_counts=True))))
print("Per-sensor fault prevalence (all windows):", Ysens_all.mean(axis=0).round(3))
"""))

cells.append(code("""def balanced_subsample_multi(Xw, y, Ysens, max_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return Xw[keep], y[keep], Ysens[keep]

W_s, y_s, Ysens_s = balanced_subsample_multi(W_all, y_all, Ysens_all, max_per_class=MAX_PER_CLASS, seed=RANDOM_SEED)

if W_s.ndim != 3:
    raise ValueError(f"Error: W_s harus 3D (N, WIN, 4). Shape={W_s.shape}.")

print("After balanced subsample:", W_s.shape, "| 17-class:", dict(zip(*np.unique(y_s, return_counts=True))))
print("Per-sensor fault prevalence (subsampled):", Ysens_s.mean(axis=0).round(3))
"""))

cells.append(md("""# Entropy Features — EDM-Fuzzy & JSD-Fuzzy (rich)

Fungsi identik `JSD_Fuzzy_ANN_Paper.ipynb`. JSD-Fuzzy pakai varian *rich*
(`[jsd, fe, mean_m, std_m]`/skala, 4 fitur/skala) yang terbukti menaikkan
akurasi 17-kelas dari ~0.34 ke ~0.42 dibanding varian lama (1 fitur/skala).
"""))

cells.append(code("""def coarse_grain_mean(x, s):
    n = (len(x) // s) * s
    return x[:n].reshape(-1, s).mean(axis=1) if n > 0 else np.array([], dtype=float)

def embed_matrix(y, m_):
    L = len(y)
    return np.lib.stride_tricks.sliding_window_view(y, m_) if L >= m_ else np.empty((0, m_), dtype=float)

def fuzzy_phi(V, r, n_ref=256, seed=0):
    rng = np.random.default_rng(seed); N = V.shape[0]
    if N < 3: return np.nan
    ref = rng.choice(N, size=n_ref, replace=False) if N > n_ref else np.arange(N)
    A = V[ref]; a2 = np.sum(A * A, axis=1, keepdims=True); b2 = np.sum(V * V, axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2 * (A @ V.T), 0.0); rr = r * r + 1e-24
    mu = 1.0 / (1.0 + d2 / rr); mu[np.arange(len(ref)), ref] = 0.0
    return (mu.sum(axis=1) / (N - 1)).mean()

def fuzzy_similarity_samples(V, r, n_ref=256, seed=0):
    rng = np.random.default_rng(seed); N = V.shape[0]
    if N < 3: return np.array([], dtype=float)
    idx = np.arange(N); ref = rng.choice(idx, size=n_ref, replace=False) if N > n_ref else idx
    A = V[ref]; a2 = np.sum(A * A, axis=1, keepdims=True); b2 = np.sum(V * V, axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2 * (A @ V.T), 0.0); d = np.sqrt(d2)
    mu = 1.0 / (1.0 + (d / (r + 1e-12)) ** 2)
    for ri, i in enumerate(ref):
        mu[ri, i] = np.nan
    return mu[~np.isnan(mu)].ravel()

def edm_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=256, seed=0):
    out = []
    for s in scales:
        y = coarse_grain_mean(x, s)
        if len(y) < (m + 2):
            out.append(np.nan); continue
        r = r_ratio * np.std(y, ddof=1)
        phi_m = fuzzy_phi(embed_matrix(y, m), r, n_ref=n_ref, seed=seed + 11 * s)
        phi_m1 = fuzzy_phi(embed_matrix(y, m + 1), r, n_ref=n_ref, seed=seed + 17 * s)
        if not phi_m or not phi_m1 or phi_m <= 0 or phi_m1 <= 0 or np.isnan(phi_m) or np.isnan(phi_m1):
            out.append(np.nan)
        else:
            out.append(np.log(phi_m / phi_m1))
    return np.array(out, dtype=float)

def jsd_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=256, seed=0, bins=20):
    out = []; bin_edges = np.linspace(0.0, 1.0, bins + 1); eps = 1e-12
    for s in scales:
        y = coarse_grain_mean(x, s)
        if len(y) < (m + 2):
            out.extend([np.nan] * 4); continue
        r = r_ratio * np.std(y, ddof=1)
        mu_m = fuzzy_similarity_samples(embed_matrix(y, m), r, n_ref=n_ref, seed=seed + 11 * s)
        mu_m1 = fuzzy_similarity_samples(embed_matrix(y, m + 1), r, n_ref=n_ref, seed=seed + 17 * s)
        if len(mu_m) == 0 or len(mu_m1) == 0:
            out.extend([np.nan] * 4); continue
        p, _ = np.histogram(mu_m, bins=bin_edges); q, _ = np.histogram(mu_m1, bins=bin_edges)
        p = p.astype(float); q = q.astype(float)
        if p.sum() == 0 or q.sum() == 0:
            out.extend([np.nan] * 4); continue
        p /= p.sum(); q /= q.sum(); mm = 0.5 * (p + q)
        jsd = 0.5 * (np.sum(p * np.log((p + eps) / (mm + eps))) + np.sum(q * np.log((q + eps) / (mm + eps))))
        fe = np.log((mu_m.mean() + eps) / (mu_m1.mean() + eps))
        out.extend([jsd, fe, mu_m.mean(), mu_m.std()])
    return np.array(out, dtype=float)

# clamp scales so coarse-graining never goes empty for this WIN
min_len = m + 2; max_scale = max(1, WIN // min_len)
scales = np.array([s for s in scales if s <= max_scale] or [1], dtype=int)
S = len(scales)
print("Using scales:", scales.tolist())
"""))

cells.append(code("""from joblib import Parallel, delayed

def sanitize_features(F, name="F"):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()

def compute_features_entropy(W, scales, method, m=2, r_ratio=0.2, n_ref=256, jsd_bins=20, seed=0, n_jobs=-1):
    Nwin, win, ns = W.shape
    method_key = method.strip().lower()

    def entropy_1d(x, seed_local):
        if method_key == "edm-fuzzy":
            return edm_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=seed_local)
        if method_key == "jsd-fuzzy":
            return jsd_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=seed_local, bins=jsd_bins)
        raise ValueError(f"Unknown method: {method}")

    def one_window(i):
        feats = [entropy_1d(W[i, :, s], seed_local=seed + 1000 * i + 19 * s) for s in range(ns)]
        return np.concatenate(feats, axis=0)

    F = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one_window)(i) for i in range(Nwin))
    return np.vstack(F)

F_by_method = {}
for name in METHOD_LIST:
    log_stage(f"entropy features: {name} ({W_s.shape[0]} windows)")
    Fm, mtr = run_with_metrics(f"Entropy {name}", lambda n=name: compute_features_entropy(
        W_s, scales=scales, method=n, m=m, r_ratio=r_ratio, n_ref=n_ref, jsd_bins=jsd_bins, seed=7, n_jobs=N_JOBS))
    F_by_method[name] = sanitize_features(Fm, name=f"F_{name}")
    print(name, "entropy feature shape:", F_by_method[name].shape)
"""))

cells.append(md("""# Fitur Hibrida (Entropy + Time-Domain)

Sama seperti `JSD_Fuzzy_Broker_PerSensor.ipynb`: entropy murni digabung 12
fitur time-domain/sensor (mean/std/rms/ptp/mad/skew/kurt/slope/maxΔ/trend/zcr/
HF-energy) supaya offset/bias (yang entropy buta) tetap tertangkap fitur level.
Fitur hibrida ini dipakai untuk **Stage 1 dan Stage 2**.
"""))

cells.append(code("""from scipy import stats as _sstats

def compute_time_features(W):
    N, WINl, ns = W.shape
    t = np.arange(WINl, dtype=float); tc = t - t.mean(); tv = (tc ** 2).mean() + 1e-12
    feats = []
    for s in range(ns):
        x = np.asarray(W[:, :, s], dtype=float)
        mu = x.mean(1); sd = x.std(1); rms = np.sqrt((x ** 2).mean(1))
        ptp = x.max(1) - x.min(1); mad = np.abs(x - mu[:, None]).mean(1)
        sk = _sstats.skew(x, axis=1, bias=False); ku = _sstats.kurtosis(x, axis=1, bias=False)
        d = np.diff(x, axis=1); sm = np.abs(d).mean(1); smx = np.abs(d).max(1)
        tr = (x * tc[None, :]).mean(1) / tv
        sc = np.sign(x - mu[:, None]); zcr = (np.abs(np.diff(sc, axis=1)) > 0).mean(1)
        xf = np.abs(np.fft.rfft(x - mu[:, None], axis=1)); pw = xf ** 2
        half = max(1, pw.shape[1] // 2); hf = pw[:, half:].sum(1) / (pw.sum(1) + 1e-12)
        feats.extend([mu, sd, rms, ptp, mad, sk, ku, sm, smx, tr, zcr, hf])
    T = np.column_stack(feats)
    return np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

T_feats = compute_time_features(W_s)
FEAT_by_method = {name: np.hstack([F_by_method[name], T_feats]) for name in METHOD_LIST}
for name, Fh in FEAT_by_method.items():
    print(name, "hybrid feature shape:", Fh.shape)
"""))

cells.append(md("""# Stage 1 — Klasifikasi 17 Kelas (EDM-Fuzzy vs JSD-Fuzzy)

Jawaban langsung untuk pesan Bu Luh #2: akurasi & classification report per
kelas, **17 kelas di EDM-Fuzzy** dan **17 kelas di JSD-Fuzzy**, split/seed
identik untuk keduanya.
"""))

cells.append(code("""from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              roc_auc_score, hamming_loss, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

TEST_FRAC = 0.25

# split index sekali -> dipakai ulang di Stage 2 supaya benar-benar "chained"
# dari test set yang sama (bukan split baru).
idx_all = np.arange(len(y_s))
idx_tr, idx_te = train_test_split(idx_all, test_size=TEST_FRAC, random_state=RANDOM_SEED, stratify=y_s)

def build_hidden_candidates(I, O):
    base = sorted(set([max(8, I // 4), max(16, I // 2), max(32, int(np.floor((2/3)*I + O))), I, min(2*I, 512)]))
    cand = [(h,) for h in base]
    for h1 in base:
        for h2 in sorted(set([max(4, h1 // 4), max(8, h1 // 2)])):
            cand.append((h1, h2))
    return list(dict.fromkeys(cand))[:20]

def train_ann_17class(F_local, name):
    Xtr, Xte, ytr, yte = F_local[idx_tr], F_local[idx_te], y_s[idx_tr], y_s[idx_te]
    I = Xtr.shape[1]; O = len(np.unique(y_s))
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("mlp", MLPClassifier(max_iter=400, random_state=RANDOM_SEED, early_stopping=True, n_iter_no_change=10))])
    grid = {"mlp__hidden_layer_sizes": build_hidden_candidates(I, O),
            "mlp__alpha": [1e-4, 1e-3, 1e-2], "mlp__learning_rate_init": [1e-3, 3e-4], "mlp__activation": ["relu", "tanh"]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    gs = GridSearchCV(pipe, grid, cv=cv, n_jobs=ANN_GRID_N_JOBS, scoring="accuracy")
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_; pred = best.predict(Xte)
    metrics = {"best_params": gs.best_params_, "best_cv_acc": float(gs.best_score_),
               "test_acc": float(accuracy_score(yte, pred)), "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0))}
    return {"gs": gs, "Xte": Xte, "yte": yte, "pred": pred, "metrics": metrics}

train_results = {}
for name in METHOD_LIST:
    log_stage(f"Stage1 ANN grid: {name}")
    res, mtr = run_with_metrics(f"Stage1 ANN {name}", lambda n=name: train_ann_17class(FEAT_by_method[n], n))
    train_results[name] = res
    print(name, "| test_acc:", round(res["metrics"]["test_acc"], 4), "| macro_f1:", round(res["metrics"]["macro_f1"], 4))

stage1_summary = pd.DataFrame({n: r["metrics"] for n, r in train_results.items()}).T
export_df(stage1_summary.reset_index().rename(columns={"index": "Method"}), "stage1_17class_summary")
stage1_summary
"""))

cells.append(code("""# === Stage 1: classification report per kelas (EDM-Fuzzy & JSD-Fuzzy) ===
for name, res in train_results.items():
    print(f"=== 17-kelas | {name} ===")
    print(classification_report(res["yte"], res["pred"], labels=np.arange(len(scenario_names)),
                                  target_names=scenario_names, zero_division=0))
    print(f"Accuracy: {res['metrics']['test_acc']:.4f}  Macro-F1: {res['metrics']['macro_f1']:.4f}\\n")
    cm = confusion_matrix(res["yte"], res["pred"], labels=np.arange(len(scenario_names)))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=scenario_names).plot(xticks_rotation=45, ax=ax, colorbar=False)
    ax.set_title(f"Stage 1 — Confusion Matrix 17-kelas — {name}")
    plt.tight_layout(); plt.savefig(f"exports/stage1_cm_{name.replace('-','_')}.png", dpi=120); plt.show()
"""))

cells.append(md("""# Stage 2 — Identifikasi Sensor (chained dari Stage 1)

Untuk window yang berlabel fault (kelas != normal) di **split test yang sama**
(`idx_te`), latih & evaluasi `MultiOutputClassifier(MLPClassifier)` -> 4
keputusan biner `[S1,S2,S3,S4]` menunjuk sensor yang benar-benar fault.

Dua varian evaluasi:
- **oracle**: filter faulty window pakai label 17-kelas TRUE (batas atas performa Stage 2).
- **end-to-end**: filter pakai label 17-kelas PREDIKSI Stage 1 (angka realistis, termasuk error Stage 1 yang ikut terbawa).
"""))

cells.append(code("""def train_sensor_id(F_local, Ysens_local, y_true_local, y_pred_stage1, seed=RANDOM_SEED):
    Xtr, Xte = F_local[idx_tr], F_local[idx_te]
    Ytr, Yte = Ysens_local[idx_tr], Ysens_local[idx_te]
    ytr_true = y_true_local[idx_tr]
    yte_true = y_true_local[idx_te]

    tr_mask = ytr_true != 0  # train sensor-ID hanya dari window fault (ground truth)
    Xtr_f, Ytr_f = Xtr[tr_mask], Ytr[tr_mask]

    base = MLPClassifier(max_iter=300, random_state=seed, early_stopping=True, n_iter_no_change=12)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("mo", MultiOutputClassifier(base))])
    I = Xtr_f.shape[1]
    grid = {"mo__estimator__hidden_layer_sizes": [(max(16, I // 2),), (I,), (max(16, I // 2), max(8, I // 4))],
            "mo__estimator__alpha": [1e-4, 1e-3]}
    gs = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro", n_jobs=N_JOBS)
    gs.fit(Xtr_f, Ytr_f)
    best = gs.best_estimator_

    def eval_on(mask, tag):
        if mask.sum() == 0:
            return None
        Xe, Ye = Xte[mask], Yte[mask]
        pred = best.predict(Xe)
        proba = np.column_stack([e.predict_proba(Xe)[:, 1] for e in best.named_steps["mo"].estimators_])
        per = {}
        for j, sname in enumerate(SENSORS):
            p, r, f, _ = precision_recall_fscore_support(Ye[:, j], pred[:, j], average="binary", zero_division=0)
            try: auc = roc_auc_score(Ye[:, j], proba[:, j])
            except Exception: auc = np.nan
            per[sname] = dict(Acc=accuracy_score(Ye[:, j], pred[:, j]), Prec=p, Rec=r, F1=f, AUC=auc)
        return dict(tag=tag, n=int(mask.sum()), per=per,
                    subset_acc=accuracy_score(Ye, pred), hamming=hamming_loss(Ye, pred),
                    macro_f1=f1_score(Ye, pred, average="macro", zero_division=0))

    oracle = eval_on(yte_true != 0, "oracle")
    end2end = eval_on(y_pred_stage1 != 0, "end_to_end")
    return oracle, end2end

sensor_rows = []
sensor_results = {}
for name in METHOD_LIST:
    log_stage(f"Stage2 sensor-ID grid: {name}")
    (oracle, e2e), mtr = run_with_metrics(f"Stage2 sensor-ID {name}", lambda n=name: train_sensor_id(
        FEAT_by_method[n], Ysens_s, y_s, train_results[n]["pred"]))
    sensor_results[name] = dict(oracle=oracle, end_to_end=e2e)
    for res in (oracle, e2e):
        if res is None: continue
        for sname, m in res["per"].items():
            sensor_rows.append({"Method": name, "Eval": res["tag"], "Sensor": sname, "n_test": res["n"],
                                 "Accuracy": round(m["Acc"], 3), "Precision": round(m["Prec"], 3),
                                 "Recall": round(m["Rec"], 3), "F1": round(m["F1"], 3),
                                 "ROC_AUC": round(m["AUC"], 3) if not np.isnan(m["AUC"]) else np.nan})
        sensor_rows.append({"Method": name, "Eval": res["tag"], "Sensor": "SUBSET(all4)", "n_test": res["n"],
                             "Accuracy": round(res["subset_acc"], 3), "Precision": np.nan, "Recall": np.nan,
                             "F1": round(res["macro_f1"], 3), "ROC_AUC": np.nan})

sensor_summary = pd.DataFrame(sensor_rows)
print(sensor_summary.to_string(index=False))
export_df(sensor_summary, "stage2_sensor_id_summary")
"""))

cells.append(code("""# === Plot: F1 per-sensor, EDM-Fuzzy vs JSD-Fuzzy (end-to-end, chained dari Stage 1) ===
fig, ax = plt.subplots(figsize=(8, 5))
sub = sensor_summary[(sensor_summary.Eval == "end_to_end") & (sensor_summary.Sensor != "SUBSET(all4)")]
piv = sub.pivot(index="Sensor", columns="Method", values="F1").reindex(SENSORS)
piv.plot.bar(ax=ax, rot=0)
ax.set_title("Stage 2 — F1 per sensor (end-to-end, chained dari Stage 1)")
ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
plt.tight_layout(); plt.savefig("exports/stage2_f1_per_sensor.png", dpi=120); plt.show()
"""))

cells.append(md("""## Ringkasan untuk Bu Luh

- **Pesan 1** ("sensor mana yang ada fault") -> dijawab Stage 2 di atas: per
  window fault, model menunjuk sensor `[S1,S2,S3,S4]` mana yang kena, dengan
  F1/Precision/Recall/ROC-AUC per sensor (tabel `stage2_sensor_id_summary.csv`).
- **Pesan 2** ("akurasi tetap per klasifikasi, 17 kelas EDM & 17 kelas JSD,
  setelah itu identifikasi sensor") -> Stage 1 tetap 17 kelas utuh untuk kedua
  metode (tabel `stage1_17class_summary.csv` + classification report di atas),
  dan Stage 2 dijalankan **setelah itu**, dari split test yang sama (chained,
  bukan task terpisah seperti `JSD_Fuzzy_Broker_PerSensor.ipynb`).
- Angka `end_to_end` di Stage 2 sudah termasuk error Stage 1 yang ikut terbawa
  (window yang salah diklasifikasi 17-kelas ikut masuk evaluasi sensor-ID);
  `oracle` adalah batas atas kalau Stage 1 sempurna.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9"}}, "nbformat": 4, "nbformat_minor": 5}

with open("/Users/kelvin/apps/public-files/JSD_Fuzzy_TwoStage_SensorID.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote JSD_Fuzzy_TwoStage_SensorID.ipynb | cells:", len(cells))
