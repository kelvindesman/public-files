#!/usr/bin/env python3
"""Build the "Cara B" notebook: which sensor carries WHICH fault type.

Answers Bu Luh's voice note (2026-07-19 17.30): "kalau kita pakai kombinasi 3
fault, dari situ kita tahu biasnya ada di sensor mana, atau driftnya ada di
sensor mana".

Why the earlier notebooks could not answer it:
  - archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb  -> injection faulted ALL 4 sensors at once,
    so its 4 label columns were identical (prevalence 0.882 on every sensor,
    ROC-AUC 0.5). Its 0.925 F1 is an artefact of always predicting "fault" on an
    88%-positive column, NOT sensor discrimination.
  - 04_Klasifikasi_Dulu_Baru_Sensor_Mana.ipynb -> fixed that with sensor-SELECTIVE
    injection, so "which sensor" became answerable. But every faulted sensor
    still received the SAME fault combination, so "which fault type on which
    sensor" still had no ground truth.

This notebook changes the data generation once more: each fault type present in
a scenario is assigned to its OWN random subset of sensors. So in scenario
"drift+bias", drift may hit sensors {1,3} while bias hits {2,3} -- exactly the
situation Bu Luh describes. Ground truth becomes a (sensor x fault-type) matrix
= 16 binary labels per window.

Stage 1 (17-class, which fault types are present in the system) is kept intact.
Stage 2 predicts the 16 labels and is reported three ways:
  - per (sensor, fault type)   -> "bias-nya di sensor mana"
  - per sensor (any fault)     -> "sensor mana yang rusak"
  - per fault type (any sensor)-> "fault jenis apa yang ada"
"""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# 05 — Fault Jenis Apa, di Sensor Mana? (Cara B)

**Pertanyaan yang dibuktikan notebook ini:**
Kalau ada kombinasi beberapa fault sekaligus, bisakah sistem menunjuk
*bias-nya ada di sensor mana* dan *drift-nya ada di sensor mana* — bukan cuma
"sensor ini rusak"?

**Kenapa notebook sebelumnya belum bisa menjawab:**

| Notebook | Cara injeksi fault | Yang bisa dijawab |
|---|---|---|
| PerSensor (lama, diarsipkan) | keempat sensor di-fault **bersamaan** | ❌ tidak ada — keempat kolom label identik (prevalensi 0,882 semua, ROC-AUC 0,5). Angka F1 0,925-nya artefak "selalu tebak fault" |
| `04` Two-Stage | fault kena **sebagian sensor** (acak) | ✅ sensor mana yang rusak |
| `05` ini | **tiap jenis fault** punya subset sensornya sendiri | ✅ jenis fault apa, di sensor mana |

**Perubahan inti:** pada skenario `drift+bias`, drift bisa kena sensor {1,3}
sedangkan bias kena sensor {2,3}. Jadi ground-truth-nya matriks
(4 sensor × 4 jenis fault) = **16 label biner per window**.

**Alur:**
- **Tahap 1** — klasifikasi 17 kelas (kombinasi jenis fault apa yang ada di sistem). Tidak diubah.
- **Tahap 2** — prediksi 16 label, dilaporkan 3 sudut pandang:
  1. per (sensor × jenis fault) → *"bias-nya di sensor mana"*
  2. per sensor (fault apa saja) → *"sensor mana yang rusak"*
  3. per jenis fault (sensor mana saja) → *"jenis fault apa yang muncul"*
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

RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE", "fast")
METHOD_LIST = ["EDM-Fuzzy", "JSD-Fuzzy"]
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

cells.append(code("""# === Speed / sampling toggles ===
DS = 4
WIN = 256
STRIDE = 128
MAX_PER_CLASS = 200
RANDOM_SEED = 42
# spike fault ticks every tau=int(1/p)=66 samples -> per-window fault ratio caps
# at ~1/66=0.0152. Threshold must stay below that or the "spike" scenario yields
# zero windows.
FAULT_RATIO_THR = 0.01

# Cara B: tiap skenario diulang REPEATS_PER_SCENARIO kali; tiap pengulangan,
# SETIAP jenis fault dalam skenario itu diberi subset sensor acak sendiri.
# Ini yang bikin "drift di sensor A, bias di sensor B" punya ground truth.
REPEATS_PER_SCENARIO = 4

SENSORS = ["S1", "S2", "S3", "S4"]
FAULT_TYPES = ["drift", "spike", "bias", "hardware"]

S = 10; m = 2; r_ratio = 0.2; n_ref = 128; jsd_bins = 40
scales = np.arange(1, S + 1)

print("config:", dict(DS=DS, WIN=WIN, STRIDE=STRIDE, MAX_PER_CLASS=MAX_PER_CLASS,
                       REPEATS_PER_SCENARIO=REPEATS_PER_SCENARIO, FAULT_TYPES=FAULT_TYPES))
"""))

cells.append(md("# Load Data (4 Sensor) — output broker = satu tabel gabungan"))

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

cells.append(md("""# Injeksi Fault — Tiap Jenis Fault Punya Sensornya Sendiri

Definisi 17 skenario (nama & jenis fault-nya) **tidak diubah** dari notebook
paper, supaya Tahap 1 tetap sebanding. Yang berubah cuma **penempatannya**:

- Notebook `04`: skenario `drift+bias` → pilih subset sensor, tiap sensor terpilih
  dapat drift **dan** bias sekaligus.
- Notebook `05` (ini): drift dapat subset sensornya sendiri, bias dapat subset
  sensornya sendiri, dipilih independen.

Ground truth jadi matriks `(window, 4 sensor, 4 jenis fault)` → 16 label biner.
"""))

cells.append(code("""# --- Fault simulators (identik notebook paper) ---
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

# satu simulator per JENIS fault -> dipakai untuk menempatkan fault per sensor
SIMULATOR = {
    "drift":    (simulate_drift_fault,    {"intensity": 0.02}),
    "spike":    (simulate_spike_fault,    {"intensity": 0.08, "p": 0.015}),
    "bias":     (simulate_bias_fault,     {"bias": 0.08}),
    "hardware": (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05}),
}

# 17 skenario: nama -> daftar JENIS fault yang muncul (urutan tidak penting).
# "malfunc" pada notebook lama = hardware. Skenario "faulty" = satu fault acak.
SCENARIO_TYPES = {
    "faulty": None,  # diisi acak 1 jenis per pengulangan
    "drift": ["drift"],
    "spike": ["spike"],
    "bias": ["bias"],
    "hardware": ["hardware"],
    "bias+malfunc": ["bias", "hardware"],
    "spike+malfunc": ["spike", "hardware"],
    "spike+bias": ["spike", "bias"],
    "drift+malfunc": ["drift", "hardware"],
    "drift+bias": ["drift", "bias"],
    "drift+spike": ["drift", "spike"],
    "spike+bias+malfunc": ["spike", "bias", "hardware"],
    "drift+bias+malfunc": ["drift", "bias", "hardware"],
    "spike+drift+malfunc": ["spike", "drift", "hardware"],
    "drift+spike+bias": ["drift", "spike", "bias"],
    "spike+bias+malfunc+drift": ["spike", "bias", "hardware", "drift"],
}
scenario_names = ["normal"] + list(SCENARIO_TYPES.keys())
print("Skenario:", len(SCENARIO_TYPES), "+ normal =", len(scenario_names), "kelas")
print("Jenis fault:", FAULT_TYPES, "-> label Tahap 2 =", len(SENSORS) * len(FAULT_TYPES))
"""))

cells.append(code("""from numpy.lib.stride_tricks import sliding_window_view

def make_windows(X, win, stride):
    Xn = np.asarray(X, dtype=np.float32); N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    view = sliding_window_view(Xn, window_shape=win, axis=0)
    starts = np.arange(0, N - win + 1, stride, dtype=int)
    return view[starts], starts

def inject_per_type_per_sensor(X, fault_types, rng):
    \"\"\"Tiap jenis fault dapat subset sensor acak SENDIRI.

    Return:
      Y : (T,4) sinyal setelah fault
      M : (T,4,4) mask boolean per (sensor, jenis fault)
    \"\"\"
    Y = X.copy()
    M = np.zeros((X.shape[0], len(SENSORS), len(FAULT_TYPES)), dtype=bool)
    for ftype in fault_types:
        ti = FAULT_TYPES.index(ftype)
        k = int(rng.integers(1, len(SENSORS) + 1))          # 1..4 sensor kena
        subset = rng.choice(len(SENSORS), size=k, replace=False)
        f, kw = SIMULATOR[ftype]
        for s in subset:
            y, mm = f(Y[:, s], **kw, seed=int(rng.integers(1e9)))
            Y[:, s] = y
            M[:, s, ti] |= mm
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M

def window_labels_per_sensor_type(M, win, stride, thr=0.01):
    \"\"\"(T,4,4) mask -> (Nwin,4,4) label. Rasio fault dalam window > thr.

    CATATAN: sliding_window_view(arr, window_shape=win, axis=0) menaruh sumbu
    window di BELAKANG -> (Nwin, 4, 4, win). Rata-rata diambil di axis=-1
    (waktu), bukan axis=1.
    \"\"\"
    T = M.shape[0]
    if win > T:
        return np.zeros((0, len(SENSORS), len(FAULT_TYPES)), dtype=bool)
    Wm = sliding_window_view(M, window_shape=win, axis=0)[::stride]  # (Nwin,4,4,win)
    return Wm.mean(axis=-1) > thr

rng_master = np.random.default_rng(RANDOM_SEED)
datasets, labels, label16 = [], [], []

# normal: tidak ada fault sama sekali
W0, _ = make_windows(X_ds, WIN, STRIDE)
datasets.append(W0)
labels.append(np.zeros(len(W0), dtype=int))
label16.append(np.zeros((len(W0), len(SENSORS), len(FAULT_TYPES)), dtype=bool))

for k, (name, ftypes) in enumerate(SCENARIO_TYPES.items(), start=1):
    n_win = 0
    for _rep in range(REPEATS_PER_SCENARIO):
        types_now = ([FAULT_TYPES[int(rng_master.integers(len(FAULT_TYPES)))]]
                     if ftypes is None else ftypes)
        Y, M = inject_per_type_per_sensor(X_ds, types_now, rng_master)
        lab = window_labels_per_sensor_type(M, WIN, STRIDE, thr=FAULT_RATIO_THR)
        keep = lab.any(axis=(1, 2))          # window yang benar-benar memuat fault
        Wk, _ = make_windows(Y, WIN, STRIDE)
        Wk, lab = Wk[keep], lab[keep]
        datasets.append(Wk)
        labels.append(np.full(len(Wk), k, dtype=int))
        label16.append(lab)
        n_win += len(Wk)
    print(f"{name:28s} windows={n_win}")

W_all = np.concatenate(datasets, axis=0)
y_all = np.concatenate(labels, axis=0)
L_all = np.concatenate(label16, axis=0).astype(int)      # (N,4,4)

if W_all.ndim == 3 and W_all.shape[1] == len(SENSORS) and W_all.shape[2] == WIN:
    W_all = W_all.transpose(0, 2, 1)

print("\\nTotal windows:", W_all.shape)
prev = pd.DataFrame(L_all.mean(axis=0), index=SENSORS, columns=FAULT_TYPES).round(3)
print("\\nPrevalensi fault per (sensor x jenis) -- HARUS berbeda-beda,")
print("kalau semua sama berarti injeksi masih menyamakan sensor:")
print(prev.to_string())
"""))

cells.append(code("""def balanced_subsample_multi(Xw, y, L, max_per_class=200, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return Xw[keep], y[keep], L[keep]

W_s, y_s, L_s = balanced_subsample_multi(W_all, y_all, L_all, max_per_class=MAX_PER_CLASS, seed=RANDOM_SEED)
Y16 = L_s.reshape(len(L_s), -1)                       # (N,16) urutan: sensor-major
LABEL16_NAMES = [f"{s}:{t}" for s in SENSORS for t in FAULT_TYPES]

print("After balanced subsample:", W_s.shape)
print("Label Tahap 2:", Y16.shape, "->", LABEL16_NAMES)
prev16 = pd.Series(Y16.mean(axis=0), index=LABEL16_NAMES).round(3)
print("\\nPrevalensi 16 label:"); print(prev16.to_string())
"""))

cells.append(md("# Fitur — Entropy (EDM-Fuzzy & JSD-Fuzzy) + Time-Domain"))

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

min_len = m + 2; max_scale = max(1, WIN // min_len)
scales = np.array([s for s in scales if s <= max_scale] or [1], dtype=int)
S = len(scales)
print("Using scales:", scales.tolist())
"""))

cells.append(code("""from joblib import Parallel, delayed
from scipy import stats as _sstats

def sanitize_features(F):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()

def compute_features_entropy(W, scales, method, m=2, r_ratio=0.2, n_ref=256, jsd_bins=20, seed=0, n_jobs=-1):
    Nwin, win, ns = W.shape
    key = method.strip().lower()

    def entropy_1d(x, seed_local):
        if key == "edm-fuzzy":
            return edm_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=seed_local)
        if key == "jsd-fuzzy":
            return jsd_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=seed_local, bins=jsd_bins)
        raise ValueError(f"Unknown method: {method}")

    def one_window(i):
        return np.concatenate([entropy_1d(W[i, :, s], seed_local=seed + 1000 * i + 19 * s) for s in range(ns)])

    return np.vstack(Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one_window)(i) for i in range(Nwin)))

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
    return np.nan_to_num(np.column_stack(feats), nan=0.0, posinf=0.0, neginf=0.0)

T_feats = compute_time_features(W_s)
FEAT_by_method = {}
for name in METHOD_LIST:
    log_stage(f"entropy features: {name} ({W_s.shape[0]} windows)")
    Fm, _ = run_with_metrics(f"Entropy {name}", lambda n=name: compute_features_entropy(
        W_s, scales=scales, method=n, m=m, r_ratio=r_ratio, n_ref=n_ref, jsd_bins=jsd_bins, seed=7, n_jobs=N_JOBS))
    FEAT_by_method[name] = np.hstack([sanitize_features(Fm), T_feats])
    print(name, "hybrid feature shape:", FEAT_by_method[name].shape)
"""))

cells.append(md("""# Tahap 1 — Klasifikasi 17 Kelas

Sama seperti notebook paper: kombinasi jenis fault apa yang sedang terjadi di
sistem. Split test-nya dipakai ulang di Tahap 2 supaya benar-benar berantai.
"""))

cells.append(code("""from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                              roc_auc_score, hamming_loss, classification_report)

TEST_FRAC = 0.25
idx_all = np.arange(len(y_s))
idx_tr, idx_te = train_test_split(idx_all, test_size=TEST_FRAC, random_state=RANDOM_SEED, stratify=y_s)

def build_hidden_candidates(I, O):
    base = sorted(set([max(8, I // 4), max(16, I // 2), max(32, int(np.floor((2/3)*I + O))), I, min(2*I, 512)]))
    cand = [(h,) for h in base]
    for h1 in base:
        cand.append((h1, max(8, h1 // 2)))
    return list(dict.fromkeys(cand))[:12]

def train_stage1(F_local):
    Xtr, Xte, ytr, yte = F_local[idx_tr], F_local[idx_te], y_s[idx_tr], y_s[idx_te]
    I = Xtr.shape[1]; O = len(np.unique(y_s))
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("mlp", MLPClassifier(max_iter=400, random_state=RANDOM_SEED, early_stopping=True, n_iter_no_change=10))])
    grid = {"mlp__hidden_layer_sizes": build_hidden_candidates(I, O),
            "mlp__alpha": [1e-4, 1e-3], "mlp__activation": ["relu", "tanh"]}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_SEED),
                      n_jobs=ANN_GRID_N_JOBS, scoring="accuracy")
    gs.fit(Xtr, ytr)
    pred = gs.best_estimator_.predict(Xte)
    return {"yte": yte, "pred": pred,
            "metrics": {"test_acc": float(accuracy_score(yte, pred)),
                         "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0))}}

stage1 = {}
for name in METHOD_LIST:
    log_stage(f"Tahap 1 (17 kelas): {name}")
    res, _ = run_with_metrics(f"Stage1 {name}", lambda n=name: train_stage1(FEAT_by_method[n]))
    stage1[name] = res
    print(f"  {name} | acc={res['metrics']['test_acc']:.4f} macro-F1={res['metrics']['macro_f1']:.4f}")

stage1_tbl = pd.DataFrame({n: r["metrics"] for n, r in stage1.items()}).T
export_df(stage1_tbl.reset_index().rename(columns={"index": "Method"}), "cara_b_stage1_17class")
stage1_tbl
"""))

cells.append(md("""# Tahap 2 — 16 Label (Sensor × Jenis Fault)

Satu `MultiOutputClassifier(MLPClassifier)` dengan 16 keluaran biner. Dilatih
hanya dari window ber-fault (kelas ≠ normal), lalu dievaluasi pada window uji
yang **diprediksi fault oleh Tahap 1** (end-to-end, jadi error Tahap 1 ikut
terhitung).

Kolom yang di training set-nya cuma punya satu kelas (misal sensor tertentu
tidak pernah kena jenis fault tertentu) di-*skip* dan dilaporkan, bukan
dipaksakan — supaya tidak ada angka palsu.
"""))

cells.append(code("""def train_stage2(F_local, Y16_local, y_true, y_pred_stage1):
    Xtr, Xte = F_local[idx_tr], F_local[idx_te]
    Ytr, Yte = Y16_local[idx_tr], Y16_local[idx_te]
    tr_mask = y_true[idx_tr] != 0
    Xtr_f, Ytr_f = Xtr[tr_mask], Ytr[tr_mask]

    # buang kolom degenerate (cuma 1 kelas di train) -> dilaporkan terpisah
    usable = [j for j in range(Ytr_f.shape[1]) if len(np.unique(Ytr_f[:, j])) > 1]
    skipped = [LABEL16_NAMES[j] for j in range(Ytr_f.shape[1]) if j not in usable]

    base = MLPClassifier(max_iter=300, random_state=RANDOM_SEED, early_stopping=True, n_iter_no_change=12)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                      ("mo", MultiOutputClassifier(base))])
    I = Xtr_f.shape[1]
    grid = {"mo__estimator__hidden_layer_sizes": [(max(16, I // 2),), (I,)],
            "mo__estimator__alpha": [1e-4, 1e-3]}
    gs = GridSearchCV(pipe, grid, cv=3, scoring="f1_macro", n_jobs=N_JOBS)
    gs.fit(Xtr_f, Ytr_f[:, usable])
    best = gs.best_estimator_

    ev = y_pred_stage1 != 0                       # end-to-end: pakai prediksi Tahap 1
    Xe, Ye = Xte[ev], Yte[ev][:, usable]
    pred = best.predict(Xe)
    proba = np.column_stack([e.predict_proba(Xe)[:, 1] for e in best.named_steps["mo"].estimators_])

    rows = []
    for jj, j in enumerate(usable):
        p, r, f, _ = precision_recall_fscore_support(Ye[:, jj], pred[:, jj], average="binary", zero_division=0)
        try: auc = roc_auc_score(Ye[:, jj], proba[:, jj])
        except Exception: auc = np.nan
        sensor, ftype = LABEL16_NAMES[j].split(":")
        rows.append({"Sensor": sensor, "Fault": ftype, "Prevalensi": round(Ye[:, jj].mean(), 3),
                      "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f, 3),
                      "ROC_AUC": round(auc, 3) if not np.isnan(auc) else np.nan})
    return dict(rows=rows, skipped=skipped, n_eval=int(ev.sum()), usable=usable,
                pred=pred, Ye=Ye,
                subset_acc=accuracy_score(Ye, pred), hamming=hamming_loss(Ye, pred),
                macro_f1=f1_score(Ye, pred, average="macro", zero_division=0))

stage2 = {}
all_rows = []
for name in METHOD_LIST:
    log_stage(f"Tahap 2 (16 label): {name}")
    res, _ = run_with_metrics(f"Stage2 {name}", lambda n=name: train_stage2(
        FEAT_by_method[n], Y16, y_s, stage1[n]["pred"]))
    stage2[name] = res
    for r in res["rows"]:
        all_rows.append({"Method": name, **r})
    print(f"  {name} | n_eval={res['n_eval']} macro-F1={res['macro_f1']:.3f} "
          f"subset-acc={res['subset_acc']:.3f} hamming={res['hamming']:.3f}")
    if res["skipped"]:
        print(f"  [skip] kolom tanpa variasi di train: {res['skipped']}")

detail = pd.DataFrame(all_rows)
print("\\n=== Tahap 2: F1 per (sensor x jenis fault) ===")
print(detail.to_string(index=False))
export_df(detail, "cara_b_stage2_sensor_x_faulttype")
"""))

cells.append(code("""# === Tabel pivot: "bias-nya di sensor mana" — baca per baris ===
for name in METHOD_LIST:
    sub = detail[detail.Method == name]
    if sub.empty: continue
    piv = sub.pivot(index="Fault", columns="Sensor", values="F1")
    print(f"\\n=== {name} — F1 deteksi (baris = jenis fault, kolom = sensor) ===")
    print(piv.to_string())
    export_df(piv.reset_index(), f"cara_b_pivot_{name.replace('-','_')}", index=False)
"""))

cells.append(code("""# === Dua sudut pandang turunan: per sensor, dan per jenis fault ===
summary_rows = []
for name in METHOD_LIST:
    sub = detail[detail.Method == name]
    if sub.empty: continue
    for sensor, g in sub.groupby("Sensor"):
        summary_rows.append({"Method": name, "Sudut pandang": "per sensor", "Item": sensor,
                              "F1 rata-rata": round(g.F1.mean(), 3), "AUC rata-rata": round(g.ROC_AUC.mean(), 3)})
    for ftype, g in sub.groupby("Fault"):
        summary_rows.append({"Method": name, "Sudut pandang": "per jenis fault", "Item": ftype,
                              "F1 rata-rata": round(g.F1.mean(), 3), "AUC rata-rata": round(g.ROC_AUC.mean(), 3)})

summary = pd.DataFrame(summary_rows)
print("=== Ringkasan Tahap 2 ===")
print(summary.to_string(index=False))
export_df(summary, "cara_b_stage2_summary")
"""))

cells.append(code("""# === Plot: heatmap F1 (jenis fault x sensor) per metode ===
fig, axes = plt.subplots(1, len(METHOD_LIST), figsize=(6 * len(METHOD_LIST), 4.5))
if len(METHOD_LIST) == 1: axes = [axes]
for ax, name in zip(axes, METHOD_LIST):
    sub = detail[detail.Method == name]
    if sub.empty:
        ax.set_visible(False); continue
    piv = sub.pivot(index="Fault", columns="Sensor", values="F1")
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=10)
    ax.set_title(f"{name} — F1 deteksi per sensor & jenis fault")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig("exports/cara_b_heatmap_f1.png", dpi=120); plt.show()
print("[Saved] exports/cara_b_heatmap_f1.png")
"""))

cells.append(md("""## Cara membaca hasil ini

- **Heatmap / tabel pivot** menjawab langsung pertanyaan Bu Luh: baris = jenis
  fault, kolom = sensor. Angka F1 tinggi di baris `bias` kolom `S2` berarti
  sistem memang bisa bilang *"bias-nya ada di sensor 2"*.
- **Kolom `Prevalensi`** wajib dilihat bareng F1. Kalau prevalensi mendekati 1
  dan ROC-AUC ≈ 0,5, artinya model cuma menebak "selalu fault" — itu bukan
  deteksi. Ini persis kesalahan yang membuat hasil notebook per-sensor lama tidak sah.
- **`subset-acc`** = seluruh 16 label benar sekaligus dalam satu window (ukuran
  paling ketat). **`hamming`** = rata-rata label yang salah.
- Angka Tahap 2 di sini **end-to-end**: window uji disaring pakai prediksi
  Tahap 1, jadi kesalahan Tahap 1 sudah ikut terhitung — bukan angka ideal.

## Batasnya

- `bias` cenderung paling sulit karena intensitas yang diinjeksikan (0,08) lebih
  kecil dari variasi alami antar-window; ini batas fisik data, bukan
  kekurangan fitur. Menaikkan intensitas injeksi akan menaikkan angkanya.
- Fault ditempatkan acak per pengulangan skenario, jadi prevalensi tiap
  (sensor × jenis) tidak sama rata. Kolom `Prevalensi` di tabel detail
  menunjukkan hal ini apa adanya.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9"}}, "nbformat": 4, "nbformat_minor": 5}

OUT = "/Users/kelvin/apps/public-files/05_Fault_Jenis_Apa_di_Sensor_Mana.ipynb"
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "| cells:", len(cells))
