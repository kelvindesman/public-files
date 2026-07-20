#!/usr/bin/env python3
"""Build 07_CV_5_Skenario_Performa_Komputasi_dan_Sensor.ipynb.

Menjawab permintaan terbaru:
  1) perbandingan pakai CROSS-VALIDATION (bukan single split) untuk 5 skenario
     ladder S1..S5,
  2) fault detection dilaporkan dua sisi: performa (akurasi/precision/recall/F1)
     DAN biaya komputasi (CPU time, memori puncak, wall time, latensi inferensi),
  3) arsitektur broker dipertahankan (4 sensor dikumpulkan jadi satu tabel
     ter-align, TIDAK dirata-rata),
  4) output tambahan: sensor mana yang rusak (asal fault-nya), dirantai dari
     hasil klasifikasi tiap skenario.

Data: data_sensor.csv (dataset terbaru, 281.721 baris, 2025-09-14 .. 2025-12-21).
"""
import json

REPO_RAW = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"
OUT = "/Users/kelvin/apps/public-files/07_CV_5_Skenario_Performa_Komputasi_dan_Sensor.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""<!-- HEADER-KLAIM -->
# 07 — Perbandingan Cross-Validation 5 Skenario: Performa + Biaya Komputasi + Sensor Mana

| | |
|---|---|
| **Pertanyaan** | Dengan **data terbaru** (`data_sensor.csv`), bagaimana perbandingan EDM-Fuzzy vs JSD-Fuzzy untuk **5 skenario (S1–S5)** kalau dievaluasi pakai **cross-validation**, dan berapa **ongkos komputasinya**? Lalu, sensor mana yang rusak? |
| **Yang diukur** | (a) **Performa**: akurasi, precision, recall, F1 (macro) — rata-rata ± simpangan baku antar-fold. (b) **Komputasi**: CPU time, wall time, memori puncak, latensi inferensi per window. (c) **Identifikasi sensor**: F1 per sensor S1–S4. |
| **Arsitektur** | **Broker** mengumpulkan 4 sensor jadi satu tabel ter-align waktu (identitas sensor dipertahankan, **tidak** dirata-rata — lihat notebook `02`), lalu windowing → fitur entropy multiskala + time-domain → ANN. |
| **Kenapa CV, bukan single split** | Single split hanya memberi satu angka tanpa ketidakpastian. Stratified K-Fold memberi rata-rata **dan** sebaran, jadi selisih EDM vs JSD bisa dinilai berarti atau tidak. |
| **Data** | `data_sensor.csv` — 281.721 baris, 2025-09-14 s.d. 2025-12-21, kolom `kelembaban1..kelembaban4`, tanpa nilai kosong. |

---
"""))

cells.append(md("""## Peta isi notebook

```
Broker  : 4 sensor -> satu tabel ter-align waktu (data_sensor.csv)
   |
Windowing (WIN/STRIDE) + injeksi fault SENSOR-SELECTIVE
   |                                     \\
   |                                      +-- ground-truth per sensor (untuk Tahap 2)
Fitur  : EDM-Fuzzy | JSD-Fuzzy (multiskala) + 12 fitur time-domain per sensor
   |
Tahap 1: 5 skenario S1..S5, Stratified 5-Fold CV, ANN
   |          -> performa  (akurasi, precision, recall, F1) mean +/- std
   |          -> komputasi (cpu_s, wall_s, peak_mem_mb, latensi inferensi)
   |
Tahap 2: sensor mana yang rusak, dirantai dari prediksi Tahap 1 (CV yang sama)
```

**5 skenario (ladder menurut banyaknya fault yang bercampur)** — sama persis
dengan notebook `03`, supaya angkanya bisa dibandingkan lintas notebook:

| Skenario | Isi | Jumlah kelas |
|---|---|---|
| S1 | normal vs faulty | 2 |
| S2 | normal + 4 fault tunggal (drift, spike, bias, hardware) | 5 |
| S3 | normal + 6 kombinasi dua fault | 7 |
| S4 | normal + 4 kombinasi tiga fault | 5 |
| S5 | normal + kombinasi empat fault | 2 |
"""))

cells.append(code("""# === Runtime guard — jalankan sel ini PALING AWAL ===
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
        print(f"[budget] SKIP {label}: sisa {left/60:.1f} menit, butuh ~{need_s/60:.1f} menit", flush=True)
        return False
    return True

def log_stage(label):
    print(f"[t+{elapsed_s()/60:6.1f} min] {label}", flush=True)

log_stage(f"runtime guard aktif | budget={KAGGLE_TIME_BUDGET_H} jam | BLAS threads=1")
"""))

cells.append(code("""# === Konfigurasi global ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, logging
import time as _time, tracemalloc, platform
from pathlib import Path
from IPython.display import FileLink, display

warnings.filterwarnings("ignore")

RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE", "fast")   # fast | paper
METHOD_LIST = ["EDM-Fuzzy", "JSD-Fuzzy"]
RANDOM_SEED = 42
N_JOBS = -1
N_SPLITS = 5                      # Stratified K-Fold
EXPORT_DIR = "exports"
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

def export_df(df, name, index=False):
    p = Path(EXPORT_DIR) / f"{name}.csv"
    df.to_csv(p, index=index)
    display(FileLink(str(p)))
    return str(p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_with_metrics(label, fn):
    \"\"\"Ukur wall time, CPU time, dan memori puncak untuk satu blok kerja.\"\"\"
    tracemalloc.start()
    t0 = _time.perf_counter(); c0 = _time.process_time()
    result = fn()
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    metrics = {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024)}
    logging.info("%s | wall=%.2fs cpu=%.2fs peak_mem=%.1f MB",
                 label, metrics["wall_s"], metrics["cpu_s"], metrics["peak_mem_mb"])
    return result, metrics

print("Mesin :", platform.processor() or platform.machine(), "| CPU count:", os.cpu_count())
print("Profil:", RUNTIME_PROFILE, "| metode:", METHOD_LIST, "| K-Fold:", N_SPLITS)
"""))

cells.append(code("""# === Toggle sampling & windowing ===
# Data baru ~281k baris (vs ~90k data lama) -> DS dinaikkan ke 8 supaya jumlah
# window (dan ongkos CV) tetap sebanding dengan notebook sebelumnya.
DS = int(os.environ.get("DS", 8))
WIN = 256
STRIDE = 128
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 200))
# spike menyala tiap tau=int(1/p)=66 sampel -> rasio per window mentok ~1/66=0.0152.
# ambang harus di bawah itu, kalau tidak skenario "spike" menghasilkan 0 window.
FAULT_RATIO_THR = 0.01

SENSORS = ["S1", "S2", "S3", "S4"]
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]   # injeksi sensor-selective -> ground-truth per sensor

S = 10; m = 2; r_ratio = 0.2; n_ref = 128; jsd_bins = 40
scales = np.arange(1, S + 1)

print("config:", dict(DS=DS, WIN=WIN, STRIDE=STRIDE, MAX_PER_CLASS=MAX_PER_CLASS,
                      N_SPLITS=N_SPLITS, S=S))
"""))

cells.append(md("""# 1. Broker — kumpulkan 4 sensor jadi satu tabel

Output broker = satu tabel ter-align waktu berisi keempat sensor. Identitas
sensor **dipertahankan** (4 kolom terpisah), bukan dilebur jadi satu sinyal —
alasannya sudah dibuktikan di notebook `02`.

Sumber data: `data_sensor.csv` (dataset terbaru). Kalau file ada di direktori
kerja atau di `/kaggle/input/...` file itu dipakai; kalau tidak, diunduh dari
GitHub (**Kaggle: Settings → Internet → On**).
"""))

cells.append(code("""import requests, glob
from io import StringIO

DATA_URL = "__REPO_RAW__"
DATA_NAME = "data_sensor.csv"

def load_sensor_data():
    # 1) file lokal / dataset Kaggle
    cands = [DATA_NAME, f"../input/{DATA_NAME}"] + glob.glob(f"/kaggle/input/**/{DATA_NAME}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            print("Sumber data: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    # 2) unduh dari GitHub
    print("Sumber data: unduh ->", DATA_URL)
    r = requests.get(DATA_URL, timeout=120); r.raise_for_status()
    return pd.read_csv(StringIO(r.text), index_col=0)

df = load_sensor_data()
cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Error-nya jelas: kolom {missing} tidak ada di {DATA_NAME}. Kolom tersedia: {list(df.columns)}")

X_df = pd.DataFrame(df[cols].to_numpy(dtype=float), columns=cols).ffill().bfill()
X_df = X_df.fillna(X_df.median(numeric_only=True))
if X_df.isna().any().any():
    raise ValueError("Error-nya jelas: X masih ada NaN setelah imputasi.")

X = X_df.to_numpy()
X_ds = X[::DS]
print("Rentang waktu :", df.index[0], "->", df.index[-1])
print("Tabel broker  :", X.shape, "-> setelah downsample DS=%d:" % DS, X_ds.shape)
display(X_df.describe().round(2))
""".replace("__REPO_RAW__", REPO_RAW)))

cells.append(code("""# === Sanity plot: 4 kanal sensor apa adanya (belum ada fault injeksi) ===
fig, ax = plt.subplots(figsize=(13, 4))
seg = X_ds[:3000]
for j, c in enumerate(cols):
    ax.plot(seg[:, j], lw=0.8, label=c)
ax.set_title("Output broker — 4 kanal kelembaban (potongan 3000 sampel, data terbaru)")
ax.set_xlabel("sampel (setelah downsample)"); ax.set_ylabel("kelembaban")
ax.legend(ncol=4, fontsize=9)
plt.tight_layout(); plt.savefig("exports/07_broker_sinyal.png", dpi=120); plt.show()
"""))

cells.append(md("""# 2. Injeksi fault — sensor-selective

Simulator fault dan definisi 16 skenario kombinasi identik notebook `01`/`03`,
jadi angkanya sebanding. Bedanya: fault hanya disuntik ke **subset sensor acak**
(1..4 sensor), sensor di luar subset tetap bersih. Ini yang membuat ground-truth
"sensor mana yang rusak" jadi genuin — kalau keempat sensor selalu kena bersamaan,
label per-sensor jadi kembar dan hasil Tahap 2 tidak sah (lihat catatan di README).
"""))

cells.append(code("""# --- Simulator fault (identik notebook 01/03) ---
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
print("Kondisi:", len(SCENARIOS), "fault + normal =", len(SCENARIOS) + 1)
"""))

cells.append(code("""from numpy.lib.stride_tricks import sliding_window_view

def make_windows(Xa, win, stride):
    Xn = np.asarray(Xa, dtype=np.float32); N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    view = sliding_window_view(Xn, window_shape=win, axis=0)
    starts = np.arange(0, N - win + 1, stride, dtype=int)
    return view[starts], starts

def inject_faults_multisensor(Xa, scenario_faults, sensor_subset, seed=0):
    rng = np.random.default_rng(seed)
    Y = Xa.copy(); M = np.zeros_like(Y, dtype=bool)
    for s in sensor_subset:
        y, m_ = simulate_multiple_faults(Y[:, s], scenario_faults, seed=int(rng.integers(1e9)))
        Y[:, s] = y; M[:, s] = m_
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M

def window_fault_label_per_sensor(mask, win, stride, thr=0.02):
    # sliding_window_view(mask, win, axis=0) -> (Nwin, 4, win): sumbu kanal tetap
    # di tengah, sumbu waktu ditempel di BELAKANG. Rata-rata di axis=2.
    T = len(mask)
    if win > T:
        return np.zeros((0, mask.shape[1]), dtype=bool)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]
    return (Wm.mean(axis=2) > thr)

rng_master = np.random.default_rng(RANDOM_SEED)
datasets, labels, sens_labels = [], [], []
condition_names = ["normal"] + list(SCENARIOS.keys())

W0, _ = make_windows(X_ds, WIN, STRIDE)
datasets.append(W0); labels.append(np.zeros(len(W0), dtype=int))
sens_labels.append(np.zeros((len(W0), 4), dtype=bool))
print(f"{'normal':28s} windows={len(W0)}")

for k, (name, faults) in enumerate(SCENARIOS.items(), start=1):
    n_scenario = 0
    for subset_size in SENSOR_SUBSET_SIZES:
        subset = rng_master.choice(4, size=subset_size, replace=False)
        Y, M = inject_faults_multisensor(X_ds, faults, subset, seed=int(rng_master.integers(1e9)))
        sens_win = window_fault_label_per_sensor(M, WIN, STRIDE, thr=FAULT_RATIO_THR)
        keep_win = sens_win.any(axis=1)
        Wk, _ = make_windows(Y, WIN, STRIDE)
        Wk = Wk[keep_win]; sens_win = sens_win[keep_win]
        datasets.append(Wk); labels.append(np.full(len(Wk), k, dtype=int)); sens_labels.append(sens_win)
        n_scenario += len(Wk)
    print(f"{name:28s} windows={n_scenario}")

W_all = np.concatenate(datasets, axis=0)
y_all = np.concatenate(labels, axis=0)
Ysens_all = np.concatenate(sens_labels, axis=0).astype(int)
if W_all.ndim == 3 and W_all.shape[1] == 4 and W_all.shape[2] == WIN:
    W_all = W_all.transpose(0, 2, 1)

print("\\nTotal window:", W_all.shape)
print("Prevalensi fault per sensor:", Ysens_all.mean(axis=0).round(3),
      "<- kalau keempatnya identik ~1.0, label per-sensor kembar dan Tahap 2 tidak sah")
"""))

cells.append(code("""def balanced_subsample(Xw, y, Ysens, max_per_class=200, seed=0):
    rng = np.random.default_rng(seed); keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return Xw[keep], y[keep], Ysens[keep]

W_s, y_s, Ysens_s = balanced_subsample(W_all, y_all, Ysens_all, MAX_PER_CLASS, RANDOM_SEED)
if W_s.ndim != 3:
    raise ValueError(f"Error: W_s harus 3D (N, WIN, 4). Shape={W_s.shape}")
print("Setelah balanced subsample:", W_s.shape)
print("Jumlah window per kondisi:", dict(zip(*np.unique(y_s, return_counts=True))))
print("Prevalensi per sensor    :", Ysens_s.mean(axis=0).round(3))
"""))

cells.append(md("""# 3. Ekstraksi fitur — EDM-Fuzzy vs JSD-Fuzzy (+ time-domain)

Fungsi entropy identik notebook `01`. JSD-Fuzzy memakai varian *rich*
(`[jsd, fe, mean_mu, std_mu]` per skala). Keduanya digabung 12 fitur time-domain
per sensor supaya offset/bias — yang entropy buta terhadapnya — tetap tertangkap.

**Ongkos ekstraksi fitur ikut diukur** dan dilaporkan terpisah dari ongkos
latih model, karena di sistem nyata ekstraksi fitur ini yang jalan tiap window.
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

min_len = m + 2; max_scale = max(1, WIN // min_len)
scales = np.array([s for s in scales if s <= max_scale] or [1], dtype=int)
S = len(scales)
print("Skala yang dipakai:", scales.tolist())
"""))

cells.append(code("""from joblib import Parallel, delayed

def sanitize_features(F):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()

def compute_features_entropy(W, scales, method, m=2, r_ratio=0.2, n_ref=256,
                             jsd_bins=20, seed=0, n_jobs=-1):
    Nwin, win, ns = W.shape
    key = method.strip().lower()

    def entropy_1d(x, seed_local):
        if key == "edm-fuzzy":
            return edm_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=seed_local)
        if key == "jsd-fuzzy":
            return jsd_fuzzy_entropy_1d(x, scales=scales, m=m, r_ratio=r_ratio, n_ref=n_ref,
                                        seed=seed_local, bins=jsd_bins)
        raise ValueError(f"Metode tidak dikenal: {method}")

    def one_window(i):
        return np.concatenate([entropy_1d(W[i, :, s], seed_local=seed + 1000 * i + 19 * s)
                               for s in range(ns)], axis=0)

    F = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one_window)(i) for i in range(Nwin))
    return np.vstack(F)

from scipy import stats as _sstats

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

# --- ekstraksi + pencatatan ongkosnya ---
feat_cost_rows = []
FEAT_by_method = {}

T_feats, tm = run_with_metrics("Fitur time-domain", lambda: compute_time_features(W_s))
feat_cost_rows.append({"Tahap": "fitur time-domain", "Metode": "(bersama)",
                        "wall_s": round(tm["wall_s"], 2), "cpu_s": round(tm["cpu_s"], 2),
                        "peak_mem_mb": round(tm["peak_mem_mb"], 1),
                        "ms_per_window": round(1000 * tm["wall_s"] / len(W_s), 3)})

for name in METHOD_LIST:
    log_stage(f"ekstraksi fitur entropy: {name} ({W_s.shape[0]} window)")
    Fm, mtr = run_with_metrics(f"Fitur entropy {name}", lambda n=name: compute_features_entropy(
        W_s, scales=scales, method=n, m=m, r_ratio=r_ratio, n_ref=n_ref,
        jsd_bins=jsd_bins, seed=7, n_jobs=N_JOBS))
    Fm = sanitize_features(Fm)
    FEAT_by_method[name] = np.hstack([Fm, T_feats])
    feat_cost_rows.append({"Tahap": "fitur entropy", "Metode": name,
                            "wall_s": round(mtr["wall_s"], 2), "cpu_s": round(mtr["cpu_s"], 2),
                            "peak_mem_mb": round(mtr["peak_mem_mb"], 1),
                            "ms_per_window": round(1000 * mtr["wall_s"] / len(W_s), 3)})
    print(f"{name}: entropy {Fm.shape} + time-domain {T_feats.shape} -> hibrida {FEAT_by_method[name].shape}")

feat_cost = pd.DataFrame(feat_cost_rows)
print("\\n=== Ongkos komputasi ekstraksi fitur ===")
print(feat_cost.to_string(index=False))
export_df(feat_cost, "07_ongkos_ekstraksi_fitur")
"""))

cells.append(md("""# 4. Definisi 5 skenario (ladder)

Sama dengan notebook `03`: skenario disusun menurut **berapa banyak jenis fault
yang bercampur**, dari yang paling mudah (S1: fault vs tidak) sampai paling
spesifik (S5: keempat jenis fault sekaligus).
"""))

cells.append(code("""ALL_FAULT = [c for c in condition_names if c != "normal"]

LADDER = {
    "S1_Normal_vs_Faulty": [("normal", ["normal"]), ("faulty", ALL_FAULT)],
    "S2_Fault_Tunggal":    [("normal", ["normal"]), ("drift", ["drift"]), ("spike", ["spike"]),
                             ("bias", ["bias"]), ("hardware", ["hardware"])],
    "S3_Dua_Fault":        [("normal", ["normal"]), ("bias+HW", ["bias+malfunc"]),
                             ("drift+bias", ["drift+bias"]), ("drift+HW", ["drift+malfunc"]),
                             ("spike+bias", ["spike+bias"]), ("drift+spike", ["drift+spike"]),
                             ("spike+HW", ["spike+malfunc"])],
    "S4_Tiga_Fault":       [("normal", ["normal"]), ("drift+bias+HW", ["drift+bias+malfunc"]),
                             ("drift+spike+bias", ["drift+spike+bias"]),
                             ("spike+bias+HW", ["spike+bias+malfunc"]),
                             ("spike+drift+HW", ["spike+drift+malfunc"])],
    "S5_Empat_Fault":      [("normal", ["normal"]), ("drift+spike+bias+HW", ["spike+bias+malfunc+drift"])],
}

cond_to_idx = {c: i for i, c in enumerate(condition_names)}

def build_scenario_labels(classes):
    \"\"\"-> (indeks window terpilih, label kelas skenario, nama kelas).\"\"\"
    idx_list, y_list = [], []
    for ci, (cname, conds) in enumerate(classes):
        want = [cond_to_idx[c] for c in conds if c in cond_to_idx]
        sel = np.where(np.isin(y_s, want))[0]
        idx_list.append(sel); y_list.append(np.full(len(sel), ci, dtype=int))
    keep = np.concatenate(idx_list); yy = np.concatenate(y_list)
    return keep, yy, [c[0] for c in classes]

for sc, cl in LADDER.items():
    keep, yy, names = build_scenario_labels(cl)
    print(f"{sc:22s} {len(names)} kelas | n_window={len(keep):5d} | {dict(zip(*np.unique(yy, return_counts=True)))}")
"""))

cells.append(md("""## Catatan penting soal keseimbangan kelas S1

Window `normal` hanya dibangkitkan sekali, sedangkan tiap kondisi fault diulang
untuk beberapa subset sensor. Kalau dibiarkan, kelas `faulty` di S1 jadi ~16x
lebih banyak dan akurasi 0,9x hanya efek menebak kelas mayoritas. Karena itu
tiap skenario **diseimbangkan ke jumlah kelas terkecil** sebelum CV, dan yang
dilaporkan sebagai angka utama adalah **F1 macro**, bukan akurasi mentah.
"""))

cells.append(code("""def rebalance(keep, yy, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n_min = min(np.bincount(yy)[np.unique(yy)])
    sel = []
    for c in np.unique(yy):
        idx = np.where(yy == c)[0]
        sel.append(rng.choice(idx, size=n_min, replace=False) if len(idx) > n_min else idx)
    sel = np.concatenate(sel); rng.shuffle(sel)
    return keep[sel], yy[sel]

SCEN_DATA = {}
for sc, cl in LADDER.items():
    keep, yy, names = build_scenario_labels(cl)
    keep, yy = rebalance(keep, yy)
    SCEN_DATA[sc] = dict(keep=keep, y=yy, class_names=names)
    print(f"{sc:22s} setelah diseimbangkan: n={len(keep):5d} | {dict(zip(*np.unique(yy, return_counts=True)))}")
"""))

cells.append(md("""# 5. Tahap 1 — Fault detection dengan Stratified 5-Fold CV

Untuk tiap **skenario x metode**:

- **Performa** — akurasi, precision (macro), recall (macro), F1 (macro), dilaporkan
  sebagai rata-rata ± simpangan baku antar-fold, dari `cross_validate`.
- **Komputasi** — `fit_time` dan `score_time` per fold dari sklearn, ditambah
  CPU time total (`time.process_time`), memori puncak (`tracemalloc`), dan
  **latensi inferensi per window** (dipakai untuk menaksir apakah muat jalan
  online di broker).

Arsitektur ANN **dipatok tetap** (bukan GridSearch) supaya perbandingan ongkos
komputasi antar-metode adil — kalau tiap metode dapat arsitektur berbeda hasil
tuning, waktu latihnya tidak bisa dibandingkan.
"""))

cells.append(code("""from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                              roc_auc_score, hamming_loss, confusion_matrix,
                              ConfusionMatrixDisplay, classification_report)

ANN_HIDDEN = (128, 64)
ANN_MAX_ITER = 400

def make_ann_pipeline():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=ANN_HIDDEN, activation="relu", alpha=1e-3,
                               max_iter=ANN_MAX_ITER, random_state=RANDOM_SEED,
                               early_stopping=True, n_iter_no_change=12)),
    ])

SCORING = {"accuracy": "accuracy", "precision": "precision_macro",
           "recall": "recall_macro", "f1": "f1_macro"}

def cv_stage1(F, y):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    tracemalloc.start()
    c0 = _time.process_time(); t0 = _time.perf_counter()
    cvres = cross_validate(make_ann_pipeline(), F, y, cv=skf, scoring=SCORING,
                           n_jobs=N_JOBS, return_train_score=False)
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    # prediksi out-of-fold: dipakai Tahap 2 dan confusion matrix
    pred_oof = cross_val_predict(make_ann_pipeline(), F, y, cv=skf, n_jobs=N_JOBS)
    n_test_per_fold = len(y) / N_SPLITS
    return cvres, pred_oof, {
        "wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024),
        "fit_s_per_fold": float(np.mean(cvres["fit_time"])),
        "score_s_per_fold": float(np.mean(cvres["score_time"])),
        "infer_ms_per_window": float(1000 * np.mean(cvres["score_time"]) / n_test_per_fold),
    }

print("ANN:", ANN_HIDDEN, "| CV:", N_SPLITS, "fold stratified | scoring:", list(SCORING))
"""))

cells.append(code("""# === Jalankan Tahap 1: 5 skenario x 2 metode ===
perf_rows, comp_rows = [], []
STAGE1 = {}

for sc, dat in SCEN_DATA.items():
    keep, yy = dat["keep"], dat["y"]
    for meth in METHOD_LIST:
        if not budget_ok(600, f"{sc}/{meth}"):
            continue
        log_stage(f"Tahap 1 CV | {sc} | {meth}")
        F = FEAT_by_method[meth][keep]
        cvres, pred_oof, cost = cv_stage1(F, yy)
        STAGE1[(sc, meth)] = dict(pred_oof=pred_oof, keep=keep, y=yy,
                                  class_names=dat["class_names"])

        perf_rows.append({
            "Skenario": sc, "Metode": meth, "n_kelas": len(dat["class_names"]),
            "n_window": len(yy), "n_fitur": F.shape[1],
            "Akurasi": round(cvres["test_accuracy"].mean(), 4),
            "Akurasi_std": round(cvres["test_accuracy"].std(), 4),
            "Precision": round(cvres["test_precision"].mean(), 4),
            "Precision_std": round(cvres["test_precision"].std(), 4),
            "Recall": round(cvres["test_recall"].mean(), 4),
            "Recall_std": round(cvres["test_recall"].std(), 4),
            "F1": round(cvres["test_f1"].mean(), 4),
            "F1_std": round(cvres["test_f1"].std(), 4),
        })
        comp_rows.append({
            "Skenario": sc, "Metode": meth,
            "cpu_s_total": round(cost["cpu_s"], 1),
            "wall_s_total": round(cost["wall_s"], 1),
            "peak_mem_mb": round(cost["peak_mem_mb"], 1),
            "fit_s_per_fold": round(cost["fit_s_per_fold"], 2),
            "score_s_per_fold": round(cost["score_s_per_fold"], 3),
            "infer_ms_per_window": round(cost["infer_ms_per_window"], 3),
        })
        print(f"  {sc:22s} {meth:10s} acc={perf_rows[-1]['Akurasi']:.3f} "
              f"F1={perf_rows[-1]['F1']:.3f}+/-{perf_rows[-1]['F1_std']:.3f} "
              f"| cpu={comp_rows[-1]['cpu_s_total']}s mem={comp_rows[-1]['peak_mem_mb']}MB")

perf_tbl = pd.DataFrame(perf_rows)
comp_tbl = pd.DataFrame(comp_rows)
"""))

cells.append(code("""# === TABEL 1 — PERFORMA fault detection (mean +/- std antar-fold) ===
print("=== Performa fault detection — Stratified %d-Fold CV ===" % N_SPLITS)
show = perf_tbl.copy()
for c in ["Akurasi", "Precision", "Recall", "F1"]:
    show[c] = show[c].map("{:.3f}".format) + " ± " + show[c + "_std"].map("{:.3f}".format)
show = show[["Skenario", "Metode", "n_kelas", "n_window", "n_fitur", "Akurasi", "Precision", "Recall", "F1"]]
print(show.to_string(index=False))
export_df(perf_tbl, "07_performa_cv_5skenario")
display(show)
"""))

cells.append(code("""# === TABEL 2 — BIAYA KOMPUTASI ===
print("=== Biaya komputasi per skenario x metode ===")
print("cpu_s_total / wall_s_total = seluruh %d fold; infer_ms_per_window = latensi prediksi 1 window" % N_SPLITS)
print(comp_tbl.to_string(index=False))
export_df(comp_tbl, "07_komputasi_cv_5skenario")

print("\\n=== Ongkos rata-rata per metode (lintas 5 skenario) ===")
comp_ring = comp_tbl.groupby("Metode")[["cpu_s_total", "wall_s_total", "peak_mem_mb",
                                          "fit_s_per_fold", "infer_ms_per_window"]].mean().round(3)
print(comp_ring.to_string())
export_df(comp_ring.reset_index(), "07_komputasi_ringkas_per_metode")

print("\\nCatatan: ekstraksi fitur (tabel di atas) sering jadi biaya dominan di")
print("sistem nyata, karena jalan tiap window; latih model hanya sekali di awal.")
display(feat_cost)
"""))

cells.append(code("""# === Plot: performa vs biaya ===
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

perf_tbl.pivot(index="Skenario", columns="Metode", values="F1").plot.bar(
    ax=axes[0, 0], rot=20, yerr=perf_tbl.pivot(index="Skenario", columns="Metode", values="F1_std"), capsize=3)
axes[0, 0].set_title("F1 macro per skenario (mean ± std antar-fold)")
axes[0, 0].set_ylabel("F1 macro"); axes[0, 0].set_ylim(0, 1.05)

perf_tbl.pivot(index="Skenario", columns="Metode", values="Akurasi").plot.bar(ax=axes[0, 1], rot=20)
axes[0, 1].set_title("Akurasi per skenario"); axes[0, 1].set_ylabel("Akurasi"); axes[0, 1].set_ylim(0, 1.05)

comp_tbl.pivot(index="Skenario", columns="Metode", values="cpu_s_total").plot.bar(ax=axes[1, 0], rot=20)
axes[1, 0].set_title("CPU time total (%d fold)" % N_SPLITS); axes[1, 0].set_ylabel("detik")

comp_tbl.pivot(index="Skenario", columns="Metode", values="peak_mem_mb").plot.bar(ax=axes[1, 1], rot=20)
axes[1, 1].set_title("Memori puncak"); axes[1, 1].set_ylabel("MB")

plt.tight_layout(); plt.savefig("exports/07_performa_vs_komputasi.png", dpi=120); plt.show()
print("[Tersimpan] exports/07_performa_vs_komputasi.png")
"""))

cells.append(code("""# === Confusion matrix out-of-fold, metode terbaik per skenario ===
best_per_scen = perf_tbl.loc[perf_tbl.groupby("Skenario")["F1"].idxmax()]
n = len(best_per_scen)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
axes = np.atleast_1d(axes)
for ax, (_, row) in zip(axes, best_per_scen.iterrows()):
    st = STAGE1[(row["Skenario"], row["Metode"])]
    cm = confusion_matrix(st["y"], st["pred_oof"], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=st["class_names"]).plot(
        ax=ax, colorbar=False, values_format=".2f", xticks_rotation=45, cmap="Blues")
    ax.set_title(f"{row['Skenario']}\\n{row['Metode']} F1={row['F1']:.3f}", fontsize=9)
plt.tight_layout(); plt.savefig("exports/07_confusion_oof.png", dpi=120); plt.show()
print("Metode terbaik per skenario:")
print(best_per_scen[["Skenario", "Metode", "Akurasi", "F1"]].to_string(index=False))
"""))

cells.append(md("""# 6. Tahap 2 — Output sensor mana yang rusak (asal fault)

Setelah Tahap 1 memutuskan sebuah window fault, Tahap 2 menunjuk **sensor mana**
yang membawanya. Bentuknya `MultiOutputClassifier` dengan 4 label biner
`[S1, S2, S3, S4]`, dilatih hanya dari window fault.

Dievaluasi dua cara supaya jelas sumber error-nya:

| Evaluasi | Artinya |
|---|---|
| `oracle` | dijalankan pada window yang **memang** fault → batas atas kemampuan Tahap 2 |
| `end_to_end` | dijalankan pada window yang **diprediksi** fault oleh Tahap 1 → angka apa adanya, termasuk kesalahan Tahap 1 |

**Kolom `Prevalensi` wajib dibaca.** Kalau prevalensi ≈ 1 dan ROC-AUC ≈ 0,5,
model hanya menebak "semua sensor rusak" — itu cacat yang membatalkan hasil
notebook per-sensor versi lama.
"""))

cells.append(code("""def cv_stage2(F, y_task, Ysens_task):
    \"\"\"Identifikasi sensor dengan CV yang sama; hasil dikumpulkan out-of-fold.\"\"\"
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.full(Ysens_task.shape, -1, dtype=int)
    oof_proba = np.full(Ysens_task.shape, np.nan, dtype=float)
    tracemalloc.start(); c0 = _time.process_time(); t0 = _time.perf_counter()

    for tr, te in skf.split(F, y_task):
        tr_f = tr[y_task[tr] != 0]                     # latih hanya dari window fault
        if len(tr_f) < 20 or Ysens_task[tr_f].sum() == 0:
            continue
        base = MLPClassifier(hidden_layer_sizes=(max(16, F.shape[1] // 2),), alpha=1e-3,
                             max_iter=300, random_state=RANDOM_SEED,
                             early_stopping=True, n_iter_no_change=12)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                          ("mo", MultiOutputClassifier(base))])
        pipe.fit(F[tr_f], Ysens_task[tr_f])
        oof_pred[te] = pipe.predict(F[te])
        try:
            oof_proba[te] = np.column_stack([e.predict_proba(F[te])[:, 1]
                                             for e in pipe.named_steps["mo"].estimators_])
        except Exception:
            pass

    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    cost = {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024)}

    def evaluate(mask, tag):
        mask = mask & (oof_pred[:, 0] >= 0)
        if mask.sum() == 0:
            return []
        Ye, P, Pr = Ysens_task[mask], oof_pred[mask], oof_proba[mask]
        rows = []
        for j, sname in enumerate(SENSORS):
            p, r, f, _ = precision_recall_fscore_support(Ye[:, j], P[:, j], average="binary", zero_division=0)
            try:
                auc = roc_auc_score(Ye[:, j], Pr[:, j])
            except Exception:
                auc = np.nan
            rows.append({"Eval": tag, "Sensor": sname, "n_window": int(mask.sum()),
                          "Prevalensi": round(float(Ye[:, j].mean()), 3),
                          "Akurasi": round(accuracy_score(Ye[:, j], P[:, j]), 3),
                          "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f, 3),
                          "ROC_AUC": round(auc, 3) if not np.isnan(auc) else np.nan})
        rows.append({"Eval": tag, "Sensor": "SEMUA-4-BENAR", "n_window": int(mask.sum()),
                      "Prevalensi": np.nan, "Akurasi": round(accuracy_score(Ye, P), 3),
                      "Precision": np.nan, "Recall": np.nan,
                      "F1": round(f1_score(Ye, P, average="macro", zero_division=0), 3),
                      "ROC_AUC": np.nan, "Hamming": round(hamming_loss(Ye, P), 3)})
        return rows

    return evaluate, cost

sensor_rows, sensor_cost_rows = [], []
for sc, dat in SCEN_DATA.items():
    for meth in METHOD_LIST:
        if (sc, meth) not in STAGE1:
            continue
        if not budget_ok(400, f"Tahap2 {sc}/{meth}"):
            continue
        log_stage(f"Tahap 2 identifikasi sensor | {sc} | {meth}")
        st = STAGE1[(sc, meth)]
        keep, yy = st["keep"], st["y"]
        F = FEAT_by_method[meth][keep]
        Ysens_task = Ysens_s[keep]
        evaluate, cost = cv_stage2(F, yy, Ysens_task)
        for mask, tag in ((yy != 0, "oracle"), (st["pred_oof"] != 0, "end_to_end")):
            for row in evaluate(mask, tag):
                row.update({"Skenario": sc, "Metode": meth})
                sensor_rows.append(row)
        sensor_cost_rows.append({"Skenario": sc, "Metode": meth,
                                  "cpu_s": round(cost["cpu_s"], 1),
                                  "wall_s": round(cost["wall_s"], 1),
                                  "peak_mem_mb": round(cost["peak_mem_mb"], 1)})

sensor_tbl = pd.DataFrame(sensor_rows)
sensor_cost = pd.DataFrame(sensor_cost_rows)
print("Tahap 2 selesai:", len(sensor_tbl), "baris hasil")
"""))

cells.append(code("""# === TABEL 3 — sensor mana yang rusak (end-to-end, dirantai dari Tahap 1) ===
cols_show = ["Skenario", "Metode", "Eval", "Sensor", "n_window", "Prevalensi",
             "Akurasi", "Precision", "Recall", "F1", "ROC_AUC"]
e2e = sensor_tbl[sensor_tbl.Eval == "end_to_end"]
print("=== Identifikasi sensor rusak — end-to-end ===")
print(e2e[cols_show].to_string(index=False))
export_df(sensor_tbl, "07_identifikasi_sensor_5skenario")

print("\\n=== Ringkas: F1 rata-rata 4 sensor per skenario x metode ===")
ring_sensor = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
               .groupby(["Skenario", "Metode"])[["F1", "ROC_AUC", "Prevalensi"]].mean().round(3))
print(ring_sensor.to_string())
export_df(ring_sensor.reset_index(), "07_identifikasi_sensor_ringkas")

print("\\n=== Biaya komputasi Tahap 2 ===")
print(sensor_cost.to_string(index=False))
export_df(sensor_cost, "07_komputasi_tahap2")
"""))

cells.append(code("""# === Peta panas: F1 identifikasi sensor per skenario ===
piv = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
       .pivot_table(index=["Skenario", "Metode"], columns="Sensor", values="F1"))
fig, ax = plt.subplots(figsize=(7, 0.55 * len(piv) + 2))
im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels([f"{a} | {b}" for a, b in piv.index], fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=8)
ax.set_title("F1 identifikasi sensor rusak (end-to-end)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig("exports/07_peta_sensor.png", dpi=120); plt.show()
print("[Tersimpan] exports/07_peta_sensor.png")
"""))

cells.append(code("""# === Ringkasan gabungan: satu tabel untuk laporan ===
merged = perf_tbl.merge(comp_tbl, on=["Skenario", "Metode"])
sens_f1 = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
           .groupby(["Skenario", "Metode"])["F1"].mean().rename("F1_sensor").reset_index())
sens_all4 = (e2e[e2e.Sensor == "SEMUA-4-BENAR"]
             .groupby(["Skenario", "Metode"])["Akurasi"].mean().rename("Semua_4_benar").reset_index())
final = (merged.merge(sens_f1, on=["Skenario", "Metode"], how="left")
               .merge(sens_all4, on=["Skenario", "Metode"], how="left"))
final = final[["Skenario", "Metode", "n_kelas", "Akurasi", "Precision", "Recall", "F1",
               "cpu_s_total", "peak_mem_mb", "infer_ms_per_window",
               "F1_sensor", "Semua_4_benar"]].round(3)
print("=== RINGKASAN — performa, komputasi, dan identifikasi sensor ===")
print(final.to_string(index=False))
export_df(final, "07_ringkasan_lengkap")
display(final)
log_stage("selesai")
"""))

cells.append(md("""## Ringkasan — apa yang dibuktikan notebook ini

1. **Perbandingan pakai cross-validation, bukan satu split.** Tiap angka disertai
   simpangan baku antar-fold, jadi selisih EDM-Fuzzy vs JSD-Fuzzy bisa dinilai
   berarti atau hanya derau split.
2. **Fault detection dilaporkan dua sisi.** Performa (akurasi/precision/recall/F1)
   dan biaya (CPU time, wall time, memori puncak, latensi inferensi per window).
   Metode yang menang akurasi belum tentu menang ongkos — kolom
   `infer_ms_per_window` yang menentukan apakah muat dijalankan online di broker.
3. **Broker tetap sebagai pengumpul.** 4 sensor disatukan jadi satu tabel
   ter-align waktu, identitas sensor dipertahankan (bukan dirata-rata) — dasarnya
   di notebook `02`.
4. **Sensor rusak bisa ditunjuk**, dirantai dari hasil klasifikasi tiap skenario,
   lengkap dengan pembanding `oracle` vs `end_to_end` supaya jelas berapa error
   yang berasal dari Tahap 1.

**Cara baca angkanya:** utamakan **F1 macro**, bukan akurasi mentah — beberapa
skenario punya jumlah kelas berbeda sehingga akurasi tidak sebanding lintas
skenario. Untuk Tahap 2, baca `F1` bersama `Prevalensi` dan `ROC_AUC`;
prevalensi tinggi + AUC ≈ 0,5 berarti model hanya menebak.

**File keluaran** ada di folder `exports/`:
`07_performa_cv_5skenario.csv`, `07_komputasi_cv_5skenario.csv`,
`07_ongkos_ekstraksi_fitur.csv`, `07_identifikasi_sensor_5skenario.csv`,
`07_ringkasan_lengkap.csv`, plus gambar `07_*.png`.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "| cells:", len(cells))
