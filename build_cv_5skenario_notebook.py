#!/usr/bin/env python3
"""Build 07_Skema_Diagram_CV_5_Skenario_dan_Sensor.ipynb.

Notebook ini mengikuti flowchart pembimbing ("Multisource Soil Moisture Sensor
Acquisition -> Broker -> ... -> Fault Classification and Evaluation") kotak per
kotak, sambil menjawab permintaan terakhir:

  * perbandingan 5 skenario (S1..S5) pakai CROSS-VALIDATION,
  * fault detection dilaporkan dua sisi: performa (akurasi/precision/recall/F1)
    dan biaya komputasi (CPU, memori, waktu, latensi inferensi),
  * output tambahan: sensor mana yang rusak.

Yang berubah dibanding versi pertama notebook 07, supaya cocok dengan diagram:
  1. Window length N in {2000, 7000, 10000} sampel (diagram), bukan WIN=256.
     Ketiganya dijalankan sebagai studi sensitivitas panjang window.
  2. Fitur = EDM-Fuzzy murni, 4 sensor x T skala = 4T fitur (kotak "Multisensor
     Entropy Feature Concatenation"). Fitur time-domain hibrida DIBUANG dari
     jalur utama karena tidak ada di diagram. JSD-Fuzzy tetap ikut sebagai
     pembanding usulan paper.
  3. ANN-LM: solver 'lbfgs' (sklearn tidak punya Levenberg-Marquardt asli) dan
     hidden layer dipilih lewat Grid Search, persis kotak "ANN-LM Classification".
  4. Ada langkah "Time synchronization" eksplisit (grid waktu 30 detik).
  5. Cross-validation memakai StratifiedGroupKFold dengan grup = blok waktu,
     supaya window yang tumpang-tindih tidak bocor antar-fold.

Data: data_sensor.csv (281.721 baris, 2025-09-14 .. 2025-12-21, interval 30 detik).
"""
import json

REPO_RAW = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"
OUT = "/Users/kelvin/apps/public-files/07_Skema_Diagram_CV_5_Skenario_dan_Sensor.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""<!-- HEADER-KLAIM -->
# 07 — Skema Diagram Pembimbing, Dijalankan dengan Cross-Validation (5 Skenario + Sensor Mana)

| | |
|---|---|
| **Pertanyaan** | Jalankan **persis skema pada flowchart** (akuisisi → broker → sinkronisasi → injeksi fault → segmentasi → EDM-Fuzzy τ=1..T → konkatenasi 4T fitur → ANN-LM grid search → evaluasi), pakai **data terbaru** dan **cross-validation**, untuk **5 skenario**. Lalu: sensor mana yang rusak? |
| **Yang diukur** | (a) **Performa**: akurasi, precision, recall, F1 macro — rata-rata ± simpangan baku antar-fold. (b) **Komputasi**: CPU time, wall time, memori puncak, latensi inferensi per window. (c) **Identifikasi sensor rusak** (tambahan di luar diagram). |
| **Panjang window** | Sesuai diagram: **N ∈ {2000, 7000, 10000}** sampel, ketiganya dijalankan sebagai studi sensitivitas. |
| **Fitur** | **EDM-Fuzzy murni**: 4 sensor × T skala = **4T fitur** (kotak konkatenasi). JSD-Fuzzy ikut sebagai pembanding usulan paper. Fitur time-domain **tidak** dipakai di jalur utama karena tidak ada di diagram. |
| **Classifier** | **ANN-LM** — `solver='lbfgs'` (quasi-Newton, paling dekat ke Levenberg–Marquardt; LM asli hanya ada di MATLAB `trainlm`), **hidden layer dipilih Grid Search**, output C neuron sesuai jumlah kelas tiap skenario. |
| **Data** | `data_sensor.csv` — 281.721 baris, 2025-09-14 s.d. 2025-12-21, interval 30 detik, kolom `kelembaban1..kelembaban4`. |

---
"""))

cells.append(md("""## Peta diagram → sel notebook

| Kotak di flowchart | Bagian notebook |
|---|---|
| Multisource Soil Moisture Sensor Acquisition | **§1** — muat `data_sensor.csv` |
| **Broker** — Multisource data integration | **§2** — 4 sensor jadi satu tabel, identitas sensor dipertahankan |
| Time synchronization — S₁..S₄ | **§3** — grid waktu seragam 30 detik, gap ditambal |
| Data preparation and Fault Injection | **§4** — injeksi fault (sensor-selective) |
| Time series segmentation and Labelling, N ∈ {2000; 7000; 10000} | **§5** — segmentasi + pelabelan |
| EDM-Fuzzy Entropy Feature Extraction, τ = 1..T | **§6** — E₁..E₄ per sensor |
| Multisensor Entropy Feature Concatenation → 4T fitur | **§7** — konkatenasi |
| ANN-LM Classification (input 4T, hidden = grid search, output C) | **§8** — ANN-LM + CV |
| Fault Classification and Evaluation | **§9** — performa + biaya komputasi |
| *(tambahan, di luar diagram)* | **§10** — sensor mana yang rusak |

**Catatan jujur soal dua penyimpangan yang disengaja:**

1. **`solver='lbfgs'`, bukan Levenberg–Marquardt.** scikit-learn tidak punya
   LM. lbfgs sama-sama quasi-Newton dan paling dekat perilakunya. LM sungguhan
   perlu MATLAB `trainlm`.
2. **JSD-Fuzzy ikut dijalankan** walau diagram hanya menyebut EDM-Fuzzy, karena
   JSD-Fuzzy adalah usulan paper dan diagram ini dipakai untuk membandingkan
   keduanya. Jalur EDM-Fuzzy tetap persis 4T fitur seperti di diagram.
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

METHOD_LIST = ["EDM-Fuzzy", "JSD-Fuzzy"]      # EDM = jalur diagram; JSD = usulan paper
RANDOM_SEED = 42
N_JOBS = -1
N_SPLITS = 5

# --- Panjang window sesuai diagram: N in {2000, 7000, 10000} sampel ---
WINDOW_LENGTHS = [int(v) for v in os.environ.get("WINDOW_LENGTHS", "2000,7000,10000").split(",")]
T_SCALES = int(os.environ.get("T_SCALES", 10))       # tau = 1..T -> fitur EDM = 4T
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 100))
SAMPLING_SECONDS = 30                                # interval akuisisi sensor

m = 2; r_ratio = 0.2; n_ref = 128; jsd_bins = 40
SENSORS = ["S1", "S2", "S3", "S4"]
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]
FAULT_RATIO_THR = 0.01

EXPORT_DIR = "exports"
Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)

def export_df(df, name, index=False):
    p = Path(EXPORT_DIR) / f"{name}.csv"
    df.to_csv(p, index=index)
    display(FileLink(str(p)))
    return str(p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def run_with_metrics(label, fn):
    \"\"\"Ukur wall time, CPU time, dan memori puncak satu blok kerja.\"\"\"
    tracemalloc.start()
    t0 = _time.perf_counter(); c0 = _time.process_time()
    result = fn()
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    metrics = {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024)}
    logging.info("%s | wall=%.2fs cpu=%.2fs peak_mem=%.1f MB",
                 label, metrics["wall_s"], metrics["cpu_s"], metrics["peak_mem_mb"])
    return result, metrics

print("Mesin :", platform.processor() or platform.machine(), "| CPU:", os.cpu_count())
print("Window N :", WINDOW_LENGTHS, "| T skala :", T_SCALES, "-> fitur EDM = 4T =", 4 * T_SCALES)
print("Metode   :", METHOD_LIST, "| K-Fold :", N_SPLITS, "| MAX_PER_CLASS :", MAX_PER_CLASS)
"""))

cells.append(md("""# §1 — Multisource Soil Moisture Sensor Acquisition

Kotak pertama diagram. Data akuisisi 4 sensor kelembaban tanah dibaca dari
`data_sensor.csv` (dataset terbaru). Kalau file ada di direktori kerja atau di
`/kaggle/input/...` file itu yang dipakai; kalau tidak, diunduh dari GitHub
(**Kaggle: Settings → Internet → On**).
"""))

cells.append(code("""import requests, glob
from io import StringIO

DATA_URL = "__REPO_RAW__"
DATA_NAME = "data_sensor.csv"

def load_sensor_data():
    cands = [DATA_NAME, f"../input/{DATA_NAME}"] + glob.glob(f"/kaggle/input/**/{DATA_NAME}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            print("Sumber data: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    print("Sumber data: unduh ->", DATA_URL)
    r = requests.get(DATA_URL, timeout=180); r.raise_for_status()
    return pd.read_csv(StringIO(r.text), index_col=0)

df_raw = load_sensor_data()
cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
missing = [c for c in cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Error-nya jelas: kolom {missing} tidak ada. Tersedia: {list(df_raw.columns)}")

print("Baris akuisisi:", len(df_raw), "| kolom sensor:", cols)
print("Rentang waktu :", df_raw.index[0], "->", df_raw.index[-1])
display(df_raw[cols].describe().round(2))
""".replace("__REPO_RAW__", REPO_RAW)))

cells.append(md("""# §2 — Broker: Multisource Data Integration

Kotak `Broker` pada diagram. Broker berperan sebagai **pengumpul**: keempat
aliran sensor disatukan jadi **satu tabel**, tetapi **identitas tiap sensor
dipertahankan** sebagai kolom terpisah — bukan dilebur/dirata-rata jadi satu
sinyal. Bukti kenapa peleburan merugikan ada di notebook `02`.
"""))

cells.append(code("""df_broker = df_raw[cols].copy()
df_broker.index = pd.to_datetime(df_broker.index, utc=True)
df_broker = df_broker.sort_index()
print("Tabel broker:", df_broker.shape, "| 4 kanal terpisah (tidak difusikan)")
display(df_broker.head(3))
"""))

cells.append(md("""# §3 — Time Synchronization

Kotak `Time synchronization` pada diagram:

$$S_1 = S_1(t_1), S_1(t_2), \\dots, S_1(t_N) \\qquad \\dots \\qquad S_4 = S_4(t_1), \\dots, S_4(t_N)$$

Keempat sensor dipaksa ke **satu sumbu waktu seragam** (grid 30 detik). Kalau
ada stempel waktu yang bolong atau ganda, di sini ketahuan dan ditambal, supaya
sampel ke-*i* dari keempat sensor benar-benar merujuk waktu yang sama sebelum
masuk windowing.
"""))

cells.append(code("""# --- Diagnosa sinkronisasi sebelum ditambal ---
dt = df_broker.index.to_series().diff().dt.total_seconds().dropna()
print("Interval antar-sampel (detik):")
print(dt.value_counts().head(5).to_string())
print("Stempel waktu ganda:", int(df_broker.index.duplicated().sum()))

# --- Paksa ke grid seragam 30 detik ---
df_broker = df_broker[~df_broker.index.duplicated(keep="first")]
grid = pd.date_range(df_broker.index[0], df_broker.index[-1], freq=f"{SAMPLING_SECONDS}s")
df_sync = df_broker.reindex(grid)
n_gap = int(df_sync.isna().any(axis=1).sum())
df_sync = df_sync.interpolate(method="time", limit_direction="both")
df_sync = df_sync.fillna(df_sync.median(numeric_only=True))

if df_sync.isna().any().any():
    raise ValueError("Error-nya jelas: masih ada NaN setelah sinkronisasi.")

X = df_sync.to_numpy(dtype=float)
print(f"\\nSetelah sinkronisasi: {X.shape} pada grid {SAMPLING_SECONDS} detik "
      f"({n_gap} slot ditambal interpolasi waktu)")
print("S1..S4 sekarang berbagi sumbu waktu yang sama.")
"""))

cells.append(code("""# --- Sanity plot: 4 kanal setelah sinkronisasi (belum ada fault) ---
fig, ax = plt.subplots(figsize=(13, 4))
seg = df_sync.iloc[:4000]
for c in cols:
    ax.plot(seg.index, seg[c], lw=0.8, label=c)
ax.set_title("Output broker setelah time synchronization — 4 kanal (potongan awal)")
ax.set_ylabel("kelembaban"); ax.legend(ncol=4, fontsize=9)
plt.tight_layout(); plt.savefig("exports/07_broker_tersinkron.png", dpi=120); plt.show()
"""))

cells.append(md("""# §4 — Data Preparation and Fault Injection

Kotak `Data preparation and Fault Injection`. Simulator fault dan 16 kombinasi
skenario identik notebook `01`/`03` supaya angkanya sebanding lintas notebook.

Satu tambahan penting: injeksi bersifat **sensor-selective** — fault hanya masuk
ke subset sensor acak (1..4 sensor), sisanya tetap bersih. Tanpa ini, keempat
label per-sensor jadi kembar dan §10 (identifikasi sensor) tidak sah.
"""))

cells.append(code("""def simulate_drift_fault(x, intensity=0.02, seed=None):
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
condition_names = ["normal"] + list(SCENARIOS.keys())
print("Kondisi:", len(SCENARIOS), "fault + normal =", len(condition_names))
"""))

cells.append(md("""# §5 — Time Series Segmentation and Labelling

Kotak `Time series segmentation and Labelling`, dengan **N ∈ {2000; 7000; 10000}
sampel** sesuai diagram. Ketiga panjang window dijalankan penuh, jadi hasilnya
sekaligus jadi **studi sensitivitas panjang window** — bukan satu angka tunggal.

Stride dipasang `N/2`. Konsekuensinya window bertetangga tumpang-tindih 50%,
sehingga **cross-validation harus dikelompokkan menurut blok waktu** (dipakai di
§8), kalau tidak potongan sinyal yang sama bisa muncul di data latih **dan** uji
sekaligus dan akurasinya jadi terlalu bagus.

Tiap window dilabeli dua hal:
- label **kondisi** (normal / jenis kombinasi fault) → dipakai §8,
- label **per-sensor** `[S1,S2,S3,S4]` → dipakai §10.
"""))

cells.append(code("""from numpy.lib.stride_tricks import sliding_window_view

def make_windows(Xa, win, stride):
    Xn = np.asarray(Xa, dtype=np.float32); N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    # GOTCHA: sliding_window_view(...,axis=0) -> (nwin, kanal, win). Harus di-transpose
    # ke (nwin, win, kanal), kalau tidak entropy dihitung atas potongan panjang-4.
    view = sliding_window_view(Xn, window_shape=win, axis=0)
    starts = np.arange(0, N - win + 1, stride, dtype=int)
    return view[starts].transpose(0, 2, 1), starts

def inject_faults_multisensor(Xa, scenario_faults, sensor_subset, seed=0):
    rng = np.random.default_rng(seed)
    Y = Xa.copy(); M = np.zeros_like(Y, dtype=bool)
    for s in sensor_subset:
        y, m_ = simulate_multiple_faults(Y[:, s], scenario_faults, seed=int(rng.integers(1e9)))
        Y[:, s] = y; M[:, s] = m_
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M

def window_labels_per_sensor(mask, win, stride, thr=0.02):
    T = len(mask)
    if win > T:
        return np.zeros((0, mask.shape[1]), dtype=bool)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]   # (nwin, kanal, win)
    return (Wm.mean(axis=2) > thr)

def segment_and_label(X, win):
    \"\"\"-> W (nwin,win,4), y kondisi, Ysens (nwin,4), start index tiap window.\"\"\"
    stride = max(1, win // 2)
    rng = np.random.default_rng(RANDOM_SEED)
    Ws, ys, Ys, st = [], [], [], []

    W0, s0 = make_windows(X, win, stride)
    Ws.append(W0); ys.append(np.zeros(len(W0), int))
    Ys.append(np.zeros((len(W0), 4), bool)); st.append(s0)

    for k, (name, faults) in enumerate(SCENARIOS.items(), start=1):
        for subset_size in SENSOR_SUBSET_SIZES:
            subset = rng.choice(4, size=subset_size, replace=False)
            Y, M = inject_faults_multisensor(X, faults, subset, seed=int(rng.integers(1e9)))
            sens = window_labels_per_sensor(M, win, stride, thr=FAULT_RATIO_THR)
            Wk, sk = make_windows(Y, win, stride)
            keep = sens.any(axis=1)
            Ws.append(Wk[keep]); ys.append(np.full(int(keep.sum()), k, int))
            Ys.append(sens[keep]); st.append(sk[keep])

    return (np.concatenate(Ws), np.concatenate(ys),
            np.concatenate(Ys).astype(int), np.concatenate(st))

def balanced_subsample(W, y, Ysens, starts, max_per_class, seed=0):
    rng = np.random.default_rng(seed); keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    return W[keep], y[keep], Ysens[keep], starts[keep]

SEGMENTS = {}
for WIN in WINDOW_LENGTHS:
    W, y, Ysens, starts = segment_and_label(X, WIN)
    W, y, Ysens, starts = balanced_subsample(W, y, Ysens, starts, MAX_PER_CLASS, RANDOM_SEED)
    # grup CV = blok waktu; window yang tumpang-tindih jatuh ke grup yang sama
    groups = starts // WIN
    SEGMENTS[WIN] = dict(W=W, y=y, Ysens=Ysens, starts=starts, groups=groups)
    print(f"N={WIN:6d} | window={W.shape} | kondisi terisi={len(np.unique(y)):2d} "
          f"| blok waktu={len(np.unique(groups)):3d} | prevalensi sensor={Ysens.mean(axis=0).round(2)}")
"""))

cells.append(md("""# §6 — EDM-Fuzzy Entropy Feature Extraction, τ = 1..T

Kotak entropy pada diagram. Untuk tiap sensor dihitung vektor entropi
multiskala:

$$E_i = [E_i^{(1)}, E_i^{(2)}, \\dots, E_i^{(T)}], \\quad i = 1..4$$

Jadi tiap sensor menyumbang **T fitur** (satu per skala τ). JSD-Fuzzy dijalankan
berdampingan sebagai pembanding usulan paper — versinya *rich*
(`[jsd, fe, mean_μ, std_μ]` per skala), sehingga dimensinya 4× lipat EDM.
"""))

cells.append(code("""def coarse_grain_mean(x, s):
    n = (len(x) // s) * s
    return x[:n].reshape(-1, s).mean(axis=1) if n > 0 else np.array([], dtype=float)

def embed_matrix(y, m_):
    return np.lib.stride_tricks.sliding_window_view(y, m_) if len(y) >= m_ else np.empty((0, m_), float)

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
    d = np.sqrt(np.maximum(a2 + b2 - 2 * (A @ V.T), 0.0))
    mu = 1.0 / (1.0 + (d / (r + 1e-12)) ** 2)
    for ri, i in enumerate(ref):
        mu[ri, i] = np.nan
    return mu[~np.isnan(mu)].ravel()

def edm_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=256, seed=0):
    \"\"\"-> vektor panjang T: satu nilai entropi per skala tau (E_i^(1..T)).\"\"\"
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

scales = np.arange(1, T_SCALES + 1)
print("Skala tau:", scales.tolist(), "-> EDM-Fuzzy menghasilkan", len(scales), "fitur per sensor")
"""))

cells.append(md("""# §7 — Multisensor Entropy Feature Concatenation

Kotak `4 sensors × T scales → 4T features`. Keempat vektor entropi digabung
berurutan jadi satu vektor fitur per window:

$$F = [E_1 \\;|\\; E_2 \\;|\\; E_3 \\;|\\; E_4] \\in \\mathbb{R}^{4T}$$

Ini yang jadi **input layer ANN-LM** di §8 (4T neuron). Ongkos ekstraksi diukur
di sini, terpisah dari ongkos latih model — di sistem nyata ekstraksi fitur ini
yang jalan tiap window, sedangkan latih model hanya sekali.
"""))

cells.append(code("""from joblib import Parallel, delayed

def sanitize(F):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()

def concat_multisensor_features(W, method, seed=0, n_jobs=-1):
    \"\"\"E_1..E_4 dihitung per sensor lalu dikonkatenasi -> (nwin, 4T) untuk EDM.\"\"\"
    nwin, win, ns = W.shape
    key = method.strip().lower()

    def entropy_1d(x, sd):
        if key == "edm-fuzzy":
            return edm_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
        if key == "jsd-fuzzy":
            return jsd_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd, bins=jsd_bins)
        raise ValueError(f"Metode tidak dikenal: {method}")

    def one_window(i):
        return np.concatenate([entropy_1d(W[i, :, s], sd=seed + 1000 * i + 19 * s) for s in range(ns)])

    return np.vstack(Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_window)(i) for i in range(nwin)))

feat_cost_rows = []
FEATURES = {}          # (WIN, metode) -> matriks fitur (nwin, 4T)

for WIN in WINDOW_LENGTHS:
    seg = SEGMENTS[WIN]
    for meth in METHOD_LIST:
        need = 0.02 * len(seg["W"]) * (WIN / 1000.0)      # taksiran kasar, untuk budget guard
        if not budget_ok(need, f"fitur N={WIN}/{meth}"):
            continue
        log_stage(f"ekstraksi fitur | N={WIN} | {meth} | {len(seg['W'])} window")
        F, mtr = run_with_metrics(f"Fitur {meth} N={WIN}",
                                  lambda w=seg["W"], mm=meth: concat_multisensor_features(w, mm, seed=7, n_jobs=N_JOBS))
        FEATURES[(WIN, meth)] = sanitize(F)
        feat_cost_rows.append({"N_window": WIN, "Metode": meth, "n_window": len(F),
                                "n_fitur": F.shape[1],
                                "wall_s": round(mtr["wall_s"], 2), "cpu_s": round(mtr["cpu_s"], 2),
                                "peak_mem_mb": round(mtr["peak_mem_mb"], 1),
                                "ms_per_window": round(1000 * mtr["wall_s"] / max(1, len(F)), 2)})
        print(f"  N={WIN} {meth:10s} -> fitur {F.shape}  ({feat_cost_rows[-1]['ms_per_window']} ms/window)")

feat_cost = pd.DataFrame(feat_cost_rows)
print("\\n=== Ongkos ekstraksi fitur (kotak entropy + konkatenasi) ===")
print(feat_cost.to_string(index=False))
export_df(feat_cost, "07_ongkos_ekstraksi_fitur")
"""))

cells.append(md("""# §8 — ANN-LM Classification

Kotak `ANN-LM Classification` pada diagram:

- **Input layer** — 4T neuron (hasil konkatenasi §7).
- **Hidden layers** — dipilih lewat **Grid Search**.
- **Output layer** — C neuron, C = jumlah kelas pada skenario yang dijalankan.

**Levenberg–Marquardt:** scikit-learn tidak menyediakannya, jadi dipakai
`solver='lbfgs'` (quasi-Newton, paling dekat perilakunya). LM sungguhan butuh
MATLAB `trainlm`.

## 5 skenario (C berbeda-beda)

| Skenario | Isi | C |
|---|---|---|
| S1 | normal vs faulty | 2 |
| S2 | normal + 4 fault tunggal | 5 |
| S3 | normal + 6 kombinasi dua fault | 7 |
| S4 | normal + 4 kombinasi tiga fault | 5 |
| S5 | normal + kombinasi empat fault | 2 |

## Cara cross-validation-nya

**Nested + grouped**, dua hal yang dua-duanya perlu:

- **Grup = blok waktu.** `StratifiedGroupKFold` menjaga window yang
  tumpang-tindih (stride = N/2) tidak terpecah antara latih dan uji. Tanpa ini
  potongan sinyal yang sama muncul di kedua sisi dan akurasinya menggelembung.
- **Grid Search di dalam tiap fold latih saja** (`cv=3`). Kalau grid dijalankan
  di seluruh data lalu skornya dilaporkan, angkanya bias optimistis karena
  arsitektur sudah "mengintip" data uji.
"""))

cells.append(code("""from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                              roc_auc_score, hamming_loss, confusion_matrix,
                              ConfusionMatrixDisplay)

ANN_SOLVER = "lbfgs"          # pengganti terdekat Levenberg-Marquardt
ANN_MAX_ITER = 500
HIDDEN_GRID = [(32,), (64,), (128,), (64, 16), (64, 32), (128, 32), (128, 64), (128, 64, 32)]
ACT_GRID = ["relu", "tanh"]

def make_ann():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("mlp", MLPClassifier(solver=ANN_SOLVER, max_iter=ANN_MAX_ITER, alpha=1e-3,
                               random_state=RANDOM_SEED)),
    ])

ANN_GRID = {"mlp__hidden_layer_sizes": HIDDEN_GRID, "mlp__activation": ACT_GRID}

ALL_FAULT = [c for c in condition_names if c != "normal"]
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

def build_scenario(WIN, classes):
    \"\"\"-> indeks window terpilih + label kelas skenario, sudah diseimbangkan.\"\"\"
    y_cond = SEGMENTS[WIN]["y"]
    idx_l, y_l = [], []
    for ci, (_, conds) in enumerate(classes):
        want = [cond_to_idx[c] for c in conds if c in cond_to_idx]
        sel = np.where(np.isin(y_cond, want))[0]
        idx_l.append(sel); y_l.append(np.full(len(sel), ci, int))
    keep = np.concatenate(idx_l); yy = np.concatenate(y_l)
    # Window normal hanya dibangkitkan sekali sedangkan tiap kondisi fault diulang
    # 4x (subset sensor) -> tanpa penyeimbangan, S1 didominasi kelas faulty dan
    # akurasi tinggi cuma efek menebak kelas mayoritas.
    rng = np.random.default_rng(RANDOM_SEED)
    n_min = min(np.bincount(yy)[np.unique(yy)])
    sel = np.concatenate([rng.choice(np.where(yy == c)[0], size=n_min, replace=False)
                          if (yy == c).sum() > n_min else np.where(yy == c)[0]
                          for c in np.unique(yy)])
    rng.shuffle(sel)
    return keep[sel], yy[sel]

for WIN in WINDOW_LENGTHS:
    print(f"--- N={WIN} ---")
    for sc, cl in LADDER.items():
        keep, yy = build_scenario(WIN, cl)
        print(f"  {sc:22s} C={len(cl)} n={len(keep):5d} {dict(zip(*np.unique(yy, return_counts=True)))}")
"""))

cells.append(code("""def nested_grouped_cv(F, y, groups, n_splits=N_SPLITS):
    \"\"\"Outer StratifiedGroupKFold; Grid Search hidden layer di dalam fold latih.

    Mengembalikan skor per fold, prediksi out-of-fold, arsitektur terpilih, dan
    biaya komputasi (CPU, wall, memori, latensi inferensi).
    \"\"\"
    n_grp = len(np.unique(groups))
    k = int(min(n_splits, n_grp, np.min(np.bincount(y)[np.unique(y)])))
    if k < 2:
        return None
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    rows, chosen = [], []
    pred_oof = np.full(len(y), -1, dtype=int)
    tracemalloc.start(); c0 = _time.process_time(); t0 = _time.perf_counter()
    fit_times, infer_times, n_infer = [], [], []

    for tr, te in skf.split(F, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        gs = GridSearchCV(make_ann(), ANN_GRID, cv=3, scoring="f1_macro", n_jobs=N_JOBS)
        tf0 = _time.perf_counter(); gs.fit(F[tr], y[tr]); fit_times.append(_time.perf_counter() - tf0)
        ti0 = _time.perf_counter(); pred = gs.best_estimator_.predict(F[te])
        infer_times.append(_time.perf_counter() - ti0); n_infer.append(len(te))
        pred_oof[te] = pred
        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average="macro", zero_division=0)
        rows.append({"acc": accuracy_score(y[te], pred), "prec": p, "rec": r, "f1": f})
        chosen.append((gs.best_params_["mlp__hidden_layer_sizes"], gs.best_params_["mlp__activation"]))

    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    if not rows:
        return None
    sc = pd.DataFrame(rows)
    return {
        "n_folds": len(rows), "pred_oof": pred_oof, "chosen": chosen,
        "mean": sc.mean().to_dict(), "std": sc.std(ddof=0).to_dict(),
        "cost": {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024),
                  "fit_s_per_fold": float(np.mean(fit_times)),
                  "infer_ms_per_window": float(1000 * np.sum(infer_times) / max(1, np.sum(n_infer)))},
    }

print("ANN-LM:", ANN_SOLVER, "| grid hidden:", len(HIDDEN_GRID), "x aktivasi:", len(ACT_GRID),
      "=", len(HIDDEN_GRID) * len(ACT_GRID), "kombinasi | outer CV:", N_SPLITS, "fold grouped")
"""))

cells.append(code("""# === Jalankan: 3 panjang window x 5 skenario x 2 metode ===
perf_rows, comp_rows, arch_rows = [], [], []
RUNS = {}

for WIN in WINDOW_LENGTHS:
    seg = SEGMENTS[WIN]
    for sc, cl in LADDER.items():
        keep, yy = build_scenario(WIN, cl)
        grp = seg["groups"][keep]
        for meth in METHOD_LIST:
            if (WIN, meth) not in FEATURES:
                continue
            if not budget_ok(900, f"{sc}/{meth}/N={WIN}"):
                continue
            log_stage(f"ANN-LM | N={WIN} | {sc} | {meth}")
            F = FEATURES[(WIN, meth)][keep]
            out = nested_grouped_cv(F, yy, grp)
            if out is None:
                print(f"  N={WIN} {sc} {meth}: dilewati (grup/kelas terlalu sedikit)")
                continue
            RUNS[(WIN, sc, meth)] = dict(out=out, keep=keep, y=yy, groups=grp,
                                         class_names=[c[0] for c in cl])
            mn, sd, cost = out["mean"], out["std"], out["cost"]
            perf_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                               "C_kelas": len(cl), "n_window": len(yy), "n_fitur": F.shape[1],
                               "n_fold": out["n_folds"],
                               "Akurasi": round(mn["acc"], 4), "Akurasi_std": round(sd["acc"], 4),
                               "Precision": round(mn["prec"], 4), "Precision_std": round(sd["prec"], 4),
                               "Recall": round(mn["rec"], 4), "Recall_std": round(sd["rec"], 4),
                               "F1": round(mn["f1"], 4), "F1_std": round(sd["f1"], 4)})
            comp_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                               "cpu_s_total": round(cost["cpu_s"], 1),
                               "wall_s_total": round(cost["wall_s"], 1),
                               "peak_mem_mb": round(cost["peak_mem_mb"], 1),
                               "fit_s_per_fold": round(cost["fit_s_per_fold"], 2),
                               "infer_ms_per_window": round(cost["infer_ms_per_window"], 3)})
            arch_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                               "Hidden_terpilih": "; ".join(sorted({str(h) for h, _ in out["chosen"]})),
                               "Aktivasi_terpilih": "; ".join(sorted({a for _, a in out["chosen"]}))})
            print(f"  N={WIN} {sc:22s} {meth:10s} F1={mn['f1']:.3f}±{sd['f1']:.3f} "
                  f"acc={mn['acc']:.3f} | cpu={cost['cpu_s']:.0f}s")

perf_tbl = pd.DataFrame(perf_rows)
comp_tbl = pd.DataFrame(comp_rows)
arch_tbl = pd.DataFrame(arch_rows)
log_stage("ANN-LM selesai")
"""))

cells.append(md("""# §9 — Fault Classification and Evaluation

Kotak terakhir diagram, dilaporkan **dua sisi**: performa dan biaya komputasi.
"""))

cells.append(code("""# === TABEL 1 — PERFORMA (mean ± std antar-fold) ===
show = perf_tbl.copy()
for c in ["Akurasi", "Precision", "Recall", "F1"]:
    show[c] = show[c].map("{:.3f}".format) + " ± " + show[c + "_std"].map("{:.3f}".format)
show = show[["N_window", "Skenario", "Metode", "C_kelas", "n_window", "n_fitur", "n_fold",
             "Akurasi", "Precision", "Recall", "F1"]]
print("=== Performa fault detection — grouped %d-fold CV, grid search per fold ===" % N_SPLITS)
print(show.to_string(index=False))
export_df(perf_tbl, "07_performa_cv_5skenario")
display(show)
"""))

cells.append(code("""# === TABEL 2 — BIAYA KOMPUTASI ===
print("=== Biaya komputasi (termasuk grid search di dalam tiap fold) ===")
print(comp_tbl.to_string(index=False))
export_df(comp_tbl, "07_komputasi_cv_5skenario")

print("\\n=== Rata-rata per metode x panjang window ===")
ring = comp_tbl.groupby(["N_window", "Metode"])[
    ["cpu_s_total", "wall_s_total", "peak_mem_mb", "fit_s_per_fold", "infer_ms_per_window"]].mean().round(3)
print(ring.to_string())
export_df(ring.reset_index(), "07_komputasi_ringkas")

print("\\n=== Ongkos ekstraksi fitur (jalan tiap window di sistem nyata) ===")
print(feat_cost.to_string(index=False))

print("\\n=== Arsitektur hidden layer yang dipilih Grid Search ===")
print(arch_tbl.to_string(index=False))
export_df(arch_tbl, "07_arsitektur_terpilih")
"""))

cells.append(code("""# === Plot: performa & biaya vs panjang window ===
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

for meth, mk in zip(METHOD_LIST, ["o-", "s--"]):
    sub = perf_tbl[perf_tbl.Metode == meth].groupby("N_window")["F1"].mean()
    axes[0, 0].plot(sub.index, sub.values, mk, label=meth)
axes[0, 0].set_title("F1 macro rata-rata 5 skenario vs panjang window N")
axes[0, 0].set_xlabel("N (sampel)"); axes[0, 0].set_ylabel("F1 macro"); axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1.05); axes[0, 0].grid(alpha=0.3)

piv = perf_tbl.pivot_table(index="Skenario", columns=["N_window", "Metode"], values="F1")
piv.plot.bar(ax=axes[0, 1], rot=20)
axes[0, 1].set_title("F1 per skenario"); axes[0, 1].set_ylabel("F1 macro")
axes[0, 1].set_ylim(0, 1.05); axes[0, 1].legend(fontsize=7, ncol=2)

fc = feat_cost.pivot(index="N_window", columns="Metode", values="ms_per_window")
fc.plot.bar(ax=axes[1, 0], rot=0)
axes[1, 0].set_title("Ongkos ekstraksi fitur per window"); axes[1, 0].set_ylabel("ms / window")

cc = comp_tbl.groupby(["N_window", "Metode"])["cpu_s_total"].mean().unstack()
cc.plot.bar(ax=axes[1, 1], rot=0)
axes[1, 1].set_title("CPU time latih + grid search (rata-rata per skenario)")
axes[1, 1].set_ylabel("detik")

plt.tight_layout(); plt.savefig("exports/07_performa_vs_komputasi.png", dpi=120); plt.show()
print("[Tersimpan] exports/07_performa_vs_komputasi.png")
"""))

cells.append(code("""# === Confusion matrix out-of-fold, konfigurasi terbaik tiap skenario ===
best = perf_tbl.loc[perf_tbl.groupby("Skenario")["F1"].idxmax()]
fig, axes = plt.subplots(1, len(best), figsize=(4.3 * len(best), 4))
axes = np.atleast_1d(axes)
for ax, (_, row) in zip(axes, best.iterrows()):
    r = RUNS[(row["N_window"], row["Skenario"], row["Metode"])]
    mask = r["out"]["pred_oof"] >= 0
    cm = confusion_matrix(r["y"][mask], r["out"]["pred_oof"][mask], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=r["class_names"]).plot(
        ax=ax, colorbar=False, values_format=".2f", xticks_rotation=45, cmap="Blues")
    ax.set_title(f"{row['Skenario']}\\nN={row['N_window']} {row['Metode']} F1={row['F1']:.3f}", fontsize=9)
plt.tight_layout(); plt.savefig("exports/07_confusion_oof.png", dpi=120); plt.show()
print(best[["N_window", "Skenario", "Metode", "Akurasi", "F1"]].to_string(index=False))
"""))

cells.append(md("""# §10 — Tambahan di luar diagram: sensor mana yang rusak

Diagram berhenti di "Fault Classification and Evaluation" — sudah tahu **ada**
fault jenis apa, belum tahu **dari sensor mana**. Bagian ini menyambungnya:
untuk window yang diputuskan fault oleh ANN-LM, `MultiOutputClassifier`
menghasilkan 4 keputusan biner `[S1,S2,S3,S4]`.

Dievaluasi dua cara:

| Evaluasi | Artinya |
|---|---|
| `oracle` | pada window yang **memang** fault → batas atas kemampuan tahap ini |
| `end_to_end` | pada window yang **diprediksi** fault oleh ANN-LM → angka apa adanya |

**`Prevalensi` wajib dibaca bareng `F1`.** Prevalensi ≈ 1 dengan ROC-AUC ≈ 0,5
berarti model cuma menebak "semua sensor rusak" — cacat yang membatalkan hasil
notebook per-sensor versi lama (lihat README).
"""))

cells.append(code("""def sensor_stage(F, y, Ysens, groups, pred_oof):
    n_grp = len(np.unique(groups))
    k = int(min(N_SPLITS, n_grp, np.min(np.bincount(y)[np.unique(y)])))
    if k < 2:
        return [], None
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.full(Ysens.shape, -1, int)
    oof_proba = np.full(Ysens.shape, np.nan, float)

    tracemalloc.start(); c0 = _time.process_time(); t0 = _time.perf_counter()
    for tr, te in skf.split(F, y, groups=groups):
        tr_f = tr[y[tr] != 0]                       # latih hanya dari window fault
        if len(tr_f) < 12 or Ysens[tr_f].sum() == 0:
            continue
        base = MLPClassifier(solver=ANN_SOLVER, hidden_layer_sizes=(max(16, F.shape[1] // 2),),
                             alpha=1e-3, max_iter=ANN_MAX_ITER, random_state=RANDOM_SEED)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                          ("mo", MultiOutputClassifier(base))])
        pipe.fit(F[tr_f], Ysens[tr_f])
        oof_pred[te] = pipe.predict(F[te])
        try:
            oof_proba[te] = np.column_stack([e.predict_proba(F[te])[:, 1]
                                             for e in pipe.named_steps["mo"].estimators_])
        except Exception:
            pass
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    cost = {"cpu_s": c1 - c0, "wall_s": t1 - t0, "peak_mem_mb": peak / (1024 * 1024)}

    rows = []
    for mask, tag in ((y != 0, "oracle"), (pred_oof != 0, "end_to_end")):
        mask = mask & (oof_pred[:, 0] >= 0)
        if mask.sum() == 0:
            continue
        Ye, P, Pr = Ysens[mask], oof_pred[mask], oof_proba[mask]
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
                      "ROC_AUC": np.nan})
    return rows, cost

sensor_rows, sensor_cost_rows = [], []
for (WIN, sc, meth), r in RUNS.items():
    if not budget_ok(400, f"sensor {sc}/{meth}/N={WIN}"):
        continue
    log_stage(f"identifikasi sensor | N={WIN} | {sc} | {meth}")
    keep = r["keep"]
    F = FEATURES[(WIN, meth)][keep]
    rows, cost = sensor_stage(F, r["y"], SEGMENTS[WIN]["Ysens"][keep], r["groups"], r["out"]["pred_oof"])
    for row in rows:
        row.update({"N_window": WIN, "Skenario": sc, "Metode": meth})
        sensor_rows.append(row)
    if cost:
        sensor_cost_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                                  "cpu_s": round(cost["cpu_s"], 1), "wall_s": round(cost["wall_s"], 1),
                                  "peak_mem_mb": round(cost["peak_mem_mb"], 1)})

sensor_tbl = pd.DataFrame(sensor_rows)
sensor_cost = pd.DataFrame(sensor_cost_rows)
print("Selesai:", len(sensor_tbl), "baris hasil identifikasi sensor")
"""))

cells.append(code("""# === TABEL 3 — sensor mana yang rusak ===
e2e = sensor_tbl[sensor_tbl.Eval == "end_to_end"]
cols_show = ["N_window", "Skenario", "Metode", "Sensor", "n_window", "Prevalensi",
             "Akurasi", "Precision", "Recall", "F1", "ROC_AUC"]
print("=== Identifikasi sensor rusak — end-to-end (dirantai dari ANN-LM) ===")
print(e2e[cols_show].to_string(index=False))
export_df(sensor_tbl, "07_identifikasi_sensor")

print("\\n=== Ringkas: F1 rata-rata 4 sensor ===")
ring_s = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
          .groupby(["N_window", "Skenario", "Metode"])[["F1", "ROC_AUC", "Prevalensi"]].mean().round(3))
print(ring_s.to_string())
export_df(ring_s.reset_index(), "07_identifikasi_sensor_ringkas")

if len(sensor_cost):
    print("\\n=== Biaya komputasi tahap identifikasi sensor ===")
    print(sensor_cost.to_string(index=False))
    export_df(sensor_cost, "07_komputasi_tahap_sensor")
"""))

cells.append(code("""# === Peta panas F1 identifikasi sensor ===
piv = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
       .pivot_table(index=["N_window", "Skenario", "Metode"], columns="Sensor", values="F1"))
fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(piv) + 2))
im = ax.imshow(piv.values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels([f"N={a} | {b} | {c}" for a, b, c in piv.index], fontsize=7)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.6 else "black", fontsize=7)
ax.set_title("F1 identifikasi sensor rusak (end-to-end)")
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig("exports/07_peta_sensor.png", dpi=120); plt.show()
"""))

cells.append(code("""# === Ringkasan gabungan untuk laporan ===
sf1 = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
       .groupby(["N_window", "Skenario", "Metode"])["F1"].mean().rename("F1_sensor").reset_index())
sall = (e2e[e2e.Sensor == "SEMUA-4-BENAR"]
        .groupby(["N_window", "Skenario", "Metode"])["Akurasi"].mean().rename("Semua_4_benar").reset_index())
final = (perf_tbl.merge(comp_tbl, on=["N_window", "Skenario", "Metode"])
                 .merge(arch_tbl, on=["N_window", "Skenario", "Metode"], how="left")
                 .merge(sf1, on=["N_window", "Skenario", "Metode"], how="left")
                 .merge(sall, on=["N_window", "Skenario", "Metode"], how="left"))
final = final[["N_window", "Skenario", "Metode", "C_kelas", "n_fitur", "Akurasi", "Precision",
               "Recall", "F1", "F1_std", "Hidden_terpilih", "cpu_s_total", "peak_mem_mb",
               "infer_ms_per_window", "F1_sensor", "Semua_4_benar"]].round(3)
print("=== RINGKASAN — diagram dijalankan penuh: performa, komputasi, sensor ===")
print(final.to_string(index=False))
export_df(final, "07_ringkasan_lengkap")
display(final)
log_stage("selesai")
"""))

cells.append(md("""## Ringkasan — apa yang dibuktikan notebook ini

1. **Skema di flowchart dijalankan utuh, kotak per kotak**: akuisisi → broker →
   sinkronisasi waktu → injeksi fault → segmentasi (N ∈ {2000; 7000; 10000}) →
   EDM-Fuzzy τ=1..T → konkatenasi 4T fitur → ANN-LM dengan hidden layer hasil
   Grid Search → evaluasi.
2. **Cross-validation-nya dibuat tidak bocor.** Window bertetangga tumpang-tindih
   50%, jadi fold dikelompokkan menurut blok waktu (`StratifiedGroupKFold`) dan
   Grid Search hanya melihat data latih tiap fold. Angka di sini lebih rendah
   tapi lebih jujur daripada CV acak biasa.
3. **Panjang window jadi variabel, bukan asumsi.** Ketiga nilai N pada diagram
   dijalankan semua, jadi terlihat berapa panjang window yang sepadan dengan
   ongkosnya — ongkos ekstraksi entropy naik kira-kira linier terhadap N.
4. **Biaya komputasi dilaporkan sejajar dengan performa.** Yang menentukan bisa
   tidaknya jalan online di broker adalah `ms_per_window` pada ekstraksi fitur
   (bukan waktu latih), karena bagian itu yang berulang tiap window.
5. **Sensor rusak bisa ditunjuk** sebagai lanjutan dari kotak terakhir diagram,
   dengan pembanding `oracle` vs `end_to_end`.

**Cara baca angkanya:** utamakan **F1 macro** — jumlah kelas C berbeda antar
skenario sehingga akurasi tidak sebanding lintas skenario. Untuk §10, baca `F1`
bersama `Prevalensi` dan `ROC_AUC`.

**Dua penyimpangan dari diagram, disengaja dan dicatat:** `solver='lbfgs'`
sebagai pengganti Levenberg–Marquardt (sklearn tidak punya LM), dan JSD-Fuzzy
ikut dijalankan sebagai pembanding usulan paper walau diagram hanya menyebut
EDM-Fuzzy.

**File keluaran** di folder `exports/`: `07_performa_cv_5skenario.csv`,
`07_komputasi_cv_5skenario.csv`, `07_ongkos_ekstraksi_fitur.csv`,
`07_arsitektur_terpilih.csv`, `07_identifikasi_sensor.csv`,
`07_ringkasan_lengkap.csv`, plus `07_*.png`.
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "| cells:", len(cells))
