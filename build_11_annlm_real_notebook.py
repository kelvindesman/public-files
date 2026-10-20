#!/usr/bin/env python3
"""Build 11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb.

Notebook ini = notebook 10 (JSD-Fuzzy vs EDM-Fuzzy, ANN-LM asli) DITAMBAH
Bagian 3: identifikasi sensor yang rusak memakai LM asli juga — single
`lm_fit` langsung pada target multi-hot 4 sensor (keluaran linear + ambang
0,5), dirantai dari prediksi out-of-fold Bagian 2 (end-to-end, jujur mencakup
error Tahap 1). Tidak ada `lbfgs`/`adam` di notebook ini: SEMUA classifier
(Levenberg–Marquardt betulan — implementasi sendiri dari notebook 09:
langkah Gauss-Newton terdamping (JtJ + lam*D) d = -Jt r + rasio gain +
Jacobian analitik + self-check), diparameterkan untuk grid search
(arsitektur, aktivasi tanh/relu, iterasi).

Notebook ini adalah realisasi "Rencana Paper Q3 Lanjutan: JSD-Fuzzy vs EDM-Fuzzy".
Basisnya notebook 07 (skema diagram pembimbing, data_sensor.csv, 5 skenario,
ANN-LM grid search, cross-validation berkelompok), tetapi fokusnya dipersempit
persis seperti dokumen rencana:

  * Hanya DUA metode: EDM-Fuzzy (baseline) vs JSD-Fuzzy (usulan).
    CMSE dan Fuzzy Entropy TIDAK ikut.
  * Perubahan tunggal yang diklaim: Euclidean Distance pada tahap similarity
    computation Multiscale Fuzzy Entropy diganti Jensen-Shannon Divergence.
    Semua tahap lain (coarse-graining, embedding, centroid shifting, fungsi
    keanggotaan fuzzy, rumus entropi) DIBUAT IDENTIK, supaya perbedaan hasil
    benar-benar berasal dari model similarity.
  * Struktur fitur dibuat sebanding: tiap sensor menghasilkan T nilai entropi
    (tau = 1..T; T = 15 pada run ini, `T_SCALES`) untuk KEDUA metode
    -> n x 15 per sensor, n x 60 setelah konkatenasi 4 sensor. (Versi JSD "rich"
    4 komponen/skala di notebook 01-07 sengaja tidak dipakai di sini karena
    tidak sebanding.)
  * RQ1..RQ6 dokumen rencana dijawab satu per satu:
      RQ1 - validitas feature matrix + tabel entropi tau = 1..T
      RQ2 - stabilitas: CV = std/mean per Skenario x Kelas x Skala + CV reduction
      RQ3 - separabilitas: boxplot mean entropy across scales + ukuran overlap IQR
      RQ4 - performa LM asli (grid search terpisah untuk tiap metode & skenario & window)
            + ROC-AUC OvR + confusion matrix kedua metode
      RQ5 - paired t-test F1 lima skenario, per panjang window + pooled deskriptif
      RQ6 - identifikasi sensor (LM asli multi-hot, end-to-end dari OOF RQ4)
  * Computational cost DILAPORKAN sebagai catatan, tetapi TIDAK diklaim sebagai
    keunggulan (sesuai keputusan scope).

Data: data_sensor.csv (281.721 baris, 2025-09-14 .. 2025-12-21, interval 30 detik).
"""
import json
import notebook_fault_figures as nff

REPO_RAW = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"
OUT = "/Users/kelvin/apps/public-files/11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

# ---------------------------------------------------------------- §0 posisi
cells.append(md(r"""<!-- HEADER-KLAIM -->
# 11 — EDM-Fuzzy vs JSD-Fuzzy lengkap dengan LM asli (klasifikasi + sensor-ID)

| | |
|---|---|
| **Posisi artikel** | Lanjutan paper pertama (ACS) yang memakai **EDM-Fuzzy + ANN-LM**. Di sini yang dikembangkan adalah **model similarity**-nya: Euclidean Distance diganti **Jensen-Shannon Divergence** → **JSD-Fuzzy**. |
| **Jenis artikel** | **Paper metode.** Smart irrigation & sensor kelembaban tanah tetap konteks aplikasi, tetapi kontribusi utamanya pengembangan similarity model di dalam Multiscale Fuzzy Entropy. |
| **Yang dibandingkan** | **Hanya dua**: EDM-Fuzzy (baseline) vs JSD-Fuzzy (usulan). CMSE dan Fuzzy Entropy **tidak** masuk paper ini. |
| **Classifier** | ANN-LM **asli dari paper** (Levenberg–Marquardt betulan: langkah Gauss-Newton terdamping + rasio gain + Jacobian analitik, implementasi sendiri dari notebook 09 — **bukan** `lbfgs`) — **dicari model terbaik terpisah untuk tiap metode, tiap skenario, dan tiap panjang window**, supaya JSD-Fuzzy tidak dipaksa memakai konfigurasi yang hanya optimal untuk EDM-Fuzzy. |
| **Skenario** | Lima skenario yang sama dengan paper EDM-Fuzzy/ACS (S1..S5) × **tiga panjang window** (N = 200/700/1000) = 15 sel perbandingan. |
| **Computational cost** | Diukur dan dilaporkan, **tetapi bukan klaim**. Artikel tidak mengklaim JSD-Fuzzy lebih efisien / siap real-time / siap edge. |

## Novelty statement

> The novelty of this study lies in the development of **JSD-Fuzzy**, a
> Jensen-Shannon Divergence-based similarity model for Multiscale Fuzzy Entropy.
> The proposed method modifies the similarity computation in EDM-Fuzzy by
> replacing Euclidean Distance with Jensen-Shannon Divergence to provide a
> distribution-aware representation for soil moisture sensor fault detection.

## Research questions dan di sel mana dijawab

| RQ | Pertanyaan | Alat ukur | Bagian |
|---|---|---|---|
| **RQ1** | Bagaimana JSD diintegrasikan sebagai model similarity dalam Multiscale Fuzzy Entropy? | tabel entropi τ = 1..T + validitas feature matrix | §6, §7, §7a, §7b |
| **RQ2** | Apakah fitur entropy JSD-Fuzzy lebih stabil dari EDM-Fuzzy? | CV = std/mean, mean CV, CV reduction | §8 |
| **RQ3** | Apakah distribusi fitur JSD-Fuzzy lebih terpisah antar kelas? | boxplot mean entropy + overlap IQR | §9 |
| **RQ4** | Apakah JSD-Fuzzy menaikkan performa fault detection? | akurasi, precision, recall, F1, ROC-AUC OvR, confusion matrix | §10 |
| **RQ5** | Apakah selisih F1 signifikan secara statistik? | paired t-test pada F1 lima skenario, per window + pooled | §11 |
| **RQ6** | Sensor mana yang rusak (identifikasi, LM asli juga)? | F1/AUC per sensor end-to-end + per skenario + per jumlah fault | §12 |

## Kejujuran metodologi yang dipegang notebook ini

1. **Satu perubahan saja.** EDM-Fuzzy dan JSD-Fuzzy memakai coarse-graining,
   embedding, centroid shifting, fungsi keanggotaan fuzzy, dan rumus entropi
   yang **identik**. Yang berbeda hanya cara jarak antar-vektor dihitung.
2. **Struktur fitur sebanding.** Dua-duanya menghasilkan T nilai entropi per
   sensor (τ = 1..T) → matriks n × T per sensor, n × 4T setelah konkatenasi
   4 sensor. Tidak ada metode yang diberi dimensi lebih banyak.
3. **Perbandingan adil.** Dataset, skenario, pembagian data (StratifiedGroupKFold
   dengan grup = blok waktu), ruang hyperparameter, kriteria pemilihan model
   (macro-F1 validasi), dan metrik evaluasi **sama persis** untuk kedua metode.
4. **Klaim mengikuti hasil.** Kalau JSD-Fuzzy tidak menang di suatu skenario,
   itu ditulis apa adanya. Tidak ada klaim "selalu lebih baik".
"""))

cells.append(md(r"""## §0 — Baca ini dulu (versi tanpa istilah)

**Masalahnya.** Sensor kelembaban tanah bisa rusak diam-diam: nilainya
menggeser pelan (*drift*), meloncat sesaat (*spike*), bergeser tetap
(*bias*), atau ngadat (*hardware malfunction*). Kalau tidak ketahuan, sistem
irigasi menyiram di waktu yang salah.

**Cara kerjanya.** Sinyal dipotong jadi jendela waktu. Tiap jendela diukur
"seberapa tidak beraturan" polanya — itu yang disebut **entropi**. Diukur pada
banyak tingkat kekasaran (τ = 1..T), jadi tiap sensor menyumbang T angka.
Empat sensor → 40 angka. Empat puluh angka itu diberikan ke jaringan saraf
tiruan (ANN) yang menebak: normal atau rusak, dan rusaknya jenis apa.

**Apa yang diubah paper ini.** Di dalam perhitungan entropi ada langkah
"seberapa mirip potongan A dengan potongan B". Paper pertama memakai **jarak
garis lurus** (Euclidean): selisih angka per angka lalu diakarkan.

Paper ini menggantinya dengan **Jensen-Shannon Divergence**: tiap potongan
diubah dulu menjadi **sebaran nilai** (seperti membuat histogram kecil), lalu
yang dibandingkan adalah **bentuk sebarannya**, bukan selisih angkanya.

**Analogi.** Dua kelas ujian punya rata-rata sama, jadi "jarak"-nya nol. Tetapi
kelas A nilainya menumpuk di tengah, kelas B terbelah jadi dua kubu. Jarak
garis lurus bilang "sama"; membandingkan sebaran bilang "beda jauh". Sinyal
sensor yang berderau dan berubah-ubah lebih mirip kasus kedua.

**Apa yang dicek.** Lima hal berurutan: (1) apakah metode barunya benar-benar
jadi dan angkanya sah, (2) apakah angkanya lebih stabil, (3) apakah antar-kelas
lebih terpisah, (4) apakah tebakan ANN jadi lebih benar, (5) apakah bedanya
bukan kebetulan.
"""))

# ---------------------------------------------------------------- runtime guard
cells.append(code(r"""# === Runtime guard — jalankan sel ini PALING AWAL ===
import os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

NOTEBOOK_START = time.time()
_NOTEBOOK_CPU0 = time.process_time()   # acuan CPU proses utama untuk cek silang akuntansi
KAGGLE_TIME_BUDGET_H = float(os.environ.get("KAGGLE_TIME_BUDGET_H", 11.5))

def elapsed_s(): return time.time() - NOTEBOOK_START
def time_left_sec(): return KAGGLE_TIME_BUDGET_H * 3600.0 - elapsed_s()

def budget_ok(need_s=0.0, label=""):
    left = time_left_sec()
    if left < need_s:
        print(f"[budget] SKIP {label}: sisa {left/60:.1f} menit, butuh ~{need_s/60:.1f} menit", flush=True)
        return False
    return True

def log_stage(label):
    _trace_end()
    print(f"[t+{elapsed_s()/60:6.1f} min] {label}", flush=True)
    _trace_begin(label)

# --- Akuntansi CPU penuh untuk publikasi: tiap fase dicatat wall + CPU + memori.
# Batasan jujur: `process_time` hanya menghitung proses utama; CPU pekerja
# joblib (proses terpisah) diukur di dalam tugasnya masing-masing dan
# dijumlahkan di sel ekstraksi fitur & ANN-LM. tracemalloc hanya melacak
# alokasi Python (bukan buffer numpy); RSS dicatat sebagai high-water global.
_CPU_ROWS = []
_OPEN_SECTION = None

def _trace_begin(label):
    global _OPEN_SECTION
    try:
        import tracemalloc as _tm
        if not _tm.is_tracing():
            _tm.start()
    except Exception:
        pass
    _OPEN_SECTION = {"label": label, "t0": time.perf_counter(), "c0": time.process_time()}

def _trace_end():
    global _OPEN_SECTION
    if _OPEN_SECTION is None:
        return None
    t1 = time.perf_counter(); c1 = time.process_time()
    peak = 0.0; rss = None
    try:
        import tracemalloc as _tm
        if _tm.is_tracing():
            _, peak = _tm.get_traced_memory()
            _tm.stop()
    except Exception:
        pass
    try:
        import resource as _rs
        rss = _rs.getrusage(_rs.RUSAGE_SELF).ru_maxrss / 1024.0  # KB -> MB di Linux
    except Exception:
        pass
    row = {"bagian": _OPEN_SECTION["label"],
           "wall_s": round(t1 - _OPEN_SECTION["t0"], 2),
           "cpu_parent_s": round(c1 - _OPEN_SECTION["c0"], 2),
           "cpu_workers_s": 0.0,
           "cpu_all_s": round(c1 - _OPEN_SECTION["c0"], 2),
           "py_peak_mb": round(peak / (1024 * 1024), 1) if peak else 0.0,
           "rss_highwater_mb": round(rss, 1) if rss is not None else None}
    _CPU_ROWS.append(row)
    _OPEN_SECTION = None
    return row

def _phase(label):
    # Batas fase tanpa cetakan [t+...] (untuk sel yang tidak punya log_stage).
    _trace_end()
    print(f"[cpu-trace] fase: {label}", flush=True)
    _trace_begin(label)

log_stage(f"runtime guard aktif | budget={KAGGLE_TIME_BUDGET_H} jam | BLAS threads=1")
"""))

cells.append(code(r"""# === Konfigurasi global ===
_phase("§0 konfigurasi global")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, logging
import time as _time, tracemalloc, platform
from pathlib import Path
from IPython.display import FileLink, display

warnings.filterwarnings("ignore")

METHOD_LIST = ["MSE", "CMSE", "EDM-Fuzzy", "JSD-Fuzzy"]      # MSE & CMSE ditambah sesuai request
RANDOM_SEED = 42
N_JOBS = -1
N_SPLITS = 5

# --- Laju sampling setelah preprocessing: rata-rata per 5 menit ---
RESAMPLE_RULE = os.environ.get("RESAMPLE_RULE", "5min")
SAMPLING_SECONDS = 300 if RESAMPLE_RULE else 30

# --- Panjang window ---
# Paper ini memakai SATU panjang window (bukan studi sensitivitas seperti notebook 07):
# N = 200 sampel pada laju 5 menit = 0,69 hari = setara N = 2000 pada laju 30 detik,
# yaitu konfigurasi terbaik pada notebook 07. Bisa ditimpa lewat env WINDOW_LENGTHS.
_DEFAULT_WIN = "200,700,1000" if RESAMPLE_RULE else "2000"
WINDOW_LENGTHS = [int(v) for v in os.environ.get("WINDOW_LENGTHS", _DEFAULT_WIN).split(",")]
WIN_MAIN = WINDOW_LENGTHS[0]

T_SCALES = int(os.environ.get("T_SCALES", 15))       # tau = 1..T -> 15 fitur per sensor per metode
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 100))

DRIFT_PER_SAMPLE = 0.02 * (SAMPLING_SECONDS / 30)

# --- Parameter entropi, DIPAKAI SAMA oleh kedua metode ---
m = 2                 # embedding dimension
r_ratio = 0.2         # toleransi fuzzy, relatif terhadap SD sinyal
n_ref = int(os.environ.get("N_REF", 128))   # cacah vektor acuan (biar tidak O(N^2))

# --- Parameter khusus pemetaan vektor -> distribusi pada JSD-Fuzzy ---
JSD_BINS = int(os.environ.get("JSD_BINS", 8))        # cacah bin sebaran
JSD_H_RATIO = float(os.environ.get("JSD_H_RATIO", 1.0))   # lebar kernel = h_ratio x lebar bin
JSD_RANGE_SD = float(os.environ.get("JSD_RANGE_SD", 3.0)) # rentang bin = +-3 SD
JSD_R_RATIO = float(os.environ.get("JSD_R_RATIO", 0.2))   # toleransi fuzzy pada ruang JSD

SENSORS = ["S1", "S2", "S3", "S4"]
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]
N_REPEAT_SUBSET = int(os.environ.get("N_REPEAT_SUBSET", 8))
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
    # Ukur wall time, CPU time, dan memori puncak satu blok kerja.
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
print("Window N :", WINDOW_LENGTHS, "| T skala :", T_SCALES,
      "-> fitur per sensor =", T_SCALES, "| setelah konkat 4 sensor =", 4 * T_SCALES)
print("Metode   :", METHOD_LIST, "| K-Fold :", N_SPLITS, "| MAX_PER_CLASS :", MAX_PER_CLASS)
print("Parameter bersama : m =", m, "| r_ratio =", r_ratio, "| n_ref =", n_ref)
print("Parameter JSD     : bins =", JSD_BINS, "| h_ratio =", JSD_H_RATIO,
      "| range = +-", JSD_RANGE_SD, "SD | r_jsd_ratio =", JSD_R_RATIO)
"""))

# ---------------------------------------------------------------- §1 data
cells.append(md(r"""# §1 — Akuisisi data multisource

Empat sensor kelembaban tanah, dibaca dari `data_sensor.csv` (281.721 baris,
14 Sep – 21 Des 2025, interval 30 detik). Kalau file ada di direktori kerja atau
`/kaggle/input/...`, file itu dipakai; kalau tidak, diunduh dari GitHub
(**Kaggle: Settings → Internet → On**).
"""))

cells.append(code(r"""import requests, glob
_phase("§1 unduh data")
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

# ---------------------------------------------------------------- §2-3 broker + sync
cells.append(md(r"""# §2 — Broker & sinkronisasi waktu

Broker berperan sebagai **pengumpul**: empat aliran sensor disatukan jadi satu
tabel, **identitas tiap sensor dipertahankan** sebagai kolom terpisah (tidak
dilebur/dirata-rata jadi satu sinyal).

Lalu keempatnya dipaksa ke satu sumbu waktu seragam:

$$S_1 = S_1(t_1), \dots, S_1(t_N) \qquad \dots \qquad S_4 = S_4(t_1), \dots, S_4(t_N)$$

**Preprocessing.** Akuisisi 30 detik dirata-ratakan jadi satu sampel per 5 menit
(bukan mengambil tiap ke-10, supaya derau ikut teredam). Stempel waktu ganda
dibuang, bolong ditambal interpolasi waktu. Kemiringan drift dikalikan 10
(`DRIFT_PER_SAMPLE`) supaya drift per satuan **waktu** sama seperti versi 30 detik.

Pencilan dan pembacaan macet **dilaporkan, tidak dibuang** — yang diteliti justru
fault; membuang pembacaan aneh sebelum pemodelan sama saja menghapus barang
bukti sebelum penyidikan.
"""))

cells.append(code(r"""df_broker = df_raw[cols].copy()
_phase("§2 broker sinkronisasi")
df_broker.index = pd.to_datetime(df_broker.index, utc=True)
df_broker = df_broker.sort_index()

n_dup = int(df_broker.index.duplicated().sum())
df_broker = df_broker[~df_broker.index.duplicated(keep="first")]
df_pre = df_broker.copy()

if RESAMPLE_RULE:
    n_before = len(df_broker)
    df_broker = df_broker.resample(RESAMPLE_RULE).mean()
    print(f"Preprocessing laju sampling: rata-rata per {RESAMPLE_RULE} -> "
          f"{n_before} baris menjadi {len(df_broker)} baris")

grid = pd.date_range(df_broker.index[0], df_broker.index[-1], freq=f"{SAMPLING_SECONDS}s")
df_sync = df_broker.reindex(grid)
n_gap = int(df_sync.isna().any(axis=1).sum())
df_sync = df_sync.interpolate(method="time", limit_direction="both")
df_sync = df_sync.fillna(df_sync.median(numeric_only=True))
if df_sync.isna().any().any():
    raise ValueError("Error-nya jelas: masih ada NaN setelah sinkronisasi.")

X = df_sync.to_numpy(dtype=float)
print(f"Stempel waktu ganda dibuang: {n_dup} | slot bolong ditambal: {n_gap}")
print(f"Setelah sinkronisasi: {X.shape} pada grid {SAMPLING_SECONDS} detik "
      f"({(len(X) * SAMPLING_SECONDS)/86400:.1f} hari)")

# --- CV (koefisien variasi) data mentah per sensor, sebagai konteks RQ2 ---
def cv_pct(a):
    a = np.asarray(a, dtype=float); mu = a.mean()
    return float(a.std(ddof=1) / mu * 100.0) if mu != 0 else np.nan

cv_sensor = pd.DataFrame([{
    "Sensor": SENSORS[i], "Kolom": cols[i],
    "mean": round(float(df_sync[cols[i]].mean()), 4),
    "std": round(float(df_sync[cols[i]].std(ddof=1)), 4),
    "CV_pct": round(cv_pct(df_sync[cols[i]]), 2),
} for i in range(len(cols))])
print("\n=== CV data sensor setelah preprocessing (konteks, bukan CV entropi) ===")
print(cv_sensor.to_string(index=False))
export_df(cv_sensor, "11_cv_data_sensor")

fig, ax = plt.subplots(figsize=(13, 3.6))
seg = df_sync.iloc[:4000]
for c in cols:
    ax.plot(seg.index, seg[c], lw=0.8, label=c)
ax.set_title("Output broker setelah sinkronisasi waktu — 4 kanal (potongan awal, belum ada fault)")
ax.set_ylabel("kelembaban"); ax.legend(ncol=4, fontsize=9)
plt.tight_layout(); plt.savefig("exports/11_broker_tersinkron.png", dpi=120); plt.show()
"""))

cells.append(md(nff.qc_steps_md("§2b", "3")))

# ---------------------------------------------------------------- §4 fault injection
cells.append(md(r"""# §3 — Data preparation & fault injection

Simulator fault identik notebook `01`/`03`/`07` supaya angkanya sebanding lintas
notebook: **drift**, **spike**, **bias**, **hardware malfunction**, plus semua
kombinasinya (16 kondisi fault + normal).

Injeksi bersifat **sensor-selective**: fault hanya masuk ke subset sensor acak
(1..4 sensor), sisanya tetap bersih.
"""))

cells.append(code(r"""_phase("§3 injeksi fault")
def simulate_drift_fault(x, intensity=DRIFT_PER_SAMPLE, seed=None):
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

_D = {"intensity": DRIFT_PER_SAMPLE}
_S = {"intensity": 0.08, "p": 0.015}
_B = {"bias": 0.08}
_H = {"stuck_prob": 0.08, "loss_prob": 0.05}

SCENARIOS = {
    "faulty": [(simulate_choose_one, {"options": [(simulate_drift_fault, _D), (simulate_spike_fault, _S),
                                                   (simulate_bias_fault, _B), (simulate_hardware_fault, _H)]})],
    "drift": [(simulate_drift_fault, _D)],
    "spike": [(simulate_spike_fault, _S)],
    "bias": [(simulate_bias_fault, _B)],
    "hardware": [(simulate_hardware_fault, _H)],
    "bias+malfunc": [(simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "spike+malfunc": [(simulate_spike_fault, _S), (simulate_hardware_fault, _H)],
    "spike+bias": [(simulate_spike_fault, _S), (simulate_bias_fault, _B)],
    "drift+malfunc": [(simulate_drift_fault, _D), (simulate_hardware_fault, _H)],
    "drift+bias": [(simulate_drift_fault, _D), (simulate_bias_fault, _B)],
    "drift+spike": [(simulate_drift_fault, _D), (simulate_spike_fault, _S)],
    "spike+bias+malfunc": [(simulate_spike_fault, _S), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "drift+bias+malfunc": [(simulate_drift_fault, _D), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "spike+drift+malfunc": [(simulate_spike_fault, _S), (simulate_drift_fault, _D), (simulate_hardware_fault, _H)],
    "drift+spike+bias": [(simulate_drift_fault, _D), (simulate_spike_fault, _S), (simulate_bias_fault, _B)],
    "spike+bias+malfunc+drift": [(simulate_spike_fault, _S), (simulate_bias_fault, _B),
                                  (simulate_hardware_fault, _H), (simulate_drift_fault, _D)],
}
condition_names = ["normal"] + list(SCENARIOS.keys())
print("Kondisi:", len(SCENARIOS), "fault + normal =", len(condition_names))
"""))

cells.append(md(nff.figures_md("§3b", "4")))
cells.append(code(nff.figures_code("11_", seed_base=11000)))

# ---------------------------------------------------------------- §5 segmentation
cells.append(md(r"""# §4 — Segmentasi & pelabelan

Window **N = 200 sampel** pada laju 5 menit = **0,69 hari**, setara N = 2000 pada
laju akuisisi 30 detik (konfigurasi terbaik pada notebook 07). Stride = N/2,
jadi window bertetangga tumpang-tindih 50% — karena itu cross-validation
**harus dikelompokkan menurut blok waktu** (§10), kalau tidak potongan sinyal
yang sama muncul di data latih dan uji sekaligus.

Window yang porsi sampel ter-fault-nya ≤ 1% dibuang (melabelinya "fault"
menyesatkan — isinya praktis normal). Tidak pernah ada window setengah jadi:
`sliding_window_view` hanya membentuk window yang genap N sampel.
"""))

cells.append(code(r"""from numpy.lib.stride_tricks import sliding_window_view
_phase("§4 segmentasi window")

def make_windows(Xa, win, stride):
    Xn = np.asarray(Xa, dtype=np.float32); N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    # GOTCHA: sliding_window_view(...,axis=0) -> (nwin, kanal, win). Harus di-transpose
    # ke (nwin, win, kanal), kalau tidak entropi dihitung atas potongan panjang-4.
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
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]
    return (Wm.mean(axis=2) > thr)

SEG_AUDIT = {}

def segment_and_label(X, win):
    stride = max(1, win // 2)
    rng = np.random.default_rng(RANDOM_SEED)
    Ws, ys, Ys, st = [], [], [], []
    n_dibentuk = n_dibuang_thr = 0

    W0, s0 = make_windows(X, win, stride)
    Ws.append(W0); ys.append(np.zeros(len(W0), int))
    Ys.append(np.zeros((len(W0), 4), bool)); st.append(s0)
    n_dibentuk += len(W0)

    for k, (name, faults) in enumerate(SCENARIOS.items(), start=1):
        for rep in range(N_REPEAT_SUBSET):
            subset_size = SENSOR_SUBSET_SIZES[rep % len(SENSOR_SUBSET_SIZES)]
            subset = rng.choice(4, size=subset_size, replace=False)
            Y, M = inject_faults_multisensor(X, faults, subset, seed=int(rng.integers(1e9)))
            sens = window_labels_per_sensor(M, win, stride, thr=FAULT_RATIO_THR)
            Wk, sk = make_windows(Y, win, stride)
            keep = sens.any(axis=1)
            n_dibentuk += len(Wk); n_dibuang_thr += int((~keep).sum())
            Ws.append(Wk[keep]); ys.append(np.full(int(keep.sum()), k, int))
            Ys.append(sens[keep]); st.append(sk[keep])

    SEG_AUDIT[win] = {"N_window": win, "stride": stride,
                      "window_per_kondisi": int(len(W0)),
                      "window_dibentuk": int(n_dibentuk),
                      "dibuang_fault<=1%": int(n_dibuang_thr),
                      "window_setengah_jadi": 0}
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
    groups = starts // WIN          # grup CV = blok waktu
    SEGMENTS[WIN] = dict(W=W, y=y, Ysens=Ysens, starts=starts, groups=groups)
    print(f"N={WIN:5d} | window={W.shape} | kondisi terisi={len(np.unique(y)):2d} "
          f"| blok waktu={len(np.unique(groups)):3d} | durasi window={WIN*SAMPLING_SECONDS/86400:.2f} hari")

audit = pd.DataFrame([SEG_AUDIT[w] for w in WINDOW_LENGTHS])
print("\n=== Akuntansi window ===")
print(audit.to_string(index=False))
export_df(audit, "11_akuntansi_window")
"""))

# ---------------------------------------------------------------- §6 similarity models (RQ1 core)
cells.append(md(r"""# §5 — Model similarity: EDM-Fuzzy vs JSD-Fuzzy  ⟵ **inti RQ1**

Multiscale Fuzzy Entropy dijalankan dengan tahapan yang **sama persis** untuk
kedua metode:

1. **Coarse-graining** komposit: pada skala τ, deret dirata-ratakan tiap τ titik,
   diulang untuk semua offset $k = 0..τ-1$ lalu hasil entropinya dirata-ratakan
   (mengurangi ragam pada τ besar).
2. **Embedding**: $V_i = [y_i, y_{i+1}, \dots, y_{i+m-1}]$, dengan $m = 2$ dan $m+1 = 3$.
3. **Centroid shifting**: tiap vektor dikurangi rata-ratanya sendiri,
   $\tilde{V}_i = V_i - \bar{V}_i$ — menghapus level lokal, menyisakan pola.
4. **Similarity computation** ← **satu-satunya bagian yang berbeda**.
5. **Fungsi keanggotaan fuzzy**: $\mu_{ij} = \dfrac{1}{1 + d_{ij}/r}$.
6. **Entropi**: $E^{(\tau)} = \ln \dfrac{\phi^{m}}{\phi^{m+1}}$, dengan
   $\phi^{m}$ = rata-rata derajat keanggotaan pada dimensi $m$.

## Perbandingan tahap similarity (tabel untuk paper)

| Aspek | **EDM-Fuzzy** (baseline) | **JSD-Fuzzy** (usulan) |
|---|---|---|
| Objek yang dibandingkan | vektor embedding $\tilde{V}_i \in \mathbb{R}^m$ | sebaran nilai $P_i$ dari $\tilde{V}_i$, $P_i \in \Delta^{B-1}$ |
| Pemetaan | — (langsung) | soft-binning kernel Gauss ke $B = 8$ bin pada rentang $\pm 3\sigma$, lebar kernel $h$ = lebar bin |
| Ukuran beda | Euclidean $d_{ij} = \lVert \tilde{V}_i - \tilde{V}_j \rVert_2$ | akar Jensen-Shannon $d_{ij} = \sqrt{\mathrm{JSD}(P_i \Vert P_j)}$ |
| Sifat ukuran | metrik, tak terbatas, peka amplitudo | metrik (JS distance), **terbatas** $[0, \sqrt{\ln 2}]$, peka **bentuk sebaran** |
| Toleransi $r$ | $r = 0{,}2 \times \mathrm{SD}(x)$ | $r_{\mathrm{JSD}} = 0{,}2 \times \sqrt{\ln 2}$ (skala natural JS distance) |
| Fungsi keanggotaan | $1/(1 + d/r)$ | $1/(1 + d/r_{\mathrm{JSD}})$ — **sama bentuknya** |
| Dimensi keluaran | T nilai per sensor (τ = 1..T) | **T nilai per sensor** — sengaja dibuat sebanding |

## Kenapa JSD bisa memberi informasi berbeda

Jensen-Shannon Divergence antara dua sebaran $P$ dan $Q$:

$$\mathrm{JSD}(P \Vert Q) = \tfrac{1}{2} D_{KL}(P \Vert M) + \tfrac{1}{2} D_{KL}(Q \Vert M), \qquad M = \tfrac{1}{2}(P + Q)$$

Sifat yang relevan untuk sinyal sensor yang **non-stationary dan berderau**:

* **Terbatas.** JSD ≤ ln 2, jadi satu loncatan besar (spike) tidak mendominasi
  perhitungan seperti pada Euclidean yang tak terbatas. Similarity-nya lebih
  tahan pencilan.
* **Berbasis sebaran.** Yang dibandingkan bentuk sebaran nilai, bukan selisih
  titik-per-titik. Dua potongan dengan selisih titik besar tetapi sebaran mirip
  akan dianggap mirip — dan sebaliknya.
* **Jenuh mulus.** Setelah beda sebarannya melewati lebar kernel, penambahan
  jarak hampir tidak menambah divergensi. Ini yang membuat fiturnya **tidak**
  sekadar transformasi monoton dari Euclidean.

**Catatan jujur (penting untuk pembahasan paper).** Kalau vektor embedding
dipetakan ke sebaran lewat fungsi yang mulus dan nyaris linear (misalnya softmax
dengan temperatur lebar), JSD berperilaku ≈ kuadrat jarak Euclidean, sehingga
fiturnya nyaris kembar dengan EDM-Fuzzy. Uji pendahuluan pada data ini
menunjukkan korelasi per-fitur ≈ 0,98 untuk pemetaan seperti itu. Pemetaan
soft-binning yang dipakai di sini menurunkan korelasi tersebut ke ≈ 0,81 — jadi
JSD-Fuzzy memang membawa informasi yang berbeda, bukan sekadar penskalaan ulang.
Angka korelasinya dihitung ulang di §7a.
"""))

cells.append(code(r"""# =====================================================================
# TAHAP BERSAMA — dipakai identik oleh EDM-Fuzzy dan JSD-Fuzzy
# =====================================================================
_phase("§5 tahap bersama EDM/JSD")
EPS = 1e-12
LN2 = np.log(2.0)

def coarse_grain_cmse(x, s, k):
    # Coarse-graining komposit: offset k, faktor skala s.
    n = (len(x) - k) // s
    if n <= 0:
        return np.array([], dtype=float)
    return x[k:k + n * s].reshape(-1, s).mean(axis=1)

def embed_matrix_centered(y, m_):
    # Embedding + centroid shifting (tiap vektor dikurangi rata-ratanya sendiri).
    if len(y) < m_:
        return np.empty((0, m_), float)
    V = np.lib.stride_tricks.sliding_window_view(y, m_)
    return V - V.mean(axis=1, keepdims=True)

def _pick_ref(N, n_ref, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=n_ref, replace=False) if N > n_ref else np.arange(N)

# =====================================================================
# SIMILARITY 1 — EDM-Fuzzy: Euclidean Distance  (baseline paper pertama)
# =====================================================================
def phi_euclidean(V, r, n_ref=128, seed=0):
    N = V.shape[0]
    if N < 3:
        return np.nan
    ref = _pick_ref(N, n_ref, seed)
    A = V[ref]
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(V * V, axis=1, keepdims=True).T
    d = np.sqrt(np.maximum(a2 + b2 - 2 * (A @ V.T), 0.0))     # <-- Euclidean
    mu = 1.0 / (1.0 + d / (r + EPS))                          # fungsi keanggotaan fuzzy
    mu[np.arange(len(ref)), ref] = 0.0                        # buang pasangan diri sendiri
    return (mu.sum(axis=1) / (N - 1)).mean()

def edm_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0):
    out = []
    r = r_ratio * np.std(x, ddof=1)          # toleransi dari SD deret asli
    for s in scales:
        val_k = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m + 2):
                continue
            phi_m = phi_euclidean(embed_matrix_centered(y, m), r, n_ref, seed + 11 * s + k)
            phi_m1 = phi_euclidean(embed_matrix_centered(y, m + 1), r, n_ref, seed + 17 * s + k)
            if phi_m > 0 and phi_m1 > 0 and np.isfinite(phi_m) and np.isfinite(phi_m1):
                val_k.append(np.log(phi_m / phi_m1))
        out.append(np.mean(val_k) if val_k else np.nan)
    return np.array(out, dtype=float)

# =====================================================================
# SIMILARITY 2 — JSD-Fuzzy: Jensen-Shannon Divergence  (usulan paper ini)
#   Perbedaan TUNGGAL terhadap blok di atas: cara d_ij dihitung.
# =====================================================================
def vectors_to_distributions(V, centers, h):
    # Tiap vektor embedding (sudah centroid-shifted) diubah jadi sebaran nilai:
    # soft-binning dengan kernel Gauss -> histogram halus B bin, dinormalkan jadi
    # peluang. Dipakai kernel (bukan hard binning) supaya m titik tetap
    # menghasilkan sebaran yang mulus dan turunannya tidak lompat.
    D = (V[:, :, None] - centers[None, None, :]) / (h + EPS)
    K = np.exp(-0.5 * D * D)
    P = K.sum(axis=1)
    return P / (P.sum(axis=1, keepdims=True) + 1e-300)

def pairwise_jsd(A, B):
    # JSD(P||Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M = (P+Q)/2. Satuan nat, <= ln 2.
    a = A[:, None, :]; b = B[None, :, :]
    mm = 0.5 * (a + b)
    lm = np.log(mm + EPS)
    d = 0.5 * (np.sum(a * (np.log(a + EPS) - lm), axis=2)
               + np.sum(b * (np.log(b + EPS) - lm), axis=2))
    return np.maximum(d, 0.0)

def phi_jsd(V, centers, h, r_jsd, n_ref=128, seed=0):
    N = V.shape[0]
    if N < 3:
        return np.nan
    ref = _pick_ref(N, n_ref, seed)
    P = vectors_to_distributions(V, centers, h)
    d = np.sqrt(pairwise_jsd(P[ref], P))                      # <-- Jensen-Shannon distance
    mu = 1.0 / (1.0 + d / (r_jsd + EPS))                      # bentuk fungsi keanggotaan SAMA
    mu[np.arange(len(ref)), ref] = 0.0
    return (mu.sum(axis=1) / (N - 1)).mean()

def jsd_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0,
                         bins=JSD_BINS, h_ratio=JSD_H_RATIO,
                         range_sd=JSD_RANGE_SD, r_jsd_ratio=JSD_R_RATIO):
    out = []
    sd = np.std(x, ddof=1)
    centers = np.linspace(-range_sd * sd, range_sd * sd, bins)   # grid bin dari SD deret asli
    h = h_ratio * (centers[1] - centers[0]) if bins > 1 else sd
    r_jsd = r_jsd_ratio * np.sqrt(LN2)                           # toleransi pada skala JS distance
    for s in scales:
        val_k = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m + 2):
                continue
            phi_m = phi_jsd(embed_matrix_centered(y, m), centers, h, r_jsd, n_ref, seed + 11 * s + k)
            phi_m1 = phi_jsd(embed_matrix_centered(y, m + 1), centers, h, r_jsd, n_ref, seed + 17 * s + k)
            if phi_m > 0 and phi_m1 > 0 and np.isfinite(phi_m) and np.isfinite(phi_m1):
                val_k.append(np.log(phi_m / phi_m1))
        out.append(np.mean(val_k) if val_k else np.nan)
    return np.array(out, dtype=float)

def _cheb_pair_count(y, dim, r, n_ref=None, seed=0):
    N = len(y) - dim + 1
    if N < 2:
        return 0, 0
    V = np.lib.stride_tricks.sliding_window_view(y, dim)
    if n_ref is not None and n_ref < N:
        rng = np.random.default_rng(seed)
        ref = rng.choice(N, size=n_ref, replace=False)
        A = V[ref]
        d = np.max(np.abs(A[:, None, :] - V[None, :, :]), axis=2)
        matches = d < r
        matches[np.arange(len(ref)), ref] = False
        return int(matches.sum()), len(ref) * (N - 1)
    else:
        d = np.max(np.abs(V[:, None, :] - V[None, :, :]), axis=2)
        matches = d < r
        np.fill_diagonal(matches, False)
        return int(matches.sum()), N * (N - 1)

def sample_entropy_1d(y, m, r, n_ref=128, seed=0):
    y = np.ascontiguousarray(y)
    c_m, t_m = _cheb_pair_count(y, m, r, n_ref, seed)
    c_m1, t_m1 = _cheb_pair_count(y, m + 1, r, n_ref, seed + 1)
    if t_m == 0 or t_m1 == 0 or c_m == 0 or c_m1 == 0:
        return np.nan
    return -np.log((c_m1 / t_m1) / (c_m / t_m))

def mse_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0):
    out = []
    r = r_ratio * np.std(x, ddof=1)
    for s in scales:
        y = coarse_grain_cmse(x, s, 0)
        if len(y) < (m + 2):
            out.append(np.nan)
        else:
            out.append(sample_entropy_1d(y, m, r, n_ref, seed + 11*s))
    return np.array(out, dtype=float)

def cmse_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0):
    out = []
    r = r_ratio * np.std(x, ddof=1)
    for s in scales:
        ent_list = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m + 2):
                continue
            ent_list.append(sample_entropy_1d(y, m, r, n_ref, seed + 11*s + k))
        
        valid_ent = [e for e in ent_list if not np.isnan(e)]
        if len(valid_ent) == 0:
            out.append(np.nan)
        else:
            out.append(np.mean(valid_ent))
    return np.array(out, dtype=float)

scales = np.arange(1, T_SCALES + 1)
print("Skala tau:", scales.tolist())
print("EDM-Fuzzy  -> ", len(scales), "fitur per sensor (Euclidean)")
print("JSD-Fuzzy  -> ", len(scales), "fitur per sensor (Jensen-Shannon) — STRUKTUR SAMA")
"""))

cells.append(code(r"""# === Demo satu window: perlihatkan bahwa yang berbeda memang cuma d_ij ===
_phase("§6 demo satu window")
_xdemo = X[:WIN_MAIN, 0].astype(float)
_y = coarse_grain_cmse(_xdemo, 1, 0)
_V = embed_matrix_centered(_y, m)
_sd = np.std(_xdemo, ddof=1)
_ref = _pick_ref(_V.shape[0], 8, 0)

_a2 = np.sum(_V[_ref] ** 2, axis=1, keepdims=True); _b2 = np.sum(_V * _V, axis=1, keepdims=True).T
_deuc = np.sqrt(np.maximum(_a2 + _b2 - 2 * (_V[_ref] @ _V.T), 0.0))
_centers = np.linspace(-JSD_RANGE_SD * _sd, JSD_RANGE_SD * _sd, JSD_BINS)
_h = JSD_H_RATIO * (_centers[1] - _centers[0])
_P = vectors_to_distributions(_V, _centers, _h)
_djsd = np.sqrt(pairwise_jsd(_P[_ref], _P))

print("Contoh 8 vektor acuan x", _V.shape[0], "vektor pembanding, skala tau=1, sensor S1")
print(f"  Euclidean      : min={_deuc.min():.4f} median={np.median(_deuc):.4f} max={_deuc.max():.4f} (tak terbatas)")
print(f"  JS distance    : min={_djsd.min():.4f} median={np.median(_djsd):.4f} max={_djsd.max():.4f} "
      f"(batas atas sqrt(ln2)={np.sqrt(LN2):.4f})")
print(f"  korelasi Spearman antar-jarak: "
      f"{pd.Series(_deuc.ravel()).corr(pd.Series(_djsd.ravel()), method='spearman'):.4f}")
print("\nBaca: urutannya berkorelasi (dua-duanya ukuran ketidakmiripan), tetapi JS")
print("distance jenuh di batas atas -> pasangan yang sangat berbeda tidak lagi")
print("dibedakan besarannya, sedangkan Euclidean terus membesar. Itu sumber")
print("perbedaan perilaku fiturnya.")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].scatter(_deuc.ravel(), _djsd.ravel(), s=4, alpha=0.25)
ax[0].set_xlabel("Euclidean $d_{ij}$"); ax[0].set_ylabel("JS distance $d_{ij}$")
ax[0].set_title("Pemetaan jarak: Euclidean vs Jensen-Shannon")
for i in range(3):
    ax[1].plot(_centers, _P[_ref[i]], marker="o", lw=1.2, label=f"vektor #{_ref[i]}")
ax[1].set_title("Vektor embedding sebagai sebaran (soft-binning)")
ax[1].set_xlabel("nilai (centroid-shifted)"); ax[1].set_ylabel("peluang"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("exports/11_similarity_euclidean_vs_jsd.png", dpi=120); plt.show()
"""))

# ---------------------------------------------------------------- §7 features
cells.append(md(r"""# §6 — Ekstraksi & konkatenasi fitur multisensor

Tiap sensor menghasilkan vektor entropi multiskala

$$E_i = [E_i^{(1)}, E_i^{(2)}, \dots, E_i^{(10)}], \quad i = 1..4$$

lalu keempatnya digabung jadi satu vektor fitur per window:

$$F = [E_1 \mid E_2 \mid E_3 \mid E_4] \in \mathbb{R}^{40}$$

Berlaku **sama** untuk EDM-Fuzzy dan JSD-Fuzzy. Ongkos ekstraksi diukur di sini
(dilaporkan sebagai catatan, bukan klaim keunggulan).
"""))

cells.append(code(r"""from joblib import Parallel, delayed

def sanitize(F):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()

def entropy_1d(x, method, sd):
    key = method.strip().lower()
    if key == "edm-fuzzy":
        return edm_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    if key == "jsd-fuzzy":
        return jsd_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    if key == "mse":
        return mse_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    if key == "cmse":
        return cmse_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    raise ValueError(f"Metode tidak dikenal: {method}")

def concat_multisensor_features(W, method, seed=0, n_jobs=-1):
    nwin, win, ns = W.shape
    def one_window(i):
        # CPU pekerja diukur di sini: Parallel jalan di proses terpisah yang
        # TIDAK tercakup process_time proses utama — tanpa ini CPU undercount.
        c0 = _time.process_time()
        out = np.concatenate([entropy_1d(W[i, :, s], method, sd=seed + 1000 * i + 19 * s)
                              for s in range(ns)])
        return out, _time.process_time() - c0
    res = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one_window)(i) for i in range(nwin))
    feats = np.vstack([r[0] for r in res])
    cpu_workers = float(sum(r[1] for r in res))
    return feats, cpu_workers

FEATURES = {}          # (WIN, metode) -> (nwin, 4T)
RAW_FEATURES = {}      # (WIN, metode) -> versi sebelum sanitize, untuk audit RQ1
feat_cost_rows = []

for WIN in WINDOW_LENGTHS:
    seg = SEGMENTS[WIN]
    for meth in METHOD_LIST:
        need = 0.05 * len(seg["W"]) * (WIN / 1000.0)
        if not budget_ok(need, f"fitur N={WIN}/{meth}"):
            continue
        log_stage(f"ekstraksi fitur | N={WIN} | {meth} | {len(seg['W'])} window")
        (F, cpu_workers), mtr = run_with_metrics(f"Fitur {meth} N={WIN}",
                                   lambda w=seg["W"], mm=meth: concat_multisensor_features(w, mm, seed=7, n_jobs=N_JOBS))
        RAW_FEATURES[(WIN, meth)] = F
        FEATURES[(WIN, meth)] = sanitize(F)
        feat_cost_rows.append({"N_window": WIN, "Metode": meth, "n_window": len(F),
                               "n_fitur": F.shape[1],
                               "wall_s": round(mtr["wall_s"], 2), "cpu_s": round(mtr["cpu_s"], 2),
                               "cpu_workers_s": round(cpu_workers, 2),
                               "cpu_all_s": round(mtr["cpu_s"] + cpu_workers, 2),
                               "peak_mem_mb": round(mtr["peak_mem_mb"], 1),
                               "ms_per_window": round(1000 * mtr["wall_s"] / max(1, len(F)), 2)})
        print(f"  N={WIN} {meth:10s} -> fitur {F.shape}  ({feat_cost_rows[-1]['ms_per_window']} ms/window)")

feat_cost = pd.DataFrame(feat_cost_rows)
print("\n=== Ongkos ekstraksi fitur (catatan, BUKAN klaim paper) ===")
print(feat_cost.to_string(index=False))
export_df(feat_cost, "11_ongkos_ekstraksi_fitur")
log_stage("ekstraksi fitur selesai")
"""))

# ---------------------------------------------------------------- RQ1 validation
cells.append(md(r"""# §7 — RQ1: apakah JSD-Fuzzy sah sebagai pengganti similarity?

RQ1 **tidak** menguji performa klasifikasi. RQ1 memastikan JSD-Fuzzy berhasil
dibangun dan menghasilkan feature matrix yang **valid, lengkap, dan sebanding**
dengan EDM-Fuzzy.

Indikator keberhasilan (dari dokumen rencana):

* JSD berhasil menggantikan Euclidean Distance pada tahap similarity.
* Ada nilai entropi untuk seluruh scale factor τ = 1..T.
* Struktur feature vector JSD-Fuzzy sama dengan EDM-Fuzzy.
* Tidak ada NaN, infinity, atau nilai tidak valid.
* Kalau EDM-Fuzzy menghasilkan matriks n × 10 per sensor, JSD-Fuzzy juga.
"""))

cells.append(code(r"""# === TABEL RQ1 — validasi feature matrix ===
_phase("§7 RQ1 validasi feature matrix")
rq1_rows = []
for WIN in WINDOW_LENGTHS:
    for meth in METHOD_LIST:
        if (WIN, meth) not in RAW_FEATURES:
            continue
        Fr = RAW_FEATURES[(WIN, meth)]
        n, p = Fr.shape
        n_nan = int(np.isnan(Fr).sum()); n_inf = int(np.isinf(Fr).sum())
        # kolom konstan = fitur mati (tidak membawa informasi)
        n_const = int(np.sum(np.nanstd(Fr, axis=0) < 1e-12))
        rq1_rows.append({
            "Method": meth, "N_window": WIN,
            "Number of samples": n,
            "Scale features per sensor": T_SCALES,
            "Sensors": 4,
            "Total features": p,
            "NaN values": n_nan,
            "Infinite values": n_inf,
            "Constant features": n_const,
            "Min": round(float(np.nanmin(Fr)), 5),
            "Max": round(float(np.nanmax(Fr)), 5),
            "Status": "Valid" if (n_inf == 0 and n_const == 0 and p == 4 * T_SCALES) else "PERIKSA",
        })
rq1 = pd.DataFrame(rq1_rows)
print("=== TABEL RQ1 — validasi feature matrix (sebelum penambalan apa pun) ===")
print(rq1.to_string(index=False))
export_df(rq1, "11_rq1_validasi_feature_matrix")
display(rq1)

print("\nCatatan: kolom 'NaN values' dihitung pada matriks MENTAH. NaN hanya muncul")
print("kalau pada skala tertentu deret hasil coarse-graining < m+2 titik; nilainya")
print("ditambal median kolom sebelum masuk classifier (kolom Status tetap menilai")
print("kelengkapan struktur). Kalau angkanya 0, tidak ada penambalan sama sekali.")

# --- Bukti bahwa struktur per-sensor benar-benar n x 10 ---
for meth in METHOD_LIST:
    if (WIN_MAIN, meth) not in FEATURES:
        continue
    F = FEATURES[(WIN_MAIN, meth)]
    blocks = {SENSORS[s]: F[:, s * T_SCALES:(s + 1) * T_SCALES] for s in range(4)}
    shapes = {k: v.shape for k, v in blocks.items()}
    print(f"{meth:10s} -> blok per sensor: {shapes}")
"""))

cells.append(code(r"""# === Seberapa beda fitur JSD-Fuzzy dari EDM-Fuzzy? (bukti bukan sekadar penskalaan) ===
_phase("§7a korelasi EDM vs JSD")
if (WIN_MAIN, "EDM-Fuzzy") in FEATURES and (WIN_MAIN, "JSD-Fuzzy") in FEATURES:
    corr_parts = []
    for WIN in WINDOW_LENGTHS:
        if (WIN, "EDM-Fuzzy") not in FEATURES or (WIN, "JSD-Fuzzy") not in FEATURES:
            continue
        FE = FEATURES[(WIN, "EDM-Fuzzy")]
        FJ = FEATURES[(WIN, "JSD-Fuzzy")]
        per_feat = []
        for j in range(FE.shape[1]):
            a, b = FE[:, j], FJ[:, j]
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                continue
            per_feat.append({"N_window": WIN, "fitur": j, "sensor": SENSORS[j // T_SCALES], "tau": j % T_SCALES + 1,
                             "pearson": float(np.corrcoef(a, b)[0, 1]),
                             "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman"))})
        part = pd.DataFrame(per_feat)
        print(f"=== Korelasi per-fitur EDM-Fuzzy vs JSD-Fuzzy (N={WIN}) ===")
        print(part.groupby("tau")[["pearson", "spearman"]].mean().round(4).to_string())
        corr_parts.append(part)
    corr_tbl = pd.DataFrame(pd.concat(corr_parts, ignore_index=True)) if corr_parts else pd.DataFrame()
    print(f"\nRata-rata |pearson| lintas {len(corr_tbl)} fitur : {corr_tbl['pearson'].abs().mean():.4f}")
    print(f"Rata-rata |spearman| lintas {len(corr_tbl)} fitur: {corr_tbl['spearman'].abs().mean():.4f}")
    export_df(corr_tbl.round(4), "11_rq1_korelasi_edm_vs_jsd")
    print("\nBaca: korelasi jauh di bawah 1 -> JSD-Fuzzy membawa informasi yang berbeda,")
    print("bukan hasil penskalaan ulang EDM-Fuzzy. Kalau korelasinya ~0,99 di semua")
    print("skala, klaim 'model similarity baru' jadi lemah dan harus ditulis apa adanya.")
else:
    print("Kedua metode belum tersedia — lewati.")
"""))

# ---------------------------------------------------------------- entropy table
cells.append(md(r"""# §7b — Tabel nilai entropi τ = 1..T (RQ1 & RQ2)

Nilai entropi dirata-ratakan lintas 4 sensor untuk tiap kondisi (normal + 16
kombinasi fault), dipisah per metode. Ini tabel "Nilai entropy s = 1 sampai
s = 10" pada dokumen rencana.
"""))

cells.append(code(r"""# === TABEL entropi per kondisi x skala x metode ===
_phase("§7b tabel entropi")
def scale_matrix(F):
    # (n, 4T) -> (n, T): rata-rata lintas 4 sensor untuk tiap skala tau.
    return np.stack([F[:, [s * T_SCALES + t for s in range(4)]].mean(axis=1)
                     for t in range(T_SCALES)], axis=1)

SCALE_FEAT = {}    # (WIN, metode) -> (n, 10)
for k, F in FEATURES.items():
    SCALE_FEAT[k] = scale_matrix(F)

ent_rows = []
for WIN in WINDOW_LENGTHS:
    seg = SEGMENTS[WIN]
    ycond = seg["y"]
    for meth in METHOD_LIST:
        if (WIN, meth) not in SCALE_FEAT:
            continue
        S = SCALE_FEAT[(WIN, meth)]
        for ci, cname in enumerate(condition_names):
            sel = np.where(ycond == ci)[0]
            if len(sel) == 0:
                continue
            row = {"N_window": WIN, "Method": meth, "Condition": cname, "n": len(sel)}
            row.update({f"s{t+1}": round(float(np.nanmean(S[sel, t])), 5) for t in range(T_SCALES)})
            row["mean_all_scales"] = round(float(np.nanmean(S[sel])), 5)
            ent_rows.append(row)
ent_tbl = pd.DataFrame(ent_rows)
print("=== Nilai entropi rata-rata (rata-rata 4 sensor), tau = 1..T ===")
print(ent_tbl.to_string(index=False))
export_df(ent_tbl, "11_entropi_per_kondisi_skala")
display(ent_tbl)

# --- Kurva entropi multiskala: normal vs beberapa fault (per panjang window) ---
show = ["normal", "drift", "spike", "bias", "hardware", "spike+bias+malfunc+drift"]
for WIN in WINDOW_LENGTHS:
    fig, axes = plt.subplots(1, len(METHOD_LIST), figsize=(6 * len(METHOD_LIST), 4), squeeze=False)
    for a, meth in enumerate(METHOD_LIST):
        ax = axes[0][a]
        sub = ent_tbl[(ent_tbl["Method"] == meth) & (ent_tbl["N_window"] == WIN)]
        for cname in show:
            r = sub[sub["Condition"] == cname]
            if len(r) == 0:
                continue
            ax.plot(range(1, T_SCALES + 1), [r[f"s{t+1}"].values[0] for t in range(T_SCALES)],
                    marker="o", lw=1.4, label=cname)
        ax.set_title(f"Entropi multiskala — {meth} (N={WIN})")
        ax.set_xlabel(r"scale factor $\tau$"); ax.set_ylabel("entropi"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"exports/11_kurva_entropi_multiskala_N{WIN}.png", dpi=120); plt.show()
"""))

# ---------------------------------------------------------------- RQ2 CV
cells.append(md(r"""# §8 — RQ2: stabilitas fitur entropi (Coefficient of Variation)

$$\mathrm{CV} = \frac{\sigma}{\mu}, \qquad
\text{CV Reduction (\%)} = \frac{\mathrm{CV}_{\text{EDM}} - \mathrm{CV}_{\text{JSD}}}{\mathrm{CV}_{\text{EDM}}} \times 100$$

CV dihitung **antar-window di dalam satu kelas**, untuk tiap skenario, kelas,
metode, dan skala τ = 1..T — persis format tabel dokumen rencana
(Scenario | Class | Method | s1..sT | Mean CV).

Karena entropi bisa bernilai kecil, penyebut memakai $|\mu|$ dan baris dengan
$|\mu|$ di bawah ambang ditandai supaya CV yang meledak tidak dibaca sebagai
"tidak stabil".

**Catatan interpretasi (dipegang di pembahasan):** JSD-Fuzzy tidak harus punya
CV lebih rendah di semua skala dan semua skenario. Klaimnya mengikuti pola
hasil — misalnya "lebih stabil pada sebagian besar skala" atau "stabilitasnya
sebanding".
"""))

cells.append(code(r"""# === Definisi lima skenario (identik paper EDM-Fuzzy/ACS) ===
_phase("§7 definisi skenario")
ALL_FAULT = [c for c in condition_names if c != "normal"]
LADDER = {
    "S1_Fault_Free_vs_Faulty": [("fault free", ["normal"]), ("faulty", ALL_FAULT)],
    "S2_Single_Faults": [("fault free", ["normal"]), ("drift", ["drift"]), ("spike", ["spike"]),
                         ("bias", ["bias"]), ("hardware", ["hardware"])],
    "S3_Two_Fault_Combinations": [("fault free", ["normal"]), ("bias+HW", ["bias+malfunc"]),
                                  ("drift+bias", ["drift+bias"]), ("drift+HW", ["drift+malfunc"]),
                                  ("spike+bias", ["spike+bias"]), ("drift+spike", ["drift+spike"]),
                                  ("spike+HW", ["spike+malfunc"])],
    "S4_Three_Fault_Combinations": [("fault free", ["normal"]), ("drift+bias+HW", ["drift+bias+malfunc"]),
                                    ("drift+spike+bias", ["drift+spike+bias"]),
                                    ("spike+bias+HW", ["spike+bias+malfunc"]),
                                    ("spike+drift+HW", ["spike+drift+malfunc"])],
    "S5_Four_Fault_Combinations": [("fault free", ["normal"]),
                                   ("drift+spike+bias+HW", ["spike+bias+malfunc+drift"])],
}
SCEN_SHORT = {k: k.split("_")[0] for k in LADDER}
cond_to_idx = {c: i for i, c in enumerate(condition_names)}

def build_scenario(WIN, classes):
    # -> indeks window terpilih + label kelas skenario, sudah diseimbangkan.
    y_cond = SEGMENTS[WIN]["y"]
    idx_l, y_l = [], []
    for ci, (_, conds) in enumerate(classes):
        want = [cond_to_idx[c] for c in conds if c in cond_to_idx]
        sel = np.where(np.isin(y_cond, want))[0]
        idx_l.append(sel); y_l.append(np.full(len(sel), ci, int))
    keep = np.concatenate(idx_l); yy = np.concatenate(y_l)
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
        print(f"{sc:30s} C={len(cl)} n={len(keep):5d} {dict(zip(*np.unique(yy, return_counts=True)))}")
"""))

cells.append(code(r"""# === TABEL RQ2 — CV entropi per Skenario x Kelas x Metode x skala ===
_phase("§7 RQ2 CV entropi")
MU_MIN = 1e-3          # ambang |mean| supaya CV tidak meledak

cv_rows = []
for WIN in WINDOW_LENGTHS:
    for sc, cl in LADDER.items():
        keep, yy = build_scenario(WIN, cl)
        for meth in METHOD_LIST:
            if (WIN, meth) not in SCALE_FEAT:
                continue
            S = SCALE_FEAT[(WIN, meth)][keep]
            for ci, (cname, _) in enumerate(cl):
                sel = np.where(yy == ci)[0]
                if len(sel) < 3:
                    continue
                row = {"Window Lengths": WIN, "Scenario": SCEN_SHORT[sc], "Class": cname, "Method": meth, "n": len(sel)}
                cvs, flag = [], False
                for t in range(T_SCALES):
                    v = S[sel, t]
                    mu = float(np.nanmean(v)); sd = float(np.nanstd(v, ddof=1))
                    if abs(mu) < MU_MIN:
                        cv = np.nan; flag = True
                    else:
                        cv = sd / abs(mu)
                    cvs.append(cv); row[f"s{t+1}"] = round(cv, 4) if np.isfinite(cv) else np.nan
                row["Mean CV"] = round(float(np.nanmean(cvs)), 4)
                row["mean_kecil"] = flag
                cv_rows.append(row)

cv_tbl = pd.DataFrame(cv_rows)
print("=== TABEL RQ2 — CV entropi (Scenario | Class | Method | s1..sT | Mean CV) ===")
print(cv_tbl.to_string(index=False))
export_df(cv_tbl, "11_rq2_cv_entropi")
display(cv_tbl)
"""))

cells.append(code(r"""# === CV reduction: EDM-Fuzzy -> JSD-Fuzzy ===
_phase("§7 RQ2 CV reduction")
if len(cv_tbl) == 0:
    print("Tabel CV kosong — lewati.")
    cv_red = pd.DataFrame()
else:
    scols = [f"s{t+1}" for t in range(T_SCALES)]
    red_rows = []
    for WIN in WINDOW_LENGTHS:
        piv = cv_tbl[cv_tbl["Window Lengths"] == WIN].pivot_table(index=["Scenario", "Class"], columns="Method",
                                 values=scols + ["Mean CV"])
        for (scn, cls), _ in piv.iterrows():
            try:
                e = np.array([piv.loc[(scn, cls), (c, "EDM-Fuzzy")] for c in scols], float)
                j = np.array([piv.loc[(scn, cls), (c, "JSD-Fuzzy")] for c in scols], float)
            except KeyError:
                continue
            ok = np.isfinite(e) & np.isfinite(j) & (np.abs(e) > 1e-12)
            if ok.sum() == 0:
                continue
            red = (e[ok] - j[ok]) / e[ok] * 100.0
            red_rows.append({
                "N_window": WIN, "Scenario": scn, "Class": cls,
                "Mean CV EDM-Fuzzy": round(float(np.nanmean(e[ok])), 4),
                "Mean CV JSD-Fuzzy": round(float(np.nanmean(j[ok])), 4),
                "CV reduction (%)": round(float(np.mean(red)), 2),
                "Scales JSD lebih stabil": int(np.sum(j[ok] < e[ok])),
                "Scales dibandingkan": int(ok.sum()),
            })
    cv_red = pd.DataFrame(red_rows)
    print("=== CV reduction per Window x Skenario x Kelas (positif = JSD-Fuzzy lebih stabil) ===")
    print(cv_red.to_string(index=False))
    export_df(cv_red, "11_rq2_cv_reduction")
    display(cv_red)

    tot_win = int(cv_red["Scales JSD lebih stabil"].sum())
    tot_cmp = int(cv_red["Scales dibandingkan"].sum())
    print(f"\nRingkas RQ2: JSD-Fuzzy punya CV lebih rendah pada {tot_win} dari {tot_cmp} "
          f"pasangan (skenario x kelas x skala) = {100*tot_win/max(1,tot_cmp):.1f}%")
    print(f"Rata-rata CV reduction lintas kelas: {cv_red['CV reduction (%)'].mean():.2f}%")
    for WIN in WINDOW_LENGTHS:
        wsub = cv_red[cv_red["N_window"] == WIN]
        if len(wsub):
            ww = int(wsub["Scales JSD lebih stabil"].sum()); wc = int(wsub["Scales dibandingkan"].sum())
            print(f"  N={WIN}: {ww}/{wc} = {100*ww/max(1,wc):.1f}%, "
                  f"rata-rata reduction {wsub['CV reduction (%)'].mean():+.2f}%")
    print("Kalau angkanya negatif atau di bawah 50%, kalimat paper harus berbunyi")
    print("'stabilitas sebanding', bukan 'lebih stabil'.")
"""))

cells.append(code(r"""# === Grafik perbandingan CV (RQ2) ===
_phase("§7 grafik CV")
if len(cv_tbl) > 0:
    scols = [f"s{t+1}" for t in range(T_SCALES)]
    scen_list = [SCEN_SHORT[s] for s in LADDER]
    for WIN in WINDOW_LENGTHS:
        fig, axes = plt.subplots(1, len(scen_list), figsize=(4 * len(scen_list), 3.6), squeeze=False)
        for i, scn in enumerate(scen_list):
            ax = axes[0][i]
            sub = cv_tbl[(cv_tbl["Scenario"] == scn) & (cv_tbl["Window Lengths"] == WIN)]
            for meth, style in zip(METHOD_LIST, ["-^", "-d", "-o", "-s"]):
                s2 = sub[sub["Method"] == meth]
                if len(s2) == 0:
                    continue
                ax.plot(range(1, T_SCALES + 1), [s2[c].mean() for c in scols], style, lw=1.5,
                        ms=4, label=meth)
            ax.set_title(scn); ax.set_xlabel(r"$\tau$"); ax.grid(alpha=0.3)
            if i == 0:
                ax.set_ylabel("CV rata-rata antar kelas")
            ax.legend(fontsize=8)
        plt.suptitle(f"RQ2 — CV entropi per skala, N={WIN} (makin rendah makin stabil)")
        plt.tight_layout(); plt.savefig(f"exports/11_rq2_cv_per_skala_N{WIN}.png", dpi=120); plt.show()
"""))

# ---------------------------------------------------------------- RQ3 boxplot
cells.append(md(r"""# §9 — RQ3: separabilitas fitur (boxplot mean entropy)

Sesuai dokumen rencana, RQ3 **cukup boxplot** (tanpa PCA/t-SNE). Nilai entropi
τ = 1..T diringkas jadi satu angka per window:

$$\text{Mean Entropy} = \frac{E^{(1)} + E^{(2)} + \dots + E^{(10)}}{10}$$

lalu dikelompokkan per kelas fault. Yang dibaca: **beda median antar kelas**,
**lebar IQR**, dan **tumpang tindih IQR**.

Selain gambar, dihitung dua angka pendamping supaya klaimnya tidak
"kelihatannya lebih terpisah" saja:

* **Median spread** = (median terbesar − median terkecil) dibagi rata-rata IQR
  (makin besar makin terpisah).
* **IQR overlap** = rata-rata porsi tumpang tindih IQR pada semua pasangan kelas
  (makin kecil makin terpisah).
"""))

cells.append(code(r"""# === RQ3 — mean entropy across scales + boxplot per kelas ===
_phase("§7 RQ3 separabilitas")
def iqr_overlap(a, b):
    qa1, qa3 = np.percentile(a, [25, 75]); qb1, qb3 = np.percentile(b, [25, 75])
    inter = max(0.0, min(qa3, qb3) - max(qa1, qb1))
    union = max(qa3, qb3) - min(qa1, qb1)
    return float(inter / union) if union > 0 else 1.0

sep_rows = []
box_data = {}
for WIN in WINDOW_LENGTHS:
    for sc, cl in LADDER.items():
        keep, yy = build_scenario(WIN, cl)
        for meth in METHOD_LIST:
            if (WIN, meth) not in SCALE_FEAT:
                continue
            S = SCALE_FEAT[(WIN, meth)][keep]
            mean_ent = np.nanmean(S, axis=1)
            groups = [mean_ent[yy == ci] for ci in range(len(cl))]
            box_data[(WIN, sc, meth)] = (groups, [c[0] for c in cl])
            meds = np.array([np.median(g) for g in groups if len(g)])
            iqrs = np.array([np.subtract(*np.percentile(g, [75, 25])) for g in groups if len(g)])
            ovl = [iqr_overlap(groups[i], groups[j])
                   for i in range(len(groups)) for j in range(i + 1, len(groups))
                   if len(groups[i]) and len(groups[j])]
            sep_rows.append({
                "N_window": WIN, "Scenario": SCEN_SHORT[sc], "Method": meth, "C_kelas": len(cl),
                "Median min": round(float(meds.min()), 5), "Median max": round(float(meds.max()), 5),
                "Median spread / IQR": round(float((meds.max() - meds.min()) / (iqrs.mean() + 1e-12)), 3),
                "IQR rata-rata": round(float(iqrs.mean()), 5),
                "IQR overlap rata-rata": round(float(np.mean(ovl)) if ovl else np.nan, 3),
            })

sep_tbl = pd.DataFrame(sep_rows)
# interpretasi otomatis: mana yang lebih terpisah per window x skenario
interp = []
for WIN in WINDOW_LENGTHS:
    for scn in [SCEN_SHORT[s] for s in LADDER]:
        sub = sep_tbl[(sep_tbl["N_window"] == WIN) & (sep_tbl["Scenario"] == scn)].set_index("Method")
        if set(["EDM-Fuzzy", "JSD-Fuzzy"]).issubset(sub.index):
            d_sep = sub.loc["JSD-Fuzzy", "Median spread / IQR"] - sub.loc["EDM-Fuzzy", "Median spread / IQR"]
            d_ovl = sub.loc["JSD-Fuzzy", "IQR overlap rata-rata"] - sub.loc["EDM-Fuzzy", "IQR overlap rata-rata"]
            verdict = ("JSD-Fuzzy lebih terpisah" if (d_sep > 0 and d_ovl <= 0) else
                       "EDM-Fuzzy lebih terpisah" if (d_sep < 0 and d_ovl >= 0) else "campuran/sebanding")
            interp.append({"N_window": WIN, "Scenario": scn, "d_median_spread": round(float(d_sep), 3),
                           "d_IQR_overlap": round(float(d_ovl), 3), "Interpretation": verdict})
sep_interp = pd.DataFrame(interp)

print("=== TABEL RQ3 — separabilitas mean entropy ===")
print(sep_tbl.to_string(index=False))
export_df(sep_tbl, "11_rq3_separabilitas")
print("\n=== Interpretasi per skenario (JSD - EDM) ===")
print(sep_interp.to_string(index=False))
export_df(sep_interp, "11_rq3_interpretasi")
display(sep_tbl)
"""))

cells.append(code(r"""# === Boxplot mean entropy per kelas — EDM-Fuzzy vs JSD-Fuzzy ===
_phase("§7 boxplot")
n_sc = len(LADDER)
for WIN in WINDOW_LENGTHS:
    fig, axes = plt.subplots(len(METHOD_LIST), n_sc, figsize=(3.6 * n_sc, 3.4 * len(METHOD_LIST)),
                             squeeze=False)
    for r, meth in enumerate(METHOD_LIST):
        for c, sc in enumerate(LADDER):
            ax = axes[r][c]
            if (WIN, sc, meth) not in box_data:
                ax.axis("off"); continue
            groups, names = box_data[(WIN, sc, meth)]
            ax.boxplot(groups, labels=names, showfliers=True, widths=0.6)
            ax.set_title(f"{SCEN_SHORT[sc]} — {meth}", fontsize=10)
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.grid(alpha=0.3, axis="y")
            if c == 0:
                ax.set_ylabel("mean entropy (τ=1..T)")
    plt.suptitle(f"RQ3 — sebaran mean entropy per kelas fault, N={WIN} (makin terpisah makin baik)")
    plt.tight_layout(); plt.savefig(f"exports/11_rq3_boxplot_mean_entropy_N{WIN}.png", dpi=120); plt.show()
"""))

# ---------------------------------------------------------------- RQ4 ANN
cells.append(md(r"""# §10 — RQ4: performa fault detection dengan ANN-LM

**Keputusan metodologi (dari dokumen rencana):** karena pada paper EDM-Fuzzy tiap
skenario punya model terbaik sendiri, di sini **model ANN-LM terbaik dicari
terpisah untuk tiap metode × tiap skenario × tiap panjang window (N = 200/700/1000)**. JSD-Fuzzy tidak dipaksa memakai
konfigurasi yang hanya optimal untuk EDM-Fuzzy.

Yang dibuat **sama persis** untuk kedua metode: dataset, skenario, pembagian
data, ruang hyperparameter, kriteria pemilihan model (macro-F1 validasi), dan
metrik evaluasi.

| Bagian | Isi |
|---|---|
| Input layer | 4T neuron (4 sensor × T skala; T = 15 → 60 pada run ini) |
| Hidden layer | dipilih Grid Search: 8 arsitektur × 2 aktivasi = 16 kombinasi |
| Output layer | C neuron sesuai skenario (C = 2, 5, 7, 5, 2) |
| Solver | **LM asli dari paper** (implementasi sendiri, notebook 09): langkah $(J^TJ + \lambda\,\mathrm{diag}(J^TJ))\Delta = -J^Tr$ + rasio gain + Jacobian analitik (tanh & relu); aktivasi hidden tanh/relu ikut di-grid, keluaran linear, target one-hot + MSE |
| Validasi | **nested + grouped**: `StratifiedGroupKFold` (grup = blok waktu) di luar, Grid Search `cv=2` di dalam fold latih saja — dijalankan untuk **3 panjang window** (N = 200/700/1000), total 30 run (5 skenario × 2 metode × 3 window) |

Grup = blok waktu wajib karena stride = N/2 membuat window bertetangga
tumpang-tindih; tanpa itu potongan sinyal yang sama bisa muncul di latih **dan**
uji, dan akurasinya menggelembung.
"""))

cells.append(code(r"""# === §10a — LM asli ala paper (diangkat dari notebook 09, diparameterkan) ===
# Langkah Levenberg-Marquardt betulan: (J^T J + lam*diag(J^T J)) step = -J^T r
_phase("§10a inti LM")
# + rasio gain + Jacobian analitik (tanh maupun relu), target one-hot + MSE.
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, roc_auc_score)
from joblib import Parallel, delayed

def lm_init_layers(n_in, hidden, n_out, seed=0):
    rng = np.random.default_rng(seed)
    dims = [n_in, *hidden, n_out]
    layers = []
    for i in range(len(dims) - 1):
        lim = np.sqrt(6.0 / (dims[i] + dims[i + 1]))   # Xavier, dipakai untuk tanh & relu
        layers.append([rng.uniform(-lim, lim, (dims[i], dims[i + 1])), np.zeros(dims[i + 1])])
    return layers

def lm_pack(layers):
    return np.concatenate([W.ravel() for W, _ in layers] + [b for _, b in layers])

def lm_unpack(theta, template):
    out, i = [], 0
    for W, b in template:
        out.append([theta[i:i + W.size].reshape(W.shape), theta[i + W.size:i + W.size + b.size]])
        i += W.size + b.size
    return out

def lm_forward(layers, X, activation="tanh"):
    acts = [X]
    for li, (W, b) in enumerate(layers):
        z = acts[-1] @ W + b
        if li == len(layers) - 1:
            acts.append(z)                              # keluaran linear (target one-hot)
        elif activation == "relu":
            acts.append(np.maximum(z, 0.0))
        else:
            acts.append(np.tanh(z))
    return acts[-1], acts

def lm_residuals(theta, template, X, Y, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), X, activation)
    return (out - Y).ravel()

def lm_residual_jacobian(theta, template, X, Y, activation="tanh", mode="analytic"):
    layers = lm_unpack(theta, template)
    out, acts = lm_forward(layers, X, activation)
    m, n_out = out.shape
    if mode == "numeric":
        r0 = (out - Y).ravel()
        J = np.empty((r0.size, theta.size))
        eps = 1e-6
        for p in range(theta.size):
            tp = theta.copy(); tp[p] += eps
            op, _ = lm_forward(lm_unpack(tp, template), X, activation)
            J[:, p] = ((op - Y).ravel() - r0) / eps
        return J
    L = len(layers)
    delta = np.broadcast_to(np.eye(n_out), (m, n_out, n_out)).astype(float)
    blocks = [None] * L
    for l in range(L - 1, -1, -1):
        W, b = layers[l]
        a_prev = acts[l]
        n_in, n_out_l = W.shape
        dW = np.einsum("ip,icq->icpq", a_prev, delta, optimize=True)
        blocks[l] = (dW.reshape(m * n_out, n_in * n_out_l),
                     delta.reshape(m * n_out, n_out_l))
        if l > 0:
            back = np.einsum("icq,pq->icp", delta, W, optimize=True)
            h = acts[l]          # l>=1 -> acts[l] selalu keluaran hidden layer
            deriv = (h > 0).astype(float) if activation == "relu" else (1.0 - h ** 2)
            delta = back * deriv[:, None, :]
    J = np.empty((m * n_out, theta.size))
    col = 0
    for dW, db in blocks:
        J[:, col:col + dW.shape[1]] = dW; col += dW.shape[1]
        J[:, col:col + db.shape[1]] = db; col += db.shape[1]
    return J

LM_JAC_MODE = "analytic"

def lm_jac(theta, template, X, Y, activation="tanh"):
    return lm_residual_jacobian(theta, template, X, Y, activation, mode=LM_JAC_MODE)

def lm_fit(X, Y, n_in, hidden, n_out, activation="tanh", iters=40, seed=0,
           lam0=1e-3, max_train_s=60.0):
    template = lm_init_layers(n_in, hidden, n_out, seed)
    theta = lm_pack(template)
    lam = lam0
    hist = {"loss": [], "grad": [], "lam": [], "nfev": 0, "iters": 0, "status": "max_iter"}
    r = lm_residuals(theta, template, X, Y, activation); hist["nfev"] += 1
    E = 0.5 * float(r @ r)
    _t0 = _time.perf_counter()
    for k in range(iters):
        if _time.perf_counter() - _t0 > max_train_s:
            hist["status"] = "budget(MAX_TRAIN_S)"; break
        J = lm_jac(theta, template, X, Y, activation)
        g = J.T @ r
        A = J.T @ J
        gn = float(np.linalg.norm(g))
        hist["loss"].append(float(np.mean(r * r))); hist["grad"].append(gn); hist["lam"].append(float(lam))
        hist["iters"] = k + 1
        if gn < 1e-10:
            hist["status"] = "converged(grad)"; break
        d = np.diag(A).copy(); d = np.where(d < 1e-12, 1e-12, d)
        try:
            step = np.linalg.solve(A + lam * np.diag(d), -g)
        except np.linalg.LinAlgError:
            lam *= 10.0; continue
        if not np.all(np.isfinite(step)):
            lam *= 10.0; continue
        theta_new = theta + step
        r_new = lm_residuals(theta_new, template, X, Y, activation); hist["nfev"] += 1
        E_new = 0.5 * float(r_new @ r_new)
        denom = 0.5 * float(step @ (lam * d * step - g))
        rho = (E - E_new) / (denom if abs(denom) > 1e-300 else 1e-300)
        if E_new < E:
            if abs(E - E_new) <= 1e-12 * (1.0 + abs(E)):
                theta, r, E = theta_new, r_new, E_new
                hist["loss"].append(float(np.mean(r * r))); hist["lam"].append(float(lam))
                hist["status"] = "converged(tol)"; break
            theta, r, E = theta_new, r_new, E_new
            lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3) if rho > 0 else lam * 2.0
        else:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-12, 1e12))
    if len(hist["loss"]) == 0:
        hist["loss"].append(float(np.mean(r * r)))
    return {"theta": theta, "template": template, "hist": hist,
            "train_mse": float(np.mean(lm_residuals(theta, template, X, Y, activation) ** 2))}

def lm_predict_proba(theta, template, X, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), np.asarray(X, dtype=float), activation)
    e = out - out.max(axis=1, keepdims=True)
    P = np.exp(e); P /= P.sum(axis=1, keepdims=True)
    return P

class LMEstimator(ClassifierMixin, BaseEstimator):
    # Pembungkus sklearn supaya bisa masuk Pipeline + grid search manual.
    # Urutan mixin (ClassifierMixin dulu) wajib untuk sklearn >= 1.6.
    def __init__(self, hidden=(32,), activation="tanh", iters=40, seed=42,
                 lam0=1e-3, max_train_s=60.0):
        self.hidden = hidden
        self.activation = activation
        self.iters = iters
        self.seed = seed
        self.lam0 = lam0
        self.max_train_s = max_train_s
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(np.asarray(y))
        idx = np.searchsorted(self.classes_, np.asarray(y))
        Y = np.eye(len(self.classes_))[idx]
        res = lm_fit(X, Y, X.shape[1], list(self.hidden), len(self.classes_),
                     activation=self.activation, iters=self.iters, seed=self.seed,
                     lam0=self.lam0, max_train_s=self.max_train_s)
        self.theta_ = res["theta"]; self.template_ = res["template"]
        self.hist_ = res["hist"]; self.train_mse_ = res["train_mse"]
        return self
    def predict_proba(self, X):
        return lm_predict_proba(self.theta_, self.template_, X, self.activation)
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
"""))

cells.append(code(r"""# === §10a-check — self-check Jacobian analitik (tanh & relu) ===
_phase("§10a self-check Jacobian")
_chk_rng = np.random.default_rng(7)
_chk_template = lm_init_layers(60, [16, 8], 3, seed=7)
_chk_X = _chk_rng.normal(size=(16, 60))
_chk_Y = np.eye(3)[_chk_rng.integers(0, 3, 16)]
_chk_th = lm_pack(_chk_template)
for _act in ["tanh", "relu"]:
    _Ja = lm_residual_jacobian(_chk_th, _chk_template, _chk_X, _chk_Y, _act, mode="analytic")
    _Jn = lm_residual_jacobian(_chk_th, _chk_template, _chk_X, _chk_Y, _act, mode="numeric")
    _err = np.abs(_Ja - _Jn) / np.maximum(1e-8, np.abs(_Jn))
    print(f"  {_act:5s} max |rel err| = {_err.max():.3e}", flush=True)
    assert _err.max() < 1e-3, f"Jacobian {_act} tidak cocok!"
print("Jacobian analitik (tanh & relu) LOLOS self-check -> LM_JAC_MODE = analytic", flush=True)
del _chk_rng, _chk_template, _chk_X, _chk_Y, _chk_th, _Ja, _Jn, _err, _act
"""))

cells.append(code(r"""# === §10b — grid LM asli + nested grouped CV ===
_phase("§10b grid + CV")
# Arsitektur DIPERKECIL vs notebook 08 (maks (64,16), bukan (128,64,32)):
# langkah LM padat butuh J (m*C x P) + solve P^3, jadi P dibatasi ~5rb supaya
# 16 kombinasi x 2 inner fold x 5 outer fold x 30 run (3 window) muat < 12 jam.
# 8 arsitektur x 2 aktivasi = 16 kombinasi, SAMA untuk EDM-Fuzzy & JSD-Fuzzy.
HIDDEN_GRID = [(16,), (32,), (64,), (16, 8), (32, 8), (32, 16), (32, 32), (64, 16)]
ACT_GRID = ["tanh", "relu"]
LM_ITERS_SEARCH = 6      # iterasi pendek saat pencarian (LM konvergen cepat)
LM_ITERS_REFIT = 40      # iterasi penuh saat refit model terpilih
LM_INNER_CV = 2          # lipatan dalam pencarian (demi muat 3 window < 12 jam)
LM_SEARCH_BUDGET_S = 40.0
LM_REFIT_BUDGET_S = 150.0

def make_lm_pipeline(hidden, activation, iters, max_train_s):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lm", LMEstimator(hidden=tuple(hidden), activation=activation, iters=iters,
                           seed=RANDOM_SEED, max_train_s=max_train_s)),
    ])

def _inner_combo_score(job):
    hidden, activation, Ftr, ytr, gtr, iters, budget_s, seed = job
    # c0/c1 diukur di PROSES PEKERJA (fungsi ini jalan di worker joblib),
    # supaya CPU pencarian terhitung — process_time proses utama tidak mencakupnya.
    c0 = _time.process_time()
    try:
        skf = StratifiedGroupKFold(n_splits=LM_INNER_CV, shuffle=True, random_state=seed)
        fs, statuses = [], []
        for itr, ite in skf.split(Ftr, ytr, groups=gtr):
            if len(np.unique(ytr[itr])) < 2:
                continue
            pipe = make_lm_pipeline(hidden, activation, iters, budget_s)
            pipe.fit(Ftr[itr], ytr[itr])
            pred = pipe.predict(Ftr[ite])
            fs.append(f1_score(ytr[ite], pred, average="macro", zero_division=0))
            statuses.append(pipe.named_steps["lm"].hist_["status"])
        if not fs:
            return (tuple(hidden), activation, -1.0, ["no-inner-fold"], _time.process_time() - c0)
        return (tuple(hidden), activation, float(np.mean(fs)), statuses, _time.process_time() - c0)
    except Exception as e:
        return (tuple(hidden), activation, -1.0, [f"error: {type(e).__name__}"], _time.process_time() - c0)

def inner_search_best(Ftr, ytr, gtr, seed=RANDOM_SEED):
    combos = [(h, a) for h in HIDDEN_GRID for a in ACT_GRID]
    results = Parallel(n_jobs=N_JOBS)(
        delayed(_inner_combo_score)((h, a, Ftr, ytr, gtr, LM_ITERS_SEARCH,
                                     LM_SEARCH_BUDGET_S, seed))
        for h, a in combos)
    results.sort(key=lambda r: r[2], reverse=True)
    total_worker_cpu = float(sum(r[4] for r in results))
    best = results[0]
    return best[0], best[1], best[2], best[3], total_worker_cpu

def nested_grouped_cv(F, y, groups, n_splits=N_SPLITS):
    n_grp = len(np.unique(groups))
    k = int(min(n_splits, n_grp, np.min(np.bincount(y)[np.unique(y)])))
    if k < 2:
        return None
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    rows, chosen = [], []
    pred_oof = np.full(len(y), -1, dtype=int)
    proba_oof = None   # diisi (n, C) bila predict_proba tersedia (untuk ROC-AUC OvR)
    tracemalloc.start(); c0 = _time.process_time(); t0 = _time.perf_counter()
    fit_times, infer_times, n_infer = [], [], []
    search_budget_hits, refit_statuses = 0, []
    search_worker_cpu, refit_cpu = 0.0, 0.0

    for tr, te in skf.split(F, y, groups=groups):
        if len(np.unique(y[tr])) < 2:
            continue
        tf0 = _time.perf_counter()
        best_h, best_a, best_score, statuses, worker_cpu = inner_search_best(F[tr], y[tr], groups[tr])
        fit_times.append(_time.perf_counter() - tf0)
        search_worker_cpu += worker_cpu
        search_budget_hits += sum(1 for s in statuses
                                  if str(s).startswith("budget") or str(s).startswith("error"))
        pipe = make_lm_pipeline(best_h, best_a, LM_ITERS_REFIT, LM_REFIT_BUDGET_S)
        tf1 = _time.perf_counter(); rc0 = _time.process_time()
        pipe.fit(F[tr], y[tr])
        refit_cpu += _time.process_time() - rc0
        fit_times.append(_time.perf_counter() - tf1)
        refit_statuses.append(pipe.named_steps["lm"].hist_["status"])
        ti0 = _time.perf_counter(); pred = pipe.predict(F[te])
        infer_times.append(_time.perf_counter() - ti0); n_infer.append(len(te))
        pred_oof[te] = pred
        try:
            Pb = pipe.predict_proba(F[te])
            if proba_oof is None:
                proba_oof = np.full((len(y), Pb.shape[1]), np.nan)
            proba_oof[te] = Pb
        except Exception:
            pass
        p, r, f, _ = precision_recall_fscore_support(y[te], pred, average="macro", zero_division=0)
        rows.append({"acc": accuracy_score(y[te], pred), "prec": p, "rec": r, "f1": f})
        chosen.append((tuple(best_h), best_a))

    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    if not rows:
        return None
    sc = pd.DataFrame(rows)
    print(f"  search: {search_budget_hits} status budget/error | refit: {refit_statuses}", flush=True)
    return {"n_folds": len(rows), "pred_oof": pred_oof, "proba_oof": proba_oof, "chosen": chosen,
            "fold_f1": sc["f1"].tolist(),
            "mean": sc.mean().to_dict(), "std": sc.std(ddof=0).to_dict(),
            "cost": {"wall_s": t1 - t0, "cpu_s": c1 - c0, "peak_mem_mb": peak / (1024 * 1024),
                     "cpu_parent_s": c1 - c0,
                     "cpu_workers_s": search_worker_cpu,
                     # refit jalan di proses utama -> sudah tercakup cpu_parent;
                     # cpu_all = parent + pekerja (tanpa hitung ganda refit).
                     "cpu_all_s": (c1 - c0) + search_worker_cpu,
                     "refit_cpu_s": refit_cpu,
                     "fit_s_per_fold": float(np.mean(fit_times)),
                     "infer_ms_per_window": float(1000 * np.sum(infer_times) / max(1, np.sum(n_infer)))}}

print("ANN-LM: algoritma LM asli (paper, implementasi sendiri) | grid:",
      len(HIDDEN_GRID), "x", len(ACT_GRID), "=",
      len(HIDDEN_GRID) * len(ACT_GRID), "kombinasi | outer CV:", N_SPLITS, "fold grouped",
      "| search:", LM_ITERS_SEARCH, "iter | refit:", LM_ITERS_REFIT, "iter")
"""))

cells.append(code(r"""# === Jalankan: 3 panjang window x 5 skenario x 2 metode ===
perf_rows, comp_rows, arch_rows = [], [], []
RUNS = {}

for WIN in WINDOW_LENGTHS:
    seg = SEGMENTS[WIN]
    for sc, cl in LADDER.items():
        keep, yy = build_scenario(WIN, cl)
        grp = seg["groups"][keep]
        for meth in ["EDM-Fuzzy", "JSD-Fuzzy"]:
            if (WIN, meth) not in FEATURES:
                continue
            # Guard per run DILONGGARKAN vs notebook 08 (900 -> 2400 dtk): LM padat
            # memang lebih lambat dari lbfgs; estimasi total 30 run ~9 jam < budget 11,5 jam.
            if not budget_ok(2400, f"{sc}/{meth}/N={WIN}"):
                continue
            log_stage(f"ANN-LM | N={WIN} | {sc} | {meth}")
            F = FEATURES[(WIN, meth)][keep]
            out = nested_grouped_cv(F, yy, grp)
            if out is None:
                print(f"  {sc} {meth}: dilewati (grup/kelas terlalu sedikit)")
                continue
            RUNS[(WIN, sc, meth)] = dict(out=out, keep=keep, y=yy, groups=grp,
                                         class_names=[c[0] for c in cl])
            mn, sd, cost = out["mean"], out["std"], out["cost"]
            try:
                _Pb = out["proba_oof"]; _ok = ~np.isnan(_Pb).any(axis=1)
                _C = int(yy.max()) + 1
                if _ok.sum() > 0 and len(np.unique(yy[_ok])) == _C and _C >= 2:
                    if _C == 2:
                        _auc = float(roc_auc_score(yy[_ok], _Pb[_ok][:, 1]))
                    else:
                        _auc = float(roc_auc_score(np.eye(_C)[yy[_ok]], _Pb[_ok],
                                                   average="macro", multi_class="ovr"))
                else:
                    _auc = float("nan")
            except Exception:
                _auc = float("nan")
            perf_rows.append({"N_window": WIN, "Scenario": SCEN_SHORT[sc], "Method": meth,
                              "C_kelas": len(cl), "n_window": len(yy), "n_fitur": F.shape[1],
                              "n_fold": out["n_folds"],
                              "Accuracy": round(mn["acc"], 4), "Accuracy_std": round(sd["acc"], 4),
                              "Precision": round(mn["prec"], 4), "Precision_std": round(sd["prec"], 4),
                              "Recall": round(mn["rec"], 4), "Recall_std": round(sd["rec"], 4),
                              "F1": round(mn["f1"], 4), "F1_std": round(sd["f1"], 4),
                              "ROC_AUC_OvR": (round(_auc, 4) if np.isfinite(_auc) else np.nan)})
            comp_rows.append({"N_window": WIN, "Scenario": SCEN_SHORT[sc], "Method": meth,
                              "cpu_s_total": round(cost["cpu_s"], 1),
                              "wall_s_total": round(cost["wall_s"], 1),
                              "cpu_workers_s_total": round(cost["cpu_workers_s"], 1),
                              "cpu_all_s_total": round(cost["cpu_all_s"], 1),
                              "peak_mem_mb": round(cost["peak_mem_mb"], 1),
                              "fit_s_per_fold": round(cost["fit_s_per_fold"], 2),
                              "infer_ms_per_window": round(cost["infer_ms_per_window"], 3)})
            arch_rows.append({"N_window": WIN, "Scenario": SCEN_SHORT[sc], "Method": meth,
                              "Best hidden layers": "; ".join(sorted({str(h) for h, _ in out["chosen"]})),
                              "Best activation": "; ".join(sorted({str(a) for _, a in out["chosen"]}))})
            print(f"  {SCEN_SHORT[sc]} {meth:10s} F1={mn['f1']:.3f}+-{sd['f1']:.3f} "
                  f"acc={mn['acc']:.3f} | cpu={cost['cpu_s']:.0f}s (pekerja {cost['cpu_workers_s']:.0f}s)")

perf_tbl = pd.DataFrame(perf_rows)
comp_tbl = pd.DataFrame(comp_rows)
arch_tbl = pd.DataFrame(arch_rows)
log_stage("ANN-LM selesai")
"""))

cells.append(code(r"""# === TABEL RQ4 — performa + arsitektur terpilih (format tabel paper) ===
_phase("§10 RQ4 tabel performa")
if len(perf_tbl) == 0:
    print("Tidak ada hasil ANN — semua konfigurasi dilewati (cek budget waktu).")
else:
    tbl4 = perf_tbl.merge(arch_tbl, on=["N_window", "Scenario", "Method"], how="left")
    diff_all = []
    for WIN in WINDOW_LENGTHS:
        sub4 = tbl4[tbl4["N_window"] == WIN]
        if len(sub4) == 0:
            print(f"--- N={WIN}: tidak ada hasil (dilewati) ---")
            continue
        show = sub4[["Scenario", "Method", "Best hidden layers", "Best activation",
                     "Accuracy", "Precision", "Recall", "F1", "F1_std", "ROC_AUC_OvR"]]
        print(f"=== TABEL RQ4 — Best ANN-LM & classification performance (N={WIN}) ===")
        print(show.to_string(index=False))
        display(show)

        print(f"\n=== Selisih per skenario, N={WIN} (JSD-Fuzzy - EDM-Fuzzy) ===")
        piv = sub4.pivot_table(index="Scenario", columns="Method",
                               values=["Accuracy", "Precision", "Recall", "F1"])
        if set(["EDM-Fuzzy", "JSD-Fuzzy"]).issubset(sub4["Method"]):
            diff = pd.DataFrame({
                m: (piv[(m, "JSD-Fuzzy")] - piv[(m, "EDM-Fuzzy")]).round(4)
                for m in ["Accuracy", "Precision", "Recall", "F1"]
            })
            diff["JSD menang?"] = np.where(diff["F1"] > 0, "ya", np.where(diff["F1"] < 0, "tidak", "seri"))
            print(diff.to_string())
            n_win = int((diff["F1"] > 0).sum())
            print(f"\nJSD-Fuzzy unggul F1 pada {n_win} dari {len(diff)} skenario (N={WIN}). "
                  f"Rata-rata selisih F1 = {diff['F1'].mean():+.4f}")
            dtmp = diff.reset_index(); dtmp["N_window"] = WIN
            diff_all.append(dtmp)
    export_df(tbl4, "11_rq4_performa_ann")
    if diff_all:
        export_df(pd.concat(diff_all, ignore_index=True), "11_rq4_selisih")

    print("\n=== Biaya komputasi (catatan, BUKAN klaim paper) ===")
    print(comp_tbl.to_string(index=False))
    export_df(comp_tbl, "11_rq4_biaya_komputasi")
"""))

cells.append(code(r"""# === Grafik RQ4 — F1 per skenario, EDM vs JSD ===
_phase("§10 grafik RQ4")
if len(perf_tbl) > 0:
    scen_order = [SCEN_SHORT[s] for s in LADDER if SCEN_SHORT[s] in set(perf_tbl["Scenario"])]
    xs = np.arange(len(scen_order)); w = 0.36
    for WIN in WINDOW_LENGTHS:
        subw = perf_tbl[perf_tbl["N_window"] == WIN]
        if len(subw) == 0:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        for i, meth in enumerate(["EDM-Fuzzy", "JSD-Fuzzy"]):
            sub = subw[subw["Method"] == meth].set_index("Scenario")
            vals = [sub.loc[s, "F1"] if s in sub.index else np.nan for s in scen_order]
            errs = [sub.loc[s, "F1_std"] if s in sub.index else 0 for s in scen_order]
            ax[0].bar(xs + (i - 0.5) * w, vals, w, yerr=errs, capsize=3, label=meth)
            accs = [sub.loc[s, "Accuracy"] if s in sub.index else np.nan for s in scen_order]
            ax[1].bar(xs + (i - 0.5) * w, accs, w, label=meth)
        for a, t in zip(ax, ["Macro F1 (± std antar-fold)", "Accuracy"]):
            a.set_xticks(xs); a.set_xticklabels(scen_order); a.set_ylim(0, 1.02)
            a.set_title(t); a.grid(alpha=0.3, axis="y"); a.legend()
        plt.suptitle(f"RQ4 — performa fault detection N={WIN}: EDM-Fuzzy vs JSD-Fuzzy (ANN-LM)")
        plt.tight_layout(); plt.savefig(f"exports/11_rq4_performa_N{WIN}.png", dpi=120); plt.show()
"""))

cells.append(code(r"""# === Confusion matrix out-of-fold, tiap skenario x metode ===
_phase("§10 confusion matrix")
if len(RUNS) == 0:
    print("Tidak ada run tersimpan — lewati confusion matrix.")
else:
    cm_rows = []
    for WIN in WINDOW_LENGTHS:
        keys = [(w, s, m) for (w, s, m) in RUNS if w == WIN]
        keys.sort(key=lambda t: (list(LADDER).index(t[1]), ["EDM-Fuzzy", "JSD-Fuzzy"].index(t[2])))
        if not keys:
            continue
        n_sc = len(LADDER)
        fig, axes = plt.subplots(len(["EDM-Fuzzy", "JSD-Fuzzy"]), n_sc,
                                 figsize=(3.8 * n_sc, 3.6 * len(["EDM-Fuzzy", "JSD-Fuzzy"])), squeeze=False)
        for r, meth in enumerate(["EDM-Fuzzy", "JSD-Fuzzy"]):
            for c, sc in enumerate(LADDER):
                ax = axes[r][c]
                key = (WIN, sc, meth)
                if key not in RUNS:
                    ax.axis("off"); continue
                R = RUNS[key]; out = R["out"]
                mask = out["pred_oof"] >= 0
                if mask.sum() == 0:
                    ax.axis("off"); continue
                cm = confusion_matrix(R["y"][mask], out["pred_oof"][mask])
                ConfusionMatrixDisplay(cm, display_labels=R["class_names"]).plot(
                    ax=ax, colorbar=False, cmap="Blues", values_format="d")
                ax.set_title(f"{SCEN_SHORT[sc]} — {meth}", fontsize=10)
                ax.tick_params(axis="x", labelrotation=90, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
        plt.suptitle(f"RQ4 — confusion matrix out-of-fold (N={WIN})")
        plt.tight_layout(); plt.savefig(f"exports/11_rq4_confusion_matrix_N{WIN}.png", dpi=120); plt.show()

        # Ekspor confusion matrix sebagai CSV panjang, biar bisa ditabelkan di paper
        for (w, sc, meth) in keys:
            R = RUNS[(w, sc, meth)]; out = R["out"]
            mask = out["pred_oof"] >= 0
            if mask.sum() == 0:
                continue
            cm = confusion_matrix(R["y"][mask], out["pred_oof"][mask])
            for i, tn in enumerate(R["class_names"]):
                for j, pn in enumerate(R["class_names"]):
                    cm_rows.append({"N_window": w, "Scenario": SCEN_SHORT[sc], "Method": meth,
                                    "True": tn, "Pred": pn, "Count": int(cm[i, j])})
    if cm_rows:
        export_df(pd.DataFrame(cm_rows), "11_rq4_confusion_matrix")
"""))

# ---------------------------------------------------------------- RQ5 t-test
cells.append(md(r"""# §11 — RQ5: validasi statistik (paired t-test)

F1 EDM-Fuzzy dan JSD-Fuzzy disusun **berpasangan** per skenario (S1..S5),
**terpisah untuk tiap panjang window** (N = 200/700/1000), lalu diuji dengan
**paired t-test** dua sisi. Padanan Excel-nya:
`=T.TEST(B2:B6,C2:C6,2,1)`.

* p-value < 0,05 → perbedaan performa signifikan secara statistik.
* p-value ≥ 0,05 → **belum** signifikan; hasil tetap dibahas lewat rata-rata F1
  dan pola performa, tanpa mengklaim keunggulan statistik.

Catatan kejujuran: n = 5 pasang per window. Daya ujinya kecil, jadi p besar
**bukan** bukti "tidak ada beda". Ditambahkan Wilcoxon signed-rank dan Cohen's d
sebagai pendamping, plus uji tingkat-fold sebagai informasi tambahan (fold tidak
sepenuhnya independen, jadi itu bukan uji utama). Baris "pooled" (15 pasang =
5 skenario × 3 window) hanya deskriptif — window yang beda ketergantungannya
beda, jadi uji utama tetap per window.
"""))

cells.append(code(r"""# === TABEL RQ5 — paired t-test pada F1 lima skenario ===
_phase("§10 RQ5 paired t-test")
from scipy import stats

def _paired_test(a, b):
    # Uji berpasangan untuk satu kelompok (satu window, atau pooled).
    # Mengembalikan dict baris ringkasan, atau None bila tidak sah.
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b - a
    if len(d) < 2 or np.std(d) == 0:
        return None
    t_stat, p_val = stats.ttest_rel(b, a)
    try:
        w_stat, w_p = stats.wilcoxon(b, a)
    except Exception:
        w_stat, w_p = np.nan, np.nan
    cohen_d = float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))
    verdict = ("SIGNIFIKAN (p < 0,05) — selisih F1 konsisten lintas skenario"
               if p_val < 0.05 else
               "BELUM SIGNIFIKAN (p >= 0,05) — bahas lewat rata-rata F1 dan pola performa saja")
    return {"n_pairs": len(d), "mean_F1_EDM": round(float(a.mean()), 4),
            "mean_F1_JSD": round(float(b.mean()), 4), "mean_diff": round(float(d.mean()), 4),
            "t_stat": round(float(t_stat), 4), "p_value": round(float(p_val), 4),
            "wilcoxon_p": (round(float(w_p), 4) if np.isfinite(w_p) else np.nan),
            "cohen_d": round(cohen_d, 3), "verdict": verdict}

if len(perf_tbl) == 0 or not set(["EDM-Fuzzy", "JSD-Fuzzy"]).issubset(set(perf_tbl["Method"])):
    print("Butuh hasil kedua metode — lewati RQ5.")
    rq5 = pd.DataFrame(); rq5_sum = pd.DataFrame()
else:
    rq5_parts, rq5_sum_parts = [], []
    for WIN in WINDOW_LENGTHS:
        sub = perf_tbl[perf_tbl["N_window"] == WIN]
        piv = sub.pivot_table(index="Scenario", columns="Method", values="F1")
        piv = piv.dropna()
        if len(piv) == 0 or not set(["EDM-Fuzzy", "JSD-Fuzzy"]).issubset(set(piv.columns)):
            print(f"--- N={WIN}: pasangan tak lengkap — lewati RQ5 window ini ---")
            continue
        rq = pd.DataFrame({
            "N_window": WIN,
            "Scenario": piv.index,
            "EDM-Fuzzy F1": piv["EDM-Fuzzy"].round(4).values,
            "JSD-Fuzzy F1": piv["JSD-Fuzzy"].round(4).values,
        })
        rq["Difference"] = (rq["JSD-Fuzzy F1"] - rq["EDM-Fuzzy F1"]).round(4)
        print(f"=== TABEL RQ5 — F1 berpasangan (N={WIN}) ===")
        print(rq.to_string(index=False))
        rq5_parts.append(rq)

        row = _paired_test(piv["EDM-Fuzzy"].to_numpy(float), piv["JSD-Fuzzy"].to_numpy(float))
        if row is None:
            print(f"N={WIN}: selisih konstan / pasangan < 2 — t-test tidak sah.")
            continue
        print(f"\nN={WIN} | n pasangan: {row['n_pairs']} | "
              f"mean EDM={row['mean_F1_EDM']:.4f} vs JSD={row['mean_F1_JSD']:.4f} "
              f"(selisih {row['mean_diff']:+.4f})")
        print(f"N={WIN} | paired t: t={row['t_stat']:.4f}, p={row['p_value']:.4f} | "
              f"Wilcoxon p={row['wilcoxon_p']} | d={row['cohen_d']:.3f}")
        print(f"N={WIN} | Kesimpulan RQ5: {row['verdict']}")
        row["N_window"] = WIN
        rq5_sum_parts.append(row)

    # --- Pooled deskriptif: 15 pasang (5 skenario x 3 window), BUKAN uji utama ---
    if rq5_parts:
        pooled = pd.concat(rq5_parts, ignore_index=True)
        prow = _paired_test(pooled["EDM-Fuzzy F1"].to_numpy(float),
                            pooled["JSD-Fuzzy F1"].to_numpy(float))
        if prow is not None:
            print(f"\n[pooled 3 window] n={prow['n_pairs']} | "
                  f"mean EDM={prow['mean_F1_EDM']:.4f} vs JSD={prow['mean_F1_JSD']:.4f} "
                  f"(selisih {prow['mean_diff']:+.4f}) | t={prow['t_stat']:.4f}, "
                  f"p={prow['p_value']:.4f} | Wilcoxon p={prow['wilcoxon_p']} | "
                  f"d={prow['cohen_d']:.3f}")
            print("Catatan: pooled menggabung window yang ketergantungannya beda — "
                  "deskriptif saja; uji utama tetap per window.")
            prow["N_window"] = "pooled"
            rq5_sum_parts.append(prow)

    rq5 = (pd.concat(rq5_parts, ignore_index=True) if rq5_parts else pd.DataFrame())
    rq5_sum = (pd.DataFrame(rq5_sum_parts) if rq5_sum_parts else pd.DataFrame())
    if len(rq5):
        export_df(pd.concat([rq5, pd.DataFrame([{}])], ignore_index=True).fillna(""), "11_rq5_f1_berpasangan")
    if len(rq5_sum):
        export_df(rq5_sum, "11_rq5_ttest")
        display(rq5); display(rq5_sum)

    # --- Informasi tambahan: uji pada tingkat fold (bukan uji utama), per window ---
    for WIN in WINDOW_LENGTHS:
        fold_a, fold_b = [], []
        for sc in LADDER:
            ka = (WIN, sc, "EDM-Fuzzy"); kb = (WIN, sc, "JSD-Fuzzy")
            if ka in RUNS and kb in RUNS:
                fa = RUNS[ka]["out"]["fold_f1"]; fb = RUNS[kb]["out"]["fold_f1"]
                k = min(len(fa), len(fb))
                fold_a.extend(fa[:k]); fold_b.extend(fb[:k])
        if len(fold_a) >= 3 and np.std(np.array(fold_b) - np.array(fold_a)) > 0:
            t2, p2 = stats.ttest_rel(fold_b, fold_a)
            print(f"\n[tambahan] N={WIN} tingkat fold: n = {len(fold_a)} pasang, t = {t2:.4f}, p = {p2:.4f}")
    if len(rq5):
        print("Fold tidak sepenuhnya independen (data yang sama dipakai ulang antar-fold),")
        print("jadi angka ini hanya pendamping, bukan uji utama paper.")
"""))

# ------------------------------------------------------- §12 sensor-ID LM asli
cells.append(md(r"""# §12 — RQ6: sensor mana yang rusak (identifikasi, LM asli juga)

Dirantai **end-to-end** dari prediksi out-of-fold §10: hanya window yang
diprediksi fault (kelas ≠ 0) yang diteruskan — jadi error Tahap 1 ikut
terhitung, jujur apa adanya. Satu `lm_fit` langsung pada target multi-hot
4 sensor (keluaran linear + ambang 0,5; tanpa `MultiOutputClassifier`,
tanpa `lbfgs`/`adam`). Arsitektur = modus pilihan grid run tersebut
(tanpa search ulang). Metrik per sensor: prevalensi, akurasi, precision,
recall, F1, ROC-AUC (dari keluaran mentah), plus `SEMUA-4-BENAR`
(exact-match keempat sensor).
"""))

cells.append(code(r"""# === §12a — sensor-ID LM asli per (window x skenario x metode), end-to-end ===
_phase("§12 sensor-ID LM asli")
from collections import Counter
from sklearn.model_selection import train_test_split

SENSOR_NAMES = ["S1", "S2", "S3", "S4"]
SENSOR_RUNS, sens_rows = {}, []

def _modal_hidden(chosen):
    h = Counter([tuple(x) for x, _ in chosen]).most_common(1)[0][0]
    a = Counter([x for _, x in chosen]).most_common(1)[0][0]
    return list(h), a

for (WIN, sc, meth), R in RUNS.items():
    if not budget_ok(300, f"sensor-ID {sc}/{meth}/N={WIN}"):
        continue
    out, keep = R["out"], R["keep"]
    oof = out["pred_oof"]
    mask = oof != 0     # kelas 0 = "fault free" di semua skenario
    if int(mask.sum()) < 20:
        print(f"  {SCEN_SHORT[sc]} {meth:10s} N={WIN}: dilewati (lolos Tahap 1: {int(mask.sum())})")
        continue
    F = FEATURES[(WIN, meth)][keep][mask]
    Ye = SEGMENTS[WIN]["Ysens"][keep][mask].astype(float)
    tr, te = train_test_split(np.arange(len(F)), test_size=0.3, random_state=RANDOM_SEED)
    imp = SimpleImputer(strategy="median").fit(F[tr])
    scl = StandardScaler().fit(imp.transform(F[tr]))
    Xtr = scl.transform(imp.transform(F[tr])); Xte = scl.transform(imp.transform(F[te]))
    hid, act = _modal_hidden(out["chosen"])
    res = lm_fit(Xtr, Ye[tr], Xtr.shape[1], hid, 4, activation=act,
                 iters=LM_ITERS_REFIT, seed=RANDOM_SEED, max_train_s=150.0)
    Ote, _ = lm_forward(lm_unpack(res["theta"], res["template"]), Xte, act)
    pred = (Ote > 0.5).astype(int)
    row_base = {"N_window": WIN, "Scenario": SCEN_SHORT[sc], "Method": meth,
                "n_uji": len(te), "Hidden": str(tuple(hid)), "Activation": act,
                "Status": res["hist"]["status"]}
    for j, sname in enumerate(SENSOR_NAMES):
        yt, yp = Ye[te][:, j].astype(int), pred[:, j]
        p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        try:
            auc = (float(roc_auc_score(yt, Ote[:, j]))
                   if len(np.unique(yt)) == 2 else float("nan"))
        except Exception:
            auc = float("nan")
        sens_rows.append({**row_base, "Sensor": sname, "n_uji": len(te),
                          "Prevalensi": round(float(yt.mean()), 3),
                          "Akurasi": round(float(accuracy_score(yt, yp)), 3),
                          "Precision": round(float(p), 3), "Recall": round(float(r), 3),
                          "F1": round(float(f), 3),
                          "ROC_AUC": (round(auc, 3) if np.isfinite(auc) else np.nan)})
    sens_rows.append({**row_base, "Sensor": "SEMUA-4-BENAR", "n_uji": len(te),
                      "Prevalensi": np.nan, "Akurasi": round(float(np.mean((pred == Ye[te]).all(axis=1))), 3),
                      "Precision": np.nan, "Recall": np.nan, "F1": np.nan, "ROC_AUC": np.nan})
    SENSOR_RUNS[(WIN, sc, meth)] = dict(hidden=hid, act=act, status=res["hist"]["status"])
    f1s = [sens_rows[-5 + j]["F1"] for j in range(4)]
    print(f"  {SCEN_SHORT[sc]} {meth:10s} N={WIN} n={len(te):3d} "
          f"F1-sensor rata2={np.nanmean(f1s):.3f} [{res['hist']['status']}]")

sens_tbl = pd.DataFrame(sens_rows)
log_stage("sensor-ID selesai")
"""))

cells.append(code(r"""# === §12b — tabel sensor-ID: end-to-end, ringkas, per jumlah fault ===
_phase("§12 tabel sensor-ID")
if len(sens_tbl) == 0:
    print("Tidak ada hasil sensor-ID (semua dilewati budget).")
else:
    e2e = sens_tbl
    print("=== Identifikasi sensor — end-to-end LM asli (dirantai OOF §10) ===")
    print(e2e.to_string(index=False))
    export_df(e2e, "11_sensor_identifikasi")

    ring = (e2e[e2e.Sensor != "SEMUA-4-BENAR"]
            .groupby(["N_window", "Scenario", "Method"])[["F1", "ROC_AUC"]].mean().round(3))
    print("\n=== Ringkas: F1 sensor rata-rata per window x skenario x metode ===")
    print(ring.to_string())
    export_df(ring.reset_index(), "11_sensor_ringkas")

    n_fault = {"S1": "campuran", "S2": 1, "S3": 2, "S4": 3, "S5": 4}
    per = e2e[e2e.Sensor != "SEMUA-4-BENAR"].copy()
    per["Jumlah fault"] = per["Scenario"].map(n_fault)
    byn = per.groupby(["N_window", "Method", "Jumlah fault"])[["F1"]].mean().round(3)
    print("\n=== F1 sensor menurut banyaknya fault yang bercampur ===")
    print(byn.to_string())
    export_df(byn.reset_index(), "11_sensor_per_jumlah_fault")
"""))

cells.append(code(r"""# === §12c — plot F1 sensor per skenario, EDM vs JSD ===
_phase("§12 plot sensor-ID")
if len(sens_tbl) > 0:
    piv = (sens_tbl[sens_tbl.Sensor != "SEMUA-4-BENAR"]
           .groupby(["N_window", "Scenario", "Method"])["F1"].mean().unstack())
    wins = sorted(piv.index.get_level_values(0).unique()); nwin = len(wins)
    fig, axes = plt.subplots(1, nwin, figsize=(5 * nwin, 4.5), sharey=True)
    if nwin == 1:
        axes = [axes]
    for ax, WIN in zip(axes, wins):
        sub = piv.loc[WIN]
        order = [s for s in ["S1", "S2", "S3", "S4", "S5"] if s in sub.index]
        sub = sub.loc[order]
        sub.plot.bar(ax=ax, rot=0)
        ax.set_title(f"F1 identifikasi sensor (N={WIN})"); ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1 rata-rata 4 sensor")
    plt.tight_layout(); plt.savefig("exports/11_sensor_f1.png", dpi=120); plt.show()
    print("[Saved] exports/11_sensor_f1.png")
"""))

# ---------------------------------------------------------------- summary
cells.append(md(r"""# §13 — Ringkasan untuk penulisan paper

Sel di bawah merangkum jawaban enam RQ dari angka yang **baru saja dihitung**,
bukan dari asumsi. Kalimat klaimnya dibangkitkan mengikuti hasil: kalau
JSD-Fuzzy tidak menang, kalimatnya ikut berubah.
"""))

cells.append(code(r"""# === RINGKASAN RQ1..RQ6 ===
_phase("§13 ringkasan RQ")
lines = []
lines.append("RQ1 — Integrasi JSD sebagai model similarity")
if len(rq1):
    for _, r in rq1.iterrows():
        lines.append(f"    {r['Method']:10s}: n={r['Number of samples']}, "
                     f"{r['Scale features per sensor']} fitur/sensor x {r['Sensors']} sensor "
                     f"= {r['Total features']} fitur, NaN={r['NaN values']}, "
                     f"Inf={r['Infinite values']}, status={r['Status']}")
    lines.append("    -> struktur feature matrix kedua metode sebanding; JSD-Fuzzy sah dipakai "
                 "sebagai input classifier.")

lines.append("")
lines.append("RQ2 — Stabilitas (CV = std/mean)")
if 'cv_red' in dir() and len(cv_red):
    for WIN in WINDOW_LENGTHS:
        wsub = cv_red[cv_red["N_window"] == WIN]
        if not len(wsub):
            continue
        tw = int(wsub["Scales JSD lebih stabil"].sum()); tc = int(wsub["Scales dibandingkan"].sum())
        pct = 100 * tw / max(1, tc); avg = wsub["CV reduction (%)"].mean()
        if pct > 60 and avg > 0:
            kal = "JSD-Fuzzy lebih stabil pada sebagian besar skala"
        elif pct < 40:
            kal = "EDM-Fuzzy lebih stabil pada sebagian besar skala"
        else:
            kal = "stabilitas kedua metode sebanding"
        lines.append(f"    N={WIN}: JSD-Fuzzy CV lebih rendah pada {tw}/{tc} pasangan ({pct:.1f}%), "
                     f"rata-rata CV reduction {avg:+.2f}% -> {kal}.")

lines.append("")
lines.append("RQ3 — Separabilitas (boxplot mean entropy)")
if 'sep_interp' in dir() and len(sep_interp):
    for _, r in sep_interp.sort_values(["N_window", "Scenario"]).iterrows():
        lines.append(f"    N={r['N_window']} {r['Scenario']}: d(median spread)={r['d_median_spread']:+.3f}, "
                     f"d(IQR overlap)={r['d_IQR_overlap']:+.3f} -> {r['Interpretation']}")

lines.append("")
lines.append("RQ4 — Performa fault detection (ANN-LM)")
if len(perf_tbl):
    for _, r in perf_tbl.sort_values(["N_window", "Scenario", "Method"]).iterrows():
        _a = r['ROC_AUC_OvR']; _as = f"{_a:.3f}" if np.isfinite(_a) else "n/a"
        lines.append(f"    N={r['N_window']} {r['Scenario']} {r['Method']:10s} acc={r['Accuracy']:.3f} "
                     f"prec={r['Precision']:.3f} rec={r['Recall']:.3f} "
                     f"F1={r['F1']:.3f}+-{r['F1_std']:.3f} AUC={_as}")

lines.append("")
lines.append("RQ5 — Validasi statistik (per window + pooled deskriptif)")
if 'rq5_sum' in dir() and len(rq5_sum):
    for _, r in rq5_sum.iterrows():
        lines.append(f"    N={r['N_window']} mean F1: EDM={r['mean_F1_EDM']:.4f} vs JSD={r['mean_F1_JSD']:.4f} "
                     f"(selisih {r['mean_diff']:+.4f})")
        lines.append(f"    N={r['N_window']} paired t-test p={r['p_value']:.4f}; Wilcoxon p={r['wilcoxon_p']}; "
                     f"Cohen's d={r['cohen_d']}")
        lines.append(f"    N={r['N_window']} -> {r['verdict']}")

lines.append("")
lines.append("RQ6 — Identifikasi sensor (LM asli, end-to-end dari OOF §10)")
if 'sens_tbl' in dir() and len(sens_tbl):
    _rg = (sens_tbl[sens_tbl.Sensor != "SEMUA-4-BENAR"]
           .groupby(["Method"])[["F1"]].mean().round(4))
    for _m, _r in _rg.iterrows():
        lines.append(f"    {_m:10s} F1 sensor rata-rata lintas window & skenario = {_r['F1']:.4f}")
    _ex = (sens_tbl[sens_tbl.Sensor == "SEMUA-4-BENAR"]
           .groupby(["Method"])["Akurasi"].mean().round(4))
    for _m, _v in _ex.items():
        lines.append(f"    {_m:10s} exact-match SEMUA-4-BENAR rata-rata = {_v:.4f}")

ringkas = "\n".join(lines)
print(ringkas)
Path("exports/11_ringkasan_rq.txt").write_text(ringkas)
display(FileLink("exports/11_ringkasan_rq.txt"))
log_stage("selesai")
"""))

cells.append(md(r"""## Yang boleh dan tidak boleh diklaim dari notebook ini

**Boleh diklaim**

1. JSD-Fuzzy dibangun dengan mengganti **hanya** tahap similarity computation
   pada EDM-Fuzzy (Euclidean → Jensen-Shannon), semua tahap lain identik.
2. Feature matrix JSD-Fuzzy lengkap, tanpa Inf/fitur mati, dan **strukturnya
   sebanding** dengan EDM-Fuzzy (T nilai per sensor, 4T setelah konkatenasi;
   T = 15 → 60 fitur pada run ini).
3. Fitur JSD-Fuzzy **bukan** transformasi monoton dari EDM-Fuzzy — dibuktikan
   lewat korelasi per-fitur di §7a.
4. Perbandingan performa dilakukan **adil**: dataset, skenario, pembagian data,
   ruang hyperparameter, kriteria pemilihan model, dan metrik sama; grid search
   dijalankan **terpisah** untuk tiap metode × tiap skenario × tiap panjang
   window (N = 200/700/1000).
5. Angka stabilitas (CV), separabilitas (boxplot + overlap IQR), performa
   (akurasi/precision/recall/F1/confusion matrix), dan uji statistik dilaporkan
   apa adanya, termasuk ketika JSD-Fuzzy kalah.

**Tidak boleh diklaim**

1. **Efisiensi komputasi.** Ongkos dilaporkan sebagai catatan; JSD-Fuzzy
   melibatkan pemetaan ke sebaran dan operasi log, jadi tidak ada klaim lebih
   ringan, lebih cocok real-time, atau siap edge deployment.
2. **Keunggulan universal.** Hasilnya per skenario per window (15 sel); kalau
   menang di sebagian besar sel, kalimatnya "unggul pada sebagian besar
   skenario dan panjang window", bukan "lebih baik".
3. **Signifikansi statistik** kalau p ≥ 0,05. Dengan n = 5 pasang per window,
   p besar berarti belum terbukti, bukan terbukti tidak ada beda. Baris pooled
   15 pasang hanya deskriptif, bukan pengganti uji per window.
4. **Levenberg–Marquardt betulan.** Yang dipakai LM asli dari paper
   (implementasi sendiri, langkah Gauss-Newton terdamping + rasio gain +
   Jacobian analitik) — **bukan** `lbfgs`/quasi-Newton dan **bukan** MATLAB
   `trainlm`. Angka 98,6% pada paper TIDAK diklaim direproduksi di sini.
5. **Generalisasi ke fault nyata.** Fault-nya hasil injeksi terkontrol pada data
   lapangan; belum divalidasi pada sensor yang benar-benar rusak.

## Batasan yang sebaiknya masuk bagian *Limitations*

* Fault disuntikkan secara sintetis dengan intensitas tetap (drift 0,2/sampel,
  spike & bias 0,08, hardware stuck 8% / loss 5%). Sensitivitas terhadap
  intensitas fault belum diuji.
* Entropi bersifat **buta terhadap offset**: bias murni menggeser level tanpa
  mengubah keteraturan, jadi kelas *bias* memang sulit untuk kedua metode. Ini
  batas fisik representasi, bukan kegagalan salah satu similarity model.
* Tiga panjang window (N = 200/700/1000 sampel = 0,69/2,43/3,47 hari) dijalankan
  langsung di notebook ini (§10 dan §12).
* Grid arsitektur ANN DIPERKECIL vs notebook 08 (maksimum (64,16), bukan
  (128,64,32)): langkah LM padat butuh Jacobian m*C × P + penyelesaian
  persamaan normal P^3, jadi P dibatasi ~5rb supaya 16 kombinasi × 3 inner
  fold × 5 outer fold × 30 run muat di sesi Kaggle. Ruang gridnya SAMA untuk
  EDM-Fuzzy dan JSD-Fuzzy, jadi perbandingannya tetap adil.
* Iterasi LM DIBEDAKAN: pencarian grid memakai iterasi pendek
   (`LM_ITERS_SEARCH` = 6) dengan lipatan dalam `cv=2` (demi muat 30 run dalam
  < 12 jam), refit model terpilih memakai iterasi penuh (`LM_ITERS_REFIT` = 40). Tiap fit punya jaring waktu sendiri
  (`budget(MAX_TRAIN_S)` bila terpotong) dan statusnya dicetak per outer fold
  — jadi angka model besar bukan angka "konvergen" yang disembunyikan.
* Pemetaan vektor → sebaran pada JSD-Fuzzy punya dua hyperparameter tambahan
  (cacah bin dan lebar kernel). Nilai yang dipakai di sini (8 bin, lebar kernel
  = lebar bin) dipilih dari uji pendahuluan, bukan dari grid search penuh.
* Angka CPU untuk publikasi memakai `10_ongkos_per_bagian.csv`: CPU pekerja
  joblib (proses terpisah) diukur di dalam tugasnya dan dijumlahkan — tanpanya
  angka CPU *undercount*. `process_time` proses utama saja TIDAK cukup.

## File keluaran (folder `exports/`)

`10_rq1_validasi_feature_matrix.csv`, `10_rq1_korelasi_edm_vs_jsd.csv`,
`10_entropi_per_kondisi_skala.csv`, `10_rq2_cv_entropi.csv`,
`10_rq2_cv_reduction.csv`, `10_rq3_separabilitas.csv`,
`10_rq3_interpretasi.csv`, `10_rq4_performa_ann.csv`, `10_rq4_selisih.csv`,
`10_rq4_biaya_komputasi.csv`, `10_rq4_confusion_matrix.csv`,
`10_rq5_f1_berpasangan.csv`, `10_rq5_ttest.csv`, `10_cv_data_sensor.csv`,
`10_akuntansi_window.csv`, `10_ongkos_ekstraksi_fitur.csv`,
`10_ongkos_per_bagian.csv` (akuntansi CPU penuh tiap fase + TOTAL),
`10_ringkasan_rq.txt`, plus gambar `10_similarity_euclidean_vs_jsd.png`,
`10_kurva_entropi_multiskala_N{200,700,1000}.png`, `10_rq2_cv_per_skala_N{200,700,1000}.png`,
`10_rq3_boxplot_mean_entropy_N{200,700,1000}.png`, `10_rq4_performa_N{200,700,1000}.png`,
`10_rq4_confusion_matrix_N{200,700,1000}.png`, `10_broker_tersinkron.png`.
"""))

cells.append(md("## Export Custom Table (Sesuai Permintaan)"))
cells.append(code(r"""
_phase("Export Custom Table")
import matplotlib.pyplot as plt

methods = ["EDM-Fuzzy"]
classes = ["bias", "drift", "spike", "hardware"]
class_names = {"bias": "Bias", "drift": "Drift", "spike": "Spike", "hardware": "Hardware\nmalfunction"}

table_data = []
for w in WINDOW_LENGTHS:
    for c in classes:
        row = cv_tbl[(cv_tbl["Window Lengths"] == w) & (cv_tbl["Scenario"] == "S2") & (cv_tbl["Class"] == c) & (cv_tbl["Method"] == "EDM-Fuzzy")]
        if len(row):
            r = row.iloc[0]
            table_data.append({
                "Window Lengths": str(w) if c == "bias" else "",
                "Class": class_names[c],
                **{f"s{i}": r[f"s{i}"] for i in range(1, 16)}
            })

if len(table_data) > 0:
    headers = ["Window Lengths", "Class"] + [str(i) for i in range(1, 16)]
    fig, ax = plt.subplots(figsize=(14, 2 + 0.4 * len(table_data)))
    ax.axis("off")
    title = "Table 1 \u2013 Coefficients of Variation (CV) of EDM-Fuzzy Entropy across scale factors dan samples"
    ax.set_title(title, fontsize=12, pad=30, loc="left")
    
    cell_text = []
    for r in table_data:
        row_vals = [r["Window Lengths"], r["Class"]]
        for i in range(1, 16):
            try:
                row_vals.append(f"{float(r[f's{i}']):.3f}")
            except:
                row_vals.append("")
        cell_text.append(row_vals)
        
    cw = [0.1, 0.15] + [0.05]*15
    tab = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center", colWidths=cw)
    tab.auto_set_font_size(False)
    tab.set_fontsize(10)
    tab.scale(1, 2.5)
    
    for i in range(len(table_data) + 1):
        tab[i, 0].set_text_props(ha="left")
        tab[i, 1].set_text_props(ha="left")
        for j in range(len(headers)):
            c_cell = tab[i, j]
            c_cell.set_linewidth(0)
            if i == 0:
                c_cell.set_linewidth(1.2)
                c_cell.visible_edges = 'B'
                
    for j in range(len(headers)):
        tab[0, j].set_linewidth(1.2)
        tab[0, j].visible_edges = 'BT'
        tab[len(table_data), j].set_linewidth(1.2)
        tab[len(table_data), j].visible_edges = 'B'
        
    plt.text(0.6, 0.92, "Scale", ha="center", va="center", fontsize=11, transform=ax.transAxes)
    plt.plot([0.25, 0.95], [0.88, 0.88], color="black", lw=1.2, transform=ax.transAxes, clip_on=False)
    
    plt.tight_layout()
    plt.savefig("exports/11_cv_table_custom.png", dpi=300, bbox_inches="tight")
    plt.show()
"""))

cells.append(md(r"""# §14 — Akuntansi CPU penuh (untuk publikasi)

Setiap fase notebook dicatat `wall_s` (waktu dinding), `cpu_parent_s`
(`process_time` proses utama), `cpu_workers_s` (CPU pekerja joblib yang jalan
di proses terpisah — diukur di dalam tugasnya lalu dijumlahkan; tanpa ini
angka CPU *undercount*), dan `cpu_all_s` (total CPU yang benar-benar dipakai).
`py_peak_mb` = puncak alokasi Python (tracemalloc; tidak mencakup buffer
numpy), `rss_highwater_mb` = puncak RSS proses sampai akhir fase tersebut.

Tabel `10_ongkos_per_bagian.csv` + baris TOTAL di bawah adalah angka yang
dipakai di bagian metode/artikel. Biaya komputasi tetap DILAPORKAN sebagai
catatan, bukan klaim keunggulan.
"""))

cells.append(code(r"""# === §14 — akuntansi CPU penuh: tiap fase + total + cek silang ===
_trace_end()   # tutup fase "Export Custom Table"

sec_rows = [r for r in _CPU_ROWS
            if not (r["bagian"].startswith("ekstraksi fitur |")
                    or r["bagian"].startswith("ANN-LM |"))]
# Baris otoritatif ekstraksi fitur (CPU pekerja sudah dijumlahkan per window):
if len(feat_cost):
    for _, r in feat_cost.iterrows():
        sec_rows.append({"bagian": f"§6 ekstraksi fitur N={r['N_window']} {r['Metode']}",
                         "wall_s": float(r["wall_s"]), "cpu_parent_s": float(r["cpu_s"]),
                         "cpu_workers_s": float(r["cpu_workers_s"]),
                         "cpu_all_s": float(r["cpu_all_s"]),
                         "py_peak_mb": float(r["peak_mem_mb"]), "rss_highwater_mb": None})
else:
    print("PERINGATAN: feat_cost kosong — ekstraksi fitur terlewati (cek budget).", flush=True)
# Baris otoritatif ANN-LM per skenario x metode:
if len(comp_tbl):
    for _, r in comp_tbl.iterrows():
        sec_rows.append({"bagian": f"ANN-LM {r['Scenario']} {r['Method']}",
                         "wall_s": float(r["wall_s_total"]),
                         "cpu_parent_s": float(r["cpu_s_total"]),
                         "cpu_workers_s": float(r["cpu_workers_s_total"]),
                         "cpu_all_s": float(r["cpu_all_s_total"]),
                         "py_peak_mb": float(r["peak_mem_mb"]), "rss_highwater_mb": None})
else:
    print("PERINGATAN: comp_tbl kosong — ANN-LM terlewati (cek budget).", flush=True)

cpu_tbl = pd.DataFrame(sec_rows)
tot = cpu_tbl[["wall_s", "cpu_parent_s", "cpu_workers_s", "cpu_all_s"]].sum()
cpu_tbl = pd.concat([cpu_tbl, pd.DataFrame([{"bagian": "TOTAL",
                                             "wall_s": round(float(tot["wall_s"]), 2),
                                             "cpu_parent_s": round(float(tot["cpu_parent_s"]), 2),
                                             "cpu_workers_s": round(float(tot["cpu_workers_s"]), 2),
                                             "cpu_all_s": round(float(tot["cpu_all_s"]), 2),
                                             "py_peak_mb": None, "rss_highwater_mb": None}])],
                    ignore_index=True)
print("=== TABEL AKUNTANSI CPU — tiap fase + TOTAL (untuk publikasi) ===")
print(cpu_tbl.to_string(index=False))
export_df(cpu_tbl, "11_ongkos_per_bagian")
display(cpu_tbl)

# --- Cek silang: total fase vs jam global proses utama ---
_glob_wall = elapsed_s()
_glob_parent = time.process_time() - _NOTEBOOK_CPU0
print(f"Cek silang: sum(wall fase)={tot['wall_s']:.0f}s vs wall global={_glob_wall:.0f}s | "
      f"sum(cpu parent)={tot['cpu_parent_s']:.0f}s vs cpu parent global={_glob_parent:.0f}s | "
      f"TOTAL cpu_all={tot['cpu_all_s']:.0f}s (~{tot['cpu_all_s']/3600:.1f} jam-CPU)")
"""))

nb = {"cells": cells,

      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "| cells:", len(cells))
