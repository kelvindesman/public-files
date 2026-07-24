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
  1. Window length: durasi sesuai diagram; pada laju 5 menit N in {200, 700, 1000}.
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
| **Panjang window** | Data di-preprocessing jadi **1 sampel per 5 menit**, jadi cacah sampel diagram diskalakan agar **durasinya identik**: **N ∈ {200, 700, 1000}** = 0,69 / 2,43 / 3,47 hari (rasio 1 : 3,5 : 5 sama seperti 2000/7000/10000 pada laju 30 detik). Ketiganya dijalankan sebagai studi sensitivitas. |
| **Fitur** | **EDM-Fuzzy murni**: 4 sensor × T skala = **4T fitur** (kotak konkatenasi). JSD-Fuzzy ikut sebagai pembanding usulan paper. Fitur time-domain **tidak** dipakai di jalur utama karena tidak ada di diagram. |
| **Classifier** | **ANN-LM** — `solver='lbfgs'` (quasi-Newton, paling dekat ke Levenberg–Marquardt; LM asli hanya ada di MATLAB `trainlm`), **hidden layer dipilih Grid Search**, output C neuron sesuai jumlah kelas tiap skenario. |
| **Data** | `data_sensor.csv` — 281.721 baris mentah, 2025-09-14 s.d. 2025-12-21 (97,8 hari), interval akuisisi 30 detik, kolom `kelembaban1..kelembaban4`. Setelah rata-rata per 5 menit: **28.173 baris**. |

---
"""))

cells.append(md("""## §0 — Baca ini dulu (versi tanpa istilah)

Analogi seluruh notebook: **empat termometer di satu ruangan.**

1. **Akuisisi.** Empat sensor kelembaban tanah mencatat angka tiap 30 detik.
2. **Broker.** Semua catatan dikumpulkan ke satu meja — tapi tetap **empat
   lembar terpisah**, tidak dijadikan satu angka rata-rata. (Kalau dirata-rata,
   satu sensor rusak akan tersamarkan tiga yang sehat — dibuktikan di notebook `02`.)
3. **Preprocessing.** Catatan 30 detik dirangkum jadi **satu angka tiap 5 menit**
   (rata-rata 10 pembacaan). Seperti mencatat nilai rata-rata per jam pelajaran
   alih-alih tiap menit: pola tetap terlihat, derau berkurang.
4. **Potong jadi window.** Rekaman panjang dipotong jadi potongan-potongan
   berdurasi tetap, seperti memotong film jadi klip. Satu klip = satu contoh
   yang dinilai.
5. **Ekstraksi fitur (entropy).** Tiap klip diringkas jadi beberapa angka yang
   menggambarkan **seberapa tidak beraturan** sinyalnya. Analogi: menilai
   tulisan tangan bukan dari isinya, tapi dari serapi apa hurufnya.
6. **ANN.** Angka-angka ringkasan itu diberikan ke jaringan saraf tiruan untuk
   memutuskan: klip ini normal atau ada fault, dan fault jenis apa.
7. **Tahap tambahan.** Untuk klip yang divonis fault, model kedua menebak
   **sensor mana** yang rusak, lengkap dengan angka keyakinannya.

### Tiga angka yang paling sering salah baca

| Kolom | **Bukan** ini | Artinya yang benar |
|---|---|---|
| `Prevalensi` | bukan hasil prediksi model | **kunci jawaban**: berapa persen window uji yang memang sensornya rusak. Ini sifat data (karena fault-nya kita suntikkan sendiri), bukan tebakan model |
| `Akurasi` | bukan "peluang sensor rusak" | berapa persen tebakan model yang benar untuk sensor itu |
| `P(Sx rusak)` | bukan akurasi, bukan prevalensi | keyakinan model **untuk satu window tertentu** |

Analogi: `Prevalensi` = berapa banyak siswa yang benar-benar sakit di kelas;
`Akurasi` = berapa banyak tebakan dokter yang tepat; `P(rusak)` = "menurut saya
anak ini 70% kemungkinan demam".

### Aturan main sebelum menyebut angka mana pun "bagus"

- **F1 tinggi + ROC-AUC ≈ 0,5 = tidak sah.** Artinya model menebak "semua
  rusak" terus. Analogi: dokter yang menyatakan **semua** pasien sakit akan
  dapat recall 100% dan terlihat hebat — padahal tidak menilai apa-apa.
  Kegagalan seperti ini pernah terjadi di notebook lama, lihat README.
- Karena itu **setiap tabel identifikasi sensor di sini selalu memuat
  `Prevalensi`, `ROC_AUC`, dan uji kalibrasi** — supaya kecurangan semacam itu
  langsung kelihatan.

### Daftar pertanyaan yang dijawab notebook ini

Semua pertanyaan pembimbing dijawab di **§13 — Tanya-Jawab**, dengan penunjuk
ke sel yang mencetak angkanya:

1. Tabel identifikasi sensor itu artinya apa? → §0 di atas + §10 + §13-Q1
2. Bisa dites dengan data yang dikondisikan **bersih** (tanpa fault)? → **§11 kontrol negatif**
3. Bisa dites dengan fault **hanya di satu sensor**? → **§11 kontrol tunggal**
4. Bedanya tabel "F1 rata-rata 4 sensor" dengan peta panas F1? → §13-Q4
5. Preprocessing-nya per jam? → **bukan, per 5 menit** — §3
6. Yang masuk broker sudah di-downsample atau belum? → §2/§3 + §13-Q6
7. Preprocessing-nya cuma resample, atau ada validasi & pembersihan? → **§3b**
8. Panjang datanya 2000/7000/10000 atau 200/700/1000? → **§5, tabel konversi**
9. Window yang tidak genap N sampel dibuang atau tidak? → **§5, blok akuntansi window**
"""))

cells.append(md("""## Peta diagram → sel notebook

| Kotak di flowchart | Bagian notebook |
|---|---|
| Multisource Soil Moisture Sensor Acquisition | **§1** — muat `data_sensor.csv` |
| **Broker** — Multisource data integration | **§2** — 4 sensor jadi satu tabel, identitas sensor dipertahankan |
| Time synchronization — S₁..S₄ | **§3** — rata-rata per 5 menit, grid waktu seragam, gap ditambal |
| Data preparation and Fault Injection | **§4** — injeksi fault (sensor-selective) |
| Time series segmentation and Labelling, N ∈ {2000; 7000; 10000} pada laju 30 detik | **§5** — segmentasi + pelabelan, N ∈ {200; 700; 1000} pada laju 5 menit (durasi sama) |
| EDM-Fuzzy Entropy Feature Extraction, τ = 1..T | **§6** — E₁..E₄ per sensor |
| Multisensor Entropy Feature Concatenation → 4T fitur | **§7** — konkatenasi |
| ANN-LM Classification (input 4T, hidden = grid search, output C) | **§8** — ANN-LM + CV |
| Fault Classification and Evaluation | **§9** — performa + biaya komputasi |
| *(tambahan, di luar diagram)* | **§3b** — laporan validasi & pembersihan data |
| *(tambahan, di luar diagram)* | **§10** — sensor mana yang rusak, **§10b** — probabilitasnya |
| *(tambahan, di luar diagram)* | **§11** — kontrol: data bersih & fault satu sensor |
| *(tambahan, di luar diagram)* | **§13** — tanya-jawab pembimbing |

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

# --- Laju sampling setelah preprocessing: rata-rata per 5 menit ---
RESAMPLE_RULE = os.environ.get("RESAMPLE_RULE", "5min")    # "" = pakai laju mentah 30 detik
SAMPLING_SECONDS = 300 if RESAMPLE_RULE else 30            # interval antar sampel setelah resample

# --- Panjang window: durasi sama dengan diagram, cacah sampel menyesuaikan laju ---
# Diagram menulis N in {2000, 7000, 10000} pada laju 30 detik = 0,69 / 2,43 / 3,47 hari.
# Pada laju 5 menit, durasi yang sama = N in {200, 700, 1000}. Rasio 1 : 3,5 : 5 tetap.
_DEFAULT_WIN = "200,700,1000" if RESAMPLE_RULE else "2000,7000,10000"
WINDOW_LENGTHS = [int(v) for v in os.environ.get("WINDOW_LENGTHS", _DEFAULT_WIN).split(",")]
T_SCALES = int(os.environ.get("T_SCALES", 10))       # tau = 1..T -> fitur EDM = 4T
MAX_PER_CLASS = int(os.environ.get("MAX_PER_CLASS", 100))

# Drift bertambah per sampel, jadi intensitasnya harus ikut laju sampling supaya
# kemiringan drift per satuan WAKTU sama dengan versi 30 detik (0,02 per 30 detik).
DRIFT_PER_SAMPLE = 0.02 * (SAMPLING_SECONDS / 30)

m = 2; r_ratio = 0.2; n_ref = 128; jsd_bins = 40
SENSORS = ["S1", "S2", "S3", "S4"]
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]
# Tiap kondisi fault diulang N_REPEAT_SUBSET kali dengan subset sensor yang
# DIUNDI ULANG tiap kali. Tanpa ini, satu kondisi cuma punya 4 pola label sensor
# dan dua sensor bisa selalu muncul berbarengan -> kolom labelnya kembar dan
# tahap §10 tidak sah (ROC-AUC jatuh ke 0,5). Lihat §13-Q1.
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

Keempat sensor dipaksa ke **satu sumbu waktu seragam**. Kalau ada stempel waktu
yang bolong atau ganda, di sini ketahuan dan ditambal, supaya sampel ke-*i* dari
keempat sensor benar-benar merujuk waktu yang sama sebelum masuk windowing.

**Preprocessing laju sampling.** Akuisisi mentah 30 detik dirata-ratakan jadi
**satu sampel per 5 menit** (`RESAMPLE_RULE`): 281.721 baris → ±28.173 baris.
Rata-rata, bukan pengambilan tiap ke-10, supaya derau akuisisi ikut teredam dan
tidak ada sampel yang dibuang begitu saja.

Akibatnya panjang window ikut menyesuaikan — lihat §5. Satu hal yang dirapikan
bersamaan: kemiringan drift pada injeksi fault dikalikan 10 (`DRIFT_PER_SAMPLE`)
supaya drift per satuan **waktu** tetap sama seperti versi 30 detik; kalau tidak,
fault drift-nya jadi 10× lebih lemah dan angkanya tidak sebanding dengan
notebook `01`–`06`.
"""))

cells.append(code("""# --- Diagnosa sinkronisasi sebelum ditambal ---
dt = df_broker.index.to_series().diff().dt.total_seconds().dropna()
print("Interval antar-sampel (detik):")
print(dt.value_counts().head(5).to_string())
print("Stempel waktu ganda:", int(df_broker.index.duplicated().sum()))

n_dup = int(df_broker.index.duplicated().sum())
df_broker = df_broker[~df_broker.index.duplicated(keep="first")]
df_pre = df_broker.copy()          # simpan versi 30 detik untuk laporan mutu di §3b

# --- Preprocessing: rata-rata per 5 menit ---
if RESAMPLE_RULE:
    n_before = len(df_broker)
    df_broker = df_broker.resample(RESAMPLE_RULE).mean()
    print(f"\\nPreprocessing laju sampling: rata-rata per {RESAMPLE_RULE} "
          f"-> {n_before} baris menjadi {len(df_broker)} baris")

# --- Paksa ke grid seragam ---
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
print(f"Rentang total: {(len(X) * SAMPLING_SECONDS) / 86400:.1f} hari")
print("S1..S4 sekarang berbagi sumbu waktu yang sama.")
for _w in WINDOW_LENGTHS:
    _dur = _w * SAMPLING_SECONDS / 86400
    _nw = (len(X) - _w) // max(1, _w // 2) + 1 if len(X) >= _w else 0
    print(f"  N={_w:>5} -> {_dur:5.2f} hari per window, {_nw} window tersedia per kondisi")
"""))

cells.append(md("""# §3b — Validasi & pembersihan data (bukan cuma resample)

Pertanyaan pembimbing: *"apakah di data preprocessing ada hal lain yang
dilakukan, seperti validation dan cleaning?"* — **Ada.** Urutannya:

| # | Langkah | Jenis | Kenapa perlu |
|---|---|---|---|
| 1 | Kolom wajib `kelembaban1..4` ada | validasi | kalau tidak ada, program berhenti dengan pesan jelas (§1) |
| 2 | Stempel waktu dijadikan `datetime` UTC lalu diurutkan | pembersihan | data mentah tidak dijamin urut |
| 3 | Stempel waktu ganda dibuang (`keep="first"`) | pembersihan | satu waktu tidak boleh punya dua baris |
| 4 | Cek nilai kosong (NaN) per kolom | validasi | melaporkan, bukan menyembunyikan |
| 5 | Cek nilai di luar rentang fisik 0–100 % | validasi | pembacaan mustahil = sensor bermasalah, harus tercatat |
| 6 | Cek pembacaan macet (nilai sama berturut-turut ≥ 10 kali) | validasi | gejala klasik sensor *stuck-at* |
| 7 | Cek pencilan ekstrem (`|z| > 6`) | validasi | dilaporkan, **tidak dibuang** |
| 8 | Rata-rata per 5 menit | pembersihan | meredam derau akuisisi (§3) |
| 9 | Paksa ke grid waktu seragam + tambal bolong dengan interpolasi waktu | pembersihan | sampel ke-*i* keempat sensor harus merujuk waktu yang sama |
| 10 | Sisa NaN ditambal median kolom | pembersihan | jaminan tidak ada NaN masuk ekstraksi fitur |

**Kenapa pencilan dan pembacaan macet dilaporkan tapi tidak dibuang:** yang
sedang diteliti justru **fault**. Membuang pembacaan aneh sebelum pemodelan sama
saja menghapus barang bukti sebelum penyidikan. Analogi: kalau ingin melatih
detektor uang palsu, jangan buang dulu semua uang yang terlihat mencurigakan.

Referensi praktik ini: *Data quality dimensions* — kelengkapan, konsistensi,
ketepatan waktu, validitas (Batini & Scannapieco, *Data Quality*, Springer 2016);
serta penanganan deret waktu sensor pada Hodge & Austin, *A survey of outlier
detection methodologies*, Artificial Intelligence Review 22(2), 2004.
"""))

cells.append(code("""# === Laporan mutu data: sebelum vs sesudah preprocessing ===
def flatline_max_run(s):
    \"\"\"Panjang deret nilai identik berturut-turut terpanjang (gejala sensor macet).\"\"\"
    v = s.to_numpy()
    brk = np.flatnonzero(np.diff(v) != 0)
    if len(brk) == 0:
        return len(v)
    seg = np.diff(np.concatenate(([-1], brk, [len(v) - 1])))
    return int(seg.max())

qc_rows = []
for c in cols:
    a = df_pre[c]
    z = (a - a.mean()) / (a.std(ddof=0) + 1e-12)
    qc_rows.append({
        "Kolom": c,
        "n_baris_30s": len(a),
        "NaN": int(a.isna().sum()),
        "Min": round(float(a.min()), 2),
        "Max": round(float(a.max()), 2),
        "Di_luar_0_100": int(((a < 0) | (a > 100)).sum()),
        "Macet_terpanjang": flatline_max_run(a.dropna()),
        "Pencilan_z6": int((z.abs() > 6).sum()),
        "n_baris_5menit": int(df_sync[c].notna().sum()),
        "NaN_setelah_bersih": int(df_sync[c].isna().sum()),
    })
qc = pd.DataFrame(qc_rows)
print("=== Laporan mutu data (validasi + pembersihan) ===")
print(qc.to_string(index=False))
export_df(qc, "07_mutu_data")

print(f"\\nStempel waktu ganda dibuang : {n_dup}")
print(f"Slot grid yang bolong lalu ditambal interpolasi waktu : {n_gap}")
print(f"Baris: {len(df_raw)} (30 detik) -> {len(df_pre)} (setelah buang ganda) "
      f"-> {len(df_sync)} (rata-rata 5 menit, grid seragam)")
print("Tidak ada baris yang dibuang karena 'aneh' — pencilan & pembacaan macet "
      "hanya DILAPORKAN, karena justru itu bahan penelitian fault.")
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

cells.append(code("""def simulate_drift_fault(x, intensity=DRIFT_PER_SAMPLE, seed=None):
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
        (simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}),
        (simulate_spike_fault, {"intensity": 0.08, "p": 0.015}),
        (simulate_bias_fault, {"bias": 0.08}),
        (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05}),
    ]})],
    "drift": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE})],
    "spike": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015})],
    "bias": [(simulate_bias_fault, {"bias": 0.08})],
    "hardware": [(simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "bias+malfunc": [(simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+bias": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08})],
    "drift+malfunc": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+bias": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_bias_fault, {"bias": 0.08})],
    "drift+spike": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_spike_fault, {"intensity": 0.08, "p": 0.015})],
    "spike+bias+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+bias+malfunc": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "spike+drift+malfunc": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05})],
    "drift+spike+bias": [(simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE}), (simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08})],
    "spike+bias+malfunc+drift": [(simulate_spike_fault, {"intensity": 0.08, "p": 0.015}), (simulate_bias_fault, {"bias": 0.08}), (simulate_hardware_fault, {"stuck_prob": 0.08, "loss_prob": 0.05}), (simulate_drift_fault, {"intensity": DRIFT_PER_SAMPLE})],
}
condition_names = ["normal"] + list(SCENARIOS.keys())
print("Kondisi:", len(SCENARIOS), "fault + normal =", len(condition_names))
"""))

cells.append(md("""# §5 — Time Series Segmentation and Labelling

Kotak `Time series segmentation and Labelling`. Diagram menulis **N ∈ {2000;
7000; 10000} sampel** pada laju akuisisi 30 detik — yaitu window **0,69 / 2,43 /
3,47 hari**. Karena §3 sudah merata-ratakan data jadi 1 sampel per 5 menit,
cacah sampelnya diskalakan jadi **N ∈ {200; 700; 1000}** supaya **durasi
window-nya persis sama** dan rasio 1 : 3,5 : 5 pada diagram tetap terjaga.

Kenapa tidak dipaksa tetap 10.000 sampel: pada laju 5 menit itu berarti satu
window = 34,7 hari, sedangkan seluruh rekaman hanya 97,8 hari → tersisa **4
window** untuk seluruh dataset. Cross-validation berkelompok tidak bisa jalan
dengan 4 window. Dengan N = 200/700/1000 tersedia 280 / 79 / 55 window per
kondisi.

Ketiga panjang window dijalankan penuh, jadi hasilnya sekaligus jadi **studi
sensitivitas panjang window** — bukan satu angka tunggal.

Catatan kejujuran: pada N = 200 dengan τ sampai 10, deret hasil coarse-graining
skala terkasar hanya 20 titik, jadi entropi di skala itu lebih berisik daripada
di N = 1000. Itu bagian dari yang dibaca pada studi sensitivitas.

Stride dipasang `N/2`. Konsekuensinya window bertetangga tumpang-tindih 50%,
sehingga **cross-validation harus dikelompokkan menurut blok waktu** (dipakai di
§8), kalau tidak potongan sinyal yang sama bisa muncul di data latih **dan** uji
sekaligus dan akurasinya jadi terlalu bagus.

Tiap window dilabeli dua hal:
- label **kondisi** (normal / jenis kombinasi fault) → dipakai §8,
- label **per-sensor** `[S1,S2,S3,S4]` → dipakai §10.

---

## Tabel konversi panjang window — 2000/7000/10000 vs 200/700/1000

Pertanyaan pembimbing: *"vin length datanya bukan 2000, 7000, 10.000?"* —
**Dua-duanya benar, satuannya yang berbeda.** Diagram menghitung sampel pada
laju **30 detik**; notebook ini menghitung sampel setelah preprocessing **5
menit**. **Durasinya identik.**

| Diagram (laju 30 detik) | Notebook ini (laju 5 menit) | Durasi window | Window tersedia |
|---|---|---|---|
| N = 2 000 sampel | N = 200 sampel | 0,69 hari (16,7 jam) | 280 per kondisi |
| N = 7 000 sampel | N = 700 sampel | 2,43 hari | 79 per kondisi |
| N = 10 000 sampel | N = 1 000 sampel | 3,47 hari | 55 per kondisi |

Analogi: film 90 menit tetap 90 menit, entah disimpan 24 fps (129.600 frame)
atau 12 fps (64.800 frame). Yang berubah cacah frame-nya, bukan durasinya.
Rasio 1 : 3,5 : 5 pada diagram juga tetap terjaga.

Kalau tetap dipaksa **10.000 sampel pada laju 5 menit**, satu window = **34,7
hari**, sedangkan seluruh rekaman cuma 97,8 hari → tersisa **4 window** untuk
seluruh dataset; cross-validation 5-fold berkelompok tidak bisa jalan. Angka
persisnya dicetak sel di bawah.

---

## Akuntansi window: apakah ada window yang dibuang?

Pertanyaan pembimbing: *"jika ada window yang tidak cukup ... sampel, apakah
window tersebut excluded?"* Jawabannya dipisah jadi tiga hal yang sering
tertukar:

| Hal | Angka | Perlakuan |
|---|---|---|
| **Panjang window N** | 200 / 700 / 1000 sampel | **Tidak pernah ada window pendek.** `sliding_window_view` hanya membentuk window yang **genap** N sampel. Sisa ekor di ujung rekaman (< N sampel) tidak dijadikan window — jadi tidak ada window setengah jadi yang perlu dibuang |
| **`n_ref` = 128** | bukan panjang window | jumlah **vektor acuan** yang diambil acak saat menghitung kemiripan fuzzy, supaya biayanya tidak O(N²). Kalau titik yang tersedia < `n_ref`, dipakai **semuanya** (`if N > n_ref` — lihat §6), window tetap ikut, **tidak dibuang** |
| **Skala kasar τ** | butuh ≥ m+2 = 4 titik setelah coarse-graining | kalau di skala tertentu titiknya kurang, **hanya fitur skala itu** yang jadi `NaN` lalu ditambal median di §7 — **window-nya tetap ikut** |

Satu-satunya penyaringan window yang memang dilakukan: pada kondisi fault,
window yang porsi sampel ter-fault-nya **≤ 1 %** (`FAULT_RATIO_THR`) dibuang,
karena melabelinya "fault" akan menyesatkan — isinya praktis normal. Jumlahnya
dicetak sel di bawah.
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

SEG_AUDIT = {}          # akuntansi window per N, dicetak di sel berikutnya

def segment_and_label(X, win):
    \"\"\"-> W (nwin,win,4), y kondisi, Ysens (nwin,4), start index tiap window.

    Subset sensor yang terkena fault DIUNDI ULANG tiap pengulangan, sebanyak
    N_REPEAT_SUBSET kali per kondisi. Ini yang membuat keempat kolom label
    per-sensor saling bebas; kalau subset-nya cuma diundi sekali per ukuran,
    ada pasangan sensor yang selalu muncul bersama dan kolom labelnya kembar.
    \"\"\"
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

    sisa_ekor = int(len(X) - (((len(X) - win) // stride) * stride + win)) if len(X) >= win else len(X)
    SEG_AUDIT[win] = {"N_window": win, "stride": stride,
                       "window_per_kondisi": int(len(W0)),
                       "sampel_sisa_ekor": sisa_ekor,
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
    # grup CV = blok waktu; window yang tumpang-tindih jatuh ke grup yang sama
    groups = starts // WIN
    SEGMENTS[WIN] = dict(W=W, y=y, Ysens=Ysens, starts=starts, groups=groups)
    print(f"N={WIN:6d} | window={W.shape} | kondisi terisi={len(np.unique(y)):2d} "
          f"| blok waktu={len(np.unique(groups)):3d} | prevalensi sensor={Ysens.mean(axis=0).round(2)}")
"""))

cells.append(code("""# === Tabel konversi panjang window + akuntansi window (jawaban Q8 & Q9) ===
konv = pd.DataFrame([{
    "N_diagram_30detik": w * (SAMPLING_SECONDS // 30),
    "N_notebook_5menit": w,
    "Durasi_jam": round(w * SAMPLING_SECONDS / 3600, 2),
    "Durasi_hari": round(w * SAMPLING_SECONDS / 86400, 2),
    "Window_tersedia_per_kondisi": SEG_AUDIT[w]["window_per_kondisi"],
} for w in WINDOW_LENGTHS])
print("=== Konversi panjang window: satuan diagram vs satuan notebook ===")
print(konv.to_string(index=False))
export_df(konv, "07_konversi_window")

# Kalau N=10.000 dipaksa pada laju 5 menit
for w in (2000, 7000, 10000):
    dur = w * SAMPLING_SECONDS / 86400
    nw = (len(X) - w) // max(1, w // 2) + 1 if len(X) >= w else 0
    print(f"  [andai] N={w:>5} sampel pada laju {SAMPLING_SECONDS}s -> {dur:5.1f} hari/window, "
          f"{nw} window untuk SELURUH rekaman ({len(X)} sampel) -> CV 5-fold {'bisa' if nw >= 10 else 'TIDAK bisa'}")

print("\\n=== Akuntansi window: apa yang dibentuk, apa yang dibuang ===")
audit = pd.DataFrame([SEG_AUDIT[w] for w in WINDOW_LENGTHS])
print(audit.to_string(index=False))
export_df(audit, "07_akuntansi_window")
print("Baca: 'window_setengah_jadi' selalu 0 -> tidak pernah ada window dengan "
      "sampel kurang dari N; sisa ekor rekaman tidak dijadikan window sama sekali.")
print("Satu-satunya window yang dibuang: porsi sampel ter-fault <= "
      f"{FAULT_RATIO_THR:.0%} (kolom 'dibuang_fault<=1%').")
"""))

cells.append(code("""# === Uji kebebasan label per-sensor (pengaman anti label kembar) ===
# Kalau dua kolom label selalu sama, classifier per-sensor untuk keduanya akan
# identik dan ROC-AUC jatuh ke ~0,5. Cacat inilah yang membatalkan notebook lama.
for WIN in WINDOW_LENGTHS:
    Ys = SEGMENTS[WIN]["Ysens"]
    kor = np.corrcoef(Ys.T)
    off = kor[~np.eye(4, dtype=bool)]
    kembar = [(SENSORS[i], SENSORS[j]) for i in range(4) for j in range(i + 1, 4)
              if (Ys[:, i] == Ys[:, j]).all()]
    print(f"N={WIN:>5} | prevalensi per sensor={Ys.mean(axis=0).round(3)} "
          f"| korelasi antar-label maks={off.max():.2f} | pasangan kembar={kembar or 'tidak ada'}")
    if kembar:
        print(f"  !! PERINGATAN: label sensor kembar {kembar} pada N={WIN}. "
              "Hasil §10 untuk pasangan itu TIDAK sah — naikkan N_REPEAT_SUBSET.")
print("\\nKalau tidak ada pasangan kembar: keempat label sensor saling bebas, "
      "jadi tabel §10 dan §10b sah dibaca per sensor.")
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
untuk window yang diputuskan fault oleh ANN-LM, model tahap kedua menghasilkan
4 keputusan biner `[S1,S2,S3,S4]` beserta probabilitasnya.

## Cara kerjanya: tiap sensor dibandingkan dengan tetangganya

Analogi: ada empat termometer di satu ruangan. Untuk menebak mana yang rusak,
yang dilihat bukan angka satu termometer saja, tapi **selisihnya terhadap
rata-rata tiga temannya**. Termometer yang menyimpang sendirian itulah yang
mencurigakan.

Diterjemahkan ke fitur: vektor 4T dipecah lagi jadi empat blok per sensor, lalu
untuk sensor ke-*j* dibentuk masukan

$$x_j = [\\; E_j \\;|\\; \\overline{E}_{k \\ne j} \\;|\\; E_j - \\overline{E}_{k \\ne j} \\;]$$

yaitu **entropi sensor itu sendiri**, **rata-rata entropi tiga sensor lain**, dan
**selisih keduanya**. Satu model biner dilatih pada gabungan keempat sensor
(4× lebih banyak contoh latih), lalu dipakai bergantian untuk S1..S4.

**Ambang keputusan tidak dipatok 0,5.** Ambang dicari di dalam **data latih tiap
fold** (nilai yang memaksimalkan F1 di situ), lalu dipakai apa adanya di data
uji. Kalau ambangnya dipatok 0,5, model cenderung menjawab "rusak" untuk hampir
semua sensor sehingga recall 1,00 dan precision persis sama dengan prevalensi —
persis cacat yang dicurigai pembimbing. Ambang rata-rata terpilih dilaporkan di
tabel biaya §10.

Dua keuntungan yang langsung menjawab kecurigaan "modelnya cuma menebak semua
rusak":

1. Masukannya **relatif**. Kalau keempat sensor sama-sama tenang, kolom selisih
   ≈ 0 untuk semuanya, jadi tidak ada alasan menjawab "rusak".
2. Modelnya **tidak tahu nomor sensor**. Ia tidak bisa hafal "S4 biasanya
   rusak"; ia harus melihat bukti pada blok fiturnya.

Dievaluasi tiga cara:

| Evaluasi | Artinya |
|---|---|
| `oracle` | pada window yang **memang** fault → batas atas kemampuan tahap ini |
| `end_to_end` | pada window yang **diprediksi** fault oleh ANN-LM → angka apa adanya |
| `satu_sensor` | khusus window dengan **tepat satu** sensor rusak → uji kontrol §11 |

**`Prevalensi` wajib dibaca bareng `F1`.** Prevalensi ≈ 1 dengan ROC-AUC ≈ 0,5
berarti model cuma menebak "semua sensor rusak" — cacat yang membatalkan hasil
notebook per-sensor versi lama (lihat README).

Keluarannya **dua bentuk**: keputusan biner (tabel di §10) dan **probabilitas
`P(sensor rusak)` per window** (§10b). Probabilitas dikalibrasi supaya angka
0,43 benar-benar berarti "43 dari 100 window seperti ini memang rusak".
"""))

cells.append(code("""from sklearn.calibration import CalibratedClassifierCV

def stack_per_sensor(F):
    \"\"\"(nwin, 4T) -> (nwin, 4, 3T): [blok sensor | rata-rata 3 sensor lain | selisih].

    Fitur jadi RELATIF terhadap tetangga, dan model tidak pernah melihat nomor
    sensor. Itu yang mencegah jawaban malas 'semua sensor rusak'.
    \"\"\"
    n, d = F.shape
    b = d // len(SENSORS)
    B = F.reshape(n, len(SENSORS), b)
    lain = (B.sum(axis=1, keepdims=True) - B) / (len(SENSORS) - 1)
    return np.concatenate([B, lain, B - lain], axis=2)

def sensor_stage(F, y, Ysens, groups, pred_oof):
    n_grp = len(np.unique(groups))
    k = int(min(N_SPLITS, n_grp, np.min(np.bincount(y)[np.unique(y)])))
    if k < 2:
        return [], None
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
    oof_pred = np.full(Ysens.shape, -1, int)
    oof_proba = np.full(Ysens.shape, np.nan, float)

    Xs = stack_per_sensor(F)                        # (nwin, 4, 3T)
    ns = len(SENSORS)
    thresholds = []

    tracemalloc.start(); c0 = _time.process_time(); t0 = _time.perf_counter()
    for tr, te in skf.split(F, y, groups=groups):
        tr_f = tr[y[tr] != 0]                       # latih hanya dari window fault
        if len(tr_f) < 12 or Ysens[tr_f].sum() == 0:
            continue
        Xtr = Xs[tr_f].reshape(-1, Xs.shape[2])     # 4 baris per window: satu per sensor
        ytr = Ysens[tr_f].reshape(-1)
        if len(np.unique(ytr)) < 2:
            continue
        base = MLPClassifier(solver=ANN_SOLVER, hidden_layer_sizes=(max(16, Xs.shape[2] // 2),),
                             alpha=1e-3, max_iter=ANN_MAX_ITER, random_state=RANDOM_SEED)
        # Kalibrasi Platt di DALAM data latih fold: MLP mentah terlalu percaya diri,
        # angkanya belum layak dibaca sebagai probabilitas. cv=3 butuh >=3 contoh per
        # kelas; kalau tidak cukup, jatuh ke MLP mentah (proba apa adanya).
        n_cal = int(min(np.bincount(ytr.astype(int))))
        est = (CalibratedClassifierCV(base, method="sigmoid", cv=3) if n_cal >= 3 else base)
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                          ("clf", est)])
        try:
            pipe.fit(Xtr, ytr)
        except Exception:                                      # kalibrasi gagal -> MLP mentah
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                              ("clf", base)])
            pipe.fit(Xtr, ytr)
        Xte = Xs[te].reshape(-1, Xs.shape[2])
        try:
            # Ambang keputusan TIDAK dipatok 0,5. Ambang dicari di DALAM data latih
            # fold (memaksimalkan F1 di situ), lalu dipakai apa adanya di data uji.
            # Tanpa ini, model cenderung menjawab "rusak" untuk hampir semua sensor
            # -> recall 1,00 dan precision = prevalensi, persis cacat yang dikritik.
            ptr = pipe.predict_proba(Xtr)[:, 1]
            ths = np.linspace(0.05, 0.95, 19)
            th = float(ths[int(np.argmax([f1_score(ytr, (ptr >= t).astype(int), zero_division=0)
                                          for t in ths]))])
            pte = pipe.predict_proba(Xte)[:, 1]
            oof_proba[te] = pte.reshape(-1, ns)
            oof_pred[te] = (pte >= th).astype(int).reshape(-1, ns)
            thresholds.append(th)
        except Exception:
            oof_pred[te] = pipe.predict(Xte).reshape(-1, ns)
    t1 = _time.perf_counter(); c1 = _time.process_time()
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    cost = {"cpu_s": c1 - c0, "wall_s": t1 - t0, "peak_mem_mb": peak / (1024 * 1024),
            "ambang_rata2": round(float(np.mean(thresholds)), 3) if thresholds else 0.5}

    rows = []
    satu = Ysens.sum(axis=1) == 1                 # window dengan TEPAT satu sensor rusak
    for mask, tag in ((y != 0, "oracle"),
                      (pred_oof != 0, "end_to_end"),
                      ((pred_oof != 0) & satu, "satu_sensor")):
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
    return rows, cost, oof_pred, oof_proba

sensor_rows, sensor_cost_rows = [], []
PROBA = {}                       # (N, skenario, metode) -> probabilitas out-of-fold per sensor
for (WIN, sc, meth), r in RUNS.items():
    if not budget_ok(400, f"sensor {sc}/{meth}/N={WIN}"):
        continue
    log_stage(f"identifikasi sensor | N={WIN} | {sc} | {meth}")
    keep = r["keep"]
    F = FEATURES[(WIN, meth)][keep]
    Ysens_k = SEGMENTS[WIN]["Ysens"][keep]
    rows, cost, s_pred, s_proba = sensor_stage(F, r["y"], Ysens_k, r["groups"], r["out"]["pred_oof"])
    if rows:
        PROBA[(WIN, sc, meth)] = {"proba": s_proba, "pred": s_pred, "Ysens": Ysens_k,
                                   "y": r["y"], "pred_oof": r["out"]["pred_oof"]}
    for row in rows:
        row.update({"N_window": WIN, "Skenario": sc, "Metode": meth})
        sensor_rows.append(row)
    if cost:
        sensor_cost_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                                  "cpu_s": round(cost["cpu_s"], 1), "wall_s": round(cost["wall_s"], 1),
                                  "peak_mem_mb": round(cost["peak_mem_mb"], 1),
                                  "ambang_rata2": cost["ambang_rata2"]})

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

cells.append(md("""# §10b — Probabilitas: seberapa besar kemungkinan tiap sensor rusak

§10 menjawab "ya/tidak". Bagian ini menjawab **"berapa persen kemungkinannya"**:
tiap window uji menghasilkan empat angka `P(S1 rusak) … P(S4 rusak)`, keluaran
`predict_proba` yang sudah dikalibrasi Platt di dalam data latih tiap fold.

Sebuah probabilitas hanya boleh disebut probabilitas kalau **terbukti kalibrasi**,
jadi dilaporkan tiga hal sekaligus:

| Ukuran | Artinya | Bagus kalau |
|---|---|---|
| **Brier score** | rata-rata kuadrat selisih probabilitas dengan kenyataan (0/1) | makin kecil |
| **Brier skill** | perbaikan terhadap tebakan konstan = prevalensi | > 0; ≤ 0 berarti tidak lebih baik daripada menyebut angka prevalensi saja |
| **ECE** | rata-rata jarak antara probabilitas yang diucapkan dan frekuensi sebenarnya, 10 bin | makin kecil (< 0,10 layak) |

`ROC_AUC` melengkapi: ia mengukur **urutan** (sensor yang lebih mungkin rusak
dapat angka lebih tinggi) walau kalibrasinya meleset.

**Cara membaca `P = 0,43`:** dari sekumpulan window yang diberi angka ±0,43,
kira-kira 43% memang sensornya rusak — itulah yang diperiksa diagram reliabilitas.
Angka ini **bukan** akurasi, dan **bukan** peluang seumur hidup sensor rusak;
ia peluang bersyarat pada potongan sinyal sepanjang N sampel yang sedang dinilai.
"""))

cells.append(code("""# === TABEL 4 — mutu probabilitas per sensor ===
def ece_score(yt, p, n_bins=10):
    \"\"\"Expected Calibration Error: |probabilitas yang diucapkan - frekuensi nyata|.\"\"\"
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += (m.sum() / len(p)) * abs(p[m].mean() - yt[m].mean())
    return e

prob_rows = []
for (WIN, sc, meth), d in PROBA.items():
    for mask, tag in ((d["y"] != 0, "oracle"), (d["pred_oof"] != 0, "end_to_end")):
        mask = mask & (d["pred"][:, 0] >= 0) & ~np.isnan(d["proba"][:, 0])
        if mask.sum() < 5:
            continue
        Ye, Pr = d["Ysens"][mask], d["proba"][mask]
        for j, sname in enumerate(SENSORS):
            yt, p = Ye[:, j].astype(int), Pr[:, j]
            prev = float(yt.mean())
            brier = float(np.mean((p - yt) ** 2))
            base_brier = prev * (1 - prev)                 # tebakan konstan = prevalensi
            skill = 1 - brier / base_brier if base_brier > 0 else np.nan
            try:
                auc = roc_auc_score(yt, p)
            except Exception:
                auc = np.nan
            prob_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth, "Eval": tag,
                               "Sensor": sname, "n_window": int(mask.sum()),
                               "Prevalensi": round(prev, 3),
                               "P_rusak_rata2": round(float(p.mean()), 3),
                               "P_rusak_p10": round(float(np.percentile(p, 10)), 3),
                               "P_rusak_p90": round(float(np.percentile(p, 90)), 3),
                               "Brier": round(float(brier), 4),
                               "Brier_skill": round(float(skill), 3) if skill == skill else np.nan,
                               "ECE": round(float(ece_score(yt, p)), 3),
                               "ROC_AUC": round(float(auc), 3) if auc == auc else np.nan})

prob_tbl = pd.DataFrame(prob_rows)
if len(prob_tbl):
    p_e2e = prob_tbl[prob_tbl.Eval == "end_to_end"]
    print("=== Probabilitas sensor rusak — mutu angkanya (end-to-end) ===")
    print(p_e2e.drop(columns=["Eval"]).to_string(index=False))
    export_df(prob_tbl, "07_probabilitas_sensor")
    print("\\nBaca: Brier_skill <= 0 berarti probabilitasnya belum lebih berguna "
          "daripada menyebut angka prevalensi; ECE > 0,10 berarti kalibrasinya masih meleset.")
else:
    print("Tidak ada probabilitas tersimpan (semua fold gagal / terlalu kecil).")
"""))

cells.append(code("""# === Diagram reliabilitas: yang diucapkan vs yang terjadi ===
if len(prob_tbl):
    # konfigurasi terbaik = ROC-AUC rata-rata 4 sensor tertinggi (end-to-end)
    _auc = (p_e2e.groupby(["N_window", "Skenario", "Metode"])["ROC_AUC"].mean().dropna())
    key = tuple(_auc.idxmax()) if len(_auc) else list(PROBA)[0]
    d = PROBA[key]
    m = (d["pred_oof"] != 0) & (d["pred"][:, 0] >= 0) & ~np.isnan(d["proba"][:, 0])
    Ye, Pr = d["Ysens"][m], d["proba"][m]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="kalibrasi sempurna")
    edges = np.linspace(0, 1, 11)
    for j, sname in enumerate(SENSORS):
        yt, p = Ye[:, j].astype(int), Pr[:, j]
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs, ys = [], []
        for b in range(10):
            mm = idx == b
            if mm.sum() >= 3:
                xs.append(p[mm].mean()); ys.append(yt[mm].mean())
        axes[0].plot(xs, ys, "o-", label=sname)
    axes[0].set_xlabel("P(rusak) yang dikeluarkan model")
    axes[0].set_ylabel("frekuensi rusak sebenarnya")
    axes[0].set_title(f"Diagram reliabilitas — N={key[0]} | {key[1]} | {key[2]}")
    axes[0].legend(fontsize=8); axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)

    axes[1].hist([Pr[:, j] for j in range(len(SENSORS))], bins=10, label=list(SENSORS))
    axes[1].set_xlabel("P(rusak)"); axes[1].set_ylabel("jumlah window")
    axes[1].set_title("Sebaran probabilitas yang dikeluarkan")
    axes[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("exports/07_reliabilitas_sensor.png", dpi=120); plt.show()
"""))

cells.append(code("""# === TABEL 5 — contoh keluaran per window: 4 probabilitas + kenyataannya ===
if len(prob_tbl):
    d = PROBA[key]
    m = np.where((d["pred_oof"] != 0) & (d["pred"][:, 0] >= 0) & ~np.isnan(d["proba"][:, 0]))[0]
    take = m[np.linspace(0, len(m) - 1, min(15, len(m))).astype(int)]
    ex = pd.DataFrame({"window": take})
    for j, s in enumerate(SENSORS):
        ex[f"P({s} rusak)"] = d["proba"][take, j].round(3)
    for j, s in enumerate(SENSORS):
        ex[f"benar_{s}"] = d["Ysens"][take, j].astype(int)
    ex["sensor_paling_mungkin"] = [SENSORS[i] for i in d["proba"][take].argmax(axis=1)]
    ex["sensor_rusak_sebenarnya"] = ["+".join([s for j, s in enumerate(SENSORS)
                                                if d["Ysens"][i, j]]) or "-" for i in take]
    print(f"=== Contoh keluaran per window — N={key[0]} | {key[1]} | {key[2]} (end-to-end) ===")
    print(ex.to_string(index=False))
    export_df(ex, "07_contoh_probabilitas_per_window")

    rank1 = np.mean([d["Ysens"][i, d["proba"][i].argmax()] for i in m])
    print(f"\\nSensor dengan probabilitas tertinggi memang rusak pada {rank1:.1%} window "
          f"(n={len(m)}). Pembanding acak = rata-rata prevalensi.")
"""))

cells.append(md("""# §11 — Dua uji kontrol yang diminta pembimbing

> *"misal dites pakai data yang kita kondisikan data benar (tidak ada fault)
> bisa ga vin, sebagai pembanding. atau data rusak hanya di salah satu sensor"*

**Bisa, dan memang harus.** Keduanya dijalankan di sini.

## Kontrol A — data yang dikondisikan bersih (tidak ada fault sama sekali)

Window `normal` **tidak disentuh injeksi apa pun**: sinyal apa adanya dari
keempat sensor. Yang diukur: seberapa sering model salah berteriak "fault"
padahal datanya bersih — **false alarm rate**.

Analogi: alarm kebakaran diuji di ruangan yang jelas-jelas tidak terbakar. Kalau
tetap berbunyi, alarmnya yang bermasalah, bukan ruangannya.

Dua angka dilaporkan:

| Ukuran | Artinya | Bagus kalau |
|---|---|---|
| `FPR` (false positive rate) | % window bersih yang divonis fault | makin kecil |
| `Spesifisitas` = 1 − FPR | % window bersih yang benar disebut normal | makin besar |

Ditambah pembanding penting: **rata-rata `P(sensor rusak)` pada window bersih
vs pada window fault**. Kalau kedua angka itu mirip, model tidak benar-benar
membedakan apa pun — persis kecurigaan pembimbing pada tabel sebelumnya.

## Kontrol B — fault hanya di **satu** sensor

Diambil hanya window yang **tepat satu** sensornya rusak, lalu ditanya: sensor
dengan `P(rusak)` tertinggi apakah memang sensor yang rusak? (`Top-1`).

Pembanding jujurnya **tebak acak = 25 %** (1 dari 4 sensor). Angka di bawah 25 %
berarti model lebih buruk daripada melempar dadu.
"""))

cells.append(code("""# === KONTROL A — data bersih: seberapa sering alarm palsu? ===
ctrl_rows = []
for (WIN, sc, meth), r in RUNS.items():
    y, pred = r["y"], r["out"]["pred_oof"]
    m_ok = (y == 0) & (pred >= 0)                 # window normal yang punya prediksi OOF
    m_bad = (y != 0) & (pred >= 0)
    if m_ok.sum() == 0:
        continue
    fpr = float((pred[m_ok] != 0).mean())
    rec_f = float((pred[m_bad] != 0).mean()) if m_bad.sum() else np.nan
    row = {"N_window": WIN, "Skenario": sc, "Metode": meth,
           "n_window_bersih": int(m_ok.sum()), "n_window_fault": int(m_bad.sum()),
           "FPR_alarm_palsu": round(fpr, 3), "Spesifisitas": round(1 - fpr, 3),
           "Recall_fault": round(rec_f, 3) if rec_f == rec_f else np.nan}
    d = PROBA.get((WIN, sc, meth))
    if d is not None:
        okp = m_ok & ~np.isnan(d["proba"][:, 0])
        badp = m_bad & ~np.isnan(d["proba"][:, 0])
        row["P_rusak_di_window_BERSIH"] = round(float(d["proba"][okp].mean()), 3) if okp.sum() else np.nan
        row["P_rusak_di_window_FAULT"] = round(float(d["proba"][badp].mean()), 3) if badp.sum() else np.nan
        row["Selisih"] = (round(row["P_rusak_di_window_FAULT"] - row["P_rusak_di_window_BERSIH"], 3)
                          if okp.sum() and badp.sum() else np.nan)
    ctrl_rows.append(row)

ctrl_tbl = pd.DataFrame(ctrl_rows)
print("=== KONTROL A — dites pakai data yang dikondisikan bersih (tanpa fault) ===")
print(ctrl_tbl.to_string(index=False))
export_df(ctrl_tbl, "07_kontrol_data_bersih")
print("\\nBaca: Spesifisitas = benar menyebut 'normal' pada data yang memang bersih.")
print("'Selisih' > 0 berarti P(rusak) memang naik saat sensornya benar-benar rusak; "
      "kalau ~0, model tidak membedakan apa pun.")
"""))

cells.append(code("""# === KONTROL B — fault hanya di satu sensor: sensor mana yang ditunjuk? ===
top1_rows = []
for (WIN, sc, meth), d in PROBA.items():
    m = ((d["Ysens"].sum(axis=1) == 1) & (d["pred"][:, 0] >= 0) & ~np.isnan(d["proba"][:, 0]))
    if m.sum() < 5:
        continue
    Pr, Ye = d["proba"][m], d["Ysens"][m]
    tebak = Pr.argmax(axis=1)
    benar = Ye.argmax(axis=1)
    top1 = float((tebak == benar).mean())
    # top-2: sensor yang benar ada di antara dua probabilitas tertinggi
    order = np.argsort(-Pr, axis=1)[:, :2]
    top2 = float(np.mean([benar[i] in order[i] for i in range(len(benar))]))
    top1_rows.append({"N_window": WIN, "Skenario": sc, "Metode": meth,
                       "n_window_1sensor": int(m.sum()),
                       "Top1_tepat": round(top1, 3), "Top2_tepat": round(top2, 3),
                       "Tebak_acak": 0.25,
                       "Lebih_baik_dari_acak": "ya" if top1 > 0.25 else "tidak"})

top1_tbl = pd.DataFrame(top1_rows)
if len(top1_tbl):
    print("=== KONTROL B — window dengan TEPAT satu sensor rusak ===")
    print(top1_tbl.to_string(index=False))
    export_df(top1_tbl, "07_kontrol_satu_sensor")
    print("\\nBaca: Top1 = sensor dengan P(rusak) tertinggi memang sensor yang rusak. "
          "Pembanding tebak acak = 0,25.")
else:
    print("Tidak ada window dengan tepat satu sensor rusak yang lolos ke tahap ini.")

# Tabel §10 versi 'satu_sensor' (per sensor, hanya window 1-sensor-rusak)
if "satu_sensor" in set(sensor_tbl.Eval):
    s1 = sensor_tbl[(sensor_tbl.Eval == "satu_sensor") & (sensor_tbl.Sensor != "SEMUA-4-BENAR")]
    print("\\n=== Rinci per sensor pada window 1-sensor-rusak ===")
    print(s1[["N_window", "Skenario", "Metode", "Sensor", "n_window", "Prevalensi",
              "Akurasi", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))
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
cols_final = ["N_window", "Skenario", "Metode", "C_kelas", "n_fitur", "Akurasi", "Precision",
              "Recall", "F1", "F1_std", "Hidden_terpilih", "cpu_s_total", "peak_mem_mb",
              "infer_ms_per_window", "F1_sensor", "Semua_4_benar"]
if len(prob_tbl):                       # mutu probabilitas ikut masuk ringkasan
    sprob = (prob_tbl[prob_tbl.Eval == "end_to_end"]
             .groupby(["N_window", "Skenario", "Metode"])[["Brier", "Brier_skill", "ECE", "ROC_AUC"]]
             .mean().round(3)
             .rename(columns={"Brier": "Brier_sensor", "Brier_skill": "Brier_skill_sensor",
                              "ECE": "ECE_sensor", "ROC_AUC": "AUC_sensor"}).reset_index())
    final = final.merge(sprob, on=["N_window", "Skenario", "Metode"], how="left")
    cols_final += ["AUC_sensor", "Brier_sensor", "Brier_skill_sensor", "ECE_sensor"]
final = final[cols_final].round(3)
print("=== RINGKASAN — diagram dijalankan penuh: performa, komputasi, sensor ===")
print(final.to_string(index=False))
export_df(final, "07_ringkasan_lengkap")
display(final)
log_stage("selesai")
"""))

cells.append(md("""# §13 — Tanya-Jawab (pertanyaan pembimbing, dijawab satu per satu)

---

### Q1. "Tabel ini artinya kemungkinan semua sensor mengandung data rusak dengan kemungkinan S1 = 0,429, S2 = 0,469, dst. Akurasi prediksi = 0,714, gitu ya?"

**Belum tepat — kolom `Prevalensi` bukan keluaran model.**

`Prevalensi = 0,429` artinya: **dari window uji yang dinilai, 42,9 % memang
sensor 1-nya rusak.** Itu **kunci jawaban**, bukan tebakan. Angkanya kita
ketahui persis karena fault-nya kita suntikkan sendiri. Gunanya sebagai
pembanding: model baru boleh disebut berguna kalau hasilnya mengalahkan
"asal tebak mengikuti prevalensi".

`Akurasi = 0,714` artinya: **71,4 % keputusan model untuk sensor 1 itu benar.**

Analogi ujian: `Prevalensi` = berapa persen soal yang kunci jawabannya "B";
`Akurasi` = berapa persen jawaban murid yang benar. Kalau 43 % kunci jawabannya
"B", murid yang menjawab "B" untuk semua soal langsung dapat 43 % — tanpa
belajar. Karena itu **akurasi harus selalu dibaca berdampingan dengan
prevalensi**.

Yang menjawab pertanyaan *"berapa kemungkinan sensor ini rusak"* adalah
**§10b**, kolom `P(Sx rusak)` — dihitung per window, bukan per tabel.

**Kenapa `ROC_AUC` wajib dilihat lebih dulu.** F1 tinggi bisa muncul semata-mata
karena prevalensi tinggi. Karena itu rancangan notebook ini memasang tiga
pengaman terhadap jawaban malas "semua sensor rusak":

1. **Subset sensor yang terkena fault diundi ulang `N_REPEAT_SUBSET` kali** per
   kondisi, sehingga tidak ada pasangan sensor yang selalu rusak bersamaan.
   Kalau label dua sensor sampai kembar, classifier per-sensor tidak punya apa
   pun untuk dibedakan dan `ROC_AUC` otomatis jatuh ke 0,5. **§5 punya sel
   pemeriksa khusus** yang mencetak korelasi antar-label dan memberi peringatan
   kalau kekembaran itu muncul.
2. **Fiturnya relatif** — tiap sensor dinilai dari selisihnya terhadap rata-rata
   tiga sensor lain (§10), jadi "semua rusak" bukan jawaban yang mudah.
3. **Ambang keputusan dicari di data latih**, bukan dipatok 0,5, supaya recall
   tidak digelembungkan.

**Cara membaca tabel §10 yang aman, tiga langkah:**
1. Lihat `ROC_AUC` dulu. ≈ 0,5 → berhenti, sisanya tidak usah dibaca.
2. Baru lihat `F1` bersama `Prevalensi`.
3. Untuk probabilitas, syaratkan `Brier_skill > 0` **dan** `ECE < 0,10`.

---

### Q2. "Bisa nggak dites pakai data yang kita kondisikan benar (tidak ada fault) sebagai pembanding? Atau data rusak hanya di salah satu sensor?"

**Bisa — keduanya sudah dijalankan di §11.**

- **Kontrol A** memakai window `normal` yang **tidak disuntik fault sama
  sekali**, dan melaporkan `FPR` (seberapa sering alarm palsu) serta
  `Spesifisitas`. Ditambah pembanding rata-rata `P(rusak)` pada window bersih
  vs window fault — kalau dua angka itu mirip, model tidak membedakan apa pun.
- **Kontrol B** memakai window yang **tepat satu** sensornya rusak, dan
  melaporkan `Top1` — apakah sensor dengan probabilitas tertinggi memang sensor
  yang rusak. Pembandingnya tebak acak 25 %.

---

### Q3. "Bedanya tabel *Ringkas: F1 rata-rata 4 sensor* dengan peta panas *F1 identifikasi sensor* apa?"

**Angkanya sama, tingkat rinciannya berbeda.** Peta panas menampilkan F1 **per
sensor** (empat kotak per baris); tabel ringkas adalah **rata-rata keempat kotak
itu** dalam satu baris.

Contoh dari hasil sebelumnya, baris `N | S1_Normal_vs_Faulty | JSD-Fuzzy`:
peta panas 0,93 · 0,89 · 0,87 · 0,90 → rata-ratanya 0,898, persis angka di
tabel ringkas. Jadi peta panas untuk melihat **sensor mana yang paling sulit**,
tabel ringkas untuk membandingkan **skenario dan metode**.

---

### Q4. "Data jadi di-preprocessing, ngambil data tiap jam kan?"

**Bukan tiap jam — tiap 5 menit**, dan bukan "diambil" melainkan
**dirata-rata**. Sepuluh pembacaan 30 detik dirangkum jadi satu angka
(281.721 baris → 28.173 baris). Bedanya penting: mengambil sampel ke-10 membuang
9 pembacaan; merata-rata memakai semuanya dan sekaligus meredam derau. Lihat §3.

---

### Q5. "Yang masuk ke broker itu data yang sudah di-sampling per 5 menit?"

**Bukan.** Urutannya mengikuti diagram:

```
sensor (30 detik) -> BROKER -> sinkronisasi waktu -> preprocessing 5 menit -> windowing
```

Broker menerima **data mentah 30 detik** dan tugasnya hanya **mengumpulkan** 4
aliran jadi satu tabel dengan identitas sensor tetap terpisah (§2). Perata-rataan
5 menit terjadi **sesudah** broker, sebagai langkah preprocessing (§3).

Kalau nanti di lapangan perata-rataan dipindah ke perangkat (sebelum broker),
hasilnya setara **asalkan** yang dikirim benar-benar rata-rata 5 menit, bukan
satu pembacaan sesaat tiap 5 menit.

---

### Q6. "Apakah di preprocessing ada hal lain, seperti validation dan cleaning?"

**Ada, sepuluh langkah, dan sekarang dicetak sebagai tabel di §3b**: cek kolom
wajib, urut waktu, buang stempel waktu ganda, cek NaN, cek nilai di luar rentang
fisik 0–100 %, cek pembacaan macet, cek pencilan ekstrem, rata-rata 5 menit,
paksa ke grid waktu seragam + tambal bolong, lalu jaminan bebas NaN.

Yang sengaja **tidak** dilakukan: membuang pencilan. Karena yang diteliti adalah
fault, membuang pembacaan aneh sebelum pemodelan sama dengan menghapus barang
bukti.

---

### Q7. "Length datanya bukan 2000, 7000, 10.000? Di perhitungan CV pakai 2000, 7000, 10.000."

**Dua-duanya benar, beda satuan.** Diagram menghitung sampel pada laju 30 detik;
notebook ini menghitung sampel setelah preprocessing 5 menit. **Durasinya sama
persis**: 0,69 / 2,43 / 3,47 hari. Tabel konversinya dicetak di §5.

Kalau 10.000 sampel dipaksa pada laju 5 menit, satu window = 34,7 hari,
sedangkan rekaman totalnya 97,8 hari → hanya 4 window untuk seluruh dataset,
dan cross-validation 5-fold tidak bisa jalan. Angkanya juga dicetak di §5.

Semua tabel hasil sekarang memuat kolom `N_diagram_30detik` di tabel konversi
supaya tidak ada lagi kebingungan satuan.

---

### Q8. "Kalau ada window yang tidak cukup panjang, apakah window itu di-exclude?"

**Tidak ada window yang setengah jadi.** Pembentuk window hanya menghasilkan
window yang **genap** N sampel; sisa ekor rekaman yang kurang dari N sampel
tidak dijadikan window sama sekali. Jumlahnya dicetak di blok akuntansi §5.

Angka **256** (dan di notebook ini **128**) adalah `n_ref` — **banyaknya vektor
acuan** yang diambil acak saat menghitung kemiripan fuzzy, supaya biaya
hitungnya tidak O(N²). Kalau titik yang tersedia lebih sedikit dari `n_ref`,
dipakai **semuanya**; window tetap ikut dan **tidak dibuang**.

Satu-satunya penyaringan window: pada kondisi fault, window yang porsi sampel
ter-fault-nya ≤ 1 % dibuang, karena melabelinya "fault" akan menyesatkan.

---

## Rujukan

| Topik | Rujukan |
|---|---|
| Fuzzy entropy | Chen W., Wang Z., Xie H., Yu W. (2007). *Characterization of surface EMG signal based on fuzzy entropy.* IEEE Trans. Neural Syst. Rehabil. Eng. 15(2), 266–272 |
| Multiscale entropy (coarse-graining τ) | Costa M., Goldberger A.L., Peng C.-K. (2002). *Multiscale entropy analysis of complex physiologic time series.* Phys. Rev. Lett. 89(6), 068102 |
| Jensen–Shannon divergence | Lin J. (1991). *Divergence measures based on the Shannon entropy.* IEEE Trans. Inf. Theory 37(1), 145–151 |
| Kalibrasi probabilitas (Platt scaling) | Platt J. (1999). *Probabilistic outputs for support vector machines.* Advances in Large Margin Classifiers, 61–74 |
| Kenapa kalibrasi wajib diuji | Niculescu-Mizil A., Caruana R. (2005). *Predicting good probabilities with supervised learning.* ICML |
| Brier score | Brier G.W. (1950). *Verification of forecasts expressed in terms of probability.* Monthly Weather Review 78(1), 1–3 |
| Bahaya prevalensi tinggi pada F1 | Saito T., Rehmsmeier M. (2015). *The precision–recall plot is more informative than the ROC plot on imbalanced datasets.* PLoS ONE 10(3), e0118432 |
| Kebocoran data pada CV deret waktu | Bergmeir C., Benítez J.M. (2012). *On the use of cross-validation for time series predictor evaluation.* Information Sciences 191, 192–213 |
| Kebocoran akibat window tumpang-tindih | Hammerla N.Y., Plötz T. (2015). *Let's (not) stick together: pairwise similarity biases cross-validation in activity recognition.* UbiComp |
| Mutu data (validasi & pembersihan) | Batini C., Scannapieco M. (2016). *Data Quality: Concepts, Methodologies and Techniques.* Springer |
| Deteksi pencilan pada deret sensor | Hodge V.J., Austin J. (2004). *A survey of outlier detection methodologies.* Artificial Intelligence Review 22(2), 85–126 |
| Levenberg–Marquardt untuk ANN | Hagan M.T., Menhaj M.B. (1994). *Training feedforward networks with the Marquardt algorithm.* IEEE Trans. Neural Networks 5(6), 989–993 |
"""))

cells.append(md("""## Ringkasan — apa yang dibuktikan notebook ini

1. **Skema di flowchart dijalankan utuh, kotak per kotak**: akuisisi → broker →
   sinkronisasi waktu (rata-rata 5 menit) → injeksi fault → segmentasi (N ∈ {200;
   700; 1000} sampel = 0,69 / 2,43 / 3,47 hari, durasi sama dengan diagram) →
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
6. **Keluarannya bukan cuma ya/tidak, tapi probabilitas.** Tiap window
   menghasilkan `P(S1 rusak) … P(S4 rusak)`, dikalibrasi Platt di dalam data
   latih tiap fold, lalu diuji kejujurannya lewat Brier score, Brier skill,
   ECE, dan diagram reliabilitas (§10b).

**Cara baca angkanya:** utamakan **F1 macro** — jumlah kelas C berbeda antar
skenario sehingga akurasi tidak sebanding lintas skenario. Untuk §10, baca `F1`
bersama `Prevalensi` dan `ROC_AUC`. Untuk §10b, sebuah probabilitas baru boleh
dikutip kalau `Brier_skill > 0` **dan** `ECE` kecil; kalau tidak, angkanya hanya
skor urutan, bukan peluang.

**Dua penyimpangan dari diagram, disengaja dan dicatat:** `solver='lbfgs'`
sebagai pengganti Levenberg–Marquardt (sklearn tidak punya LM), dan JSD-Fuzzy
ikut dijalankan sebagai pembanding usulan paper walau diagram hanya menyebut
EDM-Fuzzy.

7. **Dua uji kontrol dijalankan** (§11): data yang dikondisikan **bersih**
   (false alarm rate) dan window dengan **tepat satu sensor rusak** (Top-1
   identifikasi vs tebak acak 25 %).
8. **Preprocessing dilaporkan, bukan diklaim** (§3b): sepuluh langkah validasi
   dan pembersihan beserta angkanya, dan pernyataan tegas bahwa pencilan
   dilaporkan tetapi tidak dibuang.

**File keluaran** di folder `exports/`: `07_performa_cv_5skenario.csv`,
`07_komputasi_cv_5skenario.csv`, `07_ongkos_ekstraksi_fitur.csv`,
`07_arsitektur_terpilih.csv`, `07_identifikasi_sensor.csv`,
`07_probabilitas_sensor.csv`, `07_contoh_probabilitas_per_window.csv`,
`07_ringkasan_lengkap.csv`, `07_mutu_data.csv`, `07_konversi_window.csv`,
`07_akuntansi_window.csv`, `07_kontrol_data_bersih.csv`,
`07_kontrol_satu_sensor.csv`, plus `07_*.png` (termasuk
`07_reliabilitas_sensor.png`).
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.9"}},
      "nbformat": 4, "nbformat_minor": 5}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "| cells:", len(cells))
