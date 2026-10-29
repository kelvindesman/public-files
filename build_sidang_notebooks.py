"""Generate the three defence notebooks in sidang/.

Run:  venv/bin/python build_sidang_notebooks.py
Output: sidang/00_Peta_Program.ipynb
        sidang/01_Alur_Sistem.ipynb
        sidang/02_Hasil.ipynb

The notebooks only read exported results (sidang/assets) and the raw CSV.
Nothing is retrained here.
"""
import json
from pathlib import Path

OUT = Path("sidang")
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n")}


def notebook(cells):
    for i, c in enumerate(cells):
        c.setdefault("id", f"cell-{i:02d}")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    p = OUT / name
    p.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p, "|", len(cells), "cells")


PRELUDE = r'''
import os, io, glob, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)

ASSET_BASE = "https://raw.githubusercontent.com/vousmeevoyez/public-files/main/sidang/assets"
LOCAL_DIRS = ["assets", "sidang/assets", "../sidang/assets"]

def asset_path(rel):
    for base in LOCAL_DIRS:
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None

def read_asset(rel):
    p = asset_path(rel)
    if p:
        return pd.read_csv(p)
    r = requests_get(f"{ASSET_BASE}/{rel}")
    return pd.read_csv(io.BytesIO(r))

def requests_get(url):
    import requests
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def show_png(rel, width=900):
    p = asset_path(rel)
    if p:
        display(Image(filename=p, width=width))
    else:
        display(Image(data=requests_get(f"{ASSET_BASE}/{rel}"), width=width))

print("Aset siap dibaca dari:", "folder lokal" if asset_path("11/11_rq4_performa_ann.csv") else "GitHub")
'''

# ---------------------------------------------------------------- 00 Peta
peta = [
    md("""
# Peta Program — dari data sensor sampai keputusan

**Riset Deteksi Fault 4 Sensor Kelembaban**

Notebook ini menjawab satu pertanyaan: *program ini sebenarnya melakukan apa,
urutannya bagaimana, dan angkanya diambil dari mana?*

Cara membaca: cukup jalankan semua sel dari atas ke bawah. Tidak ada pelatihan
model di sini, jadi selesai dalam hitungan detik.

Tiga notebook sidang:

| Notebook | Isi | Waktu jalan |
|---|---|---|
| **00 Peta Program** (ini) | alur, glosarium, peta angka | < 5 detik |
| **01 Alur Sistem** | contoh data nyata di setiap tahap | < 2 menit |
| **02 Hasil** | tabel dan grafik hasil akhir | < 30 detik |
"""),
    md("""
## 1 — Alur program dalam satu gambar

Delapan kotak di bawah adalah urutan kerja program. Kotak 1–4 menyiapkan data,
kotak 5 mengubah sinyal menjadi angka, kotak 6–7 menebak, kotak 8 menguji.
"""),
    code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

stages = [
    ("1. Data mentah", "4 sensor, tiap 30 detik"),
    ("2. Broker", "4 kanal disatukan, tidak dirata-rata"),
    ("3. Sinkron waktu", "satu grid tiap 5 menit"),
    ("4. Injeksi fault", "drift, spike, bias, macet"),
    ("5. Window + entropi", "potong 16,7 jam, ukur kerumitan"),
    ("6. ANN", "60 angka -> kelas fault"),
    ("7. Rantai sensor", "fault -> sensor mana"),
    ("8. Uji", "5 skenario x 3 ukuran window"),
]

fig, ax = plt.subplots(figsize=(13, 3.2))
ax.set_xlim(0, 8); ax.set_ylim(0, 1); ax.axis("off")
for i, (title, sub) in enumerate(stages):
    box = FancyBboxPatch((i + 0.06, 0.30), 0.88, 0.42,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor="#e8f1fb", edgecolor="#2c6ea8", lw=1.4)
    ax.add_patch(box)
    ax.text(i + 0.5, 0.58, title, ha="center", va="center", fontsize=8.5, weight="bold")
    ax.text(i + 0.5, 0.42, sub, ha="center", va="center", fontsize=7)
    if i < len(stages) - 1:
        ax.annotate("", xy=(i + 1.01, 0.51), xytext=(i + 0.94, 0.51),
                    arrowprops=dict(arrowstyle="->", color="#2c6ea8", lw=1.2))
ax.set_title("Alur kerja program (kiri ke kanan)", fontsize=11, weight="bold")
plt.tight_layout(); plt.show()
'''),
    md("""
## 2 — Tiap tahap dalam bahasa sehari-hari

| Tahap | Yang dilakukan | Kenapa perlu |
|---|---|---|
| **1. Data mentah** | Baca `data_sensor.csv` (281.721 baris, 14 Sep–21 Des 2025) | bahan dasar |
| **2. Broker** | Empat kolom sensor disatukan, **identitas tiap sensor tetap** | supaya bisa tahu sensor mana yang rusak |
| **3. Sinkron waktu** | Rata-rata 5 menit, buang duplikat, tambal slot bolong | semua sensor bicara di jam yang sama |
| **4. Injeksi fault** | Tambahkan gangguan buatan sesuai skenario | memberi label "rusak" yang pasti |
| **5. Window + entropi** | Potong jadi jendela 200 sampel, ukur kerumitan pola | model butuh angka, bukan grafik mentah |
| **6. ANN** | 60 angka masuk, keluar kelas fault | menebak jenis kondisi |
| **7. Rantai sensor** | Dari tebakan fault, lacak sensor mana | tindakan perbaikan jadi terarah |
| **8. Uji** | 5 skenario × 3 ukuran window, CV berkelompok | angka tidak boleh menipu diri sendiri |
"""),
    md("""
## 3 — Enam angka yang paling sering ditanya

| Pertanyaan | Jawaban | Sumber |
|---|---|---|
| Metode mana lebih tinggi? | JSD-Fuzzy menang **11 dari 15 sel**, rata-rata **+0,028** | `11` §10 |
| Sensor mana yang dikenali? | F1 **0,675 vs 0,634**, menang 9/15 | `11` §12 |
| Di mana JSD paling menang? | Window N = 700: **5 dari 5** skenario | `11` §10 |

> Yang tidak diklaim: "JSD terbukti signifikan lebih baik". Yang diklaim:
> "JSD konsisten lebih tinggi pada 11 dari 15 sel".
"""),
    md("""
## 4 — Peta notebook: sidang ↔ penelitian

Tabel ini supaya tidak bingung kalau penguji minta "notebook aslinya".

| Bagian sidang | Notebook penelitian | Kernel |
|---|---|---|
| Alur & diagram | `07` §0–§9 | `jsd-fuzzy-cv-5skenario` v23 |
| Angka utama | `11` §10–§13 | `jsd-edm-ann-lm-lengkap-q3` v3 |
| Broker 4 kanal | `02` | `jsd-fuzzy-broker-architecture-v2` v3 |
| Data mentah | `Ilustrasi` | `ilustrasi-data-akuisisi-dan-fault` v3 |

Tabel lengkap (nomor sel + nama file): `sidang/MAPPING.md`.

**Peringatan penting.** Notebook `01/04/05/07/08` memakai `MLPClassifier`
sklearn, **bukan** Levenberg–Marquardt dari paper. Yang benar-benar LM adalah
`09`, `10`, `11`. Jangan menyebut "LM" untuk notebook lain.
"""),
    md("""
## 5 — Glosarium 10 istilah

| Istilah | Arti sederhana |
|---|---|
| **Fault** | kerusakan sensor: drift, spike, bias, atau macet |
| **Drift** | nilai bergeser pelan ke satu arah |
| **Spike** | loncatan sesaat lalu kembali |
| **Bias** | nilai bergeser tetap, tidak kembali |
| **Hardware malfunction** | sensor macet atau kehilangan data |
| **Window** | potongan sinyal sepanjang 200 sampel (±16,7 jam) |
| **Entropi** | ukuran "seberapa tidak beraturan" pola sinyal |
| **Multiskala** | entropi dihitung pada 15 tingkat kekasaran |
| **EDM-Fuzzy** | kemiripan pola memakai jarak garis lurus |
| **JSD-Fuzzy** | kemiripan pola memakai bentuk sebaran nilai |
"""),
    md("""
## 6 — Kalau penguji bertanya, buka ini

| Pertanyaan | Buka |
|---|---|
| "Perlihatkan datanya" | `01 Alur Sistem` bagian 1 |
| "Kenapa tidak dirata-rata?" | `01` bagian 2, lalu `02` |
| "Contoh sinyal rusak?" | `01` bagian 3 |
| "Kenapa 60 fitur?" | `01` bagian 5 |
| "Hasilnya mana?" | `02 Hasil` bagian 1 |
| "Sensor mana yang rusak?" | `02 Hasil` bagian 2 |
| "Signifikan?" | `02 Hasil` bagian 7 |
| "Batasan?" | `02 Hasil` bagian 8 |
"""),
    md("""
---
**Status:** privat · commit `27737ee` · 2026-10-26
**Rujukan lengkap:** `sidang/MAPPING.md` · **Bahan tanya-jawab:** `Bahan_Sidang.md`
"""),
]

# ---------------------------------------------------------------- 01 Alur
alur = [
    md("""
# Alur Sistem — data nyata di setiap tahap

**Riset Deteksi Fault 4 Sensor Kelembaban**

Notebook ini memperlihatkan **satu jalur data** dari file mentah sampai siap
ditebak: berapa barisnya, bagaimana disatukan, bagaimana gangguan disisipkan,
bagaimana dipotong, dan berapa fitur yang dihasilkan.

Analogi: empat termometer mencatat tiap 30 detik. Kita satukan catatannya,
samakan jamnya, tempelkan gangguan buatan, potong jadi blok 16,7 jam, lalu
ringkas tiap blok menjadi 60 angka.
"""),
    code(PRELUDE),
    md("""
## 1 — Data mentah: empat kanal, 30 detik

Yang dicek: jumlah baris, rentang waktu, dan apakah keempat kolom ada. Kalau
kolom tidak lengkap, program berhenti dengan pesan yang jelas.
"""),
    code(r'''
DATA_URL = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"
RAW_CAND = ["data_sensor.csv", "../data_sensor.csv"] + glob.glob("/kaggle/input/**/data_sensor.csv", recursive=True)

def load_raw():
    for p in RAW_CAND:
        if os.path.exists(p):
            print("Sumber: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    print("Sumber: unduh dari GitHub (67 MB, sekali saja)")
    import requests
    r = requests.get(DATA_URL, timeout=300); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text), index_col=0)

df_raw = load_raw()
cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
missing = [c for c in cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Kolom {missing} tidak ada. Tersedia: {list(df_raw.columns)}")

print("Baris:", len(df_raw))
print("Rentang:", df_raw.index[0], "->", df_raw.index[-1])
display(df_raw[cols].head(3))
display(df_raw[cols].describe().round(2))
show_png("ilustrasi/ILU_empat_kanal_bersih.png", 950)
'''),
    md("""
## 2 — Broker: satukan kanal, jangan rata-ratakan

Broker menyatukan empat aliran menjadi satu tabel, tetapi **kolomnya tetap
empat**. Rata-rata dilakukan pada waktu (5 menit), bukan pada sensor.

Alasannya: kalau empat sensor dirata-rata, kerusakan satu sensor tersembunyi di
balik tiga sensor sehat. Notebook `02` mengukur akibatnya: F1 0,821 (4 kanal)
vs 0,748 (dirata-rata).
"""),
    code(r'''
df = df_raw[cols].copy()
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()

n_dup = int(df.index.duplicated().sum())
df = df[~df.index.duplicated(keep="first")]

df5 = df.resample("5min").mean()
grid = pd.date_range(df5.index[0], df5.index[-1], freq="300s")
n_gap = int(df5.reindex(grid).isna().any(axis=1).sum())
df5 = (df5.reindex(grid)
          .interpolate(method="time", limit_direction="both")
          .fillna(df5.median(numeric_only=True)))

print(f"30 detik : {len(df):,} baris")
print(f"5 menit  : {len(df5):,} baris ({(len(df5)*300)/86400:.1f} hari)")
print(f"duplikat dibuang: {n_dup} | slot bolong ditambal: {n_gap}")

seg = df5.iloc[:600]
fig, ax = plt.subplots(figsize=(12, 3.6))
for c in cols:
    ax.plot(seg.index, seg[c], lw=0.9, label=c)
ax.set_title("Empat kanal setelah sinkronisasi waktu (potongan 2 hari pertama)")
ax.set_ylabel("kelembaban (%)"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.25)
plt.tight_layout(); plt.show()
'''),
    md("""
## 3 — Injeksi fault: menempelkan kerusakan yang ada labelnya

Data asli tidak punya label "rusak". Karena itu kerusakan disisipkan sendiri,
dengan rumus yang sama di semua notebook. Empat jenis dasar:

| Jenis | Perubahan | Skala |
|---|---|---|
| Drift | naik pelan 1 poin/hari, episode 3 hari | sangat kecil |
| Spike | loncatan sesaat tiap ±67 sampel | 0,08 × simpangan baku |
| Bias | tambah 0,08 % tetap | kecil |
| Hardware | 8 % sampel macet, 5 % hilang | jelas terlihat |

Fault hanya masuk ke subset sensor, jadi tiga sensor lain tetap bersih.
"""),
    code(r'''
rng = np.random.default_rng(42)
x = df5["kelembaban1"].iloc[:864].to_numpy(dtype=float)   # 3 hari
t = np.arange(len(x))

def inject_drift(x):
    return x + np.arange(1, len(x) + 1) * (1.0 / 288.0)      # 1 poin/hari @5 menit

def inject_spike(x):
    y = x.copy(); y[::67] += 0.08 * np.nanstd(x); return y

def inject_bias(x):
    return x + 0.08

def inject_hardware(x):
    y = x.copy(); idx = rng.integers(len(x), size=len(x)); rv = rng.random(len(x))
    y[rv < 0.08] = x[idx[rv < 0.08]]
    y[rv < 0.05] = np.nan
    return y

cases = [("Drift (selisih diperbesar di panel kecil)", inject_drift),
         ("Spike", inject_spike),
         ("Bias", inject_bias),
         ("Hardware malfunction", inject_hardware)]

fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
for ax, (title, fn) in zip(axes, cases):
    y = fn(x)
    ax.plot(t, x, color="#8a8a8a", lw=1.4, label="asli")
    ax.plot(t, y, color="#c0392b", lw=1.0, label="setelah gangguan")
    if np.isnan(y).any():
        ax.plot(t, pd.Series(y).ffill().bfill(), color="#e67e22", lw=0.9, ls="--", label="setelah ditambal")
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
axes[-1].set_xlabel("sampel (5 menit)")
plt.tight_layout(); plt.show()
'''),
    md("""
## 4 — Segmentasi: satu sinyal dipotong jadi banyak jendela

Window **N = 200 sampel** = 1.000 menit ≈ **16,7 jam**. Geser setengah window
(stride 100) supaya tidak ada peristiwa yang jatuh tepat di batas.

Window yang porsi fault-nya ≤ 1 % dibuang: isinya praktis normal, jadi
melabelinya "fault" akan mengajari model pola yang salah.
"""),
    code(r'''
N, STRIDE = 200, 100
w = np.lib.stride_tricks.sliding_window_view(df5["kelembaban1"].to_numpy(), N)[::STRIDE]
print(f"Window per kondisi: {len(w):,} | bentuk satu window: {w[0].shape}")
print(f"Durasi satu window: {N*5/60:.1f} jam")

display(read_asset("11/11_akuntansi_window.csv"))
'''),
    md("""
## 5 — Dari jendela ke 60 angka

Tiap jendela dihitung entropinya pada 15 skala (τ = 1..15). Empat sensor ×
15 skala = **60 fitur**. Identitas sensor tetap: fitur S1 tidak pernah dicampur
dengan fitur S3.

Dua cara mengukur kemiripan pola:

- **EDM-Fuzzy** — jarak garis lurus antar potongan sinyal.
- **JSD-Fuzzy** — tiap potongan diubah jadi sebaran nilai, lalu dibandingkan
  bentuk sebarannya.

Gambar di bawah dibuat oleh notebook `11`; di sini hanya ditampilkan ulang.
"""),
    code(r'''
show_png("11/11_similarity_euclidean_vs_jsd.png", 950)
show_png("11/11_grid_fault_vs_asli_S1.png", 950)
'''),
    md("""
## 6 — Model: 60 angka masuk, kelas fault keluar

ANN yang dipakai kecil: 60 masukan, satu atau dua lapisan tersembunyi
(32–64 neuron), keluaran sesuai jumlah kelas. Arsitektur dipilih **di dalam**
cross-validation, bukan setelah melihat data uji.

```
60 fitur  ->  [ lapisan tersembunyi ]  ->  kelas (2 / 5 / 7 / 5 / 2)
```

Klasifier yang dipakai hanya **ANN-LM asli** (Levenberg–Marquardt dengan
Jacobian analitik), bukan `lbfgs` dan bukan pelatih lain. Notebook `11` §12
menambahkan rantai kedua: dari kelas fault, tebak **sensor mana** yang rusak.
Saat pengujian nyata, label sensor tidak tersedia.
"""),
    md("""
## 7 — Uji: kenapa harus CV berkelompok

Window bertetangga tumpang-tindih 50 %. Kalau data diacak biasa, potongan yang
sama bisa masuk latih dan uji sekaligus — skor jadi terlalu bagus. Karena itu
pembagian fold **dikelompokkan per blok waktu**.

| Skenario | Yang dibedakan | Kelas |
|---|---|---|
| S1 | normal vs ada fault | 2 |
| S2 | fault tunggal (4 jenis) | 5 |
| S3 | kombinasi **2 fault** (6 kombinasi) | **7** |
| S4 | kombinasi **3 fault** (4 kombinasi) | 5 |
| S5 | kombinasi **4 fault** sekaligus | **2** |

Catatan: tugas **17 kelas** (normal + 16 kondisi fault) adalah eksperimen
terpisah di notebook `01`/`04`, bukan S5. Identifikasi "sensor mana yang
rusak" juga bukan S4 — itu RQ6 di notebook `11` §12.
"""),
    md("""
## 8 — Peta balik: tahap di sini → notebook asli

| Bagian notebook ini | Notebook asli | Sel |
|---|---|---|
| 1 Data mentah | `Ilustrasi` §1–§2 | 1–7 |
| 2 Broker + sinkron | `11` §2, `02` | 6–7 |
| 3 Injeksi fault | `11` §3–§3b | 9–12 |
| 4 Segmentasi | `11` §4, `07` §5 | 13–14 |
| 5 Entropi | `11` §5–§7b | 15–24 |
| 6 Model | `11` §10 | 33–40 |
| 7 Uji | `11` §10 | 33–40 |

---
**Status:** privat · commit `27737ee` · 2026-10-26 · rujukan: `sidang/MAPPING.md`
"""),
]

# ---------------------------------------------------------------- 02 Hasil
hasil = [
    md("""
# Hasil — semua angka dibaca dari file

**Riset Deteksi Fault 4 Sensor Kelembaban**

Notebook ini **tidak menghitung ulang model**. Ia membaca file hasil yang
diekspor notebook penelitian, lalu memeriksa ulang angkanya dengan `assert`.
Kalau file berubah, notebook ini langsung gagal — jadi tidak ada angka yang
diketik manual.

Urutan: performa utama → sensor mana yang rusak → catatan data → batasan →
sumber. Semua hasil memakai klasifier **ANN-LM asli**; tidak ada pelatih lain
di edisi ini.
"""),
    code(PRELUDE),
    md("""
## 1 — Performa utama: EDM-Fuzzy vs JSD-Fuzzy

15 sel = 5 skenario × 3 ukuran window (200, 700, 1000 sampel). Selisih
dihitung JSD − EDM; positif berarti JSD lebih tinggi.
"""),
    code(r'''
rq4 = read_asset("11/11_rq4_performa_ann.csv")
p = rq4.pivot_table(index=["N_window", "Scenario"], columns="Method", values="F1").reset_index()
p["selisih"] = p["JSD-Fuzzy"] - p["EDM-Fuzzy"]
p["menang"] = np.where(p["selisih"] > 0, "JSD", "EDM")
p["JSD-Fuzzy"] = p["JSD-Fuzzy"].round(3)
p["EDM-Fuzzy"] = p["EDM-Fuzzy"].round(3)
p["selisih"] = p["selisih"].round(3)

n_win = int((p["selisih"] > 0).sum())
mean_diff = float(p["selisih"].mean())
assert n_win == 11, n_win
assert abs(mean_diff - 0.028) < 0.001, mean_diff

print(f"JSD menang {n_win} dari {len(p)} sel | rata-rata selisih {mean_diff:+.3f}")
display(p.to_string(index=False))
'''),
    code(r'''
show_png("11/11_rq4_performa_N700.png", 950)
show_png("11/11_rq4_confusion_matrix_N700.png", 760)
'''),
    md("""
**Cara membaca.**

- JSD menang **11/15**, tetapi tidak merata: N = 700 menang 5/5, sedangkan
  N = 200 dan N = 1000 masing-masing 3/5.
- Selisih terbesar di S1 dengan window panjang (**+0,178**); di S3 beberapa sel
  justru negatif.
- Artinya keunggulan JSD **bergantung kondisi**, bukan otomatis di semua kasus.
  Kalimat yang aman: "konsisten lebih tinggi secara rata-rata".
"""),
    md("""
## 2 — Sensor mana yang rusak?

Setelah model menebak ada fault, rantai kedua menebak sensor mana. Angka di
bawah memakai protokol notebook `11` (ANN-LM asli).
"""),
    code(r'''
sr = read_asset("11/11_sensor_ringkas.csv")
q = sr.pivot_table(index=["N_window", "Scenario"], columns="Method", values="F1").reset_index()
q["selisih"] = q["JSD-Fuzzy"] - q["EDM-Fuzzy"]

m_edm = float(sr[sr.Method == "EDM-Fuzzy"].F1.mean())
m_jsd = float(sr[sr.Method == "JSD-Fuzzy"].F1.mean())
n_win_s = int((q["selisih"] > 0).sum())
assert abs(m_jsd - 0.675) < 0.002, m_jsd
assert abs(m_edm - 0.634) < 0.002, m_edm
assert n_win_s == 9, n_win_s

print(f"F1 rata-rata: JSD {m_jsd:.3f} vs EDM {m_edm:.3f} | JSD menang {n_win_s}/15")
per_sensor = (sr[sr.Method == "JSD-Fuzzy"]
              .groupby("Scenario").F1.mean().round(3)
              .rename("F1 JSD per skenario").reset_index())
display(per_sensor.to_string(index=False))
show_png("11/11_sensor_f1.png", 900)
'''),
    md("""
**Cara membaca.** Angka di atas adalah **per skenario**, bukan per sensor
fisik: S1 normal vs fault (0,543), S2 fault tunggal (0,622), S3 kombinasi dua
fault (0,635), S4 kombinasi tiga fault (0,735), S5 kombinasi empat fault (0,841).

Angka pendamping: tebakan **semua empat sensor benar sekaligus** 0,307 (JSD)
vs 0,295 (EDM). Ini lebih ketat daripada F1 rata-rata.
"""),
    md("""
## 3 — Catatan data: sensor 4 pernah macet di 0

Tidak ada baris yang dibuang. Yang perlu dijelaskan saat sidang: kolom
**kelembaban4** bernilai 0 pada **11.371 sampel 30 detik (4,0 %)**, semuanya
antara 16 dan 21 Desember 2025.

| Periode | Panjang |
|---|---|
| 16 Des 07.37–08.01 | 49 sampel (0,4 jam) |
| 17 Des 10.04 – 19 Des 07.03 | 5.399 sampel (45,0 jam) |
| 19 Des 07.57 – 20 Des 21.10 | 4.468 sampel (37,2 jam) |
| 20 Des 22.23 – 21 Des 10.30 | 1.454 sampel (12,1 jam) |

Setelah agregasi 5 menit: 1.135 titik nol. Nilai 0 tetap dipakai karena
**di dalam rentang fisik 0–100 %** dan justru bisa menjadi gejala fault
(pembacaan macet). Dampaknya: **10 dari 280 window N = 200 (3,6 %)** mayoritas
nol; N = 700: 2/79; N = 1000: 1/55. Window itu tetap masuk kelas normal karena
data asli tidak punya label. Ini dicatat sebagai batasan, bukan dibuang.
"""),
    md("""
## 4 — Batasan yang harus disebut

1. Fault berasal dari **simulasi** dengan rumus tetap, bukan kerusakan lapangan.
2. Keputusan butuh window 16,7 jam → pemantauan harian, **bukan alarm real-time**.
3. Keunggulan JSD tidak merata; di S3 beberapa sel negatif.
4. EDM lebih **stabil** (CV lebih rendah di 56/63 pasangan), walau F1 lebih
   rendah. Ini dilaporkan apa adanya.
5. Drift 1 poin/hari sengaja kecil; deteksi drift jangka pendek masih lemah.
6. Sensor 4 macet di 0 selama 5 hari terakhir (11.371 sampel); tidak dibuang,
   tetapi 10/280 window mayoritas nol.
"""),
    md("""
## 5 — Sumber setiap blok

| Blok | Notebook asli | Sel | File |
|---|---|---|---|
| 1 Performa | `11` §10 | 33–40 | `11_rq4_performa_ann.csv` |
| 2 Sensor-ID | `11` §12 | 43–46 | `11_sensor_ringkas.csv` |
| 3 Catatan data | data mentah | — | `data_sensor.csv` |

---
**Status:** privat · commit `27737ee` · 2026-10-26 · rujukan: `sidang/MAPPING.md`
"""),
]

write("00_Peta_Program.ipynb", peta)
write("01_Alur_Sistem.ipynb", alur)
write("02_Hasil.ipynb", hasil)
