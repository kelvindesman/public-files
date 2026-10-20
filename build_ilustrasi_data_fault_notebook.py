#!/usr/bin/env python3
"""Build Ilustrasi_Data_Akuisisi_dan_Fault.ipynb.

Notebook ringkas untuk menjawab dua permintaan pembimbing (20/10/2025):
  1. penjelasan 10 langkah akuisisi data dan nasib missing value /
     nilai di luar range;
  2. grafik sinyal asli vs setelah injeksi fault untuk semua kelas.

Tanpa pelatihan model — hanya pembersihan data + simulasi fault, supaya
cepat dijalankan (beberapa menit) dan mudah diperiksa.
"""
import json
import notebook_fault_figures as nff

OUT = "/Users/kelvin/apps/public-files/Ilustrasi_Data_Akuisisi_dan_Fault.ipynb"
DATA_URL = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

cells.append(md("""# Ilustrasi Data Akuisisi dan Fault

Notebook ini menjawab dua pertanyaan pembimbing:

1. **Akuisisi data** — apa saja 10 langkah yang dilakukan, dan apa yang terjadi
   pada *missing value* serta nilai di luar rentang 0–100 %?
2. **Grafik** — bagaimana bentuk sinyal asli dibanding setelah disisipi fault,
   untuk semua kelas gangguan?

Isinya sengaja dibuat ringan: hanya pembersihan data dan simulasi fault, tanpa
pelatihan model. Data yang dipakai sama dengan notebook penelitian lainnya
(`data_sensor.csv`, 281.721 baris, 14 Sep–21 Des 2025, interval 30 detik).
"""))

cells.append(md("""## 1 — Memuat dan memeriksa data mentah

Sepuluh langkah di bawah ini adalah urutan yang sama dengan pipeline penelitian.
Kolom **Tindakan** menunjukkan mana yang membuang baris dan mana yang hanya
melaporkan.
"""))

cells.append(code("""import os, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

RANDOM_SEED = 42
SAMPLING_SECONDS = 300          # setelah rata-rata per 5 menit
RESAMPLE_RULE = "5min"
cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
SENSORS = ["S1", "S2", "S3", "S4"]
os.makedirs("exports", exist_ok=True)

DATA_URL = "%s"
DATA_NAME = "data_sensor.csv"

def load_sensor_data():
    cands = [DATA_NAME, f"../input/{DATA_NAME}"] + glob.glob(f"/kaggle/input/**/{DATA_NAME}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            print("Sumber data: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    print("Sumber data: unduh ->", DATA_URL)
    import requests
    r = requests.get(DATA_URL, timeout=180); r.raise_for_status()
    return pd.read_csv(StringIO(r.text), index_col=0)

def export_df(df, name):
    path = f"exports/{name}.csv"
    df.to_csv(path, index=False)
    print(path)

df_raw = load_sensor_data()
df_raw.index = pd.to_datetime(df_raw.index, utc=True)
df_raw = df_raw.sort_index()
print("Baris akuisisi:", len(df_raw))
print("Rentang waktu :", df_raw.index.min(), "->", df_raw.index.max())
""" % DATA_URL))

cells.append(code("""# --- Audit 10 langkah (angka dihitung dari data, bukan diketik manual) ---
audit = {}
audit["kolom_ada"] = sum(c in df_raw.columns for c in cols)
audit["baris_mentah"] = len(df_raw)

dup = int(df_raw.index.duplicated().sum())
audit["stempel_ganda"] = dup
df = df_raw[~df_raw.index.duplicated()].copy()

audit["nan_mentah"] = {c: int(df[c].isna().sum()) for c in cols}
audit["luar_range"] = {c: int(((df[c] < 0) | (df[c] > 100)).sum()) for c in cols}

def hitung_macet(s, minimal=10):
    s = pd.Series(s).reset_index(drop=True)
    sama = s.eq(s.shift())
    grup = (sama != sama.shift()).cumsum()
    panjang = sama.groupby(grup).sum()
    return int((panjang >= minimal - 1).sum())

audit["macet"] = {c: hitung_macet(df[c]) for c in cols}
audit["pencilan_z6"] = {c: int((np.abs((df[c] - df[c].mean()) / df[c].std()) > 6).sum()) for c in cols}

# Rata-rata per 5 menit -> grid seragam -> interpolasi waktu bila ada slot bolong
r = df[cols].resample(RESAMPLE_RULE).mean()
grid = pd.date_range(r.index[0], r.index[-1], freq=f"{SAMPLING_SECONDS}s")
r = r.reindex(grid)
audit["slot_bolong"] = int(r[cols].isna().all(axis=1).sum())
n_before = int(r[cols].isna().sum().sum())
df_sync = r.interpolate(method="time", limit_direction="both")
audit["isian_interpolasi"] = int(n_before - df_sync[cols].isna().sum().sum())
n_before2 = int(df_sync[cols].isna().sum().sum())
df_sync = df_sync.fillna(df_sync[cols].median())
audit["isian_median"] = int(n_before2 - df_sync[cols].isna().sum().sum())
audit["baris_akhir"] = len(df_sync)

print("Baris mentah :", audit["baris_mentah"])
print("Stempel ganda:", audit["stempel_ganda"], "-> dibuang")
print("NaN mentah   :", audit["nan_mentah"])
print("Luar 0-100%  :", audit["luar_range"], "-> dilaporkan, TIDAK dibuang")
print("Macet >=10   :", audit["macet"], "-> dilaporkan, TIDAK dibuang")
print("Pencilan |z|>6:", audit["pencilan_z6"], "-> dilaporkan, TIDAK dibuang")
print("Setelah 5 menit:", audit["baris_akhir"], "baris | slot bolong:", audit["slot_bolong"],
      "| isian interpolasi:", audit["isian_interpolasi"], "| isian median:", audit["isian_median"])
""" ))

cells.append(md(nff.qc_steps_md("2", "3")))

cells.append(code("""# --- Tabel langkah 1-10 dalam bentuk tabel ---
tabel = pd.DataFrame([
    [1, "Kolom wajib kelembaban1..4", f"{audit['kolom_ada']}/4 ada", "lanjut"],
    [2, "datetime UTC lalu diurutkan", f"{audit['baris_mentah']:,} baris terurut", "-"],
    [3, "Stempel waktu ganda dibuang", f"{audit['stempel_ganda']}", "dibuang (0 baris)"],
    [4, "Cek nilai kosong (NaN)", str(sum(audit['nan_mentah'].values())), "tidak ada yang ditambal"],
    [5, "Cek nilai di luar 0-100%", str(sum(audit['luar_range'].values())), "dilaporkan, tidak dibuang"],
    [6, "Pembacaan macet >= 10", " / ".join(f"{k}:{v}" for k, v in audit['macet'].items()), "dilaporkan, tidak dibuang"],
    [7, "Pencilan ekstrem |z| > 6", " / ".join(f"{k}:{v}" for k, v in audit['pencilan_z6'].items()), "dilaporkan, tidak dibuang"],
    [8, "Rata-rata per 5 menit", f"{audit['baris_mentah']:,} -> {audit['baris_akhir']:,} baris", "agregasi"],
    [9, "Grid seragam + interpolasi waktu", f"slot bolong {audit['slot_bolong']} -> isian {audit['isian_interpolasi']}", "tidak terpakai di data ini"],
    [10, "Sisa NaN ditambal median", f"isian {audit['isian_median']}", "tidak terpakai di data ini"],
], columns=["#", "Langkah", "Hasil", "Tindakan"])
display(tabel)
export_df(tabel, "ILU_tabel_10_langkah")
"""))

cells.append(code("""# --- Empat kanal setelah pembersihan (sebelum injeksi fault) ---
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
for ax, c, s in zip(axes, cols, SENSORS):
    ax.plot(df_sync.index, df_sync[c], lw=0.6, color="#34495e")
    ax.set_ylabel(s + " (%)", fontsize=8)
    ax.grid(alpha=0.25)
axes[-1].set_xlabel("Waktu (UTC)")
fig.suptitle("Data setelah pembersihan dan penyelarasan waktu (5 menit)", fontsize=10)
fig.tight_layout()
fig.savefig("exports/ILU_empat_kanal_bersih.png", dpi=120, bbox_inches="tight")
plt.show()
print("Simpan: exports/ILU_empat_kanal_bersih.png")
"""))

cells.append(code("""# === Simulator fault — parameter dan perilaku identik dengan notebook penelitian ===
DRIFT_PER_SAMPLE = 0.02 * (SAMPLING_SECONDS / 30)

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

_D = {"intensity": DRIFT_PER_SAMPLE}
_S = {"intensity": 0.08, "p": 0.015}
_B = {"bias": 0.08}
_H = {"stuck_prob": 0.08, "loss_prob": 0.05}

SCENARIOS = {
    "normal": [],
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

print("Kondisi:", len(SCENARIOS), "| drift per sampel:", round(DRIFT_PER_SAMPLE, 4), "%")
"""))

cells.append(md(nff.figures_md("3", "4")))
cells.append(code(nff.figures_code("ILU_", seed_base=9000)))

cells.append(md("""## 4 — Ringkasan

- Dari 281.721 baris mentah tidak ada baris yang dibuang karena nilainya
  mencurigakan: stempel ganda 0, NaN 0, nilai di luar 0–100 % 0. Nilai yang
  mencurigakan (pembacaan macet, pencilan ekstrem) **dilaporkan** karena pada
  penelitian ini justru merupakan gejala fault.
- Setelah rata-rata 5 menit, data menjadi 28.173 baris pada grid seragam;
  tidak ada slot bolong sehingga interpolasi dan isian median tidak terpakai.
- Grafik di §3 memakai parameter injeksi yang sama dengan notebook penelitian.
  Perhatikan skala: drift paling kasatmata, sedangkan spike dan bias sangat
  kecil terhadap ayunan sinyal, sehingga panelnya dilengkapi kotak selisih.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", OUT, "| cells:", len(cells))
