#!/usr/bin/env python3
"""Build 15_Data_Hipotetik_Berlabel_ANN_LM.ipynb.

Permintaan Bu Luh (chat 18/11/26): dosen beliau minta data hipotetik yang
sudah berlabel untuk SEMUA skenario, plus programnya, untuk membangun sistem
notifikasi gangguan. Basisnya "program pertama yang pakai data buatan, ANN LM
tanpa entropi" (final_databaru_optimized.ipynb, commit 91f6d77, 2025-12-14),
tetapi ANN LM-nya diupdate ke LM asli (notebook 10/11/14).

Yang dipakai ulang (dibaca dari teks builder lain, bukan disalin manual):
  * simulator fault bentuk baru tingkat "sedang"  -> NEW_SIM di build_14
  * Levenberg-Marquardt asli (Jacobian analitik)   -> lm_* di build_11

Keluaran: 15_Data_Hipotetik_Berlabel_ANN_LM.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "15_Data_Hipotetik_Berlabel_ANN_LM.ipynb"
REPO_RAW = ("https://raw.githubusercontent.com/kelvindesman/public-files/"
            "refs/heads/main/tabel_sensor4_generated.csv")

SRC11 = (HERE / "build_11_annlm_real_notebook.py").read_text()
SRC14 = (HERE / "build_14_lengkap_fault_baru_notebook.py").read_text()


def between(src, start, end):
    i = src.find(start)
    assert i >= 0, f"penanda tidak ditemukan: {start!r}"
    j = src.find(end, i)
    assert j > i, f"penanda akhir tidak ditemukan: {end!r}"
    return src[i:j]


NEW_SIM = re.search(r"NEW_SIM = r'''(.*?)'''", SRC14, re.S).group(1)
assert "def simulate_drift_fault" in NEW_SIM and "FAULT_BIT" in NEW_SIM
LM_CODE = between(SRC11, "def lm_init_layers(", "class LMEstimator(")
assert "def lm_fit(" in LM_CODE and "def lm_predict_proba(" in LM_CODE
assert "least_squares" not in LM_CODE


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# 15 — Data hipotetik berlabel (semua skenario) + ANN-LM asli tanpa entropi

**Permintaan:** dosen Bu Luh meminta *data hipotetik yang sudah berlabel untuk
semua skenario* beserta programnya, untuk membangun **sistem notifikasi
gangguan** sensor kelembaban tanah.

**Basis:** program pertama yang memakai data buatan
(`final_databaru_optimized.ipynb`, Des 2025): data `tabel_sensor4_generated.csv`,
lima skenario fault, fitur lag per sampel, ANN 1 dan 2 layer, **tanpa entropi**.

**Yang diperbarui dibanding program pertama**

| Bagian | Program pertama (Des 2025) | Notebook ini |
|---|---|---|
| Pelatihan ANN | `scipy.optimize.least_squares(method='lm')` — MINPACK, Jacobian numerik, init `randn`, 1 keluaran sigmoid | **LM asli** notebook 10/11/14: langkah Gauss-Newton terdamping + rasio gain (Nielsen), **Jacobian analitik** (lolos self-check), init Xavier, 2 keluaran one-hot + softmax |
| Simulator fault | drift 0,15 poin/**sampel** (tembus > 100 %), bias konstan 0,35, spike 0,3·std berkala | versi notebook 14 tingkat "sedang": **bias episode 5 poin**, **drift linear total 5 poin per episode**, **spike relatif 30 %** (kepadatan 1,52 %), hardware malfunction (macet 8 %, hilang 5 %); 3 episode × 3 hari lalu kembali normal; semua dijaga 0–100 % |
| Bentuk data | 4 sensor dirata-rata jadi 1 sinyal | **dua versi**: 4 sensor terpisah (label per sensor) **dan** 1 sinyal rata-rata |
| Skenario | S1..S5, sebagian kondisi dipilih acak | **S1..S5 lengkap = 16 kondisi fault** + normal, seed tetap (bisa direproduksi) |
| Pembagian data | 60/20/20 berurutan waktu | blok harian bergilir 3:1:1 (lihat §4 — dengan 60/20/20 berurutan, bagian uji tidak berisi episode drift/bias sama sekali) |
| Entropi | tidak dipakai | tetap **tidak dipakai** |

**Isi notebook**

1. §1 Data sumber + QC
2. §2 Simulator fault + peta skenario
3. §3 **Data hipotetik berlabel** (deliverable utama) → `exports/15_data_hipotetik_*.csv.gz` + kamus kolom
4. §4 Fitur & pembagian data
5. §5 ANN-LM asli (self-check Jacobian, lalu latih per kondisi)
6. §6 Hasil + prediksi per sampel (probabilitas & alarm) untuk sistem notifikasi
"""))

cells.append(code(r"""import os, time, json, io
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
import time as _time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from IPython.display import FileLink, display

NOTEBOOK_START = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"      # uji cepat lokal: data dipotong, iterasi sedikit
RANDOM_SEED = 42
EXPORT_DIR = Path("exports"); EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def export_df(df, name, gz=False, index=False):
    p = EXPORT_DIR / (f"{name}.csv.gz" if gz else f"{name}.csv")
    df.to_csv(p, index=index)
    display(FileLink(str(p)))
    return p

print("SMOKE =", SMOKE)
"""))

# ---------------------------------------------------------------- §1 data
cells.append(md(r"""# §1 — Data sumber

`tabel_sensor4_generated.csv` — data yang sama dengan program pertama: 4 sensor
kelembaban tanah (`kelembaban1..4`, %), interval ±30 detik, 2025-09-14 s.d.
2025-10-17. Satu baris berstempel 1970 (jam perangkat belum sinkron) dibuang;
baris lain tidak diubah. Sinyal ini dianggap **normal** (belum ada fault), lalu
fault disuntikkan di §3.
"""))

cells.append(code(r"""DATA_NAME = "tabel_sensor4_generated.csv"
REPO_RAW = "%s"
SENSORS = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]

def _ambil_cermin(url, timeout=180):
    # Unduh url; bila raw.githubusercontent.com diblokir (gagal 443), coba cermin jsDelivr.
    # Urutan: GitHub raw -> jsDelivr -> jsDelivr versi .xz (untuk file > 20 MB).
    # Hasil: objek dengan .text dan .content, seperti requests.Response.
    import lzma, types, requests
    asal = "https://raw.githubusercontent.com/kelvindesman/public-files/"
    calon = [(url, False)]
    if url.startswith(asal):
        jalur = url[len(asal):]
        for awalan in ("refs/heads/main/", "main/"):
            if jalur.startswith(awalan):
                jalur = jalur[len(awalan):]
        cdn = "https://cdn.jsdelivr.net/gh/kelvindesman/public-files@"
        # .xz dipatok ke commit: jsDelivr @main menolak file ini (batas ukuran paket)
        calon += [(cdn + "main/" + jalur, False),
                  (cdn + "010770339060a00277a6021f802d54df1608c2c8/" + jalur + ".xz", True)]
    galat = []
    for u, xz in calon:
        try:
            r = requests.get(u, timeout=timeout)
            r.raise_for_status()
            isi = lzma.decompress(r.content) if xz else r.content
            if u != url:
                print("Sumber cadangan dipakai ->", u)
            return types.SimpleNamespace(content=isi, text=isi.decode("utf-8"))
        except Exception as e:
            galat.append(f"{u}: {e}")
    raise RuntimeError("Data tidak bisa diunduh dari semua sumber: " + " | ".join(galat))


def load_data():
    for p in [Path(DATA_NAME), *Path("/kaggle/input").glob(f"**/{DATA_NAME}")]:
        if p.exists():
            return pd.read_csv(p)
    import requests
    r = _ambil_cermin(REPO_RAW, 120)
    return pd.read_csv(io.StringIO(r.text))

raw = load_data()
raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
n0 = len(raw)
bad_ts = raw["timestamp"].isna() | (raw["timestamp"] < pd.Timestamp("2020-01-01", tz="UTC"))
df = raw.loc[~bad_ts, ["timestamp", *SENSORS]].sort_values("timestamp").reset_index(drop=True)
if SMOKE:
    df = df.iloc[:20000].reset_index(drop=True)
dt = df["timestamp"].diff().dt.total_seconds()
qc = pd.DataFrame([{
    "baris_awal": n0, "dibuang_stempel_invalid": int(bad_ts.sum()), "baris_dipakai": len(df),
    "mulai": df["timestamp"].iloc[0], "selesai": df["timestamp"].iloc[-1],
    "interval_median_s": float(dt.median()), "NaN_sensor": int(df[SENSORS].isna().sum().sum()),
}])
display(qc)
display(df[SENSORS].describe().round(2))
assert df[SENSORS].isna().sum().sum() == 0
SAMPLING_SECONDS = 30
""" % REPO_RAW))

# ---------------------------------------------------------------- §2 simulator
cells.append(md(r"""# §2 — Simulator fault (versi notebook 14, tingkat "sedang")

| Fault | Bentuk | Parameter |
|---|---|---|
| **drift** | naik linear dari 0 ke 5 poin selama satu episode 3 hari, lalu reset (kalibrasi ulang) | 3 episode, mulai di 10 %, 40 %, 70 % panjang data |
| **bias** | offset +5 poin selama episode, lalu kembali normal | episode sama dengan drift |
| **spike** | `x(t)·(1 + 0,30)` di posisi acak | 1,52 % titik |
| **hardware** | nilai macet (diganti nilai acak lain dari seri) 8 % titik, data hilang (NaN) 5 % titik | — |
| **kombinasi** | semua komponen disuntik pada interval episode yang **sama** | — |

Label per sampel disimpan sebagai **bitmask**: drift = 1, spike = 2, bias = 4,
hardware = 8. Contoh: 6 = spike + bias. Kolom `jenis_*` di data ekspor
menuliskannya sebagai teks.

**Skenario** mengikuti program pertama:

| Skenario | Isi | Jumlah kondisi |
|---|---|---|
| S1 | satu fault, jenisnya dipilih acak (seed tetap) | 1 |
| S2 | fault tunggal: drift, spike, bias, hardware | 4 |
| S3 | kombinasi dua fault | 6 |
| S4 | kombinasi tiga fault | 4 |
| S5 | keempat fault sekaligus | 1 |
"""))

cells.append(code(r"""DRIFT_EPISODE_DAYS = 3.0
DRIFT_N_EPISODES = 3
DRIFT_SAMPLE_SECONDS = SAMPLING_SECONDS
DRIFT_TOTAL_POINTS = float(os.environ.get("DRIFT_TOTAL_POINTS", 5.0))
BIAS_EPISODE_POINTS = float(os.environ.get("BIAS_EPISODE_POINTS", 5.0))
SPIKE_REL = float(os.environ.get("SPIKE_REL", 0.30))
SPIKE_DENSITY = 0.0152

%s
_D = {"total": DRIFT_TOTAL_POINTS}
_S = {"k": SPIKE_REL}
_B = {"bias": BIAS_EPISODE_POINTS}
_H = {"stuck_prob": 0.08, "loss_prob": 0.05}

CONDITIONS = {
    "faulty": ("S1", [(simulate_choose_one, {"options": [(simulate_drift_fault, _D), (simulate_spike_fault, _S),
                                                          (simulate_bias_fault, _B), (simulate_hardware_fault, _H)]})]),
    "drift": ("S2", [(simulate_drift_fault, _D)]),
    "spike": ("S2", [(simulate_spike_fault, _S)]),
    "bias": ("S2", [(simulate_bias_fault, _B)]),
    "hardware": ("S2", [(simulate_hardware_fault, _H)]),
    "bias+malfunc": ("S3", [(simulate_bias_fault, _B), (simulate_hardware_fault, _H)]),
    "spike+malfunc": ("S3", [(simulate_spike_fault, _S), (simulate_hardware_fault, _H)]),
    "spike+bias": ("S3", [(simulate_spike_fault, _S), (simulate_bias_fault, _B)]),
    "drift+malfunc": ("S3", [(simulate_drift_fault, _D), (simulate_hardware_fault, _H)]),
    "drift+bias": ("S3", [(simulate_drift_fault, _D), (simulate_bias_fault, _B)]),
    "drift+spike": ("S3", [(simulate_drift_fault, _D), (simulate_spike_fault, _S)]),
    "spike+bias+malfunc": ("S4", [(simulate_spike_fault, _S), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)]),
    "drift+bias+malfunc": ("S4", [(simulate_drift_fault, _D), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)]),
    "spike+drift+malfunc": ("S4", [(simulate_spike_fault, _S), (simulate_drift_fault, _D), (simulate_hardware_fault, _H)]),
    "drift+spike+bias": ("S4", [(simulate_drift_fault, _D), (simulate_spike_fault, _S), (simulate_bias_fault, _B)]),
    "spike+bias+malfunc+drift": ("S5", [(simulate_spike_fault, _S), (simulate_bias_fault, _B),
                                         (simulate_hardware_fault, _H), (simulate_drift_fault, _D)]),
}
SCENARIO_ORDER = ["S1", "S2", "S3", "S4", "S5"]
assert len(CONDITIONS) == 16

def bits_to_text(b):
    b = int(b)
    if b == 0:
        return "normal"
    return "+".join(n for n, v in FAULT_BIT.items() if b & v)

print(pd.Series({k: v[0] for k, v in CONDITIONS.items()}).value_counts().sort_index())
""" % NEW_SIM))

# ---------------------------------------------------------------- §3 data berlabel
cells.append(md(r"""# §3 — Data hipotetik berlabel (deliverable utama)

Untuk **tiap kondisi** fault dibuat satu salinan data penuh dengan fault
disuntikkan, seed tetap (hasil selalu sama tiap dijalankan).

**Versi A — 4 sensor terpisah.** Fault hanya masuk ke sebagian sensor
(1–4 sensor, bergiliran antar kondisi, sensor mana dipilih acak dengan seed
tetap). Sensor lain tetap bersih — seperti kondisi lapangan, tidak semua
sensor rusak bersamaan.

**Versi B — 1 sinyal rata-rata** (seperti program pertama): rata-rata 4 sensor
bersih → `kelembaban_tanah_persen`, lalu fault disuntikkan ke sinyal itu.

Nilai yang hilang karena hardware malfunction **dibiarkan NaN** di kolom data
(itu bagian dari gejala gangguan yang harus dikenali sistem notifikasi);
nilai bersihnya tetap ada di kolom `*_asli`.

Satu file per skenario: `15_data_hipotetik_4sensor_S1..S5.csv.gz`,
`15_data_hipotetik_rata2_S1..S5.csv.gz`, plus `..._S0_normal` (data bersih
tanpa fault) dan `15_kamus_kolom.csv`.
"""))

cells.append(code(r"""X_clean = df[SENSORS].to_numpy(dtype=float)
avg_clean = X_clean.mean(axis=1)
T = len(df)
SUBSET_SIZES = [1, 2, 3, 4]

GEN_4S, GEN_AVG = {}, {}     # kondisi -> dict array
rng_master = np.random.default_rng(RANDOM_SEED)
for ci, (cond, (scen, faults)) in enumerate(CONDITIONS.items()):
    seed = int(rng_master.integers(1_000_000_000))
    rng = np.random.default_rng(seed)
    # --- versi A: 4 sensor ---
    k = SUBSET_SIZES[ci % len(SUBSET_SIZES)]
    subset = np.sort(rng.choice(4, size=k, replace=False))
    Y = X_clean.copy(); M = np.zeros((T, 4), dtype=np.uint8)
    for s in subset:
        y, m = simulate_multiple_faults(X_clean[:, s].copy(), faults, seed=int(rng.integers(1_000_000_000)))
        Y[:, s] = y; M[:, s] = m
    GEN_4S[cond] = {"scen": scen, "Y": Y, "M": M, "subset": subset}
    # --- versi B: rata-rata ---
    y, m = simulate_multiple_faults(avg_clean.copy(), faults, seed=seed + 7)
    GEN_AVG[cond] = {"scen": scen, "y": y, "m": m}

def frame_4s(cond):
    g = GEN_4S[cond]
    out = pd.DataFrame({"timestamp": df["timestamp"], "skenario": g["scen"], "kondisi": cond})
    for s, name in enumerate(SENSORS):
        out[name] = np.round(g["Y"][:, s], 4)
    for s, name in enumerate(SENSORS):
        out[f"{name}_asli"] = np.round(X_clean[:, s], 4)
    for s in range(4):
        out[f"label_s{s+1}"] = (g["M"][:, s] > 0).astype(np.int8)
    for s in range(4):
        out[f"jenis_s{s+1}"] = pd.Series(g["M"][:, s]).map(bits_to_text).to_numpy()
    lab = g["M"] > 0
    out["label_fault"] = lab.any(axis=1).astype(np.int8)
    out["sensor_rusak"] = [",".join(f"s{s + 1}" for s in np.flatnonzero(r)) for r in lab]
    out["sensor_terinjeksi"] = ",".join(f"s{s + 1}" for s in g["subset"])
    return out

def frame_avg(cond):
    g = GEN_AVG[cond]
    return pd.DataFrame({"timestamp": df["timestamp"], "skenario": g["scen"], "kondisi": cond,
                         "kelembaban_tanah_persen": np.round(g["y"], 4),
                         "kelembaban_tanah_persen_asli": np.round(avg_clean, 4),
                         "label_fault": (g["m"] > 0).astype(np.int8),
                         "jenis_fault": pd.Series(g["m"]).map(bits_to_text).to_numpy()})

# --- asersi konsistensi (sumber: definisi simulator notebook 14) ---
for cond in CONDITIONS:
    g = GEN_4S[cond]
    changed = ~np.isclose(g["Y"], X_clean, equal_nan=False)          # NaN dihitung berubah
    assert not (changed & (g["M"] == 0)).any(), f"{cond}: nilai berubah tanpa label"
    assert ((g["M"] > 0).sum() > 0) and ((g["M"] == 0).sum() > 0), cond
    v = g["Y"][~np.isnan(g["Y"])]; assert v.min() >= 0 and v.max() <= 100, cond
    untouched = np.setdiff1d(np.arange(4), g["subset"])
    assert (g["M"][:, untouched] == 0).all(), f"{cond}: sensor tak terinjeksi berlabel"
    ga = GEN_AVG[cond]
    ch = ~np.isclose(ga["y"], avg_clean)
    assert not (ch & (ga["m"] == 0)).any(), f"{cond}: rata2 berubah tanpa label"
    assert 0 < (ga["m"] > 0).sum() < T, cond
print("Asersi label lolos untuk 16 kondisi x 2 versi.")

# --- ekspor per skenario ---
normal_4s = pd.DataFrame({"timestamp": df["timestamp"], "skenario": "S0", "kondisi": "normal",
                          **{n: X_clean[:, s] for s, n in enumerate(SENSORS)},
                          **{f"label_s{s+1}": np.zeros(T, np.int8) for s in range(4)},
                          "label_fault": np.zeros(T, np.int8)})
export_df(normal_4s, "15_data_hipotetik_4sensor_S0_normal", gz=True)
export_df(pd.DataFrame({"timestamp": df["timestamp"], "skenario": "S0", "kondisi": "normal",
                        "kelembaban_tanah_persen": avg_clean, "label_fault": np.zeros(T, np.int8)}),
          "15_data_hipotetik_rata2_S0_normal", gz=True)
for scen in SCENARIO_ORDER:
    conds = [c for c, (s, _) in CONDITIONS.items() if s == scen]
    export_df(pd.concat([frame_4s(c) for c in conds], ignore_index=True),
              f"15_data_hipotetik_4sensor_{scen}", gz=True)
    export_df(pd.concat([frame_avg(c) for c in conds], ignore_index=True),
              f"15_data_hipotetik_rata2_{scen}", gz=True)

# --- ringkasan label ---
rows = []
for cond, (scen, _) in CONDITIONS.items():
    g = GEN_4S[cond]; ga = GEN_AVG[cond]
    r = {"skenario": scen, "kondisi": cond, "sensor_terinjeksi": ",".join(f"s{s + 1}" for s in g["subset"]),
         "n_sampel": T}
    for s in range(4):
        r[f"fault_s{s+1}_%"] = round(100 * float((g["M"][:, s] > 0).mean()), 2)
    r["fault_any_4sensor_%"] = round(100 * float((g["M"] > 0).any(axis=1).mean()), 2)
    r["NaN_4sensor"] = int(np.isnan(g["Y"]).sum())
    r["fault_rata2_%"] = round(100 * float((ga["m"] > 0).mean()), 2)
    r["jenis_rata2_terbanyak"] = pd.Series(ga["m"][ga["m"] > 0]).map(bits_to_text).value_counts().index[0]
    rows.append(r)
ringkasan_label = pd.DataFrame(rows)
export_df(ringkasan_label, "15_ringkasan_label")
display(ringkasan_label)
"""))

cells.append(code(r"""KAMUS = [
    ("timestamp", "waktu sampel (UTC), interval ±30 detik", "keduanya"),
    ("skenario", "S0 normal; S1 satu fault acak; S2 fault tunggal; S3 dua fault; S4 tiga fault; S5 empat fault", "keduanya"),
    ("kondisi", "nama kondisi fault yang disuntikkan pada salinan data ini (16 kondisi + normal)", "keduanya"),
    ("kelembaban1..4", "nilai sensor SETELAH fault disuntikkan (%); NaN = data hilang akibat hardware malfunction", "4 sensor"),
    ("kelembaban1..4_asli", "nilai sensor sebelum fault (bersih)", "4 sensor"),
    ("label_s1..s4", "1 = sampel sensor itu terkena fault, 0 = normal", "4 sensor"),
    ("jenis_s1..s4", "jenis fault pada sampel itu: normal / drift / spike / bias / hardware / gabungannya, mis. 'spike+bias'", "4 sensor"),
    ("label_fault", "1 = minimal satu sensor (atau sinyal rata-rata) terkena fault pada sampel itu", "keduanya"),
    ("sensor_rusak", "daftar sensor yang berlabel fault pada sampel itu, mis. 's1,s3'", "4 sensor"),
    ("sensor_terinjeksi", "sensor yang disuntik fault pada kondisi ini (konstan per kondisi)", "4 sensor"),
    ("kelembaban_tanah_persen", "rata-rata 4 sensor SETELAH fault disuntikkan ke sinyal rata-rata (%)", "rata-rata"),
    ("kelembaban_tanah_persen_asli", "rata-rata 4 sensor bersih", "rata-rata"),
    ("jenis_fault", "jenis fault pada sampel sinyal rata-rata", "rata-rata"),
    ("kode bitmask", "drift=1, spike=2, bias=4, hardware=8 (dipakai di dalam program; di CSV sudah ditulis sebagai teks)", "—"),
    ("parameter fault", f"drift total {DRIFT_TOTAL_POINTS} poin/episode, bias {BIAS_EPISODE_POINTS} poin/episode, "
                        f"spike x(1+{SPIKE_REL}) pada {SPIKE_DENSITY*100:.2f}% titik, hardware macet 8% + hilang 5%, "
                        f"{DRIFT_N_EPISODES} episode x {DRIFT_EPISODE_DAYS:g} hari", "—"),
]
kamus = pd.DataFrame(KAMUS, columns=["kolom", "arti", "versi"])
export_df(kamus, "15_kamus_kolom")
display(kamus)
"""))

cells.append(code(r"""# Contoh sinyal: asli vs setelah fault, titik berlabel fault ditandai
BLOCK_DAY = int(round(86400 / SAMPLING_SECONDS))

def plot_conditions(version):
    fig, axes = plt.subplots(4, 4, figsize=(20, 13), sharex=True)
    for ax, cond in zip(axes.ravel(), CONDITIONS):
        if version == "4sensor":
            g = GEN_4S[cond]; s = int(g["subset"][0])
            y, base, m = g["Y"][:, s], X_clean[:, s], g["M"][:, s]
            ttl = f"{CONDITIONS[cond][0]} {cond} (sensor {s+1})"
        else:
            g = GEN_AVG[cond]; y, base, m = g["y"], avg_clean, g["m"]
            ttl = f"{CONDITIONS[cond][0]} {cond} (rata-rata)"
        # zoom 2 hari di sekitar awal episode pertama supaya spike/hardware terlihat jelas
        e0 = int(0.10 * T); a, b = max(0, e0 - BLOCK_DAY // 2), min(T, e0 + 3 * BLOCK_DAY // 2)
        t = np.arange(a, b)
        ax.plot(t, base[a:b], lw=0.7, color="0.6", label="asli")
        ax.plot(t, y[a:b], lw=0.7, color="tab:blue", label="setelah fault")
        idx = a + np.flatnonzero(m[a:b] > 0)
        ax.scatter(idx, np.nan_to_num(y[idx], nan=base[idx]), s=10, color="tab:red", label="label fault", zorder=3)
        ax.set_title(ttl, fontsize=9)
    axes[0, 0].legend(fontsize=7, markerscale=6)
    fig.suptitle(f"Data hipotetik berlabel — versi {version} (zoom 2 hari di sekitar awal episode pertama; NaN = celah)", fontsize=13)
    fig.tight_layout()
    p = EXPORT_DIR / f"15_contoh_sinyal_{version}.png"; fig.savefig(p, dpi=110); plt.show()

plot_conditions("4sensor")
plot_conditions("rata2")
"""))

# ---------------------------------------------------------------- §4 fitur
cells.append(md(r"""# §4 — Fitur & pembagian data (alur program pertama)

**Fitur per sampel, tanpa entropi** (sama seperti program pertama): nilai sinyal
10 langkah sebelumnya (`lag1..lag10`) + selisih terhadap sampel sebelumnya
(`diff`). Karena data hilang (NaN) dibiarkan apa adanya, nilai yang hilang
diisi nilai terakhir yang valid, dan ditambah dua penanda: `isna` (sampel ini
hilang) dan `isna_10` (berapa sampel hilang di 10 langkah terakhir) — setara
kolom `_isna` di program pertama.

**Pembagian data.** Program pertama memakai 60/20/20 berurutan waktu. Dengan
simulator baru, episode drift/bias ada di 10 %, 40 %, 70 % data, sehingga
bagian uji (80–100 %) **tidak berisi episode drift/bias sama sekali** dan
evaluasinya tidak bermakna. Jadi data dipotong per **blok 1 hari** dan blok
dibagi bergiliran latih, latih, latih, validasi, uji (3:1:1 ≈ 60/20/20). Tiap
episode 3 hari kebagian ke beberapa bagian, dan blok tetap utuh (tidak
diacak per sampel) sehingga sampel bertetangga tidak bocor antar bagian
kecuali 10 lag di batas blok.

`StandardScaler` hanya di-*fit* di data latih; ambang probabilitas dipilih
yang memaksimalkan F1 di validasi (seperti program pertama), lalu dilaporkan
di data uji.

**Versi 4 sensor:** satu model untuk semua sensor — baris tiap sensor
ditumpuk, fitur hanya dari sensor itu sendiri. Hasil dilaporkan total dan per
sensor.
"""))

cells.append(code(r"""LAG_K = 10
BLOCK_LEN = int(round(86400 / SAMPLING_SECONDS))       # 1 hari
SPLIT_PATTERN = np.array([0, 0, 0, 1, 2])               # 0 latih, 1 validasi, 2 uji
SPLIT_NAME = {0: "latih", 1: "validasi", 2: "uji"}
split_of = SPLIT_PATTERN[(np.arange(T) // BLOCK_LEN) % len(SPLIT_PATTERN)]
valid_row = np.arange(T) >= LAG_K                        # baris awal tanpa lag lengkap dibuang
print(pd.Series(split_of[valid_row]).map(SPLIT_NAME).value_counts())

FEATURE_NAMES = [f"lag{k}" for k in range(1, LAG_K + 1)] + ["diff", "isna", "isna_10"]

def lag_features(x):
    x = np.asarray(x, dtype=float)
    na = np.isnan(x)
    xf = pd.Series(x).ffill().bfill().to_numpy()
    cols = [np.roll(xf, k) for k in range(1, LAG_K + 1)]
    cols.append(xf - np.roll(xf, 1))
    cols.append(na.astype(float))
    cols.append(pd.Series(na.astype(float)).rolling(LAG_K, min_periods=1).sum().shift(1).fillna(0).to_numpy())
    return np.column_stack(cols)       # baris < LAG_K dibuang lewat valid_row

assert lag_features(np.arange(20.0)).shape == (20, len(FEATURE_NAMES))
"""))

# ---------------------------------------------------------------- §5 LM
cells.append(md(r"""# §5 — ANN-LM asli

Kode Levenberg–Marquardt di bawah **identik** dengan notebook 10/11/14
(diambil langsung dari builder notebook 11). Ringkasnya, tiap iterasi:

1. hitung residual `r = keluaran − target` dan **Jacobian analitik** `J`
   (backpropagation per keluaran, bukan selisih hingga);
2. selesaikan `(JᵀJ + λ·diag(JᵀJ))·Δ = −Jᵀr`;
3. kalau galat turun, terima langkah dan kecilkan λ menurut rasio gain
   (aturan Nielsen); kalau naik, tolak dan besarkan λ.

Keluaran 2 unit (normal, fault) dengan target one-hot; probabilitas fault =
softmax unit kedua. Arsitektur mengikuti program pertama: **1 layer (6)** dan
**2 layer (6, 3)**, aktivasi `tanh`.
"""))

cells.append(code("import time as _time\n\n" + LM_CODE + r"""
# --- self-check: Jacobian analitik vs numerik ---
_rng = np.random.default_rng(7)
for _hidden in ([6], [6, 3]):
    _tpl = lm_init_layers(len(FEATURE_NAMES), _hidden, 2, seed=7)
    _X = _rng.normal(size=(16, len(FEATURE_NAMES))); _Y = np.eye(2)[_rng.integers(0, 2, 16)]
    _th = lm_pack(_tpl)
    _Ja = lm_residual_jacobian(_th, _tpl, _X, _Y, "tanh", mode="analytic")
    _Jn = lm_residual_jacobian(_th, _tpl, _X, _Y, "tanh", mode="numeric")
    _err = float(np.max(np.abs(_Ja - _Jn) / np.maximum(1e-8, np.abs(_Jn))))
    print(f"hidden={_hidden}: max |rel err| Jacobian = {_err:.2e}")
    assert _err < 1e-3, "Jacobian analitik tidak cocok"
print("Jacobian analitik LOLOS self-check.")
"""))

cells.append(code(r"""ARCHS = {"1 layer (6)": [6], "2 layer (6,3)": [6, 3]}
LM_ITERS = 5 if SMOKE else int(os.environ.get("LM_ITERS", 60))
LM_MAX_TRAIN_S = 300.0
MAX_TRAIN_ROWS = int(os.environ.get("MAX_TRAIN_ROWS", 20000 if SMOKE else 120000))
THRESHOLDS = np.linspace(0.05, 0.95, 91)

def best_threshold(p, y):
    best_t, best_f = 0.5, -1.0
    for t in THRESHOLDS:
        f = f1_score(y, (p > t).astype(int), zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    return best_t, best_f

def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan

def build_xy(version, cond):
    # kembalikan F (baris x fitur), y, split, sensor_id (0..3; -1 untuk rata-rata), t_index
    if version == "rata2":
        g = GEN_AVG[cond]
        F = lag_features(g["y"]); y = (g["m"] > 0).astype(int)
        keep = valid_row
        return F[keep], y[keep], split_of[keep], np.full(keep.sum(), -1), np.flatnonzero(keep)
    g = GEN_4S[cond]
    Fs, ys, sp, sid, ti = [], [], [], [], []
    for s in range(4):
        F = lag_features(g["Y"][:, s]); y = (g["M"][:, s] > 0).astype(int)
        Fs.append(F[valid_row]); ys.append(y[valid_row]); sp.append(split_of[valid_row])
        sid.append(np.full(valid_row.sum(), s)); ti.append(np.flatnonzero(valid_row))
    return (np.vstack(Fs), np.concatenate(ys), np.concatenate(sp),
            np.concatenate(sid), np.concatenate(ti))

def fit_lm_model(F, y, sp, hidden, seed):
    # latih di split latih, pilih ambang di validasi; kembalikan model (angka murni) + prediksi semua baris
    tr = np.flatnonzero(sp == 0)
    if len(tr) > MAX_TRAIN_ROWS:
        tr = np.sort(np.random.default_rng(seed).choice(tr, MAX_TRAIN_ROWS, replace=False))
    scaler = StandardScaler().fit(F[tr])
    Z = scaler.transform(F)
    c0 = _time.process_time(); t0 = _time.perf_counter()
    res = lm_fit(Z[tr], np.eye(2)[y[tr]], Z.shape[1], hidden, 2, activation="tanh",
                 iters=LM_ITERS, seed=seed, max_train_s=LM_MAX_TRAIN_S)
    wall = _time.perf_counter() - t0; cpu = _time.process_time() - c0
    p = lm_predict_proba(res["theta"], res["template"], Z)[:, 1]
    va = sp == 1
    thr, f1_val = best_threshold(p[va], y[va])
    model = {"theta": res["theta"], "shapes": np.array([W.shape for W, _ in res["template"]]),
             "mean": scaler.mean_, "scale": scaler.scale_, "ambang": thr, "activation": "tanh",
             "hidden": np.array(hidden), "lag_k": LAG_K, "feature_names": np.array(FEATURE_NAMES)}
    return model, p, thr, f1_val, res, tr, wall, cpu

def train_eval(version, cond, arch_name, hidden, seed):
    F, y, sp, sid, ti = build_xy(version, cond)
    model, p, thr, f1_val, res, tr, wall, cpu = fit_lm_model(F, y, sp, hidden, seed)
    te = sp == 2
    yhat = (p > thr).astype(int)
    row = {"versi": version, "skenario": CONDITIONS[cond][0], "kondisi": cond, "arsitektur": arch_name,
           "n_latih": len(tr), "n_uji": int(te.sum()), "fault_uji_%": round(100 * y[te].mean(), 2),
           "ambang": thr, "F1_validasi": round(f1_val, 4),
           "akurasi": accuracy_score(y[te], yhat[te]),
           "precision": precision_score(y[te], yhat[te], zero_division=0),
           "recall": recall_score(y[te], yhat[te], zero_division=0),
           "F1": f1_score(y[te], yhat[te], zero_division=0),
           "ROC_AUC": safe_auc(y[te], p[te]),
           "lm_iter": res["hist"]["iters"], "lm_status": res["hist"]["status"],
           "train_mse": res["train_mse"], "wall_s": wall, "cpu_s": cpu}
    per_sensor = []
    if version == "4sensor":
        for s in range(4):
            mk = te & (sid == s)
            yt, yh = y[mk], yhat[mk]
            neg = yt == 0
            per_sensor.append({"kondisi": cond, "skenario": CONDITIONS[cond][0], "arsitektur": arch_name,
                               "sensor": s + 1, "terinjeksi": int(s in GEN_4S[cond]["subset"]),
                               "n_fault_uji": int(yt.sum()),
                               "precision": precision_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                               "recall": recall_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                               "F1": f1_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                               "FPR": float(yh[neg].mean()) if neg.any() else np.nan})
    pred = {"t": ti, "sensor": sid, "split": sp, "label": y, "prob": p, "alarm": yhat, "thr": thr}
    return row, per_sensor, pred, res["hist"], model
"""))

cells.append(code(r"""RESULTS, PER_SENSOR, PREDS, HISTS, MODELS = [], [], {}, {}, {}
for version in ["rata2", "4sensor"]:
    for ci, cond in enumerate(CONDITIONS):
        for arch_name, hidden in ARCHS.items():
            row, ps, pred, hist, model = train_eval(version, cond, arch_name, hidden, seed=RANDOM_SEED + ci)
            RESULTS.append(row); PER_SENSOR.extend(ps); MODELS[(version, cond, arch_name)] = model
            PREDS[(version, cond, arch_name)] = pred; HISTS[(version, cond, arch_name)] = hist
            print(f"[{time.time()-NOTEBOOK_START:7.0f}s] {version:7s} {row['skenario']} {cond:26s} {arch_name:14s} "
                  f"F1={row['F1']:.3f} AUC={row['ROC_AUC']:.3f} iter={row['lm_iter']} {row['lm_status']} "
                  f"wall={row['wall_s']:.1f}s", flush=True)

hasil = pd.DataFrame(RESULTS)
for c in ["akurasi", "precision", "recall", "F1", "ROC_AUC", "train_mse", "wall_s", "cpu_s"]:
    hasil[c] = hasil[c].astype(float).round(4)
export_df(hasil, "15_hasil_ann_lm")
per_sensor = pd.DataFrame(PER_SENSOR).round(4)
export_df(per_sensor, "15_hasil_ann_lm_per_sensor")
"""))

# ---------------------------------------------------------------- §6 hasil
cells.append(md(r"""# §6 — Hasil

Tabel di bawah: F1 di data **uji** per skenario (rata-rata kondisi dalam satu
skenario), lalu rincian per kondisi. Ambang alarm dipilih di validasi.
"""))

cells.append(code(r"""ringkas = (hasil.groupby(["versi", "arsitektur", "skenario"])[["precision", "recall", "F1", "ROC_AUC"]]
           .mean().round(3).reset_index())
export_df(ringkas, "15_hasil_ringkas_per_skenario")
display(ringkas.pivot_table(index=["versi", "arsitektur"], columns="skenario", values="F1"))
display(hasil[["versi", "skenario", "kondisi", "arsitektur", "fault_uji_%", "precision", "recall", "F1",
               "ROC_AUC", "lm_iter", "wall_s"]])

if len(per_sensor):
    ps = per_sensor[per_sensor["terinjeksi"] == 1]
    display(ps.groupby(["arsitektur", "sensor"])[["precision", "recall", "F1"]].mean().round(3))
    fpr = per_sensor[per_sensor["terinjeksi"] == 0].groupby("arsitektur")["FPR"].mean().round(4)
    print("FPR rata-rata pada sensor yang TIDAK disuntik fault:"); display(fpr)

# Kurva loss LM untuk satu kondisi per skenario
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, version in zip(axes, ["rata2", "4sensor"]):
    for cond in ["faulty", "drift", "bias+malfunc", "drift+bias+malfunc", "spike+bias+malfunc+drift"]:
        h = HISTS[(version, cond, "2 layer (6,3)")]
        ax.plot(h["loss"], label=f"{CONDITIONS[cond][0]} {cond}")
    ax.set_yscale("log"); ax.set_xlabel("iterasi LM"); ax.set_ylabel("MSE latih"); ax.set_title(version)
axes[0].legend(fontsize=7); fig.tight_layout()
fig.savefig(EXPORT_DIR / "15_kurva_loss_lm.png", dpi=90); plt.show()
"""))

cells.append(md(r"""## Prediksi per sampel (untuk sistem notifikasi)

Untuk tiap versi dan skenario, model 2 layer (6,3) diterapkan ke **seluruh**
data kondisi itu. Kolom `prob_fault` = probabilitas gangguan dari ANN-LM,
`alarm` = 1 bila `prob_fault` > `ambang` (dipilih di validasi), `split` =
bagian data (latih/validasi/uji — angka evaluasi yang jujur hanya dari `uji`).
Sistem notifikasi bisa langsung membaca kolom `alarm`, atau memakai
`prob_fault` dengan ambang/aturan persistensi sendiri (mis. alarm baru dikirim
kalau `alarm` = 1 selama beberapa sampel berturut-turut).
"""))

cells.append(code(r"""PRED_ARCH = "2 layer (6,3)"
for version in ["rata2", "4sensor"]:
    for scen in SCENARIO_ORDER:
        parts = []
        for cond in [c for c, (s, _) in CONDITIONS.items() if s == scen]:
            pr = PREDS[(version, cond, PRED_ARCH)]
            part = pd.DataFrame({"timestamp": df["timestamp"].to_numpy()[pr["t"]], "skenario": scen,
                                 "kondisi": cond, "split": pd.Series(pr["split"]).map(SPLIT_NAME).to_numpy(),
                                 "label_fault": pr["label"], "prob_fault": np.round(pr["prob"], 5),
                                 "ambang": round(pr["thr"], 3), "alarm": pr["alarm"]})
            if version == "4sensor":
                part.insert(3, "sensor", pr["sensor"] + 1)
            parts.append(part)
        export_df(pd.concat(parts, ignore_index=True), f"15_prediksi_{version}_{scen}", gz=True)
"""))

cells.append(md(r"""# §7 — Model gabungan & menyimpan model

Model di §5 dilatih **per kondisi** — berguna untuk membandingkan skenario,
tetapi sistem notifikasi di lapangan tidak tahu jenis gangguannya lebih dulu.
Karena itu dilatih juga satu **model gabungan** per versi: data latih dari
**semua 16 kondisi + data normal** sekaligus, label tetap fault/normal per
sampel, pembagian blok hari yang sama. Hasil ujinya dilaporkan total dan per
kondisi. Arsitekturnya dipilih dengan grid search (F1 validasi).

**Cara menyimpan.** Model disimpan sebagai file **`.npz` (numpy murni)**, bukan
pickle scikit-learn: pickle sklearn hanya bisa dibuka dengan versi sklearn yang
sama persis (sudah terbukti gagal antara Kaggle 1.6.1 dan laptop 1.9.1).
Isi tiap file: bobot LM (`theta`, `shapes`), `mean` & `scale` StandardScaler,
`ambang` alarm, nama fitur, dan `lag_k`.

* `15_model_gabungan_rata2.npz`, `15_model_gabungan_4sensor.npz` — model untuk sistem notifikasi
* `15_model_ann_lm_semua.zip` — semua model per kondisi (16 kondisi × 2 versi × 2 arsitektur) + daftar + loader
* `15_muat_model.py` — pemuat model, hanya butuh numpy + pandas

**Cara memuat:**

```python
from muat_model_15 import muat_model, deteksi   # salin 15_muat_model.py -> muat_model_15.py
m = muat_model("15_model_gabungan_rata2.npz")
prob, alarm = deteksi(m, nilai_sensor)            # nilai_sensor: array 1 sensor, urut waktu, interval 30 detik, NaN boleh
```

Model `4sensor` dipakai per sensor: panggil `deteksi` untuk tiap kolom sensor.
Notebook ini memuat ulang setiap file yang disimpan dengan `15_muat_model.py`
dan memastikan probabilitasnya **identik** dengan model di memori.
"""))

cells.append(md(r"""## Grid search arsitektur (tanpa identifikasi sensor)

Arsitektur model gabungan dipilih dengan **grid search**: setiap kandidat
dilatih di blok hari latih, lalu yang dipilih adalah kandidat dengan **F1
validasi** tertinggi (ambang juga disetel di validasi). Data uji tidak dipakai
untuk memilih. Tidak ada tahap identifikasi sensor: model `4sensor` menilai
**satu sensor sekaligus** dari deret sensor itu sendiri, jadi sensor yang
rusak langsung terbaca dari kolom alarm sensor tersebut.
"""))

cells.append(code(r"""GABUNGAN_GRID = ({"1 layer (6)": [6], "2 layer (6,3)": [6, 3]} if SMOKE else
                 {"1 layer (6)": [6], "1 layer (12)": [12], "2 layer (6,3)": [6, 3],
                  "2 layer (12,6)": [12, 6], "2 layer (16,8)": [16, 8]})
GAB_ROWS, GAB_PER_COND, GRID_ROWS, GAB_ARCH = [], [], [], {}
for gi, version in enumerate(["rata2", "4sensor"]):
    parts = []
    # data normal (bersih) ikut sebagai contoh negatif
    if version == "rata2":
        Fn = lag_features(avg_clean)[valid_row]; n_sens = 1
    else:
        Fn = np.vstack([lag_features(X_clean[:, s])[valid_row] for s in range(4)]); n_sens = 4
    parts.append(("normal", Fn, np.zeros(len(Fn), int), np.tile(split_of[valid_row], n_sens)))
    for cond in CONDITIONS:
        F, y, sp, sid, ti = build_xy(version, cond)
        parts.append((cond, F, y, sp))
    F = np.vstack([p_[1] for p_ in parts]); y = np.concatenate([p_[2] for p_ in parts])
    sp = np.concatenate([p_[3] for p_ in parts])
    cond_of = np.concatenate([np.full(len(p_[2]), p_[0], dtype=object) for p_ in parts])
    best = None
    for arch_name, hidden in GABUNGAN_GRID.items():
        fit = fit_lm_model(F, y, sp, hidden, seed=RANDOM_SEED + 100 + gi)
        model_, p_, thr_, f1v_, res_ = fit[:5]
        te_ = sp == 2; yh_ = (p_ > thr_).astype(int)
        n_param = int(len(model_["theta"]))
        GRID_ROWS.append({"versi": version, "arsitektur": arch_name, "n_parameter": n_param,
                          "ambang": thr_, "F1_validasi": f1v_,
                          "F1_uji": f1_score(y[te_], yh_[te_], zero_division=0),
                          "AUC_uji": safe_auc(y[te_], p_[te_]),
                          "lm_iter": res_["hist"]["iters"], "lm_status": res_["hist"]["status"],
                          "wall_s": fit[6]})
        print(f"[{time.time()-NOTEBOOK_START:7.0f}s] grid {version:7s} {arch_name:15s} P={n_param:4d} "
              f"F1 val={f1v_:.3f} uji={GRID_ROWS[-1]['F1_uji']:.3f} iter={res_['hist']['iters']} "
              f"{res_['hist']['status']} wall={fit[6]:.0f}s", flush=True)
        if best is None or f1v_ > best[0]:
            best = (f1v_, arch_name, fit)
    _, GABUNGAN_ARCH, (model, p, thr, f1_val, res, tr, wall, cpu) = best
    GAB_ARCH[version] = GABUNGAN_ARCH
    print(f"   -> dipilih (F1 validasi tertinggi): {GABUNGAN_ARCH}")
    MODELS[(version, "gabungan", GABUNGAN_ARCH)] = model
    te = sp == 2; yhat = (p > thr).astype(int)
    GAB_ROWS.append({"versi": version, "kondisi": "gabungan (16 kondisi + normal)", "arsitektur": GABUNGAN_ARCH,
                     "F1_validasi": f1_val,
                     "n_latih": len(tr), "n_uji": int(te.sum()), "fault_uji_%": round(100 * y[te].mean(), 2),
                     "ambang": thr, "precision": precision_score(y[te], yhat[te], zero_division=0),
                     "recall": recall_score(y[te], yhat[te], zero_division=0),
                     "F1": f1_score(y[te], yhat[te], zero_division=0), "ROC_AUC": safe_auc(y[te], p[te]),
                     "FPR_data_normal": float(yhat[te & (cond_of == "normal")].mean()),
                     "lm_iter": res["hist"]["iters"], "wall_s": wall, "cpu_s": cpu})
    for cond in ["normal", *CONDITIONS]:
        mk = te & (cond_of == cond); yt, yh = y[mk], yhat[mk]
        GAB_PER_COND.append({"versi": version, "skenario": "S0" if cond == "normal" else CONDITIONS[cond][0],
                             "kondisi": cond, "n_fault_uji": int(yt.sum()),
                             "precision": precision_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                             "recall": recall_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                             "F1": f1_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                             "FPR": float(yh[yt == 0].mean()) if (yt == 0).any() else np.nan})
    print(f"[{time.time()-NOTEBOOK_START:7.0f}s] gabungan {version}: F1={GAB_ROWS[-1]['F1']:.3f} "
          f"AUC={GAB_ROWS[-1]['ROC_AUC']:.3f} FPR normal={GAB_ROWS[-1]['FPR_data_normal']:.3f}", flush=True)

grid_gabungan = pd.DataFrame(GRID_ROWS).round(4)
export_df(grid_gabungan, "15_grid_search_gabungan")
display(grid_gabungan)
hasil_gabungan = pd.DataFrame(GAB_ROWS).round(4)
export_df(hasil_gabungan, "15_hasil_model_gabungan")
display(hasil_gabungan)
gab_per_kondisi = pd.DataFrame(GAB_PER_COND).round(4)
export_df(gab_per_kondisi, "15_hasil_model_gabungan_per_kondisi")
display(gab_per_kondisi.pivot_table(index=["skenario", "kondisi"], columns="versi", values=["F1", "FPR"]))
"""))

cells.append(code(r"""MUAT_MODEL_PY = '''# Muat dan pakai model ANN-LM notebook 15 (hanya numpy + pandas).
#
#     from muat_model_15 import muat_model, deteksi
#     m = muat_model("15_model_gabungan_rata2.npz")
#     prob, alarm = deteksi(m, nilai_sensor)
#
# nilai_sensor: array nilai SATU sensor (atau rata-rata 4 sensor untuk model
# "rata2"), urut waktu, interval +-30 detik, NaN boleh (data hilang).
# prob[t] = probabilitas gangguan pada sampel t; alarm[t] = prob[t] > ambang.
# 10 sampel pertama belum punya lag lengkap (prob tetap dihitung, abaikan saja).
import numpy as np
import pandas as pd


def muat_model(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fitur_lag(x, lag_k=10):
    x = np.asarray(x, dtype=float)
    na = np.isnan(x)
    xf = pd.Series(x).ffill().bfill().to_numpy()
    cols = [np.roll(xf, k) for k in range(1, lag_k + 1)]
    cols.append(xf - np.roll(xf, 1))
    cols.append(na.astype(float))
    cols.append(pd.Series(na.astype(float)).rolling(lag_k, min_periods=1).sum().shift(1).fillna(0).to_numpy())
    return np.column_stack(cols)


def prob_fault(m, F):
    a = (np.asarray(F, dtype=float) - m["mean"]) / m["scale"]
    shapes = [tuple(int(v) for v in s) for s in m["shapes"]]
    theta = m["theta"]
    i = 0                                          # urutan theta = lm_unpack: W1, b1, W2, b2, ...
    for li, (r, c) in enumerate(shapes):
        W = theta[i:i + r * c].reshape(r, c); i += r * c
        b = theta[i:i + c]; i += c
        a = a @ W + b
        if li < len(shapes) - 1:
            a = np.tanh(a)
    e = a - a.max(axis=1, keepdims=True)
    P = np.exp(e); P /= P.sum(axis=1, keepdims=True)
    return P[:, 1]


def deteksi(m, nilai_sensor, ambang=None):
    p = prob_fault(m, fitur_lag(nilai_sensor, int(m["lag_k"])))
    thr = float(m["ambang"]) if ambang is None else float(ambang)
    return p, (p > thr).astype(int)
'''
(EXPORT_DIR / "15_muat_model.py").write_text(MUAT_MODEL_PY)

import importlib.util, zipfile
_spec = importlib.util.spec_from_file_location("muat_model_15", EXPORT_DIR / "15_muat_model.py")
muat_model_15 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(muat_model_15)

def nama_file_model(version, cond, arch):
    if cond == "gabungan":
        return f"15_model_gabungan_{version}.npz"
    a = "2layer" if arch.startswith("2") else "1layer"
    return f"15_model_{cond.replace('+', '_')}_{version}_{a}.npz"

MODEL_DIR = EXPORT_DIR / "model_15"; MODEL_DIR.mkdir(exist_ok=True)
index_rows = []
for (version, cond, arch), m in MODELS.items():
    fn = nama_file_model(version, cond, arch)
    target = (EXPORT_DIR if cond == "gabungan" else MODEL_DIR) / fn
    x_cek = GEN_AVG["drift+bias"]["y"] if version == "rata2" else GEN_4S["spike+malfunc"]["Y"][:, 2]
    tpl = [[np.zeros(tuple(int(v) for v in s)), np.zeros(int(s[1]))] for s in m["shapes"]]
    p_mem = lm_predict_proba(m["theta"], tpl, (lag_features(x_cek) - m["mean"]) / m["scale"])[:, 1]
    extra = {}
    if cond == "gabungan":   # contoh input + prob acuan: pemakai bisa cek loader-nya memberi angka yang sama
        extra = {"contoh_input": x_cek[:2000], "contoh_prob": p_mem[:2000]}
    np.savez(target, versi=version, kondisi=cond, arsitektur=arch, **m, **extra)
    # --- muat ulang lewat 15_muat_model.py & bandingkan dengan model di memori ---
    mm = muat_model_15.muat_model(target)
    p_file, _ = muat_model_15.deteksi(mm, x_cek)
    diff = float(np.max(np.abs(p_file[valid_row] - p_mem[valid_row])))
    assert diff < 1e-9, f"{fn}: prediksi file != memori ({diff})"
    if cond != "gabungan":   # cek juga terhadap prediksi yang diekspor di §6
        pr = PREDS[(version, cond, arch)]
        sel = pr["sensor"] == (0 if version == "4sensor" else -1)
        x_own = GEN_AVG[cond]["y"] if version == "rata2" else GEN_4S[cond]["Y"][:, 0]
        p_own, _ = muat_model_15.deteksi(mm, x_own)
        diff = max(diff, float(np.max(np.abs(p_own[pr["t"][sel]] - pr["prob"][sel]))))
        assert diff < 1e-9, f"{fn}: prediksi file != prediksi §6 ({diff})"
    index_rows.append({"file": fn, "versi": version, "kondisi": cond, "arsitektur": arch,
                       "ambang": round(float(m["ambang"]), 3), "selisih_maks_prob_saat_muat_ulang": diff})

model_index = pd.DataFrame(index_rows)
model_index.to_csv(MODEL_DIR / "15_daftar_model.csv", index=False)
(MODEL_DIR / "15_muat_model.py").write_text(MUAT_MODEL_PY)
with zipfile.ZipFile(EXPORT_DIR / "15_model_ann_lm_semua.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(MODEL_DIR.iterdir()):
        zf.write(f, f.name)
for f in sorted(MODEL_DIR.iterdir()):
    f.unlink()
MODEL_DIR.rmdir()
export_df(model_index, "15_daftar_model")
print(f"{len(model_index)} model disimpan & dimuat ulang; selisih prob maks = "
      f"{model_index['selisih_maks_prob_saat_muat_ulang'].max():.2e}")
"""))

cells.append(md(r"""## Kartu model (`15_model_card.json`)

Ringkasan yang bisa dibaca mesin untuk setiap model gabungan: arsitektur hasil
grid search, parameter LM, fitur, ambang, data latih, parameter fault
injection, dan hasil uji. Notebook inferensi (model portabel, tanpa latih)
membaca file ini dan menampilkannya.
"""))

cells.append(code(r"""import inspect, datetime, json
_lm_sig = {k: v.default for k, v in inspect.signature(lm_fit).parameters.items()
           if v.default is not inspect.Parameter.empty}
MODEL_CARD = {
    "nama": "ANN-LM deteksi gangguan sensor kelembaban tanah (tanpa entropi, per sampel)",
    "dibuat": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "notebook_latih": "kaggle.com/code/kelvindsmn/data-hipotetik-berlabel-ann-lm",
    "status_publikasi": ("Metode (ANN dilatih Levenberg-Marquardt, fitur lag tanpa entropi, skenario S1-S5) "
                         "berasal dari program pertama (Des 2025); judul/DOI publikasinya belum terverifikasi, "
                         "konfirmasi ke pembimbing sebelum disitasi. Kode program pertama tertinggal: masih "
                         "memakai scipy least_squares(method='lm') dengan Jacobian numerik dan 1 keluaran sigmoid. "
                         "Model ini dilatih ulang dengan kode yang diperbarui (LM asli, Jacobian analitik)."),
    "label": ("Label berasal dari fault injection: gangguan disuntikkan secara sintetis ke data sensor, "
              "sehingga setiap sampel tahu pasti fault/normal. Bukan label dari gangguan nyata di lapangan."),
    "data_sumber": "tabel_sensor4_generated.csv (4 sensor kelembaban tanah, interval 30 detik)",
    "fault_injection": {"drift_total_poin_per_episode": DRIFT_TOTAL_POINTS,
                        "bias_poin_per_episode": BIAS_EPISODE_POINTS,
                        "spike_relatif": SPIKE_REL, "spike_kepadatan": SPIKE_DENSITY,
                        "hardware": "8% nilai macet + 5% hilang (NaN)",
                        "episode": f"{DRIFT_N_EPISODES} episode x {DRIFT_EPISODE_DAYS:g} hari (posisi 10/40/70%)",
                        "kondisi_latih": ["normal", *CONDITIONS]},
    "fitur": {"lag_k": LAG_K, "nama": FEATURE_NAMES, "interval_sampel_detik": SAMPLING_SECONDS},
    "pembagian_data": "blok hari berulang latih,latih,latih,validasi,uji",
    "lm": {"iterasi_maks": LM_ITERS, "lambda_awal": _lm_sig.get("lam0"), "batas_waktu_s": LM_MAX_TRAIN_S,
           "baris_latih_maks": MAX_TRAIN_ROWS, "target": "one-hot 2 kelas, keluaran linear, softmax saat prediksi",
           "aktivasi_tersembunyi": "tanh", "redaman": "Marquardt diag(JtJ), lambda diperbarui aturan Nielsen"},
    "grid_search": {"kandidat": {k: v for k, v in GABUNGAN_GRID.items()},
                    "kriteria": "F1 validasi tertinggi; data uji tidak dipakai untuk memilih",
                    "identifikasi_sensor": "tidak ada; model menilai satu deret sensor sendiri-sendiri"},
    "model": {},
}
for version in ["rata2", "4sensor"]:
    arch = GAB_ARCH[version]; m = MODELS[(version, "gabungan", arch)]
    hg = hasil_gabungan[hasil_gabungan["versi"] == version].iloc[0]
    MODEL_CARD["model"][version] = {
        "file": f"15_model_gabungan_{version}.npz",
        "masukan": ("rata-rata 4 sensor (satu deret)" if version == "rata2"
                    else "satu sensor per panggilan (panggil untuk tiap kolom sensor)"),
        "arsitektur": arch, "hidden": [int(h) for h in m["hidden"]], "n_parameter": int(len(m["theta"])),
        "ambang": round(float(m["ambang"]), 4),
        "hasil_uji": {k: (None if pd.isna(hg[k]) else float(hg[k]))
                      for k in ["precision", "recall", "F1", "ROC_AUC", "FPR_data_normal"]},
        "grid": grid_gabungan[grid_gabungan["versi"] == version]
                [["arsitektur", "n_parameter", "F1_validasi", "F1_uji"]].to_dict("records"),
    }
MODEL_CARD["batasan"] = [
    "Drift kecil sulit dideteksi per sampel (recall drift rendah, lihat 15_hasil_model_gabungan_per_kondisi.csv).",
    "Dilatih pada satu set data dan fault sintetis; pada data lain ambang mungkin perlu disetel ulang.",
    "Data harus satu deret per sensor, urut waktu, interval kira-kira 30 detik, nilai persen 0-100.",
]
(EXPORT_DIR / "15_model_card.json").write_text(json.dumps(MODEL_CARD, indent=1, ensure_ascii=False))
print(json.dumps(MODEL_CARD["model"], indent=1, ensure_ascii=False))
"""))

cells.append(code(r"""files = sorted(EXPORT_DIR.glob("15_*"))
daftar = pd.DataFrame({"file": [f.name for f in files],
                       "ukuran_MB": [round(f.stat().st_size / 1e6, 2) for f in files]})
display(daftar)
# baca balik satu file data supaya yakin bisa dipakai
_chk = pd.read_csv(EXPORT_DIR / "15_data_hipotetik_4sensor_S5.csv.gz")
assert {"label_s1", "jenis_s1", "label_fault", "sensor_rusak"} <= set(_chk.columns)
assert _chk["label_fault"].isin([0, 1]).all()
print(f"Selesai. Wall total {time.time()-NOTEBOOK_START:.0f} s, {len(files)} file di {EXPORT_DIR}/")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, "| cells:", len(cells))
