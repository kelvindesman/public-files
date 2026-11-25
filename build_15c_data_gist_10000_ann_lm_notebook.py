#!/usr/bin/env python3
"""Build 15c_Data_Hipotetik_Gist_10000_ANN_LM.ipynb.

Permintaan Bu Luh (chat 22/11/26 22.42-23.17): "run data-hipotetik-berlabel-ann-lm
pakai data 10.000 yang digenerated (data hipotetik), 10.000 titik", dan
"data yang berisi 10.000 titik adalah gistfile1.txt (gist 814fc8bc...), yaitu
data yang dipakai di final-2_3_sept_2025".

Program notebook 15 (fault injection S1-S5, fitur lag tanpa entropi, ANN-LM
asli) dijalankan ulang pada data gist itu. Data gist hanya punya SATU kolom
kelembaban tanah, jadi hanya versi 1 sinyal (bukan 4 sensor).

Yang dipakai ulang (dibaca dari teks builder lain, bukan disalin manual):
  * simulator fault bentuk baru tingkat "sedang"  -> NEW_SIM di build_14
  * Levenberg-Marquardt asli (Jacobian analitik)   -> lm_* di build_11
  * pemuat model .npz                              -> MUAT_MODEL_PY di build_15

Keluaran: 15c_Data_Hipotetik_Gist_10000_ANN_LM.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "15c_Data_Hipotetik_Gist_10000_ANN_LM.ipynb"
GIST_URL = ("https://gist.githubusercontent.com/kelvindesman/814fc8bcde1d9768ca0cffa0f248e567/raw/"
            "e4cf96a4f31440968706048c98d2df456816e63c/gistfile1.txt")
GIST_SHA256 = "23d39c08d38b4028b548dd13c3636f0432586637a2f77c75b9f5666d08452871"

SRC11 = (HERE / "build_11_annlm_real_notebook.py").read_text()
SRC14 = (HERE / "build_14_lengkap_fault_baru_notebook.py").read_text()
SRC15 = (HERE / "build_15_data_hipotetik_ann_lm_notebook.py").read_text()


def between(src, start, end):
    i = src.find(start)
    assert i >= 0, f"penanda tidak ditemukan: {start!r}"
    j = src.find(end, i)
    assert j > i, f"penanda akhir tidak ditemukan: {end!r}"
    return src[i:j]


NEW_SIM = re.search(r"NEW_SIM = r'''(.*?)'''", SRC14, re.S).group(1)
assert "def simulate_drift_fault" in NEW_SIM and "FAULT_BIT" in NEW_SIM
LM_CODE = between(SRC11, "def lm_init_layers(", "class LMEstimator(")
assert "def lm_fit(" in LM_CODE and "least_squares" not in LM_CODE
MUAT_MODEL_PY = re.search(r"MUAT_MODEL_PY = '''(.*?)'''", SRC15, re.S).group(1)
assert "def deteksi(" in MUAT_MODEL_PY
MUAT_MODEL_PY = (MUAT_MODEL_PY.replace("notebook 15", "notebook 15c")
                 .replace("muat_model_15", "muat_model_15c")
                 .replace("15_model_gabungan_rata2.npz", "15c_model_gabungan.npz")
                 .replace('# "rata2"), urut waktu, interval +-30 detik, NaN boleh (data hilang).',
                          '# "rata2"), urut waktu, NaN boleh (data hilang).'))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# 15c — Program data hipotetik berlabel + ANN-LM, pada data gist 10.000 titik

**Permintaan (Bu Luh, 22/11/26):** jalankan program *data-hipotetik-berlabel-ann-lm*
(notebook 15) memakai **data 10.000 titik yang di-generate**, yaitu
`gistfile1.txt` (gist `814fc8bc…`), data yang dipakai di `final-2_3_sept_2025.ipynb`.

**Yang sama dengan notebook 15:** simulator fault S1–S5 (16 kondisi, tingkat
"sedang"), label per sampel (bitmask drift = 1, spike = 2, bias = 4,
hardware = 8), fitur lag tanpa entropi, ANN dilatih **Levenberg–Marquardt
asli** (Jacobian analitik), 1 layer (6) dan 2 layer (6,3), ambang alarm
dipilih di validasi, model gabungan + grid search, model disimpan `.npz`.

**Yang berbeda, karena bentuk datanya berbeda:**

| Bagian | Notebook 15 | Notebook ini (data gist) |
|---|---|---|
| Data | `tabel_sensor4_generated.csv`, 4 sensor, ±60 rb sampel, interval 30 detik | `gistfile1.txt`, **10.000 titik**, **1 kolom** `kelembaban_tanah_persen`, interval tidak teratur |
| Versi | 4 sensor terpisah + rata-rata | **1 sinyal** saja (tidak ada data 4 sensor) |
| Panjang episode drift/bias | 3 hari = 8.640 sampel | **1.440 sampel** (±14 % data, sama dengan paket S1 notebook 18) |
| Blok pembagian data | 1 hari | **200 sampel**, bergiliran latih, latih, latih, validasi, uji |

Perbandingan dengan hasil lama `final-2_3_sept_2025` ada di §8.
"""))

cells.append(code(r"""import os, time, json, io, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from IPython.display import FileLink, display

NOTEBOOK_START = time.time()
SMOKE = os.environ.get("SMOKE", "0") == "1"      # uji cepat lokal: iterasi sedikit
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
cells.append(md(r"""# §1 — Data sumber: `gistfile1.txt` (10.000 titik)

Isi file: `id, timestamp, suhu_udara_c, kelembaban_udara_persen,
kelembaban_tanah_persen, status_waktu, status_kesehatan, label`. Yang dipakai
hanya `kelembaban_tanah_persen` (sama dengan `final-2`). Kolom `label` di file
semuanya 0, jadi label fault dibuat sendiri lewat fault injection di §3.

Isi file diperiksa dengan SHA-256 supaya pasti data yang sama dengan
`final-2`. Ada tiga hal yang perlu diperhatikan:

1. **Baris di file tidak urut waktu.** `final-2` memakai urutan file apa adanya.
   Fitur lag menganggap baris berurutan waktu, jadi di sini data **diurutkan
   menurut `timestamp`** (stabil, lalu menurut `id`). Semua 10.000 baris tetap
   dipakai, termasuk stempel waktu kembar.
2. **Interval tidak teratur** (median ±75 detik, ada celah sampai ±3,5 jam).
   Model membaca urutan sampel, bukan jam, jadi episode dan blok dihitung
   dalam **jumlah sampel**.
3. **Rentang nilai.** Nilai di file berkisar ±0–103 %. Beberapa nilai di luar
   0–100 % dipotong ke 0–100 karena kelembaban dalam persen tidak bisa di luar
   itu, dan simulator juga menjaga domain 0–100. Jumlahnya dicetak di bawah.
"""))

cells.append(code(r"""DATA_NAME = "gistfile1.txt"
GIST_URL = "%s"
GIST_SHA256 = "%s"
BASE = "kelembaban_tanah_persen"

def load_bytes():
    for p in [Path(DATA_NAME), *Path("/kaggle/input").glob(f"**/{DATA_NAME}")]:
        if p.exists():
            return p.read_bytes(), str(p)
    import requests
    r = requests.get(GIST_URL, timeout=120); r.raise_for_status()
    return r.content, GIST_URL

raw_bytes, sumber = load_bytes()
sha = hashlib.sha256(raw_bytes).hexdigest()
print("sumber :", sumber); print("sha256 :", sha)
assert sha == GIST_SHA256, "isi gistfile1.txt berbeda dari yang dipakai final-2"
raw = pd.read_csv(io.BytesIO(raw_bytes))
assert len(raw) == 10_000

ts = pd.to_datetime(raw["timestamp"])
urut_file = bool(ts.is_monotonic_increasing)
df = (raw.assign(timestamp=ts).sort_values(["timestamp", "id"], kind="mergesort")
         .reset_index(drop=True))
x_file = df[BASE].to_numpy(dtype=float)
n_bawah, n_atas = int((x_file < 0).sum()), int((x_file > 100).sum())
x_clean = np.clip(x_file, 0.0, 100.0)
T = len(x_clean)
dt = df["timestamp"].diff().dt.total_seconds()
qc = pd.DataFrame([{
    "baris": T, "urut_waktu_di_file": urut_file, "stempel_waktu_kembar": int(df["timestamp"].duplicated().sum()),
    "mulai": df["timestamp"].iloc[0], "selesai": df["timestamp"].iloc[-1],
    "interval_median_s": float(dt.median()), "interval_maks_jam": round(float(dt.max()) / 3600, 2),
    "NaN": int(np.isnan(x_file).sum()), "nilai_<0": n_bawah, "nilai_>100": n_atas,
    "min": float(x_file.min()), "maks": float(x_file.max()), "rata2": float(x_file.mean()),
    "persen_titik_di_60_80": round(100 * float(((x_file >= 60) & (x_file <= 80)).mean()), 1),
}])
display(qc.T.rename(columns={0: "nilai"}))
assert not np.isnan(x_file).any()

fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
axes[0].plot(x_clean, lw=0.5); axes[0].set_title("kelembaban tanah, urut waktu (setelah dipotong 0-100)")
axes[0].set_xlabel("indeks sampel"); axes[0].set_ylabel("%%")
axes[1].hist(x_file, bins=60); axes[1].axvspan(60, 80, color="tab:green", alpha=0.15, label="60-80 %%")
axes[1].set_title("sebaran nilai di file"); axes[1].legend()
fig.tight_layout(); fig.savefig(EXPORT_DIR / "15c_data_sumber.png", dpi=100); plt.show()
""" % (GIST_URL, GIST_SHA256)))

# ---------------------------------------------------------------- §2 simulator
cells.append(md(r"""# §2 — Simulator fault (sama dengan notebook 15)

| Fault | Bentuk | Parameter |
|---|---|---|
| **drift** | naik linear dari 0 ke 5 poin selama satu episode, lalu reset | 3 episode × 1.440 sampel, mulai di 10 %, 40 %, 70 % data |
| **bias** | offset +5 poin selama episode, lalu kembali normal | episode sama dengan drift |
| **spike** | `x(t)·(1 + 0,30)` di posisi acak | 1,52 % titik |
| **hardware** | nilai macet 8 % titik, data hilang (NaN) 5 % titik | — |
| **kombinasi** | semua komponen pada interval episode yang **sama** | — |

| Skenario | Isi | Kondisi |
|---|---|---|
| S1 | satu fault, jenisnya dipilih acak (seed tetap) | 1 |
| S2 | fault tunggal | 4 |
| S3 | dua fault | 6 |
| S4 | tiga fault | 4 |
| S5 | empat fault | 1 |
"""))

cells.append(code(r"""SAMPLING_SECONDS = 30                 # hanya untuk mengubah "hari" -> jumlah sampel di simulator
DRIFT_EPISODE_DAYS = 0.5              # 0,5 x 86400 / 30 = 1.440 sampel per episode
DRIFT_N_EPISODES = 3
DRIFT_SAMPLE_SECONDS = SAMPLING_SECONDS
DRIFT_TOTAL_POINTS = 5.0
BIAS_EPISODE_POINTS = 5.0
SPIKE_REL = 0.30
SPIKE_DENSITY = 0.0152
EPISODE_LEN = int(round(DRIFT_EPISODE_DAYS * 86400 / DRIFT_SAMPLE_SECONDS))

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
assert len(CONDITIONS) == 16 and EPISODE_LEN == 1440

def bits_to_text(b):
    b = int(b)
    return "normal" if b == 0 else "+".join(n for n, v in FAULT_BIT.items() if b & v)

print(pd.Series({k: v[0] for k, v in CONDITIONS.items()}).value_counts().sort_index())
""" % NEW_SIM))

# ---------------------------------------------------------------- §3 data berlabel
cells.append(md(r"""# §3 — Data hipotetik berlabel

Untuk tiap kondisi dibuat satu salinan 10.000 titik dengan fault disuntikkan
(seed tetap). Nilai hilang akibat hardware malfunction dibiarkan NaN; nilai
bersih ada di kolom `*_asli`. Satu file per skenario
`15c_data_berlabel_S1..S5.csv.gz`, plus `15c_data_berlabel_S0_normal.csv.gz`.
"""))

cells.append(code(r"""GEN = {}
rng_master = np.random.default_rng(RANDOM_SEED)
for cond, (scen, faults) in CONDITIONS.items():
    seed = int(rng_master.integers(1_000_000_000))
    y, m = simulate_multiple_faults(x_clean.copy(), faults, seed=seed + 7)
    GEN[cond] = {"scen": scen, "y": y, "m": m}

for cond, g in GEN.items():          # sumber: definisi simulator notebook 14
    changed = ~np.isclose(g["y"], x_clean)
    assert not (changed & (g["m"] == 0)).any(), f"{cond}: nilai berubah tanpa label"
    assert 0 < (g["m"] > 0).sum() < T, cond
    v = g["y"][~np.isnan(g["y"])]; assert v.min() >= 0 and v.max() <= 100, cond
print("Asersi label lolos untuk 16 kondisi.")

def frame(cond):
    g = GEN[cond]
    return pd.DataFrame({"timestamp": df["timestamp"], "id_gist": df["id"], "skenario": g["scen"], "kondisi": cond,
                         BASE: np.round(g["y"], 4), f"{BASE}_asli": np.round(x_clean, 4),
                         "label_fault": (g["m"] > 0).astype(np.int8),
                         "jenis_fault": pd.Series(g["m"]).map(bits_to_text).to_numpy()})

export_df(pd.DataFrame({"timestamp": df["timestamp"], "id_gist": df["id"], "skenario": "S0", "kondisi": "normal",
                        BASE: x_clean, "label_fault": np.zeros(T, np.int8)}),
          "15c_data_berlabel_S0_normal", gz=True)
for scen in SCENARIO_ORDER:
    export_df(pd.concat([frame(c) for c, (s, _) in CONDITIONS.items() if s == scen], ignore_index=True),
              f"15c_data_berlabel_{scen}", gz=True)

ringkasan_label = pd.DataFrame([{
    "skenario": g["scen"], "kondisi": c, "n_sampel": T,
    "fault_%": round(100 * float((g["m"] > 0).mean()), 2), "NaN": int(np.isnan(g["y"]).sum()),
    "jenis_terbanyak": pd.Series(g["m"][g["m"] > 0]).map(bits_to_text).value_counts().index[0]}
    for c, g in GEN.items()])
export_df(ringkasan_label, "15c_ringkasan_label")
display(ringkasan_label)

fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=True)
e0 = int(0.10 * T); a, b = max(0, e0 - 300), min(T, e0 + EPISODE_LEN + 300)
for ax, cond in zip(axes.ravel(), CONDITIONS):
    g = GEN[cond]; t = np.arange(a, b)
    ax.plot(t, x_clean[a:b], lw=0.6, color="0.6", label="asli")
    ax.plot(t, g["y"][a:b], lw=0.6, color="tab:blue", label="setelah fault")
    idx = a + np.flatnonzero(g["m"][a:b] > 0)
    ax.scatter(idx, np.nan_to_num(g["y"][idx], nan=x_clean[idx]), s=4, color="tab:red", label="label fault", zorder=3)
    ax.set_title(f"{g['scen']} {cond}", fontsize=9)
axes[0, 0].legend(fontsize=7, markerscale=4)
fig.suptitle("Data gist 10.000 titik setelah fault injection (sekitar episode pertama)", fontsize=13)
fig.tight_layout(); fig.savefig(EXPORT_DIR / "15c_contoh_sinyal.png", dpi=100); plt.show()
"""))

# ---------------------------------------------------------------- §4 fitur
cells.append(md(r"""# §4 — Fitur & pembagian data

Fitur per sampel sama dengan notebook 15: `lag1..lag10`, `diff`, `isna`,
`isna_10`. Data dipotong per **blok 200 sampel**, dan blok dibagi bergiliran:
latih, latih, latih, validasi, uji (3:1:1 ≈ 60/20/20). Tiap episode
1.440 sampel kebagian ke ketiga bagian. `StandardScaler` hanya di-*fit* di
bagian latih. Ambang alarm dipilih di validasi, lalu hasil dilaporkan di uji.
"""))

cells.append(code(r"""LAG_K = 10
BLOCK_LEN = 200
SPLIT_PATTERN = np.array([0, 0, 0, 1, 2])               # 0 latih, 1 validasi, 2 uji
SPLIT_NAME = {0: "latih", 1: "validasi", 2: "uji"}
split_of = SPLIT_PATTERN[(np.arange(T) // BLOCK_LEN) % len(SPLIT_PATTERN)]
valid_row = np.arange(T) >= LAG_K
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
    return np.column_stack(cols)

# episode drift/bias harus muncul di ketiga bagian supaya evaluasi bermakna
ep_mask = GEN["drift"]["m"] > 0
print("sampel episode per bagian:", {SPLIT_NAME[k]: int((ep_mask & (split_of == k)).sum()) for k in range(3)})
assert all((ep_mask & (split_of == k)).sum() > 0 for k in range(3))
"""))

# ---------------------------------------------------------------- §5 LM
cells.append(md(r"""# §5 — ANN-LM asli

Kode Levenberg–Marquardt identik dengan notebook 10/11/14/15: Jacobian
analitik, langkah `(JᵀJ + λ·diag(JᵀJ))·Δ = −Jᵀr`, λ diperbarui aturan Nielsen.
Keluaran 2 unit one-hot (normal, fault) + softmax. Untuk dua kelas ini setara
dengan satu keluaran sigmoid seperti program pertama.
"""))

cells.append(code("import time as _time\n\n" + LM_CODE + r"""
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
MAX_TRAIN_ROWS = 120000
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

def build_xy(cond):
    g = GEN[cond]
    F = lag_features(g["y"]); y = (g["m"] > 0).astype(int)
    return F[valid_row], y[valid_row], split_of[valid_row], np.flatnonzero(valid_row)

def fit_lm_model(F, y, sp, hidden, seed):
    tr = np.flatnonzero(sp == 0)
    if len(tr) > MAX_TRAIN_ROWS:
        tr = np.sort(np.random.default_rng(seed).choice(tr, MAX_TRAIN_ROWS, replace=False))
    scaler = StandardScaler().fit(F[tr])
    Z = scaler.transform(F)
    c0 = time.process_time(); t0 = time.perf_counter()
    res = lm_fit(Z[tr], np.eye(2)[y[tr]], Z.shape[1], hidden, 2, activation="tanh",
                 iters=LM_ITERS, seed=seed, max_train_s=LM_MAX_TRAIN_S)
    wall = time.perf_counter() - t0; cpu = time.process_time() - c0
    p = lm_predict_proba(res["theta"], res["template"], Z)[:, 1]
    va = sp == 1
    thr, f1_val = best_threshold(p[va], y[va])
    model = {"theta": res["theta"], "shapes": np.array([W.shape for W, _ in res["template"]]),
             "mean": scaler.mean_, "scale": scaler.scale_, "ambang": thr, "activation": "tanh",
             "hidden": np.array(hidden), "lag_k": LAG_K, "feature_names": np.array(FEATURE_NAMES)}
    return model, p, thr, f1_val, res, tr, wall, cpu

RESULTS, PREDS, HISTS, MODELS = [], {}, {}, {}
for ci, cond in enumerate(CONDITIONS):
    F, y, sp, ti = build_xy(cond)
    for arch_name, hidden in ARCHS.items():
        model, p, thr, f1_val, res, tr, wall, cpu = fit_lm_model(F, y, sp, hidden, RANDOM_SEED + ci)
        te = sp == 2; yhat = (p > thr).astype(int)
        RESULTS.append({"skenario": CONDITIONS[cond][0], "kondisi": cond, "arsitektur": arch_name,
                        "n_latih": len(tr), "n_uji": int(te.sum()), "fault_uji_%": round(100 * y[te].mean(), 2),
                        "ambang": thr, "F1_validasi": round(f1_val, 4),
                        "akurasi": accuracy_score(y[te], yhat[te]),
                        "precision": precision_score(y[te], yhat[te], zero_division=0),
                        "recall": recall_score(y[te], yhat[te], zero_division=0),
                        "F1": f1_score(y[te], yhat[te], zero_division=0),
                        "ROC_AUC": safe_auc(y[te], p[te]),
                        "FPR": float(yhat[te & (y == 0)].mean()),
                        "lm_iter": res["hist"]["iters"], "lm_status": res["hist"]["status"],
                        "train_mse": res["train_mse"], "wall_s": wall, "cpu_s": cpu})
        MODELS[(cond, arch_name)] = model; HISTS[(cond, arch_name)] = res["hist"]
        PREDS[(cond, arch_name)] = {"t": ti, "split": sp, "label": y, "prob": p, "alarm": yhat, "thr": thr}
        r = RESULTS[-1]
        print(f"[{time.time()-NOTEBOOK_START:6.0f}s] {r['skenario']} {cond:26s} {arch_name:14s} "
              f"F1={r['F1']:.3f} AUC={r['ROC_AUC']:.3f} FPR={r['FPR']:.3f} iter={r['lm_iter']} wall={wall:.1f}s",
              flush=True)

hasil = pd.DataFrame(RESULTS)
for c in ["akurasi", "precision", "recall", "F1", "ROC_AUC", "FPR", "train_mse", "wall_s", "cpu_s"]:
    hasil[c] = hasil[c].astype(float).round(4)
export_df(hasil, "15c_hasil_ann_lm")
"""))

# ---------------------------------------------------------------- §6 hasil
cells.append(md(r"""# §6 — Hasil per skenario dan prediksi per sampel

F1 di bagian **uji**, rata-rata kondisi dalam satu skenario. Lalu prediksi
model 2 layer (6,3) untuk seluruh 10.000 titik tiap kondisi (`prob_fault`,
`alarm`, `split`) untuk sistem notifikasi.
"""))

cells.append(code(r"""ringkas = (hasil.groupby(["arsitektur", "skenario"])[["precision", "recall", "F1", "ROC_AUC", "FPR"]]
           .mean().round(3).reset_index())
export_df(ringkas, "15c_hasil_ringkas_per_skenario")
display(ringkas.pivot_table(index="arsitektur", columns="skenario", values="F1"))
display(hasil[["skenario", "kondisi", "arsitektur", "fault_uji_%", "precision", "recall", "F1",
               "ROC_AUC", "FPR", "lm_iter", "wall_s"]])

fig, ax = plt.subplots(figsize=(7, 4))
for cond in ["faulty", "drift", "bias+malfunc", "drift+bias+malfunc", "spike+bias+malfunc+drift"]:
    ax.plot(HISTS[(cond, "2 layer (6,3)")]["loss"], label=f"{CONDITIONS[cond][0]} {cond}")
ax.set_yscale("log"); ax.set_xlabel("iterasi LM"); ax.set_ylabel("MSE latih"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(EXPORT_DIR / "15c_kurva_loss_lm.png", dpi=90); plt.show()

for scen in SCENARIO_ORDER:
    parts = []
    for cond in [c for c, (s, _) in CONDITIONS.items() if s == scen]:
        pr = PREDS[(cond, "2 layer (6,3)")]
        parts.append(pd.DataFrame({"timestamp": df["timestamp"].to_numpy()[pr["t"]], "skenario": scen,
                                   "kondisi": cond, "split": pd.Series(pr["split"]).map(SPLIT_NAME).to_numpy(),
                                   "label_fault": pr["label"], "prob_fault": np.round(pr["prob"], 5),
                                   "ambang": round(pr["thr"], 3), "alarm": pr["alarm"]}))
    export_df(pd.concat(parts, ignore_index=True), f"15c_prediksi_{scen}", gz=True)
"""))

# ---------------------------------------------------------------- §7 gabungan
cells.append(md(r"""# §7 — Model gabungan (grid search) & menyimpan model

Satu model dilatih dari **semua 16 kondisi + data normal** sekaligus (sistem
notifikasi tidak tahu jenis gangguannya lebih dulu). Arsitektur dipilih dengan
grid search berdasarkan F1 validasi. Model disimpan sebagai `.npz` numpy murni,
lalu dimuat ulang lewat `15c_muat_model.py` dan dicek identik.
"""))

cells.append(code(r"""GABUNGAN_GRID = ({"1 layer (6)": [6], "2 layer (6,3)": [6, 3]} if SMOKE else
                 {"1 layer (6)": [6], "1 layer (12)": [12], "2 layer (6,3)": [6, 3],
                  "2 layer (12,6)": [12, 6], "2 layer (16,8)": [16, 8]})
parts = [("normal", lag_features(x_clean)[valid_row], np.zeros(valid_row.sum(), int), split_of[valid_row])]
for cond in CONDITIONS:
    F, y, sp, _ = build_xy(cond); parts.append((cond, F, y, sp))
F = np.vstack([p_[1] for p_ in parts]); y = np.concatenate([p_[2] for p_ in parts])
sp = np.concatenate([p_[3] for p_ in parts])
cond_of = np.concatenate([np.full(len(p_[2]), p_[0], dtype=object) for p_ in parts])

GRID_ROWS, best = [], None
for arch_name, hidden in GABUNGAN_GRID.items():
    fit = fit_lm_model(F, y, sp, hidden, seed=RANDOM_SEED + 100)
    model_, p_, thr_, f1v_, res_ = fit[:5]
    te_ = sp == 2; yh_ = (p_ > thr_).astype(int)
    GRID_ROWS.append({"arsitektur": arch_name, "n_parameter": int(len(model_["theta"])), "ambang": thr_,
                      "F1_validasi": f1v_, "F1_uji": f1_score(y[te_], yh_[te_], zero_division=0),
                      "AUC_uji": safe_auc(y[te_], p_[te_]), "lm_iter": res_["hist"]["iters"],
                      "lm_status": res_["hist"]["status"], "wall_s": fit[6]})
    print(f"[{time.time()-NOTEBOOK_START:6.0f}s] grid {arch_name:15s} F1 val={f1v_:.3f} "
          f"uji={GRID_ROWS[-1]['F1_uji']:.3f} wall={fit[6]:.0f}s", flush=True)
    if best is None or f1v_ > best[0]:
        best = (f1v_, arch_name, fit)
_, GAB_ARCH, (model, p, thr, f1_val, res, tr, wall, cpu) = best
print("-> dipilih (F1 validasi tertinggi):", GAB_ARCH)
MODELS[("gabungan", GAB_ARCH)] = model
te = sp == 2; yhat = (p > thr).astype(int)
hasil_gabungan = pd.DataFrame([{
    "arsitektur": GAB_ARCH, "F1_validasi": f1_val, "n_latih": len(tr), "n_uji": int(te.sum()),
    "fault_uji_%": round(100 * y[te].mean(), 2), "ambang": thr,
    "precision": precision_score(y[te], yhat[te], zero_division=0),
    "recall": recall_score(y[te], yhat[te], zero_division=0),
    "F1": f1_score(y[te], yhat[te], zero_division=0), "ROC_AUC": safe_auc(y[te], p[te]),
    "FPR_data_normal": float(yhat[te & (cond_of == "normal")].mean()), "lm_iter": res["hist"]["iters"],
    "wall_s": wall}]).round(4)
gab_per_kondisi = []
for cond in ["normal", *CONDITIONS]:
    mk = te & (cond_of == cond); yt, yh = y[mk], yhat[mk]
    gab_per_kondisi.append({"skenario": "S0" if cond == "normal" else CONDITIONS[cond][0], "kondisi": cond,
                            "n_fault_uji": int(yt.sum()),
                            "recall": recall_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                            "F1": f1_score(yt, yh, zero_division=0) if yt.sum() else np.nan,
                            "FPR": float(yh[yt == 0].mean()) if (yt == 0).any() else np.nan})
gab_per_kondisi = pd.DataFrame(gab_per_kondisi).round(4)
grid_gabungan = pd.DataFrame(GRID_ROWS).round(4)
for d_, n_ in [(grid_gabungan, "15c_grid_search_gabungan"), (hasil_gabungan, "15c_hasil_model_gabungan"),
               (gab_per_kondisi, "15c_hasil_model_gabungan_per_kondisi")]:
    export_df(d_, n_); display(d_)
"""))

cells.append(code(r"""MUAT_MODEL_PY = '''%s'''
(EXPORT_DIR / "15c_muat_model.py").write_text(MUAT_MODEL_PY)
import importlib.util, zipfile
_spec = importlib.util.spec_from_file_location("muat_model_15c", EXPORT_DIR / "15c_muat_model.py")
muat = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(muat)

MODEL_DIR = EXPORT_DIR / "model_15c"; MODEL_DIR.mkdir(exist_ok=True)
x_cek = GEN["drift+bias"]["y"]
index_rows = []
for (cond, arch), m in MODELS.items():
    a = "2layer" if arch.startswith("2") else "1layer"
    fn = "15c_model_gabungan.npz" if cond == "gabungan" else f"15c_model_{cond.replace('+', '_')}_{a}.npz"
    target = (EXPORT_DIR if cond == "gabungan" else MODEL_DIR) / fn
    tpl = [[np.zeros(tuple(int(v) for v in s)), np.zeros(int(s[1]))] for s in m["shapes"]]
    p_mem = lm_predict_proba(m["theta"], tpl, (lag_features(x_cek) - m["mean"]) / m["scale"])[:, 1]
    extra = {"contoh_input": x_cek[:2000], "contoh_prob": p_mem[:2000]} if cond == "gabungan" else {}
    np.savez(target, kondisi=cond, arsitektur=arch, **m, **extra)
    p_file, _ = muat.deteksi(muat.muat_model(target), x_cek)
    diff = float(np.max(np.abs(p_file[valid_row] - p_mem[valid_row])))
    assert diff < 1e-9, f"{fn}: prediksi file != memori ({diff})"
    if cond != "gabungan":
        pr = PREDS[(cond, arch)]
        p_own, _ = muat.deteksi(muat.muat_model(target), GEN[cond]["y"])
        diff = max(diff, float(np.max(np.abs(p_own[pr["t"]] - pr["prob"]))))
        assert diff < 1e-9, f"{fn}: prediksi file != prediksi §5 ({diff})"
    index_rows.append({"file": fn, "kondisi": cond, "arsitektur": arch, "ambang": round(float(m["ambang"]), 3),
                       "selisih_maks_prob_saat_muat_ulang": diff})
model_index = pd.DataFrame(index_rows)
model_index.to_csv(MODEL_DIR / "15c_daftar_model.csv", index=False)
(MODEL_DIR / "15c_muat_model.py").write_text(MUAT_MODEL_PY)
with zipfile.ZipFile(EXPORT_DIR / "15c_model_ann_lm_semua.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(MODEL_DIR.iterdir()):
        zf.write(f, f.name)
for f in sorted(MODEL_DIR.iterdir()):
    f.unlink()
MODEL_DIR.rmdir()
export_df(model_index, "15c_daftar_model")
print(f"{len(model_index)} model disimpan & dimuat ulang; selisih prob maks = "
      f"{model_index['selisih_maks_prob_saat_muat_ulang'].max():.2e}")
""" % MUAT_MODEL_PY))

# ---------------------------------------------------------------- §8 banding
cells.append(md(r"""# §8 — Dibanding hasil lama `final-2_3_sept_2025`

Hasil lama di `final-2` (bagian "NN Grid search") memberi F1 ≈ 0,99–1,00 di
semua skenario. Angka itu **tidak bisa dibandingkan langsung** dengan tabel
§6, karena di sel itu target yang dipakai bukan label fault:

* fungsi `_prepare_df_for_grid` mencari kolom bernama `label`. Kolom itu ada di
  gist, tetapi isinya semua 0 (satu kelas), jadi program membuat
  **`label_auto` = 1 bila nilai kelembaban ≥ kuantil 80 %** (log:
  `Using synthetic 'label_auto' | thr=83.3450 | pos_rate=0.200`). Model
  diminta menebak "nilai sedang tinggi", bukan "sampel terkena fault". Tugas
  itu mudah karena fitur lag hampir sama dengan nilai itu sendiri;
* baris tidak diurutkan waktu, dan pembagian 60/20/20 berurutan.

Notebook ini memakai label fault hasil injeksi (`label_fault`), urutan waktu,
dan blok bergiliran, sehingga angkanya mengukur deteksi gangguan yang
sebenarnya.
"""))

cells.append(code(r"""import datetime
FINAL2_F1 = {"S1": 0.9962, "S2": 0.9949, "S3": 0.9923, "S4": 1.0000, "S5": 0.9961}   # sumber: output sel NN Grid search final-2
banding = pd.DataFrame({
    "final-2 (target label_auto = nilai >= kuantil 80%)": FINAL2_F1,
    "15c 1 layer (6), target label_fault": ringkas[ringkas["arsitektur"] == "1 layer (6)"].set_index("skenario")["F1"],
    "15c 2 layer (6,3), target label_fault": ringkas[ringkas["arsitektur"] == "2 layer (6,3)"].set_index("skenario")["F1"],
}).round(3)
export_df(banding.reset_index(names="skenario"), "15c_banding_final2")
display(banding)

MODEL_CARD = {
    "nama": "ANN-LM deteksi gangguan, data gist 10.000 titik (1 sinyal, tanpa entropi)",
    "dibuat": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "data_sumber": {"file": "gistfile1.txt", "sha256": GIST_SHA256, "baris": T,
                    "diurutkan_menurut_timestamp": True, "nilai_dipotong_ke_0_100": n_bawah + n_atas},
    "fault_injection": {"drift_total_poin_per_episode": DRIFT_TOTAL_POINTS, "bias_poin_per_episode": BIAS_EPISODE_POINTS,
                        "spike_relatif": SPIKE_REL, "spike_kepadatan": SPIKE_DENSITY,
                        "hardware": "8% macet + 5% hilang (NaN)",
                        "episode": f"{DRIFT_N_EPISODES} x {EPISODE_LEN} sampel (posisi 10/40/70%)"},
    "fitur": FEATURE_NAMES, "pembagian_data": f"blok {BLOCK_LEN} sampel bergiliran latih,latih,latih,validasi,uji",
    "lm_iterasi_maks": LM_ITERS,
    "model_gabungan": {"file": "15c_model_gabungan.npz", "arsitektur": GAB_ARCH,
                       "ambang": round(float(MODELS[("gabungan", GAB_ARCH)]["ambang"]), 4),
                       "hasil_uji": hasil_gabungan.iloc[0][["precision", "recall", "F1", "ROC_AUC",
                                                            "FPR_data_normal"]].astype(float).to_dict()},
    "F1_uji_per_skenario": ringkas.pivot_table(index="arsitektur", columns="skenario", values="F1").to_dict("index"),
}
(EXPORT_DIR / "15c_model_card.json").write_text(json.dumps(MODEL_CARD, indent=1, ensure_ascii=False, default=float))
files = sorted(EXPORT_DIR.glob("15c_*"))
display(pd.DataFrame({"file": [f.name for f in files], "ukuran_MB": [round(f.stat().st_size / 1e6, 2) for f in files]}))
_chk = pd.read_csv(EXPORT_DIR / "15c_data_berlabel_S5.csv.gz")
assert len(_chk) == T and _chk["label_fault"].isin([0, 1]).all()
print(f"Selesai. Wall total {time.time()-NOTEBOOK_START:.0f} s, {len(files)} file di {EXPORT_DIR}/")
"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, "| cells:", len(cells))
