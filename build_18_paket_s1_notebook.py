#!/usr/bin/env python3
"""Build 18_Paket_S1_Data_Hipotetik_10000.ipynb.

Request (Kelvin chat 2026-11-21, pasted checklist "untuk mhs s1"):
  1. data hipotetik berlabel (10.000 data)
  2. model untuk program 1: informasi preprocessing (normalisasi data, fitur
     input), contoh penggunaan model beserta output prediksinya, sebelum
     diintegrasikan ke backend.

The notebook produces one self-contained package folder `paket_s1/`:
  * data_hipotetik_berlabel_10000.csv  exactly 10,000 rows, 4 sensors, labels per sensor
  * kamus_kolom.csv                    column dictionary
  * model_ann_lm_4sensor.npz           NB15 combined ANN-LM model (one sensor per call)
  * model_ann_lm_rata2.npz             NB15 combined model for the 4-sensor average
  * muat_model.py                      loader (numpy + pandas only), same text as NB15
  * contoh_backend.py                  streaming example: one reading in -> JSON out
  * prediksi_10000.csv                 model output on the 10,000 rows
  * model_card.json, README_S1.md

Reused text (read from other builders, not copied by hand):
  * fault simulator, "sedang" level   -> NEW_SIM in build_14
  * model loader 15_muat_model.py     -> MUAT_MODEL_PY in build_15

Output: 18_Paket_S1_Data_Hipotetik_10000.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "18_Paket_S1_Data_Hipotetik_10000.ipynb"
REPO_RAW = ("https://raw.githubusercontent.com/kelvindesman/public-files/"
            "refs/heads/main/tabel_sensor4_generated.csv")

SRC14 = (HERE / "build_14_lengkap_fault_baru_notebook.py").read_text()
SRC15 = (HERE / "build_15_data_hipotetik_ann_lm_notebook.py").read_text()
NEW_SIM = re.search(r"NEW_SIM = r'''(.*?)'''", SRC14, re.S).group(1)
LOADER = re.search(r"MUAT_MODEL_PY = '''(.*?)'''", SRC15, re.S).group(1)
assert "def simulate_drift_fault" in NEW_SIM and "FAULT_BIT" in NEW_SIM
assert "def deteksi(" in LOADER and "def muat_model(" in LOADER

BACKEND = r'''"""Contoh integrasi backend: satu pembacaan sensor masuk -> prob_fault + alarm (JSON).

Model menilai SATU deret sensor. Untuk sampel terbaru model butuh 10 nilai
sebelumnya (lag1..lag10), jadi backend menyimpan riwayat pendek per sensor.

    from contoh_backend import DetektorSensor
    det = DetektorSensor("model_ann_lm_4sensor.npz")
    hasil = det.proses("kelembaban1", 51.27)      # dipanggil tiap pembacaan baru (30 detik)
    # {"sensor": "kelembaban1", "nilai": 51.27, "prob_fault": 0.0412, "alarm": false, "siap": true}

Nilai hilang (sensor tidak mengirim data) dikirim sebagai None.
Jalankan langsung untuk demo:  python contoh_backend.py data_hipotetik_berlabel_10000.csv
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muat_model import muat_model, deteksi  # noqa: E402

PANJANG_RIWAYAT = 64   # cukup > lag_k; hanya untuk mengisi nilai hilang ke depan (ffill)


class DetektorSensor:
    def __init__(self, path_model):
        self.m = muat_model(path_model)
        self.lag_k = int(self.m["lag_k"])
        self.ambang = float(self.m["ambang"])
        self.riwayat = {}

    def proses(self, sensor_id, nilai):
        buf = self.riwayat.setdefault(sensor_id, deque(maxlen=PANJANG_RIWAYAT))
        buf.append(np.nan if nilai is None else float(nilai))
        prob, alarm = deteksi(self.m, np.array(buf))
        return {"sensor": sensor_id,
                "nilai": None if nilai is None else float(nilai),
                "prob_fault": round(float(prob[-1]), 4),
                "alarm": bool(alarm[-1]),
                "siap": len(buf) > self.lag_k}   # False = riwayat belum 11 nilai, abaikan hasilnya


if __name__ == "__main__":
    import pandas as pd
    folder = Path(__file__).resolve().parent
    csv = sys.argv[1] if len(sys.argv) > 1 else folder / "data_hipotetik_berlabel_10000.csv"
    data = pd.read_csv(csv)
    det = DetektorSensor(folder / "model_ann_lm_4sensor.npz")
    sensor = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
    n_alarm = {s: 0 for s in sensor}
    tampil = 0
    for _, baris in data.iterrows():                     # simulasi data masuk satu per satu
        for s in sensor:
            v = baris[s]
            h = det.proses(s, None if pd.isna(v) else v)
            n_alarm[s] += int(h["alarm"] and h["siap"])
            if h["alarm"] and h["siap"] and tampil < 5:
                h["timestamp"] = baris["timestamp"]
                print(json.dumps(h))
                tampil += 1
    print(json.dumps({"baris": len(data), "jumlah_alarm": n_alarm}))
'''
assert "class DetektorSensor" in BACKEND


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# 18 — Paket mahasiswa S1: data hipotetik berlabel 10.000 baris + model ANN-LM program 1

Untuk mahasiswa S1 yang akan membangun **backend sistem notifikasi gangguan**
sensor kelembaban tanah. Notebook ini menghasilkan satu folder `paket_s1/`
(juga `paket_s1.zip`) berisi:

| File | Isi |
|---|---|
| `data_hipotetik_berlabel_10000.csv` | **tepat 10.000 baris**, 4 sensor, nilai bersih + nilai ber-fault, label fault per sensor dan jenisnya |
| `kamus_kolom.csv` | arti setiap kolom |
| `model_ann_lm_4sensor.npz` | model ANN-LM program 1 (notebook 15), satu sensor per panggilan |
| `model_ann_lm_rata2.npz` | model yang sama untuk sinyal rata-rata 4 sensor |
| `muat_model.py` | pemuat model, cukup `numpy` + `pandas` |
| `contoh_backend.py` | contoh backend: satu pembacaan masuk → `prob_fault` + `alarm` (JSON) |
| `prediksi_10000.csv` | keluaran model pada 10.000 baris |
| `model_card.json`, `README_S1.md` | ringkasan preprocessing, arsitektur, ambang, batasan |

**Isi notebook**

1. §1 Data sumber (10.000 baris pertama `tabel_sensor4_generated.csv`)
2. §2 Fault injection → data berlabel
3. §3 Preprocessing: fitur input dan normalisasi, dihitung tangan untuk satu sampel
4. §4 Model: arsitektur, forward pass langkah demi langkah, ambang
5. §5 Prediksi pada 10.000 baris + skor terhadap label
6. §6 Contoh backend (streaming), dicek sama dengan prediksi batch
7. §7 README, model card, zip
"""))

cells.append(code(r"""import os, io, re, json, time, shutil, subprocess, sys, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

T0 = time.time()
RANDOM_SEED = 42
N_BARIS = 10_000
PAKET = Path(os.environ.get("PAKET_DIR", "paket_s1")); PAKET.mkdir(parents=True, exist_ok=True)
print("folder paket:", PAKET.resolve())
"""))

# ---------------------------------------------------------------- §1
cells.append(md(r"""# §1 — Data sumber

Sama dengan program pertama dan notebook 15: `tabel_sensor4_generated.csv`,
4 sensor kelembaban tanah (%), interval ±30 detik. Satu baris berstempel 1970
(jam perangkat belum sinkron) dibuang. Diambil **10.000 baris pertama** setelah
itu: ±3,5 hari waktu rekam (10.000 × 30 detik). Rekaman asli punya jeda (alat
mati), jadi rentang kalendernya lebih panjang; sel di bawah mencetak jeda
terpanjang. Seperti notebook 15, jeda dibiarkan dan semua hitungan (lag,
episode) memakai urutan sampel, bukan jam dinding. Sinyal ini dianggap normal;
fault disuntikkan di §2.
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
bad_ts = raw["timestamp"].isna() | (raw["timestamp"] < pd.Timestamp("2020-01-01", tz="UTC"))
df = (raw.loc[~bad_ts, ["timestamp", *SENSORS]].sort_values("timestamp")
         .iloc[:N_BARIS].reset_index(drop=True))
SAMPLING_SECONDS = 30
dt = df["timestamp"].diff().dt.total_seconds()
display(pd.DataFrame([{"baris": len(df), "mulai": df["timestamp"].iloc[0], "selesai": df["timestamp"].iloc[-1],
                       "interval_median_s": float(dt.median()), "jeda_>10_menit": int((dt > 600).sum()),
                       "jeda_terpanjang_jam": round(float(dt.max()) / 3600, 1),
                       "NaN": int(df[SENSORS].isna().sum().sum())}]))
assert len(df) == N_BARIS and df[SENSORS].isna().sum().sum() == 0
assert abs(dt.median() - SAMPLING_SECONDS) <= 2
""" % REPO_RAW))

# ---------------------------------------------------------------- §2
cells.append(md(r"""# §2 — Fault injection → data berlabel

Simulator **sama persis** dengan notebook 14/15 (tingkat "sedang"): drift naik
linear 5 poin per episode lalu reset, bias +5 poin selama episode, spike
`x·1,30` pada 1,52 % titik, hardware 8 % nilai macet + 5 % hilang (NaN).
Kombinasi fault disuntik pada interval episode yang sama.

**Satu-satunya perubahan: panjang episode.** Di notebook 15 satu episode = 3 hari
(60.763 sampel ≈ 21 hari waktu rekam). 10.000 sampel hanya ±3,5 hari, jadi 3 episode × 3 hari tidak
muat. Episode dipendekkan menjadi **0,5 hari waktu rekam (1.440 sampel)**, tetap 3 episode di
posisi 10 %, 40 %, 70 %. Proporsi waktu dalam episode jadi 43 % (notebook 15:
43 %). Akibatnya drift di sini naik lebih curam (5 poin per 12 jam, bukan per
3 hari).

**Tiap sensor diberi skenario berbeda**, supaya satu file 10.000 baris memuat
semua jenis fault dan skenario S2–S5 sekaligus:

| Sensor | Skenario | Kondisi |
|---|---|---|
| kelembaban1 | S2 | drift |
| kelembaban2 | S3 | spike + bias |
| kelembaban3 | S4 | drift + bias + hardware |
| kelembaban4 | S5 | spike + bias + hardware + drift |

Label per sampel disimpan sebagai **bitmask**: drift = 1, spike = 2, bias = 4,
hardware = 8 (contoh 6 = spike + bias). Kolom `jenis_s*` menuliskannya sebagai
teks, kolom `label_s*` = 1 bila ada fault apa pun.
"""))

cells.append(code(r"""DRIFT_EPISODE_DAYS = 0.5          # notebook 15: 3.0 (tidak muat di 10.000 baris)
DRIFT_N_EPISODES = 3
DRIFT_SAMPLE_SECONDS = SAMPLING_SECONDS
DRIFT_TOTAL_POINTS = 5.0
BIAS_EPISODE_POINTS = 5.0
SPIKE_REL = 0.30
SPIKE_DENSITY = 0.0152

%s
_D = {"total": DRIFT_TOTAL_POINTS}
_S = {"k": SPIKE_REL}
_B = {"bias": BIAS_EPISODE_POINTS}
_H = {"stuck_prob": 0.08, "loss_prob": 0.05}

PENUGASAN = {
    "kelembaban1": ("S2", "drift", [(simulate_drift_fault, _D)]),
    "kelembaban2": ("S3", "spike+bias", [(simulate_spike_fault, _S), (simulate_bias_fault, _B)]),
    "kelembaban3": ("S4", "drift+bias+malfunc", [(simulate_drift_fault, _D), (simulate_bias_fault, _B),
                                                 (simulate_hardware_fault, _H)]),
    "kelembaban4": ("S5", "spike+bias+malfunc+drift", [(simulate_spike_fault, _S), (simulate_bias_fault, _B),
                                                       (simulate_hardware_fault, _H), (simulate_drift_fault, _D)]),
}

def bits_to_text(b):
    b = int(b)
    return "normal" if b == 0 else "+".join(n for n, v in FAULT_BIT.items() if b & v)

X = df[SENSORS].to_numpy(float)
Y = X.copy(); M = np.zeros(X.shape, np.uint8)
rng = np.random.default_rng(RANDOM_SEED)
for s, name in enumerate(SENSORS):
    seed = int(rng.integers(1_000_000_000))
    Y[:, s], M[:, s] = simulate_multiple_faults(X[:, s].copy(), PENUGASAN[name][2], seed=seed)

ep_mask, _ = _episode_mask(N_BARIS)
data = pd.DataFrame({"timestamp": df["timestamp"].dt.strftime("%%Y-%%m-%%d %%H:%%M:%%S")})
for s, n in enumerate(SENSORS):
    data[n] = np.round(Y[:, s], 4)
for s, n in enumerate(SENSORS):
    data[f"{n}_asli"] = np.round(X[:, s], 4)
for s in range(4):
    data[f"label_s{s+1}"] = (M[:, s] > 0).astype(np.int8)
for s in range(4):
    data[f"jenis_s{s+1}"] = pd.Series(M[:, s]).map(bits_to_text).to_numpy()
for s, n in enumerate(SENSORS):
    data[f"kondisi_s{s+1}"] = f"{PENUGASAN[n][0]} {PENUGASAN[n][1]}"
lab = M > 0
data["label_fault"] = lab.any(axis=1).astype(np.int8)
data["sensor_rusak"] = [",".join(f"s{s + 1}" for s in np.flatnonzero(r)) for r in lab]
data["dalam_episode"] = ep_mask.astype(np.int8)

# --- asersi (sumber: definisi simulator notebook 14 + permintaan 10.000 data) ---
assert len(data) == N_BARIS
changed = ~np.isclose(Y, X)                                   # NaN dihitung berubah
assert not (changed & ~lab).any(), "nilai berubah tanpa label"
assert (np.isnan(Y) <= ((M & FAULT_BIT["hardware"]) > 0)).all(), "NaN tanpa label hardware"
v = Y[~np.isnan(Y)]; assert v.min() >= 0 and v.max() <= 100
for s in range(4):
    assert 0 < lab[:, s].sum() < N_BARIS, SENSORS[s]
semua = np.bitwise_or.reduce(M.ravel())
assert semua == 15, f"tidak semua jenis fault muncul: {semua}"
assert int(round(DRIFT_EPISODE_DAYS * 86400 / SAMPLING_SECONDS)) == 1440 and ep_mask.sum() == 3 * 1440

data.to_csv(PAKET / "data_hipotetik_berlabel_10000.csv", index=False)

ringkas = pd.DataFrame([{
    "sensor": n, "skenario": PENUGASAN[n][0], "kondisi": PENUGASAN[n][1],
    "baris_fault": int(lab[:, s].sum()), "fault_%%": round(100 * lab[:, s].mean(), 2),
    **{f"{j}_%%": round(100 * ((M[:, s] & b) > 0).mean(), 2) for j, b in FAULT_BIT.items()},
    "NaN": int(np.isnan(Y[:, s]).sum())} for s, n in enumerate(SENSORS)])
display(ringkas)
print(f"label_fault = 1 pada {data['label_fault'].mean():.1%%} baris; episode = {ep_mask.mean():.1%%} waktu")
display(data.iloc[1435:1445, :13])
""" % NEW_SIM))

cells.append(code(r"""fig, axs = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
t = np.arange(N_BARIS) * SAMPLING_SECONDS / 3600
for s, (ax, n) in enumerate(zip(axs, SENSORS)):
    ax.plot(t, X[:, s], color="0.6", lw=0.8, label="asli")
    ax.plot(t, Y[:, s], color="C3", lw=0.6, label="dengan fault")
    ax.fill_between(t, 0, 1, where=lab[:, s], transform=ax.get_xaxis_transform(), color="C3", alpha=0.08)
    ax.set_ylabel(f"s{s+1} (%)"); ax.set_title(f"{n}: {PENUGASAN[n][0]} {PENUGASAN[n][1]}", fontsize=9, loc="left")
axs[0].legend(loc="upper right", fontsize=8); axs[-1].set_xlabel("jam sejak awal data")
fig.tight_layout(); fig.savefig(PAKET / "gambar_data_10000.png", dpi=110); plt.show()
"""))

cells.append(code(r"""KAMUS = pd.DataFrame([
    ("timestamp", "waktu pembacaan (UTC), interval ±30 detik"),
    ("kelembaban1..4", "nilai sensor DENGAN fault (%), NaN = data hilang (gejala kerusakan hardware). Ini masukan model."),
    ("kelembaban1..4_asli", "nilai sensor sebelum fault disuntikkan (%), untuk pembanding"),
    ("label_s1..4", "1 = sampel sensor itu sedang fault, 0 = normal (kunci jawaban)"),
    ("jenis_s1..4", "jenis fault: normal, drift, spike, bias, hardware, atau gabungan seperti spike+bias"),
    ("kondisi_s1..4", "skenario dan kondisi yang disuntikkan ke sensor itu (tetap sepanjang file)"),
    ("label_fault", "1 = minimal satu sensor fault pada baris ini"),
    ("sensor_rusak", "daftar sensor yang fault pada baris ini, mis. s2,s4"),
    ("dalam_episode", "1 = baris di dalam episode gangguan (drift/bias aktif, spike/hardware hanya muncul di sini untuk kombinasi)"),
], columns=["kolom", "arti"])
KAMUS.to_csv(PAKET / "kamus_kolom.csv", index=False)
display(KAMUS)
"""))

# ---------------------------------------------------------------- §3
cells.append(md(r"""# §3 — Preprocessing: fitur input dan normalisasi

Model menilai **satu deret sensor** (satu kolom `kelembaban*`), sampel demi
sampel. Untuk sampel ke-*t* dibuat **13 fitur**:

| Fitur | Rumus | Arti |
|---|---|---|
| `lag1` … `lag10` | nilai sensor pada *t−1* … *t−10* (setelah NaN diisi nilai valid sebelumnya) | pola 5 menit terakhir |
| `diff` | *x(t) − x(t−1)* | lonjakan mendadak (spike) |
| `isna` | 1 bila *x(t)* hilang | data hilang sekarang |
| `isna_10` | jumlah data hilang di 10 sampel sebelumnya | sensor sering putus |

Catatan: nilai *x(t)* sendiri tidak masuk sebagai fitur; ia hanya masuk lewat `diff`.

**Normalisasi** (StandardScaler, dihitung pada data latih notebook 15 dan disimpan
di file model sebagai `mean` dan `scale`):

$$z_j = \frac{f_j - \text{mean}_j}{\text{scale}_j}$$

Backend **wajib memakai mean/scale dari file model**, bukan menghitung ulang dari
data baru. Sel di bawah menampilkan nilainya, lalu menghitung fitur dan *z* satu
sampel secara manual dan membandingkannya dengan fungsi `fitur_lag` di
`muat_model.py`.
"""))

cells.append(code(r"""def cari_file(nama):
    roots = [os.environ.get("MODEL_DIR", ""), "/kaggle/input", "."]
    for r in [r for r in roots if r and os.path.isdir(r)]:
        for d, _, fs in sorted(os.walk(r, followlinks=True)):
            if nama in fs:
                return Path(d) / nama
    raise FileNotFoundError(f"{nama} tidak ditemukan. Tambahkan dataset "
                            "'kelvindsmn/data-hipotetik-fault-sensor-berlabel' lewat Add Data, "
                            "atau set MODEL_DIR ke folder berisi file model notebook 15.")

SUMBER = {"model_ann_lm_4sensor.npz": "15_model_gabungan_4sensor.npz",
          "model_ann_lm_rata2.npz": "15_model_gabungan_rata2.npz"}
for baru, lama in SUMBER.items():
    shutil.copyfile(cari_file(lama), PAKET / baru)
CARD15 = json.loads(cari_file("15_model_card.json").read_text())
print("model disalin dari:", cari_file(SUMBER["model_ann_lm_4sensor.npz"]).parent)
"""))

cells.append(code("# muat_model.py — teks sama persis dengan 15_muat_model.py notebook 15\n"
                  "MUAT_MODEL_PY = '''" + LOADER + "'''\n" + r"""(PAKET / "muat_model.py").write_text(MUAT_MODEL_PY)
sys.path.insert(0, str(PAKET.resolve()))
from muat_model import muat_model, fitur_lag, prob_fault, deteksi

m = muat_model(PAKET / "model_ann_lm_4sensor.npz")
# cek pemuat: contoh input + probabilitas acuan yang disimpan notebook 15 di file model
p_cek, _ = deteksi(m, m["contoh_input"])
selisih = float(np.max(np.abs(p_cek[10:] - m["contoh_prob"][10:])))
print(f"selisih prob pemuat vs acuan notebook 15 (2.000 sampel): {selisih:.1e}")
assert selisih < 1e-9

norm = pd.DataFrame({"fitur": m["feature_names"], "mean": m["mean"], "scale": m["scale"]})
display(norm.round(5))
"""))

cells.append(code(r"""# satu sampel dihitung manual: sensor 4, sampel pertama yang punya data hilang di 10 sampel sebelumnya
x = data["kelembaban4"].to_numpy(float)
na = np.isnan(x)
t = int(np.flatnonzero(pd.Series(na.astype(float)).rolling(10, min_periods=1).sum().shift(1).fillna(0).to_numpy() > 0)[0]) + 2
xf = pd.Series(x).ffill().bfill().to_numpy()                       # isi NaN dengan nilai valid sebelumnya
manual = [xf[t - k] for k in range(1, 11)] + [xf[t] - xf[t - 1], float(na[t]), float(na[t - 10:t].sum())]
otomatis = fitur_lag(x, 10)[t]
assert np.allclose(manual, otomatis)
z = (otomatis - m["mean"]) / m["scale"]
print(f"sampel t = {t}, waktu {data['timestamp'][t]}, 12 nilai mentah terakhir:")
print(np.round(x[t - 11:t + 1], 3))
display(pd.DataFrame({"fitur": m["feature_names"], "nilai_fitur": np.round(otomatis, 4),
                      "mean": np.round(m["mean"], 4), "scale": np.round(m["scale"], 4),
                      "z (masukan model)": np.round(z, 4)}))
"""))

# ---------------------------------------------------------------- §4
cells.append(md(r"""# §4 — Model: arsitektur dan forward pass

ANN dilatih dengan **Levenberg–Marquardt** asli (Jacobian analitik) di notebook 15,
arsitektur dipilih lewat grid search pada F1 validasi.

* Model `4sensor`: 13 masukan → 12 neuron tanh → 6 neuron tanh → 2 keluaran
  (normal, fault) → softmax. 260 parameter. **Ambang 0,38.**
* Model `rata2` (untuk sinyal rata-rata 4 sensor): 13 → 16 → 8 → 2, 378 parameter,
  ambang 0,60.

`theta` di file model adalah semua bobot berurutan: W1, b1, W2, b2, W3, b3.
`alarm = 1` bila `prob_fault > ambang`. Sel berikut menjalankan forward pass
sampel §3 lapis demi lapis.
"""))

cells.append(code(r"""shapes = [tuple(int(v) for v in s) for s in m["shapes"]]
a, i = z[None, :], 0
for li, (r, c) in enumerate(shapes):
    W = m["theta"][i:i + r * c].reshape(r, c); i += r * c
    b = m["theta"][i:i + c]; i += c
    a = a @ W + b
    if li < len(shapes) - 1:
        a = np.tanh(a)
    print(f"lapis {li + 1}: W {r}x{c} + b {c}" + (" -> tanh" if li < len(shapes) - 1 else " -> keluaran linear"),
          np.round(a[0], 3))
assert i == len(m["theta"])
P = np.exp(a - a.max()); P /= P.sum()
print(f"softmax [normal, fault] = {np.round(P[0], 4)}")
p_t = float(P[0, 1]); ambang = float(m["ambang"])
print(f"prob_fault = {p_t:.4f}  ambang = {ambang:.2f}  -> alarm = {int(p_t > ambang)}"
      f"   (label sebenarnya: {data['jenis_s4'][t]})")
assert abs(p_t - prob_fault(m, fitur_lag(x, 10))[t]) < 1e-12
print("parameter:", len(m["theta"]), " arsitektur:", str(m["arsitektur"]))
"""))

# ---------------------------------------------------------------- §5
cells.append(md(r"""# §5 — Prediksi pada 10.000 baris

Model `4sensor` dipanggil untuk tiap kolom sensor. 10 sampel pertama belum punya
lag lengkap, jadi tidak dinilai. Skor dihitung terhadap `label_s*`.

Ingat: model dilatih pada episode 3 hari (notebook 15), sedangkan di sini episode
0,5 hari, jadi skor ini **bukan** skor resmi model (skor resmi ada di
`model_card.json`, data uji notebook 15). Tujuannya menunjukkan bentuk keluaran
dan perilaku model per jenis fault.
"""))

cells.append(code(r"""PRED = pd.DataFrame({"timestamp": data["timestamp"]})
P_FULL = np.zeros((N_BARIS, 4))
skor = []
for s, n in enumerate(SENSORS):
    p, al = deteksi(m, data[n].to_numpy(float)); P_FULL[:, s] = p
    PRED[f"prob_s{s+1}"] = np.round(p, 6); PRED[f"alarm_s{s+1}"] = al
    PRED[f"label_s{s+1}"] = data[f"label_s{s+1}"]
    y = data[f"label_s{s+1}"].to_numpy()[10:]; a_ = al[10:]
    tp = int(((a_ == 1) & (y == 1)).sum()); fp = int(((a_ == 1) & (y == 0)).sum()); fn = int(((a_ == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    r = {"sensor": n, "kondisi": PENUGASAN[n][1], "precision": round(prec, 4), "recall": round(rec, 4),
         "F1": round(2 * prec * rec / max(prec + rec, 1e-12), 4),
         "alarm_palsu_% (baris normal)": round(100 * fp / max(int((y == 0).sum()), 1), 2)}
    for j, b in FAULT_BIT.items():                     # recall per jenis fault
        sel = (M[10:, s] & b) > 0
        r[f"recall_{j}"] = round(float(a_[sel].mean()), 3) if sel.any() else np.nan
    skor.append(r)
PRED.to_csv(PAKET / "prediksi_10000.csv", index=False)
SKOR = pd.DataFrame(skor); SKOR.to_csv(PAKET / "skor_10000.csv", index=False)
display(SKOR)
display(Markdown("Contoh keluaran di sekitar awal episode pertama (baris 995–1004):"))
display(PRED.iloc[995:1005])
"""))

cells.append(code(r"""fig, axs = plt.subplots(2, 1, figsize=(13, 5.5), sharex=True)
w = slice(900, 2600)
tt = np.arange(N_BARIS)[w]
axs[0].plot(tt, data["kelembaban4"][w], lw=0.7, color="C3", label="kelembaban4 (S5)")
axs[0].fill_between(tt, 0, 1, where=data["label_s4"][w] == 1, transform=axs[0].get_xaxis_transform(),
                    color="C3", alpha=0.1, label="label fault")
axs[0].legend(fontsize=8, loc="upper left"); axs[0].set_ylabel("%")
axs[1].plot(tt, PRED["prob_s4"][w], lw=0.7, label="prob_fault")
axs[1].axhline(float(m["ambang"]), color="k", ls="--", lw=0.8, label=f"ambang {float(m['ambang']):.2f}")
axs[1].legend(fontsize=8, loc="upper left"); axs[1].set_xlabel("indeks baris")
fig.tight_layout(); fig.savefig(PAKET / "gambar_prediksi_s4.png", dpi=110); plt.show()
"""))

# ---------------------------------------------------------------- §6
cells.append(md(r"""# §6 — Contoh backend: satu pembacaan masuk → JSON keluar

Di backend, data datang **satu per satu** (tiap 30 detik), bukan sekaligus.
`contoh_backend.py` menyimpan riwayat pendek per sensor (64 nilai terakhir), lalu
untuk setiap pembacaan baru mengembalikan:

```json
{"sensor": "kelembaban4", "nilai": 53.1, "prob_fault": 0.71, "alarm": true, "siap": true}
```

`siap = false` selama riwayat belum 11 nilai (lag belum lengkap); abaikan hasilnya.
Alarm pertama yang dicetak jatuh di menit-menit awal data, di luar episode, jadi
itu **alarm palsu**; tingkat alarm palsu per sensor ada di tabel skor §5. Backend
sebaiknya baru mengirim notifikasi bila alarm bertahan beberapa sampel berturut-turut.
Sel berikut (1) memutar seluruh 10.000 baris lewat `DetektorSensor` dan
memastikan hasilnya **sama** dengan prediksi batch §5, lalu (2) menjalankan
`contoh_backend.py` sebagai program terpisah dan menampilkan keluarannya.
"""))

cells.append(code("CONTOH_BACKEND_PY = r'''" + BACKEND + "'''\n" + r"""(PAKET / "contoh_backend.py").write_text(CONTOH_BACKEND_PY)
import importlib
sys.modules.pop("contoh_backend", None)
from contoh_backend import DetektorSensor

det = DetektorSensor(PAKET / "model_ann_lm_4sensor.npz")
t1 = time.time(); stream = np.zeros((N_BARIS, 4)); stream_alarm = np.zeros((N_BARIS, 4), int)
for i_, baris in enumerate(data[SENSORS].itertuples(index=False)):
    for s, n in enumerate(SENSORS):
        v = baris[s]
        h = det.proses(n, None if pd.isna(v) else v)
        stream[i_, s] = h["prob_fault"]; stream_alarm[i_, s] = int(h["alarm"])
beda = float(np.max(np.abs(stream[10:] - P_FULL[10:])))
alarm_beda = int((stream_alarm[10:] != PRED[[f"alarm_s{s+1}" for s in range(4)]].to_numpy()[10:]).sum())
print(f"40.000 panggilan proses() dalam {time.time() - t1:.1f} s; baris >= 10: selisih prob maks {beda:.1e} "
      f"(JSON dibulatkan 4 desimal), alarm berbeda {alarm_beda}")
assert beda <= 5e-5 + 1e-12 and alarm_beda == 0

hasil = subprocess.run([sys.executable, str(PAKET / "contoh_backend.py"), str(PAKET / "data_hipotetik_berlabel_10000.csv")],
                       capture_output=True, text=True, timeout=600)
print(hasil.stdout[-3000:]); assert hasil.returncode == 0, hasil.stderr[-2000:]
ringkas_alarm = json.loads(hasil.stdout.strip().splitlines()[-1])["jumlah_alarm"]
for s, n in enumerate(SENSORS):
    assert ringkas_alarm[n] == int(PRED[f"alarm_s{s+1}"][10:].sum()), n
print("jumlah alarm contoh_backend.py == prediksi batch untuk ke-4 sensor")
"""))

# ---------------------------------------------------------------- §7
cells.append(md(r"""# §7 — README, model card, zip

Model card notebook 15 disalin dengan dua perubahan: bagian `paket_s1`
ditambahkan, dan kalimat status publikasi diganti karena judul/DOI paper
"ACS" yang disebut di sana belum ditemukan (Crossref/OpenAlex) dan belum
dikonfirmasi pembimbing.
"""))

cells.append(code(r"""card = json.loads(json.dumps(CARD15))
card["status_publikasi"] = ("Metode (ANN dilatih Levenberg-Marquardt, fitur lag tanpa entropi, skenario S1-S5) "
                            "berasal dari program pertama (Des 2025). Judul/DOI publikasinya belum terverifikasi; "
                            "konfirmasi ke pembimbing sebelum disitasi. Kode program pertama memakai scipy "
                            "least_squares(method='lm') dengan Jacobian numerik dan 1 keluaran sigmoid; model ini "
                            "dilatih ulang dengan LM asli (Jacobian analitik).")
for v in card["model"].values():
    v["file_paket_s1"] = "model_ann_lm_" + ("4sensor" if "4sensor" in v["file"] else "rata2") + ".npz"
card["paket_s1"] = {
    "dibuat_oleh": "kaggle.com/code/kelvindsmn/paket-s1-data-hipotetik-10000",
    "data": "data_hipotetik_berlabel_10000.csv", "baris": N_BARIS,
    "episode_hari": DRIFT_EPISODE_DAYS, "catatan_episode": "notebook 15 memakai 3 hari; dipendekkan agar 3 episode muat di 10.000 baris",
    "penugasan_sensor": {n: f"{v[0]} {v[1]}" for n, v in PENUGASAN.items()},
    "preprocessing": {"fitur": list(map(str, m["feature_names"])), "lag_k": int(m["lag_k"]),
                      "nilai_hilang": "NaN diisi nilai valid sebelumnya (ffill) sebelum lag dihitung; isna dan isna_10 menandai data hilang",
                      "normalisasi": "z = (fitur - mean) / scale, mean dan scale dari file model (StandardScaler data latih notebook 15)",
                      "mean": [round(float(v), 6) for v in m["mean"]], "scale": [round(float(v), 6) for v in m["scale"]]},
    "skor_pada_data_10000": SKOR.to_dict(orient="records"),
}
(PAKET / "model_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False))

tabel_norm = "\n".join(f"| `{f}` | {mu:.4f} | {sc:.4f} |" for f, mu, sc in zip(m["feature_names"], m["mean"], m["scale"]))
_k = ["sensor", "kondisi", "precision", "recall", "F1", "alarm_palsu_% (baris normal)"]
tabel_skor = "\n".join(["| " + " | ".join(_k) + " |", "|" + "---|" * len(_k)]
                       + ["| " + " | ".join(str(r[c]) for c in _k) + " |" for r in skor])
README = f'''# Paket S1 — data hipotetik berlabel 10.000 baris + model ANN-LM program 1

Dibuat oleh notebook Kaggle `kelvindsmn/paket-s1-data-hipotetik-10000`.

## Isi

| File | Isi |
|---|---|
| `data_hipotetik_berlabel_10000.csv` | 10.000 baris, 4 sensor, nilai dengan fault + nilai asli + label per sensor |
| `kamus_kolom.csv` | arti kolom |
| `model_ann_lm_4sensor.npz` | model ANN-LM, satu sensor per panggilan (dipakai di contoh) |
| `model_ann_lm_rata2.npz` | model untuk rata-rata 4 sensor |
| `muat_model.py` | `muat_model(path)`, `deteksi(model, deret)`; cukup numpy + pandas |
| `contoh_backend.py` | kelas `DetektorSensor`: satu pembacaan masuk, JSON keluar |
| `prediksi_10000.csv`, `skor_10000.csv` | keluaran model pada data 10.000 baris dan skornya |
| `model_card.json` | ringkasan model, preprocessing, skor resmi, batasan |

## Data

Sumber: 10.000 baris pertama `tabel_sensor4_generated.csv` (interval 30 detik).
Fault disuntikkan dengan simulator notebook 15 (drift 5 poin per episode, bias +5,
spike x1,30 pada 1,52 % titik, hardware 8 % macet + 5 % hilang), 3 episode x 0,5 hari.
Tiap sensor diberi skenario berbeda: {", ".join(f"{n} = {v[0]} {v[1]}" for n, v in PENUGASAN.items())}.
Label berasal dari fault injection (bukan gangguan nyata di lapangan).

## Preprocessing (wajib sama di backend)

1. Ambil deret SATU sensor, urut waktu. Nilai hilang = NaN.
2. Isi NaN dengan nilai valid sebelumnya (forward fill).
3. Fitur sampel t: `lag1..lag10` = nilai t-1..t-10, `diff` = x(t) - x(t-1),
   `isna` = 1 bila x(t) hilang, `isna_10` = jumlah data hilang di t-10..t-1.
4. Normalisasi: `z = (fitur - mean) / scale` dengan mean/scale DARI FILE MODEL:

| fitur | mean | scale |
|---|---|---|
{tabel_norm}

5. Forward pass: 13 -> 12 tanh -> 6 tanh -> 2 -> softmax. `prob_fault` = keluaran kelas 1.
6. `alarm = prob_fault > {float(m["ambang"]):.2f}` (model 4sensor).

## Cara pakai

```python
from muat_model import muat_model, deteksi
m = muat_model("model_ann_lm_4sensor.npz")
prob, alarm = deteksi(m, deret_satu_sensor)        # array per sampel

from contoh_backend import DetektorSensor           # versi streaming untuk backend
det = DetektorSensor("model_ann_lm_4sensor.npz")
det.proses("kelembaban1", 51.27)
# {{"sensor": "kelembaban1", "nilai": 51.27, "prob_fault": ..., "alarm": false, "siap": true}}
```

`python contoh_backend.py data_hipotetik_berlabel_10000.csv` memutar seluruh data
dan mencetak 5 alarm pertama + jumlah alarm per sensor.

## Skor pada data 10.000 baris (baris 10 ke atas)

{tabel_skor}

Skor resmi model (data uji notebook 15): F1 {card["model"]["4sensor"]["hasil_uji"]["F1"]},
precision {card["model"]["4sensor"]["hasil_uji"]["precision"]}, recall {card["model"]["4sensor"]["hasil_uji"]["recall"]}.

## Batasan

- Drift kecil sulit dideteksi per sampel (recall drift rendah).
- Episode di data ini 0,5 hari, model dilatih pada episode 3 hari.
- Model tidak mengidentifikasi sensor; panggil per sensor.
- Data harus interval kira-kira 30 detik, nilai persen 0-100.
'''
(PAKET / "README_S1.md").write_text(README)

zpath = Path(str(PAKET) + ".zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(PAKET.iterdir()):
        if f.is_file():
            zf.write(f, f"paket_s1/{f.name}")
isi = pd.DataFrame([{"file": f.name, "kB": round(f.stat().st_size / 1024, 1)} for f in sorted(PAKET.iterdir()) if f.is_file()])
display(isi)
print(f"{zpath} {zpath.stat().st_size / 1e6:.2f} MB; total waktu {time.time() - T0:.0f} s")
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {OUT.name}: {len(cells)} cells")
