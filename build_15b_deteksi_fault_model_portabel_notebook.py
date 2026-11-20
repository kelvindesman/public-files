#!/usr/bin/env python3
"""Build 15b_Deteksi_Fault_Model_Portabel.ipynb.

Turunan (fork) notebook 15 yang hanya memakai model tersimpan: tidak ada
pelatihan. Orang lain cukup menunjuk CSV mereka sendiri, lalu notebook
menghitung probabilitas gangguan dan alarm per sampel dengan model ANN-LM
gabungan dari notebook 15 (arsitektur hasil grid search, tanpa identifikasi
sensor).

Yang dipakai ulang (dibaca dari teks build_15, bukan disalin manual):
  * pemuat model 15_muat_model.py -> MUAT_MODEL_PY di build_15

Keluaran: 15b_Deteksi_Fault_Model_Portabel.ipynb
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "15b_Deteksi_Fault_Model_Portabel.ipynb"
SRC15 = (HERE / "build_15_data_hipotetik_ann_lm_notebook.py").read_text()
LOADER = re.search(r"MUAT_MODEL_PY = '''(.*?)'''", SRC15, re.S).group(1)
assert "def deteksi(" in LOADER and "def muat_model(" in LOADER


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# 15b — Deteksi gangguan sensor dengan model ANN-LM portabel (tanpa pelatihan)

Notebook ini adalah turunan dari
[Data Hipotetik Berlabel ANN LM](https://www.kaggle.com/code/kelvindsmn/data-hipotetik-berlabel-ann-lm)
(notebook 15). Bedanya: **tidak ada pelatihan**. Model yang sudah dilatih di
notebook 15 dimuat dari file, lalu dipakai untuk mendeteksi gangguan pada
**data Anda sendiri**.

| | |
|---|---|
| **Yang dideteksi** | gangguan (fault) pada sensor kelembaban tanah, **per sampel**: drift, spike, bias, kerusakan hardware (macet/data hilang), dan kombinasinya |
| **Keluaran** | `prob_fault` (0–1) dan `alarm` (0/1) untuk setiap sampel dan setiap sensor, plus daftar episode alarm untuk notifikasi |
| **Model** | ANN dilatih dengan Levenberg–Marquardt asli (Jacobian analitik), fitur 10 lag + selisih + penanda data hilang, **tanpa entropi** |
| **Butuh** | hanya `numpy`, `pandas`, `matplotlib`; tidak butuh scikit-learn, tidak butuh GPU |

## Status publikasi

Metode ini (ANN dilatih Levenberg–Marquardt, fitur lag tanpa entropi,
skenario gangguan S1–S5) **sudah dipublikasikan pada paper pertama (ACS)**.
Kode yang dipakai saat paper itu **tertinggal**: masih memakai
`scipy.optimize.least_squares(method='lm')` dengan Jacobian numerik dan satu
keluaran sigmoid. Model di notebook ini dilatih ulang dengan kode yang
diperbarui (LM asli, Jacobian analitik yang dicek terhadap Jacobian numerik),
sehingga angka di sini **tidak sama persis** dengan angka di paper.

## Dari mana labelnya? — fault injection

Model dilatih pada **data berlabel hasil fault injection**: data 4 sensor
kelembaban tanah (`tabel_sensor4_generated.csv`, interval 30 detik) diberi
gangguan sintetis dengan simulator yang parameternya diketahui (bias 5 poin,
drift naik 5 poin per episode, spike +30 % pada 1,52 % titik, hardware 8 %
macet + 5 % hilang; 3 episode × 3 hari). Karena gangguannya disuntikkan
sendiri, setiap sampel tahu pasti fault atau normal. **Label ini bukan label
gangguan nyata di lapangan** — lihat bagian *Batasan*.

## Bagaimana model dipilih — grid search, tanpa identifikasi sensor

* Satu **model gabungan** dilatih dari semua 16 kondisi gangguan + data normal
  sekaligus, karena sistem notifikasi tidak tahu jenis gangguannya lebih dulu.
* Arsitektur dipilih dengan **grid search** (beberapa ukuran 1 dan 2 lapisan
  tersembunyi); yang dipakai adalah kandidat dengan **F1 validasi**
  tertinggi. Data uji tidak dipakai untuk memilih.
* **Tidak ada tahap identifikasi sensor.** Model `4sensor` menilai satu
  sensor sekaligus dari deret sensor itu sendiri. Sensor mana yang rusak
  langsung terbaca dari kolom alarm sensor tersebut.

Parameter lengkap (arsitektur terpilih, jumlah bobot, ambang, parameter LM,
hasil grid, hasil uji) dibaca dari `15_model_card.json` di bagian 2.

## Cara memakai dengan data Anda

1. Klik **Copy & Edit** (fork) notebook ini.
2. **Add Data** → unggah CSV Anda. Isinya: satu kolom waktu dan satu kolom
   per sensor (nilai kelembaban dalam persen 0–100). Data hilang boleh
   kosong/NaN.
3. Di sel **Konfigurasi**, isi `DATA_PATH`, `KOLOM_WAKTU`, dan (bila perlu)
   `KOLOM_SENSOR`. Biarkan `DATA_PATH = None` untuk menjalankan contoh
   dengan data hipotetik berlabel.
4. **Run All**. Hasil ada di `/kaggle/working/hasil_deteksi.csv` dan
   `hasil_episode_alarm.csv`.
"""))

cells.append(md(r"""# 1 — Konfigurasi"""))

cells.append(code(r"""import os
# ---- ubah bagian ini untuk data Anda ---------------------------------------
DATA_PATH = os.environ.get("DATA_PATH") or None   # mis. "/kaggle/input/data-saya/sensor.csv"; None = contoh
KOLOM_WAKTU = "timestamp"     # nama kolom waktu
KOLOM_SENSOR = None           # mis. ["sensor_a", "sensor_b"]; None = semua kolom angka selain waktu/label
KOLOM_LABEL = None            # opsional, untuk evaluasi: {"sensor_a": "label_a", ...}; None = tanpa evaluasi
MODE = "4sensor"              # "4sensor": model dinilai per sensor | "rata2": kolom sensor dirata-rata jadi 1 deret
INTERVAL_DETIK = 30           # data diseragamkan ke interval ini (model dilatih pada 30 detik); None = apa adanya
AMBANG = None                 # None = ambang dari model (disetel di validasi notebook 15)
ALARM_BERURUTAN = 3           # notifikasi dikirim bila alarm = 1 selama >= n sampel berturut-turut (1 = langsung)
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else "hasil_15b")
# ---------------------------------------------------------------------------
"""))

cells.append(md(r"""# 2 — Muat model dan kartu model

File model dicari di data yang terpasang (`/kaggle/input/...`, dataset
*Data Hipotetik Fault Sensor Kelembaban Berlabel*). Model disimpan sebagai
`.npz` (angka numpy murni), jadi tidak bergantung pada versi scikit-learn.
"""))

cells.append(code(r"""import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

def cari_file(nama):
    roots = [os.environ.get("MODEL_DIR", ""), "/kaggle/input", "hasil15/exports", "exports", "."]
    for r in [r for r in roots if r and os.path.isdir(r)]:
        hits = sorted(glob.glob(os.path.join(r, "**", nama), recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(f"{nama} tidak ditemukan. Tambahkan dataset "
                            "'kelvindsmn/data-hipotetik-fault-sensor-berlabel' lewat Add Data, "
                            "atau set MODEL_DIR ke folder berisi file model.")

MODEL_PATH = cari_file(f"15_model_gabungan_{MODE}.npz")
CARD = json.loads(cari_file("15_model_card.json").read_text())
print("model:", MODEL_PATH)
"""))

cells.append(code("# Pemuat model — sama persis dengan 15_muat_model.py dari notebook 15\n" + LOADER))

cells.append(code(r"""m = muat_model(MODEL_PATH)
info = CARD["model"][MODE]
assert str(m["arsitektur"]) == info["arsitektur"], "kartu model tidak cocok dengan file model"
assert abs(float(m["ambang"]) - info["ambang"]) < 1e-3

# Cek pemuat: contoh input yang disimpan bersama model harus memberi prob yang sama persis
p_cek, _ = deteksi(m, m["contoh_input"])
selisih = float(np.max(np.abs(p_cek[10:] - m["contoh_prob"][10:])))   # 10 sampel awal belum punya lag lengkap
print(f"selisih maks prob terhadap acuan notebook 15: {selisih:.2e}")
assert selisih < 1e-9, "pemuat memberi hasil berbeda dari model yang disimpan"

shapes = [tuple(int(v) for v in s) for s in m["shapes"]]
hu = info["hasil_uji"]
display(Markdown(f'''
| Parameter | Nilai |
|---|---|
| Versi model | `{MODE}` — {info["masukan"]} |
| Arsitektur (hasil grid search) | {info["arsitektur"]}: {" → ".join(str(r) for r, _ in shapes)} → {shapes[-1][1]} |
| Jumlah bobot | {info["n_parameter"]} |
| Aktivasi | {CARD["lm"]["aktivasi_tersembunyi"]} (tersembunyi), {CARD["lm"]["target"]} |
| Pelatihan | Levenberg–Marquardt, maks {CARD["lm"]["iterasi_maks"]} iterasi, λ awal {CARD["lm"]["lambda_awal"]}, {CARD["lm"]["redaman"]} |
| Fitur | {len(CARD["fitur"]["nama"])}: {", ".join(CARD["fitur"]["nama"])} |
| Interval sampel | {CARD["fitur"]["interval_sampel_detik"]} detik |
| Ambang alarm | {info["ambang"]} |
| Kriteria grid search | {CARD["grid_search"]["kriteria"]} |
| Identifikasi sensor | {CARD["grid_search"]["identifikasi_sensor"]} |
| Pembagian data latih | {CARD["pembagian_data"]} |
| Hasil uji (data hipotetik) | precision {hu["precision"]:.3f}, recall {hu["recall"]:.3f}, F1 {hu["F1"]:.3f}, ROC-AUC {hu["ROC_AUC"]:.3f}, alarm palsu di data normal {100*hu["FPR_data_normal"]:.1f}% |
| Dibuat | {CARD["dibuat"]} — {CARD["notebook_latih"]} |
'''))
print("Hasil grid search:"); display(pd.DataFrame(info["grid"]))
print("Fault injection saat pelatihan:"); display(pd.Series(CARD["fault_injection"]).to_frame("nilai"))
"""))

cells.append(md(r"""# 3 — Data

Tanpa `DATA_PATH`, notebook memakai contoh: skenario S5 (keempat gangguan
sekaligus) dari data hipotetik berlabel, supaya hasil deteksi bisa
dibandingkan dengan label.

Data diurutkan menurut waktu. Bila `INTERVAL_DETIK` diisi, data yang lebih
rapat dari itu dirangkum ke interval tersebut (median per slot). Slot tanpa
data **dibuang**, bukan diisi NaN, karena data latih juga punya celah waktu
(803 celah, ±289 jam) yang dibaca model sebagai baris berurutan. NaN pada
kolom sensor (sensor tidak mengirim nilai padahal sensor lain mengirim) tetap
dibaca sebagai data hilang.
"""))

cells.append(code(r"""if DATA_PATH is None:
    DATA_PATH = str(cari_file("15_data_hipotetik_4sensor_S5.csv.gz"))
    KOLOM_WAKTU = "timestamp"
    KOLOM_SENSOR = [f"kelembaban{i}" for i in range(1, 5)]
    KOLOM_LABEL = {f"kelembaban{i}": f"label_s{i}" for i in range(1, 5)}
    print("Memakai data contoh:", DATA_PATH)

raw = pd.read_csv(DATA_PATH)
raw[KOLOM_WAKTU] = pd.to_datetime(raw[KOLOM_WAKTU], errors="coerce")
raw = raw.dropna(subset=[KOLOM_WAKTU]).sort_values(KOLOM_WAKTU)
if KOLOM_SENSOR is None:
    skip = {KOLOM_WAKTU, *(KOLOM_LABEL or {}).values()}
    KOLOM_SENSOR = [c for c in raw.columns if c not in skip and pd.api.types.is_numeric_dtype(raw[c])
                    and not c.endswith("_asli") and not c.startswith("label")]
assert KOLOM_SENSOR, "tidak ada kolom sensor"
lab_cols = list((KOLOM_LABEL or {}).values())
raw = raw[[KOLOM_WAKTU, *KOLOM_SENSOR, *lab_cols]]

if INTERVAL_DETIK:
    g = raw.set_index(KOLOM_WAKTU)
    rule = f"{int(INTERVAL_DETIK)}s"
    data = g[KOLOM_SENSOR].resample(rule).median()
    if lab_cols:
        data = data.join(g[lab_cols].resample(rule).max())
    # slot tanpa data dibuang: model dilatih pada baris berurutan, celah waktu tidak diisi NaN
    data = data.dropna(subset=KOLOM_SENSOR, how="all").reset_index()
else:
    data = raw.reset_index(drop=True)

dt = data[KOLOM_WAKTU].diff().dt.total_seconds().median()
print(f"{len(data):,} sampel, {data[KOLOM_WAKTU].min()} .. {data[KOLOM_WAKTU].max()}, interval median {dt:.0f} s")
print("sensor:", KOLOM_SENSOR)
display(data[KOLOM_SENSOR].describe().round(2))

# peringatan bila data jauh dari kondisi latih
if abs(dt - CARD["fitur"]["interval_sampel_detik"]) > 5:
    print(f"PERINGATAN: interval {dt:.0f} s, model dilatih pada {CARD['fitur']['interval_sampel_detik']} s.")
v = data[KOLOM_SENSOR].to_numpy(float)
if np.nanmin(v) < 0 or np.nanmax(v) > 100:
    print("PERINGATAN: ada nilai di luar 0–100; model dilatih pada kelembaban dalam persen.")
lo, hi = float(m["mean"][0] - 4 * m["scale"][0]), float(m["mean"][0] + 4 * m["scale"][0])
luar = np.nanmean((v < lo) | (v > hi))
if luar > 0.05:
    print(f"PERINGATAN: {100*luar:.0f}% nilai di luar rentang data latih ({lo:.1f}–{hi:.1f}); "
          "pertimbangkan menyetel ulang AMBANG.")
"""))

cells.append(md(r"""# 4 — Deteksi (tanpa pelatihan)

Model hanya **dipakai**: fitur lag dihitung dari deret tiap sensor, lalu
bobot tersimpan memberi `prob_fault`. `alarm` = 1 bila `prob_fault` >
ambang. `notifikasi` = 1 bila alarm menyala selama ≥ `ALARM_BERURUTAN`
sampel berturut-turut, untuk mengurangi alarm palsu satu-titik.
"""))

cells.append(code(r"""def berurutan(a, n):
    # 1 pada sampel yang menjadi bagian dari deret alarm sepanjang >= n
    a = np.asarray(a, int)
    if n <= 1:
        return a
    out = np.zeros_like(a)
    run_id = np.cumsum(np.r_[1, np.diff(a) != 0])
    lens = pd.Series(a).groupby(run_id).transform("size").to_numpy()
    out[(a == 1) & (lens >= n)] = 1
    return out

hasil = data[[KOLOM_WAKTU]].copy()
deret = {"rata2": data[KOLOM_SENSOR].mean(axis=1).to_numpy()} if MODE == "rata2" else \
        {c: data[c].to_numpy(float) for c in KOLOM_SENSOR}
for nama, x in deret.items():
    p, a = deteksi(m, x, AMBANG)
    p[:int(m["lag_k"])] = np.nan; a[:int(m["lag_k"])] = 0          # belum punya lag lengkap
    hasil[nama] = x
    hasil[f"prob_fault_{nama}"] = np.round(p, 5)
    hasil[f"alarm_{nama}"] = a
    hasil[f"notifikasi_{nama}"] = berurutan(a, ALARM_BERURUTAN)

episodes = []
for nama in deret:
    a = hasil[f"notifikasi_{nama}"].to_numpy()
    run_id = np.cumsum(np.r_[1, np.diff(a) != 0])
    for _, grp in hasil[a == 1].groupby(run_id[a == 1]):
        episodes.append({"sensor": nama, "mulai": grp[KOLOM_WAKTU].iloc[0], "selesai": grp[KOLOM_WAKTU].iloc[-1],
                         "n_sampel": len(grp), "prob_maks": grp[f"prob_fault_{nama}"].max()})
episodes = pd.DataFrame(episodes, columns=["sensor", "mulai", "selesai", "n_sampel", "prob_maks"])

os.makedirs(OUT_DIR, exist_ok=True)
hasil.to_csv(os.path.join(OUT_DIR, "hasil_deteksi.csv"), index=False)
episodes.to_csv(os.path.join(OUT_DIR, "hasil_episode_alarm.csv"), index=False)
ringkas = pd.DataFrame({"sensor": list(deret),
                        "alarm_%": [round(100 * hasil[f"alarm_{n}"].mean(), 2) for n in deret],
                        "notifikasi_%": [round(100 * hasil[f"notifikasi_{n}"].mean(), 2) for n in deret],
                        "episode": [int((episodes["sensor"] == n).sum()) for n in deret]})
display(ringkas)
display(episodes.head(20))
print("ditulis:", os.path.join(OUT_DIR, "hasil_deteksi.csv"), "dan hasil_episode_alarm.csv")
"""))

cells.append(code(r"""fig, axes = plt.subplots(len(deret), 1, figsize=(14, 2.6 * len(deret)), sharex=True, squeeze=False)
for ax, nama in zip(axes[:, 0], deret):
    t = hasil[KOLOM_WAKTU]
    ax.plot(t, hasil[nama], lw=0.5, color="#4a6fa5", label=nama)
    on = hasil[f"notifikasi_{nama}"] == 1
    ax.scatter(t[on], hasil[nama][on], s=4, color="#d1495b", zorder=3, label="notifikasi")
    ax.set_ylabel("%"); ax.legend(loc="upper right", fontsize=8)
    ax2 = ax.twinx(); ax2.plot(t, hasil[f"prob_fault_{nama}"], lw=0.4, color="#888", alpha=0.6)
    ax2.set_ylim(0, 1); ax2.set_ylabel("prob_fault", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "hasil_deteksi.png"), dpi=90); plt.show()
"""))

cells.append(md(r"""# 5 — Evaluasi (hanya bila ada kolom label)

Bila data punya label fault/normal per sampel (`KOLOM_LABEL`), hasil alarm
dibandingkan dengan label. Pada data contoh, labelnya berasal dari fault
injection notebook 15.
"""))

cells.append(code(r"""if KOLOM_LABEL and MODE == "4sensor":
    rows = []
    for c, lc in KOLOM_LABEL.items():
        ok = hasil[f"prob_fault_{c}"].notna() & data[lc].notna()
        y = data.loc[ok, lc].astype(int).to_numpy()
        for kol in ["alarm", "notifikasi"]:
            yh = hasil.loc[ok, f"{kol}_{c}"].to_numpy()
            tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
            fn = int(((yh == 0) & (y == 1)).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan; rec = tp / (tp + fn) if tp + fn else np.nan
            rows.append({"sensor": c, "keluaran": kol, "n_fault": int(y.sum()), "precision": prec, "recall": rec,
                         "F1": 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else (0.0 if tp + fp + fn else np.nan),
                         "alarm_palsu_%": 100 * fp / max(1, int((y == 0).sum()))})
    evaluasi = pd.DataFrame(rows).round(3)
    evaluasi.to_csv(os.path.join(OUT_DIR, "hasil_evaluasi.csv"), index=False)
    display(evaluasi)
else:
    print("Tidak ada kolom label — evaluasi dilewati.")
"""))

cells.append(md(r"""# 6 — Batasan

* **Label pelatihan berasal dari fault injection**, bukan gangguan nyata.
  Bentuk gangguan di lapangan bisa berbeda; ambang (`AMBANG`) mungkin perlu
  disetel ulang dengan sedikit data berlabel dari lokasi Anda.
* **Model ikut memakai level nilai absolut.** Fitur lag adalah nilai mentah,
  dan gangguan bias menaikkan nilai 5 poin, sehingga model belajar bahwa
  level tinggi cenderung fault. Pada data normal notebook 15, sensor 1
  (level ±54 %) mendapat alarm palsu ±20 %, sensor 3 (±46 %) hanya 0,03 %.
  Pada data Anda dengan level kelembaban berbeda, alarm palsu bisa jauh lebih
  besar. Periksa `alarm_%` pada periode yang Anda tahu normal; bila tinggi,
  naikkan `AMBANG`, atau latih ulang dengan notebook 15 memakai data Anda.
* **Drift kecil sulit dideteksi per sampel** (recall drift rendah di notebook
  15). Model paling kuat untuk bias, spike, dan kerusakan hardware.
* Model dilatih pada satu set data (4 sensor, 30 detik, ±1 bulan). Interval
  atau rentang nilai yang jauh berbeda menurunkan keandalan; notebook memberi
  peringatan di bagian 3.
* Alarm palsu pada data normal: lihat baris *Hasil uji* di bagian 2. Aturan
  `ALARM_BERURUTAN` menguranginya dengan biaya notifikasi sedikit terlambat.
* Untuk melatih ulang dengan data Anda sendiri, pakai notebook 15 (data
  berlabel + program pelatihan lengkap).
"""))

for c in [c for c in cells if c["cell_type"] == "code"]:
    src = "".join(c["source"])
    assert "lm_fit" not in src and "least_squares" not in src, "notebook inferensi tidak boleh melatih"

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("wrote", OUT, "| cells:", len(cells))
