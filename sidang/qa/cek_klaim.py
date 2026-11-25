#!/usr/bin/env python3
"""cek_klaim.py — hitung ulang angka klaim sidang dari bahan mentah.

Skrip ini membaca `01_klaim_bukti.csv` (kolom `id` dan `nilai_klaim`), lalu
untuk setiap id yang punya fungsi hitung-ulang di `HITUNG` ia menghitung
ulang angkanya langsung dari CSV ekspor, log kernel Kaggle (JSON), builder,
atau `data_sensor.csv`. Hasilnya dicetak sebagai tabel:

    id | klaim | hitung ulang | selisih | status

Status:
  COCOK        selisih <= setengah satuan digit terakhir klaim
  PEMBULATAN   (hanya klaim berdesimal) selisih <= 1,5 satuan digit terakhir (salah arah pembulatan /
               pemotongan), atau batas "p <= x" yang hanya lolos setelah dibulatkan
  Mode khusus  maks (klaim "<= x"), min (klaim ">= x" / "ada"), sekitar (klaim
               "≈ x" / "sekitar x": toleransi mutlak 0,02)
  TIDAK COCOK  selisih lebih besar
  LEWATI       bukti tidak ada di repo (mis. hanya di output Kaggle). Pakai
               --kaggle DIR berisi hasil `kaggle kernels output` per slug.

Kode keluar 1 kalau ada satu saja TIDAK COCOK, selain itu 0.

Pemakaian:
    ./venv/bin/python sidang/qa/cek_klaim.py
    ./venv/bin/python sidang/qa/cek_klaim.py --kaggle /path/ke/unduhan_kaggle

Struktur folder --kaggle: <DIR>/<slug>/exports/*.csv dan <DIR>/<slug>/<slug>.log
(persis keluaran `kaggle kernels output kelvindsmn/<slug> -p <DIR>/<slug>`).
Tidak ada pelatihan model; hanya baca file + scipy.stats.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).with_name("01_klaim_bukti.csv")
KAGGLE: Path | None = None


class BuktiTidakAda(Exception):
    """Bukti hanya ada di Kaggle / tidak ada di repo."""


# ---------------------------------------------------------------- utilitas
def h(n: str, name: str) -> Path:
    p = ROOT / f"hasil{n}" / "exports" / name
    if not p.exists():
        raise BuktiTidakAda(str(p.relative_to(ROOT)))
    return p


def k(slug: str, name: str) -> Path:
    if KAGGLE is None:
        raise BuktiTidakAda(f"kaggle:{slug}/{name}")
    p = KAGGLE / slug / name
    if not p.exists():
        raise BuktiTidakAda(f"kaggle:{slug}/{name}")
    return p


def csv_(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


_LOG_CACHE: dict[Path, list] = {}


def log_entries(p: Path) -> list:
    if p not in _LOG_CACHE:
        if not p.exists():
            raise BuktiTidakAda(str(p))
        _LOG_CACHE[p] = json.loads(p.read_text())
    return _LOG_CACHE[p]


def log_text(p: Path) -> str:
    return "".join(e.get("data", "") for e in log_entries(p))


def log_cari(p: Path, pola: str, grup: int = 1) -> float:
    m = re.search(pola, log_text(p))
    if not m:
        raise BuktiTidakAda(f"pola tidak ditemukan di {p.name}: {pola}")
    return float(m.group(grup))


def log_waktu_akhir(p: Path) -> float:
    return max(float(e.get("time", 0)) for e in log_entries(p))


def hitung_traceback(p: Path) -> int:
    return log_text(p).count("Traceback")


def LOG(n: str) -> Path:
    logs = sorted((ROOT / f"hasil{n}").glob("*.log"))
    if not logs:
        raise BuktiTidakAda(f"hasil{n}/*.log")
    return logs[0]


def KLOG(slug: str) -> Path:
    return k(slug, f"{slug}.log")


def uji_berpasangan(a, b):
    """Sama dengan builder: stats.ttest_rel(b, a) dan stats.wilcoxon(b, a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t, p = stats.ttest_rel(b, a)
    try:
        w = stats.wilcoxon(b, a).pvalue
    except ValueError:
        w = float("nan")
    d = (b - a).mean() / (b - a).std(ddof=1)
    return float(p), float(w), float(d)


def f1_pasangan(path: Path) -> pd.DataFrame:
    d = csv_(path).dropna(subset=["Scenario"])
    return d


def rata(x) -> float:
    return float(np.mean(x))


# Data sensor mentah (dibaca sekali, ringan: 281 ribu baris)
_DATA = None


def data_sensor() -> pd.DataFrame:
    global _DATA
    if _DATA is None:
        p = ROOT / "data_sensor.csv"
        if not p.exists():
            raise BuktiTidakAda("data_sensor.csv")
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        _DATA = df.sort_index()
    return _DATA


SENS = [f"kelembaban{i}" for i in range(1, 5)]


def hitung_macet(s, minimal=10):
    """Salinan persis fungsi di build_ilustrasi_data_fault_notebook.py:101."""
    s = pd.Series(s).reset_index(drop=True)
    sama = s.eq(s.shift())
    grup = (sama != sama.shift()).cumsum()
    panjang = sama.groupby(grup).sum()
    return int((panjang >= minimal - 1).sum())


def runs_nol_s4():
    z = data_sensor()["kelembaban4"].eq(0)
    g = (z != z.shift()).cumsum()
    r = z.groupby(g).agg(["first", "size"])
    return r[r["first"]]["size"].sort_values(ascending=False).tolist()


def data_5menit() -> pd.DataFrame:
    return data_sensor()[SENS].resample("5min").mean()


def window_mayoritas_nol(N: int) -> tuple[int, int]:
    s4 = data_5menit()["kelembaban4"].values
    st = N // 2
    tot = maj = 0
    for i in range(0, len(s4) - N + 1, st):
        tot += 1
        maj += (s4[i:i + N] == 0).mean() > 0.5
    return maj, tot


def segmen_lurus_hari(min_jam=6.0):
    df = data_sensor()
    d2 = df[SENS].diff().diff().abs().max(axis=1)
    lin = d2 < 1e-6
    g = (lin != lin.shift()).cumsum()
    t0 = df.index[0]
    seg = []
    for _, grp in lin.groupby(g):
        if grp.iloc[0] and len(grp) >= int(min_jam * 120):
            a = (grp.index[0] - t0).total_seconds() / 86400
            b = (grp.index[-1] - t0).total_seconds() / 86400
            seg.append((a, b))
    return seg


def menang(path: Path, kolom="Difference", N=None) -> int:
    d = f1_pasangan(path)
    if N is not None:
        d = d[d["N_window"].astype(float) == float(N)]
    return int((d[kolom] > 0).sum())


def menang_dari_performa(path: Path, nilai="F1", metode="Method", skenario="Scenario") -> pd.Series:
    p = csv_(path)
    w = p.pivot_table(index=["N_window", skenario], columns=metode, values=nilai)
    return w["JSD-Fuzzy"] - w["EDM-Fuzzy"]


def sensor_rq6(n: str, kolom: str, metode: str) -> float:
    s = csv_(h(n, f"{n}_sensor_identifikasi.csv"))
    if kolom == "F1":
        x = s[(s.Sensor != "SEMUA-4-BENAR") & (s.Method == metode)]
        return float(x.groupby(["N_window", "Scenario"]).F1.mean().mean())
    x = s[(s.Sensor == "SEMUA-4-BENAR") & (s.Method == metode)]
    return float(x.Akurasi.mean())


def cocokkan_tabel(klaim: list[tuple], tabel: pd.DataFrame, kunci: list[str], kolom: str, digit=3) -> int:
    """Hitung berapa angka klaim (kunci..., nilai) yang sama dengan tabel setelah dibulatkan."""
    ok = 0
    tol = 0.5 * 10 ** (-digit) + 1e-9
    for row in klaim:
        *kv, v = row
        m = np.ones(len(tabel), bool)
        for kk, vv in zip(kunci, kv):
            m &= (tabel[kk].astype(str) == str(vv)).values
        if m.sum() == 1 and abs(float(tabel.loc[m, kolom].iloc[0]) - v) <= tol:
            ok += 1
    return ok


def menit(det: float) -> float:
    return det / 60.0


# ------------------------------------------------------------ fungsi klaim
HITUNG: dict[str, tuple] = {}


def reg(id_, mode="sama"):
    def deco(fn):
        HITUNG[id_] = (fn, mode)
        return fn
    return deco


def nilai(id_, fn, mode="sama"):
    HITUNG[id_] = (fn, mode)


K01 = "jsd-fuzzy-ann-paper"; K02 = "jsd-fuzzy-broker-architecture-v2"; K03 = "jsd-fuzzy-scenario-ladder"
K04 = "jsd-fuzzy-twostage-sensorid"; K05 = "jsd-fuzzy-faulttype-per-sensor"; K06 = "jsd-xgb-perscenario"
K07 = "jsd-fuzzy-cv-5skenario"; K08 = "jsd-fuzzy-vs-edm-fuzzy-paper-q3"; K09 = "ann-lm-vs-bayesian-vs-ga-paper-pmdc"
K10 = "jsd-vs-edm-dengan-ann-lm-asli-paper-pmdc"; KILU = "ilustrasi-data-akuisisi-dan-fault"

# ---- NB01 (Kaggle) -----------------------------------------------------
def _nb01(met, kol):
    d = csv_(k(K01, "exports/ringkasan_fault_detection.csv"))
    return float(d.loc[d.Metode == met, kol].iloc[0])
nilai("R01-01", lambda: _nb01("JSD-Fuzzy", "Test Accuracy"))
nilai("R01-02", lambda: _nb01("EDM-Fuzzy", "Test Accuracy"))
nilai("R01-03", lambda: _nb01("JSD-Fuzzy", "Macro-F1"))
nilai("R01-04", lambda: _nb01("EDM-Fuzzy", "Macro-F1"))
def _nb01s(sc, met):
    d = csv_(k(K01, "exports/rq4_classification_results.csv"))
    return float(d[(d.Scenario == sc) & (d.Method == met)].F1_macro.iloc[0])
nilai("R01-05", lambda: _nb01s("S1", "JSD-Fuzzy"))
nilai("R01-06", lambda: _nb01s("S1", "EDM-Fuzzy"))
nilai("R01-07", lambda: float(sum(_nb01s(s, "JSD-Fuzzy") > _nb01s(s, "EDM-Fuzzy") for s in ["S2", "S3", "S4", "S5"])))

# ---- NB02 (Kaggle) -----------------------------------------------------
def _nb02(var, kol):
    d = csv_(k(K02, "exports/broker_channel_summary.csv"))
    return float(d[(d.Method == "JSD-Fuzzy") & (d.Variant == var)][kol].iloc[0])
nilai("R02-01", lambda: _nb02("A_4channel", "F1"))
nilai("R02-02", lambda: _nb02("B_fused_1ch", "F1"))
nilai("R02-03", lambda: _nb02("A_4channel", "Accuracy"))
nilai("R02-04", lambda: _nb02("B_fused_1ch", "Accuracy"))

# ---- NB03 (Kaggle) -----------------------------------------------------
def _nb03(i, met):
    d = csv_(k(K03, "exports/ladder_results.csv"))
    sc = d.Scenario.unique()[i]
    return float(d[(d.Scenario == sc) & (d.Method == met)].F1_macro.iloc[0])
for i in range(5):
    nilai(f"R03-J{i+1}", (lambda i=i: _nb03(i, "JSD-Fuzzy")))
    nilai(f"R03-E{i+1}", (lambda i=i: _nb03(i, "EDM-Fuzzy")))
nilai("R03-W", lambda: float(sum(_nb03(i, "JSD-Fuzzy") > _nb03(i, "EDM-Fuzzy") for i in range(5))))

# ---- NB04 (Kaggle v6 vs lokal v5) ---------------------------------------
def _nb04_t3(met, kag=True):
    p = k(K04, "exports/tahap2_ringkas_per_tugas.csv") if kag else h("04", "tahap2_ringkas_per_tugas.csv")
    d = csv_(p)
    return float(d[(d.Tugas == "T3_17kelas") & (d.Metode == met)].F1.iloc[0])
def _nb04_all4(met, kag=True):
    p = k(K04, "exports/tahap2_per_jumlah_fault.csv") if kag else h("04", "tahap2_per_jumlah_fault.csv")
    d = csv_(p)
    return float(d[(d.Metode == met) & (d["Jumlah fault"] == 4)]["Semua-4-benar"].iloc[0])
nilai("R04-01", lambda: _nb04_t3("JSD-Fuzzy"))
nilai("R04-02", lambda: _nb04_t3("EDM-Fuzzy"))
nilai("R04-03", lambda: _nb04_all4("JSD-Fuzzy"))
nilai("R04-04", lambda: _nb04_all4("EDM-Fuzzy"))
nilai("H04-01", lambda: _nb04_t3("EDM-Fuzzy", kag=False))
nilai("H04-02", lambda: _nb04_t3("JSD-Fuzzy", kag=False))
nilai("H04-03", lambda: _nb04_all4("EDM-Fuzzy", kag=False))
nilai("H04-04", lambda: _nb04_all4("JSD-Fuzzy", kag=False))

# ---- NB05 (Kaggle) -----------------------------------------------------
def _nb05(met, item):
    d = csv_(k(K05, "exports/cara_b_stage2_summary.csv"))
    return float(d[(d.Method == met) & (d.Item == item)]["F1 rata-rata"].iloc[0])
for j, it in enumerate(["bias", "drift", "spike", "hardware"]):
    nilai(f"R05-J{j+1}", (lambda it=it: _nb05("JSD-Fuzzy", it)))
    nilai(f"R05-E{j+1}", (lambda it=it: _nb05("EDM-Fuzzy", it)))

# ---- NB06 (Kaggle) -----------------------------------------------------
def _nb06(sc, fs):
    d = csv_(k(K06, "exports/xgb_perscenario_results.csv"))
    return float(d[(d.Scenario == sc) & (d.FeatureSet == fs)].F1_macro.iloc[0])
nilai("R06-01", lambda: _nb06("S5", "JSD"))
nilai("R06-02", lambda: _nb01("JSD-Fuzzy", "Macro-F1"))
nilai("R06-03", lambda: _nb06("S5", "JSD+Time"))
def _nb06_best():
    d = csv_(k(K06, "exports/xgb_perscenario_results.csv"))
    best = d.loc[d.groupby("Scenario").F1_macro.idxmax()]
    return float((best.FeatureSet == "JSD+Time").sum())
nilai("R06-04", _nb06_best)
nilai("R06-05", lambda: _nb06("S5", "EDM"))
nilai("M-08", lambda: float(csv_(k(K06, "exports/xgb_perscenario_results.csv")).query("FeatureSet=='JSD+Time'").F1_macro.mean()))

# ---- NB07 (Kaggle v23) -------------------------------------------------
def _nb07_d():
    return menang_dari_performa(k(K07, "exports/07_performa_cv_5skenario.csv"), "F1", "Metode", "Skenario")
nilai("R07-01", lambda: float((_nb07_d() > 0).sum()))
nilai("R07-02", lambda: float(_nb07_d().mean()))
def _nb07_s(met, kol):
    d = csv_(k(K07, "exports/07_ringkasan_lengkap.csv"))
    return float(d[d.Metode == met][kol].mean())
nilai("R07-03", lambda: _nb07_s("JSD-Fuzzy", "F1_sensor"))
nilai("R07-04", lambda: _nb07_s("EDM-Fuzzy", "F1_sensor"))
nilai("R07-05", lambda: _nb07_s("JSD-Fuzzy", "Semua_4_benar"))
nilai("R07-06", lambda: _nb07_s("EDM-Fuzzy", "Semua_4_benar"))

# ---- NB08 (Kaggle v17) -------------------------------------------------
def _nb08():
    d = f1_pasangan(k(K08, "exports/08_rq5_f1_berpasangan.csv"))
    return d["EDM-Fuzzy F1"].values, d["JSD-Fuzzy F1"].values
nilai("R08-01", lambda: rata(_nb08()[1]))
nilai("R08-02", lambda: rata(_nb08()[0]))
nilai("R08-03", lambda: rata(_nb08()[1] - _nb08()[0]))
nilai("R08-04", lambda: float(((_nb08()[1] - _nb08()[0]) > 0).sum()))
nilai("R08-05", lambda: uji_berpasangan(*_nb08())[0])
nilai("R08-06", lambda: uji_berpasangan(*_nb08())[1])

# ---- NB09 (Kaggle v3) --------------------------------------------------
def _nb09_rank(tr, kol):
    d = csv_(k(K09, "exports/09_rq2_peringkat_pelatih.csv"))
    return float(d[d.Trainer == tr][kol].iloc[0])
nilai("R09-01", lambda: _nb09_rank("BR", "mean_F1"))
nilai("R09-02", lambda: _nb09_rank("LM", "mean_F1"))
nilai("R09-03", lambda: _nb09_rank("GA", "mean_F1"))
nilai("R09-04", lambda: _nb09_rank("Adam", "mean_F1"))
nilai("R09-05", lambda: _nb09_rank("BR", "juara_sel"))
nilai("R09-06", lambda: _nb09_rank("LM", "juara_sel"))
nilai("R09-07", lambda: _nb09_rank("Adam", "juara_sel"))
nilai("R09-08", lambda: _nb09_rank("GA", "juara_sel"))
def _nb09_p(a, b):
    d = csv_(k(K09, "exports/09_rq4_statistik.csv"))
    return float(d[(d.A == a) & (d.B == b)].p_value.iloc[0])
nilai("R09-09", lambda: _nb09_p("LM", "BR"))
nilai("R09-10", lambda: max(_nb09_p("LM", "GA"), _nb09_p("BR", "GA")), "maks")
nilai("R09-11", lambda: max(_nb09_p("LM", "Adam"), _nb09_p("BR", "Adam")), "maks")
nilai("R09-12", lambda: float(hitung_traceback(KLOG(K09))))

# ---- NB10 (Kaggle v4; lokal hasil10 = run 2026-10-10 sebelum koreksi drift)
def _nb10_pair(N=None):
    d = f1_pasangan(k(K10, "exports/10_rq5_f1_berpasangan.csv"))
    if N is not None:
        d = d[d.N_window.astype(float) == N]
    return d["EDM-Fuzzy F1"].values, d["JSD-Fuzzy F1"].values
nilai("R10-01", lambda: float(menang(k(K10, "exports/10_rq5_f1_berpasangan.csv"))))
nilai("R10-01L", lambda: float(menang(h("10", "10_rq5_f1_berpasangan.csv"))))
nilai("R10-02", lambda: rata(np.subtract(*_nb10_pair(200)[::-1])))
nilai("R10-03", lambda: rata(np.subtract(*_nb10_pair(700)[::-1])))
nilai("R10-04", lambda: rata(np.subtract(*_nb10_pair(1000)[::-1])))
nilai("R10-05", lambda: rata(np.subtract(*_nb10_pair()[::-1])))
nilai("R10-06", lambda: uji_berpasangan(*_nb10_pair(700))[0])
nilai("R10-07", lambda: uji_berpasangan(*_nb10_pair(200))[0])
nilai("R10-08", lambda: uji_berpasangan(*_nb10_pair(1000))[0])
nilai("R10-09", lambda: uji_berpasangan(*_nb10_pair())[0])
nilai("R10-10", lambda: uji_berpasangan(*_nb10_pair())[1])
def _ongkos_total(path):
    d = csv_(path)
    return d[d.bagian == "TOTAL"].iloc[0]
nilai("R10-11", lambda: float(_ongkos_total(k(K10, "exports/10_ongkos_per_bagian.csv")).cpu_all_s))
nilai("R10-12", lambda: float(_ongkos_total(k(K10, "exports/10_ongkos_per_bagian.csv")).cpu_all_s) / 3600)
nilai("R10-13", lambda: menit(log_cari(KLOG(K10), r"wall global=(\d+)s")))
nilai("R10-14", lambda: float(hitung_traceback(KLOG(K10))))

# ---- NB11 (lokal v3) ---------------------------------------------------
P11 = lambda: h("11", "11_rq5_f1_berpasangan.csv")
def _nb11_pair(N=None):
    d = f1_pasangan(P11())
    if N is not None:
        d = d[d.N_window.astype(float) == N]
    return d["EDM-Fuzzy F1"].values, d["JSD-Fuzzy F1"].values
_menang11 = lambda N=None: float(menang(P11(), N=N))
_beda11 = lambda N=None: rata(_nb11_pair(N)[1] - _nb11_pair(N)[0])
_p11 = lambda N=None: uji_berpasangan(*_nb11_pair(N))[0]
_w11 = lambda N=None: uji_berpasangan(*_nb11_pair(N))[1]
_cvr11 = lambda: csv_(h("11", "11_rq2_cv_reduction.csv"))
_cv_edm = lambda: float((_cvr11()["CV reduction (%)"] < 0).sum())
_cv_pct = lambda: 100 * (_cvr11()["CV reduction (%)"] < 0).mean()
_cv_mean = lambda N: float(_cvr11().query(f"N_window=={N}")["CV reduction (%)"].mean())
_wall11 = lambda: menit(log_cari(LOG("11"), r"wall global=(\d+)s"))
_cpu11 = lambda: float(_ongkos_total(h("11", "11_ongkos_per_bagian.csv")).cpu_all_s)
def _sens_menang(n="11"):
    s = csv_(h(n, f"{n}_sensor_ringkas.csv"))
    w = s.pivot_table(index=["N_window", "Scenario"], columns="Method", values="F1")
    return float((w["JSD-Fuzzy"] > w["EDM-Fuzzy"]).sum())
def _kode_terlarang_nb11():
    nb = json.loads((ROOT / "11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb").read_text())
    kode = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    kode = "\n".join(l.split("#")[0] for l in kode.splitlines())   # buang komentar
    return float(len(re.findall(r"MLPClassifier\(|solver\s*=\s*['\"](lbfgs|adam)|torch\.optim\.Adam|optimizers\.Adam", kode)))

for _id in ["R11-01", "B-35a", "B-36", "B-K1", "H11-07", "M-01"]:
    nilai(_id, lambda: _menang11())
for _id in ["R11-02", "B-35b", "B-K2", "H11-11", "M-02"]:
    nilai(_id, lambda: _beda11())
nilai("H11-26", lambda: _beda11())
for _id in ["R11-03", "B-39c", "H11-14", "P-17", "H14-26"]:
    nilai(_id, lambda: _p11(700))
for _id in ["R11-04", "B-39a", "B-K3", "H11-17", "R-61b", "R14-05", "H14-29"]:
    nilai(_id, lambda: _p11())
for _id in ["R11-05", "B-39b", "H11-18"]:
    nilai(_id, lambda: _w11())
for _id, met in [("R11-06", "JSD-Fuzzy"), ("B-35c", "JSD-Fuzzy"), ("H11-19", "JSD-Fuzzy"), ("M-03a", "JSD-Fuzzy"), ("H14-33", "JSD-Fuzzy"),
                 ("R11-07", "EDM-Fuzzy"), ("B-35d", "EDM-Fuzzy"), ("H11-20", "EDM-Fuzzy"), ("M-03b", "EDM-Fuzzy"), ("H14-31", "EDM-Fuzzy")]:
    nilai(_id, (lambda met=met: sensor_rq6("11", "F1", met)))
for _id in ["R11-08", "B-35e", "M-03c", "H11-24"]:
    nilai(_id, lambda: _sens_menang("11"))
for _id, met in [("R11-09", "JSD-Fuzzy"), ("H11-21", "JSD-Fuzzy"), ("M-04a", "JSD-Fuzzy"), ("H14-34", "JSD-Fuzzy"),
                 ("R11-10", "EDM-Fuzzy"), ("H11-22", "EDM-Fuzzy"), ("M-04b", "EDM-Fuzzy"), ("H14-32", "EDM-Fuzzy")]:
    nilai(_id, (lambda met=met: sensor_rq6("11", "SEMUA", met)))
for _id in ["R11-11", "B-K4", "H11-05"]:
    nilai(_id, _cv_edm)
nilai("R11-12", _cv_pct); nilai("H11-06", _cv_pct)
for _id in ["R11-13", "H11-01", "H14-40"]:
    nilai(_id, _wall11)
nilai("P-01", lambda: _wall11() / 60)
nilai("H11-01b", lambda: _wall11() / 60)
nilai("R11-14", _cpu11)
nilai("R11-15", lambda: _cpu11() / 3600)
nilai("R11-16", lambda: float(hitung_traceback(LOG("11"))))
nilai("H11-02", lambda: float(hitung_traceback(LOG("11"))))
nilai("R11-17", lambda: float(log_text(LOG("11")).count("SKIP")))
nilai("R11-18", _kode_terlarang_nb11)
nilai("H11-03", lambda: float(len(json.loads((ROOT / "11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb").read_text())["cells"])))
def _n_ekspor(ext):
    return float(len(list((ROOT / "hasil11" / "exports").glob(f"*.{ext}"))))
nilai("H11-04a", lambda: _n_ekspor("csv")); nilai("H11-04b", lambda: _n_ekspor("png")); nilai("H11-04c", lambda: _n_ekspor("txt"))
nilai("H11-08", lambda: _cv_mean(200)); nilai("H11-09", lambda: _cv_mean(700)); nilai("H11-10", lambda: _cv_mean(1000))
nilai("H11-12", lambda: _beda11(200)); nilai("H11-13", lambda: _beda11(700))
nilai("H11-15", lambda: _p11(200)); nilai("H11-16", lambda: _p11(1000)); nilai("H11-14w", lambda: _w11(700))
nilai("H11-23a", lambda: rata(_nb11_pair()[0])); nilai("H11-23b", lambda: rata(_nb11_pair()[1]))
for _id, N in [("B-36a", 200), ("B-36b", 700), ("B-36c", 1000)]:
    nilai(_id, (lambda N=N: _menang11(N)))
def _tabel_sensor11():
    s = csv_(h("11", "11_sensor_ringkas.csv"))
    s["Scenario"] = s["Scenario"].astype(str)
    klaim = []
    tab = {200: [(0.420, 0.644), (0.575, 0.543), (0.582, 0.621), (0.742, 0.684), (0.815, 0.788)],
           700: [(0.622, 0.668), (0.646, 0.664), (0.653, 0.622), (0.720, 0.780), (0.834, 0.896)],
           1000: [(0.410, 0.318), (0.544, 0.658), (0.642, 0.662), (0.763, 0.742), (0.540, 0.838)]}
    for N, row in tab.items():
        for i, (e, j) in enumerate(row):
            klaim += [(N, f"S{i+1}", "EDM-Fuzzy", e), (N, f"S{i+1}", "JSD-Fuzzy", j)]
    return float(cocokkan_tabel(klaim, s, ["N_window", "Scenario", "Method"], "F1"))
nilai("H11-25", _tabel_sensor11)
def _rq3_hitung(n, label):
    d = csv_(h(n, f"{n}_rq3_interpretasi.csv"))
    return float((d.Interpretation == label).sum())
nilai("H11-27", lambda: _rq3_hitung("11", "EDM-Fuzzy lebih terpisah"))
nilai("H11-28", lambda: _rq3_hitung("11", "JSD-Fuzzy lebih terpisah"))

# Bahan_Sidang Q37 (per skenario, rata-rata 3 window, JSD)
def _q37(sc):
    s = csv_(h("11", "11_sensor_ringkas.csv"))
    return float(s[(s.Scenario == sc) & (s.Method == "JSD-Fuzzy")].F1.mean())
for _id, sc in [("B-37a", "S5"), ("B-37b", "S4"), ("B-37c", "S3"), ("B-37d", "S2"), ("B-37e", "S1")]:
    nilai(_id, (lambda sc=sc: _q37(sc)))
def _kelas_skenario():
    d = csv_(h("11", "11_rq4_performa_ann.csv"))
    return d[(d.N_window == 200) & (d.Method == "EDM-Fuzzy")].sort_values("Scenario").C_kelas.tolist()
for _id, i in [("B-19a", 0), ("B-19b", 1), ("B-19c", 2), ("B-19d", 3), ("B-19e", 4)]:
    nilai(_id, (lambda i=i: float(_kelas_skenario()[i])))

# PANDUAN tabel F1 N=700
def _f1_11(N, sc, met):
    d = csv_(h("11", "11_rq4_performa_ann.csv"))
    return float(d[(d.N_window == N) & (d.Scenario == sc) & (d.Method == met)].F1.iloc[0])
for i in range(5):
    nilai(f"P-T{i+1}E", (lambda i=i: _f1_11(700, f"S{i+1}", "EDM-Fuzzy")))
    nilai(f"P-T{i+1}J", (lambda i=i: _f1_11(700, f"S{i+1}", "JSD-Fuzzy")))

# Fakta data dari log NB11 + data_sensor.csv
for _id in ["R11-D1", "B-06", "R-81a", "R-120"]:
    nilai(_id, lambda: log_cari(LOG("11"), r"Baris akuisisi: (\d+)"))
for _id in ["B-07a", "R-81b"]:
    nilai(_id, lambda: log_cari(LOG("11"), r"menjadi (\d+) baris"))
nilai("B-07b", lambda: log_cari(LOG("11"), r"\(([\d.]+) hari\)"))
nilai("B-08a", lambda: float(data_sensor()[SENS].isna().sum().sum()))
nilai("B-08b", lambda: float(((data_sensor()[SENS] < 0) | (data_sensor()[SENS] > 100)).sum().sum()))
nilai("B-08c", lambda: float(data_sensor().index.duplicated().sum()))
nilai("B-15", lambda: log_cari(LOG("11"), r"slot bolong ditambal: (\d+)"))
nilai("B-09a", lambda: float(data_sensor()["kelembaban4"].eq(0).sum()))
nilai("B-09b", lambda: 100 * float(data_sensor()["kelembaban4"].eq(0).mean()))
nilai("B-09c", lambda: float((data_sensor().index[data_sensor()["kelembaban4"].eq(0)].min()).day))
nilai("B-09d", lambda: runs_nol_s4()[0] * 30 / 3600)
nilai("B-09e", lambda: runs_nol_s4()[1] * 30 / 3600)
nilai("B-09f", lambda: runs_nol_s4()[2] * 30 / 3600)
nilai("B-09g", lambda: float(sum(runs_nol_s4()[3:])))
nilai("B-09h", lambda: float((data_5menit()["kelembaban4"] == 0).sum()))
nilai("B-09i", lambda: float(window_mayoritas_nol(200)[0]))
nilai("B-09j", lambda: float(window_mayoritas_nol(700)[0]))
nilai("B-09k", lambda: float(window_mayoritas_nol(1000)[0]))
nilai("B-22a", lambda: float(csv_(h("11", "11_akuntansi_window.csv")).query("N_window==200")["dibuang_fault<=1%"].iloc[0]))
nilai("B-22b", lambda: float(csv_(h("11", "11_akuntansi_window.csv")).query("N_window==200").window_dibentuk.iloc[0]))
nilai("R-84", lambda: log_cari(LOG("11"), r"N=  200 .*?durasi window=([\d.]+) hari"))
nilai("B-20", lambda: log_cari(LOG("11"), r"N=  200 .*?durasi window=([\d.]+) hari"))
nilai("R-85", lambda: float(csv_(h("11", "11_akuntansi_window.csv")).window_setengah_jadi.abs().sum()))
nilai("P-12", lambda: log_cari(LOG("11"), r"n_ref = (\d+)"))
nilai("B-27", lambda: log_cari(LOG("11"), r"bins = (\d+)"))
nilai("R-ILU1", lambda: float(data_sensor().index.duplicated().sum()))
nilai("R-ILU2", lambda: float(data_sensor()[SENS].isna().sum().sum()))
nilai("R-ILU3", lambda: float(((data_sensor()[SENS] < 0) | (data_sensor()[SENS] > 100)).sum().sum()))
for i, c in enumerate(SENS):
    nilai(f"R-ILU4{'abcd'[i]}", (lambda c=c: float(hitung_macet(data_sensor().loc[~data_sensor().index.duplicated(), c]))))
    nilai(f"R-ILU5{'abcd'[i]}", (lambda c=c: float(((data_sensor()[c] - data_sensor()[c].mean()).abs() / data_sensor()[c].std() > 6).sum())))

# Koreksi simulator drift (README)
_n5 = lambda: float(log_cari(LOG("11"), r"menjadi (\d+) baris"))
nilai("R-D1", lambda: 0.2 * 288)
nilai("R-D2", lambda: 0.2 * _n5())
nilai("R-D5", lambda: float(re.search(r"DRIFT_POINTS_PER_DAY = ([\d.]+)", (ROOT / "build_11_annlm_real_notebook.py").read_text()).group(1)))
_hari_total = lambda: (data_sensor().index[-1] - data_sensor().index[0]).total_seconds() / 86400
nilai("P-07a", lambda: 0.10 * _hari_total()); nilai("P-07b", lambda: 0.40 * _hari_total()); nilai("P-07c", lambda: 0.70 * _hari_total())
def _lurus_besar():
    seg = [(a, b) for a, b in segmen_lurus_hari() if b - a >= 1.0]
    return seg
nilai("P-08", lambda: float(sum(b - a for a, b in _lurus_besar())))
nilai("P-08b", lambda: float(sum(b - a for a, b in segmen_lurus_hari())))

# Builder: anggaran sesi
_bud = lambda f: float(re.search(r'KAGGLE_TIME_BUDGET_H", ([\d.]+)', (ROOT / f).read_text()).group(1))
nilai("R-159a", lambda: _bud("build_10_jsd_vs_edm_ann_lm_asli_notebook.py"))
nilai("R-159b", lambda: _bud("build_11_annlm_real_notebook.py"))
def _protokol_beda():
    """Jumlah konstanta protokol LM/CV yang BERBEDA antara builder 10 dan 11."""
    t10 = (ROOT / "build_10_jsd_vs_edm_ann_lm_asli_notebook.py").read_text()
    t11 = (ROOT / "build_11_annlm_real_notebook.py").read_text()
    beda = 0
    for nm in ["LM_ITERS_SEARCH", "LM_ITERS_REFIT", "LM_INNER_CV", "LM_SEARCH_BUDGET_S", "LM_REFIT_BUDGET_S",
               "N_SPLITS", "HIDDEN_GRID", "MAX_PER_CLASS", "RANDOM_SEED"]:
        a = re.search(rf"^{nm}\s*=\s*(.+?)(#|$)", t10, re.M)
        b = re.search(rf"^{nm}\s*=\s*(.+?)(#|$)", t11, re.M)
        if (a and a.group(1).strip()) != (b and b.group(1).strip()):
            beda += 1
    return float(beda)
nilai("R-64", _protokol_beda)   # klaim (2026-11-20): protokol identik = 0 konstanta beda
nilai("M-05", _protokol_beda)


# Demo laptop: bukti dari run nbclient (sidang/qa/jalankan_notebook.py)
_DEMO = ROOT / "sidang" / "bukti_demo"
def _demo_nb():
    import nbformat
    p = _DEMO / "03_Demo_Model_Tersimpan.executed.ipynb"
    if not p.exists():
        raise BuktiTidakAda(str(p.relative_to(ROOT)))
    return nbformat.read(p, as_version=4)
def _demo_teks(kunci: str) -> str:
    for c in _demo_nb().cells:
        if c.cell_type != "code":
            continue
        for o in c.get("outputs", []):
            t = o.get("text") or o.get("data", {}).get("text/plain", "")
            if kunci in t:
                return t
    raise BuktiTidakAda(f"output demo berisi '{kunci}'")
def _demo_total():
    p = _DEMO / "03_Demo_Model_Tersimpan.waktu.csv"
    if not p.exists():
        raise BuktiTidakAda(str(p.relative_to(ROOT)))
    return float(csv_(p).query("sel=='TOTAL'").detik.iloc[0])
def _demo_akurasi():
    t = _demo_teks("Window diuji")
    akur = [float(x) for x in re.findall(r"\s(\d+)%\s+\d+%\s+\d+%", t)]
    if len(set(akur)) != 1:
        raise BuktiTidakAda(f"akurasi Langkah 6 tidak sama untuk kedua metode: {akur}")
    return akur[0]
def _demo_benar():
    t = _demo_teks("Semua peluang")
    benar = 0
    for baris in t.splitlines()[1:]:
        m = re.match(r"\s*\d+\s+(asli|diberi fault)\s+(S\d)\s+\S+\s+(.+?)\s+\d+%", baris)
        if not m:
            continue
        jendela, model, tebak = m.group(1), m.group(2), m.group(3).strip()
        if jendela == "asli":
            benar += tebak == "fault free"
        else:
            benar += tebak != "fault free"
    return float(benar)
nilai("P-04", lambda: round(_demo_total()))
nilai("P-05", _demo_akurasi)
nilai("P-06", _demo_benar)
def _sel_beda_10_11():
    a = f1_pasangan(k(K10, "exports/10_rq5_f1_berpasangan.csv"))
    b = f1_pasangan(P11())
    m = a.merge(b, on=["N_window", "Scenario"], suffixes=("_10", "_11"))
    return float(((m["EDM-Fuzzy F1_10"] != m["EDM-Fuzzy F1_11"]) | (m["JSD-Fuzzy F1_10"] != m["JSD-Fuzzy F1_11"])).sum())
nilai("R-64b", _sel_beda_10_11)
nilai("B-29", lambda: float(len(eval(re.search(r"^HIDDEN_GRID = (\[.*\])", (ROOT / "build_11_annlm_real_notebook.py").read_text(), re.M).group(1)))))

# ---- NB12 (lokal) ------------------------------------------------------
def _nb12_rep(f, kol):
    d = csv_(h("12", "12_nilai_representatif.csv"))
    return float(d[d.fault == f][kol].iloc[0])
def _nb12_sweep(f, v, kol):
    d = csv_(h("12", "12_sweep_fault_param.csv"))
    return float(d[(d.fault == f) & (d.value.astype(str) == v)][kol].mean())
nilai("B-46a", lambda: _nb12_sweep("hardware", "(0.08, 0.05)", "AUC"))
nilai("B-46b", lambda: _nb12_rep("hardware", "TPR_pada_representatif")); nilai("M-07b", lambda: _nb12_rep("hardware", "TPR_pada_representatif"))
nilai("B-46c", lambda: _nb12_rep("hardware", "FPR_pada_representatif")); nilai("M-07c", lambda: _nb12_rep("hardware", "FPR_pada_representatif"))
nilai("B-46d", lambda: _nb12_rep("drift", "TPR_pada_representatif"))
nilai("B-46e", lambda: _nb12_rep("drift", "FPR_pada_representatif"))
nilai("B-46f", lambda: _nb12_rep("spike", "TPR_pada_representatif"))
nilai("B-46g", lambda: _nb12_rep("spike", "FPR_pada_representatif"))
def _bias_auc_dev12():
    d = csv_(h("12", "12_sweep_fault_param.csv"))
    a = d[d.fault == "bias"].groupby("value").AUC.mean()
    return float(a.sub(0.5).abs().max() + 0.5)
nilai("B-47", _bias_auc_dev12, "sekitar")
nilai("M-07a", _bias_auc_dev12, "sekitar")
nilai("B-48a", lambda: float(csv_(h("12", "12_rekomendasi.csv")).F1_rata_validasi.iloc[0]))
def _val12(uji):
    d = csv_(h("12", "12_validasi_N700.csv"))
    return float(d[(d.bins == 4) & (d.h_ratio == 1.0) & (d.uji == uji)].F1_macro.iloc[0])
nilai("B-48b", lambda: _val12("N700")); nilai("B-48c", lambda: _val12("S1_default"))
nilai("B-48d", lambda: float(csv_(h("12", "12_sweep_jsd_interaksi.csv")).query("bins==4 and h_ratio==0.5").F1_macro.iloc[0]))

# ---- NB13 (lokal) ------------------------------------------------------
_t13 = lambda: csv_(h("13", "13_tabel_tunggal.csv"))
_c13 = lambda: csv_(h("13", "13_tabel_kombinasi.csv"))
def _tabel_tunggal13():
    t = _t13()
    big = t.sort_values("value_order").groupby(["method", "fault"]).tail(1)
    klaim = {("JSD", "bias"): (0.396, 0.367, 0.501), ("EDM", "bias"): (0.479, 0.475, 0.511),
             ("JSD", "bias_episode"): (0.570, 0.462, 0.647), ("EDM", "bias_episode"): (0.634, 0.366, 0.743),
             ("JSD", "drift"): (0.667, 0.183, 0.847), ("EDM", "drift"): (0.688, 0.280, 0.798),
             ("JSD", "spike_sd"): (0.867, 0.142, 0.922), ("EDM", "spike_sd"): (0.775, 0.258, 0.815),
             ("JSD", "spike_rel"): (0.812, 0.179, 0.865), ("EDM", "spike_rel"): (0.708, 0.254, 0.785)}
    ok = 0
    for (m, f), vals in klaim.items():
        r = big[(big.method == m) & (big.fault == f)].iloc[0]
        ok += sum(abs(r[c] - v) <= 0.0005 + 1e-9 for c, v in zip(["TPR", "FPR", "AUC"], vals))
    for m, vals in {"JSD": (0.908, 0.083, 0.963), "EDM": (0.917, 0.100, 0.972)}.items():
        r = t[(t.method == m) & (t.fault == "hardware") & (t.value == "(0.3; 0.2)")].iloc[0]
        ok += sum(abs(r[c] - v) <= 0.0005 + 1e-9 for c, v in zip(["TPR", "FPR", "AUC"], vals))
    return float(ok)
nilai("H13-01", _tabel_tunggal13)
def _tabel_penelitian13():
    t = _t13()
    klaim = [("JSD", "bias", "0.08", 0.400, 0.367), ("EDM", "bias", "0.08", 0.479, 0.467),
             ("JSD", "drift", "3.0", 0.548, 0.473), ("EDM", "drift", "3.0", 0.559, 0.344),
             ("JSD", "spike_sd", "0.08", 0.658, 0.308), ("EDM", "spike_sd", "0.08", 0.704, 0.275),
             ("JSD", "hardware", "(0.08; 0.05)", 0.829, 0.175), ("EDM", "hardware", "(0.08; 0.05)", 0.779, 0.221)]
    ok = 0
    for m, f, v, tpr, fpr in klaim:
        r = t[(t.method == m) & (t.fault == f) & (t.value.astype(str) == v)].iloc[0]
        ok += int(abs(r.TPR - tpr) <= 0.0005 + 1e-9) + int(abs(r.FPR - fpr) <= 0.0005 + 1e-9)
    return float(ok)
nilai("H13-02", _tabel_penelitian13)
def _ambang13(m):
    t = _t13()
    x = t[(t.method == m) & (t.fault == "hardware") & (t.TPR >= 0.80) & (t.FPR <= 0.10 + 1e-12)].sort_values("value_order")
    return float(str(x.value.iloc[0]).strip("()").split(";")[0])   # proporsi stuck pertama yang lolos
nilai("R13-01", lambda: _ambang13("JSD")); nilai("R13-02", lambda: _ambang13("EDM"))
def _lolos_selain_hw():
    t = _t13()
    return float(((t.fault != "hardware") & (t.TPR >= 0.80) & (t.FPR <= 0.10 + 1e-12)).sum())
nilai("R13-03", _lolos_selain_hw)
nilai("R13-04", lambda: float(_t13().query("fault=='bias'").AUC.sub(0.5).abs().max() + 0.5), "sekitar")
nilai("H13-03", lambda: float(_t13().query("fault=='bias'").AUC.sub(0.5).abs().max() + 0.5), "sekitar")
nilai("R13-05", lambda: float(_t13().query("fault=='bias_episode'").AUC.max()), "maks")
nilai("R13-06", lambda: float(_t13().query("fault=='drift' and value=='60.0' and method=='EDM'").AUC.iloc[0]))
nilai("R13-07", lambda: float(_t13().query("fault=='drift' and value=='60.0' and method=='JSD'").AUC.iloc[0]))
nilai("R13-08", lambda: float(_t13().query("fault=='spike_sd' and method=='JSD' and value=='0.08'").AUC.iloc[0]))
nilai("R13-09", lambda: float(_t13().query("fault=='spike_sd' and method=='JSD' and value=='1.5'").AUC.iloc[0]))
nilai("R13-10", lambda: float(_t13().query("fault=='spike_sd' and method=='EDM'").AUC.min()))
nilai("R13-11", lambda: float(_t13().query("fault=='spike_sd' and method=='EDM'").AUC.max()))
nilai("R13-12", lambda: float(_c13().terdeteksi.astype(str).str.lower().eq("true").sum()))
def _baris_ok13():
    a = csv_(h("13", "13_sweep_tunggal.csv")); b = csv_(h("13", "13_sweep_kombinasi.csv"))
    return float((a.status == "ok").sum() + (b.status == "ok").sum())
nilai("R13-13", _baris_ok13); nilai("H13-04", _baris_ok13)
nilai("R13-14", lambda: menit(log_waktu_akhir(LOG("13")))); nilai("H13-05", lambda: menit(log_waktu_akhir(LOG("13"))))
def _komb13():
    c = _c13()
    g = c.groupby(["method", "tingkat"])[["F1", "TPR", "FPR"]].mean()
    klaim = {("JSD", "rendah"): (0.743, 0.779, 0.290), ("JSD", "sedang"): (0.766, 0.805, 0.271), ("JSD", "tinggi"): (0.779, 0.818, 0.258),
             ("EDM", "rendah"): (0.766, 0.793, 0.258), ("EDM", "sedang"): (0.767, 0.790, 0.254), ("EDM", "tinggi"): (0.771, 0.801, 0.257)}
    ok = 0
    for key, vals in klaim.items():
        ok += sum(abs(g.loc[key, c_] - v) <= 0.0005 + 1e-9 for c_, v in zip(["F1", "TPR", "FPR"], vals))
    return float(ok)
nilai("H13-06", _komb13)
_best13 = lambda: _c13().query("method=='JSD' and kombinasi=='spike+hardware' and tingkat=='tinggi'").iloc[0]
nilai("H13-07", lambda: float(_best13().TPR)); nilai("H13-08", lambda: float(_best13().FPR)); nilai("H13-09", lambda: float(_best13().AUC))
_hw13 = lambda: _c13()[_c13().kombinasi.str.contains("hardware")].AUC
nilai("H13-10", lambda: float(_hw13().min())); nilai("H13-11", lambda: float(_hw13().max()))
_db13 = lambda m: _c13().query(f"method=='{m}' and kombinasi=='drift+bias'").AUC
nilai("H13-12", lambda: float(_db13("JSD").min())); nilai("H13-13", lambda: float(_db13("JSD").max()))
nilai("H13-14", lambda: float(_db13("EDM").min())); nilai("H13-15", lambda: float(_db13("EDM").max()))
def _naik13(m):
    g = _c13().query(f"method=='{m}'").groupby("tingkat").F1.mean()
    return float(g["tinggi"] - g["rendah"])
nilai("H13-16", lambda: _naik13("JSD")); nilai("H13-17", lambda: _naik13("EDM"))
def _fpr_min13():
    t = _t13(); c = _c13()
    x = pd.concat([t[~((t.fault == "hardware") & (t.value_order >= 0.5))].FPR, c.FPR])
    return float(x.min())
nilai("H13-18", _fpr_min13, "sekitar")

# ---- NB14 (lokal) ------------------------------------------------------
P14 = lambda: h("14", "14_rq5_f1_berpasangan.csv")
def _nb14_pair(N=None):
    d = f1_pasangan(P14())
    if N is not None:
        d = d[d.N_window.astype(float) == N]
    return d["EDM-Fuzzy F1"].values, d["JSD-Fuzzy F1"].values
for _id in ["R14-01", "H14-04"]:
    nilai(_id, lambda: float(menang(P14())))
for _id in ["R14-02", "H14-27"]:
    nilai(_id, lambda: rata(_nb14_pair()[0]))
for _id in ["R14-03", "H14-28"]:
    nilai(_id, lambda: rata(_nb14_pair()[1]))
for _id in ["R14-04", "H14-14"]:
    nilai(_id, lambda: uji_berpasangan(*_nb14_pair())[0])
nilai("H14-15", lambda: uji_berpasangan(*_nb14_pair())[1])
nilai("H14-16", lambda: uji_berpasangan(*_nb14_pair())[2])
for _id, met in [("R14-06", "EDM-Fuzzy"), ("H14-35", "EDM-Fuzzy"), ("R14-07", "JSD-Fuzzy"), ("H14-36", "JSD-Fuzzy")]:
    nilai(_id, (lambda met=met: sensor_rq6("14", "F1", met)))
for _id, met in [("R14-08", "EDM-Fuzzy"), ("H14-37", "EDM-Fuzzy"), ("R14-09", "JSD-Fuzzy"), ("H14-38", "JSD-Fuzzy")]:
    nilai(_id, (lambda met=met: sensor_rq6("14", "SEMUA", met)))
def _s2s4_14(f):
    d = f1_pasangan(P14()); d = d[d.Scenario.isin(["S2", "S3", "S4"])]
    v = np.r_[d["EDM-Fuzzy F1"].values, d["JSD-Fuzzy F1"].values]
    return float(f(v))
for _id in ["R14-10", "H14-46"]:
    nilai(_id, lambda: _s2s4_14(np.min))
for _id in ["R14-11", "H14-47"]:
    nilai(_id, lambda: _s2s4_14(np.max))
for _id in ["R14-12", "H14-01", "H14-41"]:
    nilai(_id, lambda: menit(log_cari(LOG("14"), r"wall global=(\d+)s")))
nilai("R14-13", lambda: float(hitung_traceback(LOG("14")))); nilai("H14-02", lambda: float(hitung_traceback(LOG("14"))))
def _rq4_tabel14():
    a = csv_(h("11", "11_rq4_performa_ann.csv")); b = csv_(h("14", "14_rq4_performa_ann.csv"))
    rows = """200 S1 0.653 0.634 0.612 0.702|200 S2 0.300 0.315 0.280 0.370|200 S3 0.219 0.189 0.206 0.235|200 S4 0.367 0.397 0.266 0.408|200 S5 0.761 0.777 0.685 0.834|700 S1 0.595 0.671 0.658 0.773|700 S2 0.328 0.347 0.372 0.532|700 S3 0.184 0.196 0.237 0.284|700 S4 0.339 0.426 0.334 0.311|700 S5 0.804 0.872 0.653 0.633|1000 S1 0.490 0.668 0.542 0.669|1000 S2 0.381 0.365 0.292 0.384|1000 S3 0.220 0.164 0.249 0.211|1000 S4 0.382 0.400 0.314 0.350|1000 S5 0.849 0.872 0.614 0.764"""
    ok = 0
    for r in rows.split("|"):
        N, sc, e11, j11, e14, j14 = r.split()
        for tab, met, v in [(a, "EDM-Fuzzy", e11), (a, "JSD-Fuzzy", j11), (b, "EDM-Fuzzy", e14), (b, "JSD-Fuzzy", j14)]:
            x = tab[(tab.N_window == int(N)) & (tab.Scenario == sc) & (tab.Method == met)].F1.iloc[0]
            ok += abs(x - float(v)) <= 0.0005 + 1e-9
    return float(ok)
nilai("H14-03", _rq4_tabel14)
def _kalah14():
    d = f1_pasangan(P14())
    return float((d.Difference < 0).sum())
nilai("H14-05", _kalah14)
def _akun(n, N, kol):
    return float(csv_(h(n, f"{n}_akuntansi_window.csv")).query(f"N_window=={N}")[kol].iloc[0])
nilai("H14-06", lambda: _akun("14", 200, "dibuang_fault<=1%")); nilai("H14-07", lambda: _akun("14", 700, "dibuang_fault<=1%"))
nilai("H14-08", lambda: _akun("14", 1000, "dibuang_fault<=1%"))
nilai("H14-09", lambda: _akun("11", 200, "dibuang_fault<=1%")); nilai("H14-10", lambda: _akun("11", 700, "dibuang_fault<=1%"))
nilai("H14-11", lambda: _akun("11", 1000, "dibuang_fault<=1%"))
def _n_rq1(n, N):
    d = csv_(h(n, f"{n}_rq1_validasi_feature_matrix.csv"))
    return float(d[(d.N_window == N) & (d.Method == "JSD-Fuzzy")]["Number of samples"].iloc[0])
nilai("H14-12", lambda: _n_rq1("11", 700))     # klaim: "sama dengan NB11" = 1.621
nilai("H14-13", lambda: _n_rq1("11", 1000))    # klaim: "sama dengan NB11" = 1.526
nilai("H14-17", lambda: rata(_nb14_pair(200)[0])); nilai("H14-18", lambda: rata(_nb14_pair(200)[1]))
nilai("H14-19", lambda: uji_berpasangan(*_nb14_pair(200))[0])
nilai("H14-20", lambda: rata(_nb14_pair(700)[0])); nilai("H14-21", lambda: rata(_nb14_pair(700)[1]))
nilai("H14-22", lambda: uji_berpasangan(*_nb14_pair(700))[0])
nilai("H14-23", lambda: rata(_nb14_pair(1000)[0])); nilai("H14-24", lambda: rata(_nb14_pair(1000)[1]))
nilai("H14-25", lambda: uji_berpasangan(*_nb14_pair(1000))[0])
def _rq2_pct(n, N):
    c = csv_(h(n, f"{n}_rq2_cv_reduction.csv")).query(f"N_window=={N}")
    return float(100 * c["Scales JSD lebih stabil"].sum() / c["Scales dibandingkan"].sum())
for _id, n, N in [("H14-39a", "14", 200), ("H14-39b", "14", 700), ("H14-39c", "14", 1000),
                  ("H14-39d", "11", 200), ("H14-39e", "11", 700), ("H14-39f", "11", 1000),
                  ("R11-19a", "11", 200), ("R11-19b", "11", 700), ("R11-19c", "11", 1000)]:
    nilai(_id, (lambda n=n, N=N: _rq2_pct(n, N)))
nilai("H14-39g", lambda: _rq3_hitung("14", "JSD-Fuzzy lebih terpisah"))
nilai("H14-39h", lambda: _rq3_hitung("14", "EDM-Fuzzy lebih terpisah"))
def _ekstr(n, met, N):
    d = csv_(h(n, f"{n}_ongkos_ekstraksi_fitur.csv"))
    return float(d[(d.Metode == met) & (d.N_window == N)].wall_s.iloc[0])
for _id, n, met, N in [("H14-42a", "11", "JSD-Fuzzy", 200), ("H14-42b", "11", "JSD-Fuzzy", 700), ("H14-42c", "11", "JSD-Fuzzy", 1000),
                       ("H14-42d", "14", "JSD-Fuzzy", 200), ("H14-42e", "14", "JSD-Fuzzy", 700), ("H14-42f", "14", "JSD-Fuzzy", 1000),
                       ("H14-42g", "11", "EDM-Fuzzy", 200), ("H14-42h", "11", "EDM-Fuzzy", 700), ("H14-42i", "11", "EDM-Fuzzy", 1000),
                       ("H14-42j", "14", "EDM-Fuzzy", 200), ("H14-42k", "14", "EDM-Fuzzy", 700), ("H14-42l", "14", "EDM-Fuzzy", 1000)]:
    nilai(_id, (lambda n=n, met=met, N=N: _ekstr(n, met, N)))
def _biaya(n, met, kol):
    d = csv_(h(n, f"{n}_rq4_biaya_komputasi.csv"))
    return float(d[d.Method == met][kol].mean())
nilai("H14-43a", lambda: _biaya("11", "EDM-Fuzzy", "fit_s_per_fold")); nilai("H14-43b", lambda: _biaya("11", "JSD-Fuzzy", "fit_s_per_fold"))
nilai("H14-43c", lambda: _biaya("14", "EDM-Fuzzy", "fit_s_per_fold")); nilai("H14-43d", lambda: _biaya("14", "JSD-Fuzzy", "fit_s_per_fold"))
nilai("H14-44a", lambda: _biaya("11", "EDM-Fuzzy", "infer_ms_per_window")); nilai("H14-44b", lambda: _biaya("11", "JSD-Fuzzy", "infer_ms_per_window"))
nilai("H14-44c", lambda: _biaya("14", "EDM-Fuzzy", "infer_ms_per_window")); nilai("H14-44d", lambda: _biaya("14", "JSD-Fuzzy", "infer_ms_per_window"))
def _sensorid_s(n):
    d = csv_(h(n, f"{n}_ongkos_per_bagian.csv"))
    return float(d[d.bagian == "§12 sensor-ID LM asli"].wall_s.iloc[0])
nilai("H14-45a", lambda: _sensorid_s("11")); nilai("H14-45b", lambda: _sensorid_s("14"))
def _rasio(f):
    r = []
    for n in ["11", "14"]:
        for N in [200, 700, 1000]:
            r.append(_ekstr(n, "JSD-Fuzzy", N) / _ekstr(n, "EDM-Fuzzy", N))
    return float(f(r))
nilai("H14-45c", lambda: _rasio(min)); nilai("H14-45d", lambda: _rasio(max))
nilai("H14-30", lambda: rata(_nb14_pair()[1] - _nb14_pair()[0]))
def _s5_700(n, met):
    d = f1_pasangan(h(n, f"{n}_rq5_f1_berpasangan.csv"))
    r = d[(d.N_window.astype(float) == 700) & (d.Scenario == "S5")]
    return float(r[f"{met} F1"].iloc[0])
nilai("H14-48a", lambda: _s5_700("11", "EDM-Fuzzy")); nilai("H14-48b", lambda: _s5_700("11", "JSD-Fuzzy"))
nilai("H14-48c", lambda: _s5_700("14", "EDM-Fuzzy")); nilai("H14-48d", lambda: _s5_700("14", "JSD-Fuzzy"))

# ---- NB15 (log lokal hasil15 + CSV 15_* yang tersimpan di hasil16/exports)
def _nb15(versi, sc):
    p = ROOT / "hasil16" / "exports" / "15_hasil_ringkas_per_skenario.csv"
    if not p.exists():
        raise BuktiTidakAda(str(p))
    d = csv_(p)
    return float(d[(d.versi == versi) & (d.arsitektur == "2 layer (6,3)") & (d.skenario == sc)].F1.iloc[0])
for i in range(5):
    nilai(f"R15-A{i+1}", (lambda i=i: _nb15("rata2", f"S{i+1}")))
    nilai(f"R15-B{i+1}", (lambda i=i: _nb15("4sensor", f"S{i+1}")))
def _nb15_drift(versi):
    d = csv_(ROOT / "hasil16" / "exports" / "15_hasil_ann_lm.csv")
    return float(d[(d.versi == versi) & (d.kondisi == "drift") & (d.arsitektur == "2 layer (6,3)")].F1.iloc[0])
nilai("R15-C1", lambda: _nb15_drift("4sensor")); nilai("R15-C2", lambda: _nb15_drift("rata2"))
def _nb15_fpr():
    d = csv_(ROOT / "hasil16" / "exports" / "15_hasil_ann_lm_per_sensor.csv")
    return 100 * float(d[(d.terinjeksi == 0) & (d.arsitektur == "2 layer (6,3)")].FPR.mean())
nilai("R15-D", _nb15_fpr)
nilai("R15-E", lambda: menit(log_cari(LOG("15"), r"Wall total (\d+) s")))
nilai("R15-F", lambda: float(hitung_traceback(LOG("15"))))
nilai("R15-G", lambda: float(csv_(ROOT / "hasil16" / "exports" / "15_hasil_ann_lm.csv").query("versi=='rata2' and arsitektur=='1 layer (6)'").kondisi.nunique()))

# ---- NB16 / demo -------------------------------------------------------
nilai("P-02", lambda: log_cari(ROOT / "hasil16" / "simpan-model-final-deteksi-fault.log", r"total ([\d.]+) s") / 60)
nilai("M-06", lambda: log_cari(ROOT / "hasil16" / "simpan-model-final-deteksi-fault.log", r"total ([\d.]+) s") / 60)

# ---- tambahan: fungsi kecil dan alias --------------------------------
nilai("R-84b", lambda: log_cari(LOG("11"), r"N=  700 .*?durasi window=([\d.]+) hari"))
nilai("R-84c", lambda: log_cari(LOG("11"), r"N= 1000 .*?durasi window=([\d.]+) hari"))
def _konst(nama, f="build_11_annlm_real_notebook.py"):
    return float(re.search(rf"^{nama}\s*=\s*([\d.]+)", (ROOT / f).read_text(), re.M).group(1))
nilai("R-153a", lambda: _konst("LM_ITERS_SEARCH")); nilai("R-153b", lambda: _konst("LM_INNER_CV"))
nilai("R-153c", lambda: _konst("LM_ITERS_REFIT"))
nilai("R-D6", lambda: _konst("DRIFT_POINTS_PER_DAY") * _konst("DRIFT_EPISODE_DAYS"))
nilai("B-09m", lambda: float(window_mayoritas_nol(200)[1]))
nilai("B-09n", lambda: float(window_mayoritas_nol(700)[1]))
nilai("B-09o", lambda: float(window_mayoritas_nol(1000)[1]))
nilai("B-20b", lambda: 200 * 5 / 60)
nilai("B-28", lambda: float(csv_(h("11", "11_rq4_performa_ann.csv")).n_fitur.unique()[0]))
def _neg11(N):
    d = f1_pasangan(P11())
    return float((d[d.N_window.astype(float) == N].Difference < 0).sum())
nilai("B-41", lambda: _neg11(1000), "min")
nilai("H11-04d", lambda: float(len([x for x in (ROOT / "hasil11" / "exports").iterdir() if x.is_file()])))
nilai("H11-13b", lambda: _beda11(1000))
for _id, N in [("H11-29a", 200), ("H11-29b", 700), ("H11-29c", 1000)]:
    nilai(_id, (lambda N=N: float(_cvr11().query(f"N_window=={N}")["Scales JSD lebih stabil"].sum())))
nilai("H11-29d", lambda: float(_cvr11()["Scales dibandingkan"].groupby(_cvr11().N_window).sum().iloc[0]))
_auc13 = lambda m, f: _t13().query(f"method=='{m}' and fault=='{f}'").set_index("value").AUC
nilai("H13-19a", lambda: float(_auc13("JSD", "bias_episode").min())); nilai("H13-19b", lambda: float(_auc13("JSD", "bias_episode").max()))
nilai("H13-19c", lambda: float(_auc13("EDM", "bias_episode").min())); nilai("H13-19d", lambda: float(_auc13("EDM", "bias_episode").max()))
def _drift_kecil(f):
    t = _t13().query("fault=='drift' and value!='60.0'")
    return float(f(t.AUC))
nilai("H13-23a", lambda: _drift_kecil(np.min)); nilai("H13-23b", lambda: _drift_kecil(np.max))
nilai("H13-25a", lambda: float(_auc13("JSD", "spike_rel")["0.15"])); nilai("H13-25b", lambda: float(_auc13("JSD", "spike_rel")["0.5"]))
nilai("H13-26a", lambda: float(_auc13("EDM", "spike_rel").min())); nilai("H13-26b", lambda: float(_auc13("EDM", "spike_rel").max()))
_sr13 = lambda: _t13().query("method=='JSD' and fault=='spike_rel' and value=='0.5'").iloc[0]
nilai("H13-27a", lambda: float(_sr13().TPR)); nilai("H13-27b", lambda: float(_sr13().FPR))
for _id, N, k_ in [("H14-49a", 200, 0), ("H14-49b", 200, 1), ("H14-49c", 700, 0), ("H14-49d", 700, 1),
                   ("H14-49e", 1000, 0), ("H14-49f", 1000, 1)]:
    nilai(_id, (lambda N=N, k_=k_: rata(_nb11_pair(N)[k_])))
nilai("H14-39i", lambda: float(((csv_(h("14", "14_rq3_interpretasi.csv")).Interpretation == "campuran/sebanding")
                                & (csv_(h("14", "14_rq3_interpretasi.csv")).N_window == 200)).sum()))

ALIAS = {
    # README
    "R-60a": "R10-01", "R-60b": "R10-09", "R-61a": "R11-01", "R-121": "B-08a",
    # Bahan_Sidang
    "B-40": "R11-01", "B-36": "R11-01",
    # hasil11 ringkasan
    "H11-07a": "B-36a", "H11-07b": "B-36b", "H11-07c": "B-36c", "H11-02b": "R11-17",
    "H11-30": "R11-06", "H11-30b": "R11-07", "H11-31": "R11-03", "H11-32": "R11-04", "H11-33": "R11-12",
    # NB13 draf
    "H13-21": "R13-07", "H13-22": "R13-06", "H13-24a": "R13-08", "H13-24b": "R13-09",
    "H13-24c": "R13-10", "H13-24d": "R13-11", "H13-28": "R13-12",
    # NB14 draf
    "H14-30b": "R11-02", "H14-36b": "R11-06", "H14-38b": "R11-09", "H14-50": "H11-15", "H14-51": "H11-16",
    # MAPPING
    "M-01b": "R11-01", "M-10a": "R10-01", "M-09a": "R09-01", "M-09b": "R09-02", "M-09c": "R09-09",
    "M-11a": "R04-01", "M-11b": "R04-02", "M-12a": "R04-03", "M-12b": "R04-04",
    # PANDUAN
    "P-08c": "B-07b", "P-18": "R-84b", "P-19": "B-28",
}
for _a, _b in ALIAS.items():
    HITUNG[_a] = HITUNG[_b]


# ------------------------------------------------------------- perbandingan
def parse_klaim(s: str) -> tuple[float, int]:
    s = s.strip().replace("−", "-").replace("+", "")
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", s):  # 99.162 atau 85.784
        s = s.replace(".", "")
    s = s.replace(",", ".")
    d = len(s.split(".")[1]) if "." in s else 0
    return float(s), d


def bandingkan(klaim: str, v: float, mode: str) -> tuple[str, float]:
    c, d = parse_klaim(klaim)
    tol = 0.5 * 10 ** (-d)
    sel = v - c
    eps = 1e-9
    if mode == "maks":
        if v <= c + eps:
            return "COCOK", sel
        if v <= c + tol + eps:
            return "PEMBULATAN", sel
        return "TIDAK COCOK", sel
    if mode == "min":
        if v >= c - eps:
            return "COCOK", sel
        if v >= c - tol - eps:
            return "PEMBULATAN", sel
        return "TIDAK COCOK", sel
    if mode == "sekitar":   # klaim "≈ / sekitar": toleransi mutlak 0,02
        return ("COCOK" if abs(sel) <= 0.02 + eps else "TIDAK COCOK"), sel
    if abs(sel) <= tol + eps:
        return "COCOK", sel
    if d > 0 and abs(sel) <= 3 * tol + eps:   # bilangan bulat (hitungan) tidak diberi kelonggaran
        return "PEMBULATAN", sel
    return "TIDAK COCOK", sel


def fmt(v: float) -> str:
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.4f}".rstrip("0").rstrip(".") if abs(v) >= 0.001 else f"{v:.5f}"


def main(argv=None) -> int:
    global KAGGLE
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kaggle", default=os.environ.get("CEK_KAGGLE_DIR"),
                    help="folder unduhan `kaggle kernels output` per slug")
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--perbarui-ledger", action="store_true",
                    help="tulis nilai_bukti dan status hasil hitung ulang kembali ke ledger")
    a = ap.parse_args(argv)
    # Bukti Kaggle disalin ke repo (sidang/bukti_kaggle/) sehingga default-nya lokal.
    KAGGLE = Path(a.kaggle) if a.kaggle else ROOT / "sidang" / "bukti_kaggle"
    rows = list(csv.DictReader(open(a.ledger, encoding="utf-8")))
    hasil = []
    for r in rows:
        id_ = r["id"]
        if id_ not in HITUNG:
            continue
        fn, mode = HITUNG[id_]
        try:
            v = float(fn())
        except BuktiTidakAda as e:
            hasil.append((id_, r["nilai_klaim"], "-", "-", "LEWATI", f"bukti tidak di repo: {e}"))
            continue
        st, sel = bandingkan(r["nilai_klaim"], v, mode)
        hasil.append((id_, r["nilai_klaim"], fmt(v), fmt(sel) if sel else "0", st, "" if mode == "sama" else f"mode={mode}"))
    lebar = [max(len(str(x[i])) for x in hasil + [("id", "klaim", "hitung ulang", "selisih", "status", "catatan")]) for i in range(6)]
    kepala = ("id", "klaim", "hitung ulang", "selisih", "status", "catatan")
    print(" | ".join(str(c).ljust(w) for c, w in zip(kepala, lebar)))
    print("-+-".join("-" * w for w in lebar))
    for x in hasil:
        print(" | ".join(str(c).ljust(w) for c, w in zip(x, lebar)))
    from collections import Counter
    cnt = Counter(x[4] for x in hasil)
    tanpa = len(rows) - len(hasil)
    print()
    print(f"Baris ledger: {len(rows)} | dihitung ulang otomatis: {len(hasil)} | tanpa fungsi hitung (diperiksa manual): {tanpa}")
    print("Status hitung ulang: " + ", ".join(f"{k_}={v_}" for k_, v_ in sorted(cnt.items())))
    if a.perbarui_ledger:
        per_id = {x[0]: x for x in hasil}
        for r in rows:
            x = per_id.get(r["id"])
            if x and x[4] != "LEWATI":
                r["nilai_bukti"], r["status"] = x[2], x[4]
        with open(a.ledger, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("Ledger diperbarui dari hasil hitung ulang.")
    status_ledger = Counter(r["status"] for r in rows)
    print("Status ledger: " + ", ".join(f"{k_}={v_}" for k_, v_ in sorted(status_ledger.items())))
    gagal = [x for x in hasil if x[4] in ("TIDAK COCOK", "LEWATI")]
    gagal_ledger = [r["id"] for r in rows if r["status"] in ("TIDAK COCOK", "TIDAK ADA BUKTI LOKAL")]
    if gagal:
        print("TIDAK COCOK / LEWATI: " + ", ".join(x[0] for x in gagal))
    if gagal_ledger:
        print("Ledger belum bersih: " + ", ".join(gagal_ledger))
    lulus = not gagal and not gagal_ledger
    print("GERBANG KLAIM: " + ("LULUS" if lulus else "GAGAL"))
    return 0 if lulus else 1


if __name__ == "__main__":
    sys.exit(main())
