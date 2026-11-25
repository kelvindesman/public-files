#!/usr/bin/env python3
"""Build 17_Rekonstruksi_Paper_JITK.ipynb.

Notebook 17 rekonstruksi Tabel 1 dan Tabel 2 paper
Wardhani dkk. (2026), "Evaluation of ANN-Levenberg Marquardt Models for Fault
Detection in Smart Farming System", JITK 11(4):1053-1060,
doi:10.33480/jitk.v11i4.7481.

Isi:
  1. tabel paper (disalin dari PDF, hlm. 1056-1057);
  2. kode grid search Des 2025 (final_databaru_optimized.ipynb, commit 48966f8,
     sel 5, 8-16, 18, 21) dijalankan ulang pada 10.000 titik data;
  3. sidik jari yang menjelaskan pola tabel paper;
  4. kontrol negatif (skenario tanpa fault);
  5. sebelum -> sesudah sampai versi terbaru.

Jalankan: ./venv/bin/python build_17_rekonstruksi_paper_jitk_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "17_Rekonstruksi_Paper_JITK.ipynb"

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# 17 — Rekonstruksi Tabel 1 dan Tabel 2 paper JITK 2026

**Paper:** Wardhani, L. K., Buono, A., Wahjuni, S., & Syukur, M. (2026). *Evaluation of
ANN–Levenberg Marquardt Models for Fault Detection in Smart Farming System.* JITK 11(4),
1053–1060. doi:10.33480/jitk.v11i4.7481.

**Tujuan.** Menjelaskan dari mana angka Tabel 1 (akurasi, presisi, recall, F1) dan Tabel 2
(waktu eksekusi, CPU, memori) berasal. Caranya: kode grid search versi Desember 2025
(`final_databaru_optimized.ipynb`, commit `48966f8`, sel 21) dijalankan ulang di sini. Lalu
dicatat apa yang sudah diubah sampai versi terbaru, beserta alasan dan dampaknya.

**Batas.** Data dan kode persis yang dipakai paper tidak ada di repositori. Paper menyebut
10.000 titik data hipotetik dengan rentang 60–80 %. Rekonstruksi ini memakai 10.000 baris
pertama `tabel_sensor4_generated.csv`, data yang dibaca kode Des 2025. Rentangnya ±45–50 %,
jadi angka persis tidak bisa sama. Yang diuji adalah **pola** tabel. Setiap kesimpulan di
bawah diperiksa dengan sel kode yang bisa dijalankan ulang.

| Bagian | Isi |
|---|---|
| §1 | Tabel paper, disalin dari PDF |
| §2 | Kode Des 2025 dijalankan ulang → tabel versi rekonstruksi |
| §3 | Tujuh sidik jari: kenapa tabel paper berbentuk seperti itu |
| §4 | Kontrol negatif: data **tanpa fault** tetap mendapat akurasi ±0,99 |
| §5 | Sebelum → sesudah: perubahan sampai versi terbaru |
| §6 | Ringkasan untuk sidang |
""")

md(r"""
## §1 Tabel paper (disalin dari PDF, hlm. 1056–1057)

Notasi arsitektur paper, misalnya `[1 12 4]`, dibaca: 12 neuron di hidden layer 1 dan
4 neuron di hidden layer 2. Angka 0 atau tidak ada angka ketiga berarti satu hidden layer.
""")

code(r"""
import pandas as pd
pd.set_option("display.width", 200)

paper_t1 = pd.DataFrame([
    # skenario, model, h1, h2, akurasi, presisi, recall, F1
    (1, "A", 12, 0, 0.997, 0.995, 0.992, 0.994), (1, "B", 12, 4, 0.997, 0.995, 0.992, 0.994), (1, "C", 12, 8, 0.997, 0.995, 0.992, 0.994),
    (2, "A", 3, 8, 0.999, 1.000, 0.997, 0.999),  (2, "B", 3, 12, 0.999, 1.000, 0.997, 0.999), (2, "C", 3, 4, 0.998, 1.000, 0.992, 0.996),
    (3, "A", 5, 0, 0.997, 0.997, 0.987, 0.992),  (3, "B", 5, 4, 0.997, 0.997, 0.987, 0.992),  (3, "C", 5, 8, 0.997, 0.997, 0.987, 0.992),
    (4, "A", 5, 0, 0.997, 0.987, 0.997, 0.992),  (4, "B", 5, 4, 0.997, 0.987, 0.997, 0.992),  (4, "C", 5, 8, 0.997, 0.987, 0.997, 0.992),
    (5, "A", 3, 4, 0.997, 1.000, 1.000, 1.000),  (5, "B", 3, 8, 0.997, 1.000, 1.000, 1.000),  (5, "C", 3, 0, 0.996, 0.999, 1.000, 0.999),
], columns=["skenario", "model", "h1", "h2", "akurasi", "presisi", "recall", "F1"])

paper_t2 = pd.DataFrame([
    (1, 591.88, 78.40, 4.80), (2, 58.84, 86.00, 4.81), (3, 185.21, 81.60, 4.81),
    (4, 157.28, 82.50, 4.81), (5, 33.74, 50.50, 4.81),
], columns=["skenario", "waktu_s", "cpu_persen", "memori_persen"])

print("Tabel 1 paper"); print(paper_t1.to_string(index=False))
print("\nTabel 2 paper (sama untuk model A, B, C dalam satu skenario)"); print(paper_t2.to_string(index=False))
""")

md(r"""
Tiga hal langsung terlihat dari tabel paper:

1. Dalam satu skenario, ketiga "arsitektur" hampir selalu punya metrik **identik** sampai
   tiga desimal, padahal jumlah neuronnya berbeda.
2. Tabel 2 memberi waktu, CPU, dan memori yang **identik** untuk A, B, dan C dalam satu
   skenario.
3. Skenario 5 model A/B: presisi = recall = 1,000, tetapi akurasi 0,997. Pada satu ambang,
   presisi dan recall 1 berarti tidak ada FP dan FN, sehingga akurasi harus 1. Kombinasi itu
   hanya muncul kalau akurasi dihitung dengan ambang yang berbeda dari presisi/recall.

§2–§3 menunjukkan ketiga pola ini lahir dari kode grid search Des 2025.
""")

md(r"""
## §2 Kode Desember 2025 dijalankan ulang

Kode di bawah disalin dari `final_databaru_optimized.ipynb` commit `48966f8`:
- sel 5: simulator fault;
- sel 8–16: skenario S1–S5;
- sel 18: `neural_network_detection`;
- sel 21: grid search dua tahap.

Perbaikan minimum supaya bisa jalan, tanpa mengubah perilaku:

| Perbaikan | Alasan |
|---|---|
| Kolom `kelembaban_tanah_persen` = rata-rata `kelembaban1..4` | Sel 2 di commit menimpanya dengan `None` (indentasi rusak). Ambang `thr=50.7500` di output asli sama dengan kuantil 80 % rata-rata 4 sensor, jadi saat dijalankan kolom ini berisi rata-rata. |
| Tanda koma di signature `neural_network_detection` | Di commit, komentar `# Updated ...` menelan koma, sehingga sintaks rusak. |
| `make_features` memakai `base_signal` | Di commit memakai variabel `base_col` yang tidak terdefinisi. |
| Grid dijalankan serial, bukan paralel (`loky`) | Supaya urutan deterministik dan aman di memori. Nilai CPU % tetap memakai rumus asli. |
| Pilihan fault acak S1 diberi seed | Di kode asli tidak ada seed. |

Data: 10.000 baris pertama, sesuai jumlah titik di paper.
""")

code(r"""
import os, time, platform
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
from io import StringIO
from math import isnan
import numpy as np
import psutil
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score

N_TITIK = int(os.environ.get("N_TITIK", 10000))   # paper: "Ten thousand data points"
URL = "https://raw.githubusercontent.com/kelvindesman/public-files/refs/heads/main/tabel_sensor4_generated.csv"
lokal = Path("tabel_sensor4_generated.csv")
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
        cdn = "https://cdn.jsdelivr.net/gh/kelvindesman/public-files@main/" + jalur
        calon += [(cdn, False), (cdn + ".xz", True)]
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


if lokal.exists():
    df_raw = pd.read_csv(lokal)
else:
    import requests
    r = _ambil_cermin(URL, 180)
    df_raw = pd.read_csv(StringIO(r.text))

kelembaban_cols = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
df_raw["kelembaban_tanah_persen"] = df_raw[kelembaban_cols].mean(axis=1)
df = df_raw.iloc[:N_TITIK].reset_index(drop=True).copy()
base = "kelembaban_tanah_persen"
print(f"data: {len(df_raw)} baris, dipakai {len(df)} baris pertama")
print(f"rentang {base}: {df[base].min():.2f}–{df[base].max():.2f} (median {df[base].median():.2f})")
""")

code(r"""
# ---- sel 5 asli: simulator fault ----
def simulate_drift_fault(x, intensity=0.02, seed=None):
    rng = np.random.default_rng(seed)
    alpha = intensity
    drift = np.arange(len(x)) * alpha
    y = x + drift; m = np.abs(drift) > 1e-6; return y, m

def simulate_spike_fault(x, intensity=0.08, p=0.015, seed=None):
    rng = np.random.default_rng(seed)
    tau = max(1, int(1.0 / p)) if p > 0 else len(x)
    spikes = (np.arange(len(x)) % tau == 0).astype(float) * (intensity * np.nanstd(x))
    y = x + spikes; m = spikes != 0; return y, m

def simulate_bias_fault(x, bias=0.08, seed=None):
    y = x + bias; m = np.ones(len(x), bool); return y, m

def simulate_hardware_fault(x, stuck_prob=0.08, loss_prob=0.05, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=len(x))
    m1 = rng.random(len(x)) < stuck_prob
    y = x.copy(); y[m1] = x[idx][m1]
    m2 = rng.random(len(x)) < loss_prob
    y[m2] = np.nan
    return y, (m1 | m2)

def simulate_multiple_faults(x, faults, seed=None):
    y = x.copy(); m = np.zeros(len(x), bool)
    for f, kw in faults:
        y, mi = f(y, **kw, seed=seed); m |= mi
    return y, m
""")

code(r"""
# ---- sel 8-16 asli: skenario S1-S5 (tanpa plot) ----
def bangun_skenario(df, base, seed_pilihan=2025):
    x = df[base].to_numpy()
    s = {}
    # S1: satu fault acak
    d1 = df.copy()
    pilihan = [("drift", simulate_drift_fault, {"intensity": 0.15}),
               ("spike", simulate_spike_fault, {"intensity": 0.3, "p": 0.07}),
               ("bias", simulate_bias_fault, {"bias": 0.35}),
               ("hardware", simulate_hardware_fault, {"stuck_prob": 0.2, "loss_prob": 0.1})]
    nama, fn, kw = pilihan[np.random.default_rng(seed_pilihan).integers(len(pilihan))]
    y, m = fn(x, **kw)
    d1[f"{base}_{nama}"] = y; d1["label_fault"] = m.astype(int); d1.attrs["fault_S1"] = nama
    s[1] = d1
    # S2: empat fault tunggal, masing-masing kolom sendiri
    d2 = df.copy(); ms = []
    for nama, fn, kw in [("drift", simulate_drift_fault, {"intensity": 0.15}),
                         ("spike", simulate_spike_fault, {"intensity": 0.3, "p": 0.07}),
                         ("bias", simulate_bias_fault, {"bias": 0.35}),
                         ("hardware", simulate_hardware_fault, {"stuck_prob": 0.2, "loss_prob": 0.1})]:
        y, m = fn(x, **kw); d2[f"{base}_{nama}"] = y; ms.append(m)
    d2["label_fault"] = np.logical_or.reduce(ms).astype(int)
    s[2] = d2
    # S3: enam pasangan
    B = (simulate_bias_fault, {"bias": 0.5}); H = (simulate_hardware_fault, {"stuck_prob": 0.3, "loss_prob": 0.2})
    SP = (simulate_spike_fault, {"intensity": 0.3, "p": 0.07}); D = (simulate_drift_fault, {"intensity": 0.2})
    d3 = df.copy(); ms = []
    for nama, komb in [("bias_malfunc", [B, H]), ("spike_malfunc", [SP, H]), ("spike_bias", [SP, B]),
                       ("drift_malfunc", [D, H]), ("drift_bias", [D, B]), ("drift_spike", [D, SP])]:
        y, m = simulate_multiple_faults(x, komb); d3[f"{base}_{nama}"] = y; d3[f"label_{nama}"] = m.astype(int); ms.append(m)
    d3["label_fault"] = np.logical_or.reduce(ms).astype(int)
    s[3] = d3
    # S4: empat tripel
    d4 = df.copy(); ms = []
    for nama, komb in [("spike_bias_malfunc", [SP, B, H]), ("drift_bias_malfunc", [D, B, H]),
                       ("spike_drift_malfunc", [SP, D, H]), ("drift_spike_bias", [D, SP, B])]:
        y, m = simulate_multiple_faults(x, komb); d4[f"{base}_{nama}"] = y; d4[f"label_{nama}"] = m.astype(int); ms.append(m)
    d4["label_fault"] = np.logical_or.reduce(ms).astype(int)
    s[4] = d4
    # S5: keempat fault sekaligus
    d5 = df.copy()
    y, m = simulate_multiple_faults(x, [SP, D, B, H])
    d5[f"{base}_spike_drift_bias_malfunc"] = y; d5["label_fault"] = m.astype(int)
    s[5] = d5
    return s

SKENARIO = bangun_skenario(df, base)
print("fault acak S1:", SKENARIO[1].attrs["fault_S1"])
for k, d in SKENARIO.items():
    print(f"S{k}: kolom fault = {[c for c in d.columns if c.startswith(base + '_')]}, "
          f"porsi label_fault=1: {d['label_fault'].mean():.3f}")
""")

code(r"""
# ---- sel 18 asli: neural_network_detection (perbaikan sintaks minimum) ----
def sigmoid(z): return 1 / (1 + np.exp(-z))

def neural_net_predict(params, X, d, h1, h2, two_layers):
    i = 0
    W1 = params[i:i + d * h1].reshape(d, h1); i += d * h1
    b1 = params[i:i + h1].reshape(1, h1); i += h1
    if two_layers:
        W2 = params[i:i + h1 * h2].reshape(h1, h2); i += h1 * h2
        b2 = params[i:i + h2].reshape(1, h2); i += h2
        W3 = params[i:i + h2].reshape(h2, 1); i += h2
        a1 = np.tanh(X @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2)
    else:
        W3 = params[i:i + h1].reshape(h1, 1); i += h1
        a2 = np.tanh(X @ W1 + b1)
    b3 = np.array([[params[i]]])
    return sigmoid(a2 @ W3 + b3)

def residuals(params, X, y, d, h1, h2, two_layers):
    return (neural_net_predict(params, X, d, h1, h2, two_layers) - y).ravel()

def jumlah_parameter(d, h1, h2, use_two_layers):
    return d * h1 + h1 + (h1 * h2 + h2 if use_two_layers else 0) + (h2 if use_two_layers else h1) + 1

def make_features(df, base_signal, label_col="label_fault", lag_k=10, include_base=False):
    x = df.copy()
    for k in range(1, lag_k + 1): x[f"{base_signal}_lag{k}"] = x[base_signal].shift(k)
    x[f"{base_signal}_diff"] = x[base_signal].diff()
    feats = [c for c in x.select_dtypes(include=[np.number]).columns
             if c != label_col and (include_base or c != base_signal)]
    for c in list(feats):
        if x[c].isna().any():
            x[c + "_isna"] = x[c].isna().astype(int)
            x[c] = x[c].fillna(x[c].median())
            feats.append(c + "_isna")
    mask = (~x[feats].isna().any(axis=1)) & x[label_col].notna()
    return x.loc[mask, feats].to_numpy(), x.loc[mask, label_col].astype(int).to_numpy().reshape(-1, 1), feats, mask

def time_block_split(X, y, tr_frac=0.6, va_frac=0.2):
    n = len(X); i1 = int(n * tr_frac); i2 = int(n * (tr_frac + va_frac))
    return X[:i1], X[i1:i2], X[i2:], y[:i1], y[i1:i2], y[i2:]

def neural_network_detection(faulty_df, *, target_col="label_fault", base_signal=base, use_two_layers=True,
                             h1_size=6, h2_size=3, max_nfev=60, lag_k=10, include_base=False, hide_logging=True):
    df = faulty_df.copy()
    X_all, y_all, used, mask = make_features(df, base_signal, label_col=target_col, lag_k=lag_k, include_base=include_base)
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_block_split(X_all, y_all, 0.6, 0.2)
    sc = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
    d = X_tr.shape[1]
    np.random.seed(42)
    p0 = np.random.randn(jumlah_parameter(d, h1_size, h2_size, use_two_layers))
    res = least_squares(residuals, p0, args=(X_tr, y_tr, d, h1_size, h2_size, use_two_layers),
                        max_nfev=max_nfev, method="lm")
    w = res.x
    p_va = neural_net_predict(w, X_va, d, h1_size, h2_size, use_two_layers).ravel()
    p_te = neural_net_predict(w, X_te, d, h1_size, h2_size, use_two_layers).ravel()
    out = pd.DataFrame({"split": ["val"] * len(p_va) + ["test"] * len(p_te),
                        "true_label": np.concatenate([y_va.ravel(), y_te.ravel()]),
                        "predicted_prob": np.concatenate([p_va, p_te])})
    out.attrs.update(features=used, n_params=len(w), weights=w)
    return None, out
""")

code(r"""
# ---- sel 21 asli: grid search dua tahap (dijalankan serial) ----
POS_FRAC = 0.20
grid_h1 = [3, 5, 8, 12, 16]
grid_h2 = [0, 4, 8, 12]            # "keep 0 to represent 'no second layer' in records"
max_nfev = 40; n_repeats = 2; SEARCH_LAGS = 2; FINAL_LAGS = 5; early_margin = 0.05
MAX_NFEV_FIRST_REPEAT = max(20, max_nfev // 2); MAX_NFEV_SUBSEQUENT = max_nfev
_PROC = psutil.Process(os.getpid())
N_CPUS = psutil.cpu_count(logical=True) or 1
TOTAL_RAM_MB = psutil.virtual_memory().total / 1024 / 1024
def _rss_mb_once(): return _PROC.memory_info().rss / 1024 / 1024
def _cpu_times_s():
    t = _PROC.cpu_times(); return float(t.user + t.system)

def _prepare_df_for_grid(df, base_signal=base, pos_frac=POS_FRAC, n_lags=SEARCH_LAGS):
    label_col = "label"
    if (label_col not in df.columns) or (df[label_col].nunique() < 2):
        thr = df[base_signal].quantile(1 - pos_frac); df = df.copy()
        df["label_auto"] = (df[base_signal] >= thr).astype(int)
        label_col = "label_auto"
        print(f"[info] Using synthetic '{label_col}' | thr={thr:.4f} | pos_rate={df[label_col].mean():.3f}")
    dfe = df.copy(); s = dfe[base_signal]
    dfe[base_signal + "_diff"] = s.diff().fillna(0)
    for k in range(1, n_lags + 1): dfe[f"{base_signal}_lag{k}"] = s.shift(k)
    dfe = dfe.fillna(0)
    return dfe, label_col

def _summarize_best_f1_from_arrays(y_true, p):
    # ambang dipilih dari kurva presisi-recall pada data UJI
    precs, recs, thrs = precision_recall_curve(y_true, p)
    f1s = 2 * precs * recs / (precs + recs + 1e-12)
    i = int(np.nanargmax(f1s))
    best_t = float(thrs[i]) if i < len(thrs) else 0.5
    acc = float(((p > best_t) == y_true).mean())           # akurasi memakai  p >  t
    return acc, float(precs[i]), float(recs[i]), best_t, float(f1s[i])   # presisi/recall memakai p >= t

_global_best_f1 = {"v": 0.0}   # di kode asli: Manager().Value, tidak di-reset antar skenario

def _run_config_avg(h1, h2, dfe, label_col):
    accs, precs, recs, f1s, ts = [], [], [], [], []
    for r in range(n_repeats):
        nfev = MAX_NFEV_FIRST_REPEAT if r == 0 else MAX_NFEV_SUBSEQUENT
        _, res = neural_network_detection(dfe, use_two_layers=False, h1_size=h1, h2_size=h2,
                                          max_nfev=nfev, target_col=label_col)
        te = res[res["split"] == "test"]
        acc, pr, rc, t, f1 = _summarize_best_f1_from_arrays(te["true_label"].to_numpy(np.int8),
                                                            te["predicted_prob"].to_numpy(np.float32))
        accs.append(acc); precs.append(pr); recs.append(rc); f1s.append(f1); ts.append(t)
        if r == 0 and (f1 + early_margin * 1.5) < _global_best_f1["v"] and _global_best_f1["v"] > 0.1:
            for _ in range(n_repeats - len(accs)):
                accs.append(acc); precs.append(pr); recs.append(rc); f1s.append(f1); ts.append(t)
            break
    f1m = float(np.nanmean(f1s)); _global_best_f1["v"] = max(_global_best_f1["v"], f1m)
    return dict(accuracy=float(np.mean(accs)), precision=float(np.mean(precs)), recall=float(np.mean(recs)),
                f1_best=f1m, best_t=float(np.mean(ts)))

def run_grid(df, nama, g1, g2, n_lags):
    dfe, label_col = _prepare_df_for_grid(df, n_lags=n_lags)
    t0w, t0c = time.perf_counter(), _cpu_times_s()
    rows = [dict(scenario=nama, h1_size=h1, h2_size=h2, label_used=label_col, **_run_config_avg(h1, h2, dfe, label_col))
            for h1 in g1 for h2 in g2]
    wall = time.perf_counter() - t0w; cpu = _cpu_times_s() - t0c
    lb = pd.DataFrame(rows)
    # biaya diukur SEKALI untuk seluruh grid skenario ini, lalu disalin ke setiap baris
    lb["wall_time_s"] = round(wall, 3)
    lb["cpu_percent"] = round((cpu / wall) / N_CPUS * 100, 1)
    lb["mem_percent"] = round(_rss_mb_once() / TOTAL_RAM_MB * 100, 4)
    for c in ["accuracy", "precision", "recall", "f1_best"]: lb[c] = lb[c].round(4)
    return lb.sort_values(["f1_best", "accuracy", "recall", "precision"], ascending=False).reset_index(drop=True)

def run_grid_two_stage(df, nama, coarse_frac=0.25, top_k=3):
    global n_repeats
    kecil = df.iloc[:max(1000, int(len(df) * coarse_frac))].copy()
    n0 = n_repeats; n_repeats = 1
    lb1 = run_grid(kecil, nama + " [stage1]", grid_h1, grid_h2, SEARCH_LAGS)
    n_repeats = n0
    top = lb1.head(top_k)
    g1 = sorted(set(top["h1_size"])); g2 = sorted(set(top["h2_size"]))
    return run_grid(df, nama, g1, g2, FINAL_LAGS), lb1
""")

code(r"""
t_mulai = time.perf_counter()
LB = {}
for k, dsk in SKENARIO.items():
    print(f"\n=== Scenario {k} ===")
    lb, lb1 = run_grid_two_stage(dsk, f"Scenario {k}")
    LB[k] = lb
    print(lb.head(5).to_string(index=False))
print(f"\nTotal waktu §2: {time.perf_counter() - t_mulai:.1f} s  (N_CPUS={N_CPUS}, RAM={TOTAL_RAM_MB/1024:.1f} GB)")
""")

md(r"""
### Tabel versi rekonstruksi, disusun seperti Tabel 1 dan Tabel 2 paper

Seperti di kode asli: dari tahap 1, tiga konfigurasi terbaik (`top_k=3`) dibawa ke tahap 2,
lalu dilaporkan sebagai model A, B, dan C.
""")

code(r"""
rek = []
for k, lb in LB.items():
    for huruf, (_, r) in zip("ABC", lb.head(3).iterrows()):
        rek.append(dict(skenario=k, model=huruf, h1=int(r.h1_size), h2=int(r.h2_size),
                        akurasi=r.accuracy, presisi=r.precision, recall=r.recall, F1=r.f1_best,
                        waktu_s=r.wall_time_s, cpu_persen=r.cpu_percent, memori_persen=r.mem_percent))
rek = pd.DataFrame(rek)
print("Rekonstruksi — Tabel 1"); print(rek[["skenario", "model", "h1", "h2", "akurasi", "presisi", "recall", "F1"]].to_string(index=False))
print("\nRekonstruksi — Tabel 2"); print(rek[["skenario", "model", "h1", "h2", "waktu_s", "cpu_persen", "memori_persen"]].to_string(index=False))

def ringkas(t, kol):
    g = t.groupby("skenario")[kol]
    return pd.DataFrame({"min": g.min(), "max": g.max(), "identik_dalam_skenario": g.nunique() == 1})

banding = pd.concat({"paper": ringkas(paper_t1, "akurasi"), "rekonstruksi": ringkas(rek, "akurasi")}, axis=1)
print("\nAkurasi per skenario: paper vs rekonstruksi"); print(banding.to_string())
""")

md(r"""
## §3 Tujuh sidik jari

Setiap sidik jari adalah sifat kode Des 2025 yang menghasilkan pola tertentu di tabel paper.
Sel-sel berikut memeriksanya langsung.
""")

md(r"""
### S1 — Hidden layer 2 tidak pernah dipakai

`_run_config_avg` selalu memanggil `use_two_layers=False`. Nilai `h2_size` hanya dicatat
("pass h2_size only for bookkeeping"). Jadi `[1 12]`, `[1 12 4]`, dan `[1 12 8]` adalah
**jaringan yang sama**: jumlah parameter sama, bobot awal sama (`np.random.seed(42)`), dan
prediksi sama.
""")

code(r"""
dfe1, lab1 = _prepare_df_for_grid(SKENARIO[1], n_lags=FINAL_LAGS)
hasil = {}
for h2 in [0, 4, 8]:
    _, r = neural_network_detection(dfe1, use_two_layers=False, h1_size=12, h2_size=h2, max_nfev=20, target_col=lab1)
    hasil[h2] = r
    print(f"[1 12 {h2}]: jumlah parameter = {r.attrs['n_params']}")
sama = all(np.array_equal(hasil[0]["predicted_prob"], hasil[h]["predicted_prob"]) for h in [4, 8])
print("prediksi ketiga 'arsitektur' identik:", sama)
assert sama
""")

md(r"""
### S2 — Waktu, CPU, dan memori diukur per skenario, bukan per arsitektur

`run_grid` mengukur `wall_time_s`, `cpu_percent`, dan `mem_percent` **satu kali** untuk
seluruh grid satu skenario, lalu menyalin angka itu ke setiap baris. Karena itu Tabel 2
paper identik untuk A, B, dan C. Waktu itu juga bukan waktu melatih satu model, melainkan
waktu seluruh grid tahap 2 (jumlah konfigurasi × 2 ulangan). Jadi kalimat paper "larger
networks need longer execution time" tidak bisa disimpulkan dari angka ini.
""")

code(r"""
print(rek.groupby("skenario")[["waktu_s", "cpu_persen", "memori_persen"]].nunique().rename(columns=lambda c: c + "_jumlah_nilai_unik"))
assert (rek.groupby("skenario")[["waktu_s", "cpu_persen", "memori_persen"]].nunique() == 1).all().all()
""")

md(r"""
### S3 — CPU % bergantung pada jumlah core mesin; memori % adalah ukuran proses Python

- `cpu_percent = (waktu CPU proses / waktu dinding) / N_CPUS × 100`. Nilainya bergantung pada
  jumlah core mesin dan jumlah thread pustaka numerik (BLAS) yang aktif. Kode dan grid yang
  sama memberi ±16 % di laptop 6 core (1 thread) dan ±50–75 % di Kaggle 4 core
  (`hasil17/rekonstruksi-paper-jitk-ann-lm.v3.log`, berubah sedikit tiap run); paper 50,50–86,00 %. Angka ini mengukur
  mesin, bukan arsitektur.
- `mem_percent = RSS proses Python / total RAM × 100`. Isinya pandas, data, dan seluruh
  notebook, bukan memori model. Run Des 2025 di repositori memberi 4,7252–4,7304 %; paper
  4,80–4,81 % untuk semua skenario.
""")

code(r"""
print(f"mesin ini: {N_CPUS} core logis, RAM {TOTAL_RAM_MB/1024:.1f} GB")
print(f"CPU % kalau hanya satu thread yang bekerja = 100 / N_CPUS = {100 / N_CPUS:.1f} %; "
      f"lebih tinggi berarti pustaka numerik memakai beberapa thread")
print(rek.groupby("skenario")[["cpu_persen", "memori_persen"]].first())
""")

md(r"""
### S4 — Label yang dipakai grid search bukan label fault

Skenario tidak punya kolom `label`. Karena itu `_prepare_df_for_grid` membuat `label_auto`:
**1 kalau rata-rata kelembaban ada di 20 % tertinggi**, selain itu 0. Label ini dihitung
dari sinyal **bersih**, sebelum fault disuntikkan.

Fitur yang masuk model justru memuat `kelembaban1..4`, yaitu komponen rata-rata itu sendiri,
plus `lag` dan `diff` dari rata-rata. Akibatnya label bisa dihitung persis dari fitur dengan
satu aturan sederhana, tanpa jaringan saraf.
""")

code(r"""
d5, lab5 = _prepare_df_for_grid(SKENARIO[5], n_lags=FINAL_LAGS)
thr = SKENARIO[5][base].quantile(1 - POS_FRAC)
aturan = (d5[kelembaban_cols].mean(axis=1) >= thr).astype(int)
print("fitur kelembaban1..4 ada di input model:", all(c in make_features(d5, base, label_col=lab5)[2] for c in kelembaban_cols))
print(f"akurasi aturan 'rata-rata(kelembaban1..4) >= {thr:.2f}' terhadap label_auto: {(aturan == d5[lab5]).mean():.4f}")
benar = SKENARIO[5]["label_fault"]
print(f"kesesuaian label_auto dengan label fault sebenarnya (S5): {(d5[lab5] == benar).mean():.3f}  "
      f"(porsi fault sebenarnya {benar.mean():.3f}, porsi label_auto {d5[lab5].mean():.3f})")
""")

md(r"""
### S5 — Ambang keputusan dipilih di data uji

`_summarize_best_f1_from_arrays` memilih ambang yang memaksimalkan F1 dengan
`precision_recall_curve` pada **data uji**. Ambang dari data validasi
(`neural_network_detection` menghitungnya) tidak dipakai. Ini optimistis: label uji ikut
menentukan model.

### S6 — Akurasi dan presisi/recall memakai aturan ambang berbeda

Akurasi dihitung dengan `p > t`, presisi dan recall dengan `p >= t`. Itu sebabnya paper bisa
melaporkan presisi = recall = 1,000 dengan akurasi 0,997 (S5 model A/B). Sel di bawah
mencari baris seperti itu di rekonstruksi dan di paper, lalu menghitung ulang akurasi satu
model dengan kedua aturan.

Efeknya bisa besar. Ketika keluaran sigmoid jenuh, banyak sampel punya probabilitas persis
sama dengan ambang. Sampel-sampel itu dihitung positif untuk presisi/recall, tetapi negatif
untuk akurasi. Kontrol tanpa fault di §4 memperlihatkan F1 dan akurasi yang berbeda jauh pada
model yang sama.
""")

code(r"""
def tak_konsisten(t, desimal):
    # Presisi = recall = 1 (pada presisi yang dilaporkan) berarti FP = FN = 0, jadi akurasi harus 1.
    return t[(t["presisi"].round(desimal) == 1.0) & (t["recall"].round(desimal) == 1.0) & (t["akurasi"].round(desimal) < 1.0)]
print("paper (3 desimal):"); print(tak_konsisten(paper_t1, 3).to_string(index=False))
print("\nrekonstruksi (4 desimal):"); print(tak_konsisten(rek, 4).to_string(index=False))

# Demonstrasi langsung pada satu model: ambang dari data uji, lalu akurasi dengan p > t dan p >= t.
_, r = neural_network_detection(dfe1, use_two_layers=False, h1_size=3, h2_size=0, max_nfev=20, target_col=lab1)
te = r[r["split"] == "test"]; y = te["true_label"].to_numpy(np.int8); p = te["predicted_prob"].to_numpy(np.float32)
acc_kode, pr, rc, t, f1 = _summarize_best_f1_from_arrays(y, p)
acc_satu_ambang = float(((p >= t) == y).mean())
print(f"\nS1 [1 3]: ambang dari data uji t = {t:.6f}; sampel dengan p tepat = t: {(p == np.float32(t)).sum()}")
print(f"  presisi {pr:.4f}, recall {rc:.4f}, F1 {f1:.4f}  (p >= t)")
print(f"  akurasi versi kode (p >  t): {acc_kode:.4f}")
print(f"  akurasi satu ambang (p >= t): {acc_satu_ambang:.4f}")
""")

md(r"""
### S7 — Label fault asli tidak bisa dipakai karena mask bias mencakup seluruh deret

`simulate_bias_fault` mengembalikan mask True untuk **semua** sampel. Setiap skenario yang
memuat bias (S2–S5, dan S1 kalau fault acaknya bias) punya `label_fault` = 1 di semua baris.
Satu kelas tidak bisa dilatih atau dievaluasi, jadi kode jatuh ke `label_auto`.
""")

code(r"""
for k, d in SKENARIO.items():
    print(f"S{k}: porsi label_fault=1 = {d['label_fault'].mean():.3f}  -> kelas unik: {sorted(d['label_fault'].unique())}")
""")

md(r"""
## §4 Kontrol negatif: data tanpa fault sama sekali

Kalau tabel paper mengukur deteksi fault, data **tanpa fault** seharusnya tidak bisa
mencapai akurasi tinggi. Sel ini menjalankan grid yang sama pada sinyal bersih: tanpa kolom
fault dan tanpa `label_fault`.
""")

code(r"""
bersih = df.copy()
_global_best_f1["v"] = 0.0
lb_bersih, _ = run_grid_two_stage(bersih, "Tanpa fault")
print(lb_bersih.head(3)[["scenario", "h1_size", "h2_size", "label_used", "accuracy", "precision", "recall", "f1_best"]].to_string(index=False))
print(f"\nF1 terbaik tanpa fault: {lb_bersih['f1_best'].iloc[0]:.4f} "
      f"vs rentang F1 S1-S5 rekonstruksi: {rek['F1'].min():.4f}-{rek['F1'].max():.4f}")
print(f"akurasi tanpa fault: {lb_bersih['accuracy'].iloc[0]:.4f} (lihat S6: akurasi kode memakai p > t)")
""")

md(r"""
F1 pada data tanpa fault tetap tinggi, jauh di atas F1 deteksi fault versi terbaru
(±0,45–0,75). Jadi angka 0,99x di Tabel 1
mengukur kemampuan jaringan menebak "apakah rata-rata kelembaban berada di 20 % teratas",
bukan kemampuan mendeteksi fault.

## §5 Sebelum → sesudah: perubahan sampai versi terbaru

Setiap baris menjawab satu sidik jari di §3. Angka "sesudah" diambil dari commit/file yang
disebut. Rincian per fase ada di `sidang/qa/12_jurnal_eksperimen.md`.
""")

code(r"""
evolusi = pd.DataFrame([
    ("Label (S4, S7)", "label_auto = 20 % level tertinggi dari sinyal bersih", "mask fault yang benar-benar disuntikkan; fault berupa episode sehingga ada kelas normal dan fault",
     "label harus mengukur fault", "NB15 v3 (program pertama + LM asli + label fault): F1 1 sinyal 0,753, 4 sensor 0,640", "build_15_data_hipotetik_ann_lm_notebook.py; hasil15/data-hipotetik-berlabel-ann-lm.log"),
    ("Arsitektur (S1)", "h2 dicatat, tidak dipakai", "grid 8 arsitektur (1 dan 2 hidden layer) yang benar-benar dilatih, dipilih di dalam fold",
     "perbandingan arsitektur harus nyata", "arsitektur terpilih berbeda per fold; tercatat di models/meta.json", "deteksi_fault_core.py HIDDEN_GRID"),
    ("Biaya (S2, S3)", "1 angka per skenario untuk seluruh grid; CPU % bergantung core", "waktu latih per model dan per fold; tidak ada klaim efisiensi antar arsitektur",
     "biaya harus melekat ke model yang diukur", "NB11 total 610 menit; inferensi model tersimpan < 1 s", "hasil11/Ringkasan_untuk_laporan.md; sidang/bukti_demo"),
    ("Ambang (S5, S6)", "dipilih di data uji; akurasi p>t, presisi p>=t", "dipilih di validasi/fold dalam; semua metrik dari satu prediksi",
     "data uji tidak boleh ikut memilih", "angka F1 turun ke tingkat realistis", "build_15 best_threshold(); deteksi_fault_core.py"),
    ("Evaluasi", "satu split berurutan 60/20/20", "StratifiedGroupKFold blok waktu + uji statistik + purge + label acak",
     "satu split tidak memberi variasi", "NB11: JSD 11/15 sel, p = 0,076; label acak F1 0,446–0,508", "statistik_sel.csv; efek_purge_permutasi.csv"),
    ("Simulator fault", "drift 0,15–0,2 per sampel, bias seluruh deret", "drift 1 poin/hari per episode; bias/drift/spike per episode (NB14/NB15)",
     "fault harus mungkin terjadi secara fisik", "NB11 v2 → v3: p 0,0504 → 0,076", "commit 9963cd2"),
    ("Pelatihan", "scipy least_squares (Jacobian numerik), init randn", "LM asli dengan Jacobian analitik ber-self-check, init Xavier",
     "permintaan pembimbing: LM sesuai paper", "NB09 RQ2: LM juara 4/10 sel, BR 5/10, Adam 1/10", "hasil09/exports/09_ringkasan_rq.txt"),
], columns=["aspek", "paper / kode Des 2025", "versi terbaru", "kenapa", "dampak", "sumber"])
with pd.option_context("display.max_colwidth", 80):
    print(evolusi.to_string(index=False))
""")

md(r"""
### Angka hasil terbaru dari file (bukan diketik)

Bila dataset `sidang-bahan` terpasang, sel ini membaca ringkasan statistik NB10/NB11/NB14
langsung dari `statistik_sel.csv`.
""")

code(r"""
kandidat = [Path("/kaggle/input/sidang-bahan/statistik_sel.csv"), Path(".kaggle_push/sidang_bahan_data/statistik_sel.csv"),
            Path("../.kaggle_push/sidang_bahan_data/statistik_sel.csv")]
f = next((p for p in kandidat if p.exists()), None)
if f is None:
    print("statistik_sel.csv tidak ditemukan; lewati.")
else:
    st = pd.read_csv(f)
    print(st[st["N"] == "gabungan"][["notebook", "n_sel", "menang_JSD", "rerata_selisih", "p_t", "cohen_dz", "ci95_bawah", "ci95_atas"]].to_string(index=False))
""")

md(r"""
## §6 Ringkasan untuk sidang

**Yang masih berlaku dari paper**
- Kerangka masalah: deteksi fault sensor kelembaban dengan ANN-LM.
- Empat jenis fault mengikuti Saeed dkk.
- Lima skenario.
- Pertimbangan biaya komputasi untuk perangkat edge.

**Yang tidak bisa dipertahankan dengan kode Des 2025**

| Klaim paper | Kenapa tidak bisa dipertahankan | Bukti di notebook ini |
|---|---|---|
| Akurasi 0,996–0,999 adalah akurasi deteksi fault | Labelnya ambang level sinyal bersih; data tanpa fault pun mencapai angka yang sama | §3 S4, §4 |
| Arsitektur A/B/C dibandingkan | Hidden layer 2 tidak pernah dipakai | §3 S1 |
| Waktu, CPU, dan memori per arsitektur | Diukur sekali per skenario untuk seluruh grid; CPU % bergantung jumlah core | §3 S2, S3 |
| Metrik dari data uji yang tidak memihak | Ambang dipilih dengan label uji | §3 S5 |
| Presisi = recall = 1 dengan akurasi 0,997 | Tidak konsisten pada satu ambang; artefak dua aturan ambang | §3 S6 |

**Versi terbaru menjawab semuanya:**
- label fault yang benar;
- arsitektur yang benar-benar dilatih;
- ambang dari validasi;
- CV berkelompok + uji statistik;
- simulator fisis;
- LM asli.

Harga perubahannya: angka turun ke F1 ±0,45–0,75 dan klaim dibatasi menjadi "konsisten".
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"sel-{i:02d}"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"Tersimpan {OUT.name}: {len(cells)} sel")
