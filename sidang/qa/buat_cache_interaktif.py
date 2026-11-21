#!/usr/bin/env python3
"""buat_cache_interaktif.py — siapkan bahan notebook sidang interaktif.

Notebook interaktif menghitung fault dan entropi LANGSUNG dari pilihan panel
(panjang jendela, hari mulai, sensor, kombinasi fault, intensitas, seed), memakai
fungsi penelitian `deteksi_fault_core` yang sama. Satu jendela N = 1000 butuh
±0,4 s per sensor per metode, jadi tidak perlu tabel pilihan yang dihitung di muka.

File ini menyiapkan dua hal:

1. `sidang/assets/cache_interaktif.npz` — data lapangan yang sudah disinkronkan ke
   5 menit (`X5`, sama dengan masukan `segment_and_label` di simpan_model.py),
   ringkasan per jam untuk grafik besar, dan waktu awal rekaman. Dengan ini notebook
   tidak perlu membaca `data_sensor.csv` (70 MB).
2. `sidang/assets/model_sidang.npz` — bobot semua model tersimpan untuk setiap
   panjang jendela yang tersedia di `models/` (N = 200/700/1000), sebagai numpy
   murni. Prediksi dicek sama dengan `SavedLM` asli pada fitur latih
   `models/fitur_N{N}.npz` (selisih < 1e-9).

Jalankan: ./venv/bin/python sidang/qa/buat_cache_interaktif.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import deteksi_fault_core as core  # noqa: E402

OUT = ROOT / "sidang" / "assets" / "cache_interaktif.npz"
OUT_MODEL = ROOT / "sidang" / "assets" / "model_sidang.npz"
MODELS = ROOT / "models"


def main() -> int:
    t0 = time.time()
    df = core.sync_broker(core.load_sensor_data([str(ROOT / "data_sensor.csv")]))
    X = df.to_numpy(float)
    # ringkasan seluruh data per jam untuk grafik gambaran besar
    jam = X[: (len(X) // 12) * 12].reshape(-1, 12, 4).mean(axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X5=X, ringkas_per_jam=jam, n_sampel_5menit=len(X),
                        mulai=str(df.index[0]), metode=np.array(["EDM-Fuzzy", "JSD-Fuzzy"]))
    print(f"Tersimpan {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) dalam {time.time() - t0:.0f} s")
    ekspor_model()
    return 0


def maju(p: dict, X: np.ndarray, kind: str) -> np.ndarray:
    """Forward pass numpy murni — salinan persis yang dipakai notebook sidang."""
    X = np.array(X, float, copy=True)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.broadcast_to(p["fill"], X.shape)[nan]
    a = (X - p["mean"]) / p["scale"]
    n = len(p["W"])
    for i, (W, b) in enumerate(zip(p["W"], p["b"])):
        a = a @ W + b
        if i < n - 1:
            a = np.maximum(a, 0.0) if p["act"] == "relu" else np.tanh(a)
    if kind == "sensor":
        return a
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def daftar_meta() -> dict[int, dict]:
    """{N: meta} untuk setiap panjang jendela yang modelnya ada di models/."""
    hasil = {}
    for f in sorted(MODELS.glob("meta_N*.json")):
        meta = json.loads(f.read_text())
        hasil[int(meta["window"])] = meta
    if 700 not in hasil and (MODELS / "meta.json").exists():
        hasil[700] = json.loads((MODELS / "meta.json").read_text())
    return dict(sorted(hasil.items()))


def ekspor_model():
    """Simpan bobot semua model (tanpa sklearn) + cek kesamaan prediksi per N.

    Mengimpor deteksi_fault_core memuat scikit-learn; bobot dibongkar di sini supaya
    forward pass di notebook cukup numpy. Kunci: "N{N}|{skenario}|{metode}|...".
    """
    import joblib
    simpan, info = {}, {}
    for N, meta in daftar_meta().items():
        fitur = np.load(MODELS / f"fitur_N{N}.npz")
        selisih = 0.0
        for row in meta["models"]:
            b = joblib.load(MODELS / row["file"])
            assert int(b["window"]) == N, (row["file"], b["window"])
            kunci = f"N{N}|{b['scenario_short']}|{b['method']}"
            uji = fitur["F_edm"] if b["method"] == "EDM-Fuzzy" else fitur["F_jsd"]
            for bagian, mdl in [("det", b["detektor"]), ("sen", b["sensor_id"])]:
                if mdl is None:
                    continue
                lay = core.lm_unpack(mdl.theta, mdl._template())
                p = {"fill": mdl.fill, "mean": mdl.mean, "scale": mdl.scale, "act": mdl.activation,
                     "W": [W for W, _ in lay], "b": [bb for _, bb in lay]}
                kind = "sensor" if bagian == "sen" else "kelas"
                asli = mdl.decision_function(uji) if kind == "sensor" else mdl.predict_proba(uji)
                selisih = max(selisih, float(np.abs(maju(p, uji, kind) - asli).max()))
                simpan[f"{kunci}|{bagian}|fill"] = mdl.fill
                simpan[f"{kunci}|{bagian}|mean"] = mdl.mean
                simpan[f"{kunci}|{bagian}|scale"] = mdl.scale
                simpan[f"{kunci}|{bagian}|act"] = np.array(mdl.activation)
                simpan[f"{kunci}|{bagian}|n"] = np.array(len(lay))
                for i, (W, bb) in enumerate(lay):
                    simpan[f"{kunci}|{bagian}|W{i}"] = W
                    simpan[f"{kunci}|{bagian}|b{i}"] = bb
            simpan[f"{kunci}|kelas"] = np.array(b["class_names"])
            simpan[f"{kunci}|cv_f1_nb11"] = np.array(b["cv_f1_nb11"])
        assert selisih < 1e-9, f"N={N}: forward numpy berbeda dari SavedLM: {selisih}"
        info[N] = {k: meta[k] for k in ["dibuat", "commit", "window"]}
        print(f"N = {N}: {len(meta['models'])} model, selisih maks vs SavedLM pada "
              f"{len(fitur['F_edm'])} jendela latih = {selisih:.2e}")
    simpan["_meta"] = np.array(json.dumps(info))
    np.savez_compressed(OUT_MODEL, **simpan)
    print(f"Tersimpan {OUT_MODEL.relative_to(ROOT)} (N = {list(info)})")


if __name__ == "__main__":
    sys.exit(main())
