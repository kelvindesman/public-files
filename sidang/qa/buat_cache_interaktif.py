#!/usr/bin/env python3
"""buat_cache_interaktif.py — hitung sekali semua fitur yang dipakai widget notebook sidang.

Notebook interaktif harus responsif (< 2 detik per sel), sedangkan entropi
multiskala satu window butuh beberapa detik. Karena pilihan widget terbatas,
semua kombinasinya dihitung di sini sekali lalu disimpan ke
`sidang/assets/cache_interaktif.npz`.

Pilihan: hari mulai {10, 40, 69} (ketiganya di dalam episode drift) × sensor
{S1..S4} × fault {drift, spike, bias, hardware, semua} × intensitas {×1, ×10, ×50}.
×1 = parameter simulator penelitian (deteksi_fault_core.SCENARIOS). Intensitas
mengalikan amplitudo drift/spike/bias; peluang hardware dikali min(k, 5) dan
dibatasi 0,6 (supaya tetap peluang).

Fitur dihitung dengan fungsi yang sama persis dengan penelitian
(`deteksi_fault_core.entropy_1d`), seed per sensor = 7 + 19·s seperti
`concat_multisensor_features(..., seed=7)` untuk window pertama. Sensor yang tidak
dirusak memakai fitur window bersih (identik karena data dan seed-nya sama).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import deteksi_fault_core as core  # noqa: E402

OUT = ROOT / "sidang" / "assets" / "cache_interaktif.npz"
WIN = 700
HARI = [10, 40, 69]
FAULT = ["drift", "spike", "bias", "hardware", "semua"]
INTENS = [1, 10, 50]
METODE = ["EDM-Fuzzy", "JSD-Fuzzy"]
SEED = 7


def daftar_fault(nama: str, k: float):
    D = {"intensity": core.DRIFT_PER_SAMPLE * k}
    S = {"intensity": 0.08 * k, "p": 0.015}
    B = {"bias": 0.08 * k}
    H = {"stuck_prob": min(0.6, 0.08 * min(k, 5)), "loss_prob": min(0.6, 0.05 * min(k, 5))}
    peta = {"drift": [(core.simulate_drift_fault, D)], "spike": [(core.simulate_spike_fault, S)],
            "bias": [(core.simulate_bias_fault, B)], "hardware": [(core.simulate_hardware_fault, H)]}
    if nama == "semua":   # urutan sama dengan kondisi "spike+bias+malfunc+drift"
        return peta["spike"] + peta["bias"] + peta["hardware"] + peta["drift"]
    return peta[nama]


def fitur_sensor(x, metode, s):
    return core.entropy_1d(np.asarray(x, float), metode, sd=SEED + 19 * s)


def main() -> int:
    t0 = time.time()
    X = core.sync_broker(core.load_sensor_data([str(ROOT / "data_sensor.csv")])).to_numpy(float)
    per_hari = int(86400 / core.SAMPLING_SECONDS)
    starts = [h * per_hari for h in HARI]
    sig_bersih = np.stack([X[s:s + WIN] for s in starts])                       # (3, WIN, 4)

    sig_fault = np.zeros((len(FAULT), len(INTENS), 4, len(HARI), WIN))
    porsi = np.zeros((len(FAULT), len(INTENS), 4, len(HARI)))
    jobs = []
    for fi, fn in enumerate(FAULT):
        for ki, k in enumerate(INTENS):
            for s in range(4):
                Y, M = core.inject_faults_multisensor(X, daftar_fault(fn, k), [s], seed=2026 + 31 * fi + s)
                for hi, st in enumerate(starts):
                    sig_fault[fi, ki, s, hi] = Y[st:st + WIN, s]
                    porsi[fi, ki, s, hi] = M[st:st + WIN, s].mean()
                    for mi, met in enumerate(METODE):
                        jobs.append(((fi, ki, s, hi, mi), sig_fault[fi, ki, s, hi].copy(), met, s))
    for hi in range(len(HARI)):
        for s in range(4):
            for mi, met in enumerate(METODE):
                jobs.append((("bersih", hi, s, mi), sig_bersih[hi, :, s].copy(), met, s))
    print(f"{len(jobs)} perhitungan entropi", flush=True)
    hasil = Parallel(n_jobs=2)(delayed(fitur_sensor)(x, met, s) for _, x, met, s in jobs)

    T = core.T_SCALES
    fit_bersih = np.zeros((len(METODE), len(HARI), 4 * T))
    fit_fault = np.zeros((len(METODE), len(FAULT), len(INTENS), 4, len(HARI), T))
    for (key, _, _, _), f in zip(jobs, hasil):
        if key[0] == "bersih":
            _, hi, s, mi = key
            fit_bersih[mi, hi, s * T:(s + 1) * T] = f
        else:
            fi, ki, s, hi, mi = key
            fit_fault[mi, fi, ki, s, hi] = f

    # ringkasan seluruh data per jam untuk grafik gambaran besar
    jam = X[: (len(X) // 12) * 12].reshape(-1, 12, 4).mean(axis=1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, hari=np.array(HARI), fault=np.array(FAULT), intensitas=np.array(INTENS),
                        metode=np.array(METODE), sig_bersih=sig_bersih, sig_fault=sig_fault, porsi=porsi,
                        fit_bersih=fit_bersih, fit_fault=fit_fault, ringkas_per_jam=jam,
                        n_sampel_5menit=len(X), win=WIN)
    print(f"Tersimpan {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.1f} MB) dalam {time.time() - t0:.0f} s")
    ekspor_model(fit_bersih, fit_fault)
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


def ekspor_model(fit_bersih, fit_fault):
    """Simpan bobot 10 model N=700 sebagai npz biasa (tanpa sklearn) + cek kesamaan prediksi.

    Mengimpor deteksi_fault_core memuat scikit-learn (±3–6 s), terlalu lambat untuk
    notebook sidang. Di sini bobot SavedLM dibongkar; notebook cukup numpy. Prediksi
    dibandingkan dengan SavedLM asli pada semua fitur cache; selisih harus < 1e-9.
    """
    import json
    import joblib
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    simpan, uji = {}, np.vstack([fit_bersih.reshape(-1, fit_bersih.shape[-1])])
    T = core.T_SCALES
    # tambahkan semua jendela fault (sensor rusak diganti) sebagai data uji kesamaan
    rows = []
    for mi in range(fit_fault.shape[0]):
        for fi in range(fit_fault.shape[1]):
            for ki in range(fit_fault.shape[2]):
                for s in range(4):
                    for hi in range(fit_fault.shape[4]):
                        f = fit_bersih[mi, hi].copy(); f[s * T:(s + 1) * T] = fit_fault[mi, fi, ki, s, hi]
                        rows.append(f)
    uji = np.vstack([uji, np.array(rows)])
    selisih = 0.0
    for row in meta["models"]:
        b = joblib.load(ROOT / "models" / row["file"])
        kunci = f"{b['scenario_short']}|{b['method']}"
        for bagian, mdl in [("det", b["detektor"]), ("sen", b["sensor_id"])]:
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
    assert selisih < 1e-9, f"forward numpy berbeda dari SavedLM: {selisih}"
    simpan["_meta"] = np.array(json.dumps({k: meta[k] for k in ["dibuat", "commit", "window"]}))
    out = ROOT / "sidang" / "assets" / "model_sidang.npz"
    np.savez_compressed(out, **simpan)
    print(f"Tersimpan {out.relative_to(ROOT)}; selisih maks vs SavedLM pada {len(uji)} jendela = {selisih:.2e}")


if __name__ == "__main__":
    sys.exit(main())
