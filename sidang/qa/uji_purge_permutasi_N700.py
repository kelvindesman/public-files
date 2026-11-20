#!/usr/bin/env python3
"""uji_purge_permutasi_N700.py — dua uji tambahan untuk menjawab penguji.

source: sidang/qa/02_kelemahan.md A1 (window bersebelahan berbagi 50 % sampel
antar grup CV) dan rencana sidang 2026-11-20 (kontrol negatif label acak).

Data: fitur tersimpan `models/fitur_N700.npz` (N = 700, sama dengan notebook 11 v3
dan model sidang). Untuk tiap skenario S1–S5 dan tiap metode (EDM, JSD):

  standar   StratifiedGroupKFold 5 lipatan, grup = blok waktu (sama dengan NB11)
  purge     lipatan yang sama, tetapi window latih yang rentang waktunya
            bertumpang-tindih dengan window uji mana pun dibuang dari data latih
  permutasi lipatan standar, label data latih diacak (kontrol negatif:
            F1 harus jatuh ke tingkat tebakan)

Arsitektur tetap per skenario/metode = arsitektur model tersimpan
(`models/meta.json`), LM 40 iterasi, tanpa batas waktu dinding (supaya
deterministik). Ini BUKAN ulangan penuh NB11 (tanpa grid search di dalam lipatan);
tujuannya mengukur **selisih** standar vs purge pada protokol yang sama.

Hasil: sidang/qa/hasil_uji/purge_permutasi_N700.csv (satu baris per fit,
ditulis bertahap; baris yang sudah ada dilewati saat dijalankan ulang).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import deteksi_fault_core as core  # noqa: E402

OUT = Path(__file__).with_name("hasil_uji") / "purge_permutasi_N700.csv"
KOLOM = ["skenario", "metode", "varian", "lipatan", "f1", "n_latih", "n_uji", "n_dibuang_purge",
         "hidden", "aktivasi", "status_lm", "detik"]
WIN = 700
PERMUTASI_UNTUK = {"S1", "S5"}


def tugas():
    d = np.load(ROOT / "models" / "fitur_N700.npz")
    meta = json.load(open(ROOT / "models" / "meta.json"))
    arsi = {(m["scenario_short"], m["method"]): (tuple(m["hidden"]), m["activation"]) for m in meta["models"]}
    out = []
    for nama, classes in core.LADDER.items():
        s = core.SCEN_SHORT[nama]
        idx, y = core.build_scenario(d["y_cond"], classes)
        g, st = d["groups"][idx], d["starts"][idx]
        k = int(min(core.N_SPLITS, len(np.unique(g)), np.min(np.bincount(y))))
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=core.RANDOM_SEED)
        for metode, F_all in [("EDM-Fuzzy", d["F_edm"]), ("JSD-Fuzzy", d["F_jsd"])]:
            F = F_all[idx]
            h, a = arsi[(s, metode)]
            for i, (tr, te) in enumerate(skf.split(F, y, groups=g)):
                # purge: buang window latih yang rentangnya [start, start+WIN) beririsan dengan window uji
                te_st = np.sort(st[te])
                pos = np.searchsorted(te_st, st[tr])
                kiri = np.abs(st[tr] - te_st[np.clip(pos - 1, 0, len(te_st) - 1)])
                kanan = np.abs(st[tr] - te_st[np.clip(pos, 0, len(te_st) - 1)])
                tumpang = np.minimum(kiri, kanan) < WIN
                tr_purge = tr[~tumpang]
                out.append((s, metode, "standar", i, F, y, tr, te, h, a, 0, None))
                out.append((s, metode, "purge", i, F, y, tr_purge, te, h, a, int(tumpang.sum()), None))
                if s in PERMUTASI_UNTUK:
                    out.append((s, metode, "permutasi", i, F, y, tr, te, h, a, 0, 1000 + i))
    return out


def jalankan(s, metode, varian, i, F, y, tr, te, h, a, n_buang, seed_perm):
    t0 = time.time()
    ytr = y[tr].copy()
    if seed_perm is not None:
        np.random.default_rng(seed_perm).shuffle(ytr)
    if len(np.unique(ytr)) < 2 or len(np.unique(y[te])) < 2:
        f1, status = float("nan"), "kelas_kurang"
    else:
        pipe = core.make_lm_pipeline(h, a, core.LM_ITERS_REFIT, max_train_s=1e9)
        pipe.fit(F[tr], ytr)
        f1 = f1_score(y[te], pipe.predict(F[te]), average="macro", zero_division=0)
        status = pipe.named_steps["lm"].hist_["status"]
    return {"skenario": s, "metode": metode, "varian": varian, "lipatan": i, "f1": f"{f1:.6f}",
            "n_latih": len(tr), "n_uji": len(te), "n_dibuang_purge": n_buang,
            "hidden": str(h), "aktivasi": a, "status_lm": status, "detik": f"{time.time() - t0:.1f}"}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    selesai = set()
    if OUT.exists():
        for r in csv.DictReader(open(OUT, encoding="utf-8")):
            selesai.add((r["skenario"], r["metode"], r["varian"], int(r["lipatan"])))
    semua = [t for t in tugas() if (t[0], t[1], t[2], t[3]) not in selesai]
    # urutkan: varian murah dulu tidak perlu; urutkan per skenario supaya hasil awal cepat terbaca
    print(f"{len(semua)} fit tersisa (sudah {len(selesai)})", flush=True)
    baru = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KOLOM)
        if baru:
            w.writeheader()
        par = Parallel(n_jobs=2, return_as="generator_unordered")
        for r in par(delayed(jalankan)(*t) for t in semua):
            w.writerow(r); f.flush()
            print(r["skenario"], r["metode"], r["varian"], r["lipatan"], r["f1"], r["detik"], "s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
