"""Latih SEKALI model final ANN-LM (EDM-Fuzzy dan JSD-Fuzzy) lalu simpan ke disk.

Tujuan: saat sidang/Zoom tidak perlu training ulang 10 jam. Notebook demo
(`sidang/03_Demo_Model_Tersimpan.ipynb`) cukup membuka file di `models/`.

Alur sama dengan notebook 11 v3, window N = 700 (5 menit/sampel = 2,4 hari):
data -> sinkron 5 menit -> injeksi 16 kondisi fault -> window -> entropi
multiskala EDM/JSD (4 sensor x 15 skala = 60 fitur) -> untuk tiap skenario
S1..S5: cari arsitektur dengan grid yang sama (inner grouped CV) -> latih
ulang di SEMUA window skenario itu -> simpan.

Angka performa resmi tetap dari CV notebook 11 (model di sini dilatih di semua
data, jadi akurasinya pada data latih bukan ukuran generalisasi).

Pakai:
    python simpan_model.py                  # run penuh (Kaggle: +-1 jam)
    SMOKE=1 python simpan_model.py          # uji cepat lokal (data kecil, grid kecil)
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import deteksi_fault_core as core

WIN = 700
METHODS = ["EDM-Fuzzy", "JSD-Fuzzy"]
SMOKE = os.environ.get("SMOKE", "0") == "1"
OUT = Path(os.environ.get("MODEL_DIR", "models_smoke" if SMOKE else "models"))
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

# Angka CV resmi notebook 11 v3 (hasil11/exports/11_rq4_performa_ann.csv, N=700),
# disalin supaya demo bisa menampilkan "seberapa sering model ini benar" tanpa file lain.
CV_F1_NB11 = {
    ("S1", "EDM-Fuzzy"): 0.5945, ("S1", "JSD-Fuzzy"): 0.6712,
    ("S2", "EDM-Fuzzy"): 0.3283, ("S2", "JSD-Fuzzy"): 0.3469,
    ("S3", "EDM-Fuzzy"): 0.1836, ("S3", "JSD-Fuzzy"): 0.1962,
    ("S4", "EDM-Fuzzy"): 0.3386, ("S4", "JSD-Fuzzy"): 0.4257,
    ("S5", "EDM-Fuzzy"): 0.8037, ("S5", "JSD-Fuzzy"): 0.8718,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get("GIT_COMMIT", "unknown")


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    max_per_class = 4 if SMOKE else core.MAX_PER_CLASS
    scen_keys = list(core.LADDER)[:2] if SMOKE else list(core.LADDER)
    hidden_grid = [(16,), (16, 8)] if SMOKE else core.HIDDEN_GRID
    act_grid = ["tanh"] if SMOKE else core.ACT_GRID
    if SMOKE:
        core.LM_ITERS_REFIT = 5
        log("SMOKE=1: data kecil, grid kecil, hanya untuk uji alur (bukan model sidang)")

    # 1) data + sinkron
    df_sync = core.sync_broker(core.load_sensor_data())
    X = df_sync.to_numpy(dtype=float)
    log(f"data tersinkron {X.shape} ({len(X) * core.SAMPLING_SECONDS / 86400:.1f} hari)")

    # 2) window + label (seed sama dengan notebook 11)
    W, y_cond, Ysens, starts = core.segment_and_label(X, WIN)
    W, y_cond, Ysens, starts = core.balanced_subsample(W, y_cond, Ysens, starts, max_per_class,
                                                       core.RANDOM_SEED)
    groups = starts // WIN
    log(f"window N={WIN}: {W.shape[0]} window, {len(np.unique(y_cond))} kondisi, "
        f"{len(np.unique(groups))} blok waktu")

    # 3) fitur entropi
    FEAT, feat_s = {}, {}
    for meth in METHODS:
        t0 = time.time()
        FEAT[meth] = core.sanitize(core.concat_multisensor_features(W, meth, seed=7, n_jobs=N_JOBS))
        feat_s[meth] = round(time.time() - t0, 1)
        log(f"fitur {meth}: {FEAT[meth].shape} dalam {feat_s[meth]} s")
    np.savez_compressed(OUT / f"fitur_N{WIN}.npz",
                        F_edm=FEAT["EDM-Fuzzy"], F_jsd=FEAT["JSD-Fuzzy"],
                        y_cond=y_cond, Ysens=Ysens, starts=starts, groups=groups)

    # 4) tiap skenario x metode: grid -> latih final -> simpan
    rows = []
    for sc in scen_keys:
        classes = core.LADDER[sc]
        keep, yy = core.build_scenario(y_cond, classes)
        grp = groups[keep]
        short = core.SCEN_SHORT[sc]
        for meth in METHODS:
            t0 = time.time()
            F = FEAT[meth][keep]
            ranking = core.inner_search_best(F, yy, grp, n_jobs=N_JOBS,
                                             hidden_grid=hidden_grid, act_grid=act_grid)
            best_h, best_a, best_f1 = ranking[0]
            det = core.make_lm_pipeline(best_h, best_a, core.LM_ITERS_REFIT, core.LM_REFIT_BUDGET_S)
            det.fit(F, yy)
            train_acc = float((det.predict(F) == yy).mean())

            fault = yy != 0
            sens = None
            sens_acc = None
            if fault.sum() >= 10:
                sens = core.make_sensor_pipeline(best_h, best_a)
                sens.fit(F[fault], Ysens[keep][fault])
                sens_acc = float((sens.predict(F[fault]) == Ysens[keep][fault]).all(axis=1).mean())

            tag = "JSD" if meth.startswith("JSD") else "EDM"
            fname = f"model_N{WIN}_{short}_{tag}.joblib"
            bundle = {
                "scenario": sc, "scenario_short": short, "method": meth, "window": WIN,
                "class_names": [c[0] for c in classes],
                "hidden": list(best_h), "activation": best_a,
                "inner_cv_f1": best_f1, "grid_ranking": ranking,
                # simpan bentuk numpy murni supaya bisa dibuka di versi scikit-learn mana pun
                "detektor": core.SavedLM.from_pipeline(det, "kelas"),
                "sensor_id": None if sens is None else core.SavedLM.from_pipeline(sens, "sensor"),
                "n_train": int(len(yy)), "n_train_sensor": int(fault.sum()),
                "train_acc": train_acc, "train_acc_sensor_all4": sens_acc,
                "cv_f1_nb11": CV_F1_NB11.get((short, meth)),
                "lm_status": det.named_steps["lm"].hist_["status"],
            }
            joblib.dump(bundle, OUT / fname, compress=3)
            rows.append({k: bundle[k] for k in ("scenario_short", "method", "hidden", "activation",
                                                 "inner_cv_f1", "n_train", "train_acc",
                                                 "train_acc_sensor_all4", "cv_f1_nb11", "lm_status")}
                        | {"file": fname, "train_s": round(time.time() - t0, 1)})
            log(f"{short} {meth}: hidden={best_h} act={best_a} innerF1={best_f1:.3f} "
                f"trainAcc={train_acc:.3f} -> {fname} ({rows[-1]['train_s']} s)")

    shutil.copy(core.__file__, OUT / "deteksi_fault_core.py")
    meta = {
        "dibuat": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(), "smoke": SMOKE, "window": WIN,
        "sampling_detik": core.SAMPLING_SECONDS, "t_scales": core.T_SCALES,
        "m": core.m, "r_ratio": core.r_ratio, "n_ref": core.n_ref,
        "jsd": {"bins": core.JSD_BINS, "h_ratio": core.JSD_H_RATIO,
                "range_sd": core.JSD_RANGE_SD, "r_jsd_ratio": core.JSD_R_RATIO},
        "max_per_class": max_per_class, "n_window": int(W.shape[0]),
        "feature_seconds": feat_s, "total_seconds": round(time.time() - t_start, 1),
        "python": sys.version.split()[0], "numpy": np.__version__, "sklearn": sklearn.__version__,
        "joblib": joblib.__version__, "mesin": f"{platform.machine()} cpu={os.cpu_count()}",
        "catatan": "Model dilatih di semua window skenario. Akurasi data latih BUKAN ukuran "
                   "generalisasi; angka resmi = cv_f1_nb11 (CV grouped notebook 11 v3).",
        "models": rows,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(pd.DataFrame(rows).to_string(index=False))
    log(f"selesai dalam {meta['total_seconds']} s -> {OUT}/")


if __name__ == "__main__":
    main()
