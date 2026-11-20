#!/usr/bin/env python3
"""statistik_tambahan.py — uji pelengkap untuk perbandingan JSD-Fuzzy vs EDM-Fuzzy.

source: rencana sidang 2026-11-20 (serangan statistik penguji) dan pustaka
  Nadeau & Bengio (2003) — corrected resampled t-test untuk lipatan CV yang
  data latihnya bertumpang-tindih; Cohen (1988) — ukuran efek dz untuk data
  berpasangan; Efron & Tibshirani (1993) — interval kepercayaan bootstrap;
  Holm (1979) — koreksi perbandingan berganda.

Bagian A (tingkat sel, dari ekspor yang sudah ada): untuk NB10, NB11, NB14,
  tiap N dan gabungan: rata-rata selisih F1 (JSD − EDM), t berpasangan dan
  Wilcoxon (sama dengan builder), Cohen dz, CI bootstrap 95 % (persentil,
  10.000 ulangan, seed 2026), dan p Holm untuk tiga N dalam satu notebook.
Bagian B (tingkat lipatan, dari sidang/qa/hasil_uji/purge_permutasi_N700.csv bila
  ada): per skenario, selisih F1 per lipatan untuk varian standar dan purge,
  t terkoreksi Nadeau–Bengio, dz, dan efek purge (F1 purge − F1 standar);
  kontrol permutasi dibandingkan dengan tingkat tebakan.

Keluaran: sidang/qa/hasil_uji/statistik_sel.csv, statistik_lipatan.csv, dan
ringkasan di stdout. Tidak ada pelatihan model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("hasil_uji")
SUMBER_SEL = {
    "NB10": ROOT / "sidang/bukti_kaggle/jsd-vs-edm-dengan-ann-lm-asli-paper-pmdc/exports/10_rq5_f1_berpasangan.csv",
    "NB11": ROOT / "hasil11/exports/11_rq5_f1_berpasangan.csv",
    "NB14": ROOT / "hasil14/exports/14_rq5_f1_berpasangan.csv",
}
B = 10_000
SEED = 2026
JUMLAH_KELAS = {"S1": 2, "S2": 5, "S3": 7, "S4": 5, "S5": 2}   # deteksi_fault_core.LADDER


def cohen_dz(d) -> float:
    d = np.asarray(d, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def bootstrap_ci(d, b=B, seed=SEED, alpha=0.05) -> tuple[float, float]:
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(b, len(d)), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def t_nadeau_bengio(d, n_latih: float, n_uji: float) -> tuple[float, float]:
    """Corrected resampled t (Nadeau & Bengio 2003): var × (1/k + n_uji/n_latih), df = k−1."""
    d = np.asarray(d, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt(var * (1.0 / k + n_uji / n_latih))
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _baca_sel(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    d = d[pd.to_numeric(d["N_window"], errors="coerce").notna()].copy()
    d["N_window"] = d["N_window"].astype(float).astype(int)
    for c in ["EDM-Fuzzy F1", "JSD-Fuzzy F1"]:
        d[c] = d[c].astype(float)
    return d


def bagian_a() -> pd.DataFrame:
    baris = []
    for nb, p in SUMBER_SEL.items():
        d = _baca_sel(p)
        blok = []
        for N, g in d.groupby("N_window"):
            blok.append((str(N), g))
        blok.append(("gabungan", d))
        for N, g in blok:
            a, b = g["EDM-Fuzzy F1"].to_numpy(), g["JSD-Fuzzy F1"].to_numpy()
            diff = b - a
            t, p_t = stats.ttest_rel(b, a)
            try:
                p_w = stats.wilcoxon(b, a).pvalue
            except ValueError:
                p_w = float("nan")
            lo, hi = bootstrap_ci(diff)
            baris.append({"notebook": nb, "N": N, "n_sel": len(diff), "menang_JSD": int((diff > 0).sum()),
                          "rerata_selisih": diff.mean(), "t": t, "p_t": p_t, "p_wilcoxon": p_w,
                          "cohen_dz": cohen_dz(diff), "ci95_bawah": lo, "ci95_atas": hi})
    df = pd.DataFrame(baris)
    df["p_t_holm_3N"] = np.nan
    for nb in df.notebook.unique():
        m = (df.notebook == nb) & (df.N != "gabungan")
        df.loc[m, "p_t_holm_3N"] = holm(df.loc[m, "p_t"])
    return df


def bagian_b() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    p = OUT / "purge_permutasi_N700.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    baris = []
    for (s, var), g in d[d.varian.isin(["standar", "purge"])].groupby(["skenario", "varian"]):
        w = g.pivot(index="lipatan", columns="metode", values="f1")
        if not {"EDM-Fuzzy", "JSD-Fuzzy"} <= set(w.columns):
            continue
        w = w.dropna()
        if len(w) < 2:
            continue
        diff = (w["JSD-Fuzzy"] - w["EDM-Fuzzy"]).to_numpy()
        n_latih = g.n_latih.mean(); n_uji = g.n_uji.mean()
        t, p_nb = t_nadeau_bengio(diff, n_latih, n_uji)
        _, p_t = stats.ttest_rel(w["JSD-Fuzzy"], w["EDM-Fuzzy"])
        baris.append({"skenario": s, "varian": var, "k": len(diff),
                      "F1_EDM": w["EDM-Fuzzy"].mean(), "F1_JSD": w["JSD-Fuzzy"].mean(),
                      "rerata_selisih": diff.mean(), "p_t_biasa": p_t, "t_NB": t, "p_NB": p_nb,
                      "cohen_dz": cohen_dz(diff), "n_latih": n_latih, "n_uji": n_uji})
    lip = pd.DataFrame(baris)
    # efek purge per skenario × metode, plus kontrol permutasi
    efek = []
    for (s, met), g in d.groupby(["skenario", "metode"]):
        st = g[g.varian == "standar"].set_index("lipatan").f1
        pu = g[g.varian == "purge"].set_index("lipatan").f1
        pe = g[g.varian == "permutasi"].set_index("lipatan").f1
        both = st.index.intersection(pu.index)
        efek.append({"skenario": s, "metode": met, "F1_standar": st.mean(), "F1_purge": pu.mean(),
                     "purge_minus_standar": (pu[both] - st[both]).mean() if len(both) else np.nan,
                     "lipatan": len(both), "F1_permutasi": pe.mean() if len(pe) else np.nan,
                     # kelas seimbang + tebakan acak -> F1 makro harapan ≈ 1/C
                     "tebakan_F1_makro": 1.0 / JUMLAH_KELAS[s]})
    ef = pd.DataFrame(efek)
    return lip, ef


def main() -> int:
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    a = bagian_a()
    a.to_csv(OUT / "statistik_sel.csv", index=False, float_format="%.6f")
    print("=== Bagian A: tingkat sel (15 sel = 5 skenario × 3 N) ===")
    print(a.round(4).to_string(index=False))
    bb = bagian_b()
    if bb is None:
        print("\nBagian B dilewati: hasil_uji/purge_permutasi_N700.csv belum ada.")
        return 0
    lip, ef = bb
    lip["p_NB_holm_5skenario"] = np.nan
    for var in lip.varian.unique():
        m = lip.varian == var
        lip.loc[m, "p_NB_holm_5skenario"] = holm(lip.loc[m, "p_NB"])
    lip.to_csv(OUT / "statistik_lipatan.csv", index=False, float_format="%.6f")
    ef.to_csv(OUT / "efek_purge_permutasi.csv", index=False, float_format="%.6f")
    print("\n=== Bagian B: tingkat lipatan N = 700 (arsitektur tetap) ===")
    print(lip.round(4).to_string(index=False))
    print("\n=== Efek purge dan kontrol permutasi ===")
    print(ef.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
