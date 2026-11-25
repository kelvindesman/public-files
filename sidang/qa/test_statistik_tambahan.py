"""Uji fungsi statistik pelengkap. Jalankan: ./venv/bin/python -m pytest sidang/qa/test_statistik_tambahan.py"""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
import statistik_tambahan as st  # noqa: E402


# source: Nadeau & Bengio (2003), Machine Learning 52:239–281, eq. corrected resampled t:
# t = mean(d) / sqrt((1/k + n_test/n_train) · var(d)). With n_test/n_train = 0 it is the ordinary paired t.
def test_nadeau_bengio_tanpa_koreksi_sama_dengan_t_berpasangan():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.5, 0.05, 5), rng.normal(0.55, 0.05, 5)
    t_nb, p_nb = st.t_nadeau_bengio(b - a, n_latih=1e12, n_uji=0.0)
    t, p = stats.ttest_rel(b, a)
    assert abs(t_nb - t) < 1e-9 and abs(p_nb - p) < 1e-9


# source: Nadeau & Bengio (2003): the correction can only shrink |t| (larger variance term).
def test_nadeau_bengio_koreksi_mengecilkan_t():
    d = np.array([0.02, 0.05, 0.01, 0.04, 0.03])
    t0, _ = st.t_nadeau_bengio(d, 1e12, 0.0)
    t1, p1 = st.t_nadeau_bengio(d, 400, 100)
    assert abs(t1) < abs(t0) and 0 < p1 <= 1


# source: Holm (1979), Scand. J. Statist. 6:65–70, step-down adjusted p = max_{j<=i}((m-j+1) p_(j)), capped at 1.
def test_holm_contoh_tangan():
    adj = st.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


# source: Cohen (1988) dz = mean(d)/sd(d) for paired data.
def test_cohen_dz():
    d = np.array([1.0, 2.0, 3.0])
    assert abs(st.cohen_dz(d) - 2.0 / 1.0) < 1e-12


# source: Efron & Tibshirani (1993) percentile bootstrap; CI of a constant sample collapses to the constant.
def test_bootstrap_konstan():
    lo, hi = st.bootstrap_ci(np.full(15, 0.03), b=500)
    assert abs(lo - 0.03) < 1e-12 and abs(hi - 0.03) < 1e-12


# source: sidang/qa/01_klaim_bukti.csv R11-04 (NB11 pooled paired t p = 0,076) — part A must reproduce the builder's p.
def test_bagian_a_mereproduksi_p_nb11():
    a = st.bagian_a()
    p = float(a[(a.notebook == "NB11") & (a.N == "gabungan")].p_t.iloc[0])
    assert abs(p - 0.0759) < 0.0005
