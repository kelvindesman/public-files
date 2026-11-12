#!/usr/bin/env python3
"""Build 14_Lengkap_Fault_Baru.ipynb.

Notebook 14 = pipeline notebook 11 UTUH (S1..S5 x window 200/700/1000,
grid search ANN-LM asli, grouped CV, RQ1..RQ6 termasuk identifikasi sensor,
dan biaya komputasi), tetapi simulator fault diganti bentuk baru dari uji
tambahan intensitas gangguan (notebook 13) pada tingkat "sedang":

  * bias episode 5 poin (3 episode x 3 hari, lalu kembali normal);
  * drift linear per episode, total 5 poin per episode;
  * spike relatif x(t)*(1+0,30), posisi acak, kepadatan 1,52%;
  * hardware malfunction tetap (0,08; 0,05);
  * kombinasi: semua komponen pada interval episode yang sama, dan window
    berlabel fault hanya kalau SEMUA komponen ada (spike: >= 1 titik;
    komponen lain: > 1% sampel).

Builder ini membaca build_11_annlm_real_notebook.py dan menambal bagian
fault, label window, nama keluaran, dan catatan markdown. build_11 dan
notebook 11 tidak diubah.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE / "build_11_annlm_real_notebook.py").read_text()
OUT = HERE / "14_Lengkap_Fault_Baru.ipynb"


def sub1(src, old, new):
    n = src.count(old)
    assert n == 1, f"pola harus muncul tepat sekali, muncul {n}x: {old[:70]!r}"
    return src.replace(old, new)


def sub_block(src, start, end, new):
    i = src.index(start)
    j = src.index(end, i) + len(end)
    assert src.count(start) == 1, start
    return src[:i] + new + src[j:]


src = SRC

# 1) keluaran
src = sub1(src, 'OUT = "/Users/kelvin/apps/public-files/11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb"',
           f'OUT = "{OUT}"')
src = re.sub(r'(["/])11_', r'\g<1>14_', src)  # exports/11_* -> exports/14_*, figures prefix

# 2) simulator fault baru
NEW_SIM = r'''def _episode_mask(n, days=DRIFT_EPISODE_DAYS, n_episodes=DRIFT_N_EPISODES):
    # Episode identik notebook 11: mulai di 10%, 40%, 70% deret, panjang 3 hari.
    ep_len = min(n, max(1, int(round(days * 86400.0 / DRIFT_SAMPLE_SECONDS))))
    mask = np.zeros(n, bool); ramp = np.zeros(n)
    for s in (np.linspace(0.10, 0.70, n_episodes) * n).astype(int):
        e = min(n, s + ep_len)
        mask[s:e] = True
        ramp[s:e] = np.arange(1, e - s + 1) / ep_len
    return mask, ramp

FAULT_BIT = {"drift": 1, "spike": 2, "bias": 4, "hardware": 8}

def simulate_drift_fault(x, total=DRIFT_TOTAL_POINTS, seed=None, within=None):
    # Linear per episode: naik dari 0 ke `total` poin selama 3 hari, lalu reset.
    ep, ramp = _episode_mask(len(x))
    return x + total * ramp, ep.astype(np.uint8) * FAULT_BIT["drift"]

def simulate_spike_fault(x, k=SPIKE_REL, density=SPIKE_DENSITY, seed=None, within=None):
    # Spike relatif x(t)*(1+k) di posisi acak; kepadatan dihitung atas kandidat posisi.
    rng = np.random.default_rng(None if seed is None else seed + 1)
    cand = np.arange(len(x)) if within is None else np.flatnonzero(within)
    n_sp = max(1, int(round(density * len(cand))))
    pos = rng.choice(cand, size=min(n_sp, len(cand)), replace=False)
    y = x.copy(); y[pos] = x[pos] * (1.0 + k)
    mask = np.zeros(len(x), np.uint8); mask[pos] = FAULT_BIT["spike"]
    return y, mask

def simulate_bias_fault(x, bias=BIAS_EPISODE_POINTS, seed=None, within=None):
    # Bias episode: offset selama episode, lalu kembali normal.
    ep, _ = _episode_mask(len(x))
    return x + bias * ep, ep.astype(np.uint8) * FAULT_BIT["bias"]

def simulate_hardware_fault(x, stuck_prob=0.08, loss_prob=0.05, seed=None, within=None):
    rng = np.random.default_rng(None if seed is None else seed + 2)
    n = len(x); rv = rng.random(n); idx = rng.integers(n, size=n)
    allow = np.ones(n, bool) if within is None else within
    m1 = (rv < stuck_prob) & allow; y = x.copy(); y[m1] = x[idx[m1]]
    m2 = (rv < loss_prob) & allow; y[m2] = np.nan
    return y, (m1 | m2).astype(np.uint8) * FAULT_BIT["hardware"]

def simulate_multiple_faults(x, faults, seed=None):
    # Kombinasi (>= 2 komponen): semua komponen pada interval episode yang sama.
    within = _episode_mask(len(x))[0] if len(faults) > 1 else None
    y = x.copy(); m_ = np.zeros(len(x), np.uint8)
    for f, kw in faults:
        y, mi = f(y, **kw, seed=seed, within=within); m_ |= mi
    # Jaga domain fisik: injeksi tidak boleh keluar 0-100 %.
    # m_ = bitmask komponen per sampel (FAULT_BIT), bukan bool.
    return np.clip(y, 0.0, 100.0), m_

def simulate_choose_one(x, options, seed=None, within=None):
    rng = np.random.default_rng(seed)
    f, kw = options[rng.integers(len(options))]
    y, m = f(x, **kw, seed=seed, within=within)
    return np.clip(y, 0.0, 100.0), m
'''
src = sub_block(src, "def simulate_drift_fault(", "    return np.clip(y, 0.0, 100.0), m\n", NEW_SIM)

src = sub1(src, '_D = {"intensity": DRIFT_PER_SAMPLE}\n_S = {"intensity": 0.08, "p": 0.015}\n_B = {"bias": 0.08}',
           '_D = {"total": DRIFT_TOTAL_POINTS}\n_S = {"k": SPIKE_REL}\n_B = {"bias": BIAS_EPISODE_POINTS}')

# 3) parameter tingkat "sedang"
src = sub1(src, "DRIFT_PER_SAMPLE = DRIFT_POINTS_PER_DAY * (DRIFT_SAMPLE_SECONDS / 86400.0)\n",
           "DRIFT_PER_SAMPLE = DRIFT_POINTS_PER_DAY * (DRIFT_SAMPLE_SECONDS / 86400.0)\n"
           "\n# --- Notebook 14: fault bentuk baru, tingkat \"sedang\" (notebook 13) ---\n"
           "DRIFT_TOTAL_POINTS = float(os.environ.get(\"DRIFT_TOTAL_POINTS\", 5.0))   # total per episode\n"
           "BIAS_EPISODE_POINTS = float(os.environ.get(\"BIAS_EPISODE_POINTS\", 5.0)) # offset selama episode\n"
           "SPIKE_REL = float(os.environ.get(\"SPIKE_REL\", 0.30))                    # lonjakan relatif\n"
           "SPIKE_DENSITY = 0.0152                                                   # 1,52% titik\n")

# 4) mask bitmask + label window "semua komponen ada"
src = sub1(src, "Y = Xa.copy(); M = np.zeros_like(Y, dtype=bool)",
           "Y = Xa.copy(); M = np.zeros(Y.shape, dtype=np.uint8)")
NEW_LABEL = r'''def window_labels_per_sensor(mask, win, stride, thr=0.02):
    # Notebook 14: window berlabel fault pada sensor s hanya kalau SEMUA komponen
    # yang disuntikkan ke sensor itu ada di window: spike >= 1 titik,
    # komponen lain > thr porsi sampel.
    T = len(mask)
    if win > T:
        return np.zeros((0, mask.shape[1]), dtype=bool)
    mask = np.asarray(mask, dtype=np.uint8)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]   # (nwin, sensor, win)
    lab = np.zeros(Wm.shape[:2], dtype=bool)
    for s in range(mask.shape[1]):
        req = int(np.bitwise_or.reduce(mask[:, s]))
        if req == 0:
            continue
        ok = np.ones(Wm.shape[0], dtype=bool)
        for name, b in FAULT_BIT.items():
            if req & b:
                hit = (Wm[:, s, :] & b) != 0
                ok &= (hit.sum(axis=1) >= 1) if name == "spike" else (hit.mean(axis=1) > thr)
        lab[:, s] = ok
    return lab
'''
src = sub_block(src, "def window_labels_per_sensor(", "    return (Wm.mean(axis=2) > thr)\n", NEW_LABEL)

# 4b) boxplot aman untuk matplotlib lama dan baru (argumen `labels` dihapus di versi baru)
src = sub1(src, "ax.boxplot(groups, labels=names, showfliers=True, widths=0.6)",
           "ax.boxplot(groups, showfliers=True, widths=0.6); ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)")

# 5) catatan markdown + judul, disisipkan sebelum notebook ditulis
NOTE = r'''
NOTE14 = md(r"""# 14 — Pipeline lengkap notebook 11 dengan fault bentuk baru

Notebook ini menjalankan **seluruh** pipeline notebook 11 — lima skenario S1..S5,
window N = 200/700/1000, grid search ANN-LM asli, grouped CV, RQ1..RQ6
(termasuk **identifikasi sensor**) dan **biaya komputasi** — tetapi dengan
fault bentuk baru dari uji tambahan intensitas gangguan (notebook 13),
tingkat **sedang**:

| Gangguan | Notebook 11 | Notebook 14 |
|---|---|---|
| Bias | +0,08 poin, konstan seluruh deret | **5 poin selama 3 episode × 3 hari**, lalu kembali normal |
| Drift | 1 poin/hari per episode (total 3 poin) | linear per episode, **total 5 poin** |
| Spike | periodik tiap 66 sampel, 0,08 × SD deret | **relatif x(t)·(1+0,30)**, posisi acak, kepadatan 1,52% |
| Hardware | 8% stuck + 5% hilang | tidak berubah |
| Kombinasi | komponen ditumpuk di seluruh deret | **semua komponen pada episode yang sama** |
| Label window | porsi sampel fault > 1% | **semua komponen ada**: spike ≥ 1 titik, komponen lain > 1% |

Semua bagian lain identik dengan notebook 11. Teks penjelasan di bawah yang
menyebut nilai fault lama (0,08, 1 poin/hari) berasal dari notebook 11;
nilai yang benar-benar dipakai adalah tabel di atas. Keluaran ditulis ke
`exports/14_*`.
""")
cells.insert(0, NOTE14)
for c in cells:
    if c["cell_type"] == "markdown":
        t = "".join(c["source"])
        t = t.replace("# 11 — EDM-Fuzzy vs JSD-Fuzzy lengkap", "# (Basis notebook 11) EDM-Fuzzy vs JSD-Fuzzy lengkap")
        c["source"] = t.splitlines(keepends=True)

'''
src = sub1(src, "with open(OUT, \"w\") as f:", NOTE + "with open(OUT, \"w\") as f:")

ns = {"__name__": "__main__", "__file__": str(HERE / "build_11_annlm_real_notebook.py")}
import sys
sys.path.insert(0, str(HERE))
exec(compile(src, "build_14_patched", "exec"), ns)
