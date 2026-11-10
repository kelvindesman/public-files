#!/usr/bin/env python3
"""Build 13_Uji_Intensitas_Fault.ipynb.

Uji tambahan intensitas gangguan (revisi pasca-sidang). Dikembangkan dari
Bagian A notebook 12, tetapi berdiri sendiri (kernel Kaggle terpisah):

  - dua metode fitur: JSD-Fuzzy dan EDM-Fuzzy;
  - bentuk fault baru: bias episode, spike relatif, kombinasi pada episode sama;
  - grid drift dalam total poin per episode (2, 3, 5, 9, 60);
  - bentuk lama (bias konstan, spike SD periodik, hardware) tetap sebagai pembanding.

Notebook 11 (hasil utama) dan notebook 12 tidak diubah.
"""
import json

OUT = "13_Uji_Intensitas_Fault.ipynb"
REPO_RAW = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(r"""# 13 — Uji tambahan intensitas gangguan (fault injection)

Jawaban revisi pasca-sidang. Intensitas gangguan pada hasil utama (notebook 11)
kecil, sehingga spike, bias, dan drift sering tertukar dengan *fault free*:

- spike 0,08 × SD deret (≈0,6–1,2 poin) setara noise sensor;
- bias disisipkan konstan pada seluruh deret, sehingga hilang oleh
  *centroid shifting* dan tidak mengubah entropi;
- pada gangguan kombinasi notebook 11, drift hanya ada di episode, sehingga
  hanya 11–22% window kombinasi yang benar-benar memuat drift.

Notebook ini menjalankan ulang **Bagian A notebook 12** (sapuan intensitas
fault, klasifikasi biner fault vs normal, ANN-LM tetap) untuk **JSD-Fuzzy dan
EDM-Fuzzy**, dengan bentuk gangguan tambahan:

| Gangguan | Bentuk lama (pembanding) | Bentuk tambahan |
|---|---|---|
| Bias | konstan seluruh deret: 0,08; 0,5; 1; 2; 5 poin | episode 3 × 3 hari: 2; 5; 9 poin, lalu kembali normal |
| Drift | linear per episode | bentuk sama, grid total 2; 3; 5; 9; 60 poin per episode |
| Spike | periodik tiap 66 sampel: 0,08–1,5 × SD deret | relatif x(t)·(1+k), k = 15%, 30%, 50%, posisi acak, kepadatan 1,52%; window berlabel fault kalau memuat ≥ 1 spike |
| Hardware | 3% diganti nilai acak + 5% hilang; grid (0,08; 0,05) … (0,7; 0,5) | tidak berubah |
| Kombinasi | komponen ditumpuk di seluruh deret | semua komponen pada interval episode yang sama; window berlabel kombinasi hanya kalau semua komponen ada |

Nilai penelitian saat ini (bias 0,08; drift total 3; spike 0,08 × SD; hardware
(0,08; 0,05)) tetap ada di grid sebagai pembanding.

> Notebook ini **tidak mengubah** hasil utama notebook 11 maupun notebook 12.
"""))

cells.append(md(r"""## §0 — Konfigurasi

| Variabel | Arti |
|---|---|
| `SWEEP_SMOKE=1` | uji fungsi sangat kecil (2 nilai, 1 seed, 20 window, 2 fold, 5 iterasi) |
| `SWEEP_QUICK=1` | grid sedang, 1 seed |
| `SWEEP_N_WINDOWS` | window per kelas (default 80) |
| `SWEEP_FOLDS` | lipatan CV (default 5) |
| `SWEEP_ITERS` | iterasi LM (default 40) |
| `SWEEP_SEEDS` | jumlah seed (default 3) |
| `SWEEP_METHODS` | metode, dipisah koma (default `JSD,EDM`) |
"""))

cells.append(code(r"""import os, json, time, warnings, platform, itertools
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import FileLink, display

warnings.filterwarnings("ignore")
NOTEBOOK_START = time.time()
def log_stage(label):
    print(f"[t+{(time.time()-NOTEBOOK_START)/60:6.1f} min] {label}", flush=True)

SMOKE = os.environ.get("SWEEP_SMOKE", "0") == "1"
QUICK = os.environ.get("SWEEP_QUICK", "0") == "1"
N_WINDOWS = int(os.environ.get("SWEEP_N_WINDOWS", 20 if SMOKE else 80))
N_FOLDS = int(os.environ.get("SWEEP_FOLDS", 2 if SMOKE else 5))
LM_ITERS = int(os.environ.get("SWEEP_ITERS", 5 if SMOKE else 40))
N_SEEDS = int(os.environ.get("SWEEP_SEEDS", 1 if (SMOKE or QUICK) else 3))
METHODS = [m.strip().upper() for m in os.environ.get("SWEEP_METHODS", "JSD,EDM").split(",") if m.strip()]

RANDOM_SEED = 42
SENSOR_INJECT = 2          # indeks sensor yang disuntik (0-based); S3, bebas blok nol (sama dengan notebook 12)
RESAMPLE_RULE = "5min"
SAMPLING_SECONDS = 300
WIN = 200                  # window N=200 (sama dengan notebook 11/12)
T_SCALES = 15
M_EMB = 2
R_RATIO_EDM = 0.2
N_REF = 128
FAULT_RATIO_THR = 0.01     # 1%, sama dengan notebook 11/12

# Episode (sama dengan drift notebook 11/12): 3 episode x 3 hari di 10%, 40%, 70% deret
EP_POS = (0.10, 0.40, 0.70)
EP_DAYS = 3.0
SPIKE_DENSITY = 0.0152     # 1,52% titik = kepadatan spike periodik tiap 66 sampel

# JSD default (sama dengan notebook 11)
JSD_BINS, JSD_H_RATIO, JSD_RANGE_SD, JSD_R_RATIO = 8, 1.0, 3.0, 0.2

EXPORT_DIR = Path(os.environ.get("SWEEP_EXPORT_DIR", "exports"))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

def export_df(df, name, index=False):
    p = EXPORT_DIR / f"{name}.csv"
    df.to_csv(p, index=index)
    display(FileLink(str(p)))
    return p

def export_fig(fig, name):
    p = EXPORT_DIR / f"{name}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    display(FileLink(str(p)))
    return p

print("Mesin :", platform.processor() or platform.machine(), "| CPU:", os.cpu_count())
print(f"Mode: {'SMOKE' if SMOKE else ('QUICK' if QUICK else 'FULL')} | metode={METHODS} | "
      f"window/kelas={N_WINDOWS} | fold={N_FOLDS} | iterasi LM={LM_ITERS} | seed={N_SEEDS}")
"""))

cells.append(md(r"""## §1 — Data dan sinkronisasi

Sama dengan notebook 12: data @30 detik dirata-ratakan per 5 menit, lalu
disinkronkan ke grid waktu seragam.
"""))

cells.append(code(r"""import requests, glob
from io import StringIO

DATA_URL = "__REPO_RAW__"
DATA_NAME = "data_sensor.csv"

def load_sensor_data():
    for p in [DATA_NAME, f"../input/{DATA_NAME}"] + glob.glob(f"/kaggle/input/**/{DATA_NAME}", recursive=True):
        if os.path.exists(p):
            print("Sumber data: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    print("Sumber data: unduh ->", DATA_URL)
    r = requests.get(DATA_URL, timeout=180); r.raise_for_status()
    return pd.read_csv(StringIO(r.text), index_col=0)

COLS = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
df = load_sensor_data()[COLS].copy()
df.index = pd.to_datetime(df.index, utc=True)
df = df.sort_index()
df = df[~df.index.duplicated(keep="first")]
df = df.resample(RESAMPLE_RULE).mean()
grid = pd.date_range(df.index[0], df.index[-1], freq=f"{SAMPLING_SECONDS}s")
df = df.reindex(grid).interpolate(method="time", limit_direction="both")
df = df.fillna(df.median(numeric_only=True))
X = df.to_numpy(dtype=float)
print("Bentuk data siap pakai:", X.shape, "| durasi hari:", round(len(X)*SAMPLING_SECONDS/86400, 1))
xs = X[:, SENSOR_INJECT]
print(f"Sensor disuntik S{SENSOR_INJECT+1}: SD deret {np.std(xs):.2f} poin | "
      f"rentang {xs.min():.1f}-{xs.max():.1f}")
""".replace("__REPO_RAW__", REPO_RAW)))

cells.append(md(r"""## §2 — Pembangkit fault

Setiap pembangkit mengembalikan sinyal baru dan **mask per komponen**. Aturan
label window per komponen:

- spike relatif: window memuat **≥ 1** spike;
- komponen lain: porsi sampel ter-fault dalam window > 1% (sama dengan notebook 11/12).

Window kombinasi diberi label fault **hanya kalau semua komponennya** memenuhi
aturan di atas.

Tingkat kombinasi (asumsi notebook ini, ditandai di ringkasan): komponen bias,
drift, dan spike memakai tingkat yang sama-sama berujung pada rentang galat
kalibrasi (rendah = bias 2 / drift total 2 / spike 15%, sedang = 5 / 5 / 30%,
tinggi = 9 / 9 / 50%); hardware tetap (0,08; 0,05) karena bentuknya tidak diubah.
"""))

cells.append(code(r"""def episode_mask(n, days=EP_DAYS, pos=EP_POS):
    ep_len = min(n, max(1, int(round(days * 86400.0 / SAMPLING_SECONDS))))
    mask = np.zeros(n, bool); ramp = np.zeros(n)
    for s in (np.asarray(pos) * n).astype(int):
        e = min(n, s + ep_len)
        mask[s:e] = True
        ramp[s:e] = np.arange(1, e - s + 1) / ep_len   # 0..1 dalam satu episode
    return mask, ramp

def sim_drift(x, total_points):
    # linear per episode: naik dari 0 ke total_points selama EP_DAYS hari
    ep, ramp = episode_mask(len(x))
    return x + total_points * ramp, ep.copy()

def sim_bias_const(x, points):
    return x + points, np.ones(len(x), bool)

def sim_bias_episode(x, points):
    ep, _ = episode_mask(len(x))
    return x + points * ep, ep.copy()

def sim_spike_sd(x, amplitude_sd, p=SPIKE_DENSITY):
    # bentuk notebook 11/12: periodik tiap 1/p sampel, amplitudo k x SD deret
    tau = max(1, int(1.0 / p))
    spikes = (np.arange(len(x)) % tau == 0).astype(float) * (amplitude_sd * np.nanstd(x))
    return x + spikes, spikes != 0

def sim_spike_rel(x, k, seed=0, within=None, density=SPIKE_DENSITY):
    # lonjakan relatif x(t)*(1+k) di posisi acak; kepadatan dihitung atas kandidat
    rng = np.random.default_rng(seed)
    cand = np.arange(len(x)) if within is None else np.flatnonzero(within)
    n_sp = max(1, int(round(density * len(cand))))
    pos = rng.choice(cand, size=min(n_sp, len(cand)), replace=False)
    y = x.copy(); y[pos] = x[pos] * (1.0 + k)
    mask = np.zeros(len(x), bool); mask[pos] = True
    return y, mask

def sim_hardware(x, stuck_prob=0.08, loss_prob=0.05, seed=0, within=None):
    rng = np.random.default_rng(seed)
    n = len(x); rv = rng.random(n); idx = rng.integers(n, size=n)
    allow = np.ones(n, bool) if within is None else within
    m1 = (rv < stuck_prob) & allow
    y = x.copy(); y[m1] = x[idx[m1]]
    m2 = (rv < loss_prob) & allow
    y[m2] = np.nan
    return y, (m1 | m2)

COMBO_LEVELS = {
    "rendah": {"bias": 2.0, "drift": 2.0, "spike": 0.15},
    "sedang": {"bias": 5.0, "drift": 5.0, "spike": 0.30},
    "tinggi": {"bias": 9.0, "drift": 9.0, "spike": 0.50},
}
COMBO_HW = (0.08, 0.05)
COMPONENTS = ["drift", "spike", "bias", "hardware"]
COMBOS = ["+".join(c) for r in (2, 3, 4) for c in itertools.combinations(COMPONENTS, r)]

def sim_combo(x, combo, level, seed=0):
    # semua komponen pada interval episode yang sama
    ep, _ = episode_mask(len(x))
    lv = COMBO_LEVELS[level]; y = x.copy(); masks = {}
    for comp in combo.split("+"):
        if comp == "drift":
            y, masks["drift"] = sim_drift(y, lv["drift"])
        elif comp == "bias":
            y, masks["bias"] = sim_bias_episode(y, lv["bias"])
        elif comp == "spike":
            y, masks["spike"] = sim_spike_rel(y, lv["spike"], seed=seed + 1, within=ep)
        elif comp == "hardware":
            y, masks["hardware"] = sim_hardware(y, *COMBO_HW, seed=seed + 2, within=ep)
    return y, masks

def inject(Xa, fault, value, seed=0, sensor=SENSOR_INJECT):
    # kembalikan (Y, {komponen: (mask, aturan)})
    Y = Xa.copy(); x = Y[:, sensor]
    if fault == "bias":
        y, mk = sim_bias_const(x, value); masks = {"bias": mk}
    elif fault == "bias_episode":
        y, mk = sim_bias_episode(x, value); masks = {"bias": mk}
    elif fault == "drift":
        y, mk = sim_drift(x, value); masks = {"drift": mk}
    elif fault == "spike_sd":
        y, mk = sim_spike_sd(x, value); masks = {"spike_sd": mk}
    elif fault == "spike_rel":
        y, mk = sim_spike_rel(x, value, seed=seed); masks = {"spike": mk}
    elif fault == "hardware":
        y, mk = sim_hardware(x, *value, seed=seed); masks = {"hardware": mk}
    elif fault.startswith("combo:"):
        y, masks = sim_combo(x, fault.split(":", 1)[1], value, seed=seed)
    else:
        raise ValueError(fault)
    Y[:, sensor] = y
    Y = np.clip(Y, 0.0, 100.0)
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(dtype=float), masks

def window_starts(n, win):
    return np.arange(0, n - win + 1, max(1, win // 2))

def window_label(masks, n, win):
    # True kalau SEMUA komponen memenuhi aturannya di window tersebut
    st = window_starts(n, win)
    ok = np.ones(len(st), bool)
    for comp, mk in masks.items():
        Mw = np.lib.stride_tricks.sliding_window_view(mk, win)[st]
        ok &= (Mw.sum(axis=1) >= 1) if comp == "spike" else (Mw.mean(axis=1) > FAULT_RATIO_THR)
    return ok
"""))

cells.append(md(r"""## §2b — Cek visual bentuk fault baru

Satu potongan deret sensor S3 di sekitar episode pertama: bias episode, drift,
spike relatif, dan kombinasi empat komponen. Kepadatan spike relatif dicetak.
"""))

cells.append(code(r"""n = len(X); ep, _ = episode_mask(n)
s0 = int(EP_POS[0] * n); ep_len = int(round(EP_DAYS * 86400 / SAMPLING_SECONDS))
sl = slice(max(0, s0 - ep_len // 2), min(n, s0 + ep_len + ep_len // 2))
demo = [("bias_episode", 5.0), ("drift", 9.0), ("spike_rel", 0.30), ("combo:drift+spike+bias+hardware", "sedang")]
fig, axes = plt.subplots(len(demo), 1, figsize=(12, 2.4 * len(demo)), sharex=True)
for ax, (f, v) in zip(axes, demo):
    Y, masks = inject(X, f, v, seed=0)
    ax.plot(X[sl, SENSOR_INJECT], color="gray", lw=0.8, label="asli")
    ax.plot(Y[sl, SENSOR_INJECT], color="C3", lw=0.8, label="terinjeksi")
    ax.set_title(f"{f} = {v}", fontsize=9); ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); export_fig(fig, "13_cek_bentuk_fault"); plt.show()

_, mk = sim_spike_rel(X[:, SENSOR_INJECT], 0.30, seed=0)
print(f"Kepadatan spike relatif: {mk.mean()*100:.2f}% titik (target {SPIKE_DENSITY*100:.2f}%)")
for f, v in demo:
    _, masks = inject(X, f, v, seed=0)
    lab = window_label(masks, n, WIN)
    print(f"{f}={v}: {lab.sum()} window berlabel fault dari {len(lab)}")
"""))

cells.append(md(r"""## §3 — Fitur JSD-Fuzzy dan EDM-Fuzzy

Identik dengan notebook 11. Perbedaan tunggal kedua metode: cara jarak antar
vektor embedding dihitung (Euclidean vs Jensen-Shannon pada soft binning).
"""))

cells.append(code(r"""EPS = 1e-12
LN2 = np.log(2.0)

def coarse_grain_cmse(x, s, k):
    n = (len(x) - k) // s
    if n <= 0:
        return np.array([], dtype=float)
    return x[k:k + n * s].reshape(-1, s).mean(axis=1)

def embed_matrix_centered(y, m_):
    if len(y) < m_:
        return np.empty((0, m_), float)
    V = np.lib.stride_tricks.sliding_window_view(y, m_)
    return V - V.mean(axis=1, keepdims=True)

def _pick_ref(N, nref, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=nref, replace=False) if N > nref else np.arange(N)

# ---- EDM-Fuzzy (Euclidean) ----
def phi_euclidean(V, r, n_ref=128, seed=0):
    N = V.shape[0]
    if N < 3:
        return np.nan
    ref = _pick_ref(N, n_ref, seed)
    A = V[ref]
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(V * V, axis=1, keepdims=True).T
    d = np.sqrt(np.maximum(a2 + b2 - 2 * (A @ V.T), 0.0))
    mu = 1.0 / (1.0 + d / (r + EPS))
    mu[np.arange(len(ref)), ref] = 0.0
    return (mu.sum(axis=1) / (N - 1)).mean()

def edm_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0):
    out = []
    r = r_ratio * np.std(x, ddof=1)
    for s in scales:
        val_k = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m + 2):
                continue
            phi_m = phi_euclidean(embed_matrix_centered(y, m), r, n_ref, seed + 11 * s + k)
            phi_m1 = phi_euclidean(embed_matrix_centered(y, m + 1), r, n_ref, seed + 17 * s + k)
            if phi_m > 0 and phi_m1 > 0 and np.isfinite(phi_m) and np.isfinite(phi_m1):
                val_k.append(np.log(phi_m / phi_m1))
        out.append(np.mean(val_k) if val_k else np.nan)
    return np.array(out, dtype=float)

# ---- JSD-Fuzzy (Gaussian soft binning + Jensen-Shannon) ----
def vectors_to_distributions(V, centers, h):
    D = (V[:, :, None] - centers[None, None, :]) / (h + EPS)
    K = np.exp(-0.5 * D * D)
    P = K.sum(axis=1)
    return P / (P.sum(axis=1, keepdims=True) + 1e-300)

def pairwise_jsd(A, B):
    a = A[:, None, :]; b = B[None, :, :]
    mm = 0.5 * (a + b)
    lm = np.log(mm + EPS)
    d = 0.5 * (np.sum(a * (np.log(a + EPS) - lm), axis=2)
               + np.sum(b * (np.log(b + EPS) - lm), axis=2))
    return np.maximum(d, 0.0)

def phi_jsd(V, centers, h, r_jsd, nref=128, seed=0):
    N = V.shape[0]
    if N < 3:
        return np.nan
    ref = _pick_ref(N, nref, seed)
    P = vectors_to_distributions(V, centers, h)
    d = np.sqrt(pairwise_jsd(P[ref], P))
    mu = 1.0 / (1.0 + d / (r_jsd + EPS))
    mu[np.arange(len(ref)), ref] = 0.0
    return (mu.sum(axis=1) / (N - 1)).mean()

def jsd_entropy_1d(x, scales, bins=8, h_ratio=1.0, range_sd=3.0, r_jsd_ratio=0.2,
                   m_=2, nref=128, seed=0):
    out = []
    sd = np.std(x, ddof=1)
    centers = np.linspace(-range_sd * sd, range_sd * sd, bins)
    h = h_ratio * (centers[1] - centers[0]) if bins > 1 else sd
    r_jsd = r_jsd_ratio * np.sqrt(LN2)
    for s in scales:
        vals = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m_ + 2):
                continue
            p1 = phi_jsd(embed_matrix_centered(y, m_), centers, h, r_jsd, nref, seed + 11*s + k)
            p2 = phi_jsd(embed_matrix_centered(y, m_ + 1), centers, h, r_jsd, nref, seed + 17*s + k)
            if p1 > 0 and p2 > 0 and np.isfinite(p1) and np.isfinite(p2):
                vals.append(np.log(p1 / p2))
        out.append(np.mean(vals) if vals else np.nan)
    return np.array(out, dtype=float)

def features_one_window(win2d, method):
    scales = range(1, T_SCALES + 1)
    if method == "JSD":
        f = [jsd_entropy_1d(win2d[:, s], scales, bins=JSD_BINS, h_ratio=JSD_H_RATIO,
                            range_sd=JSD_RANGE_SD, r_jsd_ratio=JSD_R_RATIO, seed=7 + 19*s)
             for s in range(win2d.shape[1])]
    elif method == "EDM":
        f = [edm_fuzzy_entropy_1d(win2d[:, s], scales, m=M_EMB, r_ratio=R_RATIO_EDM,
                                  n_ref=N_REF, seed=7 + 19*s)
             for s in range(win2d.shape[1])]
    else:
        raise ValueError(method)
    return np.concatenate(f)
"""))

cells.append(md(r"""## §4 — ANN-LM tetap dan metrik

Sama dengan notebook 12: ANN-LM `(32,16)` tanh, 40 iterasi, 5-fold
`StratifiedGroupKFold` per blok waktu. Model tidak dicari ulang per baris
supaya perbedaan bisa diatribusikan ke intensitas fault dan metode fitur.
"""))

cells.append(code(r"""from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from joblib import Parallel, delayed

def lm_init_layers(n_in, hidden, n_out, seed=0):
    rng = np.random.default_rng(seed)
    dims = [n_in, *hidden, n_out]
    layers = []
    for i in range(len(dims) - 1):
        lim = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
        layers.append([rng.uniform(-lim, lim, (dims[i], dims[i + 1])), np.zeros(dims[i + 1])])
    return layers

def lm_pack(layers):
    return np.concatenate([W.ravel() for W, _ in layers] + [b for _, b in layers])

def lm_unpack(theta, template):
    out, i = [], 0
    for W, b in template:
        out.append([theta[i:i + W.size].reshape(W.shape), theta[i + W.size:i + W.size + b.size]])
        i += W.size + b.size
    return out

def lm_forward(layers, Xm, activation="tanh"):
    acts = [Xm]
    for li, (W, b) in enumerate(layers):
        z = acts[-1] @ W + b
        if li == len(layers) - 1:
            acts.append(z)
        elif activation == "relu":
            acts.append(np.maximum(z, 0.0))
        else:
            acts.append(np.tanh(z))
    return acts[-1], acts

def lm_residuals(theta, template, Xm, Y, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), Xm, activation)
    return (out - Y).ravel()

def lm_jac(theta, template, Xm, Y, activation="tanh"):
    layers = lm_unpack(theta, template)
    out, acts = lm_forward(layers, Xm, activation)
    mm, n_out = out.shape
    L = len(layers)
    delta = np.broadcast_to(np.eye(n_out), (mm, n_out, n_out)).astype(float)
    blocks = [None] * L
    for l in range(L - 1, -1, -1):
        W, b = layers[l]
        a_prev = acts[l]
        n_in, n_out_l = W.shape
        dW = np.einsum("ip,icq->icpq", a_prev, delta, optimize=True)
        blocks[l] = (dW.reshape(mm*n_out, n_in*n_out_l), delta.reshape(mm*n_out, n_out_l))
        if l > 0:
            back = np.einsum("icq,pq->icp", delta, W, optimize=True)
            h = acts[l]
            deriv = (h > 0).astype(float) if activation == "relu" else (1.0 - h**2)
            delta = back * deriv[:, None, :]
    J = np.empty((mm*n_out, theta.size))
    col = 0
    for dW, db in blocks:
        J[:, col:col+dW.shape[1]] = dW; col += dW.shape[1]
        J[:, col:col+db.shape[1]] = db; col += db.shape[1]
    return J

def lm_fit(Xm, Y, hidden=(32, 16), activation="tanh", iters=40, seed=0,
           lam0=1e-3, max_train_s=120.0):
    template = lm_init_layers(Xm.shape[1], list(hidden), Y.shape[1], seed)
    theta = lm_pack(template); lam = lam0
    r = lm_residuals(theta, template, Xm, Y, activation)
    E = 0.5 * float(r @ r); t0 = time.perf_counter()
    for _ in range(iters):
        if time.perf_counter() - t0 > max_train_s:
            break
        J = lm_jac(theta, template, Xm, Y, activation)
        g = J.T @ r; A = J.T @ J
        if np.linalg.norm(g) < 1e-10:
            break
        d = np.where(np.diag(A) < 1e-12, 1e-12, np.diag(A).copy())
        try:
            step = np.linalg.solve(A + lam*np.diag(d), -g)
        except np.linalg.LinAlgError:
            lam *= 10.0; continue
        if not np.all(np.isfinite(step)):
            lam *= 10.0; continue
        th_new = theta + step
        r_new = lm_residuals(th_new, template, Xm, Y, activation)
        E_new = 0.5 * float(r_new @ r_new)
        denom = 0.5 * float(step @ (lam*d*step - g))
        rho = (E - E_new) / (denom if abs(denom) > 1e-300 else 1e-300)
        if E_new < E:
            theta, r, E = th_new, r_new, E_new
            lam = lam * max(1/3, 1 - (2*rho - 1)**3) if rho > 0 else lam * 2
        else:
            lam *= 2
        lam = float(np.clip(lam, 1e-12, 1e12))
    return theta, template

def lm_proba(theta, template, Xm, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), np.asarray(Xm, float), activation)
    e = out - out.max(axis=1, keepdims=True)
    P = np.exp(e); return P / P.sum(axis=1, keepdims=True)

def make_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])

def evaluate_binary(F, y, groups, seed=0):
    k = int(min(N_FOLDS, len(np.unique(groups)), np.min(np.bincount(y))))
    if k < 2:
        return None
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan); fold_f1, fold_auc = [], []
    for tr, te in sgkf.split(F, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        prep = make_pipe()
        Ftr = prep.fit_transform(F[tr]); Fte = prep.transform(F[te])
        theta, template = lm_fit(Ftr, np.eye(2)[y[tr]], iters=LM_ITERS, seed=seed)
        p = lm_proba(theta, template, Fte)[:, 1]
        oof[te] = p
        pred = (p >= 0.5).astype(int)
        fold_f1.append(f1_score(y[te], pred, average="macro", zero_division=0))
        if len(np.unique(y[te])) == 2:
            fold_auc.append(roc_auc_score(y[te], p))
    ok = ~np.isnan(oof)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return None
    pred = (oof[ok] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y[ok], pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y[ok], oof[ok])),
        "F1_macro": float(f1_score(y[ok], pred, average="macro", zero_division=0)),
        "TPR": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "FPR": float(fp / (fp + tn)) if (fp + tn) else np.nan,
        "disebut_normal": float(fn / (tp + fn)) if (tp + fn) else np.nan,
        "F1_fold_mean": float(np.mean(fold_f1)), "F1_fold_std": float(np.std(fold_f1)),
        "AUC_fold_mean": float(np.mean(fold_auc)) if fold_auc else np.nan,
        "AUC_fold_std": float(np.std(fold_auc)) if fold_auc else np.nan,
        "n": int(ok.sum()), "k_fold": int(k),
    }
"""))

cells.append(md(r"""## §5 — Window, cache normal, dan evaluasi satu konfigurasi

Fitur normal dihitung **sekali per metode**. Fault dihitung per nilai dan seed.
"""))

cells.append(code(r"""def make_windows(Xa, win):
    st = window_starts(len(Xa), win)
    W = np.lib.stride_tricks.sliding_window_view(Xa, window_shape=win, axis=0)[st]
    return W.transpose(0, 2, 1), st

def feat_matrix(W, method):
    rows = Parallel(n_jobs=-1, prefer="processes")(
        delayed(features_one_window)(W[i], method) for i in range(len(W)))
    return np.vstack(rows)

def normal_cache(method, win=WIN, seed=123):
    Wn, st = make_windows(X, win)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Wn)) if len(Wn) <= N_WINDOWS else np.sort(rng.choice(len(Wn), N_WINDOWS, replace=False))
    return feat_matrix(Wn[idx], method), st[idx]

NORMAL = {}
for method in METHODS:
    log_stage(f"Cache fitur normal {method} N={WIN}")
    NORMAL[method] = normal_cache(method)
    print(method, NORMAL[method][0].shape)

def run_one(fault, value, method, seed=0, win=WIN):
    Y, masks = inject(X, fault, value, seed=seed)
    W, st = make_windows(Y, win)
    keep = window_label(masks, len(Y), win)
    Wf, stf = W[keep], st[keep]
    if len(Wf) == 0:
        return None, 0
    rng = np.random.default_rng(seed + 77)
    take = min(N_WINDOWS, len(Wf))
    idx = np.arange(len(Wf)) if len(Wf) <= take else np.sort(rng.choice(len(Wf), take, replace=False))
    Ff = feat_matrix(Wf[idx], method)
    Fn, stn = NORMAL[method]
    n = min(len(Ff), len(Fn))
    F = np.vstack([Fn[:n], Ff[:n]])
    y = np.r_[np.zeros(n, int), np.ones(n, int)]
    groups = np.r_[stn[:n] // win, stf[idx][:n] // win]
    return evaluate_binary(F, y, groups, seed=seed), int(keep.sum())

def value_label(v):
    return f"({v[0]}; {v[1]})" if isinstance(v, tuple) else str(v)

def value_order(v):
    if isinstance(v, tuple):
        return float(v[0]) + float(v[1])
    if isinstance(v, str):
        return float(list(COMBO_LEVELS).index(v))
    return float(v)

def run_config(fault, value, method, seed=0, kind="tunggal"):
    res, n_lab = run_one(fault, value, method, seed=seed)
    base = {"jenis": kind, "fault": fault, "method": method, "value": value_label(value),
            "value_order": value_order(value), "seed": seed, "window_berlabel": n_lab}
    if res is None:
        return {**base, "status": "no_window"}
    return {**base, **res, "status": "ok"}
"""))

cells.append(md(r"""## §6 — Sapuan intensitas: fault tunggal

`*` = nilai penelitian saat ini (pembanding).

| Fault | Nilai |
|---|---|
| bias (konstan) | 0,08\*; 0,5; 1; 2; 5 poin |
| bias_episode | 2; 5; 9 poin |
| drift (total per episode) | 2; 3\*; 5; 9; 60 poin (= 0,667; 1; 1,667; 3; 20 poin/hari) |
| spike_sd (periodik) | 0,08\*; 0,2; 0,4; 0,8; 1,5 × SD deret |
| spike_rel | 15%; 30%; 50% |
| hardware | (0,08; 0,05)\*; (0,15; 0,10); (0,3; 0,2); (0,5; 0,3); (0,7; 0,5) |

Hasil disimpan ke CSV setelah setiap baris, jadi run yang terputus tetap
meninggalkan hasil parsial.
"""))

cells.append(code(r"""BASELINE = {"bias": "0.08", "drift": "3.0", "spike_sd": "0.08", "hardware": "(0.08; 0.05)"}

if SMOKE:
    GRID = {"bias": [0.08, 5.0], "bias_episode": [9.0], "drift": [3.0, 60.0],
            "spike_sd": [0.08, 1.5], "spike_rel": [0.50], "hardware": [(0.08, 0.05), (0.7, 0.5)]}
elif QUICK:
    GRID = {"bias": [0.08, 2.0, 5.0], "bias_episode": [2.0, 5.0, 9.0], "drift": [2.0, 5.0, 9.0],
            "spike_sd": [0.08, 0.4, 1.5], "spike_rel": [0.15, 0.30, 0.50],
            "hardware": [(0.08, 0.05), (0.3, 0.2), (0.7, 0.5)]}
else:
    GRID = {"bias": [0.08, 0.5, 1.0, 2.0, 5.0],
            "bias_episode": [2.0, 5.0, 9.0],
            "drift": [2.0, 3.0, 5.0, 9.0, 60.0],
            "spike_sd": [0.08, 0.2, 0.4, 0.8, 1.5],
            "spike_rel": [0.15, 0.30, 0.50],
            "hardware": [(0.08, 0.05), (0.15, 0.10), (0.30, 0.20), (0.50, 0.30), (0.70, 0.50)]}

rows = []
log_stage("Sapuan fault tunggal mulai")
for method in METHODS:
    for fault, values in GRID.items():
        for v in values:
            for sd in range(N_SEEDS):
                log_stage(f"tunggal | {method} | {fault}={value_label(v)} | seed={sd}")
                rows.append(run_config(fault, v, method, seed=sd))
                pd.DataFrame(rows).to_csv(EXPORT_DIR / "13_sweep_tunggal.csv", index=False)

dfA = pd.DataFrame(rows)
dfA["nilai_penelitian"] = [BASELINE.get(f) == v for f, v in zip(dfA["fault"], dfA["value"])]
export_df(dfA, "13_sweep_tunggal")
print(dfA.groupby(["method", "fault", "status"]).size().to_string())
"""))

cells.append(md(r"""## §7 — Sapuan kombinasi pada episode yang sama

11 kombinasi (2, 3, dan 4 komponen dari drift, spike relatif, bias episode,
hardware) × 3 tingkat (rendah, sedang, tinggi).
"""))

cells.append(code(r"""if SMOKE:
    COMBO_RUN, LEVEL_RUN = ["drift+bias", "drift+spike+bias+hardware"], ["tinggi"]
elif QUICK:
    COMBO_RUN, LEVEL_RUN = COMBOS, ["sedang"]
else:
    COMBO_RUN, LEVEL_RUN = COMBOS, list(COMBO_LEVELS)

rowsC = []
log_stage("Sapuan kombinasi mulai")
for method in METHODS:
    for combo in COMBO_RUN:
        for level in LEVEL_RUN:
            for sd in range(N_SEEDS):
                log_stage(f"kombinasi | {method} | {combo} | {level} | seed={sd}")
                r = run_config("combo:" + combo, level, method, seed=sd, kind="kombinasi")
                r["kombinasi"] = combo; r["n_komponen"] = combo.count("+") + 1; r["tingkat"] = level
                rowsC.append(r)
                pd.DataFrame(rowsC).to_csv(EXPORT_DIR / "13_sweep_kombinasi.csv", index=False)

dfC = pd.DataFrame(rowsC)
export_df(dfC, "13_sweep_kombinasi")
print(dfC.groupby(["method", "tingkat", "status"]).size().to_string())
"""))

cells.append(md(r"""## §8 — Nilai representatif dan grafik

Aturan sama dengan notebook 12: nilai terkecil dengan rata-rata **TPR ≥ 0,80**
dan **FPR ≤ 0,10** antar-seed. Kalau tidak ada, ditulis batas atas yang diuji
dengan status *tidak terdeteksi sampai batas uji*.
"""))

cells.append(code(r"""okA = dfA[dfA["status"] == "ok"].copy()
FAULT_ORDER = list(GRID.keys())

def agg(d, keys):
    return d.groupby(keys, dropna=False).agg(
        TPR=("TPR", "mean"), TPR_sd=("TPR", "std"), FPR=("FPR", "mean"), FPR_sd=("FPR", "std"),
        F1=("F1_macro", "mean"), AUC=("AUC", "mean"), n_seed=("seed", "nunique"),
        window_berlabel=("window_berlabel", "mean")).reset_index()

tabA = agg(okA, ["method", "fault", "value", "value_order"]).sort_values(["method", "fault", "value_order"])
tabA["nilai_penelitian"] = [BASELINE.get(f) == v for f, v in zip(tabA["fault"], tabA["value"])]
export_df(tabA, "13_tabel_tunggal")

rep_rows = []
for method in METHODS:
    for fault in FAULT_ORDER:
        g = tabA[(tabA["method"] == method) & (tabA["fault"] == fault)].sort_values("value_order")
        if g.empty:
            rep_rows.append({"method": method, "fault": fault, "status": "tidak ada window"}); continue
        lulus = g[(g["TPR"] >= 0.8) & (g["FPR"] <= 0.10)]
        r = lulus.iloc[0] if len(lulus) else g.iloc[-1]
        base = g[g["nilai_penelitian"]]
        rep_rows.append({
            "method": method, "fault": fault,
            "status": "terdeteksi" if len(lulus) else "tidak terdeteksi sampai batas uji",
            "nilai_representatif": r["value"], "TPR": r["TPR"], "FPR": r["FPR"], "F1": r["F1"], "AUC": r["AUC"],
            "nilai_diuji": ", ".join(g["value"]),
            "nilai_penelitian": base["value"].iloc[0] if len(base) else "",
            "TPR_penelitian": base["TPR"].iloc[0] if len(base) else np.nan,
            "FPR_penelitian": base["FPR"].iloc[0] if len(base) else np.nan,
        })
dfRep = pd.DataFrame(rep_rows)
export_df(dfRep, "13_nilai_representatif")
print(dfRep.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, fault in zip(axes.ravel(), FAULT_ORDER):
    for i, method in enumerate(METHODS):
        g = tabA[(tabA["method"] == method) & (tabA["fault"] == fault)].sort_values("value_order")
        x = np.arange(len(g))
        ax.plot(x, g["TPR"], "o-", color=f"C{i}", label=f"TPR {method}")
        ax.plot(x, g["FPR"], "s--", color=f"C{i}", alpha=0.6, label=f"FPR {method}")
        ax.set_xticks(x); ax.set_xticklabels(g["value"], rotation=30, fontsize=8)
    ax.axhline(0.8, color="gray", lw=0.8, ls=":"); ax.axhline(0.1, color="gray", lw=0.8, ls=":")
    ax.set_ylim(-0.02, 1.02); ax.set_title(fault); ax.set_xlabel("nilai"); ax.set_ylabel("laju")
    ax.legend(fontsize=7)
fig.suptitle("Fault tunggal — TPR/FPR JSD-Fuzzy vs EDM-Fuzzy", y=1.01)
fig.tight_layout(); export_fig(fig, "13_sweep_tunggal"); plt.show()
"""))

cells.append(code(r"""okC = dfC[dfC["status"] == "ok"].copy()
tabC = agg(okC, ["method", "kombinasi", "n_komponen", "tingkat"])
tabC["tingkat_urut"] = tabC["tingkat"].map({k: i for i, k in enumerate(COMBO_LEVELS)})
tabC = tabC.sort_values(["method", "n_komponen", "kombinasi", "tingkat_urut"]).drop(columns="tingkat_urut")
tabC["terdeteksi"] = (tabC["TPR"] >= 0.8) & (tabC["FPR"] <= 0.10)
export_df(tabC, "13_tabel_kombinasi")
print(tabC[["method", "kombinasi", "tingkat", "TPR", "FPR", "F1", "AUC", "window_berlabel"]].to_string(index=False))

levels = [l for l in COMBO_LEVELS if l in set(tabC["tingkat"])]
if len(tabC):
    piv = tabC.pivot_table(index="kombinasi", columns=["method", "tingkat"], values="F1")
    piv = piv.reindex(columns=[(m, l) for m in METHODS for l in levels if (m, l) in piv.columns])
    piv = piv.reindex([c for c in COMBOS if c in piv.index])
    fig, ax = plt.subplots(figsize=(1.3 * piv.shape[1] + 4, 0.45 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{m}\n{l}" for m, l in piv.columns], fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("Kombinasi pada episode sama — F1 macro (fault vs normal)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); export_fig(fig, "13_heatmap_kombinasi"); plt.show()
"""))

cells.append(md(r"""## §9 — Ringkasan

Semua angka dibaca dari tabel di atas; tidak ada angka yang ditulis tangan.
"""))

cells.append(code(r"""def fmt(v):
    try:
        return "-" if v is None or not np.isfinite(float(v)) else f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)

lines = ["# Ringkasan uji tambahan intensitas gangguan", "",
         f"- Mode: {'SMOKE' if SMOKE else ('QUICK' if QUICK else 'FULL')} | metode: {', '.join(METHODS)}",
         f"- Window N={WIN} | window/kelas maks {N_WINDOWS} | fold {N_FOLDS} | iterasi LM {LM_ITERS} | seed {N_SEEDS}",
         f"- Sensor disuntik: S{SENSOR_INJECT+1}; episode {len(EP_POS)} x {EP_DAYS:g} hari di "
         + ", ".join(f"{int(p*100)}%" for p in EP_POS) + " deret",
         "- Aturan representatif: nilai terkecil dengan TPR >= 0,80 dan FPR <= 0,10 (rata-rata seed)", "",
         "## Fault tunggal", "",
         "| Metode | Fault | Status | Nilai | TPR | FPR | Nilai penelitian | TPR penelitian | FPR penelitian |",
         "|---|---|---|---|---|---|---|---|---|"]
for _, r in dfRep.iterrows():
    lines.append(f"| {r['method']} | {r['fault']} | {r.get('status', '')} | {r.get('nilai_representatif', '')} | "
                 f"{fmt(r.get('TPR'))} | {fmt(r.get('FPR'))} | {r.get('nilai_penelitian', '') or '-'} | "
                 f"{fmt(r.get('TPR_penelitian'))} | {fmt(r.get('FPR_penelitian'))} |")

lines += ["", "## Kombinasi pada episode sama", "",
          "| Metode | Tingkat | Kombinasi terdeteksi | F1 rata-rata | TPR rata-rata | FPR rata-rata |",
          "|---|---|---|---|---|---|"]
for (method, level), g in tabC.groupby(["method", "tingkat"], sort=False):
    lines.append(f"| {method} | {level} | {int(g['terdeteksi'].sum())}/{len(g)} | {fmt(g['F1'].mean())} | "
                 f"{fmt(g['TPR'].mean())} | {fmt(g['FPR'].mean())} |")

lines += ["", "## Asumsi yang perlu dicek", "",
          "- Tingkat kombinasi (rendah/sedang/tinggi = bias 2/5/9 poin, drift total 2/5/9 poin, spike 15/30/50%) "
          "ditetapkan di notebook ini; hardware di kombinasi tetap (0,08; 0,05).",
          "- Klasifikasi biner fault vs normal pada satu sensor (S3), sama dengan notebook 12 Bagian A; "
          "bukan klasifikasi 16 kondisi notebook 11.",
          "- Spike relatif dan hardware pada kombinasi hanya disisipkan di dalam episode."]
ringkasan = "\n".join(lines) + "\n"
(EXPORT_DIR / "13_ringkasan.md").write_text(ringkasan)
print(ringkasan)
log_stage("selesai")
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                     "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(OUT, "w"), indent=1, ensure_ascii=False)
print("wrote", OUT, "cells:", len(cells))
