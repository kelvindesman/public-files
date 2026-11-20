"""Inti pipeline notebook 11 (EDM-Fuzzy vs JSD-Fuzzy + ANN-LM asli) sebagai modul.

Isi file ini disalin dari sel notebook `11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb`
(dibangun dari `build_11_annlm_real_notebook.py`) supaya:

1. model yang sudah dilatih bisa disimpan dengan joblib dan dibuka lagi di
   proses lain (kelas `LMEstimator` harus bisa di-import, bukan cuma ada di
   dalam notebook), dan
2. notebook demo sidang memakai rumus yang PERSIS sama dengan run penelitian.

Yang dibuang dibanding notebook 11 hanya pengukur waktu/CPU dan guard budget
Kaggle. Rumus entropi, simulator fault, segmentasi, LM, dan grid tidak diubah.
"""
import glob
import os
import time as _time
from io import StringIO

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Konfigurasi (nilai default = notebook 11 v3)
# --------------------------------------------------------------------------
RANDOM_SEED = 42
N_SPLITS = 5
RESAMPLE_RULE = "5min"
SAMPLING_SECONDS = 300
T_SCALES = 15
MAX_PER_CLASS = 100

DRIFT_POINTS_PER_DAY = 1.0
DRIFT_EPISODE_DAYS = 3.0
DRIFT_N_EPISODES = 3
DRIFT_SAMPLE_SECONDS = SAMPLING_SECONDS
DRIFT_PER_SAMPLE = DRIFT_POINTS_PER_DAY * (DRIFT_SAMPLE_SECONDS / 86400.0)

m = 2
r_ratio = 0.2
n_ref = 128

JSD_BINS = 8
JSD_H_RATIO = 1.0
JSD_RANGE_SD = 3.0
JSD_R_RATIO = 0.2

SENSORS = ["S1", "S2", "S3", "S4"]
SENSOR_SUBSET_SIZES = [1, 2, 3, 4]
N_REPEAT_SUBSET = 8
FAULT_RATIO_THR = 0.01

COLS = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
DATA_URL = "https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/data_sensor.csv"
DATA_NAME = "data_sensor.csv"

EPS = 1e-12
LN2 = np.log(2.0)
scales = np.arange(1, T_SCALES + 1)

HIDDEN_GRID = [(16,), (32,), (64,), (16, 8), (32, 8), (32, 16), (32, 32), (64, 16)]
ACT_GRID = ["tanh", "relu"]
LM_ITERS_SEARCH = 6
LM_ITERS_REFIT = 40
LM_INNER_CV = 2
LM_SEARCH_BUDGET_S = 40.0
LM_REFIT_BUDGET_S = 150.0


# --------------------------------------------------------------------------
# §1–§2 data + broker sinkronisasi
# --------------------------------------------------------------------------
def load_sensor_data(extra_paths=()):
    cands = [*extra_paths, DATA_NAME, f"../input/{DATA_NAME}"] + glob.glob(
        f"/kaggle/input/**/{DATA_NAME}", recursive=True)
    for p in cands:
        if os.path.exists(p):
            print("Sumber data: file lokal ->", p)
            return pd.read_csv(p, index_col=0)
    import requests
    print("Sumber data: unduh ->", DATA_URL)
    r = requests.get(DATA_URL, timeout=180)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text), index_col=0)


def sync_broker(df_raw):
    """30 detik -> rata-rata 5 menit di grid waktu rapat (sama dengan §2 notebook 11)."""
    df_broker = df_raw[COLS].copy()
    df_broker.index = pd.to_datetime(df_broker.index, utc=True)
    df_broker = df_broker.sort_index()
    df_broker = df_broker[~df_broker.index.duplicated(keep="first")]
    df_broker = df_broker.resample(RESAMPLE_RULE).mean()
    grid = pd.date_range(df_broker.index[0], df_broker.index[-1], freq=f"{SAMPLING_SECONDS}s")
    df_sync = df_broker.reindex(grid)
    df_sync = df_sync.interpolate(method="time", limit_direction="both")
    df_sync = df_sync.fillna(df_sync.median(numeric_only=True))
    if df_sync.isna().any().any():
        raise ValueError("Masih ada NaN setelah sinkronisasi.")
    return df_sync


# --------------------------------------------------------------------------
# §3 simulator fault
# --------------------------------------------------------------------------
def simulate_drift_fault(x, intensity=DRIFT_PER_SAMPLE, episode_days=DRIFT_EPISODE_DAYS,
                         n_episodes=DRIFT_N_EPISODES, seed=None):
    n = len(x)
    drift = np.zeros(n, dtype=float)
    ep_len = max(1, int(round(episode_days * 86400.0 / DRIFT_SAMPLE_SECONDS)))
    ep_len = min(ep_len, n)
    for s in (np.linspace(0.10, 0.70, n_episodes) * n).astype(int):
        e = min(n, s + ep_len)
        drift[s:e] = intensity * np.arange(1, e - s + 1)
    return x + drift, drift > 1e-12


def simulate_spike_fault(x, intensity=0.08, p=0.015, seed=None):
    tau = max(1, int(1.0 / p)) if p > 0 else len(x)
    spikes = (np.arange(len(x)) % tau == 0).astype(float) * (intensity * np.nanstd(x))
    return x + spikes, spikes != 0


def simulate_bias_fault(x, bias=0.08, seed=None):
    return x + bias, np.ones(len(x), bool)


def simulate_hardware_fault(x, stuck_prob=0.08, loss_prob=0.05, seed=None):
    rng = np.random.default_rng(seed)
    n = len(x)
    rv = rng.random(n)
    idx = rng.integers(n, size=n)
    m1 = rv < stuck_prob
    y = x.copy()
    y[m1] = x[idx[m1]]
    m2 = rv < loss_prob
    y[m2] = np.nan
    return y, (m1 | m2)


def simulate_multiple_faults(x, faults, seed=None):
    y = x.copy()
    m_ = np.zeros(len(x), bool)
    for f, kw in faults:
        y, mi = f(y, **kw, seed=seed)
        m_ |= mi
    return np.clip(y, 0.0, 100.0), m_


def simulate_choose_one(x, options, seed=None):
    rng = np.random.default_rng(seed)
    f, kw = options[rng.integers(len(options))]
    y, m_ = f(x, **kw, seed=seed)
    return np.clip(y, 0.0, 100.0), m_


_D = {"intensity": DRIFT_PER_SAMPLE}
_S = {"intensity": 0.08, "p": 0.015}
_B = {"bias": 0.08}
_H = {"stuck_prob": 0.08, "loss_prob": 0.05}

SCENARIOS = {
    "faulty": [(simulate_choose_one, {"options": [(simulate_drift_fault, _D), (simulate_spike_fault, _S),
                                                   (simulate_bias_fault, _B), (simulate_hardware_fault, _H)]})],
    "drift": [(simulate_drift_fault, _D)],
    "spike": [(simulate_spike_fault, _S)],
    "bias": [(simulate_bias_fault, _B)],
    "hardware": [(simulate_hardware_fault, _H)],
    "bias+malfunc": [(simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "spike+malfunc": [(simulate_spike_fault, _S), (simulate_hardware_fault, _H)],
    "spike+bias": [(simulate_spike_fault, _S), (simulate_bias_fault, _B)],
    "drift+malfunc": [(simulate_drift_fault, _D), (simulate_hardware_fault, _H)],
    "drift+bias": [(simulate_drift_fault, _D), (simulate_bias_fault, _B)],
    "drift+spike": [(simulate_drift_fault, _D), (simulate_spike_fault, _S)],
    "spike+bias+malfunc": [(simulate_spike_fault, _S), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "drift+bias+malfunc": [(simulate_drift_fault, _D), (simulate_bias_fault, _B), (simulate_hardware_fault, _H)],
    "spike+drift+malfunc": [(simulate_spike_fault, _S), (simulate_drift_fault, _D), (simulate_hardware_fault, _H)],
    "drift+spike+bias": [(simulate_drift_fault, _D), (simulate_spike_fault, _S), (simulate_bias_fault, _B)],
    "spike+bias+malfunc+drift": [(simulate_spike_fault, _S), (simulate_bias_fault, _B),
                                  (simulate_hardware_fault, _H), (simulate_drift_fault, _D)],
}
condition_names = ["normal"] + list(SCENARIOS.keys())


# --------------------------------------------------------------------------
# §4 segmentasi window
# --------------------------------------------------------------------------
def make_windows(Xa, win, stride):
    Xn = np.asarray(Xa, dtype=np.float32)
    N = Xn.shape[0]
    if N < win:
        return np.empty((0, win, Xn.shape[1]), dtype=np.float32), np.array([], dtype=int)
    view = sliding_window_view(Xn, window_shape=win, axis=0)
    starts = np.arange(0, N - win + 1, stride, dtype=int)
    return view[starts].transpose(0, 2, 1), starts


def inject_faults_multisensor(Xa, scenario_faults, sensor_subset, seed=0):
    rng = np.random.default_rng(seed)
    Y = Xa.copy()
    M = np.zeros_like(Y, dtype=bool)
    for s in sensor_subset:
        y, m_ = simulate_multiple_faults(Y[:, s], scenario_faults, seed=int(rng.integers(1e9)))
        Y[:, s] = y
        M[:, s] = m_
    Ydf = pd.DataFrame(Y).ffill().bfill()
    Ydf = Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M


def window_labels_per_sensor(mask, win, stride, thr=0.02):
    T = len(mask)
    if win > T:
        return np.zeros((0, mask.shape[1]), dtype=bool)
    Wm = sliding_window_view(mask, window_shape=win, axis=0)[::stride]
    return (Wm.mean(axis=2) > thr)


def segment_and_label(X, win):
    stride = max(1, win // 2)
    rng = np.random.default_rng(RANDOM_SEED)
    Ws, ys, Ys, st = [], [], [], []
    W0, s0 = make_windows(X, win, stride)
    Ws.append(W0)
    ys.append(np.zeros(len(W0), int))
    Ys.append(np.zeros((len(W0), 4), bool))
    st.append(s0)
    for k, (name, faults) in enumerate(SCENARIOS.items(), start=1):
        for rep in range(N_REPEAT_SUBSET):
            subset_size = SENSOR_SUBSET_SIZES[rep % len(SENSOR_SUBSET_SIZES)]
            subset = rng.choice(4, size=subset_size, replace=False)
            Y, M = inject_faults_multisensor(X, faults, subset, seed=int(rng.integers(1e9)))
            sens = window_labels_per_sensor(M, win, stride, thr=FAULT_RATIO_THR)
            Wk, sk = make_windows(Y, win, stride)
            keep = sens.any(axis=1)
            Ws.append(Wk[keep])
            ys.append(np.full(int(keep.sum()), k, int))
            Ys.append(sens[keep])
            st.append(sk[keep])
    return (np.concatenate(Ws), np.concatenate(ys),
            np.concatenate(Ys).astype(int), np.concatenate(st))


def balanced_subsample(W, y, Ysens, starts, max_per_class, seed=0):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > max_per_class:
            idx = rng.choice(idx, size=max_per_class, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return W[keep], y[keep], Ysens[keep], starts[keep]


# --------------------------------------------------------------------------
# §5 entropi multiskala: tahap bersama + dua similarity
# --------------------------------------------------------------------------
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


def _pick_ref(N, n_ref, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(N, size=n_ref, replace=False) if N > n_ref else np.arange(N)


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


def vectors_to_distributions(V, centers, h):
    D = (V[:, :, None] - centers[None, None, :]) / (h + EPS)
    K = np.exp(-0.5 * D * D)
    P = K.sum(axis=1)
    return P / (P.sum(axis=1, keepdims=True) + 1e-300)


def pairwise_jsd(A, B):
    a = A[:, None, :]
    b = B[None, :, :]
    mm = 0.5 * (a + b)
    lm = np.log(mm + EPS)
    d = 0.5 * (np.sum(a * (np.log(a + EPS) - lm), axis=2)
               + np.sum(b * (np.log(b + EPS) - lm), axis=2))
    return np.maximum(d, 0.0)


def phi_jsd(V, centers, h, r_jsd, n_ref=128, seed=0):
    N = V.shape[0]
    if N < 3:
        return np.nan
    ref = _pick_ref(N, n_ref, seed)
    P = vectors_to_distributions(V, centers, h)
    d = np.sqrt(pairwise_jsd(P[ref], P))
    mu = 1.0 / (1.0 + d / (r_jsd + EPS))
    mu[np.arange(len(ref)), ref] = 0.0
    return (mu.sum(axis=1) / (N - 1)).mean()


def jsd_fuzzy_entropy_1d(x, scales, m=2, r_ratio=0.2, n_ref=128, seed=0,
                         bins=JSD_BINS, h_ratio=JSD_H_RATIO,
                         range_sd=JSD_RANGE_SD, r_jsd_ratio=JSD_R_RATIO):
    out = []
    sd = np.std(x, ddof=1)
    centers = np.linspace(-range_sd * sd, range_sd * sd, bins)
    h = h_ratio * (centers[1] - centers[0]) if bins > 1 else sd
    r_jsd = r_jsd_ratio * np.sqrt(LN2)
    for s in scales:
        val_k = []
        for k in range(s):
            y = coarse_grain_cmse(x, s, k)
            if len(y) < (m + 2):
                continue
            phi_m = phi_jsd(embed_matrix_centered(y, m), centers, h, r_jsd, n_ref, seed + 11 * s + k)
            phi_m1 = phi_jsd(embed_matrix_centered(y, m + 1), centers, h, r_jsd, n_ref, seed + 17 * s + k)
            if phi_m > 0 and phi_m1 > 0 and np.isfinite(phi_m) and np.isfinite(phi_m1):
                val_k.append(np.log(phi_m / phi_m1))
        out.append(np.mean(val_k) if val_k else np.nan)
    return np.array(out, dtype=float)


def sanitize(F):
    Fdf = pd.DataFrame(F)
    if Fdf.isna().any().any():
        Fdf = Fdf.fillna(Fdf.median(numeric_only=True))
    return Fdf.to_numpy()


def entropy_1d(x, method, sd):
    key = method.strip().lower()
    if key == "edm-fuzzy":
        return edm_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    if key == "jsd-fuzzy":
        return jsd_fuzzy_entropy_1d(x, scales, m=m, r_ratio=r_ratio, n_ref=n_ref, seed=sd)
    raise ValueError(f"Metode tidak dikenal: {method}")


def _one_window(Wi, method, seed_i):
    return np.concatenate([entropy_1d(Wi[:, s], method, sd=seed_i + 19 * s)
                           for s in range(Wi.shape[1])])


def concat_multisensor_features(W, method, seed=0, n_jobs=-1):
    """(nwin, win, 4) -> (nwin, 4T). Seed per window sama dengan notebook 11."""
    nwin = W.shape[0]
    feats = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_one_window)(np.asarray(W[i], dtype=float), method, seed + 1000 * i)
        for i in range(nwin))
    return np.vstack(feats)


# --------------------------------------------------------------------------
# §7 lima skenario
# --------------------------------------------------------------------------
ALL_FAULT = [c for c in condition_names if c != "normal"]
LADDER = {
    "S1_Fault_Free_vs_Faulty": [("fault free", ["normal"]), ("faulty", ALL_FAULT)],
    "S2_Single_Faults": [("fault free", ["normal"]), ("drift", ["drift"]), ("spike", ["spike"]),
                         ("bias", ["bias"]), ("hardware", ["hardware"])],
    "S3_Two_Fault_Combinations": [("fault free", ["normal"]), ("bias+HW", ["bias+malfunc"]),
                                  ("drift+bias", ["drift+bias"]), ("drift+HW", ["drift+malfunc"]),
                                  ("spike+bias", ["spike+bias"]), ("drift+spike", ["drift+spike"]),
                                  ("spike+HW", ["spike+malfunc"])],
    "S4_Three_Fault_Combinations": [("fault free", ["normal"]), ("drift+bias+HW", ["drift+bias+malfunc"]),
                                    ("drift+spike+bias", ["drift+spike+bias"]),
                                    ("spike+bias+HW", ["spike+bias+malfunc"]),
                                    ("spike+drift+HW", ["spike+drift+malfunc"])],
    "S5_Four_Fault_Combinations": [("fault free", ["normal"]),
                                   ("drift+spike+bias+HW", ["spike+bias+malfunc+drift"])],
}
SCEN_SHORT = {k: k.split("_")[0] for k in LADDER}
cond_to_idx = {c: i for i, c in enumerate(condition_names)}


def build_scenario(y_cond, classes):
    idx_l, y_l = [], []
    for ci, (_, conds) in enumerate(classes):
        want = [cond_to_idx[c] for c in conds if c in cond_to_idx]
        sel = np.where(np.isin(y_cond, want))[0]
        idx_l.append(sel)
        y_l.append(np.full(len(sel), ci, int))
    keep = np.concatenate(idx_l)
    yy = np.concatenate(y_l)
    rng = np.random.default_rng(RANDOM_SEED)
    n_min = min(np.bincount(yy)[np.unique(yy)])
    sel = np.concatenate([rng.choice(np.where(yy == c)[0], size=n_min, replace=False)
                          if (yy == c).sum() > n_min else np.where(yy == c)[0]
                          for c in np.unique(yy)])
    rng.shuffle(sel)
    return keep[sel], yy[sel]


# --------------------------------------------------------------------------
# §10a LM asli (Levenberg–Marquardt)
# --------------------------------------------------------------------------
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


def lm_forward(layers, X, activation="tanh"):
    acts = [X]
    for li, (W, b) in enumerate(layers):
        z = acts[-1] @ W + b
        if li == len(layers) - 1:
            acts.append(z)
        elif activation == "relu":
            acts.append(np.maximum(z, 0.0))
        else:
            acts.append(np.tanh(z))
    return acts[-1], acts


def lm_residuals(theta, template, X, Y, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), X, activation)
    return (out - Y).ravel()


def lm_residual_jacobian(theta, template, X, Y, activation="tanh"):
    layers = lm_unpack(theta, template)
    out, acts = lm_forward(layers, X, activation)
    m_, n_out = out.shape
    L = len(layers)
    delta = np.broadcast_to(np.eye(n_out), (m_, n_out, n_out)).astype(float)
    blocks = [None] * L
    for l in range(L - 1, -1, -1):
        W, b = layers[l]
        a_prev = acts[l]
        n_in, n_out_l = W.shape
        dW = np.einsum("ip,icq->icpq", a_prev, delta, optimize=True)
        blocks[l] = (dW.reshape(m_ * n_out, n_in * n_out_l),
                     delta.reshape(m_ * n_out, n_out_l))
        if l > 0:
            back = np.einsum("icq,pq->icp", delta, W, optimize=True)
            h = acts[l]
            deriv = (h > 0).astype(float) if activation == "relu" else (1.0 - h ** 2)
            delta = back * deriv[:, None, :]
    J = np.empty((m_ * n_out, theta.size))
    col = 0
    for dW, db in blocks:
        J[:, col:col + dW.shape[1]] = dW
        col += dW.shape[1]
        J[:, col:col + db.shape[1]] = db
        col += db.shape[1]
    return J


def lm_fit(X, Y, n_in, hidden, n_out, activation="tanh", iters=40, seed=0,
           lam0=1e-3, max_train_s=60.0):
    template = lm_init_layers(n_in, hidden, n_out, seed)
    theta = lm_pack(template)
    lam = lam0
    hist = {"loss": [], "grad": [], "lam": [], "nfev": 0, "iters": 0, "status": "max_iter"}
    r = lm_residuals(theta, template, X, Y, activation)
    hist["nfev"] += 1
    E = 0.5 * float(r @ r)
    _t0 = _time.perf_counter()
    for k in range(iters):
        if _time.perf_counter() - _t0 > max_train_s:
            hist["status"] = "budget(MAX_TRAIN_S)"
            break
        J = lm_residual_jacobian(theta, template, X, Y, activation)
        g = J.T @ r
        A = J.T @ J
        gn = float(np.linalg.norm(g))
        hist["loss"].append(float(np.mean(r * r)))
        hist["grad"].append(gn)
        hist["lam"].append(float(lam))
        hist["iters"] = k + 1
        if gn < 1e-10:
            hist["status"] = "converged(grad)"
            break
        d = np.diag(A).copy()
        d = np.where(d < 1e-12, 1e-12, d)
        try:
            step = np.linalg.solve(A + lam * np.diag(d), -g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue
        if not np.all(np.isfinite(step)):
            lam *= 10.0
            continue
        theta_new = theta + step
        r_new = lm_residuals(theta_new, template, X, Y, activation)
        hist["nfev"] += 1
        E_new = 0.5 * float(r_new @ r_new)
        denom = 0.5 * float(step @ (lam * d * step - g))
        rho = (E - E_new) / (denom if abs(denom) > 1e-300 else 1e-300)
        if E_new < E:
            if abs(E - E_new) <= 1e-12 * (1.0 + abs(E)):
                theta, r, E = theta_new, r_new, E_new
                hist["loss"].append(float(np.mean(r * r)))
                hist["lam"].append(float(lam))
                hist["status"] = "converged(tol)"
                break
            theta, r, E = theta_new, r_new, E_new
            lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3) if rho > 0 else lam * 2.0
        else:
            lam *= 2.0
        lam = float(np.clip(lam, 1e-12, 1e12))
    if len(hist["loss"]) == 0:
        hist["loss"].append(float(np.mean(r * r)))
    return {"theta": theta, "template": template, "hist": hist,
            "train_mse": float(np.mean(lm_residuals(theta, template, X, Y, activation) ** 2))}


def lm_predict_proba(theta, template, X, activation="tanh"):
    out, _ = lm_forward(lm_unpack(theta, template), np.asarray(X, dtype=float), activation)
    e = out - out.max(axis=1, keepdims=True)
    P = np.exp(e)
    P /= P.sum(axis=1, keepdims=True)
    return P


class LMEstimator(ClassifierMixin, BaseEstimator):
    """Klasifikasi kelas tunggal (target one-hot + MSE), sama dengan notebook 11."""

    def __init__(self, hidden=(32,), activation="tanh", iters=40, seed=42,
                 lam0=1e-3, max_train_s=60.0):
        self.hidden = hidden
        self.activation = activation
        self.iters = iters
        self.seed = seed
        self.lam0 = lam0
        self.max_train_s = max_train_s

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_ = np.unique(np.asarray(y))
        idx = np.searchsorted(self.classes_, np.asarray(y))
        Y = np.eye(len(self.classes_))[idx]
        res = lm_fit(X, Y, X.shape[1], list(self.hidden), len(self.classes_),
                     activation=self.activation, iters=self.iters, seed=self.seed,
                     lam0=self.lam0, max_train_s=self.max_train_s)
        self.theta_ = res["theta"]
        self.template_ = res["template"]
        self.hist_ = res["hist"]
        self.train_mse_ = res["train_mse"]
        return self

    def predict_proba(self, X):
        return lm_predict_proba(self.theta_, self.template_, X, self.activation)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class LMSensorID(BaseEstimator):
    """Sensor mana yang rusak: satu LM pada target multi-hot 4 sensor (ambang 0,5), pola §12."""

    def __init__(self, hidden=(32,), activation="tanh", iters=40, seed=42, max_train_s=150.0):
        self.hidden = hidden
        self.activation = activation
        self.iters = iters
        self.seed = seed
        self.max_train_s = max_train_s

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        res = lm_fit(X, Y, X.shape[1], list(self.hidden), Y.shape[1], activation=self.activation,
                     iters=self.iters, seed=self.seed, max_train_s=self.max_train_s)
        self.theta_ = res["theta"]
        self.template_ = res["template"]
        self.hist_ = res["hist"]
        return self

    def decision_function(self, X):
        out, _ = lm_forward(lm_unpack(self.theta_, self.template_), np.asarray(X, dtype=float),
                            self.activation)
        return out

    def predict(self, X):
        return (self.decision_function(X) > 0.5).astype(int)


def make_lm_pipeline(hidden, activation, iters, max_train_s, seed=RANDOM_SEED):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lm", LMEstimator(hidden=tuple(hidden), activation=activation, iters=iters,
                           seed=seed, max_train_s=max_train_s)),
    ])


def make_sensor_pipeline(hidden, activation, iters=LM_ITERS_REFIT, max_train_s=LM_REFIT_BUDGET_S):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("lm", LMSensorID(hidden=tuple(hidden), activation=activation, iters=iters,
                          seed=RANDOM_SEED, max_train_s=max_train_s)),
    ])


# --------------------------------------------------------------------------
# §10b pencarian arsitektur (inner grouped CV, sama dengan notebook 11)
# --------------------------------------------------------------------------
def _inner_combo_score(hidden, activation, Ftr, ytr, gtr, iters, budget_s, seed):
    try:
        skf = StratifiedGroupKFold(n_splits=LM_INNER_CV, shuffle=True, random_state=seed)
        fs = []
        for itr, ite in skf.split(Ftr, ytr, groups=gtr):
            if len(np.unique(ytr[itr])) < 2:
                continue
            pipe = make_lm_pipeline(hidden, activation, iters, budget_s)
            pipe.fit(Ftr[itr], ytr[itr])
            fs.append(f1_score(ytr[ite], pipe.predict(Ftr[ite]), average="macro", zero_division=0))
        return tuple(hidden), activation, (float(np.mean(fs)) if fs else -1.0)
    except Exception:
        return tuple(hidden), activation, -1.0


def inner_search_best(Ftr, ytr, gtr, seed=RANDOM_SEED, n_jobs=-1,
                      hidden_grid=None, act_grid=None):
    hidden_grid = hidden_grid or HIDDEN_GRID
    act_grid = act_grid or ACT_GRID
    results = Parallel(n_jobs=n_jobs)(
        delayed(_inner_combo_score)(h, a, Ftr, ytr, gtr, LM_ITERS_SEARCH, LM_SEARCH_BUDGET_S, seed)
        for h in hidden_grid for a in act_grid)
    results.sort(key=lambda r: r[2], reverse=True)
    return results


# --------------------------------------------------------------------------
# Bantu demo
# --------------------------------------------------------------------------
FAULT_PILIHAN = {
    "drift": [(simulate_drift_fault, _D)],
    "spike": [(simulate_spike_fault, _S)],
    "bias": [(simulate_bias_fault, _B)],
    "hardware": [(simulate_hardware_fault, _H)],
}


def inject_demo(X_win, fault_names, sensor_idx, seed=0):
    """Beri fault ke satu window (win, 4). fault_names: list, mis. ["spike"] atau ["drift", "bias"]."""
    faults = []
    for f in fault_names:
        faults += FAULT_PILIHAN[f]
    return inject_faults_multisensor(np.asarray(X_win, dtype=float), faults, sensor_idx, seed=seed)


# --------------------------------------------------------------------------
# Model portabel: tidak tergantung versi scikit-learn
# --------------------------------------------------------------------------
class SavedLM:
    """Model ANN-LM tersimpan dalam bentuk numpy murni.

    Pipeline sklearn (SimpleImputer -> StandardScaler -> LM) yang di-pickle
    hanya bisa dibuka dengan versi scikit-learn yang sama persis (Kaggle 1.6.1
    vs laptop 1.9.1 gagal: atribut internal SimpleImputer berubah). Kelas ini
    menyimpan angka yang dipakai saat prediksi saja: nilai pengisi NaN (median
    data latih), mean dan skala StandardScaler, bobot LM, dan aktivasi.
    Hasil prediksinya identik dengan pipeline asal.

    kind = "kelas"  -> predict_proba / predict (softmax atas keluaran linear)
    kind = "sensor" -> decision_function / predict (ambang 0,5 per sensor)
    """

    def __init__(self, kind, fill, mean, scale, theta, shapes, activation, classes=None):
        self.kind = kind
        self.fill = np.asarray(fill, dtype=float)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.shapes = [tuple(map(int, s)) for s in shapes]
        self.activation = activation
        self.classes = None if classes is None else np.asarray(classes)

    @classmethod
    def from_pipeline(cls, pipe, kind):
        imp, sc, lm = pipe.named_steps["imp"], pipe.named_steps["sc"], pipe.named_steps["lm"]
        return cls(kind=kind, fill=imp.statistics_, mean=sc.mean_, scale=sc.scale_,
                   theta=lm.theta_, shapes=[W.shape for W, _ in lm.template_],
                   activation=lm.activation,
                   classes=getattr(lm, "classes_", None))

    def _template(self):
        return [[np.zeros(s), np.zeros(s[1])] for s in self.shapes]

    def _prep(self, X):
        X = np.array(X, dtype=float, copy=True)
        if X.ndim == 1:
            X = X[None, :]
        nan = np.isnan(X)
        if nan.any():
            X[nan] = np.broadcast_to(self.fill, X.shape)[nan]
        return (X - self.mean) / self.scale

    def decision_function(self, X):
        out, _ = lm_forward(lm_unpack(self.theta, self._template()), self._prep(X), self.activation)
        return out

    def predict_proba(self, X):
        out = self.decision_function(X)
        e = out - out.max(axis=1, keepdims=True)
        P = np.exp(e)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, X):
        if self.kind == "sensor":
            return (self.decision_function(X) > 0.5).astype(int)
        return self.classes[self.predict_proba(X).argmax(axis=1)]


# --------------------------------------------------------------------------
# Demo: injeksi PERSIS seperti saat training
# --------------------------------------------------------------------------
_KOMPONEN = {"malfunc": "hardware"}


def condition_for(fault_names):
    """{"spike", "hardware"} -> "spike+malfunc" (nama kondisi di SCENARIOS)."""
    want = set(fault_names)
    for name in SCENARIOS:
        if name == "faulty":
            continue
        parts = {_KOMPONEN.get(p, p) for p in name.split("+")}
        if parts == want:
            return name
    raise KeyError(f"Kombinasi fault {sorted(want)} tidak ada. Pilihan: drift, spike, bias, hardware")


def inject_like_training(X_full, fault_names, sensor_idx, seed=0):
    """Beri fault ke SELURUH deret (seperti segment_and_label), urutan komponen sama
    dengan definisi kondisi di SCENARIOS. Window demo dipotong SESUDAH ini, jadi
    bentuk fault (mis. hardware menyalin nilai dari hari lain, drift hanya di
    episode 3 hari) sama dengan data latih."""
    cond = condition_for(fault_names)
    Y, M = inject_faults_multisensor(np.asarray(X_full, dtype=float), SCENARIOS[cond], sensor_idx, seed=seed)
    return cond, Y, M


def drift_episode_days(n_samples):
    ep = int(round(DRIFT_EPISODE_DAYS * 86400.0 / DRIFT_SAMPLE_SECONDS))
    return [(s * SAMPLING_SECONDS / 86400, (s + ep) * SAMPLING_SECONDS / 86400)
            for s in (np.linspace(0.10, 0.70, DRIFT_N_EPISODES) * n_samples).astype(int)]
