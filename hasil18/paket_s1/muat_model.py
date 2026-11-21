# Muat dan pakai model ANN-LM notebook 15 (hanya numpy + pandas).
#
#     from muat_model_15 import muat_model, deteksi
#     m = muat_model("15_model_gabungan_rata2.npz")
#     prob, alarm = deteksi(m, nilai_sensor)
#
# nilai_sensor: array nilai SATU sensor (atau rata-rata 4 sensor untuk model
# "rata2"), urut waktu, interval +-30 detik, NaN boleh (data hilang).
# prob[t] = probabilitas gangguan pada sampel t; alarm[t] = prob[t] > ambang.
# 10 sampel pertama belum punya lag lengkap (prob tetap dihitung, abaikan saja).
import numpy as np
import pandas as pd


def muat_model(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fitur_lag(x, lag_k=10):
    x = np.asarray(x, dtype=float)
    na = np.isnan(x)
    xf = pd.Series(x).ffill().bfill().to_numpy()
    cols = [np.roll(xf, k) for k in range(1, lag_k + 1)]
    cols.append(xf - np.roll(xf, 1))
    cols.append(na.astype(float))
    cols.append(pd.Series(na.astype(float)).rolling(lag_k, min_periods=1).sum().shift(1).fillna(0).to_numpy())
    return np.column_stack(cols)


def prob_fault(m, F):
    a = (np.asarray(F, dtype=float) - m["mean"]) / m["scale"]
    shapes = [tuple(int(v) for v in s) for s in m["shapes"]]
    theta = m["theta"]
    i = 0                                          # urutan theta = lm_unpack: W1, b1, W2, b2, ...
    for li, (r, c) in enumerate(shapes):
        W = theta[i:i + r * c].reshape(r, c); i += r * c
        b = theta[i:i + c]; i += c
        a = a @ W + b
        if li < len(shapes) - 1:
            a = np.tanh(a)
    e = a - a.max(axis=1, keepdims=True)
    P = np.exp(e); P /= P.sum(axis=1, keepdims=True)
    return P[:, 1]


def deteksi(m, nilai_sensor, ambang=None):
    p = prob_fault(m, fitur_lag(nilai_sensor, int(m["lag_k"])))
    thr = float(m["ambang"]) if ambang is None else float(ambang)
    return p, (p > thr).astype(int)
