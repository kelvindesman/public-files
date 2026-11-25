"""Contoh integrasi backend: satu pembacaan sensor masuk -> prob_fault + alarm (JSON).

Model menilai SATU deret sensor. Untuk sampel terbaru model butuh 10 nilai
sebelumnya (lag1..lag10), jadi backend menyimpan riwayat pendek per sensor.

    from contoh_backend import DetektorSensor
    det = DetektorSensor("model_ann_lm_4sensor.npz")
    hasil = det.proses("kelembaban1", 51.27)      # dipanggil tiap pembacaan baru (30 detik)
    # {"sensor": "kelembaban1", "nilai": 51.27, "prob_fault": 0.0412, "alarm": false, "siap": true}

Nilai hilang (sensor tidak mengirim data) dikirim sebagai None.
Jalankan langsung untuk demo:  python contoh_backend.py data_hipotetik_berlabel_10000.csv
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from muat_model import muat_model, deteksi  # noqa: E402

PANJANG_RIWAYAT = 64   # cukup > lag_k; hanya untuk mengisi nilai hilang ke depan (ffill)


class DetektorSensor:
    def __init__(self, path_model):
        self.m = muat_model(path_model)
        self.lag_k = int(self.m["lag_k"])
        self.ambang = float(self.m["ambang"])
        self.riwayat = {}

    def proses(self, sensor_id, nilai):
        buf = self.riwayat.setdefault(sensor_id, deque(maxlen=PANJANG_RIWAYAT))
        buf.append(np.nan if nilai is None else float(nilai))
        prob, alarm = deteksi(self.m, np.array(buf))
        return {"sensor": sensor_id,
                "nilai": None if nilai is None else float(nilai),
                "prob_fault": round(float(prob[-1]), 4),
                "alarm": bool(alarm[-1]),
                "siap": len(buf) > self.lag_k}   # False = riwayat belum 11 nilai, abaikan hasilnya


if __name__ == "__main__":
    import pandas as pd
    folder = Path(__file__).resolve().parent
    csv = sys.argv[1] if len(sys.argv) > 1 else folder / "data_hipotetik_berlabel_10000.csv"
    data = pd.read_csv(csv)
    det = DetektorSensor(folder / "model_ann_lm_4sensor.npz")
    sensor = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]
    n_alarm = {s: 0 for s in sensor}
    tampil = 0
    for _, baris in data.iterrows():                     # simulasi data masuk satu per satu
        for s in sensor:
            v = baris[s]
            h = det.proses(s, None if pd.isna(v) else v)
            n_alarm[s] += int(h["alarm"] and h["siap"])
            if h["alarm"] and h["siap"] and tampil < 5:
                h["timestamp"] = baris["timestamp"]
                print(json.dumps(h))
                tampil += 1
    print(json.dumps({"baris": len(data), "jumlah_alarm": n_alarm}))
