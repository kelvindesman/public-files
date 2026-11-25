"""Checks on the S1 student package produced by notebook 18 (hasil18/paket_s1).

Run after: PAKET_DIR=hasil18/paket_s1 MODEL_DIR=<NB15 model folder> \
    python sidang/qa/jalankan_notebook.py 18_Paket_S1_Data_Hipotetik_10000.ipynb --keluar hasil18
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAKET = ROOT / "hasil18" / "paket_s1"
SENSORS = ["kelembaban1", "kelembaban2", "kelembaban3", "kelembaban4"]

pytestmark = pytest.mark.skipif(not PAKET.exists(), reason="run notebook 18 first")


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(PAKET / "data_hipotetik_berlabel_10000.csv")


# source: Kelvin chat 2026-11-21, pasted checklist "untuk mhs s1: 1. data hipotetik berlabel (10.000 data)"
def test_data_tepat_10000_baris_berlabel_semua_jenis_fault(data):
    assert len(data) == 10_000
    for s in range(1, 5):
        assert {f"label_s{s}", f"jenis_s{s}", f"kelembaban{s}_asli"} <= set(data.columns)
    jenis = set()
    for s in range(1, 5):
        for j in data[f"jenis_s{s}"].unique():
            jenis |= set(j.split("+"))
    assert {"normal", "drift", "spike", "bias", "hardware"} <= jenis
    assert 0 < data["label_fault"].sum() < len(data)


# source: NB14 simulator definition (build_14 NEW_SIM): every changed value carries a fault label
def test_nilai_berubah_selalu_berlabel(data):
    for s, n in enumerate(SENSORS, 1):
        y = data[n].to_numpy(float)
        x = data[f"{n}_asli"].to_numpy(float)
        berubah = ~np.isclose(y, x)                       # NaN counts as changed
        assert not (berubah & (data[f"label_s{s}"].to_numpy() == 0)).any(), n
        nan = np.isnan(y)
        assert data.loc[nan, f"jenis_s{s}"].str.contains("hardware").all(), n


# source: Kelvin chat 2026-11-21, pasted checklist "untuk mhs s1: 2. ... informasi preprocessing ...
# normalisasi data dan fitur inputnya. contoh penggunaan model beserta output prediksinya"
def test_model_card_memuat_preprocessing_dan_contoh_backend_jalan(data):
    card = json.loads((PAKET / "model_card.json").read_text())
    pre = card["paket_s1"]["preprocessing"]
    assert len(pre["fitur"]) == len(pre["mean"]) == len(pre["scale"]) == 13
    assert "ACS" not in card["status_publikasi"]           # unverified claim, journal gap 3
    sys.path.insert(0, str(PAKET))
    from contoh_backend import DetektorSensor
    from muat_model import deteksi
    det = DetektorSensor(PAKET / "model_ann_lm_4sensor.npz")
    x = data["kelembaban4"].to_numpy(float)[:300]
    stream = [det.proses("s4", None if np.isnan(v) else v) for v in x]
    p, a = deteksi(det.m, x)
    assert all(h["siap"] for h in stream[10:]) and not stream[9]["siap"]
    assert max(abs(h["prob_fault"] - p[i]) for i, h in enumerate(stream) if i >= 10) <= 5e-5 + 1e-12
    assert [int(h["alarm"]) for h in stream[10:]] == list(a[10:])
    pred = pd.read_csv(PAKET / "prediksi_10000.csv")
    assert len(pred) == 10_000 and {"prob_s1", "alarm_s4"} <= set(pred.columns)
