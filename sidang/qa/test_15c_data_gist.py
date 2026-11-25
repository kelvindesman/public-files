"""Checks on notebook 15c: the notebook-15 program run on the 10,000-point gist data.

Run after the Kaggle output is downloaded into hasil15c/ (exports/15c_*).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
HASIL = ROOT / "hasil15c"
EXP = HASIL / "exports"
GIST_SHA256 = "23d39c08d38b4028b548dd13c3636f0432586637a2f77c75b9f5666d08452871"

pytestmark = pytest.mark.skipif(not EXP.exists(), reason="download the 15c Kaggle output first")


# source: Bu Luh chat 22/11/26 23.17: "Data yang berisi 10.000 titik adalah gistfile1.txt (gist 814fc8bc...),
# yaitu data yang dipakai di final-2_3_sept_2025"; final-2 cell 1 loads that gist URL
def test_data_yang_dipakai_adalah_gist_10000_titik():
    nb = json.loads((ROOT / "15c_Data_Hipotetik_Gist_10000_ANN_LM.ipynb").read_text())
    src = "".join("".join(c["source"]) for c in nb["cells"])
    assert "814fc8bcde1d9768ca0cffa0f248e567" in src and GIST_SHA256 in src
    if (HASIL / "gistfile1.txt").exists():
        assert hashlib.sha256((HASIL / "gistfile1.txt").read_bytes()).hexdigest() == GIST_SHA256
    for s in ["S0_normal", "S1", "S2", "S3", "S4", "S5"]:
        d = pd.read_csv(EXP / f"15c_data_berlabel_{s}.csv.gz")
        assert len(d) % 10_000 == 0 and d.groupby("kondisi").size().eq(10_000).all()


# source: Bu Luh chat 22/11/26 23.01: "run data-hipotetik-berlabel-ann-lm ... pakai data 10.000"
# (program of notebook 15: 16 fault conditions S1-S5, ANN-LM 1 and 2 layer, combined model)
def test_program_nb15_lengkap_pada_data_gist():
    hasil = pd.read_csv(EXP / "15c_hasil_ann_lm.csv")
    assert hasil["kondisi"].nunique() == 16
    assert set(hasil["skenario"]) == {"S1", "S2", "S3", "S4", "S5"}
    assert set(hasil["arsitektur"]) == {"1 layer (6)", "2 layer (6,3)"}
    assert hasil["F1"].between(0, 1).all()
    daftar = pd.read_csv(EXP / "15c_daftar_model.csv")
    assert len(daftar) == 33 and daftar["selisih_maks_prob_saat_muat_ulang"].max() < 1e-9
    card = json.loads((EXP / "15c_model_card.json").read_text())
    assert card["data_sumber"]["sha256"] == GIST_SHA256 and card["data_sumber"]["baris"] == 10_000
    assert card["lm_iterasi_maks"] == 60, "full run expected, not SMOKE"
