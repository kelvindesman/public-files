#!/usr/bin/env python3
"""Build 16_Simpan_Model_Final.ipynb (dijalankan sekali di Kaggle).

Notebook ini menulis `deteksi_fault_core.py` dan `simpan_model.py` dari repo ke
folder kerja Kaggle, lalu menjalankan `simpan_model.py`. Keluarannya (folder
`models/` + `models.zip`) dipakai notebook demo sidang supaya demo tidak perlu
training ulang.
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 700   # 200 / 700 / 1000
OUT = HERE / ("16_Simpan_Model_Final.ipynb" if WIN == 700 else f"16_Simpan_Model_Final_N{WIN}.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
core_src = (HERE / "deteksi_fault_core.py").read_text()
train_src = (HERE / "simpan_model.py").read_text()

cells = [
    md(r"""
# 16 — Simpan model final ANN-LM (untuk demo tanpa training ulang)

Run penelitian notebook 11 makan **10,2 jam** karena mengulang ekstraksi
entropi dan grid search di tiap lipatan CV. Untuk demo cukup **satu** model
per skenario per metode yang sudah jadi, lalu disimpan ke file.

| | |
|---|---|
| Window | N = 700 (5 menit/sampel = 2,4 hari) — satu-satunya window dengan selisih F1 signifikan (p = 0,027) di notebook 11 |
| Fitur | entropi multiskala EDM-Fuzzy dan JSD-Fuzzy, 4 sensor × 15 skala = 60 fitur |
| Classifier | ANN-LM asli (Levenberg–Marquardt) — kode sama persis dengan notebook 11 |
| Arsitektur | dicari ulang dengan grid yang sama (8 hidden × 2 aktivasi, inner grouped CV 2 lipatan) |
| Model akhir | dilatih di semua window skenario, lalu disimpan `joblib` |
| Keluaran | `models/model_N700_S{1..5}_{EDM,JSD}.joblib`, `models/fitur_N700.npz`, `models/meta.json`, `models/deteksi_fault_core.py` |

**Catatan jujur.** Angka performa resmi tetap dari CV notebook 11 v3. Model di
sini dilatih di semua data, jadi akurasi pada data latih yang dicetak di bawah
bukan ukuran generalisasi.
"""),
    code("%%writefile deteksi_fault_core.py\n" + core_src),
    code("%%writefile simpan_model.py\n" + train_src),
    code(f"""import os, subprocess, sys
os.environ["GIT_COMMIT"] = "{commit}"
os.environ.setdefault("MODEL_DIR", "models")
os.environ["WIN"] = "{WIN}"
r = subprocess.run([sys.executable, "simpan_model.py"], text=True)
assert r.returncode == 0, "simpan_model.py gagal"
"""),
    code(r"""import json, os, shutil, pandas as pd
shutil.make_archive("models", "zip", "models")
meta = json.load(open(f"models/meta_N{os.environ['WIN']}.json"))
print("dibuat", meta["dibuat"], "| commit", meta["commit"], "| total", meta["total_seconds"], "s")
pd.DataFrame(meta["models"])
"""),
]

if WIN != 700:
    hari = {200: "0,69", 1000: "3,47"}[WIN]
    teks = "".join(cells[0]["source"])
    teks = teks.replace("# 16 — Simpan model final ANN-LM", f"# 16 — Simpan model final ANN-LM, N = {WIN}")
    teks = teks.replace("| Window | N = 700 (5 menit/sampel = 2,4 hari) — satu-satunya window dengan selisih F1 signifikan (p = 0,027) di notebook 11 |",
                        f"| Window | N = {WIN} (5 menit/sampel = {hari} hari) — pilihan panjang jendela untuk notebook sidang interaktif |")
    teks = teks.replace("model_N700_", f"model_N{WIN}_").replace("fitur_N700", f"fitur_N{WIN}").replace("models/meta.json", f"models/meta_N{WIN}.json")
    cells[0]["source"] = teks.splitlines(keepends=True)

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                  "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print("ditulis:", OUT)
