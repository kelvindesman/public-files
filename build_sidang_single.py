"""Merge the three defence notebooks into one, with links to the finished kernels.

Run:  venv/bin/python build_sidang_single.py
Output: sidang/00_Sidang_Deteksi_Fault.ipynb

Content is copied from the already-tested notebooks; only the headings are
renumbered and a link table is prepended. Nothing is recomputed.
"""
import json
import re
from pathlib import Path

OUT = Path("sidang")
SOURCES = [
    ("Bagian I — Peta Program", "00_Peta_Program.ipynb"),
    ("Bagian II — Alur Sistem", "01_Alur_Sistem.ipynb"),
    ("Bagian III — Hasil", "02_Hasil.ipynb"),
]
TARGET = OUT / "00_Sidang_Deteksi_Fault.ipynb"

INTRO = """# Sidang — Deteksi Fault 4 Sensor Kelembaban

**Satu notebook, tiga bagian:** peta program, alur data nyata, dan hasil.

Semua angka **dibaca dari file hasil** yang sudah diekspor notebook penelitian.
Notebook ini tidak melatih ulang model; ia hanya menampilkan dan menjelaskan.
Waktu jalan ± 10 detik.

**Cara pakai saat sidang:** buka bagian yang ditanyakan penguji. Tiap bagian
punya nomor sendiri di daftar isi di bawah.
"""

LINKS = """## Tautan notebook penelitian (semua sudah selesai dijalankan)

Kernel di bawah adalah sumber angka. Statusnya **Private**; buka sambil login
akun `kelvindsmn`. Notebook ini meringkasnya, bukan menggantikannya.

| # | Isi | Kernel |
|---|---|---|
| 01 | Eksperimen dasar JSD-Fuzzy vs EDM-Fuzzy | [JSD_Fuzzy_ANN_Paper](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-ann-paper) |
| 02 | Arsitektur broker 4 kanal | [JSD_Fuzzy_Broker_Architecture_v2](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-broker-architecture-v2) |
| 03 | Tangga skenario S1–S5 | [JSD_Fuzzy_Scenario_Ladder](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-scenario-ladder) |
| 04 | Two-stage: fault dulu, sensor kemudian | [JSD_Fuzzy_TwoStage_SensorID](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-twostage-sensorid) |
| 05 | Jenis fault per sensor fisik | [JSD_Fuzzy_FaultType_per_Sensor](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-faulttype-per-sensor) |
| 06 | ANN vs XGBoost per skenario | [JSD_XGB_perScenario](https://www.kaggle.com/code/kelvindsmn/jsd-xgb-perscenario) |
| 07 | Pipeline lengkap 5 skenario (CV berkelompok) | [JSD_Fuzzy_CV_5Skenario](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-cv-5skenario) |
| 08 | Perbandingan paper Q3 | [JSD_Fuzzy_vs_EDM_Fuzzy_Paper_Q3](https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-vs-edm-fuzzy-paper-q3) |
| 09 | Empat algoritma pelatih ANN | [ANN_LM_vs_Bayesian_vs_GA](https://www.kaggle.com/code/kelvindsmn/ann-lm-vs-bayesian-vs-ga-paper-pmdc) |
| 10 | ANN-LM asli vs JSD | [JSD_vs_EDM_ANN_LM_Asli](https://www.kaggle.com/code/kelvindsmn/jsd-vs-edm-dengan-ann-lm-asli-paper-pmdc) |
| 11 | Hasil lengkap + statistik + sensor-ID | [JSD_EDM_ANN_LM_Lengkap_Q3](https://www.kaggle.com/code/kelvindsmn/jsd-edm-ann-lm-lengkap-q3) |
| — | Ilustrasi data akuisisi & fault | [Ilustrasi_Data_Akuisisi_dan_Fault](https://www.kaggle.com/code/kelvindsmn/ilustrasi-data-akuisisi-dan-fault) |

Pemetaan rinci (bagian → sel → kernel) ada di `sidang/MAPPING.md`.
"""

TOC = """### Daftar isi

**I — Peta Program**
1. Alur program dalam satu gambar
2. Tiap tahap dalam bahasa sehari-hari
3. Enam angka yang paling sering ditanya
4. Peta notebook: sidang ↔ penelitian
5. Glosarium 10 istilah
6. Kalau penguji bertanya, buka ini

**II — Alur Sistem**
7. Data mentah: empat kanal, 30 detik
8. Broker: satukan kanal, jangan rata-ratakan
9. Injeksi fault: menempelkan kerusakan yang ada labelnya
10. Segmentasi: satu sinyal dipotong jadi banyak jendela
11. Dari jendela ke 60 angka
12. Model: 60 angka masuk, kelas fault keluar
13. Uji: kenapa harus CV berkelompok
14. Peta balik: tahap di sini → notebook asli

**III — Hasil**
15. Performa utama: EDM-Fuzzy vs JSD-Fuzzy
16. Sensor mana yang rusak?
17. Jenis fault apa, di sensor mana?
18. Two-stage: klasifikasi dulu, sensor kemudian
19. Empat algoritma pelatih ANN
20. ANN vs XGBoost: batasnya di mana?
21. Satu kalimat statistik (hanya satu)
22. Batasan yang harus disebut
23. Sumber setiap blok
"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.rstrip()}


def load_cells(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb["cells"]


def notebook(cells):
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
        if c["cell_type"] == "code":
            c["execution_count"] = None
            c["outputs"] = []
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    cells = [md(INTRO), md(LINKS), md(TOC)]
    seen_code = set()
    number = 0

    for title, name in SOURCES:
        src = load_cells(OUT / name)
        cells.append(md(f"---\n\n# {title}"))
        for c in src[1:]:
            body = "".join(c["source"])
            if c["cell_type"] == "code":
                if body in seen_code:
                    continue
                seen_code.add(body)
            elif body.startswith("## "):
                number += 1
                body = re.sub(r"^## \d+ — ", f"## {number} — ", body, count=1)
                if not re.match(rf"^## {number} — ", body):
                    body = re.sub(r"^## ", f"## {number} — ", body, count=1)
            cells.append({"cell_type": c["cell_type"], "metadata": {}, "source": body})

    TARGET.write_text(json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", TARGET, "|", len(cells), "cells | headings", number)


if __name__ == "__main__":
    main()
