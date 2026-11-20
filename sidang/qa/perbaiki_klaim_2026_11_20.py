#!/usr/bin/env python3
"""perbaiki_klaim_2026_11_20.py — koreksi sekali-jalan atas klaim TIDAK COCOK.

Sumber: audit `sidang/qa/01_ringkasan_klaim.md` (2026-11-20). Setiap penggantian
teks di bawah menyebut id ledger yang dikoreksi. Skrip berhenti kalau teks lama
tidak ditemukan tepat satu kali, jadi aman dijalankan ulang (yang sudah
diperbaiki dilewati).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = Path(__file__).with_name("01_klaim_bukti.csv")

GANTI = [
    # README ---------------------------------------------------------------
    ("README.md", ["R06-02", "R06-04"],
     "XGBoost + fitur JSD F1 0,443 (ANN-LM 0,394); tambah fitur waktu → **0,608**, terbaik di 5 skenario. Fitur EDM saja 0,254",
     "Angka skenario S5: XGBoost + fitur JSD F1 0,443 (MLP sklearn notebook 01: 0,394); tambah fitur waktu → **0,608**. JSD + waktu terbaik di 3 dari 5 skenario (S1, S2, S5), fitur waktu saja terbaik di S3 dan S4; rata-rata 5 skenario JSD + waktu 0,750. Fitur EDM saja (S5) 0,254"),
    ("README.md", ["R09-11"], "vs Adam (p ≤ 0,025)", "vs Adam (p ≤ 0,026)"),
    ("README.md", ["R10-01"], "**JSD-Fuzzy unggul di 10 dari 15 sel** (run v4)", "**JSD-Fuzzy unggul di 11 dari 15 sel** (run v4)"),
    ("README.md", ["R10-03"], "N=700 +0,056 · N=1000 +0,030 · pooled +0,031", "N=700 +0,055 · N=1000 +0,030 · pooled +0,031"),
    ("README.md", ["R13-05"], "bias episode AUC ≤ 0,74", "bias episode AUC ≤ 0,75"),
    ("README.md", ["R13-11"], "daripada EDM (0,76–0,82)", "daripada EDM (0,76–0,81)"),
    ("README.md", ["R15-H", "R15-I"],
     "Kaggle `data-hipotetik-berlabel-ann-lm` v1 COMPLETE, wall 30 mnt, 0 Traceback. Hasil di `hasil15/`",
     "Kaggle `data-hipotetik-berlabel-ann-lm` v1 selesai (log `hasil15/data-hipotetik-berlabel-ann-lm.log`: wall 30 mnt, 0 Traceback). Ekspor `15_*` ada di `hasil16/exports/`"),
    ("README.md", ["R-D3"], "versi lama 300–19.000× terlalu cepat", "versi lama 339–19.200× terlalu cepat"),
    ("README.md", ["R-60a"], "-> JSD unggul 10/15 sel, pooled p = 0,052", "-> JSD unggul 11/15 sel, pooled p = 0,052"),
    ("README.md", ["R-64"],
     "berbeda (p = 0,052 vs 0,076) karena `11` memakai protokol CV dan anggaran\n"
     "komputasi yang lebih besar. Keduanya harus ditulis apa adanya — batas\n"
     "signifikansi 0,05 tidak dilampaui dengan aman, jadi klaim tetap deskriptif.",
     "berbeda (p = 0,052 vs 0,076) walaupun konstanta protokolnya identik\n"
     "(LM_ITERS_SEARCH 6, LM_ITERS_REFIT 40, LM_INNER_CV 2, grid dan seed sama);\n"
     "yang beda hanya batas sesi Kaggle (10,5 vs 11,5 jam). F1 sama persis di 8\n"
     "dari 15 sel. Tujuh sel yang beda (N200 S2–S4, N700 S2/S4, N1000 S2/S4) adalah\n"
     "sel yang pelatihannya dihentikan batas waktu dinding `MAX_TRAIN_S`, sehingga\n"
     "hasilnya tidak deterministik antar-run. Jumlah sel menang sama (11/15).\n"
     "Keduanya ditulis apa adanya — batas signifikansi 0,05 tidak dilampaui\n"
     "dengan aman, jadi klaim tetap deskriptif."),
    ("README.md", ["R-96", "R-98a", "R-98b"],
     "\"Per-Sensor Cara A\") pernah melaporkan **F1 0,925** untuk identifikasi sensor.\n"
     "**Angka itu tidak boleh dipakai.** Injeksi fault-nya mengenai keempat sensor\n"
     "sekaligus, sehingga keempat kolom label identik (prevalensi 0,882 di semua\n"
     "sensor) dan ROC-AUC-nya 0,50 — setara tebak acak. Angka tinggi itu murni efek\n"
     "\"selalu menebak fault\".",
     "\"Per-Sensor Cara A\") pernah melaporkan F1 yang sangat tinggi untuk\n"
     "identifikasi sensor. **Angka itu tidak boleh dipakai** dan sengaja tidak\n"
     "dikutip di sini, karena notebook arsipnya tidak menyimpan output sehingga\n"
     "angkanya tidak bisa dibuktikan ulang. Injeksi fault-nya mengenai keempat\n"
     "sensor sekaligus, sehingga keempat kolom label identik: model cukup selalu\n"
     "menebak \"fault\" untuk mendapat F1 tinggi tanpa bisa membedakan sensor."),
    # Bahan_Sidang ----------------------------------------------------------
    ("Bahan_Sidang.md", ["B-09c", "B-09g"],
     "semuanya 16–21 Des 2025, berupa tiga blok macet: 45 jam, 37 jam, dan 12 jam\n"
     "(plus 49 sampel pendek).",
     "berupa tiga blok macet 45 jam, 37 jam, dan 12 jam (semuanya 16–21 Des 2025),\n"
     "plus 50 sampel pendek (49 di 16–21 Des dan 1 pada 8 Des 2025 pukul 14.06)."),
    ("Bahan_Sidang.md", ["B-29"],
     "kisi (32,32), (64,16), (64,), aktivasi relu/tanh",
     "kisi 8 arsitektur — (16,), (32,), (64,), (16,8), (32,8), (32,16), (32,32),\n"
     "(64,16) — aktivasi relu/tanh"),
    # MAPPING ---------------------------------------------------------------
    ("sidang/MAPPING.md", ["M-05", "M-10a"],
     "- Karena protokol/anggaran beda, **jumlah sel menang juga beda**: `10` 10/15,\n"
     "  `11` 11/15. Yang dipakai di sidang: **11/15** (notebook 11).",
     "- Protokol CV `10` dan `11` identik; yang beda hanya batas sesi Kaggle\n"
     "  (10,5 vs 11,5 jam). Jumlah sel menang **sama**: `10` 11/15, `11` 11/15.\n"
     "  p gabungan beda (0,052 vs 0,076) karena 7 sel S2/S4 dihentikan batas waktu\n"
     "  latih dinding. Yang dipakai di sidang: notebook 11."),
    # hasil11 -----------------------------------------------------------------
    ("hasil11/Ringkasan_untuk_laporan.md", ["H11-08", "H11-09", "H11-10"],
     "pasangan (45,7%, rata-rata −26%) · N=700 59/315 (18,7%, −28%) · N=1000\n"
     "52/315 (16,5%, −31%) → **EDM-Fuzzy lebih stabil pada 56 dari 63 sel\n"
     "(89%)**.",
     "pasangan (45,7%, rata-rata −19,1%) · N=700 59/315 (18,7%, −16,5%) · N=1000\n"
     "52/315 (16,5%, −16,4%) → **EDM-Fuzzy lebih stabil pada 56 dari 63 baris\n"
     "kelas (N × skenario × kelas, 89%)**."),
    ("hasil11/Ringkasan_untuk_laporan.md", ["H11-13b"], "N=1000 +0,030 · pooled +0,028", "N=1000 +0,029 · pooled +0,028"),
    ("hasil11/Ringkasan_untuk_laporan.md", ["H11-27", "H11-28"],
     "Campuran: JSD lebih terpisah di S1-N200/N700, S3-N700/N1000, S2-N1000,\n"
     "S5-N1000; EDM lebih terpisah di S2-N200, S4-N200, S5-N200, S1-N1000;\n"
     "sisanya sebanding.",
     "JSD lebih terpisah di 11 sel (N200 S3, S4; seluruh N700; N1000 S2–S5);\n"
     "EDM lebih terpisah di 0 sel; 4 sel campuran (N200 S1, S2, S5; N1000 S1)."),
    # hasil13 -----------------------------------------------------------------
    ("hasil13/13_draf_laporan.md", ["H13-25b"], "dari 0,81 ke 0,87 pada spike", "dari 0,81 ke 0,86 pada spike"),
    ("hasil13/13_draf_laporan.md", ["H13-24d"], "EDM relatif datar (0,76–0,82 dan", "EDM relatif datar (0,76–0,81 dan"),
    ("hasil13/13_draf_laporan.md", ["H13-12"], "(AUC 0,67–0,73 JSD", "(AUC 0,66–0,73 JSD"),
    # hasil14 -----------------------------------------------------------------
    ("hasil14/14_draf_laporan.md", ["H14-12", "H14-13"],
     "Jumlah sampel per\nkelas yang masuk classifier tetap sama dengan notebook 11 (RQ1: n = 1.700 /\n1.621 / 1.526).",
     "Jumlah sampel per\nkelas yang masuk classifier (RQ1): N = 200 tetap 1.700 seperti notebook 11,\n"
     "tetapi N = 700 dan 1.000 turun menjadi 1.621 / 1.526 (notebook 11: 1.675 / 1.651)."),
    ("hasil14/14_draf_laporan.md", ["H14-43a", "H14-43b", "H14-43c", "H14-43d"],
     "| Fit ANN-LM per fold, rata-rata EDM / JSD | 75,5 / 75,1 s | 93,2 / 98,0 s |",
     "| Fit ANN-LM per fold, rata-rata EDM / JSD | 93,2 / 98,0 s | 75,5 / 75,1 s |"),
    ("hasil14/14_draf_laporan.md", ["H14-44a", "H14-44b", "H14-44c", "H14-44d"],
     "| Inferensi per window, rata-rata EDM / JSD | 0,089 / 0,088 ms | 0,097 / 0,087 ms |",
     "| Inferensi per window, rata-rata EDM / JSD | 0,097 / 0,087 ms | 0,089 / 0,088 ms |"),
    ("hasil14/14_draf_laporan.md", ["H14-45d"], "1,7–6 kali lebih lama", "1,7–6,8 kali lebih lama"),
    ("hasil14/14_draf_laporan.md", ["H14-45e"],
     "Total run lebih\nsingkat karena identifikasi sensor lebih cepat. Fit ANN-LM per fold lebih\n"
     "lama, dan beberapa fold berhenti karena batas waktu latih (`MAX_TRAIN_S`).",
     "Total run lebih\nsingkat karena identifikasi sensor dan fit ANN-LM per fold sama-sama lebih\n"
     "cepat (fit 75 s vs 93–98 s per fold), walau beberapa fold tetap berhenti karena\n"
     "batas waktu latih (`MAX_TRAIN_S`)."),
    ("hasil14/14_draf_laporan.md", ["H14-30"], "menjadi +0,077 (p = 0,0006", "menjadi +0,076 (p = 0,0006"),
    # PANDUAN_DEMO ------------------------------------------------------------
    ("sidang/PANDUAN_DEMO.md", ["P-03", "P-04"],
     "selesai (di Kaggle terukur sekitar 2,5 menit termasuk menyalakan mesin;\n"
     "   di laptop 64–90 detik).",
     "selesai (di Kaggle waktu kernel sekitar 2,1 menit menurut log\n"
     "   `hasil16/demo-kaggle-v2.log`, belum termasuk menyalakan mesin; di laptop\n"
     "   63 detik, 21 detik tanpa Langkah 7 — bukti `sidang/bukti_demo/`)."),
    ("sidang/PANDUAN_DEMO.md", ["P-05"],
     "Di run terakhir kedua metode benar 80 %.",
     "Di run uji 2026-11-20 kedua metode benar 80 % (24 dari 30)."),
    ("sidang/PANDUAN_DEMO.md", ["P-06"], "5 dari 8 tebakan benar (S1-EDM", "6 dari 8 tebakan benar (S1-EDM"),
]

# id -> (klaim baru, nilai_klaim baru, status, catatan tambahan). None = biarkan.
LEDGER_BARU = {
    "R06-02": ("(MLP sklearn notebook 01: 0,394)", "0,394", "COCOK", "Label diperbaiki 2026-11-20."),
    "R06-04": ("JSD + waktu terbaik di 3 dari 5 skenario", "3", None, "Diperbaiki 2026-11-20."),
    "R09-11": ("vs Adam (p ≤ 0,026)", "0,026", None, "Batas dinaikkan ke 0,026 (p LM-Adam 0,0252)."),
    "R10-01": ("**JSD-Fuzzy unggul di 11 dari 15 sel**", "11", None, "Diperbaiki 2026-11-20."),
    "R10-03": ("N=700 +0,055", "0,055", None, None),
    "R13-05": ("bias episode AUC ≤ 0,75", "0,75", None, None),
    "R13-11": ("daripada EDM (0,76–0,81)", "0,81", None, None),
    "R15-H": ("v1 selesai (log hasil15: wall 30 mnt, 0 Traceback)", "v1 selesai", "COCOK",
              "Klaim kini merujuk log lokal hasil15 (v1), bukan status Kaggle saat ini."),
    "R15-I": ("Ekspor `15_*` ada di `hasil16/exports/`", "hasil16/exports/", "COCOK", "Rujukan diperbaiki 2026-11-20."),
    "R-D3": ("versi lama 339–19.200× terlalu cepat", "339–19.200", "COCOK", "Rentang ditulis persis 2026-11-20."),
    "R-60a": ("-> JSD unggul 11/15 sel", "11", None, None),
    "R-64": ("walaupun konstanta protokolnya identik", "0", None, "Klaim dibalik 2026-11-20: protokol identik."),
    "R-96": ("(dihapus) F1 arsip label kembar", "-", "DIHAPUS", "Angka dihapus dari README 2026-11-20 karena tidak ada output arsip."),
    "R-98a": ("(dihapus) prevalensi arsip", "-", "DIHAPUS", "Angka dihapus dari README 2026-11-20."),
    "R-98b": ("(dihapus) ROC-AUC arsip", "-", "DIHAPUS", "Angka dihapus dari README 2026-11-20."),
    "B-09c": ("1 pada 8 Des 2025 pukul 14.06", "8", None, None),
    "B-09g": ("plus 50 sampel pendek", "50", None, None),
    "B-29": ("kisi 8 arsitektur", "8", None, None),
    "M-05": ("Protokol CV `10` dan `11` identik", "0", None, "Klaim dibalik 2026-11-20."),
    "M-10a": ("`10` 11/15, `11` 11/15", "11", None, None),
    "H11-08": ("rata-rata −19,1%", "-19,1", None, None),
    "H11-09": ("N=700 (−16,5%)", "-16,5", None, None),
    "H11-10": ("N=1000 (−16,4%)", "-16,4", None, None),
    "H11-13b": ("N=1000 +0,029", "0,029", None, None),
    "H11-27": ("EDM lebih terpisah di 0 sel", "0", None, None),
    "H11-28": ("JSD lebih terpisah di 11 sel", "11", None, None),
    "H13-25b": ("dari 0,81 ke 0,86 pada spike", "0,86", None, None),
    "H13-24d": ("EDM relatif datar (0,76–0,81", "0,81", None, None),
    "H13-12": ("(AUC 0,66–0,73 JSD", "0,66", None, None),
    "H14-12": ("notebook 11: 1.675", "1675", None, None),
    "H14-13": ("notebook 11: 1.651", "1651", None, None),
    "H14-43a": ("NB11 fit EDM 93,2 s", "93,2", None, "Kolom ditukar balik 2026-11-20."),
    "H14-43b": ("NB11 fit JSD 98,0 s", "98,0", None, None),
    "H14-43c": ("NB14 fit EDM 75,5 s", "75,5", None, None),
    "H14-43d": ("NB14 fit JSD 75,1 s", "75,1", None, None),
    "H14-44a": ("NB11 inferensi EDM 0,097 ms", "0,097", None, None),
    "H14-44b": ("NB11 inferensi JSD 0,087 ms", "0,087", None, None),
    "H14-44c": ("NB14 inferensi EDM 0,089 ms", "0,089", None, None),
    "H14-44d": ("NB14 inferensi JSD 0,088 ms", "0,088", None, None),
    "H14-45d": ("1,7–6,8 kali lebih lama", "6,8", None, None),
    "H14-45e": ("fit ANN-LM per fold sama-sama lebih cepat", "lebih cepat", "COCOK",
                "Kesimpulan dibalik 2026-11-20 mengikuti CSV (NB14 75 s vs NB11 93-98 s)."),
    "H14-30": ("menjadi +0,076", "0,076", None, None),
    "P-03": ("waktu kernel sekitar 2,1 menit", "2,1", "COCOK", "Log demo-kaggle-v2: ~128 s waktu kernel."),
    "P-04": ("di laptop 63 detik", "63", None, "Bukti sidang/bukti_demo/03_Demo_Model_Tersimpan.waktu.csv."),
    "P-05": ("kedua metode benar 80 % (24 dari 30)", "80", None, "Bukti sidang/bukti_demo/03_Demo_Model_Tersimpan.executed.ipynb."),
    "P-06": ("6 dari 8 tebakan benar", "6", None, "Klaim lama 5/8 salah hitung; uraiannya sendiri memberi 6/8."),
}


def main() -> None:
    for rel, ids, lama, baru in GANTI:
        p = ROOT / rel
        t = p.read_text(encoding="utf-8")
        n = t.count(lama)
        if n == 0 and baru in t:
            print(f"lewati (sudah) {rel} {ids}")
            continue
        assert n == 1, f"{rel} {ids}: teks lama ditemukan {n} kali"
        p.write_text(t.replace(lama, baru), encoding="utf-8")
        print(f"diperbaiki {rel} {ids}")

    hasil04 = ROOT / "hasil04" / "Ringkasan_untuk_laporan.md"
    t = hasil04.read_text(encoding="utf-8")
    catatan = ("> **Usang (2026-11-20):** angka di file ini berasal dari run v5, sebelum koreksi\n"
               "> drift. Angka yang dipakai di README dan sidang adalah run v6, buktinya di\n"
               "> `sidang/bukti_kaggle/jsd-fuzzy-twostage-sensorid/`.\n\n")
    if not t.startswith("> **Usang"):
        hasil04.write_text(catatan + t, encoding="utf-8")

    rows = list(csv.DictReader(open(LEDGER, encoding="utf-8")))
    kolom = list(rows[0].keys())
    for r in rows:
        bf = r["bukti_file"]
        m = re.match(r"\[KAGGLE\] kaggle kernels output kelvindsmn/([\w-]+) -> (.*)", bf)
        if m:
            r["bukti_file"] = f"sidang/bukti_kaggle/{m.group(1)}/{m.group(2)}"
            if r["status"] == "TIDAK ADA BUKTI LOKAL":
                if "Kaggle mengonfirmasi (COCOK" in r["catatan"]:
                    r["status"] = "COCOK"
                elif "Kaggle mengonfirmasi (PEMBULATAN" in r["catatan"]:
                    r["status"] = "PEMBULATAN"
        if r["id"] in LEDGER_BARU:
            klaim, nilai, status, cat = LEDGER_BARU[r["id"]]
            r["klaim"], r["nilai_klaim"] = klaim, nilai
            if status:
                r["status"] = status
            if cat:
                r["catatan"] = (cat + " " + r["catatan"]).strip()
    with open(LEDGER, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom)
        w.writeheader()
        w.writerows(rows)
    print("ledger diperbarui")


if __name__ == "__main__":
    main()
