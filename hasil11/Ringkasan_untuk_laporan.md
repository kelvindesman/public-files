# Notebook 11 — bahan untuk laporan (EDM vs JSD, SEMUA classifier LM asli)

Sumber: Kaggle `kelvindsmn/jsd-edm-ann-lm-lengkap-q3` (version 3, COMPLETE,
wall 610 mnt / 10,2 jam, 0 Traceback, 0 SKIP). Versi 3 = run ulang setelah
simulator drift dikoreksi (1 poin/hari dalam episode 3 hari).
Notebook: `11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb` (54 sel, dibangun dari
`build_11_annlm_real_notebook.py`).
Link: https://www.kaggle.com/code/kelvindsmn/jsd-edm-ann-lm-lengkap-q3

Kejujuran metodologi: tidak ada `lbfgs`/`adam`/`MLPClassifier` di notebook
ini. Semua classifier = `lm_fit` (Levenberg–Marquardt: Gauss-Newton
terdamping + rasio gain + Jacobian analitik + self-check, dari notebook 09).
Bagian sensor-ID (§12) = satu `lm_fit` langsung pada target multi-hot
4 sensor (ambang 0,5), dirantai end-to-end dari prediksi out-of-fold §10.

File mentah: `hasil11/exports/` (49 file: 21 CSV + 27 PNG + 1 TXT; tidak ikut
commit karena `exports/` di-gitignore — unduh ulang dari halaman Outputs
kernel bila perlu).

## Bagian 1 — CV stabilitas fitur (4 metode × 3 window × skala)

EDM-Fuzzy dan JSD-Fuzzy: 0 NaN/Inf di semua window (matriks Valid).
MSE/CMSE: ribuan NaN pada N=200 (ditambal median — pola yang sama seperti 08).

Hasil JSD vs EDM (CV reduction): N=200 JSD lebih stabil pada 144/315
pasangan (45,7%, rata-rata −26%) · N=700 59/315 (18,7%, −28%) · N=1000
52/315 (16,5%, −31%) → **EDM-Fuzzy lebih stabil pada 56 dari 63 sel
(89%)**. Ini temuan jujur yang harus masuk laporan apa adanya.

## Bagian 2 — Klasifikasi 5 skenario (LM asli, grouped CV)

Selisih F1 (JSD − EDM): **JSD unggul pada 10 dari 15 sel**
(N=200: 3/5 · N=700: 5/5 · N=1000: 3/5).
Rata-rata selisih: N=200 +0,002 · N=700 +0,053 · N=1000 +0,030 · pooled +0,028.
F1 rata-rata: EDM 0,458 vs JSD 0,486.

Uji berpasangan: **N=700 signifikan** (p = 0,027; Wilcoxon 0,0625);
N=200 p = 0,85 · N=1000 p = 0,50; **pooled 15 pasang p = 0,076**
(t), Wilcoxon 0,107 — belum signifikan secara gabungan, jadi statusnya
tetap "belum signifikan", bukan "lebih baik".
ROC-AUC OvR tersedia per sel di `11_rq4_performa_ann.csv`; confusion matrix
kedua metode per skenario di `11_rq4_confusion_matrix.csv` + PNG.

## Bagian 3 — Sensor fault identification (RQ6, LM asli)

F1 sensor rata-rata lintas window & skenario:
**EDM-Fuzzy 0,634 vs JSD-Fuzzy 0,675.**
Exact-match SEMUA-4-BENAR: EDM 0,295 vs JSD 0,307.

| N | S1 E/J | S2 E/J | S3 E/J | S4 E/J | S5 E/J |
|---|---|---|---|---|---|
| 200 | 0,420/0,644 | 0,575/0,543 | 0,582/0,621 | 0,742/0,684 | 0,815/0,788 |
| 700 | 0,622/0,668 | 0,646/0,664 | 0,653/0,622 | 0,720/0,780 | 0,834/0,896 |
| 1000 | 0,410/0,318 | 0,544/0,658 | 0,642/0,662 | 0,763/0,742 | 0,540/0,838 |

(E = EDM-Fuzzy, J = JSD-Fuzzy. JSD unggul di 9 dari 15 sel; pengecualian:
S2-N200, S4-N200, S5-N200, S3-N700, S1-N1000, S4-N1000 — tulis apa adanya.)

## Separabilitas (RQ3)

Campuran: JSD lebih terpisah di S1-N200/N700, S3-N700/N1000, S2-N1000,
S5-N1000; EDM lebih terpisah di S2-N200, S4-N200, S5-N200, S1-N1000;
sisanya sebanding. Detail per sel di `11_rq3_interpretasi.csv` + boxplot PNG
per window.

## Kalimat usulan untuk laporan

> Seluruh classifier pada studi ini memakai Levenberg–Marquardt yang
> diimplementasikan sendiri (langkah Gauss-Newton terdamping, rasio gain,
> Jacobian analitik terverifikasi self-check) — tanpa solver quasi-Newton.
> JSD-Fuzzy mengungguli EDM-Fuzzy pada 10 dari 15 sel klasifikasi
> (rata-rata +0,03) dan pada identifikasi sensor (F1 0,675 vs 0,634), namun
> hanya window N=700 yang signifikan (p = 0,027); uji gabungan 15 pasang
> belum signifikan (p = 0,076), sehingga klaim ditulis sebagai keunggulan
> deskriptif. Pada stabilitas fitur (CV), EDM-Fuzzy justru lebih stabil pada
> 89% sel — dilaporkan apa adanya.
