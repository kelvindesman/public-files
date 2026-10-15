# Notebook 11 — bahan untuk laporan (EDM vs JSD, SEMUA classifier LM asli)

Sumber: Kaggle `kelvindsmn/jsd-edm-ann-lm-lengkap-q3` (version 1, COMPLETE,
wall 525 mnt / 8,75 jam, 0 Traceback, 0 SKIP).
Notebook: `11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb` (51 sel, dibangun dari
`build_11_annlm_real_notebook.py`).
Link: https://www.kaggle.com/code/kelvindsmn/jsd-edm-ann-lm-lengkap-q3

Kejujuran metodologi: tidak ada `lbfgs`/`adam`/`MLPClassifier` di notebook
ini. Semua classifier = `lm_fit` (Levenberg–Marquardt: Gauss-Newton
terdamping + rasio gain + Jacobian analitik + self-check, dari notebook 09).
Bagian sensor-ID (§12) = satu `lm_fit` langsung pada target multi-hot
4 sensor (ambang 0,5), dirantai end-to-end dari prediksi out-of-fold §10.

File mentah: `hasil11/exports/` (37 file: CSV + PNG; tidak ikut commit karena
`exports/` di-gitignore — unduh ulang dari halaman Outputs kernel bila perlu).

## Bagian 1 — CV stabilitas fitur (4 metode × 3 window × skala)

EDM-Fuzzy dan JSD-Fuzzy: 0 NaN/Inf di semua window (matriks Valid).
MSE/CMSE: ribuan NaN pada N=200 (ditambal median — pola yang sama seperti 08).

Hasil JSD vs EDM (CV reduction): N=200 JSD lebih stabil pada 124/315
pasangan (39,4%, rata-rata −26,3%) · N=700 74/315 (23,5%, −28,2%) · N=1000
39/315 (12,4%, −30,9%) → **EDM-Fuzzy lebih stabil pada sebagian besar skala**.
Ini temuan jujur yang harus masuk laporan apa adanya.

## Bagian 2 — Klasifikasi 5 skenario (LM asli, grouped CV)

Selisih F1 (JSD − EDM): **JSD unggul pada 11 dari 15 sel**
(N=200: 4/5 · N=700: 3/5 · N=1000: 4/5).
Rata-rata selisih: N=200 +0,037 · N=700 +0,034 · N=1000 +0,019 · pooled +0,030.

Uji berpasangan: per window **belum signifikan**
(p = 0,12 / 0,42 / 0,29); pooled 15 pasang p = 0,0511 (t maupun Wilcoxon) —
tipis di atas 0,05, jadi statusnya tetap "belum signifikan", bukan "lebih baik".
ROC-AUC OvR tersedia per sel di `11_rq4_performa_ann.csv`; confusion matrix
kedua metode per skenario di `11_rq4_confusion_matrix.csv` + PNG.

## Bagian 3 — Sensor fault identification (RQ6, LM asli)

F1 sensor rata-rata lintas window & skenario:
**EDM-Fuzzy 0,722 vs JSD-Fuzzy 0,770.**
Exact-match SEMUA-4-BENAR: EDM 0,363 vs JSD 0,466.

| N | S1 E/J | S2 E/J | S3 E/J | S4 E/J | S5 E/J |
|---|---|---|---|---|---|
| 200 | 0,601/0,730 | 0,640/0,687 | 0,656/0,733 | 0,768/0,878 | 0,840/0,910 |
| 700 | 0,582/0,650 | 0,640/0,634 | 0,718/0,760 | 0,832/0,939 | 0,895/0,817 |
| 1000 | 0,631/0,656 | 0,597/0,665 | 0,605/0,680 | 0,898/0,912 | 0,924/0,895 |

(E = EDM-Fuzzy, J = JSD-Fuzzy. JSD unggul di 12 dari 15 sel; pengecualian:
S2-N700, S5-N700, S5-N1000 — tulis apa adanya.)

## Separabilitas (RQ3)

Campuran: JSD lebih terpisah di S1 (semua window) dan S5-N200/N700;
EDM lebih terpisah di S3-N700/N1000, S2-N1000, S5-N1000; sisanya sebanding.
Detail per sel di `11_rq3_interpretasi.csv` + boxplot PNG per window.

## Kalimat usulan untuk laporan

> Seluruh classifier pada studi ini memakai Levenberg–Marquardt yang
> diimplementasikan sendiri (langkah Gauss-Newton terdamping, rasio gain,
> Jacobian analitik terverifikasi self-check) — tanpa solver quasi-Newton.
> JSD-Fuzzy mengungguli EDM-Fuzzy pada 11 dari 15 sel klasifikasi
> (rata-rata +0,02 s.d. +0,04) dan pada identifikasi sensor (F1 0,770 vs
> 0,722), namun uji berpasangan per window belum signifikan (p ≥ 0,05;
> pooled p = 0,051), sehingga klaim ditulis sebagai keunggulan deskriptif.
> Pada stabilitas fitur (CV), EDM-Fuzzy justru lebih stabil pada sebagian
> besar skala — dilaporkan apa adanya.
