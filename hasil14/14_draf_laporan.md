# Draf laporan — Pipeline lengkap dengan fault bentuk baru (notebook 14)

Sumber angka: `hasil14/exports/14_*.csv` dan `14_ringkasan_rq.txt`, dari kernel
Kaggle `kelvindsmn/jsd-edm-lengkap-fault-baru`, versi 1. Run selesai
(COMPLETE) dalam 478 menit, tanpa Traceback dan tanpa SKIP.
Pembanding: notebook 11 v3 (`hasil11/exports/11_*`), yang tetap menjadi hasil
utama dan tidak diubah.

## Rancangan

Notebook 14 menjalankan pipeline notebook 11 secara utuh: S1–S5, window
N = 200/700/1000, grid search ANN-LM asli, grouped CV, RQ1–RQ6 termasuk
identifikasi sensor, dan biaya komputasi. Yang berubah hanya bentuk fault,
memakai tingkat "sedang" dari uji intensitas (notebook 13):

| Gangguan | Notebook 11 | Notebook 14 |
|---|---|---|
| Bias | +0,08 poin, konstan sepanjang deret | 5 poin selama 3 episode × 3 hari, lalu kembali normal |
| Drift | 1 poin/hari per episode (total 3 poin) | linear per episode, total 5 poin |
| Spike | periodik, 0,08 × SD | relatif x(t)·(1+0,30), posisi acak, kepadatan 1,52% |
| Hardware | 8% stuck + 5% hilang | tidak berubah |
| Kombinasi | komponen ditumpuk sepanjang deret | semua komponen pada episode yang sama |
| Label window | porsi sampel fault > 1% | semua komponen ada: spike ≥ 1 titik, komponen lain > 1% |

Karena fault kini hanya terjadi di dalam episode, lebih banyak window fault
yang dibuang karena porsinya terlalu kecil. Jumlah yang dibuang: 26.548 dari
36.120 (N = 200), 7.108 dari 10.191 (N = 700) dan 4.635 dari 7.095
(N = 1000). Di notebook 11 jumlahnya 1.992, 536 dan 344. Jumlah sampel per
kelas yang masuk classifier tetap sama dengan notebook 11 (RQ1: n = 1.700 /
1.621 / 1.526).

## RQ4 — Performa deteksi (F1 makro ANN-LM, 16 kondisi)

| N | Skenario | NB11 EDM | NB11 JSD | NB14 EDM | NB14 JSD |
|---|---|---|---|---|---|
| 200 | S1 | 0,653 | 0,634 | 0,612 | 0,702 |
| 200 | S2 | 0,300 | 0,315 | 0,280 | 0,370 |
| 200 | S3 | 0,219 | 0,189 | 0,206 | 0,235 |
| 200 | S4 | 0,367 | 0,397 | 0,266 | 0,408 |
| 200 | S5 | 0,761 | 0,777 | 0,685 | 0,834 |
| 700 | S1 | 0,595 | 0,671 | 0,658 | 0,773 |
| 700 | S2 | 0,328 | 0,347 | 0,372 | 0,532 |
| 700 | S3 | 0,184 | 0,196 | 0,237 | 0,284 |
| 700 | S4 | 0,339 | 0,426 | 0,334 | 0,311 |
| 700 | S5 | 0,804 | 0,872 | 0,653 | 0,633 |
| 1000 | S1 | 0,490 | 0,668 | 0,542 | 0,669 |
| 1000 | S2 | 0,381 | 0,365 | 0,292 | 0,384 |
| 1000 | S3 | 0,220 | 0,164 | 0,249 | 0,211 |
| 1000 | S4 | 0,382 | 0,400 | 0,314 | 0,350 |
| 1000 | S5 | 0,849 | 0,872 | 0,614 | 0,764 |

JSD-Fuzzy unggul pada 12 dari 15 sel di notebook 14 (notebook 11: 11/15).
EDM-Fuzzy unggul pada N = 700 S4, N = 700 S5 dan N = 1000 S3.

## RQ5 — Uji statistik F1

| N | NB11: EDM / JSD, p (t berpasangan) | NB14: EDM / JSD, p (t berpasangan) |
|---|---|---|
| 200 | 0,460 / 0,462, p = 0,854 | 0,410 / 0,510, **p = 0,010** |
| 700 | 0,450 / 0,502, **p = 0,027** | 0,451 / 0,507, p = 0,199 |
| 1000 | 0,464 / 0,494, p = 0,499 | 0,402 / 0,476, p = 0,095 |
| gabungan | 0,458 / 0,486, p = 0,076 | 0,421 / 0,497, **p = 0,0006** (Wilcoxon p = 0,0015, d = 1,13) |

## RQ6 — Identifikasi sensor

| Ukuran | NB11 EDM | NB11 JSD | NB14 EDM | NB14 JSD |
|---|---|---|---|---|
| F1 sensor rata-rata | 0,634 | 0,675 | 0,693 | 0,791 |
| Semua 4 sensor benar | 0,295 | 0,307 | 0,356 | 0,483 |

## RQ2/RQ3 — Stabilitas dan separabilitas

- RQ2 (CV entropi): polanya sama dengan notebook 11. Kedua metode sebanding
  pada N = 200, dan EDM-Fuzzy lebih stabil pada N = 700 dan 1000 (JSD lebih
  rendah CV-nya pada 48,3% / 31,1% / 17,8% pasangan; NB11 45,7% / 18,7% /
  16,5%).
- RQ3 (boxplot mean entropy): JSD-Fuzzy lebih terpisah pada 11 dari 15 sel,
  dan tidak ada sel yang dimenangi EDM. Empat sel lainnya campuran, semuanya
  pada N = 200 (S1, S2, S4, S5).

## Biaya komputasi (Kaggle 4 CPU)

| Ukuran | NB11 | NB14 |
|---|---|---|
| Wall total run | 610 menit | 478 menit |
| Ekstraksi fitur JSD-Fuzzy, N = 200 / 700 / 1000 | 324 / 1.055 / 1.628 s | 345 / 922 / 1.340 s |
| Ekstraksi fitur EDM-Fuzzy, N = 200 / 700 / 1000 | 186 / 219 / 241 s | 202 / 222 / 220 s |
| Fit ANN-LM per fold, rata-rata EDM / JSD | 75,5 / 75,1 s | 93,2 / 98,0 s |
| Inferensi per window, rata-rata EDM / JSD | 0,089 / 0,088 ms | 0,097 / 0,087 ms |
| Identifikasi sensor (§12) | 2.133 s | 1.114 s |

Pola biayanya sama: ekstraksi JSD-Fuzzy 1,7–6 kali lebih lama daripada
EDM-Fuzzy, dan biaya inferensi per window hampir sama. Total run lebih
singkat karena identifikasi sensor lebih cepat. Fit ANN-LM per fold lebih
lama, dan beberapa fold berhenti karena batas waktu latih (`MAX_TRAIN_S`).

## Temuan utama

1. Dengan fault yang lebih besar dan berbentuk episode, keunggulan JSD-Fuzzy
   atas EDM-Fuzzy menjadi lebih jelas. Selisih F1 gabungan naik dari +0,028
   (p = 0,076, tidak signifikan) menjadi +0,077 (p = 0,0006, signifikan).
2. Identifikasi sensor membaik pada kedua metode, dan paling besar pada
   JSD-Fuzzy: F1 sensor naik 0,675 → 0,791, dan semua-4-benar naik
   0,307 → 0,483.
3. Skenario dengan banyak kombinasi (S2–S4) tetap sulit: F1 makro 0,21–0,53.
   Hasil ini sejalan dengan notebook 13, yang menunjukkan bias dan drift
   kecil hampir tidak terlihat oleh fitur entropi.
4. S5 justru turun dibanding notebook 11 (N = 700: 0,80/0,87 → 0,65/0,63).
   Label "semua komponen ada" lebih ketat, dan kini fault hanya ada di dalam
   episode.

## Catatan

- Tingkat "sedang" dan aturan kombinasi diambil dari notebook 13. Tab
  "Perubahan program" pada dokumen usulan tidak tersedia, jadi nilai-nilai
  itu adalah asumsi.
- Hasil utama skripsi tetap notebook 11 v3. Notebook 14 adalah uji tambahan
  untuk menjawab revisi pasca-sidang.
