# Fault Sensor Identification — bahan untuk laporan (Paper Q3)

Sumber: re-run Kaggle `kelvindsmn/jsd-fuzzy-twostage-sensorid` (version 5,
2026-10-15, status COMPLETE, 0 Traceback).
Notebook: `04_Klasifikasi_Dulu_Baru_Sensor_Mana.ipynb` — Stage 1 klasifikasi,
Stage 2 identifikasi sensor (`MultiOutputClassifier`, 4 label biner S1..S4)
dirantai dari window yang lolos Stage 1 (end-to-end). Kedua metode dijalankan
pada split test yang sama: **EDM-Fuzzy dan JSD-Fuzzy**.

File mentah: `hasil04/exports/tahap2_ringkas_per_tugas.csv`,
`tahap2_per_jumlah_fault.csv`, `tahap2_per_skenario.csv`,
`tahap2_identifikasi_sensor.csv`, `tahap1_akurasi_per_tugas.csv`.

Link Kaggle:
- 04 (fault identification, kedua metode):
  https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-twostage-sensorid
- 05 (jenis fault apa di sensor mana, 16 label):
  https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-faulttype-per-sensor
- 07 (skema flowchart + §10 sensor mana, §11 kontrol):
  https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-cv-5skenario
- 08 (paper Q3 — hanya klasifikasi S1..S5, TIDAK ada tabel sensor):
  https://www.kaggle.com/code/kelvindsmn/jsd-fuzzy-vs-edm-fuzzy-paper-q3

## Tabel 1 — F1 identifikasi sensor (rata-rata S1..S4, end-to-end)

| Tugas Tahap 1 | EDM-Fuzzy F1 | EDM-Fuzzy ROC-AUC | JSD-Fuzzy F1 | JSD-Fuzzy ROC-AUC |
|---|---|---|---|---|
| T1 biner (fault vs non-fault) | 0,721 | 0,712 | 0,821 | 0,735 |
| T2 single-5 (normal + 4 fault tunggal) | 0,756 | 0,682 | 0,740 | 0,660 |
| T3 17 kelas (penuh) | 0,860 | 0,784 | 0,899 | 0,881 |

## Tabel 2 — F1 sensor menurut banyaknya fault yang bercampur (tugas 17 kelas)

| Jumlah fault | EDM-Fuzzy F1 sensor | EDM-Fuzzy semua-4-benar | JSD-Fuzzy F1 sensor | JSD-Fuzzy semua-4-benar |
|---|---|---|---|---|
| 1 | 0,740 | 0,353 | 0,781 | 0,445 |
| 2 | 0,866 | 0,624 | 0,907 | 0,682 |
| 3 | 0,951 | 0,802 | 0,967 | 0,862 |
| 4 | 0,977 | 0,880 | 0,997 | 0,980 |

## Tabel 3 — Tahap 1 (konteks: error Tahap 1 diteruskan ke Tahap 2)

| Tugas | EDM-Fuzzy akurasi | EDM-Fuzzy macro-F1 | JSD-Fuzzy akurasi | JSD-Fuzzy macro-F1 |
|---|---|---|---|---|
| T1 biner | 0,661 | 0,653 | 0,729 | 0,719 |
| T2 single-5 | 0,578 | 0,532 | 0,635 | 0,563 |
| T3 17 kelas | 0,496 | 0,482 | 0,553 | 0,539 |

## Kalimat usulan untuk laporan

> Identifikasi sensor yang rusak dilakukan dua tahap: Tahap 1
> mengklasifikasikan window (biner, 5 kelas, dan 17 kelas, masing-masing untuk
> EDM-Fuzzy dan JSD-Fuzzy), Tahap 2 menunjuk sensor mana (S1..S4) yang membawa
> fault pada window yang lolos Tahap 1. Pada tugas 17 kelas, F1 identifikasi
> sensor rata-rata mencapai 0,860 (EDM-Fuzzy) dan 0,899 (JSD-Fuzzy); JSD-Fuzzy
> unggul pada semua tingkat campuran fault, dengan selisih terbesar pada
> campuran 1 fault (0,781 vs 0,740) dan kinerja hampir sempurna pada 4 fault
> (0,997 vs 0,977).

Catatan jujur untuk laporan: pada tugas T2_single5, JSD-Fuzzy sedikit di bawah
EDM-Fuzzy (0,740 vs 0,756) — jangan tulis "JSD menang di semua tugas".
