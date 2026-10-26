# MAPPING — notebook sidang → notebook penelitian

Dokumen ini satu-satunya acuan kalau ada pertanyaan "angka ini dari mana?".
Tiga notebook sidang **tidak menghitung ulang** model; mereka membaca hasil yang
sudah diekspor notebook penelitian, lalu menampilkannya dengan bahasa sederhana.

- Commit acuan: `27737ee` (angka 01–11 sesudah koreksi drift)
- Tanggal: 2026-10-26
- Display name Kaggle: **Riset Deteksi Fault 4 Sensor Kelembaban**
- Semua kernel penelitian berstatus **Private**; tautan hanya untuk penguji.

---

## Tabel A — bagian sidang → notebook asli → sel → kernel

| Bagian sidang | Notebook asli | Sel | Kernel (versi) |
|---|---|---|---|
| Peta: alur 8 tahap | `07` §0–§9 | 1–43 | `jsd-fuzzy-cv-5skenario` v23 |
| Peta: ringkasan angka | `11` §13 | 47–48 | `jsd-edm-ann-lm-lengkap-q3` v3 |
| Alur: data mentah | `Ilustrasi` §1–§2 | 1–7 | `ilustrasi-data-akuisisi-dan-fault` v3 |
| Alur: broker 4 kanal | `02` | 1–13 | `jsd-fuzzy-broker-architecture-v2` v3 |
| Alur: sinkron 5 menit | `07` §3–§3c | 9–16 | `jsd-fuzzy-cv-5skenario` v23 |
| Alur: injeksi fault | `07` §4–§4b | 17–20 | `jsd-fuzzy-cv-5skenario` v23 |
| Alur: segmentasi | `07` §5 | 21–25 | `jsd-fuzzy-cv-5skenario` v23 |
| Alur: entropi | `11` §5–§7b | 15–24 | `jsd-edm-ann-lm-lengkap-q3` v3 |
| Hasil: performa utama | `11` §10 | 33–40 | `jsd-edm-ann-lm-lengkap-q3` v3 |
| Hasil: sensor mana rusak | `11` §12 | 43–46 | `jsd-edm-ann-lm-lengkap-q3` v3 |
| Hasil: jenis fault | `05` | 13–19 | `jsd-fuzzy-faulttype-per-sensor` v2 |
| Hasil: two-stage | `04` | 15–20 | `jsd-fuzzy-twostage-sensorid` v6 |
| Hasil: algoritma pelatih | `09` §8–§11 | 21–39 | `ann-lm-vs-bayesian-vs-ga-paper-pmdc` v3 |
| Hasil: ANN vs XGBoost | `06` | 36–41 | `jsd-xgb-perscenario` v4 |
| Statistik (1 kalimat) | `11` §11 | 41–42 | `jsd-edm-ann-lm-lengkap-q3` v3 |

Nomor sel = indeks 0-based di file `.ipynb` (sel yang sama dengan yang dibuka
di Kaggle).

## Tabel B — angka mana yang dikutip

| Kalau ditanya | Kutip dari | Jangan kutip dari | Alasan |
|---|---|---|---|
| Perbandingan EDM vs JSD utama | `11` v3 | `07`, `08` | 11 memakai ANN-LM asli, bukan `lbfgs` |
| Sensor mana yang rusak | `11` §12 | `07` §10 | 07 masih `lbfgs` |
| Algoritma pelatih terbaik | `09` v3 | `01`, `03` | 01/03 memakai sklearn, bukan LM |
| Broker 4 kanal vs digabung | `02` v3 | — | khusus notebook 02 |
| Jenis fault per sensor | `05` v2 | — | khusus notebook 05 |
| ANN vs XGBoost | `06` v4 | — | khusus notebook 06 |

Catatan supersesi:

- `11` = versi lengkap dari `10` (10 tanpa identifikasi sensor).
- `10` = `08` dengan LM asli menggantikan `lbfgs`.
- Karena protokol/anggaran beda, **jumlah sel menang juga beda**: `10` 10/15,
  `11` 11/15. Yang dipakai di sidang: **11/15** (notebook 11).
- `07` dan `08` tetap dirujuk untuk **diagram/alur**, bukan untuk angka akhir.

## Tabel C — jenis pelatih (sumber salah klaim paling sering)

| Notebook | Pelatih | Catatan |
|---|---|---|
| `07`, `08` | sklearn `MLPClassifier` (`lbfgs`) | bukan LM paper |
| `01`, `04`, `05` | sklearn `MLPClassifier` | bukan LM paper |
| `09` | `fit_lm` sendiri + Bayesian + GA + Adam | LM versi paper |
| `10`, `11` | `lm_fit` + Jacobian analitik | LM versi paper, tanpa `lbfgs` |
| `06` | `XGBClassifier` | pembanding non-ANN |

Saat sidang, sebut pelatih sesuai tabel ini. Jangan menyebut "LM" untuk
notebook 01/04/05/07/08.

## Tabel D — arti file di `sidang/assets/`

| Folder | File | Isi |
|---|---|---|
| `04` | `tahap2_ringkas_per_tugas.csv` | F1 tahap-2 per tingkat kesulitan (T1/T2/T3) |
| `04` | `tahap2_per_jumlah_fault.csv` | F1 vs jumlah fault bercampur, termasuk semua-4-benar |
| `05` | `cara_b_stage2_summary.csv` | F1 per sensor dan per jenis fault |
| `05` | `cara_b_stage2_sensor_x_faulttype.csv` | rincian 32 baris sensor × fault |
| `05` | `cara_b_heatmap_f1.png` | peta panas F1 |
| `06` | `xgb_perscenario_results.csv` | F1 XGBoost per skenario × set fitur |
| `07` | `07_performa_cv_5skenario.csv` | CV 5 skenario (protokol 07) |
| `07` | `07_identifikasi_sensor_ringkas.csv` | sensor-ID protokol 07 |
| `07` | `07_mutu_data.csv` | pemeriksaan mutu data |
| `07` | `07_broker_tersinkron.png` | 4 kanal tersinkron |
| `07` | `07_contoh_fault_vs_asli_S1.png` | contoh injeksi |
| `07` | `07_grid_fault_vs_asli_S1.png` | grid 16 kondisi |
| `08` | `08_ringkasan_rq.txt` | ringkasan RQ1–RQ5 protokol 08 |
| `09` | `09_rq2_performa.csv` | F1 4 pelatih × 5 skenario × 2 set fitur |
| `09` | `09_rq2_peringkat_pelatih.csv` | rata-rata F1 dan juara per pelatih |
| `09` | `09_rq2_performa.png` | grafik performa pelatih |
| `11` | `11_rq4_performa_ann.csv` | F1 15 sel (5 skenario × 3 ukuran window) |
| `11` | `11_rq4_selisih.csv` | selisih JSD − EDM + status menang |
| `11` | `11_rq4_performa_N700.png` | grafik N=700 |
| `11` | `11_rq4_confusion_matrix_N700.png` | confusion matrix |
| `11` | `11_sensor_ringkas.csv` | F1 sensor-ID 15 sel |
| `11` | `11_sensor_f1.png` | grafik sensor-ID |
| `11` | `11_rq5_ttest.csv` | uji berpasangan (lampiran privat) |
| `11` | `11_akuntansi_window.csv` | jumlah window & pembuangan |
| `11` | `11_rq4_biaya_komputasi.csv` | CPU/wall per sel |
| `11` | `11_broker_tersinkron.png` | 4 kanal tersinkron |
| `11` | `11_contoh_fault_vs_asli_S1.png`, `_S4.png` | contoh injeksi |
| `11` | `11_grid_fault_vs_asli_S1.png` | grid 16 kondisi |
| `11` | `11_similarity_euclidean_vs_jsd.png` | EDM vs JSD |
| `ilustrasi` | `ILU_empat_kanal_bersih.png` | data mentah 4 kanal |
| `ilustrasi` | `ILU_grid_fault_vs_asli_S1.png` | grid injeksi |
| `ilustrasi` | `ILU_tabel_10_langkah.csv` | 10 langkah pembersihan |

## Tabel E — kernel & versi

| # | Slug | Versi | Status |
|---|---|---|---|
| 01 | `jsd-fuzzy-ann-paper` | 6 | private |
| 02 | `jsd-fuzzy-broker-architecture-v2` | 3 | private |
| 03 | `jsd-fuzzy-scenario-ladder` | 3 | private |
| 04 | `jsd-fuzzy-twostage-sensorid` | 6 | private |
| 05 | `jsd-fuzzy-faulttype-per-sensor` | 2 | private |
| 06 | `jsd-xgb-perscenario` | 4 | private |
| 07 | `jsd-fuzzy-cv-5skenario` | 23 | private |
| 08 | `jsd-fuzzy-vs-edm-fuzzy-paper-q3` | 17 | private |
| 09 | `ann-lm-vs-bayesian-vs-ga-paper-pmdc` | 3 | private |
| 10 | `jsd-vs-edm-dengan-ann-lm-asli-paper-pmdc` | 4 | private |
| 11 | `jsd-edm-ann-lm-lengkap-q3` | 3 | private |
| — | `ilustrasi-data-akuisisi-dan-fault` | 3 | private |

Tiga notebook sidang (v1, semua private):

| Notebook | Slug |
|---|---|
| `00_Peta_Program.ipynb` | `peta-program-deteksi-fault-4-sensor-kelembaban` |
| `01_Alur_Sistem.ipynb` | `alur-sistem-deteksi-fault-4-sensor-kelembaban` |
| `02_Hasil.ipynb` | `hasil-deteksi-fault-4-sensor-kelembaban` |

Status private dan display name `Riset Deteksi Fault 4 Sensor Kelembaban`
diverifikasi di Kaggle pada 2026-10-26.

## Catatan angka yang sering tertukar

| Angka | Nilai benar | Sumber |
|---|---|---|
| Sel JSD menang (notebook 11) | **11/15** | `11_rq4_selisih.csv` + log |
| Rata-rata selisih F1 | **+0,028** | log pooled |
| Sensor-ID F1 | **0,675 vs 0,634** (menang 9/15) | `11_sensor_ringkas.csv` |
| Semua-4-benar | **0,307 vs 0,295** | log kernel 11 |
| Tahap-2 17 kelas | **0,902 vs 0,846** | `tahap2_ringkas_per_tugas.csv` |
| Semua-4-benar (04) | **0,920 vs 0,800** | `tahap2_per_jumlah_fault.csv` |
| Pelatih terbaik | BR 0,429 ≈ LM 0,433 (p = 0,81) | `09_rq2_peringkat_pelatih.csv` |
| XGBoost JSD+Time | **0,750** rata-rata 5 skenario | `xgb_perscenario_results.csv` |
