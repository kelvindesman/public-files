# MAPPING — notebook sidang → notebook penelitian

Dokumen ini satu-satunya acuan kalau ada pertanyaan "angka ini dari mana?".
Notebook sidang **tidak menghitung ulang** model; ia membaca hasil yang sudah
diekspor notebook penelitian, lalu menampilkannya dengan bahasa sederhana.

- Commit acuan: `27737ee` (angka 01–11 sesudah koreksi drift)
- Tanggal: 2026-10-26
- Display name Kaggle: **Riset Deteksi Fault 4 Sensor Kelembaban**
- Semua kernel berstatus **Private**; tautan hanya untuk penguji.
- Notebook sidang utama: `sidang/00_Sidang_Deteksi_Fault.ipynb`
  ([Kaggle](https://www.kaggle.com/code/kelvindsmn/sidang-deteksi-fault-4-sensor-kelembaban))
  — menggabungkan Peta, Alur, dan Hasil dalam satu kernel + tabel tautan ke 12
  notebook penelitian. Versi Kaggle: **v2** (performa + sensor-ID + catatan
  data; statistik dan pelatih lain tidak ditampilkan).

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
| Hasil: catatan data 0 | data mentah | — | `data_sensor.csv` |
| Hasil: batasan | `11` §13 | 47–48 | `jsd-edm-ann-lm-lengkap-q3` v3 |

Tidak ditampilkan di edisi sidang (tetap ada di arsip penelitian):
jenis fault (`05`), two-stage (`04`), empat pelatih (`09`), ANN vs XGBoost
(`06`), dan statistik (`11` §11). Alasannya: fokus sidang adalah perbandingan
EDM vs JSD dengan **satu** protokol ANN-LM.

Nomor sel = indeks 0-based di file `.ipynb` (sel yang sama dengan yang dibuka
di Kaggle).

## Tabel B — angka mana yang dikutip

| Kalau ditanya | Kutip dari | Jangan kutip dari | Alasan |
|---|---|---|---|
| Perbandingan EDM vs JSD utama | `11` v3 | `07`, `08` | 11 memakai ANN-LM asli, bukan `lbfgs` |
| Sensor mana yang rusak | `11` §12 | `07` §10 | 07 masih `lbfgs` |
| Broker 4 kanal vs digabung | `02` v3 | — | khusus notebook 02 |
| (arsip) pelatih terbaik | `09` v3 | `01`, `03` | 01/03 memakai sklearn, bukan LM |
| (arsip) jenis fault per sensor | `05` v2 | — | khusus notebook 05 |
| (arsip) ANN vs XGBoost | `06` v4 | — | khusus notebook 06 |

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
| `11` | `11_rq4_performa_ann.csv` | F1 15 sel (5 skenario × 3 ukuran window) |
| `11` | `11_rq4_selisih.csv` | selisih JSD − EDM + status menang |
| `11` | `11_rq4_performa_N700.png` | grafik N=700 |
| `11` | `11_rq4_confusion_matrix_N700.png` | confusion matrix |
| `11` | `11_sensor_ringkas.csv` | F1 sensor-ID 15 sel |
| `11` | `11_sensor_f1.png` | grafik sensor-ID |
| `11` | `11_akuntansi_window.csv` | jumlah window & pembuangan |
| `11` | `11_similarity_euclidean_vs_jsd.png` | EDM vs JSD |
| `11` | `11_grid_fault_vs_asli_S1.png` | grid 16 kondisi |
| `ilustrasi` | `ILU_empat_kanal_bersih.png` | data mentah 4 kanal |
| `ilustrasi` | `ILU_grid_fault_vs_asli_S1.png` | grid injeksi |
| `ilustrasi` | `ILU_tabel_10_langkah.csv` | 10 langkah pembersihan |

Aset untuk bagian yang tidak lagi ditampilkan (04/05/06/07/08/09) sudah
dihapus dari repo; file aslinya tetap ada di kernel Kaggle masing-masing.

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
| 12 | `sensitivitas-parameter-fault-dan-jsd` | 1 | private |
| — | `ilustrasi-data-akuisisi-dan-fault` | 3 | private |

Notebook sidang (semua private):

| Notebook | Slug | Status |
|---|---|---|
| `00_Sidang_Deteksi_Fault.ipynb` | `sidang-deteksi-fault-4-sensor-kelembaban` | **utama** (v2) |
| `00_Peta_Program.ipynb` | `peta-program-deteksi-fault-4-sensor-kelembaban` | digantikan |
| `01_Alur_Sistem.ipynb` | `alur-sistem-deteksi-fault-4-sensor-kelembaban` | digantikan |
| `02_Hasil.ipynb` | `hasil-deteksi-fault-4-sensor-kelembaban` | digantikan |
| `12_Sensitivitas_Parameter.ipynb` | `sensitivitas-parameter-fault-dan-jsd` | pendukung (bukan sidang) |

Status private dan display name `Riset Deteksi Fault 4 Sensor Kelembaban`
diverifikasi di Kaggle pada 2026-10-26.

Hasil eksperimen 12 disimpan di `hasil12/exports/` (CSV + PNG + ringkasan).
Angka pendukung: bias AUC ~0,50 (0,08–5 poin) karena centroid shifting;
hardware (0,3; 0,2) TPR 0,908/FPR 0,083; rekomendasi binning diuji dan
divalidasi N700/S1 (`12_rekomendasi.csv`).

## Catatan angka yang sering tertukar

| Angka | Nilai benar | Sumber |
|---|---|---|
| Sel JSD menang (notebook 11) | **11/15** | `11_rq4_selisih.csv` + log |
| Rata-rata selisih F1 | **+0,028** | log pooled |
| Sensor-ID F1 | **0,675 vs 0,634** (menang 9/15) | `11_sensor_ringkas.csv` |
| Semua-4-benar | **0,307 vs 0,295** | log kernel 11 |
| (arsip) Tahap-2 17 kelas | **0,902 vs 0,846** | `tahap2_ringkas_per_tugas.csv` |
| (arsip) Semua-4-benar (04) | **0,920 vs 0,800** | `tahap2_per_jumlah_fault.csv` |
| (arsip) Pelatih terbaik | BR 0,429 ≈ LM 0,433 (p = 0,81) | `09_rq2_peringkat_pelatih.csv` |
| (arsip) XGBoost JSD+Time | **0,750** rata-rata 5 skenario | `xgb_perscenario_results.csv` |
