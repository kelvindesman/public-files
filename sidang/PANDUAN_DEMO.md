# Panduan Demo — Deteksi Fault 4 Sensor Kelembaban

Panduan ini untuk siapa saja yang perlu menjalankan atau menonton demo
(Kelvin, pembimbing, penguji). Tidak perlu paham pemrograman.

**Intinya:** model sudah dilatih dan **disimpan ke file**. Demo hanya memakai
model itu, jadi selesai dalam **sekitar 1 menit**, bukan 10 jam.

---

## 1. Kenapa tidak perlu training ulang?

| | Run penelitian (notebook 11) | Demo (notebook ini) |
|---|---|---|
| Yang dikerjakan | hitung fitur ribuan potongan data + cari arsitektur terbaik + uji 5 lipatan, diulang 15 kali | buka model tersimpan, hitung fitur untuk beberapa potongan baru, lalu tebak |
| Waktu | 10,2 jam di Kaggle | sekitar 1 menit |
| Gunanya | menghasilkan **angka resmi** (F1, uji statistik) | **memperlihatkan** cara kerja sistem secara langsung |

Model disimpan sekali oleh notebook 16 (`simpan_model.py`, Kaggle
`simpan-model-final-deteksi-fault`). Isinya 10 model: 5 skenario × 2 metode
(EDM-Fuzzy dan JSD-Fuzzy), window N = 700.

## 2. Tautan (semua Private, buka dengan akun Kaggle Kelvin)

| Untuk apa | Tautan |
|---|---|
| **Demo** (dipakai saat Zoom) | https://www.kaggle.com/code/kelvindsmn/demo-model-tersimpan-deteksi-fault |
| File model + data (Dataset) | https://www.kaggle.com/datasets/kelvindsmn/model-deteksi-fault-tersimpan |
| Proses melatih + menyimpan model (58 menit, sekali saja) | https://www.kaggle.com/code/kelvindsmn/simpan-model-final-deteksi-fault |
| Ringkasan hasil sidang | https://www.kaggle.com/code/kelvindsmn/sidang-deteksi-fault-4-sensor-kelembaban |
| Run penelitian lengkap (bukti 10 jam + log) | https://www.kaggle.com/code/kelvindsmn/jsd-edm-ann-lm-lengkap-q3 |

Kalau penguji ingin melihat sendiri, kernel bisa dibagikan lewat tombol
**Share** di Kaggle (tetap Private, hanya orang yang diundang).

## 3. Persiapan (10 menit sebelum Zoom)

1. Login Kaggle, buka tautan **Demo**.
2. Klik **Copy & Edit** (pojok kanan atas). Notebook terbuka di mode edit.
3. Pastikan di panel kanan **Internet: On**.
4. Menu **Run → Run all** sekali sebagai pemanasan. Tunggu sampai semua sel
   selesai (di Kaggle waktu kernel sekitar 2,1 menit menurut log
   `hasil16/demo-kaggle-v2.log`, belum termasuk menyalakan mesin; di laptop
   63 detik, 21 detik tanpa Langkah 7 — bukti `sidang/bukti_demo/`). Kalau tidak ada tulisan merah, siap dipakai.
5. Di Zoom, bagikan layar jendela browser itu saja (bukan seluruh layar).

**Cadangan kalau internet bermasalah:** di laptop, buka
`sidang/03_Demo_Model_Tersimpan.ipynb` di VS Code atau Jupyter, pilih kernel
`venv` repo ini, lalu **Run All**. Model ada di folder `models/`, data di
`data_sensor.csv` — tidak butuh internet.

## 4. Naskah demo singkat (sekitar 5 menit)

| Menit | Tunjukkan | Katakan |
|---|---|---|
| 0:00 | judul notebook | "Ini demo pakai model yang sudah dilatih. Tidak ada training ulang, jadi cepat." |
| 0:30 | **Langkah 1** — tabel model | "Ada 10 model: 5 skenario kerusakan × 2 metode. Kolom kanan adalah F1 resmi dari penelitian." |
| 1:00 | **Langkah 3** — grafik biru/merah | "Biru data lapangan asli. Merah data yang sama setelah sensor S2 kami beri empat kerusakan: naik pelan (drift), lonjakan, geser, dan nilai jatuh/macet." |
| 2:00 | **Langkah 4** — kurva entropi | "Sinyal diubah jadi 60 angka keteraturan pola. Kerusakan menggeser kurvanya." |
| 3:00 | **Langkah 5** — tabel tebakan | "Jawaban benar: window asli *fault free*, window rusak *faulty*. Satu contoh bisa benar bisa salah; peluang dekat 50 % artinya model ragu." |
| 4:00 | **Langkah 6** — uji banyak window | "Ini yang adil: 30 potongan acak. Di run uji 2026-11-20 kedua metode benar 80 % (24 dari 30)." |
| 4:30 | (opsional) **Langkah 7** | "Kalau ingin lihat proses training: ini versi mini, loss turun tiap iterasi Levenberg–Marquardt." |

## 5. Cara mengganti contoh (sel *Konfigurasi*)

Ubah salah satu baris ini, lalu **Run all** lagi:

```python
HARI_MULAI = 10                  # potongan data mulai hari ke-berapa (0–95)
SENSOR_RUSAK = ["S2"]            # contoh lain: ["S1", "S3"] atau ["S1","S2","S3","S4"]
FAULT = ["drift", "spike", "bias", "hardware"]   # satu atau kombinasi
SEED = 1                         # ganti angka = pola kerusakan acak yang lain
```

Dua hal soal `HARI_MULAI`:

- **Drift** hanya ada di tiga episode 3 hari (hari 9,8–12,8 · 39,1–42,1 ·
  68,5–71,5), persis seperti saat training. Kalau memilih `drift`, pakai hari
  **10** atau **69**. Di luar itu notebook menulis "di luar episode drift".
- **Hindari hari 1–3, 17–18, 33–49, dan 77–83.** Di file `data_sensor.csv`
  bagian itu berupa garis lurus sempurna (keempat sensor turun/naik lurus),
  jadi grafiknya membingungkan. Lihat catatan di bagian 8.

Arti jenis kerusakan:

| Nama | Artinya di sensor nyata |
|---|---|
| `drift` | nilai pelan-pelan naik (sensor bergeser, perlu kalibrasi ulang) |
| `spike` | lonjakan sesaat (gangguan listrik/derau) |
| `bias` | nilai bergeser tetap (offset) |
| `hardware` | nilai macet atau hilang (kabel/modul bermasalah) |

Jumlah jenis kerusakan menentukan model yang dipakai: 1 jenis → model S2,
2 → S3, 3 → S4, 4 → S5. Model S1 (normal vs rusak) selalu ikut.

## 6. Cara membaca hasil

- **Tebakan** — jawaban model. Benar kalau window *asli* = `fault free` dan
  window *diberi fault* = `faulty` / nama kerusakannya.
- **Yakin** — peluang tebakan (50 % pada 2 kelas = ragu, 90 % = sangat yakin).
- **Sensor ditebak rusak** — sensor dengan skor di atas 0,5.
- **Akurasi** di Langkah 6 — persentase benar dari banyak window acak.

## 7. Kalau model salah tebak (ini normal, jangan panik)

Hasil contoh default (hari 10, empat fault di S2) pada run uji 2026-11-20:
6 dari 8 tebakan benar (S1-EDM benar dua-duanya, S5-JSD benar dua-duanya;
S1-JSD menyebut window asli *faulty*, S5-EDM melewatkan window rusak).
Contoh ini **tidak** dipilih supaya hasilnya bagus — dipilih karena keempat
kerusakan terlihat jelas di grafik.

Model **tidak sempurna**, dan itu sudah dilaporkan jujur di penelitian:

| Skenario (N = 700) | F1 EDM-Fuzzy | F1 JSD-Fuzzy |
|---|---|---|
| S1 normal vs rusak | 0,59 | 0,67 |
| S2 satu jenis kerusakan | 0,33 | 0,35 |
| S3 dua jenis | 0,18 | 0,20 |
| S4 tiga jenis | 0,34 | 0,43 |
| S5 empat jenis sekaligus | 0,80 | 0,87 |

Sumber: `hasil11/exports/11_rq4_performa_ann.csv` (notebook 11 v3).

Jawaban yang bisa dipakai:

- *"Satu contoh bisa salah. Ukuran yang benar adalah F1 dari cross-validation
  di tabel tadi, dan Langkah 6 yang menguji banyak window."*
- **Bias saja hampir tidak terdeteksi** — memang begitu: langkah *centroid
  shifting* di dalam entropi menghapus pergeseran tetap (AUC ≈ 0,50, temuan
  notebook 12). Itu batasan yang sudah ditulis, bukan bug.
- Kombinasi empat kerusakan (S5) paling mudah dikenali; dua kerusakan (S3)
  paling sulit.
- Tebakan dengan peluang dekat 50 % bisa berbalik kalau jalan ulang dengan
  seed lain: entropi memakai 128 titik acuan yang diambil acak (`n_ref`),
  jadi fiturnya sedikit berubah. Itu sebabnya yang dinilai adalah banyak
  window (Langkah 6) dan CV, bukan satu contoh.
- Akurasi Langkah 6 (sekitar 80 %) lebih tinggi dari F1 CV karena posisi
  potongannya juga dipakai saat melatih (kerusakannya baru). Notebook menulis
  catatan ini sendiri; angka resmi tetap F1 CV.

## 8. Pertanyaan yang mungkin muncul

**"Model ini sama dengan yang di penelitian?"** Rumus fitur, simulator
kerusakan, dan algoritma LM sama persis (disalin tanpa perubahan ke
`deteksi_fault_core.py`). Arsitektur dicari dengan grid yang sama, lalu model
dilatih di semua data skenario dan disimpan. Angka resmi tetap dari CV
notebook 11.

**"Kenapa window 700?"** Satu-satunya panjang window yang selisih F1 JSD vs
EDM-nya signifikan (p = 0,027).

**"Kenapa ada bagian grafik yang lurus sempurna?"** Di `data_sensor.csv`
ada empat bagian (total sekitar 26 dari 98 hari: hari 1–3, 17–18,6, 33–49,9,
77–83) yang nilainya berubah lurus sempurna di keempat sensor. Itu ada di file
data aslinya, bukan dari pengolahan di notebook. Kemungkinan besar bagian yang
ditambal saat pengambilan data — cek ke sumber data sebelum menjawab pasti.

**"Bisa dipakai di alat sungguhan?"** Bisa dalam bentuk ini: data 2,4 hari
terakhir → hitung 60 angka entropi → model tersimpan menebak. Klaim
real-time/edge tidak dibuat di penelitian.

## 9. Kalau ada masalah

| Gejala | Solusi |
|---|---|
| `Folder models/ tidak ditemukan` | Di Kaggle: panel kanan **Input → Add Input → Datasets → `model-deteksi-fault-tersimpan`**. Di laptop: pastikan folder `models/` ada di repo. |
| `Sumber data: unduh` lalu error | Internet mati. Pakai cadangan laptop (bagian 3). |
| `HARI_MULAI terlalu besar` | Isi angka 0–95. |
| `Kombinasi fault [...] tidak ada` | Pakai nama persis: `drift`, `spike`, `bias`, `hardware`. Semua 15 kombinasi tersedia. |
| `PERHATIAN: ... <= 1% sampel` | Potongan itu tidak cukup rusak (biasanya karena hanya `drift` di luar episode). Ganti `HARI_MULAI` ke 10 atau 69. |
| Sel lama sekali (> 3 menit) | Klik **Run → Restart & clear outputs**, lalu **Run all**. |

---

Dibuat 2026-11-20. Model: notebook 16 (`simpan_model.py`). Demo:
`build_sidang_demo_notebook.py` → `sidang/03_Demo_Model_Tersimpan.ipynb`.
