# Deteksi Fault 4 Sensor Kelembaban — Multiscale Entropy + ANN

Riset perbandingan **EDM-Fuzzy** vs **JSD-Fuzzy** entropy sebagai fitur untuk
mendeteksi fault pada 4 sensor kelembaban, dengan classifier ANN (MLP).

Tiap notebook di sini membuktikan **satu hal**, dan namanya sudah menyebutkan
hal itu. Baca berurutan dari 01.

## Daftar notebook

| # | Notebook | Pertanyaan yang dijawab | Jawabannya |
|---|---|---|---|
| 01 | `01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb` | Metode entropy mana yang paling bagus jadi fitur? | **JSD-Fuzzy *rich*** — akurasi 17 kelas naik 0,34 → 0,42 dibanding JSD versi lama; unggul di keempat jenis fault |
| 02 | `02_Sensor_Digabung_Broker_Jangan_Dirata_rata.ipynb` | Setelah dikumpulkan broker, 4 sensor boleh dirata-rata jadi 1? | **Tidak.** 4 jalur F1 0,845 vs dirata-rata 0,797. Spike & hardware paling dirugikan karena waktunya beda per sensor |
| 03 | `03_Akurasi_per_Tingkat_Kombinasi_Fault.ipynb` | Kalau dipecah menurut banyaknya fault yang bercampur, seberapa akurat? | **JSD-Fuzzy menang di 4 dari 5 tingkat** (S1 0,80 · S2 0,68 · S3 0,79 · S4 0,78 · S5 0,97) |
| 04 | `04_Klasifikasi_Dulu_Baru_Sensor_Mana.ipynb` | Setelah tahu ada fault, sensor **mana** yang rusak? | **Bisa, dan dirantai dari hasil klasifikasi.** Dijalankan untuk 3 bentuk: biner, 5 kelas, dan 17 kelas |
| 05 | `05_Fault_Jenis_Apa_di_Sensor_Mana.ipynb` | Bias-nya di sensor mana, drift-nya di sensor mana? | **Bisa** — 16 label (4 sensor × 4 jenis fault), hasilnya berupa peta sensor × jenis fault |
| 06 | `06_ANN_vs_XGBoost.ipynb` | Batasnya di classifier atau di fiturnya? | Pembanding XGBoost untuk menguji apakah ANN yang jadi penghambat |
| 07 | `07_CV_5_Skenario_Performa_Komputasi_dan_Sensor.ipynb` | Pakai **data terbaru** dan **cross-validation**: bagaimana perbandingan 5 skenario, berapa ongkos komputasinya, dan sensor mana yang rusak? | Satu notebook berisi tiga tabel: performa (akurasi/precision/recall/F1 ± std antar-fold), biaya komputasi (CPU, memori, waktu, latensi inferensi), dan identifikasi sensor rusak |

## Alur pemikirannya

```
01  metode mana yang terbaik           -> JSD-Fuzzy
02  bentuk datanya bagaimana           -> broker kumpulkan, JANGAN dirata-rata
03  seberapa akurat per tingkat        -> makin banyak fault bercampur, makin sulit
04  sensor mana yang rusak             -> dirantai setelah klasifikasi
05  jenis fault apa di sensor mana     -> peta sensor x jenis fault
06  apakah ANN penghambatnya           -> pembanding XGBoost
07  data terbaru + cross-validation    -> performa + biaya komputasi + sensor mana
```

Notebook 04 dan 05 menjawab pertanyaan pembimbing secara berurutan: 04 menjawab
*"sensor mana yang fault"*, 05 menjawab *"bias-nya di sensor mana, drift-nya di
sensor mana"*.

## Penting: satu hasil lama yang TIDAK sah

`archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb` (dulu
"Per-Sensor Cara A") pernah melaporkan **F1 0,925** untuk identifikasi sensor.
**Angka itu tidak boleh dipakai.** Injeksi fault-nya mengenai keempat sensor
sekaligus, sehingga keempat kolom label identik (prevalensi 0,882 di semua
sensor) dan ROC-AUC-nya 0,50 — setara tebak acak. Angka tinggi itu murni efek
"selalu menebak fault". Penggantinya yang sah adalah notebook `04`.

## Isi folder `archive/`

| File | Kenapa diarsipkan |
|---|---|
| `LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb` | hasilnya tidak sah (lihat atas) |
| `SUMBER_01a_Paper_Q3_RQ1_RQ5.ipynb` | sudah digabung ke notebook `01` |
| `SUMBER_01b_Dev_JSD_Rich.ipynb` | sudah digabung ke notebook `01` |
| `LAMA_EDM_v6_dengan_downsampling.ipynb` | percobaan awal, hanya EDM-Fuzzy |
| `LAMA_EDM_v6_tanpa_downsampling.ipynb` | varian tanpa downsampling/windowing |
| `LAMA_Revisi2_Deteksi_per_Sensor.ipynb` | pipeline generasi awal |
| `LAMA_Anchor_Cepat.ipynb` | alat bantu pembanding, tidak dipakai lagi |

## Data

Notebook `01`–`06` mengunduh sendiri `tabel_sensor4_generated.csv` dari GitHub
saat dijalankan, jadi **internet harus aktif** (di Kaggle: Settings → Internet →
On). Kolomnya `kelembaban1..kelembaban4`.

Notebook `07` memakai dataset terbaru **`data_sensor.csv`** (281.721 baris,
2025-09-14 s.d. 2025-12-21, tanpa nilai kosong). Loader-nya memakai file lokal
kalau ada (termasuk `/kaggle/input/...`), kalau tidak ada baru mengunduh dari
GitHub.

## Menjalankan

**Di Kaggle** (cara yang dipakai selama ini):

```bash
kaggle kernels push -p <folder berisi kernel-metadata.json + notebook>
kaggle kernels status  kelvindsmn/<slug>
kaggle kernels output  kelvindsmn/<slug> -p ./hasil
```

Kernel yang sudah ada: `jsd-fuzzy-ann-paper` (01),
`jsd-fuzzy-broker-architecture-v2` (02), `jsd-fuzzy-scenario-ladder` (03),
`jsd-fuzzy-twostage-sensorid` (04).

**Di laptop:**

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn scipy joblib matplotlib requests
jupyter notebook
```

Kecepatan diatur lewat variabel lingkungan `RUNTIME_PROFILE` (`fast`, default,
~15–25 menit di Kaggle · `paper`, beberapa jam) dan `KAGGLE_TIME_BUDGET_H`
(pengaman supaya pekerjaan berhenti sendiri sebelum Kaggle mematikan sesi di
jam ke-12).

## Notebook dibangun dari skrip

Beberapa notebook dihasilkan oleh skrip supaya isinya bisa diulang persis:

| Skrip | Menghasilkan |
|---|---|
| `build_notebook.py` | `01_...` (gabungan dua notebook di `archive/`) |
| `build_twostage_notebook.py` | `04_...` |
| `build_faulttype_persensor_notebook.py` | `05_...` |
| `add_headers.py` | menyisipkan header "apa yang dibuktikan" ke tiap notebook |

Kalau mengubah isi notebook-notebook itu, ubah **skripnya**, lalu jalankan
ulang — bukan mengedit `.ipynb`-nya langsung.
