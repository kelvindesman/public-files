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
| 07 | `07_Skema_Diagram_CV_5_Skenario_dan_Sensor.ipynb` | Skema flowchart pembimbing dijalankan utuh dengan **data terbaru** dan **cross-validation**: 5 skenario, berapa ongkos komputasinya, sensor mana yang rusak? | Satu notebook mengikuti flowchart kotak per kotak (broker → sinkronisasi → segmentasi → EDM-Fuzzy 4T fitur → ANN-LM grid search), plus **§3b laporan validasi & pembersihan data**, **§11 dua uji kontrol** (data bersih & fault satu sensor), dan **§13 tanya-jawab pembimbing** |
| 08 | `08_JSD_Fuzzy_vs_EDM_Fuzzy_Paper_Q3.ipynb` | Kalau JSD-Fuzzy diuji di dalam kerangka paper (5 skenario S1..S5, 4T fitur, grouped CV), apakah tetap menang dari EDM-Fuzzy? | **JSD-Fuzzy F1 rata-rata 0,627 vs EDM 0,572** (selisih +0,056), tapi uji berpasangan **belum signifikan** (p ≥ 0,05) — jadi ditulis "unggul pada sebagian besar sel", bukan "lebih baik" |
| 09 | `09_ANN_LM_vs_Bayesian_vs_GA.ipynb` | Dari empat algoritma pelatihan ANN — **LM, Bayesian regularization, GA, Adam** — mana yang paling cocok untuk deteksi fault? | **BR dan LM setara, keduanya jauh di atas GA dan Adam.** F1 rata-rata 10 sel (5 skenario × 2 set fitur): BR 0,504 · LM 0,504 · GA 0,335 · Adam 0,321. Juara terbanyak: BR 5 sel, LM 4, Adam 1, GA 0. LM vs BR **tidak signifikan** (p = 0,99); LM/BR vs GA (p ≤ 0,003) dan vs Adam (p ≤ 0,04) **signifikan**. Jacobian analitik lolos self-check (rel err 6,9e-05), jadi Jacobian numerik **tidak** dipakai. Run Kaggle 112 menit, 0 Traceback |
| 10 | `10_JSD_Fuzzy_vs_EDM_Fuzzy_ANN_LM_Asli.ipynb` | **Permintaan pembimbing**: ulangi perbandingan JSD vs EDM (notebook 08), tapi pembelajaran ANN-LM diganti **LM asli dari paper** (bukan `lbfgs`) | **JSD-Fuzzy unggul di 11 dari 15 sel** (5 skenario × 3 window, run v3): rata-rata selisih F1 N=200 +0,038 · N=700 +0,035 · N=1000 +0,020 · pooled +0,031. Uji berpasangan **per window tidak signifikan** (p = 0,12 / 0,42 / 0,28; n=5 power rendah); **pooled 15 pasang signifikan p = 0,048** (t maupun Wilcoxon; deskriptif — ketergantungan antar-window beda). Protokol v3: inner CV 2, search 6 iter, refit 40, grid 16 kombinasi. Jacobian analitik lolos self-check (~1e-9). **Akuntansi CPU penuh**: TOTAL 93.740 detik (~26,0 jam-CPU), wall 573 mnt, 0 Traceback, 0 SKIP — tabel `10_ongkos_per_bagian.csv` |
| 11 | `11_LM_Asli_EDM_vs_JSD_Lengkap.ipynb` | Versi **lengkap** dari 10: EDM vs JSD dengan **semua** classifier LM asli, ditambah **identifikasi sensor** (RQ6) yang dirantai end-to-end dari prediksi out-of-fold | **JSD unggul di 12 dari 15 sel**, rata-rata +0,030; tapi **pooled p = 0,050** → statusnya tetap "belum signifikan". Sensor-ID: F1 **0,765 vs 0,721**, exact-match semua-4-benar **0,454 vs 0,363**. **Stabilitas fitur (CV): EDM lebih stabil** di 61–88% pasangan — dilaporkan apa adanya. Tidak ada `lbfgs`/`Adam`/`MLPClassifier`. Wall 588 mnt, **94.490 detik CPU (~26,2 jam-CPU)**, 0 Traceback, 0 SKIP |
| — | `Ilustrasi_Data_Akuisisi_dan_Fault.ipynb` | Penjelasan 10 langkah akuisisi (nasib missing value & nilai di luar 0–100 %) dan grafik sinyal asli vs setelah injeksi fault untuk semua kelas | **Tidak ada baris dibuang karena nilainya mencurigakan**: stempel ganda 0, NaN 0, luar range 0; pembacaan macet 556/654/659/657 kejadian dan pencilan \|z\|>6 13/0/0/0 hanya **dilaporkan**. Grafik: 5 panel + grid 16 kelas per sensor, NaN dibiarkan celah + versi tertambal. Kaggle `ilustrasi-data-akuisisi-dan-fault` v2 COMPLETE |

## Alur pemikirannya

```
01  metode mana yang terbaik           -> JSD-Fuzzy
02  bentuk datanya bagaimana           -> broker kumpulkan, JANGAN dirata-rata
03  seberapa akurat per tingkat        -> makin banyak fault bercampur, makin sulit
04  sensor mana yang rusak             -> dirantai setelah klasifikasi
05  jenis fault apa di sensor mana     -> peta sensor x jenis fault
06  apakah ANN penghambatnya           -> pembanding XGBoost
07  skema flowchart + cross-validation -> performa + biaya komputasi + sensor mana
08  JSD vs EDM di dalam kerangka paper -> dua set fitur pada pipeline yang sama
09  algoritma pelatihan mana yang menang -> BR & LM setara, di atas GA & Adam
10  ulangi 08 dengan LM asli paper      -> JSD unggul 11/15 sel, pooled p = 0,048
11  10 + identifikasi sensor (RQ6)      -> JSD unggul, tapi pooled p = 0,050
```

Catatan: `10` dan `11` menjalankan pipeline yang sama tetapi hasil pooled-nya
berbeda tipis (p = 0,048 vs 0,050). Keduanya harus ditulis apa adanya — batas
signifikansi 0,05 tidak dilampaui dengan aman, jadi klaim tetap deskriptif.

Notebook 04 dan 05 menjawab pertanyaan pembimbing secara berurutan: 04 menjawab
*"sensor mana yang fault"*, 05 menjawab *"bias-nya di sensor mana, drift-nya di
sensor mana"*.

## Notebook 07 — peta pertanyaan pembimbing → bagian notebook

| Pertanyaan | Dijawab di |
|---|---|
| Tabel identifikasi sensor itu artinya apa? `Prevalensi` vs `Akurasi` vs `P(rusak)` | **§0** (versi tanpa istilah) + **§13-Q1** |
| Bisa dites dengan data yang dikondisikan bersih (tanpa fault)? | **§11 Kontrol A** — FPR / spesifisitas + `P(rusak)` di window bersih vs fault |
| Bisa dites dengan fault hanya di satu sensor? | **§11 Kontrol B** — Top-1 vs tebak acak 25 % |
| Beda tabel "F1 rata-rata 4 sensor" dengan peta panas F1? | **§13-Q3** — angka sama, peta panas per sensor, tabel ringkas rata-ratanya |
| Preprocessing-nya per jam? | **§3** — bukan; **rata-rata per 5 menit** (281.721 → 28.173 baris) |
| Yang masuk broker sudah di-downsample? | **§13-Q5** — belum; broker terima 30 detik, perata-rataan sesudahnya |
| Ada validasi & pembersihan, bukan cuma resample? | **§3b** — 10 langkah + tabel `07_mutu_data.csv` |
| Panjang data 2000/7000/10000 atau 200/700/1000? | **§5** — tabel konversi; durasinya identik (0,69 / 2,43 / 3,47 hari) |
| Window yang tidak genap N sampel di-exclude? | **§5** — blok akuntansi window; `window_setengah_jadi` selalu 0 |

Tiga pengaman anti "model cuma menebak semua sensor rusak" di notebook 07:
subset sensor yang di-fault **diundi ulang tiap pengulangan** (+ sel pemeriksa
label kembar di §5), fitur tahap sensor dibuat **relatif terhadap rata-rata tiga
sensor lain**, dan **ambang keputusan dicari di data latih** tiap fold, bukan
dipatok 0,5.

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
2025-09-14 s.d. 2025-12-21, tanpa nilai kosong). Notebook `08`–`11` memakai
dataset yang sama. Loader-nya memakai file lokal kalau ada (termasuk
`/kaggle/input/...`), kalau tidak ada baru mengunduh dari GitHub.

## Menjalankan

**Di Kaggle** (cara yang dipakai selama ini):

```bash
kaggle kernels push -p <folder berisi kernel-metadata.json + notebook>
kaggle kernels status  kelvindsmn/<slug>
kaggle kernels output  kelvindsmn/<slug> -p ./hasil
```

Kernel yang sudah ada: `jsd-fuzzy-ann-paper` (01),
`jsd-fuzzy-broker-architecture-v2` (02), `jsd-fuzzy-scenario-ladder` (03),
`jsd-fuzzy-twostage-sensorid` (04), `jsd-fuzzy-cv-5skenario` (07),
`jsd-fuzzy-vs-edm-fuzzy-paper-q3` (08),
`ann-lm-vs-bayesian-vs-ga-paper-pmdc` (09),
`jsd-vs-edm-dengan-ann-lm-asli-paper-pmdc` (10),
`jsd-edm-ann-lm-lengkap-q3` (11).

**Di laptop:**

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy pandas scikit-learn scipy joblib matplotlib requests
jupyter notebook
```

Kecepatan diatur lewat variabel lingkungan `RUNTIME_PROFILE` (`fast` | `paper`).
Notebook `09` memakai `paper` sebagai default (LM 60 iterasi, BR 40 ronde,
GA 60×40, Adam 300 epoch); `fast` memperkecil separuhnya. Notebook `10` dan
`11` tidak memakai `RUNTIME_PROFILE` — protokolnya tetap (LM search 6 iterasi,
inner CV 2, refit 40). Dua pengaman waktu ikut bekerja: `MAX_TRAIN_S` (default
45 dtk per pelatihan — statusnya menjadi `budget(MAX_TRAIN_S)`, bukan
"konvergen") dan `KAGGLE_TIME_BUDGET_H` (default 10,5 jam untuk `08`–`10`,
11,5 jam untuk `11`, supaya sesi berhenti sendiri sebelum Kaggle mematikan
sesi di jam ke-12).

## Notebook dibangun dari skrip

Beberapa notebook dihasilkan oleh skrip supaya isinya bisa diulang persis:

| Skrip | Menghasilkan |
|---|---|
| `build_notebook.py` | `01_...` (gabungan dua notebook di `archive/`) |
| `build_cv_5skenario_notebook.py` | `07_...` |
| `build_twostage_notebook.py` | `04_...` |
| `build_faulttype_persensor_notebook.py` | `05_...` |
| `build_q3_jsd_vs_edm_notebook.py` | `08_...` |
| `build_09_ann_training_algorithms_notebook.py` | `09_...` (LM vs Bayesian vs GA vs Adam) |
| `build_10_jsd_vs_edm_ann_lm_asli_notebook.py` | `10_...` (08 diulang dengan LM asli paper) |
| `build_11_annlm_real_notebook.py` | `11_...` (10 + identifikasi sensor RQ6, tanpa `lbfgs`) |
| `add_headers.py` | menyisipkan header "apa yang dibuktikan" ke tiap notebook |

Kalau mengubah isi notebook-notebook itu, ubah **skripnya**, lalu jalankan
ulang — bukan mengedit `.ipynb`-nya langsung.
