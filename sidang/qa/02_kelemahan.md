# 02 — Daftar kelemahan (disusun sebelum penguji menemukannya)

Disusun dari kode dan hasil mentah, dikelompokkan menurut jenis ancaman
validitas. Tiap butir: **apa kelemahannya**, **bukti** (file:baris), dan
**jawaban 2 kalimat** (akui → dampak → mitigasi). Angka yang disebut harus
sudah berstatus COCOK di `01_klaim_bukti.csv`.

Status: **[DIKETAHUI]** sudah ditulis di dokumen penelitian; **[BARU]**
ditemukan saat menyusun bahan ini dan belum ada di laporan.

---

## A. Validitas internal (apakah perbandingannya adil dan bebas bocor?)

**A1. Kebocoran antar grup CV yang bersebelahan [BARU]**
- Bukti: `build_11_annlm_real_notebook.py:601` `groups = starts // WIN`,
  stride = WIN/2 (`:557`). Window yang mulai di WIN/2 (grup 0) mencakup
  [WIN/2, 1,5·WIN] dan tumpang-tindih 50 % dengan window yang mulai di WIN
  (grup 1). Grup tetangga bisa jatuh ke fold berbeda.
- Jawaban: "Pengelompokan per blok waktu mencegah window yang sama persis bocor,
  tetapi window di batas blok masih berbagi setengah sampelnya dengan blok
  sebelah, jadi skor absolutnya sedikit optimistis. Kebocoran ini berlaku sama
  untuk EDM dan JSD karena keduanya memakai window, grup, dan fold yang
  identik, sehingga **selisih** keduanya tetap sah; perbaikannya adalah memberi
  celah (purge) satu window di setiap batas fold."
- Tindak lanjut: kalau ada waktu, jalankan ulang satu sel (mis. N=700) dengan
  purge dan laporkan selisih F1.

**A2. Satu sinyal bersih dipakai berulang di semua kondisi**
- Bukti: `segment_and_label` (`build_11…:556–595`) menyuntik fault ke salinan
  data yang sama untuk tiap kondisi; grup = blok waktu yang sama lintas kondisi.
- Jawaban: "Semua kelas berasal dari latar sinyal yang sama, jadi model tidak
  bisa membedakan kelas dari 'hari yang berbeda'; ini justru memaksa model
  belajar dari pola fault. Karena grup CV dihitung dari posisi waktu, blok
  yang sama di semua kondisi selalu masuk fold yang sama."

**A3. Anggaran waktu latih memotong sebagian fold**
- Bukti: `LM_SEARCH_BUDGET_S = 40`, `LM_REFIT_BUDGET_S = 150`
  (`build_11…:1613–1614`); `hasil14/14_draf_laporan.md` "beberapa fold berhenti
  karena batas waktu latih".
- Jawaban: "LM dibatasi waktu per fit supaya 15 sel muat di Kaggle; batasnya
  sama untuk kedua metode dan dimensi fiturnya sama (60), jadi tidak ada metode
  yang diberi anggaran lebih. Status henti setiap fold tercatat di log, sehingga
  bisa ditunjukkan berapa yang terpotong."
- Siapkan: jumlah fold `budget(MAX_TRAIN_S)` per metode dari log (lihat
  `01_klaim_bukti.csv`).

**A4. Pencarian arsitektur memakai iterasi pendek**
- Bukti: `LM_ITERS_SEARCH = 6`, `LM_ITERS_REFIT = 40` (`build_11…:1610–1611`).
- Jawaban: "LM konvergen cepat, jadi 6 iterasi cukup untuk mengurutkan kandidat;
  model akhir dilatih ulang 40 iterasi. Prosedurnya identik untuk kedua metode."

**A5. Bobot awal tersusun lain dari maksudnya (`lm_pack` vs `lm_unpack`) [BARU]**
- Bukti: `deteksi_fault_core.py:452–462`. `lm_pack` menyusun semua W lalu semua
  b; `lm_unpack` membaca W1, b1, W2, b2. Latih dan prediksi sama-sama lewat
  `lm_unpack`, jadi konsisten; yang berubah hanya bobot awal.
- Jawaban: "Ada ketidakcocokan urutan saat inisialisasi, sehingga sebagian bobot
  awal tidak persis mengikuti Xavier; pelatihan dan prediksi tetap konsisten
  karena memakai urutan yang sama. Karena EDM dan JSD memakai fungsi yang sama
  dengan seed yang sama, dampaknya tidak memihak satu metode; perbaikannya satu
  baris dan hasil uji ulang bisa dilampirkan."
- Detail angka: `04_walkthrough_kode.md` bagian "Cacat yang diketahui".

**A6. Parameter fault NB14 dipilih setelah melihat NB13 [DIKETAHUI, sebagian]**
- Bukti: `hasil14/14_draf_laporan.md` §Catatan — tingkat "sedang" diambil dari
  notebook 13, dan tab "Perubahan program" di dokumen usulan tidak tersedia.
- Risiko: penguji bisa bertanya apakah tingkat fault dipilih supaya JSD menang.
- Jawaban: "Tingkat 'sedang' adalah titik tengah kisi intensitas di notebook 13,
  ditetapkan sebelum notebook 14 dijalankan dan tidak diubah setelahnya. Hasil
  utama skripsi tetap notebook 11; notebook 14 saya laporkan sebagai uji
  tambahan, lengkap dengan sel di mana EDM menang (N=700 S4, S5; N=1000 S3)."

## B. Validitas konstruk (apakah yang diukur memang yang dimaksud?)

**B1. Fault simulasi, bukan kerusakan nyata [DIKETAHUI]** — Q10 `Bahan_Sidang.md`.
Jawaban siap: label pasti dan bisa diulang; berlaku untuk fault yang dimodelkan.

**B2. Entropi buta terhadap bias konstan karena centroid shifting [DIKETAHUI]**
- Bukti: `embed_matrix_centered` (`deteksi_fault_core.py:274`); NB12/13: AUC
  bias ≈ 0,50 berapa pun besarnya.
- Jawaban: "Ini keterbatasan representasi, bukan besarnya fault: rata-rata tiap
  vektor dikurangkan sebelum entropi dihitung, jadi offset hilang. Saran saya
  menambah fitur level (rata-rata window atau selisih antar sensor)."

**B3. Drift kecil tidak terlihat [DIKETAHUI]** — NB13: AUC ≤ 0,70 untuk total
2–9 poin; NB15 (tanpa entropi) drift tunggal F1 0,49–0,71. Jawaban: drift
realistis berada di bawah ayunan alami sinyal dalam satu window.

**B4. Bentuk fungsi keanggotaan berbeda dari rumus buku [perlu dijelaskan]**
- Bukti: `phi_euclidean` / `phi_jsd`: `mu = 1/(1 + d/r)` (`build_11…:706,756`),
  bukan `exp(-d^n/r)` seperti fuzzy entropy Chen dkk.
- Jawaban: "Saya memakai fungsi keanggotaan rasional yang turun monoton, sama
  untuk EDM dan JSD, sehingga satu-satunya perbedaan antar metode tetap pada
  ukuran jarak. Penurunannya ada di `03_algoritma_dengan_tangan.md`."

**B5. JSD dihitung dari vektor pendek (m = 2) yang diubah jadi sebaran 8 bin**
- Bukti: `vectors_to_distributions` (`deteksi_fault_core.py:317`), `JSD_BINS=8`.
- Pertanyaan yang mungkin muncul: "Sebaran dari dua angka itu bermakna?"
- Jawaban: "Dua angka tidak cukup untuk histogram biasa, karena itu dipakai soft
  binning dengan kernel halus selebar `h`, sehingga tiap vektor menjadi sebaran
  halus atas 8 pusat bin. Sensitivitas jumlah bin dan lebar kernel diuji di
  notebook 12 (rekomendasi `bins=4`, `h_ratio=1,0`)."

**B6. Asal-usul data [PERLU DIPASTIKAN SEBELUM SIDANG]**
- Pengamatan: baris awal `data_sensor.csv` bernilai pecahan yang naik rata
  (`id` 1,0 → 1,875 → 2,75), tanda hasil interpolasi ke grid 30 detik.
  `tabel_sensor4_generated.csv` bernama "generated" dan disebut Bu Luh "data
  buatan".
- Penguji bisa bertanya "data ini asli atau buatan?". Jawabannya harus pasti:
  dari mana sensor diambil, siapa yang menginterpolasi, dan apa yang
  "generated" pada tabel kedua. **Tanyakan ke Bu Luh**; jangan mengarang.

## C. Validitas kesimpulan statistik

**C1. Pooled NB11 tidak signifikan (p = 0,076; Wilcoxon 0,107) [DIKETAHUI]**
— klaim yang dipakai "konsisten", bukan "terbukti". Q39–Q41 `Bahan_Sidang.md`.

**C2. Uji gabungan memakai 15 sel yang tidak saling bebas [BARU]**
- Lima skenario yang sama muncul di tiga ukuran window, jadi 15 pasangan bukan
  15 pengamatan independen (pseudo-replikasi). Uji per N hanya punya n = 5
  (df = 4) — dayanya rendah.
- Jawaban: "Uji gabungan saya baca sebagai indikasi arah, bukan bukti kuat,
  karena sel-selnya berbagi skenario; uji per ukuran window yang lebih tepat
  hanya punya lima pasangan sehingga dayanya rendah. Karena itu saya
  menekankan konsistensi arah dan ukuran efek, bukan p-value."

**C3. Banyak uji tanpa koreksi perbandingan berganda [BARU]**
- NB11: 3 uji per N + 1 gabungan; NB14 menambah 4 lagi.
- Jawaban: "Dengan koreksi Bonferroni untuk 4 uji (α = 0,0125), hanya NB14
  gabungan p = 0,0006 yang tetap signifikan; N=700 NB11 (p = 0,027) tidak.
  Itu sejalan dengan klaim saya yang sudah berhati-hati."

**C4. Variasi antar seed tidak dilaporkan**
- Setiap sel satu seed; tidak ada selang kepercayaan antar ulangan.
- Jawaban: "Variasi yang saya laporkan adalah antar fold CV; ulangan multi-seed
  belum dilakukan karena satu run penuh butuh ±8–10 jam Kaggle. Ini saya
  masukkan sebagai saran."

**C5. EDM lebih stabil (56/63 pasangan) [DIKETAHUI]** — kalimat siap pakai no. 3.

**C6. NB10 dan NB11 memberi p berbeda untuk pipeline yang sama [DIKETAHUI]**
— README: beda protokol CV dan anggaran komputasi. Jawaban: keduanya dilaporkan
apa adanya; perbedaan itu sendiri menunjukkan hasil gabungan belum stabil.

## D. Validitas eksternal

**D1.** Satu lokasi, 4 sensor, ±98 hari; sensor kapasitif murah.
**D2.** Window 16,7 jam → pemantauan harian, bukan alarm real-time (Q43).
**D3.** S2–S4 tetap sulit (F1 0,21–0,53 di NB14) — model belum siap memilah
kombinasi fault.
**D4.** FPR kelas normal ±0,13 (NB13) — penyiraman/pengeringan alami menaikkan
entropi. Saran: konteks penyiraman/cuaca.

## E. Notebook 15 (data hipotetik, tanpa entropi)

**E1.** Drift tunggal F1 0,49 (4 sensor) / 0,71 (rata-rata).
**E2.** Pembagian data berbeda dari program pertama (blok harian 3:1:1, bukan
60/20/20 berurutan) — alasannya: bagian uji 60/20/20 tidak memuat episode.
**E3.** FPR sensor yang tidak disuntik fault ±6 %.
**E4.** S1 kebetulan memilih jenis fault berbeda di dua versi (seed berbeda).
**E5.** Fitur hanya lag; tidak memakai sensor tetangga — karena itu bias/drift
lemah. Bukan bagian klaim utama skripsi.

---

## Prioritas sebelum sidang

1. B6 — pastikan asal-usul data (tanya Bu Luh).
2. A1 dan A5 — siapkan jawaban; jika sempat, uji ulang satu sel dengan purge
   dan dengan `lm_pack` diperbaiki untuk menunjukkan dampaknya kecil.
3. C2/C3 — hafalkan kalimat Bonferroni.

---

## F. Temuan tambahan dari pemeriksaan kode baris per baris [BARU]

Semua dibuktikan oleh `cek_cacat_kode.py` (jalankan:
`./venv/bin/python sidang/qa/cek_cacat_kode.py`). Detail di
`04_walkthrough_kode.md` bagian "Cacat yang diketahui".

**F1. Tahap identifikasi sensor memakai split acak, bukan grouped**
- Bukti: `build_11_annlm_real_notebook.py:2065`
  `train_test_split(..., test_size=0.3, random_state=RANDOM_SEED)`.
- Dampak: window tumpang-tindih bisa masuk latih dan uji sekaligus → F1 sensor-ID
  (0,675 vs 0,634; NB14 0,693 vs 0,791) optimistis. Berlaku sama untuk EDM dan
  JSD.
- Jawaban: "Tahap klasifikasi memakai grouped CV, tetapi tahap identifikasi
  sensor memakai pembagian acak 70/30, sehingga angka F1 sensornya lebih
  optimistis daripada tahap pertama. Perbandingan EDM–JSD tetap memakai
  pembagian yang sama; perbaikannya mengganti split itu dengan GroupShuffleSplit
  berdasarkan blok waktu."

**F2. Bobot awal: lapisan terakhir mulai dari nol (lanjutan A5)**
- `lm_pack`/`lm_unpack` berbeda urutan → untuk jaringan 2 lapisan tersembunyi,
  100 % bobot lapisan keluaran mulai dari nol, sehingga pada iterasi pertama
  kolom Jacobian lapisan-lapisan awal bernilai nol; 1 lapisan dengan 2 keluaran:
  50 %. Hanya W1 yang benar-benar Xavier.
- Semua jalur (latih, prediksi, `SavedLM`, `LMSensorID`, `.npz` NB15) memakai
  urutan `lm_unpack` → konsisten, prediksi model tersimpan identik.
- Teks skripsi **jangan** menulis "inisialisasi Xavier dengan bias nol".
- Jawaban: lihat A5 — sama untuk kedua metode; perbaikan satu baris; perlu run
  ulang untuk angka baru.

**F3. Langkah LM yang ditolak tetap menghabiskan jatah iterasi**
- `lm_fit`: tiap penolakan λ×2 dan iterasi tetap terhitung; pencarian arsitektur
  hanya 6 iterasi (`LM_ITERS_SEARCH`). Pada contoh mini dibutuhkan 7 penolakan
  sebelum langkah pertama diterima, jadi pada pencarian sebagian kandidat bisa
  hampir tidak bergerak dari bobot awal (risiko; belum diukur di data asli —
  cek `lm_status`/`iters` di log).
- Jawaban: "Pemilihan arsitektur memakai iterasi pendek; kandidat yang belum
  bergerak dinilai rendah sehingga pemilihan cenderung ke arsitektur yang
  cepat konvergen. Prosedur ini identik untuk kedua metode."

**F4. Simulator NB11: hardware 'macet' efektif 3 %, bukan 8 %**
- `simulate_hardware_fault` (`deteksi_fault_core.py:135–146`): titik 'hilang'
  (rv < 0,05) adalah subset titik 'macet' (rv < 0,08) lalu ditimpa NaN.
  Total titik rusak tetap 8 %: 5 % NaN + 3 % macet.
- Jawaban: "Total proporsi titik rusak tetap 8 %, tetapi komposisinya 5 % hilang
  dan 3 % macet, bukan 8 % + 5 %. Deskripsi di laporan saya perbaiki."

**F5. Simulator NB11: drift, spike, bias tidak memakai seed**
- Posisi spike identik untuk semua seed dan semua sensor (periodik tiap
  1/p sampel). NB14 sudah memakai posisi acak ber-seed.
- Jawaban: "Pada NB11 spike sengaja periodik sehingga posisinya sama di semua
  sensor; variasi antar sensor hanya dari subset sensor yang disuntik. NB14
  memperbaikinya dengan posisi acak."

**F6. Wilcoxon dengan 5 pasangan tidak mungkin p < 0,05**
- Uji eksak dua sisi dengan n = 5: p minimum = 2/2⁵ = 0,0625. Jadi "Wilcoxon
  tidak signifikan" per ukuran window bukan bukti tidak ada efek.
- Jawaban: "Per ukuran window Wilcoxon hanya punya lima pasangan, sehingga
  secara matematis p terkecilnya 0,0625; karena itu saya membaca uji-t dan
  ukuran efek, dan menyebut hasilnya deskriptif."

**F7. Penyimpangan dari rumus buku (harus bisa dijelaskan, bukan bug)**
- Jarak Euclidean, bukan Chebyshev (Chen 2007).
- Fungsi keanggotaan 1/(1+d/r), bukan exp(−dⁿ/r).
- Coarse-graining komposit (Wu 2013), bukan satu offset (Costa).
- Titik acuan acak n_ref = 128, set berbeda untuk m dan m+1.
- JSD soft-histogram mengabaikan urutan nilai di dalam vektor (pola naik dan
  turun dianggap sama) → entropi JSD-Fuzzy bisa bernilai negatif
  (contoh: −0,115 pada deret 8 titik).
  Bukti kode: `vectors_to_distributions` menjumlahkan kernel atas elemen vektor
  (`K.sum(axis=1)`), jadi template [−1, 1] dan [1, −1] memberi JSD = 0 persis
  (dicek 2026-11-20), sedangkan Euclidean memberi d = 2,83. JSD-Fuzzy mengukur
  kemiripan *sebaran nilai*, bukan *bentuk pola*; tulis ini di bab metode.
- Argumen `r_ratio` pada `jsd_fuzzy_entropy_1d` tidak dipakai.
- Rujukan rumus dan contoh angka: `03_algoritma_dengan_tangan.md`.

## Prioritas sebelum sidang (diperbarui)

1. B6 — asal-usul data (tanya Bu Luh).
2. F1 dan A1 — kebocoran: siapkan jawaban; idealnya uji ulang satu sel
   dengan GroupShuffleSplit + purge.
3. F2/A5 — hapus klaim "Xavier dengan bias nol" dari teks; siapkan jawaban.
4. F4 — perbaiki deskripsi hardware di laporan (5 % hilang + 3 % macet).
5. C2/C3/F6 — hafalkan kalimat statistik.
