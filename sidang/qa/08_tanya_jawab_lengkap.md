# 08 — Tanya-jawab lengkap, tiga lapis

Satu dokumen untuk latihan dan untuk catatan internal notebook
`sidang/04_Sidang_Interaktif.ipynb`. Tiap bagian diawali penanda
`<!-- bagian: nama -->`; builder notebook menyalin isi bagian itu ke sel
catatan internal yang sesuai, jadi **ubah di sini, lalu bangun ulang notebook**
(`./venv/bin/python build_sidang_interaktif_notebook.py`).

Tiap pertanyaan punya tiga lapis:

- **30 detik** — kalimat yang diucapkan di depan penguji.
- **Untuk mahasiswa baru** — penjelasan dari nol, dengan contoh angka kecil
  yang bisa dihitung di kertas.
- **Bukti** — tempat membuka angka/kode di layar (notebook §, file:baris,
  CSV, pustaka). Angka hanya boleh disebut kalau berstatus COCOK di
  `01_klaim_bukti.csv` atau dibaca dari CSV di `hasil_uji/`.

Sumber yang digabung: `Bahan_Sidang.md` (Q1–Q48), `05_bank_pertanyaan.md`
(P1–P33), `02_kelemahan.md` (A–F), `03_algoritma_dengan_tangan.md`,
`07_pustaka_dan_kebaruan.md`, `hasil_uji/statistik_sel.csv`. Nomor asal
ditulis di akhir judul, misalnya (Q1, P9).

---

<!-- bagian: pembuka -->

**T1. Bagaimana memakai notebook ini saat sidang?**
- *30 detik:* "Notebook ini memuat model yang sudah dilatih dan fitur yang
  sudah dihitung, jadi setiap sel selesai di bawah dua detik. Bapak/Ibu bisa
  memilih hari, sensor, jenis gangguan, dan intensitasnya, lalu melihat sinyal,
  kurva entropi, dan tebakan model berubah."
- *Untuk mahasiswa baru:* melatih ulang model butuh jam (NB11 di Kaggle
  ±10 jam). Saat presentasi kita tidak melatih; kita **memuat** bobot jaringan
  yang sudah tersimpan, lalu hanya melakukan perkalian matriks (forward pass).
  Itu seperti memakai kalkulator yang rumusnya sudah jadi.
- *Bukti:* `sidang/bukti_demo/04_Sidang_Interaktif.waktu.csv` (13 sel kode,
  total 9,7 s, sel terlama 1,7 s); `sidang/qa/test_notebook_sidang.py`
  (540 kombinasi panel tanpa galat, tiap panggilan < 1 s).

**T2. Apakah model di notebook ini sama dengan model penelitian?**
- *30 detik:* "Sama persis. Bobotnya diekspor dari model tersimpan N = 700 dan
  prediksinya dibandingkan dengan model asli: selisih maksimum nol."
- *Untuk mahasiswa baru:* model asli disimpan sebagai objek Python yang butuh
  scikit-learn (lambat diimpor, ±3–6 detik). Kita ambil angka-angkanya saja
  (bobot W, bias b, rata-rata dan skala normalisasi) ke file `.npz`, lalu
  menulis ulang forward pass dengan numpy. Supaya yakin tidak ada salah salin,
  keluaran kedua versi dibandingkan pada semua jendela uji.
- *Bukti:* `sidang/qa/buat_cache_interaktif.py` fungsi `ekspor_model`
  (assert selisih < 1e-9, terukur 0,00e+00); uji
  `test_bobot_numpy_sama_dengan_model_tersimpan`.

**T3. Kenapa catatan ini tidak tampil saat presentasi?**
- *30 detik:* "Catatan internal hanya untuk persiapan saya. Ada tombol mode
  presentasi dan ada salinan presentasi yang sel catatannya dibuang otomatis."
- *Bukti:* `sidang/qa/ekspor_notebook_sidang.py` (gagal kalau masih ada sel
  bertanda `catatan-internal`); `sidang/04_Sidang_Interaktif_presentasi.ipynb`.

<!-- bagian: masalah -->

**T4. Penelitian ini tentang apa? (Q1)**
- *30 detik:* "Mendeteksi gangguan pada empat sensor kelembaban tanah dari
  pola sinyalnya. Sinyal dipotong menjadi jendela, tiap sensor diringkas dengan
  entropi multiskala, lalu jaringan saraf yang dilatih Levenberg–Marquardt
  menebak jenis gangguan dan sensor yang rusak. Yang saya uji: mengganti ukuran
  kemiripan Euclidean (EDM-Fuzzy) dengan Jensen–Shannon (JSD-Fuzzy)."
- *Untuk mahasiswa baru:* bayangkan empat termometer di satu kebun. Kalau satu
  mulai "melantur", angka rata-ratanya mungkin tetap wajar, tetapi *polanya*
  berubah: terlalu rata (macet), tiba-tiba melonjak (spike), atau pelan-pelan
  bergeser (drift). Entropi adalah angka yang mengukur seberapa "rumit" pola
  itu. Jaringan saraf belajar dari 60 angka entropi (4 sensor × 15 skala) untuk
  memutuskan "normal atau rusak, dan rusak jenis apa".
- *Bukti:* NB11 §0; `deteksi_fault_core.py` (pipeline lengkap).

**T5. Apa kebaruannya? (Q5)**
- *30 detik:* "Tempat JSD dipasang: sebagai jarak antar pasangan template di
  dalam fuzzy entropy. Penelusuran pustaka tidak menemukan karya yang sama;
  yang ada memakai jarak geometris di dalam entropi, atau memakai JSD pada level
  seluruh sinyal."
- *Untuk mahasiswa baru:* fuzzy entropy membandingkan setiap potongan pendek
  sinyal (template) dengan potongan lain. Biasanya "beda" diukur dengan
  penggaris (jarak Euclidean). Di sini setiap potongan diubah dulu menjadi
  histogram kecil, lalu dua histogram dibandingkan dengan JSD — ukuran beda dua
  sebaran peluang.
- *Bukti:* `07_pustaka_dan_kebaruan.md` (b) tabel prior-art dan log pencarian
  (f); sumber baseline Zhou dkk. 2021, IEEE TII, doi:10.1109/TII.2020.3009139.

**T6. EDM-Fuzzy itu buatan Anda?**
- *30 detik:* "Bukan. EDM-Fuzzy dari Zhou dkk. 2021. Saya memakainya sebagai
  pembanding dan mengganti hanya ukuran jaraknya."
- *Bukti:* `pustaka.bib` kunci `zhou2021edmfuzzy`.

**T7. Kenapa empat sensor, bukan satu, dan kenapa tidak dirata-rata? (Q2, Q13)**
- *30 detik:* "Rata-rata menyembunyikan sensor mana yang rusak dan menurunkan
  skor: F1 0,821 untuk empat kanal terpisah vs 0,748 setelah dirata-rata."
- *Untuk mahasiswa baru:* kalau sensor A naik 10 dan sensor B turun 10,
  rata-ratanya tidak berubah — gangguannya hilang dari pandangan.
- *Bukti:* NB02.

<!-- bagian: data -->

**T8. Data dari mana, berapa banyak? (Q6, Q7)**
- *30 detik:* "Empat sensor kelembaban tanah, 14 September–21 Desember 2025,
  sampel tiap 30 detik, 281.721 baris. Saya rata-ratakan per 5 menit menjadi
  28.173 baris, 97,8 hari."
- *Untuk mahasiswa baru:* 30 detik → 5 menit artinya setiap 10 angka diganti
  rata-ratanya. Contoh: 40, 41, 40, 42, 41, 40, 41, 40, 42, 41 → 40,8. Derau
  kecil teredam, pola harian tetap.
- *Bukti:* NB11 §1–§2; `data_sensor.csv`.

**T9. Nilai nol di S4 diapakan? (Q9)**
- *30 detik:* "Tidak dibuang, dilabeli normal, dan dilaporkan sebagai batasan:
  11.371 sampel (4,0 %), tiga blok macet di 16–21 Desember plus 50 sampel
  pendek."
- *Untuk mahasiswa baru:* nol masih di rentang fisik 0–100 %, jadi tidak bisa
  disebut "salah ketik". Pembacaan macet di nol justru gejala gangguan; kalau
  dibuang, kita menyembunyikan kasus sulit.
- *Bukti:* NB11 §2b; Bahan Q9.

**T10. Missing value dan outlier? Sinkronisasi waktu? (Q8, Q14, Q15)**
- *30 detik:* "Tidak ada NaN dan tidak ada nilai di luar 0–100 %. Stempel waktu
  diubah ke UTC, diurutkan, duplikat dibuang, lalu dipaksa ke grid 5 menit;
  slot bolong nol."
- *Bukti:* NB11 §2 sel 7, §2b sel 8.

**T11. "Broker" itu apa? (Q12)**
- *30 detik:* "Komponen yang menyatukan empat aliran sensor menjadi satu tabel
  berwaktu sama tanpa melebur identitas sensornya; bukan message broker
  produksi."
- *Bukti:* NB02 sel 1–13; `deteksi_fault_core.sync_broker`.

<!-- bagian: fault -->

**T12. Kenapa gangguannya disimulasikan? (Q10)**
- *30 detik:* "Data asli tidak berlabel rusak dan saya tidak boleh merusak
  sensor. Simulasi memberi label pasti dan bisa diulang dengan seed tetap.
  Batasnya: hasil berlaku untuk gangguan yang dimodelkan."
- *Untuk mahasiswa baru:* ini seperti latihan pemadam kebakaran dengan api
  buatan: kita tahu persis di mana apinya, jadi bisa menilai apakah petugas
  menemukannya.
- *Bukti:* NB11 §3; `02_kelemahan.md` B (validitas konstruk).

**T13. Tulis model tiap gangguan. (Q17, P17)**
- *30 detik:* "Drift: tambahan yang naik linear 1 poin/hari selama episode
  3 hari. Spike: lonjakan sesaat dengan peluang 1,5 %. Bias: geser tetap.
  Hardware: sebagian titik macet atau hilang."
- *Untuk mahasiswa baru:* sinyal 40, 40, 40, 40.
  Bias +2 → 42, 42, 42, 42. Spike di titik ke-3 → 40, 40, 47, 40.
  Drift → 40, 40,003, 40,007, 40,010 (1 poin/hari = 0,00347 poin per 5 menit).
  Macet → 40, 41, 41, 41 (nilai terakhir diulang).
- *Bukti:* `deteksi_fault_core.py` `simulate_drift_fault`,
  `simulate_spike_fault`, `simulate_bias_fault`, `simulate_hardware_fault`;
  `04_walkthrough_kode.md` baris 18.

**T14. Kenapa drift hanya 1 poin/hari? (Q11, Q42)**
- *30 detik:* "Agar tidak mustahil secara fisik: versi awal 57,6 poin/hari dan
  menembus 100 % dalam sehari. Rentang 0,003–0,17 poin/hari yang saya pakai
  sebagai pembanding adalah **asumsi skenario**; saya tidak menemukan sumber
  terbit yang mengukur laju drift sensor kelembaban tanah per hari."
- *Untuk mahasiswa baru:* dalam jendela N = 200 (16,7 jam) drift hanya
  menambah ±0,7 poin, lebih kecil dari ayunan harian sinyal. Karena itu drift
  memang paling sulit dideteksi, dan itu dilaporkan.
- *Bukti:* README "Koreksi simulator drift"; `07_pustaka_dan_kebaruan.md` (e).
- *Serangan lanjutan:* "Jadi skenario Anda tidak realistis?" → "Arahnya
  realistis (lambat), besarnya asumsi. Uji intensitas NB13 menunjukkan drift
  baru terdeteksi (AUC 0,80–0,85) pada total 60 poin."

**T15. Hardware malfunction 8 % macet dan 5 % hilang? (F4)**
- *30 detik:* "Total titik rusak 8 %: 5 % hilang (NaN) dan 3 % macet, karena
  titik hilang adalah bagian dari titik macet lalu ditimpa NaN. Deskripsi lama
  saya koreksi."
- *Bukti:* `deteksi_fault_core.py` `simulate_hardware_fault`;
  `02_kelemahan.md` F4.

**T16. Kenapa bias konstan tidak terdeteksi? (Q47)**
- *30 detik:* "Karena setiap template dikurangi rata-ratanya sebelum
  dibandingkan. Geser tetap ikut hilang. AUC bias sekitar 0,50 dari 0,08 sampai
  5 poin."
- *Untuk mahasiswa baru:* template [42, 43] dikurangi rata-rata 42,5 menjadi
  [−0,5, 0,5]; template tanpa bias [40, 41] juga menjadi [−0,5, 0,5]. Entropi
  melihat keduanya identik. Panel fault "bias" di notebook menunjukkan kurva
  entropi hampir tidak bergerak.
- *Bukti:* NB12 `12_sweep_fault_param.csv`; `embed_matrix_centered`.

**T17. Kapan jendela dilabel rusak? (Q22, P18)**
- *30 detik:* "Kalau porsi titik rusak di jendela > 1 %; jendela di bawah itu
  dibuang supaya model tidak belajar label yang salah."
- *Bukti:* `FAULT_RATIO_THR`; `11_akuntansi_window.csv`.

<!-- bagian: entropi -->

**T18. Apa itu entropi di sini? (Q3, Q23, P9)**
- *30 detik:* "Fuzzy entropy mengukur seberapa sering pola pendek yang mirip
  tetap mirip ketika diperpanjang satu titik. Sinyal teratur: entropi kecil.
  Sinyal acak: entropi besar."
- *Untuk mahasiswa baru:* rumusnya FE = ln(φ_m / φ_{m+1}). φ_m adalah rata-rata
  "kemiripan" antar semua pasangan template sepanjang m = 2. Kalau pasangan yang
  mirip di panjang 2 tetap mirip di panjang 3, φ hampir sama dan ln-nya dekat 0.
  Contoh angka pada deret x = [3, 5, 4, 6, 5, 7, 6, 4]: r = 0,2 · sd =
  0,261861; jarak Euclidean template [−1, 1] dan [0,5, −0,5] = 2,121320,
  keanggotaan μ = 1/(1 + 2,121320/0,261861) = 0,109879.
- *Bukti:* `03_algoritma_dengan_tangan.md` (b), dicek otomatis oleh
  `cek_algoritma_tangan.py`; Chen dkk. 2007 (`chen2007fuzzyen`).

**T19. Kenapa multiskala τ = 1..15 dan 60 fitur? (Q24, Q28)**
- *30 detik:* "Gangguan berbeda tampak di skala berbeda: spike di skala halus,
  drift di skala kasar. 15 skala × 4 sensor = 60 fitur."
- *Untuk mahasiswa baru:* skala τ = 2 artinya rata-ratakan tiap dua titik:
  [3, 5, 4, 6] → [4, 5]. Komposit: hitung juga dengan geseran awal (mulai dari
  titik kedua) lalu rata-ratakan hasilnya, supaya tidak bergantung pada titik
  awal.
- *Bukti:* `coarse_grain_cmse`; Costa dkk. 2002, Wu dkk. 2013.

**T20. Di mana tepatnya EDM dan JSD berbeda? (Q4, Q26, Q27, P10)**
- *30 detik:* "Hanya di jarak d antar template. EDM: jarak Euclidean. JSD:
  tiap template diubah jadi histogram lunak 8 bin, jarak = akar JSD. Embedding,
  centroid shifting, fungsi keanggotaan, dan rumus entropi identik."
- *Untuk mahasiswa baru:* EDM seperti membandingkan dua foto piksel demi
  piksel. JSD seperti membandingkan dua foto dari histogram warnanya saja.
  Contoh JSD dari hitungan tangan: d = √0,011811 = 0,108678, ambang
  r_JSD = 0,2·√ln2 = 0,166511, sehingga μ = 1/(1 + 0,108678/0,166511) = 0,605078.
- *Bukti:* `deteksi_fault_core.py` `phi_euclidean`, `vectors_to_distributions`,
  `pairwise_jsd`, `phi_jsd`; `03_algoritma_dengan_tangan.md` (c).

**T21. Kenapa r = 0,2·SD dan r_JSD = 0,2·√ln2? (Q25)**
- *30 detik:* "0,2·SD adalah toleransi lazim dari literatur sample/fuzzy
  entropy, diskalakan dengan simpangan baku supaya adil untuk amplitudo
  berbeda. JSD (log natural) terbatas di [0, ln 2], jadi akar JSD maksimum √ln2;
  ambangnya 20 % dari maksimum itu."
- *Bukti:* `JSD_R_RATIO`, `jsd_fuzzy_entropy_1d`; Richman & Moorman 2000.

**T22. Sifat JSD yang penting? (P11)**
- *30 detik:* "Simetris, selalu berhingga, terbatas [0, ln 2], dan akarnya
  metrik. KL tidak simetris dan bisa tak hingga kalau ada bin nol."
- *Bukti:* Lin 1991; Endres & Schindelin 2003.

**T23. Apa kelemahan konseptual JSD-Fuzzy? (F7)**
- *30 detik:* "Histogram tidak peduli urutan. Template naik [−1, 1] dan turun
  [1, −1] menghasilkan histogram sama, jadi JSD = 0 persis, padahal Euclidean
  memberi 2,83. JSD-Fuzzy mengukur kemiripan sebaran nilai, bukan bentuk pola.
  Akibat lain: nilainya bisa negatif pada deret pendek."
- *Untuk mahasiswa baru:* kata "TAS" dan "SAT" punya huruf yang sama. Kalau
  hanya menghitung huruf, keduanya dianggap kembar.
- *Bukti:* `02_kelemahan.md` F7 (dicek 2026-11-20 dengan
  `vectors_to_distributions` + `pairwise_jsd`); `07_pustaka_dan_kebaruan.md`
  catatan jujur.
- *Serangan lanjutan:* "Kalau buta urutan, kenapa JSD menang?" → "Gangguan yang
  diuji mengubah sebaran amplitudo di dalam template (spike, macet); itu yang
  ditangkap JSD. Untuk gangguan yang hanya membalik arah, JSD tidak peka. Ini
  saya tulis sebagai batasan."

**T24. Kenapa n_ref = 128 titik acuan? (P14)**
- *30 detik:* "Semua pasangan berbiaya kuadrat; 128 acuan acak dengan seed
  tetap memberi aproksimasi yang sama untuk kedua metode."
- *Bukti:* `_pick_ref`.

**T25. Kenapa JSD lebih lambat? (P15)**
- *30 detik:* "Tiap template harus diubah menjadi histogram dan JSD butuh
  logaritma per bin; Euclidean cukup selisih kuadrat. Biaya ada di ekstraksi
  fitur; inferensi per jendela setara."
- *Bukti:* NB14 `hasil14` (1,7–6,8× pada ekstraksi fitur).

<!-- bagian: model -->

**T26. ANN-nya seperti apa? (Q29)**
- *30 detik:* "Feed-forward, 60 masukan, satu atau dua lapisan tersembunyi
  dari kisi 8 arsitektur, aktivasi relu/tanh, keluaran softmax. Arsitektur
  dipilih di dalam cross-validation."
- *Bukti:* NB11 §10; `models/meta.json`.

**T27. Tulis rumus update LM dan arti λ. (P1, P2, P3)**
- *30 detik:* "(JᵀJ + λ·diag(JᵀJ))·Δ = −Jᵀr. λ kecil: mendekati Gauss–Newton,
  cepat dekat minimum. λ besar: langkah kecil searah gradien, aman. Galat naik:
  langkah ditolak, λ dikali 2. Galat turun: diterima, λ dikecilkan menurut
  rasio ρ (aturan Nielsen)."
- *Untuk mahasiswa baru:* LM seperti menuruni bukit dalam kabut. Kalau
  langkah besar membuat Anda naik (galat bertambah), Anda mundur dan mencoba
  langkah lebih kecil. Di contoh mainan: iterasi 1 ditolak (ρ = −0,58),
  λ 0,001 → 0,002; pada iterasi 9 galat turun dari 1,453047 ke 0,194302.
- *Bukti:* `03_algoritma_dengan_tangan.md` (a); `lm_fit`; Marquardt 1963,
  Hagan & Menhaj 1994, Madsen dkk. 2004.

**T28. Bagaimana Jacobian analitik dihitung dan dibuktikan? (Q31, P4)**
- *30 detik:* "Backpropagation per keluaran. Dibuktikan dengan membandingkan
  terhadap Jacobian numerik beda hingga."
- *Untuk mahasiswa baru:* turunan numerik: geser satu bobot sedikit (ε), lihat
  berapa keluaran berubah, bagi ε. Kalau rumus analitik dan cara numerik
  memberi angka sama sampai ±10⁻¹⁰, rumusnya benar.
- *Bukti:* NB11 §9a; `03` (d); `cek_algoritma_tangan.py` (Jacobian tangan vs
  kode, selisih 5,6e-17).

**T29. LM itu untuk regresi; klasifikasinya bagaimana? (P5)**
- *30 detik:* "Target one-hot, kuadrat terkecil pada keluaran linear, softmax
  saat prediksi — cara yang lazim di `trainlm` MATLAB. Kelemahannya
  probabilitasnya tidak terkalibrasi."

**T30. Kenapa LM, bukan Adam? (Q30, P7)**
- *30 detik:* "LM metode pada paper acuan dan cocok untuk jaringan kecil. Di
  arsip NB09 LM 0,433, BR 0,429, GA 0,306, Adam 0,268; LM dan BR tidak berbeda,
  keduanya jauh di atas GA dan Adam."
- *Bukti:* NB09 §10.

**T31. Kenapa jaringan dibatasi kecil? (P6)**
- *30 detik:* "JᵀJ berukuran P×P dan harus diselesaikan tiap iterasi, biayanya
  O(P³). Karena itu parameter dibatasi sekitar 5 ribu."

**T32. Jelaskan nested grouped CV. (Q33, P20)**
- *30 detik:* "Luar: StratifiedGroupKFold 5 lipatan, grup = blok waktu. Dalam:
  2 lipatan untuk memilih arsitektur dengan macro-F1. Model terpilih dilatih
  ulang di data latih luar dan dinilai di data uji luar."
- *Untuk mahasiswa baru:* jendela bertetangga berbagi 50 % isi. Kalau satu di
  latih dan tetangganya di uji, model "sudah pernah melihat" separuh soal ujian.
  Mengelompokkan per blok waktu memisahkan soal latihan dan soal ujian.
- *Bukti:* NB11 §10; Roberts dkk. 2017, Bergmeir & Benítez 2012.

**T33. Kenapa macro-F1? (P23)**
- *30 detik:* "Kelas tidak seimbang; macro-F1 memberi bobot sama pada tiap
  kelas, sehingga kelas sulit tidak tertutup kelas mudah."
- *Untuk mahasiswa baru:* 90 jendela normal dan 10 rusak. Model yang selalu
  menjawab "normal" akurasinya 90 %, tetapi F1 kelas rusak 0. Macro-F1 =
  rata-rata (F1 normal ≈ 0,95, F1 rusak 0) ≈ 0,47 — angka itu jujur.

**T34. Bobot awal Xavier? (A5, F2)**
- *30 detik:* "Jujur: urutan `lm_pack`/`lm_unpack` berbeda, sehingga sebagian
  bobot lapisan keluaran mulai dari nol. Konsisten saat latih dan prediksi,
  sama untuk kedua metode. Teks skripsi tidak menyebut 'Xavier dengan bias
  nol'."
- *Bukti:* `02_kelemahan.md` A5, F2.

<!-- bagian: prediksi -->

**T35. Bagaimana membaca tebakan di panel?**
- *30 detik:* "Tabel pertama: tebakan model N = 700 (S1 normal/rusak dan S2
  atau S5 jenis gangguan) untuk jendela asli dan jendela yang diberi gangguan,
  EDM dan JSD berdampingan, dengan tingkat keyakinan. Tabel kedua: skor tiap
  sensor; skor > 0,5 berarti sensor itu ditebak rusak."
- *Untuk mahasiswa baru:* softmax mengubah skor mentah menjadi peluang yang
  jumlahnya 1. Skor [2, 0] → e²/(e² + e⁰) = 7,39/8,39 = 0,88 untuk kelas
  pertama.
- *Bukti:* `maju` di `buat_cache_interaktif.py`; sel prediksi notebook.

**T36. Kenapa tebakan kadang salah di panel?**
- *30 detik:* "Karena F1 validasi silang skenario ini memang belum sempurna,
  dan panel memakai satu jendela, bukan rata-rata. Contoh salah lebih jujur
  daripada demo yang dipilih-pilih."
- *Bukti:* `|cv_f1_nb11` di `model_sidang.npz`; `11_rq4_performa_ann.csv`.

**T37. Bisa dipakai real-time? (Q43, P32)**
- *30 detik:* "Belum. Satu keputusan N = 700 butuh 58 jam data; cocok untuk
  pemantauan harian. Untuk notifikasi cepat ada NB15 (per sampel, tanpa
  entropi) dengan aturan persistensi."
- *Bukti:* NB15, NB15b.

<!-- bagian: hasil -->

**T38. Hasil utamanya? (Q35, Q36)**
- *30 detik:* "JSD-Fuzzy lebih tinggi di 11 dari 15 sel, rata-rata selisih
  F1 +0,028. Per ukuran jendela: N = 200 menang 3/5, N = 700 5/5,
  N = 1000 3/5."
- *Untuk mahasiswa baru:* satu sel = satu skenario (S1–S5) × satu ukuran
  jendela (200/700/1000). Di tiap sel kedua metode dilatih dengan protokol yang
  sama; kita catat selisih F1-nya.
- *Bukti:* NB11 §10; `11_rq4_selisih.csv`; `statistik_sel.csv` baris NB11.

**T39. Hasil identifikasi sensor? (Q37, F1)**
- *30 detik:* "JSD 0,675 vs EDM 0,634, menang 9 dari 15. Catatan jujur: tahap
  ini memakai pembagian acak 70/30, jadi angka absolutnya optimistis; selisih
  kedua metode tetap adil karena pembagiannya sama."
- *Bukti:* NB11 §12; `11_sensor_ringkas.csv`; `02_kelemahan.md` F1.

**T40. JSD lebih tinggi, EDM lebih apa?**
- *30 detik:* "EDM lebih stabil (56 dari 63 pasangan). Keduanya saya
  tampilkan."
- *Bukti:* Bahan "Kalimat siap pakai" 3.

**T41. NB10, NB11, NB14 — kenapa ada tiga hasil?**
- *30 detik:* "NB10 dan NB11 protokol sama, berbeda anggaran sesi; NB14
  memakai simulator fault 'sedang' dengan posisi acak ber-seed. Arah ketiganya
  sama: JSD lebih tinggi."
- *Bukti:* `statistik_sel.csv` (NB10 11/15, NB11 11/15, NB14 12/15).

<!-- bagian: statistik -->

**T42. Apakah hasilnya signifikan? (Q39, Q41)**
- *30 detik:* "NB11 gabungan belum: uji-t p = 0,076, Wilcoxon 0,107. Ukuran
  efek sedang (d_z = 0,49), dan interval bootstrap 95 % selisih F1 [+0,002;
  +0,058] tidak memuat nol. Jadi saya menyebutnya konsisten lebih tinggi, bukan
  terbukti. NB14 gabungan signifikan: p = 0,0006, d_z = 1,13, CI [+0,043;
  +0,108]."
- *Untuk mahasiswa baru:* p menjawab "kalau sebenarnya tidak ada beda,
  seberapa sering kita melihat beda sebesar ini karena kebetulan?". p = 0,076
  berarti sekitar 7,6 %, di atas batas 5 %. Bootstrap: ambil ulang 15 selisih
  secara acak dengan pengembalian 10.000 kali, hitung rata-rata tiap kali, lalu
  lihat 2,5 % dan 97,5 % terbawah/teratas.
- *Bukti:* `sidang/qa/hasil_uji/statistik_sel.csv`; `statistik_tambahan.py`.
- *Serangan lanjutan:* "CI tidak memuat nol tapi p > 0,05 — mana yang benar?"
  → "Keduanya pendekatan berbeda pada n = 15: bootstrap persentil cenderung
  sempit pada sampel kecil, uji-t lebih konservatif. Saya ambil yang
  konservatif untuk klaim."

**T43. N = 700 signifikan, kan? (Q41)**
- *30 detik:* "Tanpa koreksi p = 0,027; setelah koreksi Holm untuk tiga ukuran
  jendela p = 0,081. Jadi tidak saya klaim signifikan setelah koreksi."
- *Untuk mahasiswa baru:* menguji tiga hal sekaligus menaikkan peluang satu
  'beruntung'. Holm: urutkan p, kalikan yang terkecil dengan 3, berikutnya
  dengan 2, dan seterusnya (dengan aturan tidak boleh turun).
- *Bukti:* `statistik_sel.csv` kolom `p_t_holm_3N`; Holm 1979.

**T44. Uji-t pada lipatan CV itu bias, kan? (P22, C2)**
- *30 detik:* "Ya, lipatan berbagi data latih sehingga varians diremehkan.
  Saya menambahkan uji-t terkoreksi Nadeau–Bengio. Uji utama saya per sel
  (skenario × N), bukan per lipatan."
- *Untuk mahasiswa baru:* koreksi Nadeau–Bengio memperbesar varians dengan
  faktor (1/k + n_uji/n_latih); hasilnya |t| selalu lebih kecil.
- *Bukti:* `statistik_tambahan.py` `t_nadeau_bengio`, keluaran per lipatan
  `hasil_uji/statistik_lipatan.csv` (dari run purge N = 700); Nadeau & Bengio
  2003; Dietterich 1998; uji `test_nadeau_bengio_*`.

**T45. Kenapa Wilcoxon per N tidak pernah signifikan? (F6)**
- *30 detik:* "Dengan lima pasangan, p eksak dua sisi terkecil 2/2⁵ = 0,0625.
  Jadi secara matematis tidak mungkin < 0,05."

**T46. Ukuran efek? (P25)**
- *30 detik:* "d_z NB11 gabungan 0,49 (sedang), NB14 1,13 (besar)."
- *Bukti:* `statistik_sel.csv` kolom `cohen_dz`; Cohen 1988.

<!-- bagian: kelemahan -->

**T47. Apakah masih ada kebocoran data? (A1, P21)**
- *30 detik:* "Ada sedikit di batas grup: jendela terakhir satu blok bisa
  tumpang-tindih dengan jendela pertama blok tetangga. Sama untuk kedua metode.
  Saya ukur dengan purge: jendela latih yang tumpang-tindih dengan jendela uji
  dibuang, lalu model N = 700 dilatih ulang."
- *Untuk mahasiswa baru:* purge = membuang soal latihan yang isinya beririsan
  dengan soal ujian. Kalau skor turun banyak setelah purge, berarti skor lama
  terbantu kebocoran.
- *Bukti:* `sidang/qa/uji_purge_permutasi_N700.py`,
  `hasil_uji/efek_purge_permutasi.csv`. Catatan: purge juga mengurangi data
  latih ±20 %, jadi penurunan skor tidak seluruhnya akibat kebocoran.

**T48. Bagaimana tahu model tidak belajar artefak? (kontrol negatif)**
- *30 detik:* "Label saya acak lalu model dilatih ulang dengan protokol sama.
  Kalau pipeline jujur, F1-nya turun ke tingkat tebak-tebakan."
- *Untuk mahasiswa baru:* untuk 2 kelas, tebakan acak memberi F1 sekitar 0,5.
  Kalau model dengan label acak masih jauh di atas itu, ada yang bocor.
- *Bukti:* `hasil_uji/efek_purge_permutasi.csv` varian `permutasi` (S1, S5).

**T49. Validitas eksternal? (D)**
- *30 detik:* "Satu lokasi, satu musim, empat sensor, gangguan disimulasikan.
  Hasil berlaku untuk gangguan yang dimodelkan; langkah berikutnya data lapangan
  yang benar-benar rusak."
- *Bukti:* `02_kelemahan.md` D.

**T50. Data asli atau buatan? (B6, P16)**
- *30 detik:* "Gangguan disuntikkan secara simulasi ke `data_sensor.csv`."
  Asal-usul `data_sensor.csv` **belum dipastikan**: baris awalnya tampak hasil
  interpolasi ke grid 30 detik. Jawab hanya setelah dikonfirmasi ke pembimbing;
  jangan mengarang.
- *Bukti:* `02_kelemahan.md` B6.

**T51. p = 0,0006 di NB14 — apakah karena gangguan diperbesar sampai JSD menang?
(P24, A6)**
- *30 detik:* "Simulator NB14 ditetapkan sebelum membandingkan metode dan sama
  untuk keduanya. Tetap signifikan setelah koreksi ganda."
- *Bukti:* `02_kelemahan.md` A6, C3.

<!-- bagian: penutup -->

**T52. Kalimat penutup.**
- "Saya mengganti ukuran kemiripan di dalam fuzzy entropy dari Euclidean
  menjadi Jensen–Shannon dan mengujinya sampai identifikasi sensor."
- "JSD-Fuzzy lebih tinggi di 11 dari 15 sel (+0,028); secara gabungan NB11
  belum signifikan (p = 0,076), sehingga saya menyebutnya konsisten, bukan
  terbukti. Pada simulator NB14 selisihnya signifikan (p = 0,0006)."
- "Batasan yang saya laporkan: gangguan disimulasikan, JSD buta urutan,
  sedikit kebocoran di batas grup, dan bias konstan tidak terdeteksi oleh
  representasi terpusat."

**T53. Langkah berikutnya? (Q44)**
- Data lapangan yang benar-benar rusak; jendela adaptif untuk deteksi lebih
  cepat; fitur level untuk bias; GroupShuffleSplit pada tahap sensor-ID.

**T54. Kalau ditanya angka yang tidak ada di layar? (Q45)**
- "Saya cek di `01_klaim_bukti.csv` atau notebook sumbernya." Jangan mengarang
  angka baru.
