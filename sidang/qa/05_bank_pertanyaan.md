# 05 — Bank pertanyaan penguji (versi menyerang)

Pelengkap `Bahan_Sidang.md` (Q1–Q48), bukan pengganti. Di sini isinya
pertanyaan lanjutan yang muncul **setelah** jawaban pertama: "kenapa",
"buktikan", "bagaimana kalau". Tiap jawaban dibuat untuk ±30 detik dan
menunjuk bukti. Angka hanya boleh disebut kalau berstatus COCOK di
`01_klaim_bukti.csv`.

Kode level: **[L1]** konsep, **[L2]** algoritma/rumus, **[L3]** baris kode.

---

## 1. Levenberg–Marquardt

**P1 [L2]. Tulis rumus update LM.**
(JᵀJ + λ·diag(JᵀJ))·Δ = −Jᵀr, lalu θ ← θ + Δ kalau galat turun. J = turunan
residual terhadap semua bobot, r = keluaran − target. Contoh angka:
`03_algoritma_dengan_tangan.md` bagian (a).

**P2 [L2]. Apa arti λ? Kapan naik, kapan turun?**
λ kecil → langkah mendekati Gauss–Newton (cepat dekat minimum); λ besar →
langkah kecil searah gradien turun (aman jauh dari minimum). Galat naik →
langkah ditolak, λ dinaikkan (×2). Galat turun → diterima, λ dikecilkan
menurut rasio gain ρ (aturan Nielsen: λ·max(1/3, 1−(2ρ−1)³)).
Kode: `deteksi_fault_core.py` `lm_fit` (baris ±511–568).

**P3 [L2]. Kenapa diag(JᵀJ), bukan identitas?**
Itu versi Marquardt (1963): redaman diskalakan per parameter sehingga bobot
dengan kelengkungan besar tidak mendominasi. Identitas adalah versi Levenberg
(1944).

**P4 [L3]. Bagaimana Jacobian analitik dihitung?**
Backpropagation per keluaran: delta dimulai dari identitas (keluaran linear),
dimundurkan lewat tiap lapisan dengan turunan tanh (1 − h²) atau relu; blok
dW = aktivasi sebelumnya ⊗ delta. Kebenarannya dibuktikan dengan
membandingkan terhadap Jacobian numerik (selisih relatif < 1e-3; di NB15
1,3e-5 dan 4,3e-9). Kode: `lm_residual_jacobian`.

**P5 [L2]. LM itu untuk regresi kuadrat terkecil; klasifikasinya bagaimana?**
Target dibuat one-hot dan keluaran linear dilatih meminimalkan jumlah kuadrat
residual; probabilitas kelas didapat dari softmax keluaran saat prediksi.
Ini cara "LM untuk klasifikasi" yang lazim di toolbox MATLAB (`trainlm` +
target one-hot). Kelemahannya: bukan cross-entropy, jadi probabilitas tidak
terkalibrasi — karena itu ambang dipilih di validasi.

**P6 [L1]. Kenapa LM tidak dipakai untuk jaringan besar?**
JᵀJ berukuran P×P dan harus diselesaikan tiap iterasi (O(P³)), J berukuran
(m·kelas)×P. Karena itu arsitektur dibatasi ±5 ribu parameter
(`build_11…:1600–1609`).

**P7 [L1]. Bukti LM lebih cocok daripada Adam?**
Notebook 09: F1 rata-rata LM 0,433, BR 0,429, GA 0,306, Adam 0,268; LM vs BR
tidak beda (p = 0,81), LM/BR vs GA dan Adam signifikan. (Cek status di 01.)

**P8 [L3]. Apa yang terjadi di `lm_pack` / `lm_unpack`?**
Lihat `02_kelemahan.md` A5 dan `04_walkthrough_kode.md` "Cacat yang
diketahui". Jawab jujur: konsisten saat latih/prediksi, bobot awal tidak
persis Xavier, dampaknya sama untuk kedua metode.

## 2. Entropi fuzzy, EDM, JSD

**P9 [L2]. Tulis rumus fuzzy entropy.**
FuzzyEn(m, r) = ln φ_m(r) − ln φ_{m+1}(r), φ_m = rata-rata kemiripan antar
pasangan vektor embedding berdimensi m (setelah dikurangi rata-ratanya).
Multiskala: sinyal di-coarse-grain dulu pada skala τ. Hitungan tangan:
`03` bagian (b).

**P10 [L2]. Di mana tepatnya EDM dan JSD berbeda dalam rumus?**
Hanya di d(i,j): EDM = jarak Euclidean antar vektor; JSD = Jensen–Shannon
divergence antara sebaran halus kedua vektor. Fungsi keanggotaan
μ = 1/(1 + d/r), embedding, centroid shifting, dan rumus entropi identik.

**P11 [L2]. Sifat JSD yang relevan?**
Simetris, selalu berhingga, terbatas di [0, ln 2] (atau [0,1] basis 2), dan
akar JSD adalah metrik. Berbeda dengan KL yang tidak simetris dan bisa tak
hingga kalau ada bin nol.

**P12 [L2]. Bagaimana vektor 2 angka jadi sebaran?**
Soft binning: tiap nilai disebar ke 8 pusat bin dengan kernel halus lebar h,
rentang ±3 SD, lalu dinormalisasi. Lihat `02` B5 dan NB12 (bins=4,
h_ratio=1,0 disarankan).

**P13 [L2]. Kenapa μ = 1/(1 + d/r), bukan exp(−dⁿ/r)?**
Lihat `02` B4. Monoton turun, bernilai 1 saat d = 0, sama untuk kedua metode.

**P14 [L1]. Kenapa n_ref = 128 titik acuan?**
Menghitung semua pasangan O(n²) terlalu mahal untuk ribuan window ×15 skala ×
4 sensor; kemiripan dirata-rata terhadap 128 vektor acuan yang dipilih dengan
seed tetap. Ini aproksimasi yang sama untuk kedua metode.

**P15 [L1]. Kenapa JSD 1,7–6× lebih lambat?**
Tiap vektor harus diubah jadi sebaran dan JSD butuh log per bin, sedangkan
Euclidean cukup selisih kuadrat. Inferensi per window tetap setara (±0,09 ms)
karena biaya ada di ekstraksi fitur. (Cek angka di 01.)

## 3. Data, fault, label

**P16 [L1]. Data asli atau buatan?** — `02` B6. **Wajib dipastikan dulu.**

**P17 [L2]. Tulis model matematis tiap fault.**
NB11: drift x + α·k di episode (α = 1 poin/hari dalam sampel), spike periodik
+0,08·SD, bias +0,08, hardware: 8 % nilai diganti nilai acak lain (macet),
5 % NaN. NB14: bias +5 selama episode, drift naik linear 0→5 per episode,
spike x·(1+0,30) acak 1,52 %. Kode: `simulate_*_fault`.

**P18 [L3]. Window kapan dilabel fault?**
NB11: porsi sampel fault > 1 % (`FAULT_RATIO_THR`); window di bawah itu dibuang.
NB14: semua komponen kombinasi harus ada (spike ≥ 1 titik, lainnya > 1 %).

**P19 [L1]. Bagaimana sensor-ID dirantai dari prediksi?**
Lihat `04_walkthrough_kode.md` (`LMSensorID`, §12 NB11): prediksi kelas
out-of-fold → model per sensor → F1 per sensor dan "semua-4-benar".

## 4. Evaluasi dan statistik

**P20 [L2]. Jelaskan nested grouped CV Anda.**
Luar: StratifiedGroupKFold 5 lipatan, grup = blok waktu. Dalam: 2 lipatan
untuk memilih arsitektur/aktivasi dengan macro-F1. Model terpilih dilatih
ulang di lipatan luar latih, dinilai di lipatan luar uji.

**P21 [L2]. Apakah masih ada kebocoran?** — `02` A1 (jujur: ya, di batas grup;
sama untuk kedua metode).

**P22 [L2]. Kenapa uji-t berpasangan? Asumsinya?**
Pasangan = sel yang sama (skenario × N) dengan dua metode. Asumsi: selisih
berdistribusi normal — dengan n = 5 tidak bisa dicek, karena itu Wilcoxon
dilaporkan juga.

**P23 [L2]. Kenapa macro-F1?**
Kelas tidak seimbang setelah pembuangan window; macro-F1 memberi bobot sama
pada tiap kelas sehingga kelas sulit (kombinasi) tidak tertutup kelas mudah.

**P24 [L2]. p = 0,0006 di NB14 — apakah karena fault diperbesar sampai JSD
menang?** — `02` A6 + C3 (tetap signifikan setelah Bonferroni).

**P25 [L1]. Ukuran efek?** NB14 Cohen d = 1,13 (cek di 01). NB11 selisih F1
rata-rata +0,028.

## 5. Implementasi dan reprodusibilitas

**P26 [L3]. Bagaimana memastikan hasil bisa diulang?**
Seed tetap di semua langkah acak, kernel Kaggle tercatat versinya, builder
`build_*.py` menghasilkan notebook dari teks yang sama, log Kaggle disimpan di
`hasil*/`. Klaim dicek ulang otomatis oleh `cek_klaim.py`.

**P27 [L3]. Kenapa model disimpan numpy, bukan pickle?**
Pickle sklearn dari Kaggle (1.6.1) gagal dibuka di laptop (1.9.1).
`SavedLM` / `.npz` hanya menyimpan angka (pengisi NaN, mean, skala, bobot) dan
prediksinya identik (selisih 0).

**P28 [L3]. Kenapa `sliding_window_view` perlu di-transpose?**
Hasilnya berbentuk (n_window, kanal, panjang); entropi harus dihitung per
kanal sepanjang waktu, jadi diubah ke (n_window, panjang, kanal). Tanpa itu
entropi dihitung atas potongan sepanjang 4. Kode: `make_windows`.

**P29 [L1]. Berapa biaya komputasinya?** NB11 wall 610 menit, ±27,5 jam-CPU;
NB14 478 menit. (Cek di 01.)

## 6. Pertanyaan "bagaimana kalau"

**P30.** Bagaimana kalau sensor kelima ditambahkan? → fitur bertambah 15, model
dilatih ulang; arsitektur dan metode tidak berubah.
**P31.** Bagaimana kalau hanya satu sensor? → tidak ada pembanding antar sensor;
lokalisasi hilang (notebook 02).
**P32.** Bagaimana dipakai di lapangan? → model tersimpan (NB16 / NB15 `.npz`),
satu keputusan per window 16,7 jam; untuk notifikasi cepat pakai NB15 (per
sampel, tanpa entropi) dengan aturan persistensi.
**P33.** Bagaimana kalau penguji minta demo langsung? → `sidang/PANDUAN_DEMO.md`.

---

## Latihan

1. Tutup kolom jawaban, jawab lisan ≤ 30 detik, buka bukti di layar ≤ 20 detik.
2. Untuk P1, P4, P9, P10: tulis rumus di kertas tanpa melihat.
3. Untuk P8, P21, P24: latih kalimat "akui → dampak → mitigasi" sampai lancar.
