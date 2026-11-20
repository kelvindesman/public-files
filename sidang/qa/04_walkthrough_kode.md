# 04 — Walkthrough kode inti (per fungsi) + cacat yang diketahui

Acuan baris: `deteksi_fault_core.py` (disingkat **core**) pada commit `422232b`,
`build_11_annlm_real_notebook.py` (**b11**), `build_14_lengkap_fault_baru_notebook.py`
(**b14**), `build_15_data_hipotetik_ann_lm_notebook.py` (**b15**),
`simpan_model.py`. Bentuk array: `T` = jumlah sampel deret (5 menit),
`N`/`win` = panjang jendela, `nwin` = jumlah jendela, `P` = jumlah parameter.

Bukti angka untuk klaim di dokumen ini: `sidang/qa/cek_cacat_kode.py`
(keluaran diringkas di bagian akhir) dan `sidang/qa/cek_algoritma_tangan.py`.

---

## 1. Simulator fault (core:113-188)

### `simulate_drift_fault(x, intensity, episode_days, n_episodes, seed)` — core:113-122
Menambah tanjakan linear di 3 episode (mulai 10 %, 40 %, 70 % deret), tiap
episode 3 hari, kemiringan 1 poin/hari (= 0,00347 poin per sampel 5 menit).
Masukan `x (T,)`; keluaran `(x + drift (T,), mask bool (T,))`.

- **Kenapa episode, bukan drift sepanjang deret?** Drift sepanjang deret membuat
  semua jendela fault dan tak ada pembanding di deret yang sama; episode memberi
  jendela fault dan normal dari sensor yang sama. Posisi tetap (core:119) supaya
  dapat diulang.
- **Kenapa `seed` tidak dipakai?** Drift deterministik; argumen `seed` hanya ada
  supaya semua simulator punya tanda tangan sama untuk `simulate_multiple_faults`
  (core:152). Konsekuensinya: 8 ulangan per skenario (core:236) memakai pola
  drift yang sama, yang berbeda hanya subset sensornya.

### `simulate_spike_fault(x, intensity=0.08, p=0.015, seed)` — core:125-128
Lonjakan setiap `tau = int(1/p) = 66` sampel, tinggi `0.08 * nanstd(x)`.
Keluaran `(x + spikes, mask)`.

- **Kenapa `nanstd`?** Bila hardware fault (NaN) diterapkan lebih dulu di
  kombinasi, `std` biasa akan NaN.
- **Kenapa posisi periodik, bukan acak?** Deterministik dan dapat diulang; tetapi
  akibatnya posisi spike identik di semua sensor/ulangan (`cek_cacat_kode.py` §7).
  Tinggi 0,08·sd berada **di bawah** toleransi $r=0{,}2$·sd, jadi spike ini
  sengaja halus. Notebook 14 menggantinya dengan spike relatif $x(1+0{,}30)$ di
  posisi acak (b14:67-75).

### `simulate_bias_fault(x, bias=0.08, seed)` — core:131-132
Menambah konstanta 0,08 ke seluruh deret; semua sampel ditandai fault.
- **Apa yang dilihat fitur entropi?** **Tidak ada.** Lihat "Cacat yang diketahui"
  butir C2: kedua entropi invarian terhadap pergeseran konstan.

### `simulate_hardware_fault(x, stuck_prob=0.08, loss_prob=0.05, seed)` — core:135-145
Satu bilangan acak `rv` per sampel: `rv < 0.08` → nilai diganti nilai dari
indeks acak lain ("stuck"), `rv < 0.05` → NaN ("loss").
- **Kenapa `rv` yang sama untuk dua mask?** Supaya kedua kejadian saling eksklusif
  per sampel dan tak perlu dua generator. Akibatnya loss ⊂ stuck: porsi
  efektif stuck 3 %, loss 5 %, total 8 % (`cek_cacat_kode.py` §6). "Stuck" di
  sini sebenarnya *random replacement*, bukan nilai macet konstan.
- **Ke mana NaN-nya?** Diisi `ffill/bfill` di `inject_faults_multisensor`
  (core:213), jadi data hilang tampil sebagai nilai datar = nilai terakhir.

### `simulate_multiple_faults(x, faults, seed)` — core:148-154
Menerapkan daftar fault berurutan, OR-kan mask, lalu `np.clip(y, 0, 100)`.
- **Kenapa clip 0–100?** Kelembaban tanah dalam % fisik tidak bisa keluar
  0–100; tanpa clip, bias/drift/spike bisa menghasilkan nilai mustahil yang
  terlalu mudah dideteksi (bocoran label dari nilai di luar domain). `np.clip`
  membiarkan NaN tetap NaN.
- **Urutan berpengaruh?** Ya: spike setelah hardware memakai `nanstd`; drift
  setelah hardware menambah tanjakan ke nilai yang sudah diganti. Urutan dikunci
  di `SCENARIOS` (core:169-188) dan `inject_like_training` memakai urutan yang
  sama untuk demo (core:787-794).

### `simulate_choose_one` — core:157-161; `SCENARIOS` — core:169-188
Skenario "faulty" memilih satu dari empat fault secara acak per sensor.
16 kondisi fault + "normal" = `condition_names` (core:189).

---

## 2. Segmentasi dan label

### `make_windows(Xa, win, stride)` — core:195-202
`(T, 4)` → `(nwin, win, 4)` dan `starts (nwin,)`, `nwin = floor((T-win)/stride)+1`.
- **Kenapa `transpose(0, 2, 1)`?** `sliding_window_view(Xn, win, axis=0)` pada
  array `(T,4)` menaruh sumbu jendela **di akhir**: `(T-win+1, 4, win)`.
  Transpose mengembalikan ke `(nwin, win, 4)` supaya `W[i][:, s]` = deret sensor
  `s` (dipakai `_one_window`, core:385).
- **Kenapa `sliding_window_view`?** Hanya *view*, tanpa menyalin data; baru
  disalin saat diindeks `view[starts]`. `float32` menghemat memori.

### `inject_faults_multisensor(Xa, scenario_faults, sensor_subset, seed)` — core:205-215
Menyuntik fault hanya ke kolom sensor di `sensor_subset`, seed per sensor dari
satu RNG induk. Keluaran `Y (T,4)` (NaN sudah diisi) dan `M (T,4) bool`.
- **Kenapa disuntik ke seluruh deret lalu dipotong, bukan per jendela?**
  Episode drift 3 hari lebih panjang dari jendela; hardware "stuck" menyalin
  nilai dari waktu lain. Hanya injeksi di deret penuh yang memberi bentuk fault
  realistis. Demo memakai jalur sama (`inject_like_training`, core:787).
- **Kenapa `ffill().bfill()` lalu median?** Entropi tidak bisa menghitung NaN;
  ffill meniru logger yang mengulang nilai terakhir; bfill untuk NaN di awal;
  median untuk kolom yang seluruhnya NaN.

### `window_labels_per_sensor(mask, win, stride, thr)` — core:218-223
`(T,4)` → `(nwin, 4) bool`: sensor `s` di jendela `i` berlabel fault bila porsi
sampel fault-nya `> thr`. Dipanggil dengan `thr = FAULT_RATIO_THR = 0.01` (core:240).
- **Kenapa ambang 1 %?** Untuk $N=200$ itu > 2 sampel. Jendela yang hanya
  tersentuh 1–2 sampel fault praktis identik dengan jendela normal; memberinya
  label fault = noise label.
- **Kenapa `[::stride]` di sini sama dengan `make_windows`?** `starts =
  arange(0, T-win+1, stride)` dan `[::stride]` pada view berukuran `T-win+1`
  menghasilkan indeks yang sama, sehingga label dan jendela sejajar.

### `segment_and_label(X, win)` — core:226-248
Membuat jendela normal (kelas 0) dari deret asli, lalu untuk tiap 16 kondisi
fault × 8 ulangan (ukuran subset 1,2,3,4,1,2,3,4) menyuntik, memberi label, dan
menyimpan hanya jendela yang punya ≥ 1 sensor fault. Keluaran `W (nwin,win,4)`,
`y (nwin,)` kondisi 0..16, `Ysens (nwin,4)` multi-hot, `starts (nwin,)`.
- **Kenapa stride `win//2` (50 % tumpang-tindih)?** Menggandakan jumlah jendela
  tanpa data baru dan tidak melewatkan fault di batas jendela. Harga yang
  dibayar: jendela bertetangga berbagi 50 % sampel, jadi **CV acak bocor**; ini
  alasan wajib grouped CV (bagian 6).
- **Kenapa jendela fault dengan ≤ 1 % sampel fault dibuang, tidak diberi label
  normal?** (core:242 `keep = sens.any(axis=1)`) Jendela itu berasal dari deret
  yang sudah disuntik; melabelinya "normal" mencampur data hampir-normal ke kelas
  normal, melabelinya "fault" memberi label tanpa isi. Membuang keduanya paling
  bersih. Kelas normal hanya diambil dari deret asli.
- **Kenapa `RANDOM_SEED` lokal (core:228)?** Supaya subset sensor dan seed injeksi
  sama di setiap pemanggilan — notebook 11, 14, `simpan_model.py`, dan demo
  memperoleh jendela identik.

### `balanced_subsample(W, y, Ysens, starts, max_per_class, seed)` — core:251-261
Maks 100 jendela per kondisi, lalu diacak. Keluaran sama bentuknya, lebih kecil.
- **Kenapa 100?** Biaya ekstraksi entropi (2 metode × 4 sensor × 15 skala) dan
  LM padat ($O(P^3)$ per iterasi) dibatasi; seimbang per kondisi juga mencegah
  kondisi yang jendelanya banyak (bias: semua jendela berlabel) mendominasi.

### `build_scenario(y_cond, classes)` — core:421-436 (b11:1156)
Memetakan 17 kondisi ke kelas skenario S1–S5 (`LADDER`, core:402-416) dan
undersample tiap kelas ke ukuran kelas terkecil.
- **Kenapa undersample?** Di S1 kelas "faulty" menggabung 16 kondisi (≈1600
  jendela) vs 100 normal; tanpa penyeimbangan akurasi 94 % bisa dicapai dengan
  menebak "faulty" saja.

---

## 3. Entropi multiskala

### `coarse_grain_cmse(x, s, k)` — core:267-271
Rata-rata blok tak-tumpang ukuran `s` mulai offset `k`; `(n,)` → `((n-k)//s,)`.
- **Kenapa ada offset `k`?** Composite MSE (Wu 2013): di skala besar deret
  coarse-grained pendek; merata-ratakan $s$ versi bergeser mengurangi varians
  estimasi. Costa asli hanya $k=0$.
- **Kenapa `reshape(-1, s).mean(1)`?** Setara rumus Costa
  $\frac1\tau\sum$ tanpa loop Python.

### `embed_matrix_centered(y, m_)` — core:274-278
`(L,)` → `(L-m+1, m)`, tiap baris dikurangi rata-ratanya.
- **Kenapa dipusatkan?** Mengikuti Chen 2007 (FuzzyEn mengurangi baseline
  lokal): yang dibandingkan **bentuk** pola, bukan level. Efek sampingnya:
  fault bias tak terlihat (Cacat C2).

### `_pick_ref` — core:281-283; `phi_euclidean(V, r, n_ref, seed)` — core:286-297
`V (Nm, m)` → skalar $\phi$. Jarak Euclid dari ≤ 128 vektor acuan ke semua
vektor lewat $\lVert a\rVert^2 + \lVert b\rVert^2 - 2a\cdot b$, keanggotaan
$1/(1+d/r)$, diagonal acuan dinolkan, rata-rata baris dibagi $N-1$.
- **Kenapa `n_ref = 128`?** Matriks jarak penuh $O(N^2)$: untuk $N=1000$ itu 1 juta
  pasangan × 15 skala × banyak offset × 4 sensor × ±1700 jendela × 2 metode.
  Dengan 128 acuan biayanya $O(128N)$; $\phi$ adalah rata-rata, sehingga
  subsampel acuan memberi penduga tak bias dengan varians kecil.
- **Kenapa `np.maximum(..., 0)` sebelum `sqrt`?** Rumus ekspansi kuadrat bisa
  memberi −1e-16 akibat pembulatan.
- **Kenapa `1/(1+d/r)` bukan `exp(-d^n/r)`?** Bukan dari Chen; pilihan rancangan
  (dipakai juga di notebook-notebook sebelumnya). Keanggotaan rasional tidak
  pernah 0 dan turun lebih lambat, sehingga $\phi>0$ selalu dan $\ln$ aman. Ini
  harus disebut sebagai modifikasi, bukan FuzzyEn baku.
- **Kenapa `r + EPS`?** Jendela datar (sd = 0, mis. sensor macet panjang) →
  $r=0$ → pembagian nol.

### `edm_fuzzy_entropy_1d(x, scales, m, r_ratio, n_ref, seed)` — core:300-314
`(n,)` → `(15,)` entropi per skala.
- **Kenapa $r$ dihitung sekali dari deret asli (core:302)?** Costa: $r$ tetap
  antarskala supaya penurunan varians akibat perataan ikut terukur sebagai
  perubahan entropi.
- **Kenapa `len(y) < m+2` dilewati dan syarat `phi > 0`, `isfinite`?** Butuh
  ≥ 3 vektor $m+1$ untuk $\phi$ (core:288); $\ln$ tidak terdefinisi untuk 0/NaN.
  Bila semua offset gagal, skala itu NaN → `sanitize`.
- **Kenapa seed berbeda untuk $\phi^m$ dan $\phi^{m+1}$ (core:309-310)?** Tidak
  ada alasan teoretis; acuan berbeda menambah sedikit varians. Lebih rapi bila
  acuan sama (Chen memakai himpunan $i$ yang sama).

### `vectors_to_distributions(V, centers, h)` — core:317-321
`(Nm, m)` → `(Nm, 8)`: histogram lunak berkernel Gauss, tiap baris dijumlah 1.
- **Kenapa kernel, bukan histogram keras?** Dengan $m=2$ atau 3 nilai, histogram
  keras hampir selalu 1–3 bin bernilai 1 lainnya 0 → JS sering $\ln 2$ (maksimum)
  dan tak stabil terhadap pergeseran kecil. Kernel memberi distribusi halus,
  kontinu terhadap $v$.
- **Kenapa `+ 1e-300` di penyebut?** Bila semua nilai jauh di luar ±3 sd, semua
  $K\approx0$.

### `pairwise_jsd(A, B)` — core:324-331
`(nA, 8)`, `(nB, 8)` → `(nA, nB)` JS natural log (≤ ln 2).
- **Kenapa `EPS` di dalam log?** $0\log0$ didefinisikan 0; `a*log(a+EPS)` memberi
  0 tanpa peringatan.
- **Kenapa `np.maximum(d, 0)`?** Pembulatan bisa membuat JS ≈ −1e-17, lalu
  `sqrt` NaN.

### `phi_jsd` — core:334-343; `jsd_fuzzy_entropy_1d` — core:346-365
Sama dengan jalur EDM, tetapi jarak = $\sqrt{\mathrm{JS}}$ antar distribusi
vektor, ambang $r_\text{JSD} = 0{,}2\sqrt{\ln2}$.
- **Kenapa $\sqrt{\mathrm{JS}}$, bukan JS?** $\sqrt{\mathrm{JS}}$ adalah metrik
  (Endres–Schindelin 2003), sejajar dengan jarak Euclid di EDM.
- **Kenapa $r_\text{JSD} = 0{,}2\sqrt{\ln 2}$?** Batas atas jarak JS natural =
  $\sqrt{\ln2}$; 0,2 meniru `r_ratio = 0.2` di EDM, jadi toleransi = 20 % dari
  rentang jarak.
- **Kenapa pusat bin ±3 sd dari deret jendela?** Vektor terpusat berada di
  sekitar 0 dengan sebaran ≈ sd; ±3 sd mencakup hampir semua nilai.
- **Pertanyaan tajam:** histogram tidak mengenal urutan — vektor naik dan turun
  dianggap kembar (03, bagian c), dan entropi bisa negatif. Jawaban jujur: JSD
  mengukur kemiripan **sebaran nilai** dalam pola, bukan bentuk berurut; itu
  perbedaan konseptual dari EDM, bukan kesalahan implementasi. Parameter
  `r_ratio` di fungsi ini tidak dipakai.

### `sanitize(F)` — core:368-372
Isi NaN dengan median kolom. Kolom yang seluruhnya NaN tetap NaN (lalu
`SimpleImputer` di pipeline). Dipanggil sebelum CV, tetapi median kolom
fitur tanpa label → tidak membocorkan label; di jendela 200/700/1000 NaN
praktis tidak muncul ($n \ge 5\tau$).

### `entropy_1d` / `_one_window` / `concat_multisensor_features` — core:375-395
`(nwin, win, 4)` → `(nwin, 60)`: 15 skala × 4 sensor digabung.
- **Kenapa seed `seed + 1000*i` dan `+ 19*s`?** Setiap jendela dan sensor
  mendapat subsampel acuan sendiri yang deterministik, **dan sama untuk EDM dan
  JSD** — perbandingan adil.
- **Kenapa `joblib` proses?** Perhitungan numpy kecil-kecil per jendela;
  paralel per jendela memakai semua inti.

---

## 4. Jaringan saraf Levenberg–Marquardt

### `lm_init_layers(n_in, hidden, n_out, seed)` — core:442-449
Daftar `[W (d_i, d_{i+1}), b (d_{i+1},)]`, $W \sim U(\pm\sqrt{6/(d_i+d_{i+1})})$, $b=0$.
- **Kenapa Xavier?** Menjaga varians aktivasi tanh di daerah linear saat awal
  sehingga $1-h^2$ tidak jenuh. Untuk ReLU, He init lebih lazim; kode memakai
  Xavier untuk keduanya (komentar b11:1434).
- **Tetapi:** niat Xavier ini **tidak sampai** ke jaringan karena cacat
  pack/unpack (bagian "Cacat yang diketahui", C1).

### `lm_pack(layers)` — core:452-453 dan `lm_unpack(theta, template)` — core:456-461
`lm_pack`: list → vektor `(P,)` berurutan **[W1, W2, …, b1, b2, …]**.
`lm_unpack`: vektor → list, membaca **[W1, b1, W2, b2, …]**.
- **Kenapa vektor datar?** LM bekerja pada satu vektor parameter (Jacobian
  `(m·C, P)`, `solve` `P×P`).
- **Kenapa `template`?** Hanya untuk bentuk (shape) tiap lapis.
- Lihat C1.

### `lm_forward(layers, X, activation)` — core:464-474
`(n, n_in)` → keluaran `(n, n_out)` dan daftar aktivasi tiap lapis.
- **Kenapa lapis terakhir linear?** Target one-hot + kuadrat galat: keluaran
  linear membuat Jacobian lapis terakhir sederhana ($\delta=I$) dan LM klasik
  memang untuk least squares. Softmax hanya saat prediksi.

### `lm_residuals` — core:477-479; `lm_residual_jacobian` — core:482-508
`(P,)` → `r (n·C,)` dan `J (n·C, P)` analitik.
- **Kenapa Jacobian residual per keluaran, bukan gradien loss?** LM butuh
  $J$ penuh untuk $J^\top J$; backprop biasa hanya memberi $J^\top r$. Kode
  merambatkan $\delta$ berbentuk `(n, C, units)` = satu backprop per keluaran.
- **Kenapa urutan kolom W lalu b per lapis (core:503-507)?** Harus sama dengan
  `lm_unpack` — dan memang sama (dicek numerik, 03 bagian d).
- **Biaya memori:** `(n·C) × P`; untuk (64,16), 60 fitur, 7 kelas, P = 5063 dan
  ±560 jendela latih → ~3900 × 5063 ≈ 160 MB. Ini alasan grid arsitektur kecil
  (komentar b11:1604-1606).

### `lm_fit(X, Y, n_in, hidden, n_out, activation, iters, seed, lam0, max_train_s)` — core:511-567
Loop LM (rinci di 03 bagian a). Keluaran `theta`, `template`, `hist`, `train_mse`.
- **Kenapa `diag(JᵀJ)` bukan $I$?** Marquardt 1963: invarian skala parameter.
- **Kenapa dijepit `1e-12` (core:537)?** Kolom Jacobian nol (unit ReLU mati,
  atau baris W nol akibat C1) → diagonal 0 → $A+\lambda D$ singular.
- **Kenapa `max_train_s` (core:522)?** `solve` $O(P^3)$; satu arsitektur besar
  bisa memakan menit per iterasi. Budget waktu (40 s pencarian, 150 s refit;
  core:73-74) menjamin seluruh nested CV (30 run × 5 fold × 16 kombinasi × 2
  inner fold) selesai dalam batas 12 jam Kaggle. Status `budget(MAX_TRAIN_S)`
  dicatat di `hist` dan dihitung di b11:1679-1680, jadi jujur terlihat.
- **Kenapa berhenti di `gn < 1e-10` dan penurunan relatif `1e-12`?** Kriteria
  stasioner dan stagnasi standar (Madsen–Nielsen–Tingleff).
- **Lihat juga:** iterasi ditolak menghabiskan jatah iterasi (03, a.1 butir 5).

### `lm_predict_proba` — core:570-575
Softmax stabil (`out - max`) atas keluaran linear. Probabilitas tidak
terkalibrasi; ROC-AUC (b11) hanya butuh urutan skor, jadi tetap sah.

### `LMEstimator` — core:578-608
Pembungkus scikit-learn: label → one-hot (`searchsorted` pada `classes_`), lalu
`lm_fit`.
- **Kenapa `ClassifierMixin` lebih dulu?** scikit-learn ≥ 1.6 memeriksa urutan
  MRO untuk tag estimator (komentar b11:1559).
- **Kenapa `__init__` hanya menyimpan argumen?** Kontrak `BaseEstimator`
  (`get_params/clone`).

### `LMSensorID` — core:611-637
Satu LM untuk target multi-hot 4 sensor; keluaran linear, ambang 0,5.
- **Kenapa satu model 4 keluaran, bukan 4 model?** Berbagi lapis tersembunyi;
  hemat waktu; tak butuh `MultiOutputClassifier`.
- **Kenapa ambang 0,5?** Target 0/1 dengan MSE → keluaran linear menduga
  $P(\text{fault}\mid x)$ secara kasar; 0,5 = titik tengah.

### `make_lm_pipeline` / `make_sensor_pipeline` — core:640-655
`SimpleImputer(median) → StandardScaler → LM`.
- **Kenapa di dalam pipeline?** Median dan mean/sd di-*fit* hanya pada fold latih
  → tidak bocor ke fold uji.

---

## 5. Pencarian arsitektur

### `_inner_combo_score` — core:661-673; `inner_search_best(Ftr, ytr, gtr, ...)` — core:676-684
16 kombinasi (8 arsitektur × tanh/relu) dinilai dengan `StratifiedGroupKFold`
2 lipatan **di dalam** data latih luar, skor macro-F1, diurutkan.
- **Kenapa grup juga di inner CV?** Sama seperti luar: jendela bertumpang-tindih
  tidak boleh terbelah antara latih-dalam dan validasi-dalam.
- **Kenapa hanya 6 iterasi LM saat pencarian?** Peringkat arsitektur cukup
  dibaca dari awal kurva; refit 40 iterasi. Risikonya: dengan penolakan beruntun
  (03 bagian a) 6 iterasi bisa nyaris tak bergerak, sehingga peringkat agak
  acak.
- **Kenapa `except: return -1.0` (core:672)?** Satu kombinasi yang gagal
  (mis. memori) tidak menggagalkan seluruh run; b11 mencatat statusnya. Di core
  versi ini pesan errornya dibuang — lebih baik dicatat.

---

## 6. Validasi di notebook 11 (b11)

### Grup CV — b11:601 (`groups = starts // WIN`)
- **Kenapa grouped CV wajib?** Stride N/2 → jendela bertetangga berbagi 50 %
  sampel; jendela kondisi fault berbeda pada `starts` yang sama berasal dari
  **potongan deret asli yang sama**. CV acak menaruh "kembaran" di latih dan uji
  → skor terlalu tinggi. Grup = blok waktu memastikan potongan yang sama tidak
  muncul di kedua sisi.
- **Sisa kebocoran:** lihat C3 — batas antar grup masih tumpang-tindih 50 %.

### `nested_grouped_cv(F, y, groups)` — b11:1657-1720
Luar: `StratifiedGroupKFold(k)`, `k = min(5, #grup, kelas terkecil)` (b11:1659);
dalam: `inner_search_best` hanya pada fold latih; refit arsitektur terbaik 40
iterasi; prediksi OOF untuk confusion matrix dan rantai sensor-ID.
- **Kenapa nested?** Memilih arsitektur dengan data uji yang sama → skor optimis.
  Nested memisahkan pemilihan model dari penilaian.
- **Kenapa Stratified?** Tiap fold memuat semua kelas meski grup memaksa
  pembagian kasar.

### Uji berpasangan — b11:1929-1950 (`_paired_test`)
Paired t-test pada F1 5 skenario (EDM vs JSD) per panjang jendela, plus
Wilcoxon dan Cohen's $d_z = \bar d/s_d$.
- **Kenapa berpasangan?** Skenario sama, fold sama, fitur beda → selisih per
  skenario menghilangkan variasi tingkat kesulitan skenario.
- **Pertanyaan tajam:** $n=5$ pasangan, df = 4, daya uji rendah. **Wilcoxon
  eksak dua sisi dengan $n=5$ tidak pernah bisa $p<0{,}05$** (p minimum
  $2/2^5 = 0{,}0625$), jadi kolom Wilcoxon hanya pendamping. Uji tingkat fold
  (b11:2011-2024) melanggar independensi (fold berbagi data) dan sudah ditandai
  "bukan uji utama" di kode.

### Rantai sensor-ID — b11:2054-2097
Hanya jendela yang **diprediksi** fault oleh tahap 1 (OOF, b11:2059) yang
diteruskan; target `Ysens` multi-hot; arsitektur = modus pilihan grid
(b11:2069); `lm_fit` langsung, ambang 0,5.
- **Kenapa dirantai dari OOF?** Kesalahan tahap 1 (normal diprediksi fault →
  target semua 0; fault terlewat → hilang) ikut terhitung, seperti sistem nyata.
- **Cacat:** pembagian latih/uji tahap ini acak, bukan grouped (C4).

---

## 7. Notebook 14 dan 15

### b14 — bentuk fault baru (b14:50-104) dan label "semua komponen ada" (b14:123-143)
Builder menambal teks b11: drift linear per episode (total 5 poin), bias
episode 5 poin, spike relatif ×1,30 acak 1,52 %, hardware dibatasi ke episode
di kombinasi; mask jadi **bitmask** (`FAULT_BIT`) supaya label jendela kombinasi
mensyaratkan setiap komponen hadir (spike ≥ 1 titik, lainnya > 1 %).
- **Kenapa bitmask?** Mask bool tidak tahu komponen mana yang ada; jendela
  "drift+spike" yang hanya berisi spike akan berlabel kombinasi padahal tidak.
- **Kenapa `seed + 1`, `seed + 2` (b14:69, b14:83)?** Spike dan hardware di satu
  kombinasi menerima seed yang sama; offset membuat aliran acaknya berbeda.
- **Kenapa `sub1` meng-assert tepat satu kemunculan (b14:29-32)?** Tambalan
  berbasis teks; bila b11 berubah, builder gagal keras daripada diam-diam
  menghasilkan notebook salah.

### b15 — fitur lag tanpa entropi, simpan `.npz`
- `lag_features` (b15:437-445): `lag1..lag10`, `diff`, `isna`, `isna_10` per
  sampel. **Kenapa `np.roll`?** Cepat; baris awal yang "membungkus" dari ujung
  deret dibuang lewat `valid_row` (b15:432).
- **Pembagian per blok 1 hari 3:1:1 (b15:427-431)?** Pembagian kronologis
  60/20/20 menaruh semua episode di latih; blok bergilir menyebar episode ke
  latih/validasi/uji. Lag 10 sampel di batas blok tetap bocor sedikit (diakui di
  teks b15:416).
- **Kenapa `.npz` + `allow_pickle=False` (b15:789, b15:736)?** File hanya berisi
  array angka; memuatnya tidak mengeksekusi kode (pickle bisa), dan tidak
  bergantung versi scikit-learn. Pembaca `prob_fault` (b15:751-765) menulis ulang
  maju jaringan dengan urutan $W_1,b_1,W_2,b_2$ = `lm_unpack` (diverifikasi
  sama, `cek_cacat_kode.py` §2), dan setiap file dimuat ulang lalu dibandingkan
  dengan prediksi di memori (assert < 1e-9, b15:790-805).

### `SavedLM` — core:709-766 (dipakai `simpan_model.py:133-134`)
Menyimpan `fill` (median imputer), `mean`, `scale`, `theta`, `shapes`,
`activation`, `classes`; prediksi = isi NaN → standarisasi → `lm_forward(lm_unpack(...))`.
- **Kenapa tidak mem-pickle Pipeline sklearn?** Pickle Pipeline menyimpan atribut
  internal `SimpleImputer`/`StandardScaler`; Kaggle (sklearn 1.6.1) vs laptop
  (1.9.1) gagal dibuka (docstring core:712-716). `SavedLM` hanya berisi array
  numpy + satu kelas dari core.
- **Masih pickle?** Ya: `simpan_model.py:140` memakai `joblib.dump` (pickle) untuk
  bundle. Bedanya, isi bundle tidak bergantung versi sklearn; file tetap harus
  dari sumber tepercaya. Format bebas-pickle sungguhan = `.npz` di b15.
- **Konsisten dengan pelatihan?** Ya: `_template()` hanya membuat bentuk, dan
  `lm_unpack` dipakai sama seperti saat `fit`/`predict`
  (`cek_cacat_kode.py` §2: selisih 0,0 untuk kelas dan sensor).

### `condition_for` / `inject_like_training` — core:775-794
Memetakan pilihan fault demo (mis. {spike, hardware}) ke nama kondisi
("spike+malfunc") dan menyuntik ke **deret penuh** dengan urutan komponen yang
sama dengan pelatihan.
- **Kenapa tidak menyuntik ke satu jendela saja (seperti `inject_demo`,
  core:698)?** Episode drift dan salinan "stuck" butuh konteks deret penuh;
  menyuntik per jendela menghasilkan bentuk fault yang tidak pernah dilihat model.

---

## Cacat yang diketahui

### C1. `lm_pack` dan `lm_unpack` tidak saling invers (core:452-461; salinan identik b11:1438-1445)

**Faktanya.** `lm_pack` menulis $[W_1, W_2, \dots, W_L, b_1, \dots, b_L]$;
`lm_unpack` membaca $[W_1, b_1, W_2, b_2, \dots]$. Hanya dipakai satu kali
per pelatihan: `theta = lm_pack(template)` di `lm_fit` (core:514). Setelah itu
**semua** jalur memakai `lm_unpack` dengan urutan yang sama: `lm_residuals`,
`lm_residual_jacobian` (kolom J disusun W,b per lapis, core:503-507),
`lm_predict_proba`, `LMSensorID.decision_function`, rantai sensor-ID
(b11:2072), `SavedLM.decision_function` (core:754), dan pembaca `.npz`
notebook 15 (b15:755-760). Jadi pelatihan, prediksi, dan model tersimpan
**konsisten**; yang salah hanya **nilai awal** $\theta_0$.

**Apa yang sebenarnya terjadi pada $\theta_0$** (diverifikasi,
`cek_cacat_kode.py` §1):

- $W_1$ selalu benar (Xavier).
- $b_1$ berisi $n_1$ bobot pertama $W_2$ (bukan nol).
- Lapis berikutnya bergeser; nol-nol bias asli terdorong ke ekor vektor sehingga
  **lapis terakhir sebagian/seluruhnya nol**, dan $b_L=0$.
- Contoh 2-2-2: $b_1 = [0.767, 1.011]$, $W_2 = [[0.261, 0.562],[0,0]]$.
- Konfigurasi grid penelitian (60 fitur):

| arsitektur | keluaran | $W$ terakhir nol | unit tersembunyi terakhir tanpa bobot keluar |
|---|---|---|---|
| 1 lapis (16/32/64) | 2 (S1, S5) | 50 % | separuh unit |
| 1 lapis | 5 (S2, S4) | 20 % | 3/16, 6/32, 12/64 |
| 1 lapis | 7 (S3) | 14,3 % | 2/16, 4/32, 9/64 |
| 1 lapis | 4 (sensor-ID) | 25 % | seperempat unit |
| 2 lapis (16,8), (32,8), (32,16), (32,32), (64,16) | 2 | **100 %** | semua |
| (32,8), (64,16) | 5 dan 4 | **100 %** | semua |
| 2 lapis lain | 5/7/4 | 29–75 % | sebagian |

Pada kasus 100 %: keluaran awal = 0 untuk semua masukan (MSE awal tepat 0,5
untuk 2 kelas), dan **semua kolom Jacobian lapis 1–2 nol** di iterasi pertama
(dicek: 0 dari 124 kolom tak nol pada jaringan 10-[8,4]-2). Iterasi pertama
hanya bisa menggerakkan lapis terakhir; mulai iterasi kedua semua lapis ikut.
Simetri tidak terjadi karena $W_1$ (dan sebagian lapis tengah) tetap acak.

**Dampak terukur (data mainan, `cek_cacat_kode.py` §3, 5 seed):**

| kasus | MSE latih akhir, kode asli | MSE latih akhir, pack diperbaiki |
|---|---|---|
| [8], 3 kelas, 6 iterasi | 0,171 | 0,211 |
| [8], 3 kelas, 15 iterasi | 0,062 | 0,064 |
| [8,4], 2 kelas, 6 iterasi | 0,176 | 0,203 |
| [8,4], 2 kelas, 15 iterasi | 0,017 | 0,060 |

Di contoh ini init yang "teracak" **tidak lebih buruk** (bahkan sedikit lebih
baik: keluaran awal mendekati 0 dekat dengan target one-hot rata-rata). Ini
bukan bukti umum, tetapi menunjukkan dampaknya kecil dan tidak searah.

**Jawaban jujur untuk penguji:**

> "Ya, ada ketidaksesuaian urutan antara `lm_pack` dan `lm_unpack`. Karena
> semua perhitungan sesudah inisialisasi — residual, Jacobian, langkah LM,
> prediksi, dan model yang disimpan — memakai `lm_unpack` yang sama, jaringan
> yang dilatih dan yang dievaluasi adalah jaringan yang sama; tidak ada bobot
> yang tertukar saat prediksi. Akibatnya hanya pada **titik awal**: bias lapis
> pertama terisi angka kecil dan sebagian bobot lapis terakhir mulai dari nol,
> jadi inisialisasinya bukan Xavier murni seperti yang saya tulis. LM adalah
> metode orde dua; sesudah satu iterasi diterima, lapis terakhir sudah terisi,
> dan pada uji kecil hasil akhirnya setara dengan inisialisasi yang benar.
> Untuk perbandingan EDM-Fuzzy vs JSD-Fuzzy, kedua metode memakai fungsi
> pelatihan yang sama, seed yang sama (`RANDOM_SEED=42`), grid arsitektur yang
> sama, dan pembagian fold yang sama, sehingga inisialisasi yang sama-sama
> menyimpang itu tidak menguntungkan salah satu metode. Yang perlu saya ralat
> di naskah adalah klaim 'inisialisasi Xavier dengan bias nol'."

**Perbaikan** (satu baris; perlu menjalankan ulang notebook karena $\theta_0$
berubah, sehingga angka akan bergeser sedikit):

```python
def lm_pack(layers):
    return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in layers])
```

Sesudah perbaikan `lm_unpack(lm_pack(t)) == t` (dicek). Model yang sudah
tersimpan tetap sah karena `theta` tersimpan dibaca dengan `lm_unpack` yang tidak
berubah.

### C2. Fault bias konstan tidak terlihat oleh fitur entropi (core:131-132, 274-278, 302, 351)

Kedua entropi memusatkan tiap vektor (mengurangi rata-ratanya) dan menghitung
$r$ serta pusat bin dari sd jendela. Menambah konstanta $b$ ke seluruh deret
tidak mengubah vektor terpusat, sd, $r$, maupun pusat bin, jadi fitur **identik**
(dicek: $\max|F(x+0{,}08)-F(x)| \approx 10^{-14}$ untuk EDM, $10^{-12}$ untuk JSD;
sama untuk $x+5$). Jendela bias di notebook 11 memakai `starts` yang sama dengan
jendela normal, sehingga fiturnya kembar dengan jendela normal kecuali di mana
hardware/clip ikut berperan.

Bukti di hasil: recall kelas bias pada S2 notebook 11 = 19 % / 21 % (N=200,
EDM/JSD), 35 % / 37 % (N=700), 29 % / 20 % (N=1000) dengan peluang acak 20 %
(5 kelas) — `hasil11/exports/11_rq4_confusion_matrix.csv`. Jendela bias juga
masuk kelas "faulty" S1, sehingga membatasi skor S1.

**Jawaban jujur:** "Entropi mengukur keteraturan pola, bukan level; bias
konstan secara matematis tidak terdeteksi oleh fitur ini, dan hasil kelas bias
memang setingkat tebakan. Deteksi bias butuh fitur level/perbandingan antar
sensor (mis. selisih terhadap median 4 sensor) — itulah alasan notebook 14
memakai bias **episode** (ada tepi naik-turun) dan notebook 15 memakai fitur lag
yang berisi level." Ini berlaku sama untuk EDM dan JSD, jadi tidak membiaskan
perbandingan keduanya, tetapi menurunkan skor absolut S1–S4.

### C3. Grup CV `starts // WIN` masih bocor di batas grup (b11:601, `simpan_model.py:85`)

Dengan stride $N/2$, grup $g$ berisi jendela mulai $gN$ dan $gN + N/2$. Jendela
$gN + N/2$ menutup $[gN+N/2,\,(g+1)N + N/2)$, separuhnya sama dengan jendela
pertama grup $g+1$. Pada 40 jendela beruntun, 19 dari 39 pasangan bertumpang
tindih (49 %) berada di grup **berbeda** (`cek_cacat_kode.py` §5). Bila dua grup
bertetangga jatuh di fold berbeda, 50 % sampel jendela uji sudah dilihat saat
latih (dengan fault yang mungkin berbeda).

Dampak: optimistis untuk **kedua** metode secara sama (fold identik), jadi
perbandingan relatif tetap adil; skor absolut sedikit terlalu tinggi.
Perbaikan: grup lebih besar (mis. `starts // (k·WIN)`) **dan** buang
(*purge*) jendela yang melintasi batas grup, atau `stride = WIN` untuk evaluasi.

### C4. Sensor-ID memakai `train_test_split` acak (b11:2065)

Tahap 2 membagi jendela yang lolos tahap 1 secara acak 70/30, bukan per grup.
Jendela bertumpang-tindih dan jendela dari potongan deret yang sama (kondisi
fault berbeda) bisa berada di latih dan uji → F1 sensor-ID optimistis (untuk
kedua metode). Perbaikan: `GroupShuffleSplit` dengan `groups[keep][mask]`.

### C5. Hal kecil yang perlu diketahui (bukan penyebab hasil salah)

- Aturan $\lambda$ saat ditolak tetap ×2 (core:562), bukan $\nu$ berlipat
  (Nielsen); iterasi ditolak memakan jatah `iters` (03, a.1).
- `phi_euclidean`/`phi_jsd` memakai acuan acak berbeda untuk $m$ dan $m+1$
  (core:309-310, 360-361).
- `simulate_hardware_fault`: loss ⊂ stuck, stuck efektif 3 % (core:140-143).
- `simulate_drift/spike/bias_fault` mengabaikan `seed`; spike periodik di posisi
  sama di semua sensor (core:125-128).
- `jsd_fuzzy_entropy_1d(r_ratio=...)` tidak dipakai (core:346).
- `_inner_combo_score` di core membuang pesan error (core:672-673).
- Entropi JSD bisa negatif (03, bagian c) — perlu disebut bila penguji melihat
  nilai < 0 di kurva.

---

## Ringkasan keluaran `cek_cacat_kode.py` (2026-11-20)

```
1. lm_pack vs lm_unpack (2-2-2, seed 0)
  theta = lm_pack = [0.3355 -0.5639 -1.1244 -1.1843 0.7674 1.011 0.2612 0.5621 0 0 0 0]
  dipakai jaringan: W1 benar; b1 = [0.7674 1.011]; W2 = [[0.2612 0.5621],[0 0]]; b2 = [0 0]
  lm_unpack(pack_benar(t)) == t -> OK ; lm_pack(lm_unpack(theta)) == theta ? False
2. Pipeline vs SavedLM predict_proba: maks selisih = 0.00e+00
   Pipeline sensor vs SavedLM decision_function: maks selisih = 0.00e+00
   Urutan baca prob_fault (notebook 15) == lm_unpack ? True
3. [8,4]->2 di theta awal: kolom Jacobian lapis 1-2 yang tak nol = 0/124; lapis 3 = 10/10
4. EDM: maks |F(x+0.08) - F(x)| = 1.44e-14 ; maks |F(x+5) - F(x)| = 6.88e-15
   JSD: maks |F(x+0.08) - F(x)| = 1.48e-12 ; maks |F(x+5) - F(x)| = 3.73e-13
5. WIN=200/700/1000: 39 pasangan bertumpang-tindih, 19 di grup BERBEDA (49%), tumpang-tindih 50%
6. porsi NaN (loss) = 0.0500 ; 'stuck' tersisa = 0.0300 ; total mask = 0.0800
7. posisi spike seed=1 dan seed=999: [0 66 132 198 ...] identik: True
   tinggi spike 0.08*sd = 0.0565 vs r = 0.2*sd = 0.1415
8. JSD-FuzzyEn deret 8 titik tau=1: -0.1151
```
