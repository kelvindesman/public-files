# 07 — Pustaka dan Posisi Kebaruan (JSD-Fuzzy vs EDM-Fuzzy + ANN-LM)

Tanggal penelusuran: **2026-11-20**. Semua referensi di bagian (b) dan (d) sudah dicek ada (judul, penulis, tahun, venue, DOI) lewat Crossref API / OpenAlex / halaman penerbit. BibTeX lengkap: `sidang/qa/pustaka.bib` (51 entri, kunci seperti `chen2007fuzzyen`). Yang tidak bisa dicek ada di bagian (e).

Ringkasan metode yang diposisikan (dari `deteksi_fault_core.py`):
- **EDM-Fuzzy** (`phi_euclidean`, `edm_fuzzy_entropy_1d`): vektor embedding dipusatkan (dikurangi rerata), jarak **Euclidean** antar vektor, keanggotaan fuzzy `mu = 1/(1 + d/r)`, r = 0,2·SD, 128 vektor referensi acak, coarse-graining komposit (`coarse_grain_cmse`), skala 1..15, m = 2.
- **JSD-Fuzzy** (`vectors_to_distributions`, `pairwise_jsd`, `phi_jsd`, `jsd_fuzzy_entropy_1d`): tiap vektor embedding terpusat diubah jadi **histogram lunak** (8 bin pada ±3 SD, kernel Gauss), lalu jarak antar vektor = **akar JSD** (jarak Jensen–Shannon, metrik menurut Endres & Schindelin 2003), dimasukkan ke fungsi keanggotaan yang sama `mu = 1/(1 + d/r_jsd)` dengan r_jsd = 0,2·√ln2.

---

## (a) Paper dasar EDM-Fuzzy: hasil pencarian

**Kesimpulan: paper "ACS" milik mahasiswa (EDM-Fuzzy + ANN-LM untuk fault sensor kelembaban tanah) TIDAK ditemukan** di Crossref, OpenAlex, Semantic Scholar (kena rate-limit), maupun web search. Kemungkinan belum terindeks, masih dalam proses, atau terbit dengan judul yang tidak memuat kata kunci yang dicari.

Yang ditemukan:

1. **Asal metode EDM-Fuzzy (pasti):** Zhou, R., Wang, X., Wan, J., & Xiong, N. (2021). *EDM-Fuzzy: An Euclidean Distance Based Multiscale Fuzzy Entropy Technology for Diagnosing Faults of Industrial Systems.* IEEE Transactions on Industrial Informatics, 17(6), 4046–4054. https://doi.org/10.1109/TII.2020.3009139. Abstraknya menyebut dua kelemahan SampEn yang diperbaiki: (1) jarak Chebyshev (maksimum elemen) "not able to fully reflect the distance of the two vectors", (2) kemiripan biner 0/1 menyebabkan lompatan nilai entropi. EDM-Fuzzy memakai **jarak Euclidean + kemiripan kontinu 0..1**. Diuji pada data bearing. Penulisnya dari Tiongkok, bukan Indonesia. Penerapan kedua oleh kelompok yang sama: Zhou dkk. (2021), *Security and Communication Networks*, https://doi.org/10.1155/2021/5560185 (trafik jaringan anomali).
   → **Wajib disitasi** sebagai sumber baseline; jangan sebut EDM-Fuzzy sebagai metode buatan paper pertama mahasiswa.

2. **Kandidat terdekat untuk "paper pertama" mahasiswa (cocok konteks, tapi tanpa EDM-Fuzzy di abstrak):** Wardhani, L. K., Buono, A., Wahjuni, S., & Syukur, M. (2026). *Evaluation of ANN–Levenberg Marquardt Models for Fault Detection in Smart Farming System.* JITK (Jurnal Ilmu Pengetahuan dan Teknologi Komputer), 11(4), 1053–1060. https://doi.org/10.33480/jitk.v11i4.7481 (IPB University; terbit 6 Mei 2026).
   Bukti kecocokan dengan repo: data **hipotetik** kelembaban tanah sistem smart farming **cabai**, empat fault **drift, bias, spike, malfunction**, **lima skenario** deteksi, ANN-LM. Semua ini sama dengan notebook (S1..S5, `simulate_drift_fault`, "hardware malfunction").
   Ketidakcocokan: abstrak (Crossref dan Garuda) **tidak menyebut fuzzy entropy / EDM-Fuzzy / multiscale**; venue-nya JITK, bukan "ACS". Jadi paper ini kemungkinan **paper pendamping/sebelumnya dari kelompok yang sama**, bukan paper ACS yang dirujuk builder.
   → Tindakan: tanyakan ke mahasiswa judul/DOI paper ACS; bila belum terbit, sitasi sebagai "accepted/in press" dengan judul persisnya. Sementara itu Wardhani dkk. (2026) aman disitasi untuk ANN-LM di konteks yang sama.

3. Tidak ditemukan karya lain (Indonesia atau luar) yang menggabungkan *multiscale fuzzy entropy* + *ANN-LM* + *sensor kelembaban tanah*. Karya MSE terdekat pada data kelembaban tanah hanya analisis skala/regionalisasi (mis. SMOSMANIA, Sun & Niu 2019 *Entropy* 21(6):548), bukan deteksi fault.

---

## (b) Prior-art dan posisi kebaruan

Pertanyaan: apakah sudah ada yang memakai JSD / KL / jarak berbasis distribusi sebagai **ukuran kemiripan antar vektor embedding di dalam** SampEn / FuzzyEn / ApEn / MSE?

**Jawaban singkat:** tidak ditemukan. Yang sudah ada terbagi tiga keluarga, dan tidak satu pun sama dengan JSD-Fuzzy:
1. **Ganti metrik jarak antar vektor** di dalam SampEn/FuzzyEn — tetapi metriknya geometris (Euclidean, kosinus, range, sigmoid), bukan divergensi distribusi.
2. **Entropi dari distribusi jarak** (DistEn, DSEn, diversity entropy) — distribusi dibentuk dari *kumpulan jarak*, lalu dihitung Shannon entropy-nya; vektor tidak diubah jadi distribusi, dan tidak ada divergensi antar pasangan vektor.
3. **JSD pada pola ordinal** (kompleksitas statistik Rosso, PJSD, iraversibilitas) — JSD dihitung antara **distribusi pola permutasi seluruh sinyal** dan distribusi acuan (uniform / sinyal lain / sinyal terbalik), bukan antara dua vektor embedding di dalam penghitungan pencocokan template.

| Karya | Apa yang mirip | Beda dengan JSD-Fuzzy | URL/DOI |
|---|---|---|---|
| Zhou dkk. 2021, EDM-Fuzzy (IEEE TII) | Multiscale fuzzy entropy, ganti Chebyshev → Euclidean, kemiripan kontinu; dipakai untuk diagnosis fault | Jarak geometris antar vektor mentah; tidak ada konversi ke distribusi atau divergensi. Ini **baseline** kita. | https://doi.org/10.1109/TII.2020.3009139 |
| Chen dkk. 2007, FuzzyEn (IEEE TNSRE) | Keanggotaan fuzzy eksponensial, vektor dipusatkan (baseline dibuang) | Jarak Chebyshev (maks. selisih elemen) | https://doi.org/10.1109/TNSRE.2007.897025 |
| Xie dkk. 2008, NS-SampEn (Phys. Lett. A) | Ganti fungsi Heaviside dengan fungsi kemiripan nonlinier (sigmoid) | Mengubah fungsi kemiripan, jarak tetap Chebyshev; tidak berbasis distribusi | https://doi.org/10.1016/j.physleta.2008.10.049 |
| Chanwimalueang & Mandic 2017, Cosine Similarity Entropy (Entropy) | Ganti jarak antar vektor embedding di SampEn dengan **jarak sudut**; ada versi multiskala (MCSE) | Kemiripan geometris (sudut), bukan divergensi distribusi; hasil akhir Shannon entropy dari proporsi kecocokan | https://doi.org/10.3390/e19120652 |
| Omidvarnia dkk. 2018, Range Entropy (Entropy) | Ganti Chebyshev dengan jarak "range" ternormalisasi di ApEn/SampEn | Jarak berbasis rentang amplitudo, bukan distribusi | https://doi.org/10.3390/e20120962 |
| Li dkk. 2015, Distribution Entropy / DistEn (MBEC) | "Distribusi" dan "probability density" dari vektor ke vektor | Mengestimasi PDF dari **semua jarak (Chebyshev) antar vektor**, lalu Shannon entropy PDF itu. Tidak ada divergensi antar dua vektor; tidak ada fungsi keanggotaan fuzzy | https://doi.org/10.1007/s11517-014-1216-0 |
| Wang dkk. 2021, Multiscale Diversity Entropy (IEEE TII) | Multiskala, diagnosis fault, "distribution of cosine similarity between adjacent orbits" | Distribusi dari nilai kosinus antar orbit **bertetangga**, lalu entropi; tidak ada JSD | https://doi.org/10.1109/TII.2020.3022369 |
| Shi 2025, Improved Multi-scale Divergence Entropy (Front. Mech. Eng.) | Namanya "divergence entropy", multiskala, fault mesin | Isinya kemiripan kosinus antar trajektori bertetangga + histogram; **tidak memakai KL/JS divergence** (dicek di teks lengkap) | https://doi.org/10.3389/fmech.2025.1690084 |
| Wang dkk. 2025, Distance Similarity Entropy / DSEn (RESS) | Kernel Gauss untuk kemiripan vektor, estimasi densitas, fault bearing | Kernel Gauss diterapkan pada **jarak elemen-per-elemen** antar vektor bertetangga, lalu densitas kemiripan; tidak ada konversi vektor → histogram dan tidak ada JSD | https://doi.org/10.1016/j.ress.2024.110643 |
| Rostaghi & Azami 2016, Dispersion Entropy (IEEE SPL) | Memetakan vektor ke pola diskret lalu Shannon entropy | Vektor dipetakan ke **pola kelas** (simbol), bukan histogram; tidak ada divergensi antar vektor | https://doi.org/10.1109/LSP.2016.2542881 |
| Bandt & Pompe 2002, Permutation Entropy (PRL) | Vektor embedding → objek diskret (pola ordinal) | Shannon entropy distribusi pola; tanpa jarak antar vektor | https://doi.org/10.1103/PhysRevLett.88.174102 |
| Lamberti dkk. 2004 (Physica A); Rosso dkk. 2007 (PRL) — kompleksitas statistik Jensen–Shannon | **JSD dipakai** pada analisis kompleksitas deret waktu | JSD antara distribusi pola ordinal **seluruh sinyal** dan distribusi **uniform** (disequilibrium); bukan kemiripan antar vektor di dalam FuzzyEn | https://doi.org/10.1016/j.physa.2003.11.005 ; https://doi.org/10.1103/PhysRevLett.99.154102 |
| Li, Shang & Zhang 2019 (Nonlinear Dyn.); Zunino dkk. 2022, PJSD (PRE) | JSD antar distribusi pola permutasi | JSD antara distribusi pola dua sinyal / sinyal maju vs terbalik (iraversibilitas); level sinyal, bukan level pasangan vektor | https://doi.org/10.1007/s11071-019-04950-6 ; https://doi.org/10.1103/PhysRevE.105.045310 |
| Fan, Song & Li 2024, Multiscale PJSD (FNL) | **Multiskala + JSD** | Sama seperti PJSD (pola ordinal, level sinyal), tujuan iraversibilitas; tidak ada fuzzy entropy | https://doi.org/10.1142/S0219477525500130 |
| Yao dkk. 2026, Fractional Reverse PJSD + multiskala (ICCEA, IEEE) | **Paling dekat secara aplikasi: JSD + multiskala + diagnosis fault bearing** | JSD (fraksional) antara distribusi pola permutasi dan **distribusi acuan uniform**; tidak ada pencocokan template antar vektor, tidak ada fungsi keanggotaan fuzzy | https://doi.org/10.1109/ICCEA69962.2026.11743731 |
| Yao, Yao & Wang 2021 (Chaos Solitons Fractals) | **KL divergence** di ruang keadaan rekonstruksi | KL dipakai sebagai "kernel probabilistic difference" atas fluktuasi **jarak vektor terhadap toleransi** untuk analisis non-ekuilibrium; bukan kemiripan antar vektor di FuzzyEn (detail rumus hanya dari abstrak) | https://doi.org/10.1016/j.chaos.2021.111568 |

**Posisi kebaruan.** Kombinasi yang belum ditemukan dalam literatur yang ditelusuri: (i) setiap vektor embedding (terpusat) diubah menjadi distribusi probabilitas kecil (histogram lunak berkernel Gauss), (ii) **jarak Jensen–Shannon** antar dua distribusi itu dipakai sebagai d di dalam fungsi keanggotaan fuzzy, (iii) di dalam kerangka multiscale (komposit) fuzzy entropy, (iv) untuk deteksi fault sensor kelembaban tanah dengan ANN-LM. Unsur (iii)–(iv) sendiri tidak baru (Zhou 2021; Wardhani 2026); JSD pada deret waktu juga tidak baru (Rosso 2007, PJSD). Yang baru adalah **tempat** JSD dipasang: sebagai ukuran kemiripan pasangan template di dalam FuzzyEn.

**Catatan jujur yang harus disiapkan untuk penguji (bukan dari literatur, dari kode):**
- Histogram lunak dari vektor terpusat **tidak peka urutan** elemen: untuk m = 2, vektor terpusat selalu berbentuk [+d/2, −d/2] atau [−d/2, +d/2]; keduanya menghasilkan histogram identik, sehingga JSD = 0 untuk pola naik vs turun dengan besar sama. JSD-Fuzzy mengukur kemiripan **sebaran amplitudo dalam template**, bukan bentuk/urutan template. Ini beda konseptual dengan SampEn/FuzzyEn klasik dan perlu disebut sebagai sifat (atau keterbatasan) metode.
- Jarak yang dipakai adalah **akar JSD** (jarak Jensen–Shannon), yang merupakan metrik (Endres & Schindelin 2003); sitasi ini sebaiknya ditambahkan karena "JSD" mentah bukan metrik.
- Penelusuran literatur tidak pernah bisa membuktikan "belum pernah ada"; klaim harus diberi batas "sepanjang penelusuran penulis".

---

## (c) Rumusan klaim kebaruan yang aman

**Bahasa Indonesia:**
> Sepanjang penelusuran literatur penulis (Crossref, OpenAlex, Google/Semantic Scholar, per 20 November 2026), belum ditemukan studi yang menggunakan Jensen–Shannon divergence sebagai ukuran ketidakmiripan antar vektor embedding di dalam fungsi keanggotaan *multiscale fuzzy entropy*. Penelitian ini mengusulkan JSD-Fuzzy, yaitu modifikasi EDM-Fuzzy (Zhou dkk., 2021) yang mengubah setiap vektor embedding menjadi distribusi probabilitas (histogram lunak berkernel Gauss) lalu mengganti jarak Euclidean dengan jarak Jensen–Shannon, dan mengevaluasinya untuk deteksi fault sensor kelembaban tanah dengan ANN Levenberg–Marquardt. Jensen–Shannon divergence sendiri telah dipakai dalam analisis kompleksitas deret waktu berbasis pola ordinal (Rosso dkk., 2007; Zunino dkk., 2022); kebaruan penelitian ini terletak pada penempatannya sebagai ukuran kemiripan template di dalam fuzzy entropy, bukan pada JSD atau fuzzy entropy itu sendiri.

**English:**
> To the best of our knowledge (literature search in Crossref, OpenAlex and Google/Semantic Scholar as of 20 November 2026), Jensen–Shannon divergence has not previously been used as the dissimilarity between embedding vectors inside the membership function of multiscale fuzzy entropy. We propose JSD-Fuzzy, a modification of EDM-Fuzzy (Zhou et al., 2021) that maps each embedding vector to a probability distribution (a Gaussian-kernel soft histogram) and replaces the Euclidean distance with the Jensen–Shannon distance, and we evaluate it for soil-moisture sensor fault detection with a Levenberg–Marquardt-trained ANN. Jensen–Shannon divergence has been used before in ordinal-pattern complexity analysis of time series (Rosso et al., 2007; Zunino et al., 2022); the novelty lies in its use as the template-similarity measure within fuzzy entropy, not in JSD or fuzzy entropy themselves.

Hindari kalimat: "pertama di dunia", "belum pernah ada yang memakai JSD untuk entropi", "metode entropi baru berbasis distribusi" (DistEn sudah ada), "EDM-Fuzzy dari paper kami".

---

## (d) Daftar sitasi per pilihan metode

Format: **pilihan metode → sitasi → kenapa (1 baris)**. Kunci BibTeX dalam kurung.

**Entropi**
- Sample entropy sebagai dasar pencocokan template → Richman & Moorman (2000), AJP-Heart 278(6):H2039–H2049, doi:10.1152/ajpheart.2000.278.6.H2039 (`richman2000sampen`) → definisi SampEn, rasio φ_m/φ_{m+1} tanpa self-match.
- (latar) Approximate entropy → Pincus (1991), PNAS 88(6):2297–2301, doi:10.1073/pnas.88.6.2297 (`pincus1991apen`) → pendahulu SampEn.
- Fuzzy entropy, keanggotaan kontinu, vektor dipusatkan → Chen dkk. (2007), IEEE TNSRE 15(2):266–272, doi:10.1109/TNSRE.2007.897025 (`chen2007fuzzyen`) → sumber ide keanggotaan fuzzy dan pengurangan baseline vektor.
- m = 2, r = 0,2·SD → Richman & Moorman (2000); Chen dkk. (2007) → nilai baku yang dipakai di kedua paper asal.
- Multiscale (coarse-graining, skala 1..15) → Costa dkk. (2002), PRL 89:068102, doi:10.1103/PhysRevLett.89.068102 (`costa2002mse`); Costa dkk. (2005), PRE 71:021906, doi:10.1103/PhysRevE.71.021906 (`costa2005mse`) → definisi MSE.
- Coarse-graining komposit (rerata atas k offset, `coarse_grain_cmse`) → Wu dkk. (2013), Entropy 15(3):1069–1084, doi:10.3390/e15031069 (`wu2013cmse`) → mengurangi varians estimasi pada skala besar/deret pendek.
- Baseline EDM-Fuzzy (Euclidean + fuzzy) → Zhou dkk. (2021), IEEE TII 17(6):4046–4054, doi:10.1109/TII.2020.3009139 (`zhou2021edmfuzzy`) → sumber asli baseline.

**Divergensi**
- Jensen–Shannon divergence → Lin (1991), IEEE TIT 37(1):145–151, doi:10.1109/18.61115 (`lin1991jsd`) → definisi JSD, simetris, terbatas (≤ ln 2), tak butuh kontinuitas absolut.
- Akar JSD sebagai metrik (yang dipakai kode: `np.sqrt(pairwise_jsd(...))`) → Endres & Schindelin (2003), IEEE TIT 49(7):1858–1860, doi:10.1109/TIT.2003.813506 (`endres2003jsdmetric`) → membenarkan √JSD sebagai jarak di fungsi keanggotaan.
- Alasan tidak memakai KL → Kullback & Leibler (1951), Ann. Math. Stat. 22(1):79–86, doi:10.1214/aoms/1177729694 (`kullback1951kl`) → KL asimetris dan tak terbatas bila ada bin nol.

**Levenberg–Marquardt**
- Algoritma LM → Levenberg (1944), QAM 2(2):164–168, doi:10.1090/qam/10666 (`levenberg1944`); Marquardt (1963), SIAM J. 11(2):431–441, doi:10.1137/0111030 (`marquardt1963`) → asal langkah Gauss–Newton terdamping.
- LM untuk melatih jaringan saraf feedforward (Jacobian per bobot) → Hagan & Menhaj (1994), IEEE TNN 5(6):989–993, doi:10.1109/72.329697 (`hagan1994lm`) → acuan baku LM pada ANN.
- Pembaruan damping gaya Nielsen (rasio gain ρ, μ·max{1/3, 1−(2ρ−1)³}, ν = 2) → Madsen, Nielsen & Tingleff (2004), *Methods for Non-Linear Least Squares Problems*, 2nd ed., IMM-DTU (`madsen2004nlls`) → sumber primer terverifikasi (rekaman IMM-DTU) untuk LM dengan update μ berbasis rasio gain; keberadaan rekaman sudah dicek, tetapi isi PDF tidak dibuka, jadi cocokkan nomor persamaan sebelum dikutip. Laporan Nielsen 1999 lihat bagian (e).
- ANN-LM untuk fault sensor kelembaban tanah (konteks yang sama) → Wardhani dkk. (2026), JITK 11(4):1053–1060, doi:10.33480/jitk.v11i4.7481 (`wardhani2026annlm`) → studi pendahulu di domain dan skenario yang sama.

**Validasi dan uji statistik**
- StratifiedGroupKFold (implementasi) → Pedregosa dkk. (2011), JMLR 12:2825–2830 (`pedregosa2011sklearn`) → pustaka yang dipakai.
- Grup = blok waktu (menghindari kebocoran autokorelasi) → Roberts dkk. (2017), Ecography 40(8):913–929, doi:10.1111/ecog.02881 (`roberts2017cv`); Bergmeir & Benítez (2012), Inf. Sci. 191:192–213, doi:10.1016/j.ins.2011.12.028 (`bergmeir2012tscv`) → CV acak pada data berstruktur temporal terlalu optimistis; blocking dianjurkan.
- Uji t berpasangan pada lipatan CV bisa bias (fold tidak independen) → Dietterich (1998), Neural Comput. 10(7):1895–1923, doi:10.1162/089976698300017197 (`dietterich1998tests`) → peringatan Type I error tinggi.
- Koreksi varians (corrected resampled t-test) → Nadeau & Bengio (2003), Mach. Learn. 52(3):239–281, doi:10.1023/A:1024068626366 (`nadeau2003inference`) → koreksi tumpang-tindih data latih.
- Replikabilitas uji → Bouckaert & Frank (2004), PAKDD, LNCS 3056:3–12, doi:10.1007/978-3-540-24775-3_3 (`bouckaert2004replicability`) → hasil uji t berulang-CV tidak stabil tanpa koreksi.
- Uji Wilcoxon berpasangan atas 15 sel → Wilcoxon (1945), Biometrics Bull. 1(6):80–83, doi:10.2307/3001968 (`wilcoxon1945`); Demšar (2006), JMLR 7:1–30 (`demsar2006comparisons`) → Wilcoxon dianjurkan untuk membandingkan dua metode atas banyak dataset/sel.
- Koreksi perbandingan ganda → Holm (1979), Scand. J. Stat. 6(2):65–70, doi:10.2307/4615733 (`holm1979`) → kendali FWER bertahap.
- Ukuran efek (Cohen's d / d_z) → Cohen (1988), *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., Erlbaum, doi:10.4324/9780203771587 (`cohen1988power`) → ambang kecil/sedang/besar.
- Interval kepercayaan bootstrap → Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall, doi:10.1201/9780429246593 (`efron1993bootstrap`) → CI tanpa asumsi normal.

**Fault sensor (taksonomi dan deteksi)**
- Taksonomi fault (drift, offset/bias, spike, stuck-at, noise) → Ni dkk. (2009), ACM TOSN 5(3):25, doi:10.1145/1525856.1525863 (`ni2009faulttypes`) → definisi jenis fault yang disimulasikan.
- Prevalensi fault nyata (SHORT/spike, CONSTANT/stuck, NOISE) → Sharma, Golubchik & Govindan (2010), ACM TOSN 6(3):23, doi:10.1145/1754414.1754419 (`sharma2010sensorfaults`) → fault memang muncul pada dataset lapangan.
- Survei strategi deteksi fault WSN → Muhammed & Shaikh (2017), JNCA 78:267–287, doi:10.1016/j.jnca.2016.10.019 (`muhammed2017fdwsn`) → posisi metode berbasis fitur + klasifier.
- Klasifier ML untuk fault WSN → Zidi dkk. (2018), IEEE Sens. J. 18(1):340–347, doi:10.1109/JSEN.2017.2771226 (`zidi2018svm`) → contoh deteksi fault berbasis pembelajaran.
- QC data kelembaban tanah in situ (spike, break, saturasi) → Dorigo dkk. (2013), Vadose Zone J. 12(3), doi:10.2136/vzj2012.0097 (`dorigo2013ismnqc`) → fault/anomali nyata pada jaringan sensor kelembaban tanah global.
- Fault sensor kelembaban tanah pertanian → Ludeña-Choez dkk. (2019), Comput. Electron. Agric. 161:214–224, doi:10.1016/j.compag.2018.06.033 (`ludena2019nmf`); Jihani dkk. (2023), Results Eng. 20:101395, doi:10.1016/j.rineng.2023.101395 (`jihani2023kalman`); Deforce dkk. (2023), arXiv:2305.05495 (`deforce2023rogue`) → pembanding domain irigasi/pertanian.

**Laju drift 0,003–0,17 poin/hari:** **tidak ditemukan sumber** (lihat (e)).

---

## (e) Not verified

| Item | Status | Saran |
|---|---|---|
| Paper "ACS" mahasiswa (EDM-Fuzzy + ANN-LM, sensor kelembaban tanah) | Tidak ditemukan di Crossref/OpenAlex/web. Kandidat terdekat: Wardhani dkk. 2026 JITK (tanpa EDM-Fuzzy di abstrak). | Minta judul/DOI dari mahasiswa; jangan sitasi sebelum ada. |
| Nielsen, H. B. (1999). *Damping parameter in Marquardt's method.* Tech. Rep. IMM-REP-1999-05, IMM-DTU. | Hanya terlihat sebagai sitasi sekunder (catatan levmar UTK, komentar kode GSL `nielsen.c`); rekaman primer di pubdb IMM / DTU Orbit tidak ditemukan. | Sitasi Madsen, Nielsen & Tingleff (2004) (terverifikasi) untuk aturan update μ; bila tetap ingin Nielsen 1999, tulis sesuai sitasi sekunder dan tandai. |
| Sumber laju drift sensor kelembaban tanah nyata "0,003–0,17 poin/hari" (komentar di `build_11_annlm_real_notebook.py` baris 253–254) | Tidak ada sumber yang memberi laju drift per hari/tahun. Yang ada: drift diakui tapi tidak dikuantifikasi (mis. jaringan LoRaWAN kapasitif 90 hari "no significant drift"; review kalibrasi FDR 2026 hanya menyebut drift linear karena penuaan/korosi tanpa laju). | Jangan klaim angka itu sebagai "drift nyata" tanpa sumber; ubah jadi "asumsi skenario" atau turunkan dari data lapangan sendiri. |
| Cohen (1988) dan Efron & Tibshirani (1993) | Terverifikasi, tetapi DOI yang ada merujuk terbitan ulang (Routledge 2013; Chapman & Hall/CRC). | Aman; tahun edisi asli tetap 1988/1993. |
| Yao, Yao & Wang (2021) — detail rumus KL | Bibliografi terverifikasi; cara pakai KL hanya dari abstrak. | Cukup untuk tabel prior-art; jangan kutip rumus. |

---

## (f) Search log (2026-11-20)

Mesin: WebSearch (standard/extended), Crossref REST API (`/works/<DOI>` dan `query.bibliographic`), OpenAlex API, Europe PMC API, halaman penerbit (JMLR, IMM-DTU pubdb, arXiv, Frontiers, RePEc, Garuda). Semantic Scholar API: HTTP 429 (rate-limit), tidak dipakai. Halaman JITK (nusamandiri) HTTP 403; data diambil dari Garuda dan Crossref.

Query web:
1. `multiscale fuzzy entropy Levenberg-Marquardt neural network soil moisture sensor fault detection`
2. `Jensen-Shannon divergence fuzzy entropy similarity embedding vectors`
3. `"sample entropy" OR "fuzzy entropy" "Kullback-Leibler" divergence instead of Chebyshev distance embedding vectors similarity`
4. `EDM fuzzy entropy Euclidean distance membership fuzzy entropy sensor fault Indonesia` → menemukan Zhou dkk. 2021
5. `fuzzy entropy different distance metrics Euclidean Chebyshev cosine comparison FuzzyEn`
6. `"EDM-Fuzzy" soil moisture sensor fault detection neural network`
7. `distance similarity entropy rolling bearing 2024 Reliability Engineering System Safety ...` → DSEn
8. `entropy measure embedding vectors probability distribution divergence similarity "Jensen-Shannon" sample entropy variant time series complexity` → PJSD, CSE, kompleksitas JS
9. `"Jensen-Shannon" "fuzzy entropy" time series fault diagnosis multiscale` → nihil
10. `Wasserstein OR Kullback-Leibler distance between state vectors entropy measure time series complexity "phase space" distribution` → Yao 2021
11. `soil moisture sensor fault detection entropy features artificial neural network Levenberg Marquardt Indonesia smart irrigation drift spike bias`
12. `"multiscale fuzzy entropy" "soil moisture" sensor fault` → nihil
13. `deteksi fault sensor kelembaban tanah multiscale fuzzy entropy jaringan saraf tiruan Levenberg-Marquardt` → Wardhani dkk. 2026
14. `Wardhani Buono Wahjuni Syukur fuzzy entropy soil moisture sensor fault chili` → nihil
15. `"Applied Computer Science" journal soil moisture sensor fault detection entropy neural network Levenberg-Marquardt 2025` → nihil (selain Wardhani 2026)
16. `"multiscale" "Jensen-Shannon divergence" entropy time series fault diagnosis feature` → Shi 2025 (ternyata kosinus)
17. `sample entropy variant histogram of each embedding vector compared distributions ... Entropy MDPI` → nihil
18. `modified fuzzy entropy similarity measure replaced "Hellinger" OR "Bhattacharyya" OR "Kullback" OR "Jensen" between embedded vectors entropy time series` → nihil (hanya NS-SampEn, FuzzyMEn)
19. `Nielsen "Damping parameter in Marquardt's method" IMM-REP-1999-05 DTU` → hanya sitasi sekunder
20. `site:imm.dtu.dk Nielsen damping parameter Marquardt 1999` → nihil
21. `Madsen Nielsen Tingleff 2004 "Methods for non-linear least squares problems" pdf imm.dtu.dk` → pubdb 3215
22. `soil moisture sensor long-term drift rate per day capacitance sensor calibration drift study` → tidak ada laju
23. `in situ soil moisture sensor drift "per year" OR "per day" volumetric water content drift quantified long-term field TDR capacitance` → tidak ada laju

Query Crossref (`query.bibliographic`): `Wardhani Buono Wahjuni fuzzy entropy`; `Wardhani Buono Wahjuni soil moisture sensor`; `Luh Kesuma Wardhani fault`; `EDM-Fuzzy soil moisture sensor fault`; `multiscale fuzzy entropy Levenberg-Marquardt sensor fault smart farming`; `Euclidean distance multiscale fuzzy entropy soil moisture`; `Lamberti Intensive entropic non-triviality measure`; `Permutation Jensen-Shannon distance ...`; `Sensor nodes fault detection ... NMF`; `Kalman filter based sensor fault detection ... smart irrigation`; `Deforce Baesens Diels Serral rogue soil moisture`; `Holm sequentially rejective`; `Fan Song Li multiscale irreversibility PJSD` (→ juga menemukan Yao dkk. 2026 ICCEA).

Verifikasi DOI langsung (Crossref `/works/`): 45 DOI, semua OK kecuali satu DOI Lamberti yang salah ketik (diganti 10.1016/j.physa.2003.11.005 hasil pencarian judul) dan DOI arXiv (DataCite, dicek di arxiv.org). Holm 1979, Madsen 2004, Pedregosa 2011 dicek via OpenAlex + halaman JSTOR/IMM/JMLR. Demšar 2006 via jmlr.org.
