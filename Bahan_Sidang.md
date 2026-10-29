# Bahan Sidang — tanya-jawab dan peta logika

Dokumen ini dipakai saat latihan tanya-jawab. Setiap jawaban menunjuk
notebook/sel supaya bisa langsung dibuka kalau penguji minta bukti.
Rujukan lengkap ada di `sidang/MAPPING.md`.

**Aturan menjawab:** angka tanpa sumber jangan disebut. Kalau tidak yakin,
jawab "saya cek di notebook 11 §10" lalu buka. Jangan mengklaim "lebih baik
secara signifikan" kecuali menyebut p < 0,05.

---

## A. Gambaran besar

**Q1. Penelitian ini tentang apa?**
Mendeteksi kerusakan sensor kelembaban tanah dari pola sinyalnya, lalu
menentukan sensor mana yang rusak dan jenis kerusakannya. Caranya: potong
sinyal jadi jendela waktu, ukur entropi multiskala tiap sensor, beri ke ANN.
Yang diuji: mengganti ukuran kemiripan Euclidean (EDM-Fuzzy) dengan
Jensen–Shannon Divergence (JSD-Fuzzy). Rujukan: `11` §0 sel 1.

**Q2. Kenapa empat sensor, bukan satu?**
Karena satu sensor tidak bisa memberi tahu apakah nilai aneh itu gangguan
lokal atau pola normal. Empat kanal yang disinkronkan mempertahankan identitas
masing-masing, sehingga kerusakan bisa dilokalisasi per sensor. Rujukan: `02`
dan `11` §2.

**Q3. Kenapa entropi, bukan rata-rata/standar deviasi?**
Rata-rata dan standar deviasi hanya menangkap besar dan sebaran nilai.
Entropi menangkap **kerumitan pola** — sinyal yang macet atau berulang punya
entropi rendah walau rata-ratanya normal. Itu yang membuat fault seperti
*stuck* dan *drift* lebih terlihat. Rujukan: `11` §5.

**Q4. Apa beda EDM-Fuzzy dan JSD-Fuzzy?**
Keduanya mengukur "seberapa mirip dua potongan sinyal". EDM memakai jarak
garis lurus antar angka. JSD mengubah tiap potongan menjadi sebaran nilai
(histogram kecil), lalu membandingkan bentuk sebarannya. Rujukan: `11` §5
sel 15–17 dan gambar `11_similarity_euclidean_vs_jsd.png`.

**Q5. Apa kontribusi/kebaruannya?**
Tiga hal: (1) JSD-Fuzzy sebagai pengganti similarity di dalam entropi fuzzy,
(2) evaluasi berantai sampai identifikasi sensor dan jenis fault, (3) laporan
jujur bahwa keunggulannya konsisten tetapi belum kuat secara statistik
gabungan. Rujukan: `11` §13.

## B. Data

**Q6. Data dari mana dan berapa lama?**
Empat sensor kelembaban tanah, 14 September–21 Desember 2025, sampel tiap
30 detik: 281.721 baris. Rujukan: `Ilustrasi` §1 dan `11` §1.

**Q7. Kenapa 30 detik diubah jadi 5 menit?**
Rata-rata per 5 menit meredam derau dan memperkecil jumlah sampel tanpa
membuang informasi harian. Hasilnya 28.173 baris (97,8 hari). Rujukan: `11`
§2 sel 7.

**Q8. Missing value dan outlier dibuang?**
Tidak. Di data ini tidak ada NaN dan tidak ada nilai di luar 0–100 %.
Pembacaan macet dan pencilan **dilaporkan, tidak dibuang**, karena justru bisa
jadi gejala fault. Yang dibuang hanya cacat struktural (stempel waktu ganda),
dan jumlahnya nol. Rujukan: `11` §2b sel 8.

**Q9. Nilai 0 di sensor S4 diapakan?**
Tetap dipakai dan dilabeli normal. Totalnya 11.371 sampel 30 detik (4,0 %),
semuanya 16–21 Des 2025, berupa tiga blok macet: 45 jam, 37 jam, dan 12 jam
(plus 49 sampel pendek). Setelah agregasi 5 menit: 1.135 titik nol. Nilainya
masih di dalam rentang fisik 0–100 %, jadi tidak dibuang; ini justru gejala
pembacaan macet. Dampaknya 10 dari 280 window N=200 (3,6 %) mayoritas nol
(N=700: 2/79; N=1000: 1/55). Dicatat sebagai batasan. Rujukan: `11` §2b dan
`data_sensor.csv`.

**Q10. Kenapa fault-nya simulasi, bukan data rusak asli?**
Karena data asli tidak punya label "rusak" dan tidak boleh merusak sensor
sungguhan. Simulasi memberi label yang pasti dan bisa diulang dengan seed
tetap. Batasnya diakui: hasil berlaku untuk fault yang dimodelkan, bukan semua
kerusakan fisik. Rujukan: `11` §3.

**Q11. Kenapa drift-nya cuma 1 poin per hari?**
Agar setara drift sensor nyata (0,003–0,17 poin/hari), bukan rampa besar yang
mustahil. Drift dibuat per episode 3 hari lalu di-reset, jadi totalnya kecil
dan nilai tetap 0–100 %. Konsekuensinya drift memang sulit dideteksi — itu
dilaporkan apa adanya. Rujukan: `11` §3 sel 10.

## C. Broker dan sinkronisasi

**Q12. "Broker" di sini maksudnya apa?**
Komponen yang mengumpulkan empat aliran sensor menjadi satu tabel dan
menyelaraskan waktunya, tanpa melebur identitas sensor. Ini menyiapkan data
untuk pemodelan, bukan message broker produksi. Rujukan: `02` sel 1–13.

**Q13. Kenapa empat sensor tidak dirata-rata jadi satu?**
Karena rata-rata menyembunyikan sensor mana yang rusak. Hasilnya juga lebih
rendah: F1 0,821 (4 kanal) vs 0,748 (dirata-rata); akurasi 0,829 vs 0,726.
Rujukan: `02`.

**Q14. Bagaimana waktu disinkronkan?**
Stempel waktu diubah ke UTC, diurutkan, duplikat dibuang, lalu semua kanal
dipaksa ke grid 5 menit. Slot yang bolong ditambal interpolasi waktu.
Rujukan: `11` §2 sel 7.

**Q15. Kalau timestamp bolong banyak?**
Di dataset ini tidak ada slot bolong (0 isian). Mekanismenya tetap ada:
interpolasi waktu lalu median sebagai jaring pengaman. Rujukan: `11` §2b.

**Q16. Apa yang dibuktikan notebook 02?**
Bahwa menggabungkan empat kanal lebih baik daripada merata-ratakan, terutama
untuk spike dan hardware malfunction yang waktunya berbeda antar sensor.
Rujukan: `02` sel 14.

## D. Fault dan window

**Q17. Jenis fault-nya apa saja?**
Empat dasar: drift (geser pelan), spike (loncatan sesaat), bias (geser tetap),
hardware malfunction (macet/hilang). Kombinasi dua, tiga, dan empat dipakai
juga, total 16 kondisi fault + normal = 17 kelas. Rujukan: `11` §3.

**Q18. Bagaimana memilih sensor yang kena fault?**
Injeksi bersifat sensor-selective: satu sampai empat sensor dipilih acak,
sisanya dibiarkan bersih. Jadi model tidak bisa menebak "semua rusak".
Rujukan: `11` §3 dan `04` §6.

**Q19. Kenapa 17 kelas, bukan 5?**
17 kelas (normal + 16 kondisi fault) adalah eksperimen terpisah di notebook
`01`/`04`, bukan S5. Ladder S1–S5 di notebook `11` justru memecah tugas itu:
S1 2 kelas, S2 5, S3 7 (6 kombinasi 2 fault), S4 5 (4 kombinasi 3 fault),
S5 2 (kombinasi 4 fault). Identifikasi "sensor mana yang rusak" juga terpisah
(RQ6, `11` §12), bukan S4. Rujukan: `11` §7.

**Q20. Window N = 200 itu berapa lama?**
200 sampel × 5 menit = 1.000 menit ≈ 16,7 jam (0,69 hari). Setara N = 2000
pada laju 30 detik, konfigurasi terbaik di notebook 07. Rujukan: `11` §4.

**Q21. Kenapa stride 50 %?**
Agar jumlah window cukup banyak tanpa kehilangan peristiwa di batas. Karena
window bertetangga tumpang-tindih, cross-validation harus dikelompokkan
menurut blok waktu. Rujukan: `11` §4 dan §10.

**Q22. Kenapa window dengan fault ≤ 1 % dibuang?**
Window seperti itu isinya praktis normal; melabelinya "fault" akan mengajari
model pola yang salah. Pada N = 200, 1.992 dari 36.120 window dibuang.
Rujukan: `11_akuntansi_window.csv`.

## E. Entropi

**Q23. Entropi yang dipakai entropi apa?**
Entropi sampel/fuzzy pada sinyal yang di-embed (dimensi m = 2), dihitung pada
beberapa skala waktu. "Fuzzy" berarti kemiripan diberi bobot, bukan ya/tidak.
Rujukan: `11` §5.

**Q24. Kenapa multiskala τ = 1..15?**
Fault berbeda muncul pada kekasaran berbeda: spike terlihat di skala halus,
drift di skala kasar. 15 skala × 4 sensor = 60 fitur. Rujukan: `11` §6.

**Q25. r = 0,2 × simpangan baku itu apa?**
Toleransi "dianggap mirip" pada entropi fuzzy. Dikalikan simpangan baku supaya
adil untuk sinyal dengan amplitudo berbeda. Nilainya sama untuk EDM dan JSD,
jadi perbandingannya setara. Rujukan: `11` §5.

**Q26. EDM-Fuzzy menghitung apa?**
Jarak Euclidean antar vektor yang di-embed, lalu dipetakan menjadi bobot
kemiripan fuzzy. Semakin kecil jarak, semakin mirip. Rujukan: `11` §5.

**Q27. JSD-Fuzzy mengubah apa?**
Sebelum dibandingkan, tiap vektor diubah menjadi sebaran (8 bin, rentang
±3 simpangan baku, kernel halus). Kemiripan dihitung dari bentuk sebaran,
bukan selisih angka. Rujukan: `11` §5 sel 15–17.

**Q28. Kenapa 60 fitur, bukan 4?**
Karena tiap sensor menyumbang 15 nilai (satu per skala), lalu keempatnya
digabung. Identitas sensor dipertahankan supaya bisa dilacak. Rujukan: `11` §6.

## F. Model dan evaluasi

**Q29. ANN-nya seperti apa?**
Jaringan feed-forward: 60 masukan, satu atau dua lapisan tersembunyi dari
kisi (32,32), (64,16), (64,), aktivasi relu/tanh, keluaran softmax. Arsitektur
dipilih di dalam cross-validation, bukan setelah melihat data uji. Rujukan:
`11` §10.

**Q30. Kenapa Levenberg–Marquardt, bukan Adam?**
LM adalah metode pelatihan pada paper acuan dan cocok untuk jaringan kecil.
Edisi sidang hanya menampilkan ANN-LM; perbandingan dengan BR, GA, dan Adam
ada di arsip notebook `09` dan tidak ditampilkan agar protokolnya tidak
tercampur. Rujukan: `11` §10.

**Q31. Apa itu Jacobian analitik?**
Turunan error terhadap semua bobot dihitung langsung dengan aturan rantai,
bukan didekati beda hingga. Hasilnya lebih presisi dan cepat. Kebenarannya
diperiksa dengan membandingkan terhadap Jacobian numerik (§9a) — lolos.
Rujukan: `11` §9a dan `09` §9.

**Q32. Kenapa hanya ANN-LM yang ditampilkan?**
Karena fokus penelitian adalah perbandingan metode entropi (EDM vs JSD), bukan
perbandingan classifier. Pelatih lain dan XGBoost sudah diuji di arsip `09`
dan `06`, tetapi tidak dimunculkan supaya klaim tetap satu protokol: ANN-LM
asli dengan Jacobian analitik. Rujukan: `11` §10.

**Q33. Kenapa cross-validation harus dikelompokkan?**
Window bertetangga tumpang-tindih 50 %. Kalau diacak biasa, potongan sinyal
yang sama bisa muncul di data latih dan uji — skor jadi terlalu optimistis.
GroupKFold per blok waktu mencegah kebocoran itu. Rujukan: `11` §10.

**Q34. Apakah program penelitian masih satu alur?**
Ya. Notebook `11` tetap satu program end-to-end: muat data → broker → injeksi
fault → segmentasi → entropi → ANN-LM → evaluasi → sensor-ID. Notebook sidang
adalah lapisan penyaji: ia membaca CSV ekspor dan memeriksa angkanya dengan
`assert`, tanpa melatih ulang. Rujukan: `11` §2–§13.

## G. Hasil

**Q35. Hasil utamanya apa?**
JSD-Fuzzy menang di 11 dari 15 sel (5 skenario × 3 ukuran window), rata-rata
selisih F1 +0,028. Identifikasi sensor juga lebih baik: 0,675 vs 0,634
(menang 9/15). Rujukan: `11` §10 dan §12.

**Q36. Berapa skenario yang dimenangkan?**
11 dari 15 sel. Kalau dihitung per ukuran window: N = 200 menang 3/5,
N = 700 menang 5/5, N = 1000 menang 3/5. Rujukan: `11_rq4_selisih.csv`.

**Q37. Skenario mana yang paling mudah dikenali sensornya?**
S5 (F1 0,841), lalu S4 (0,735), S3 (0,635), S2 (0,622), S1 (0,543) — angka
JSD-Fuzzy per skenario. Rujukan: `11_sensor_ringkas.csv`.

**Q38. Apa bedanya notebook sidang dengan notebook `11` selain drift?**
`11` menghitung semuanya (RQ1–RQ6, CV entropi, separabilitas, statistik,
confusion matrix, sensor-ID); notebook sidang hanya menampilkan performa utama
dan sensor-ID. Sidang memakai bahasa sederhana + aset gambar, sedangkan `11`
memuat kode pipeline lengkap. Angka keduanya sama karena sama-sama dari `11`
versi v3 (setelah koreksi drift). Rujukan: `11` §10–§12.

**Q39. Apakah hasilnya signifikan?** *(tidak ditampilkan di notebook)*
Tidak secara gabungan: pooled p = 0,076 (Wilcoxon 0,107). Hanya N = 700 yang
signifikan (p = 0,027). Karena itu klaim yang benar: **konsisten lebih tinggi**,
bukan "terbukti signifikan". Rujukan: `11` §11 (lampiran privat).

**Q40. Kalau begitu, kenapa tetap dipakai?** *(tidak ditampilkan di notebook)*
Karena arahnya konsisten (11/15), keunggulan muncul di skenario sulit, dan
biayanya sebanding. Nilainya sebagai bukti pendukung, bukan klaim kemenangan
mutlak. Rujukan: `11` §13.

## H. Batasan dan tindak lanjut

**Q41. Kenapa N = 700 signifikan tetapi gabungan tidak?** *(tidak ditampilkan di notebook)*
Karena arah efek berbeda antar ukuran window (ada sel yang negatif di N = 1000).
Menggabungkan tiga kondisi menambah varians, sehingga rasio efek/derau tidak
naik. Uji per ukuran window lebih tepat dibaca satu per satu. Rujukan:
`11_rq5_ttest.csv` + `11` §11.

**Q42. Kenapa drift sulit dideteksi?**
Karena drift hanya 1 poin/hari; dalam window 16,7 jam pergeserannya di bawah
ayunan alami sinyal. Deteksi drift lebih bergantung pada pola entropi jangka
panjang daripada selisih nilai. Rujukan: `11` §3 dan `11` §10.

**Q43. Apakah bisa dipakai real-time?**
Belum. Satu keputusan butuh window 16,7 jam, jadi cocok untuk pemantauan
harian, bukan alarm detik. Untuk real-time perlu window lebih pendek dan
pelatihan ulang. Rujukan: `11` §4 dan `11_rq4_biaya_komputasi.csv`.

**Q44. Apa langkah berikutnya?**
(1) Uji pada data lapangan yang benar-benar rusak, (2) window adaptif untuk
deteksi lebih cepat, (3) uji jumlah sensor lebih banyak, (4) kalibrasi ambang
keputusan untuk operasi irigasi. Rujukan: `11` §13 dan `08` §12.

**Q45. Kalau penguji menanyakan angka yang tidak ada di notebook?**
Angka pendukung (statistik, pelatih lain, XGBoost, jenis fault, two-stage)
tetap tersedia di `Bahan_Sidang.md` dan arsip kernel penelitian, tetapi
sengaja tidak dimunculkan di notebook sidang supaya klaim tetap satu
protokol: ANN-LM. Sebutkan sumbernya, jangan mengarang angka baru.

---

## Kalimat siap pakai (hafalkan tiga ini)

1. "Saya mengganti ukuran kemiripan di dalam entropi fuzzy dari jarak
   Euclidean menjadi Jensen–Shannon Divergence, lalu menguji dampaknya sampai
   identifikasi sensor."
2. "JSD-Fuzzy menang di 11 dari 15 sel dengan selisih rata-rata +0,028;
   secara gabungan belum signifikan (p = 0,076), jadi saya menyebutnya
   konsisten, bukan terbukti."
3. "Yang saya laporkan apa adanya: JSD lebih tinggi, tetapi EDM lebih stabil
   (56 dari 63 pasangan). Keduanya saya tampilkan."

## Peta cepat kalau penguji menunjuk layar

| Penguji menunjuk | Buka |
|---|---|
| Grafik 4 kanal | `11` §2 atau `ILU_empat_kanal_bersih.png` |
| Contoh sinyal kena fault | `11` §3b, `ILU_grid_fault_vs_asli_S1.png` |
| Tabel 17 kelas | `04` §15, `05` §13 |
| Grafik EDM vs JSD | `11` §5, `11_similarity_euclidean_vs_jsd.png` |
| Tabel F1 utama | `11` §10, `11_rq4_performa_ann.csv` |
| Identifikasi sensor | `11` §12, `11_sensor_ringkas.csv` |
| Uji statistik | `11` §11 (privat) |
| ANN vs XGBoost | `06` §36–41 |
| Empat pelatih | `09` §10 |
