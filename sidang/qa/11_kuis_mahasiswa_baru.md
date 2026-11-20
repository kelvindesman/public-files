# 11 — Kuis 10 soal untuk mahasiswa ilmu komputer tahun pertama

Tujuan: membuktikan bahwa penjelasan di notebook interaktif (salinan
internal) bisa dipahami orang yang belum pernah belajar entropi atau jaringan
saraf. Yang diuji adalah **penjelasannya**, bukan mahasiswanya.

Cara pakai:
1. Beri mahasiswa `sidang/04_Sidang_Interaktif.ipynb` (salinan internal) dan
   20 menit membaca serta memainkan panel. Kandidat tidak boleh menjelaskan.
2. Mahasiswa mengerjakan kuis tanpa membuka notebook, 15 menit.
3. Nilai dengan kunci. Target: ≥ 8/10. Soal yang salah menunjukkan bagian
   notebook yang penjelasannya perlu diperbaiki (kolom "Bagian").

## Soal

1. Sinyal 30 detik dirata-ratakan per 5 menit. Berapa sampel 30 detik yang
   menjadi satu sampel 5 menit?
   (a) 5 (b) 10 (c) 30 (d) 300

2. Mana yang paling tepat menggambarkan gangguan *bias*?
   (a) lonjakan sesaat (b) nilai bergeser tetap (c) nilai naik pelan-pelan
   (d) nilai macet atau hilang

3. Sinyal yang sangat teratur (berulang persis) akan punya entropi …
   (a) besar (b) kecil (c) negatif selalu (d) tidak bisa dihitung

4. Ada 4 sensor dan 15 skala. Berapa banyak fitur yang masuk ke jaringan
   saraf?
   (a) 4 (b) 15 (c) 19 (d) 60

5. Template [42, 43] dan [40, 41] dikurangi rata-ratanya masing-masing.
   Hasilnya?
   (a) berbeda 2 poin (b) sama-sama [−0,5; 0,5] (c) sama-sama [0; 1]
   (d) tidak bisa dibandingkan

6. Apa satu-satunya perbedaan EDM-Fuzzy dan JSD-Fuzzy?
   (a) jumlah sensor (b) cara mengukur jarak antar template
   (c) jaringan saraf yang dipakai (d) data yang dipakai

7. Template naik [−1, 1] dan turun [1, −1]. Menurut JSD-Fuzzy jaraknya …
   (a) 0 (b) 1 (c) 2,83 (d) tak hingga

8. Dalam Levenberg–Marquardt, apa yang terjadi kalau satu langkah membuat
   galat **naik**?
   (a) langkah diterima (b) langkah ditolak dan λ diperbesar
   (c) pelatihan berhenti (d) λ dijadikan nol

9. Model yang selalu menjawab "normal" pada 90 jendela normal dan 10 jendela
   rusak. Kenapa macro-F1 lebih jujur daripada akurasi?
   (jawab dalam satu kalimat)

10. p = 0,076 untuk selisih F1 gabungan. Kalimat mana yang benar?
    (a) JSD terbukti lebih baik (b) tidak ada perbedaan sama sekali
    (c) JSD konsisten lebih tinggi, tetapi belum signifikan pada α = 0,05
    (d) EDM lebih baik

## Kunci dan bagian notebook yang diuji

| No | Kunci | Bagian notebook / T di `08` |
|---|---|---|
| 1 | (b) 10 | 2 data / T8 |
| 2 | (b) | 3 fault / T13 |
| 3 | (b) | 4 entropi / T18 |
| 4 | (d) 60 | 4 entropi / T19 |
| 5 | (b) | 3 fault / T16 |
| 6 | (b) | 4 entropi / T20 |
| 7 | (a) 0 | 4 entropi / T23 |
| 8 | (b) | 5 LM / T27 |
| 9 | akurasi 90 % padahal kelas rusak tidak pernah ditemukan; macro-F1 merata-ratakan F1 tiap kelas (rusak = 0) sehingga ≈ 0,47 | 6/7 hasil / T33 |
| 10 | (c) | 8 statistik / T42 |

## Hasil

| Mahasiswa (inisial) | Tanggal | Skor /10 | Soal salah | Bagian yang diperbaiki |
|---|---|---|---|---|
| | | | | |
