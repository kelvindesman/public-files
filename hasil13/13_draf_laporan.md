# Draf laporan — Uji tambahan intensitas gangguan (notebook 13)

Sumber angka: `hasil13/exports/13_tabel_tunggal.csv`, `13_tabel_kombinasi.csv`,
`13_nilai_representatif.csv` (kernel Kaggle `kelvindsmn/uji-intensitas-fault-jsd-edm`,
versi 1, mode FULL, 512 menit, 156 + 198 baris, semua berstatus `ok`).
Hasil utama (notebook 11 v3) tidak berubah.

## Usulan subbab hasil: Uji tambahan intensitas gangguan

### Rancangan uji

Uji tambahan menjalankan ulang sapuan intensitas gangguan (Bagian A notebook 12)
untuk dua metode fitur, JSD-Fuzzy dan EDM-Fuzzy. Model dibuat tetap supaya
perbedaan hasil hanya berasal dari intensitas gangguan dan metode fitur:
ANN-LM (32, 16) tanh, 40 iterasi, 5-fold StratifiedGroupKFold per blok waktu,
window N = 200, 3 seed. Tugasnya klasifikasi biner (fault vs normal) dengan
gangguan disisipkan pada sensor S3. Window berlabel fault kalau porsi sampel
terganggu lebih dari 1%. Untuk spike relatif, window berlabel fault kalau memuat
minimal satu spike. Window kombinasi berlabel fault hanya kalau semua
komponennya ada.

Sebuah gangguan disebut terdeteksi pada nilai terkecil yang memberi rata-rata
TPR ≥ 0,80 dan FPR ≤ 0,10.

### Hasil fault tunggal

| Gangguan | Nilai diuji | JSD-Fuzzy: TPR / FPR / AUC pada nilai terbesar | EDM-Fuzzy: TPR / FPR / AUC pada nilai terbesar | Status |
|---|---|---|---|---|
| Bias konstan | 0,08–5 poin | 0,396 / 0,367 / 0,501 | 0,479 / 0,475 / 0,511 | tidak terdeteksi (AUC ≈ 0,50 pada semua nilai) |
| Bias episode | 2; 5; 9 poin | 0,570 / 0,462 / 0,647 | 0,634 / 0,366 / 0,743 | tidak terdeteksi |
| Drift (total per episode) | 2; 3; 5; 9; 60 poin | 0,667 / 0,183 / 0,847 | 0,688 / 0,280 / 0,798 | tidak terdeteksi |
| Spike berbasis SD (periodik) | 0,08–1,5 × SD | 0,867 / 0,142 / 0,922 | 0,775 / 0,258 / 0,815 | tidak terdeteksi (FPR > 0,10) |
| Spike relatif | 15%; 30%; 50% | 0,812 / 0,179 / 0,865 | 0,708 / 0,254 / 0,785 | tidak terdeteksi (FPR > 0,10) |
| Hardware malfunction | (0,08; 0,05)–(0,7; 0,5) | terdeteksi mulai (0,3; 0,2): 0,908 / 0,083 / 0,963 | terdeteksi mulai (0,3; 0,2): 0,917 / 0,100 / 0,972 | terdeteksi |

Pada nilai penelitian saat ini, kedua metode masih jauh dari ambang deteksi:

| Gangguan (nilai penelitian) | JSD-Fuzzy TPR / FPR | EDM-Fuzzy TPR / FPR |
|---|---|---|
| Bias 0,08 poin | 0,400 / 0,367 | 0,479 / 0,467 |
| Drift total 3 poin (1 poin/hari) | 0,548 / 0,473 | 0,559 / 0,344 |
| Spike 0,08 × SD | 0,658 / 0,308 | 0,704 / 0,275 |
| Hardware (0,08; 0,05) | 0,829 / 0,175 | 0,779 / 0,221 |

Temuan utama:

1. **Bias konstan tidak dapat dideteksi oleh fitur entropi**, berapa pun besarnya.
   AUC tetap sekitar 0,50 dari 0,08 sampai 5 poin pada kedua metode. Centroid
   shifting mengurangi setiap vektor embedding dengan rata-ratanya sendiri,
   sehingga offset konstan hilang sebelum entropi dihitung. Ini batas
   representasi, bukan soal besar gangguan.
2. **Bias episode sedikit lebih terlihat**, tetapi tetap di bawah ambang. AUC naik
   menjadi 0,64–0,65 (JSD) dan 0,67–0,74 (EDM). Yang tertangkap hanya window
   yang memuat tepi episode, yaitu lompatan naik dan turun. Window di tengah
   episode tetap tampak normal karena alasan yang sama dengan bias konstan.
3. **Drift baru terpisah jelas pada total 60 poin per episode** (AUC 0,85 JSD,
   0,80 EDM). Pada total 2–9 poin, yang sesuai dengan rentang galat kalibrasi,
   AUC hanya 0,59–0,70. Drift lambat dalam rentang galat kalibrasi tidak dapat
   dibedakan dari variasi alami.
4. **Spike makin terdeteksi seiring amplitudo, dan JSD-Fuzzy lebih peka.** JSD naik
   dari AUC 0,70 (0,08 × SD) ke 0,92 (1,5 × SD) dan dari 0,81 ke 0,87 pada spike
   relatif 15% → 50%. EDM relatif datar (0,76–0,82 dan 0,77–0,79). Spike relatif
   50% pada JSD mencapai TPR 0,812, tetapi FPR 0,179 masih di atas 0,10.
5. **Hardware malfunction tetap paling mudah dideteksi.** Kedua metode lolos
   ambang mulai (0,3; 0,2), sama dengan hasil notebook 12 v1.

### Hasil kombinasi pada episode yang sama

Tingkat kombinasi: rendah = bias 2 poin, drift total 2 poin, spike 15%;
sedang = 5 poin, 5 poin, 30%; tinggi = 9 poin, 9 poin, 50%. Hardware di dalam
kombinasi tetap (0,08; 0,05).

| Metode | Tingkat | Kombinasi lolos ambang | F1 rata-rata | TPR rata-rata | FPR rata-rata |
|---|---|---|---|---|---|
| JSD-Fuzzy | rendah | 0/11 | 0,743 | 0,779 | 0,290 |
| JSD-Fuzzy | sedang | 0/11 | 0,766 | 0,805 | 0,271 |
| JSD-Fuzzy | tinggi | 0/11 | 0,779 | 0,818 | 0,258 |
| EDM-Fuzzy | rendah | 0/11 | 0,766 | 0,793 | 0,258 |
| EDM-Fuzzy | sedang | 0/11 | 0,767 | 0,790 | 0,254 |
| EDM-Fuzzy | tinggi | 0/11 | 0,771 | 0,801 | 0,257 |

- Tidak ada kombinasi yang lolos ambang FPR ≤ 0,10. Kombinasi terbaik adalah
  spike + hardware tingkat tinggi pada JSD-Fuzzy (TPR 0,927, FPR 0,134,
  AUC 0,942).
- Kombinasi yang memuat hardware mencapai AUC 0,81–0,94.
  Drift + bias adalah yang terlemah (AUC 0,67–0,73 JSD, 0,69–0,74 EDM), sesuai
  dengan temuan bahwa kedua komponen itu hampir tidak terlihat sendiri-sendiri.
- Menaikkan tingkat dari rendah ke tinggi hanya menaikkan F1 rata-rata
  0,036 (JSD) dan 0,005 (EDM). Deteksi kombinasi ditentukan oleh jenis
  komponennya, bukan besarnya.
- JSD-Fuzzy unggul (AUC lebih tinggi pada ketiga tingkat) pada spike + hardware
  dan drift + spike + hardware. EDM-Fuzzy unggul pada keempat kombinasi tanpa
  hardware (drift + bias, drift + spike, spike + bias, drift + spike + bias).
  Pada bias + hardware dan spike + bias + hardware hasilnya bergantian.

### Catatan FPR

FPR kelas normal tidak pernah turun di bawah sekitar 0,13, kecuali pada hardware
malfunction besar. Window normal juga memuat perubahan alami, misalnya
penyiraman dan pengeringan, yang entropinya mirip gangguan. Ambang FPR ≤ 0,10
karenanya ketat untuk fitur entropi pada data ini.

## Usulan perubahan Keterbatasan

Ganti kalimat bahwa uji intensitas hanya diusulkan dengan:

> Uji tambahan intensitas gangguan menunjukkan bahwa fitur entropi multiskala
> (JSD-Fuzzy maupun EDM-Fuzzy) tidak dapat mendeteksi bias konstan berapa pun
> besarnya, karena centroid shifting menghapus offset sebelum entropi dihitung.
> Bias episode hanya tertangkap pada tepi episode. Drift dengan total 2–9 poin
> per episode, yang setara dengan galat kalibrasi sensor kapasitif murah, juga
> belum dapat dibedakan dari kondisi normal (AUC ≤ 0,70). Hanya hardware
> malfunction yang lolos ambang deteksi (TPR ≥ 0,80, FPR ≤ 0,10), mulai dari
> proporsi (0,3; 0,2). Uji ini memakai klasifikasi biner pada satu sensor, bukan
> 16 kondisi seperti hasil utama.

## Usulan perubahan Saran

> 1. Tambahkan fitur yang mempertahankan level sinyal, misalnya rata-rata window,
>    selisih terhadap sensor tetangga, atau residu terhadap model prediksi, untuk
>    mendeteksi bias dan drift yang tidak tertangkap oleh entropi.
> 2. Pertahankan JSD-Fuzzy untuk gangguan yang mengubah bentuk distribusi,
>    yaitu spike dan hardware malfunction; pada uji ini JSD-Fuzzy lebih peka
>    terhadap amplitudo spike dibanding EDM-Fuzzy.
> 3. Kurangi FPR kelas normal dengan memasukkan konteks penyiraman atau cuaca,
>    karena perubahan alami pada data normal ikut menaikkan entropi.

## Asumsi yang perlu dicocokkan dengan dokumen usulan

- Tingkat kombinasi (rendah/sedang/tinggi) ditetapkan di notebook 13, karena tab
  "Perubahan program" pada dokumen usulan tidak tersedia.
- Spike relatif dan hardware pada kombinasi hanya disisipkan di dalam episode.
- Gangguan disisipkan pada satu sensor (S3), sama dengan notebook 12 Bagian A.
