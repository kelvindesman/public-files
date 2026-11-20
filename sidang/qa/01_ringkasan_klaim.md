# Ringkasan audit klaim → bukti (sidang)

## Status sesudah koreksi (2026-11-20, branch `sidang-prep`)

Semua temuan di bawah sudah diperbaiki di dokumen sumbernya oleh
`sidang/qa/perbaiki_klaim_2026_11_20.py` (setiap penggantian menyebut id ledger).

- Output Kaggle (CSV + log, tanpa gambar) disalin ke `sidang/bukti_kaggle/<slug>/`,
  jadi 83 klaim yang dulu "TIDAK ADA BUKTI LOKAL" kini dicek dari repo.
- Notebook demo dijalankan ulang tanpa layar (`sidang/qa/jalankan_notebook.py`);
  output dan waktu tiap sel ada di `sidang/bukti_demo/`. Klaim demo P-04–P-06
  kini dihitung dari file itu. Klaim "5 dari 8" ternyata salah hitung (6 dari 8).
- Tiga angka arsip label kembar (R-96, R-98a/b) dihapus dari README karena
  notebook arsipnya tidak punya output (status DIHAPUS).
- Log NB15 di `hasil15/` sekarang run v2 (wall 34 menit), README disesuaikan.

Hasil `./venv/bin/python sidang/qa/cek_klaim.py` (kode keluar 0):

```
Baris ledger: 476 | dihitung ulang otomatis: 466 | tanpa fungsi hitung (diperiksa manual): 10
Status hitung ulang: COCOK=466
Status ledger: COCOK=473, DIHAPUS=3
GERBANG KLAIM: LULUS
```

Bagian di bawah ini adalah temuan audit **sebelum** koreksi, disimpan sebagai jejak.

---

Tanggal: 2026-11-20. Ledger lengkap: `sidang/qa/01_klaim_bukti.csv`. Skrip hitung ulang: `sidang/qa/cek_klaim.py`.
Bukti diambil dari bahan mentah (CSV ekspor, log kernel JSON, builder, `data_sensor.csv`), bukan dari ringkasan.
Uji statistik dihitung ulang dengan `scipy.stats.ttest_rel(b, a)` dan `scipy.stats.wilcoxon(b, a)` dari F1 per sel, sama seperti builder (scipy 1.18.1).

## Hitungan

Ledger memuat **476 klaim** dari README.md, Bahan_Sidang.md, sidang/MAPPING.md, sidang/PANDUAN_DEMO.md, hasil04/hasil11 Ringkasan, dan draf laporan notebook 13 dan 14.

| Status | Jumlah |
|---|---|
| COCOK | 349 |
| PEMBULATAN | 10 |
| TIDAK COCOK | 26 |
| TIDAK ADA BUKTI LOKAL | 91 (83 di antaranya cocok dengan unduhan output Kaggle; 8 tidak punya bukti sama sekali) |

Sebanyak 463 klaim dihitung ulang otomatis oleh skrip. Sisanya 13 diperiksa manual, misalnya klaim berupa teks atau klaim tanpa file bukti.
Klaim notebook 01–10 berstatus TIDAK ADA BUKTI LOKAL karena angka terbarunya hanya ada di output Kaggle. Folder `hasil04`, `hasil07`–`hasil10` di repo masih berisi run lama sebelum koreksi drift (hasil04 masih v5).
Kolom `catatan` di ledger menyebut apakah Kaggle mengonfirmasi angkanya.

## Klaim TIDAK COCOK (nilai benar → file yang perlu diperbaiki)

1. **NB10 "JSD unggul 10 dari 15 sel"**. Angka yang benar adalah **11/15**: N200 S2, S4, S5; seluruh N700; N1000 S1, S4, S5. Perbaiki README.md:22, README.md:60, dan sidang/MAPPING.md:59. Output Kaggle v4 dan hasil10 lokal sama-sama memberi 11.
2. **Alasan p NB10 berbeda dari NB11: "protokol CV dan anggaran komputasi lebih besar"**. Klaim ini ada di README.md:64–66 dan MAPPING.md:59. Builder 10 dan 11 memakai protokol yang identik (LM_ITERS_SEARCH 6, LM_ITERS_REFIT 40, LM_INNER_CV 2, anggaran 40/150 s, grid dan seed sama). Yang berbeda hanya `KAGGLE_TIME_BUDGET_H`, 10,5 jam vs 11,5 jam. Lihat bagian kontradiksi di bawah.
3. **README.md:18 (NB06), "ANN-LM 0,394"**. Angka 0,394 berasal dari notebook 01 yang memakai sklearn `MLPClassifier`, bukan LM (MAPPING Tabel C). Labelnya harus diganti.
4. **README.md:18, "terbaik di 5 skenario"**. JSD+Time hanya terbaik di 3 skenario (S1, S2, S5); fitur Time menang di S3 dan S4. Selain itu, 0,443, 0,608, dan 0,254 adalah nilai **S5 saja**, padahal README tidak menyebut S5.
5. **README.md:26, "Hasil di `hasil15/`"**. File 15_* ada di `hasil16/exports/`, sedangkan `hasil15/` hanya berisi log.
6. **Bahan_Sidang.md:65, "semuanya 16–21 Des 2025"**. Satu sampel nol jatuh pada 2025-12-08 14:06.
7. **Bahan_Sidang.md:66, "plus 49 sampel pendek"**. Angka yang benar 50: 49 sampel di 16–21 Des ditambah 1 sampel tanggal 8 Des. Total 11.371 benar.
8. **Bahan_Sidang.md:176, "kisi (32,32), (64,16), (64,)"**. `HIDDEN_GRID` berisi **8** arsitektur: (16,), (32,), (64,), (16,8), (32,8), (32,16), (32,32), (64,16).
9. **hasil11/Ringkasan_untuk_laporan.md:26–27, rata-rata CV reduction −26% / −28% / −31%**. Nilai yang benar **−19,1% / −16,5% / −16,4%**, sesuai 11_rq2_cv_reduction.csv dan log NB11 baris 780–782.
10. **hasil11/Ringkasan_untuk_laporan.md:61–62 (RQ3)**. Klaimnya: JSD lebih terpisah di 6 sel, EDM di 4 sel (S2/S4/S5-N200, S1-N1000). Data menunjukkan JSD lebih terpisah di **11 sel**, EDM di **0 sel**, dan 4 sel campuran (N200 S1, S2, S5; N1000 S1).
11. **hasil14/14_draf_laporan.md:29–30, "jumlah sampel per kelas tetap sama dengan notebook 11 (1.700 / 1.621 / 1.526)"**. Angka itu milik NB14. NB11 punya 1.700 / **1.675 / 1.651**, jadi hanya N=200 yang sama.
12. **hasil14/14_draf_laporan.md:88–89, kolom NB11 dan NB14 tertukar**. Nilai yang benar:
    - NB11: fit 93,2 / 98,0 s dan inferensi 0,097 / 0,087 ms.
    - NB14: fit 75,5 / 75,1 s dan inferensi 0,089 / 0,088 ms.

    Kesimpulan di baris 94–95 ("Fit ANN-LM per fold lebih lama") karenanya terbalik, karena NB14 justru lebih cepat. Tabel ini mencakup 8 baris ledger: H14-43a–d, H14-44a–d, dengan dua di antaranya berstatus PEMBULATAN. Kesimpulannya tercatat sebagai H14-45e.
13. **hasil14/14_draf_laporan.md:92, "1,7–6 kali"**. Rentang yang benar **1,7–6,8 kali** (NB11 N=1000: 1.627,7 / 240,7 s).

## PEMBULATAN (arah atau batas pembulatan meleset sedikit)

| Klaim | Nilai | Catatan |
|---|---|---|
| README:24 dan draf 13:62, EDM spike "0,76–0,82" | maks 0,8145 | seharusnya 0,81 |
| README:24, bias episode "AUC ≤ 0,74" | 0,7426 | |
| draf 13:61, JSD spike relatif "0,87" | 0,8649 | |
| draf 13:86, drift+bias JSD "0,67" | 0,6649 | |
| hasil11 Ringkasan:34, "N=1000 +0,030" | 0,0295 | |
| draf 14:101, "+0,077" | 0,07648 | 3 desimal = 0,076 |
| README:34, "300–19.000×" | 339–19.200× | |

Satu klaim lain hanya lolos setelah dibulatkan tetapi buktinya hanya ada di Kaggle, sehingga berstatus TIDAK ADA BUKTI LOKAL: README:21 "vs Adam p ≤ 0,025", dengan p LM–Adam = 0,0252.

## Klaim tanpa bukti sama sekali

- README.md:96–99: arsip label kembar (F1 0,925, prevalensi 0,882, AUC 0,50). Notebook arsip tidak punya output.
- README.md:26: "Kaggle v1 COMPLETE" (NB15). Log lokal v1 memang selesai (1804 s), tetapi kernel kini sedang menjalankan versi baru dan unduhannya kosong.
- PANDUAN_DEMO.md:42: "2,5 menit termasuk menyalakan mesin". Log hanya mencatat waktu kernel sekitar 2,1 menit.
- PANDUAN_DEMO.md:43: "64–90 detik di laptop".
- PANDUAN_DEMO.md:60 dan 136: "80 %".
- PANDUAN_DEMO.md:106: "5 dari 8 tebakan".

Notebook demo tidak menyimpan output, dan tabelnya tidak ikut tercetak di stdout log.

## Kontradiksi antardokumen

- **Mengapa p NB10 (0,052) berbeda dari NB11 (0,076).** Menurut README.md:64–66, penyebabnya protokol CV dan anggaran komputasi NB11 lebih besar. Bukti tidak mendukung bagian protokol karena konstanta protokol di kedua builder identik. Hanya batas sesi `KAGGLE_TIME_BUDGET_H` yang berbeda.
  - Dari 15 sel, 8 punya F1 identik di kedua notebook dan 7 berbeda: N200 S2, S3, S4; N700 S2, S4; N1000 S2, S4 (`_sel_beda_10_11` di skrip). Perbedaannya terkumpul di sel S2/S4, yaitu sel yang pelatihannya berhenti karena `budget(MAX_TRAIN_S)`. Karena batas itu diukur dengan jam dinding, hasilnya tidak deterministik antar-run.
  - Jadi perbedaan p berasal dari variasi run, bukan dari protokol yang lebih besar. Jumlah sel menang pun sama, 11/15, sehingga MAPPING:59 ("jumlah sel menang juga beda") keliru.
- **hasil04/Ringkasan_untuk_laporan.md vs README dan MAPPING.** hasil04 masih memuat angka v5 (17 kelas 0,860/0,899; 4 fault semua-4-benar 0,880/0,980). README dan MAPPING memakai v6 (0,846/0,902; 0,800/0,920). Angka v5 cocok dengan ekspor lokal hasil04, tetapi sudah usang.
- **README NB06 vs MAPPING Tabel C.** README menyebut 0,394 sebagai "ANN-LM", sedangkan MAPPING menyatakan notebook 01 memakai sklearn MLP.
- **README NB06 vs MAPPING:149.** Angka README adalah S5 saja (0,608), sedangkan MAPPING memakai rata-rata 5 skenario (0,750). Keduanya benar untuk basisnya masing-masing, tetapi basis README tidak disebut.
- **Satuan 56/63.** README dan Bahan menyebut "pasangan", hasil11 Ringkasan menyebut "sel". Yang dihitung sebenarnya 63 baris kelas (N × skenario × kelas). Angkanya benar.
- **Draf NB14.** Tabel biaya (kolom tertukar) bertentangan dengan angka di CSV, dan kesimpulan di bawahnya mengikuti tabel yang salah.
- **PANDUAN_DEMO.md:152, "sekitar 26 hari garis lurus".** Angka ini benar untuk empat segmen panjang (26,3 hari). Kalau segmen lurus ≥ 6 jam ikut dihitung, totalnya 37,0 hari.

## Hasil `cek_klaim.py`

Perintah: `nice -n 19 ./venv/bin/python sidang/qa/cek_klaim.py --kaggle <scratchpad>/kaggle`. **Kode keluar: 1** karena ada 23 TIDAK COCOK dalam hitung ulang.
Tanpa `--kaggle`, 88 klaim berstatus LEWATI dan kode keluar tetap 1 (19 TIDAK COCOK).

Hitung ulang skrip membandingkan **angka** saja. Karena itu R06-02 (angka 0,394 benar, label "ANN-LM" salah) muncul COCOK di skrip tetapi TIDAK COCOK di ledger. Klaim teks atau manual (R15-H, R15-I, R-96, R-98a/b, P-03–P-06, H14-45e, R-D3, B-06b, P-09) tidak ikut dihitung skrip.

```
id      | klaim  | hitung ulang | selisih  | status      | catatan     
--------+--------+--------------+----------+-------------+-------------
R01-01  | 0,406  | 0.4061       | 0.00010  | COCOK       |             
R01-02  | 0,230  | 0.2303       | 0.00030  | COCOK       |             
R01-03  | 0,394  | 0.3938       | -0.00020 | COCOK       |             
R01-04  | 0,217  | 0.2174       | 0.00040  | COCOK       |             
R01-05  | 0,484  | 0.4844       | 0.00040  | COCOK       |             
R01-06  | 0,484  | 0.4838       | -0.00020 | COCOK       |             
R01-07  | 4      | 4            | 0        | COCOK       |             
R02-01  | 0,821  | 0.821        | 0        | COCOK       |             
R02-02  | 0,748  | 0.748        | 0        | COCOK       |             
R02-03  | 0,829  | 0.829        | 0        | COCOK       |             
R02-04  | 0,726  | 0.726        | 0        | COCOK       |             
R03-J1  | 0,819  | 0.819        | 0        | COCOK       |             
R03-E1  | 0,766  | 0.766        | 0        | COCOK       |             
R03-J2  | 0,668  | 0.668        | 0        | COCOK       |             
R03-E2  | 0,492  | 0.492        | 0        | COCOK       |             
R03-J3  | 0,805  | 0.805        | 0        | COCOK       |             
R03-E3  | 0,494  | 0.494        | 0        | COCOK       |             
R03-J4  | 0,814  | 0.814        | 0        | COCOK       |             
R03-E4  | 0,644  | 0.644        | 0        | COCOK       |             
R03-J5  | 1,000  | 1            | 0        | COCOK       |             
R03-E5  | 0,990  | 0.99         | 0        | COCOK       |             
R03-W   | 5      | 5            | 0        | COCOK       |             
R04-01  | 0,902  | 0.902        | 0        | COCOK       |             
R04-02  | 0,846  | 0.846        | 0        | COCOK       |             
R04-03  | 0,920  | 0.92         | 0        | COCOK       |             
R04-04  | 0,800  | 0.8          | 0        | COCOK       |             
R05-J1  | 0,448  | 0.448        | 0        | COCOK       |             
R05-E1  | 0,142  | 0.142        | 0        | COCOK       |             
R05-J2  | 0,868  | 0.868        | 0        | COCOK       |             
R05-E2  | 0,696  | 0.696        | 0        | COCOK       |             
R05-J3  | 0,601  | 0.601        | 0        | COCOK       |             
R05-E3  | 0,191  | 0.191        | 0        | COCOK       |             
R05-J4  | 0,917  | 0.917        | 0        | COCOK       |             
R05-E4  | 0,924  | 0.924        | 0        | COCOK       |             
R06-01  | 0,443  | 0.4428       | -0.00020 | COCOK       |             
R06-02  | 0,394  | 0.3938       | -0.00020 | COCOK       |             
R06-03  | 0,608  | 0.6076       | -0.00040 | COCOK       |             
R06-04  | 5      | 3            | -2       | TIDAK COCOK |             
R06-05  | 0,254  | 0.2542       | 0.00020  | COCOK       |             
R07-01  | 13     | 13           | 0        | COCOK       |             
R07-02  | 0,063  | 0.0632       | 0.00018  | COCOK       |             
R07-03  | 0,790  | 0.7905       | 0.00047  | COCOK       |             
R07-04  | 0,719  | 0.7193       | 0.00033  | COCOK       |             
R07-05  | 0,460  | 0.4604       | 0.00040  | COCOK       |             
R07-06  | 0,357  | 0.3572       | 0.00020  | COCOK       |             
R08-01  | 0,505  | 0.5049       | -0.00010 | COCOK       |             
R08-02  | 0,453  | 0.4526       | -0.00038 | COCOK       |             
R08-03  | 0,052  | 0.0523       | 0.00028  | COCOK       |             
R08-04  | 5      | 5            | 0        | COCOK       |             
R08-05  | 0,035  | 0.0354       | 0.00044  | COCOK       |             
R08-06  | 0,0625 | 0.0625       | 0        | COCOK       |             
R09-01  | 0,429  | 0.4287       | -0.00030 | COCOK       |             
R09-02  | 0,433  | 0.4333       | 0.00030  | COCOK       |             
R09-03  | 0,306  | 0.3057       | -0.00030 | COCOK       |             
R09-04  | 0,268  | 0.2675       | -0.00050 | COCOK       |             
R09-05  | 5      | 5            | 0        | COCOK       |             
R09-06  | 4      | 4            | 0        | COCOK       |             
R09-07  | 1      | 1            | 0        | COCOK       |             
R09-08  | 0      | 0            | 0        | COCOK       |             
R09-09  | 0,81   | 0.8135       | 0.0035   | COCOK       |             
R09-10  | 0,002  | 0.00030      | -0.0017  | COCOK       | mode=maks   
R09-11  | 0,025  | 0.0252       | 0.00020  | PEMBULATAN  | mode=maks   
R09-12  | 0      | 0            | 0        | COCOK       |             
R10-01  | 10     | 11           | 1        | TIDAK COCOK |             
R10-02  | 0,007  | 0.0067       | -0.00034 | COCOK       |             
R10-03  | 0,056  | 0.0555       | -0.00052 | PEMBULATAN  |             
R10-04  | 0,030  | 0.0303       | 0.00028  | COCOK       |             
R10-05  | 0,031  | 0.0308       | -0.00019 | COCOK       |             
R10-06  | 0,016  | 0.0161       | 0.00009  | COCOK       |             
R10-07  | 0,64   | 0.643        | 0.003    | COCOK       |             
R10-08  | 0,48   | 0.4834       | 0.0034   | COCOK       |             
R10-09  | 0,052  | 0.0519       | -0.00011 | COCOK       |             
R10-10  | 0,041  | 0.0413       | 0.00026  | COCOK       |             
R10-11  | 85784  | 85784.21     | 0.21     | COCOK       |             
R10-12  | 23,8   | 23.8289      | 0.0289   | COCOK       |             
R10-13  | 525    | 525.0333     | 0.0333   | COCOK       |             
R10-14  | 0      | 0            | 0        | COCOK       |             
R11-01  | 11     | 11           | 0        | COCOK       |             
R11-02  | 0,028  | 0.0281       | 0.00011  | COCOK       |             
R11-03  | 0,027  | 0.0271       | 0.00007  | COCOK       |             
R11-04  | 0,076  | 0.0759       | -0.00007 | COCOK       |             
R11-05  | 0,107  | 0.107        | -0.00001 | COCOK       |             
R11-06  | 0,675  | 0.6753       | 0.00030  | COCOK       |             
R11-07  | 0,634  | 0.6338       | -0.00017 | COCOK       |             
R11-08  | 9      | 9            | 0        | COCOK       |             
R11-09  | 0,307  | 0.3067       | -0.00027 | COCOK       |             
R11-10  | 0,295  | 0.2953       | 0.00033  | COCOK       |             
R11-11  | 56     | 56           | 0        | COCOK       |             
R11-12  | 89     | 88.8889      | -0.1111  | COCOK       |             
R11-13  | 610    | 610.2333     | 0.2333   | COCOK       |             
R11-14  | 99162  | 99162.45     | 0.45     | COCOK       |             
R11-15  | 27,5   | 27.5451      | 0.0451   | COCOK       |             
R11-16  | 0      | 0            | 0        | COCOK       |             
R11-17  | 0      | 0            | 0        | COCOK       |             
R11-18  | 0      | 0            | 0        | COCOK       |             
R13-01  | 0,3    | 0.3          | 0        | COCOK       |             
R13-02  | 0,3    | 0.3          | 0        | COCOK       |             
R13-03  | 0      | 0            | 0        | COCOK       |             
R13-04  | 0,50   | 0.516        | 0.016    | COCOK       | mode=sekitar
R13-05  | 0,74   | 0.7426       | 0.0026   | PEMBULATAN  | mode=maks   
R13-06  | 0,80   | 0.7978       | -0.0022  | COCOK       |             
R13-07  | 0,85   | 0.8467       | -0.0033  | COCOK       |             
R13-08  | 0,70   | 0.7038       | 0.0038   | COCOK       |             
R13-09  | 0,92   | 0.9222       | 0.0022   | COCOK       |             
R13-10  | 0,76   | 0.7593       | -0.00073 | COCOK       |             
R13-11  | 0,82   | 0.8145       | -0.0055  | PEMBULATAN  |             
R13-12  | 0      | 0            | 0        | COCOK       |             
R13-13  | 354    | 354          | 0        | COCOK       |             
R13-14  | 512    | 512.4949     | 0.4949   | COCOK       |             
R14-01  | 12     | 12           | 0        | COCOK       |             
R14-02  | 0,421  | 0.421        | -0.00005 | COCOK       |             
R14-03  | 0,497  | 0.4974       | 0.00043  | COCOK       |             
R14-04  | 0,0006 | 0.00062      | 0.00002  | COCOK       |             
R14-05  | 0,076  | 0.0759       | -0.00007 | COCOK       |             
R14-06  | 0,693  | 0.6925       | -0.00048 | COCOK       |             
R14-07  | 0,791  | 0.7905       | -0.00048 | COCOK       |             
R14-08  | 0,356  | 0.3563       | 0.00033  | COCOK       |             
R14-09  | 0,483  | 0.4831       | 0.00013  | COCOK       |             
R14-10  | 0,21   | 0.206        | -0.004   | COCOK       |             
R14-11  | 0,53   | 0.5324       | 0.0024   | COCOK       |             
R14-12  | 478    | 478.0167     | 0.0167   | COCOK       |             
R14-13  | 0      | 0            | 0        | COCOK       |             
R15-A1  | 0,90   | 0.897        | -0.003   | COCOK       |             
R15-B1  | 0,98   | 0.976        | -0.004   | COCOK       |             
R15-A2  | 0,85   | 0.853        | 0.003    | COCOK       |             
R15-B2  | 0,75   | 0.754        | 0.004    | COCOK       |             
R15-A3  | 0,83   | 0.83         | 0        | COCOK       |             
R15-B3  | 0,75   | 0.753        | 0.003    | COCOK       |             
R15-A4  | 0,88   | 0.881        | 0.001    | COCOK       |             
R15-B4  | 0,80   | 0.795        | -0.005   | COCOK       |             
R15-A5  | 0,96   | 0.958        | -0.002   | COCOK       |             
R15-B5  | 0,90   | 0.897        | -0.003   | COCOK       |             
R15-C1  | 0,49   | 0.4944       | 0.0044   | COCOK       |             
R15-C2  | 0,71   | 0.7094       | -0.00060 | COCOK       |             
R15-D   | 6      | 6.1625       | 0.1625   | COCOK       |             
R15-E   | 30     | 30.0667      | 0.0667   | COCOK       |             
R15-F   | 0      | 0            | 0        | COCOK       |             
R15-G   | 16     | 16           | 0        | COCOK       |             
R-ILU1  | 0      | 0            | 0        | COCOK       |             
R-ILU2  | 0      | 0            | 0        | COCOK       |             
R-ILU3  | 0      | 0            | 0        | COCOK       |             
R-ILU4a | 556    | 556          | 0        | COCOK       |             
R-ILU4b | 654    | 654          | 0        | COCOK       |             
R-ILU4c | 659    | 659          | 0        | COCOK       |             
R-ILU4d | 657    | 657          | 0        | COCOK       |             
R-ILU5a | 13     | 13           | 0        | COCOK       |             
R-ILU5b | 0      | 0            | 0        | COCOK       |             
R-ILU5c | 0      | 0            | 0        | COCOK       |             
R-ILU5d | 0      | 0            | 0        | COCOK       |             
R-D1    | 57,6   | 57.6         | 0        | COCOK       |             
R-D2    | 5635   | 5634.6       | -0.4     | COCOK       |             
R-D5    | 1,0    | 1            | 0        | COCOK       |             
R-D6    | 3      | 3            | 0        | COCOK       |             
R-60a   | 10     | 11           | 1        | TIDAK COCOK |             
R-60b   | 0,052  | 0.0519       | -0.00011 | COCOK       |             
R-61a   | 11     | 11           | 0        | COCOK       |             
R-61b   | 0,076  | 0.0759       | -0.00007 | COCOK       |             
R-64    | 1      | 0            | -1       | TIDAK COCOK | mode=min    
R-81a   | 281721 | 281721       | 0        | COCOK       |             
R-81b   | 28173  | 28173        | 0        | COCOK       |             
R-84    | 0,69   | 0.69         | 0        | COCOK       |             
R-84b   | 2,43   | 2.43         | 0        | COCOK       |             
R-84c   | 3,47   | 3.47         | 0        | COCOK       |             
R-85    | 0      | 0            | 0        | COCOK       |             
R-120   | 281721 | 281721       | 0        | COCOK       |             
R-121   | 0      | 0            | 0        | COCOK       |             
R-153a  | 6      | 6            | 0        | COCOK       |             
R-153b  | 2      | 2            | 0        | COCOK       |             
R-153c  | 40     | 40           | 0        | COCOK       |             
R-159a  | 10,5   | 10.5         | 0        | COCOK       |             
R-159b  | 11,5   | 11.5         | 0        | COCOK       |             
B-06    | 281721 | 281721       | 0        | COCOK       |             
B-07a   | 28173  | 28173        | 0        | COCOK       |             
B-07b   | 97,8   | 97.8         | 0        | COCOK       |             
B-08a   | 0      | 0            | 0        | COCOK       |             
B-08b   | 0      | 0            | 0        | COCOK       |             
B-08c   | 0      | 0            | 0        | COCOK       |             
B-09a   | 11371  | 11371        | 0        | COCOK       |             
B-09b   | 4,0    | 4.0363       | 0.0363   | COCOK       |             
B-09c   | 16     | 8            | -8       | TIDAK COCOK |             
B-09d   | 45     | 44.9917      | -0.0083  | COCOK       |             
B-09e   | 37     | 37.2333      | 0.2333   | COCOK       |             
B-09f   | 12     | 12.1167      | 0.1167   | COCOK       |             
B-09g   | 49     | 50           | 1        | TIDAK COCOK |             
B-09h   | 1135   | 1135         | 0        | COCOK       |             
B-09i   | 10     | 10           | 0        | COCOK       |             
B-09m   | 280    | 280          | 0        | COCOK       |             
B-09j   | 2      | 2            | 0        | COCOK       |             
B-09n   | 79     | 79           | 0        | COCOK       |             
B-09k   | 1      | 1            | 0        | COCOK       |             
B-09o   | 55     | 55           | 0        | COCOK       |             
B-15    | 0      | 0            | 0        | COCOK       |             
B-19a   | 2      | 2            | 0        | COCOK       |             
B-19b   | 5      | 5            | 0        | COCOK       |             
B-19c   | 7      | 7            | 0        | COCOK       |             
B-19d   | 5      | 5            | 0        | COCOK       |             
B-19e   | 2      | 2            | 0        | COCOK       |             
B-20    | 0,69   | 0.69         | 0        | COCOK       |             
B-20b   | 16,7   | 16.6667      | -0.0333  | COCOK       |             
B-22a   | 1992   | 1992         | 0        | COCOK       |             
B-22b   | 36120  | 36120        | 0        | COCOK       |             
B-27    | 8      | 8            | 0        | COCOK       |             
B-28    | 60     | 60           | 0        | COCOK       |             
B-29    | 3      | 8            | 5        | TIDAK COCOK |             
B-35a   | 11     | 11           | 0        | COCOK       |             
B-35b   | 0,028  | 0.0281       | 0.00011  | COCOK       |             
B-35c   | 0,675  | 0.6753       | 0.00030  | COCOK       |             
B-35d   | 0,634  | 0.6338       | -0.00017 | COCOK       |             
B-35e   | 9      | 9            | 0        | COCOK       |             
B-36    | 11     | 11           | 0        | COCOK       |             
B-36a   | 3      | 3            | 0        | COCOK       |             
B-36b   | 5      | 5            | 0        | COCOK       |             
B-36c   | 3      | 3            | 0        | COCOK       |             
B-37a   | 0,841  | 0.8407       | -0.00033 | COCOK       |             
B-37b   | 0,735  | 0.7353       | 0.00033  | COCOK       |             
B-37c   | 0,635  | 0.635        | -0.00000 | COCOK       |             
B-37d   | 0,622  | 0.6217       | -0.00033 | COCOK       |             
B-37e   | 0,543  | 0.5433       | 0.00033  | COCOK       |             
B-39a   | 0,076  | 0.0759       | -0.00007 | COCOK       |             
B-39b   | 0,107  | 0.107        | -0.00001 | COCOK       |             
B-39c   | 0,027  | 0.0271       | 0.00007  | COCOK       |             
B-40    | 11     | 11           | 0        | COCOK       |             
B-41    | 1      | 2            | 1        | COCOK       | mode=min    
B-46a   | 0,897  | 0.8967       | -0.00033 | COCOK       |             
B-46b   | 0,908  | 0.9083       | 0.00033  | COCOK       |             
B-46c   | 0,083  | 0.0833       | 0.00033  | COCOK       |             
B-46d   | 0,667  | 0.6667       | -0.00033 | COCOK       |             
B-46e   | 0,183  | 0.1828       | -0.00020 | COCOK       |             
B-46f   | 0,854  | 0.8542       | 0.00017  | COCOK       |             
B-46g   | 0,154  | 0.1542       | 0.00017  | COCOK       |             
B-47    | 0,50   | 0.5019       | 0.0019   | COCOK       | mode=sekitar
B-48a   | 0,805  | 0.8048       | -0.00023 | COCOK       |             
B-48b   | 0,766  | 0.7659       | -0.00015 | COCOK       |             
B-48c   | 0,844  | 0.8437       | -0.00030 | COCOK       |             
B-48d   | 0,735  | 0.7348       | -0.00015 | COCOK       |             
B-K1    | 11     | 11           | 0        | COCOK       |             
B-K2    | 0,028  | 0.0281       | 0.00011  | COCOK       |             
B-K3    | 0,076  | 0.0759       | -0.00007 | COCOK       |             
B-K4    | 56     | 56           | 0        | COCOK       |             
H11-01  | 610    | 610.2333     | 0.2333   | COCOK       |             
H11-01b | 10,2   | 10.1706      | -0.0294  | COCOK       |             
H11-02  | 0      | 0            | 0        | COCOK       |             
H11-02b | 0      | 0            | 0        | COCOK       |             
H11-03  | 54     | 54           | 0        | COCOK       |             
H11-04d | 49     | 49           | 0        | COCOK       |             
H11-04a | 21     | 21           | 0        | COCOK       |             
H11-04b | 27     | 27           | 0        | COCOK       |             
H11-04c | 1      | 1            | 0        | COCOK       |             
H11-29a | 144    | 144          | 0        | COCOK       |             
H11-29d | 315    | 315          | 0        | COCOK       |             
R11-19a | 45,7   | 45.7143      | 0.0143   | COCOK       |             
H11-08  | -26    | -19.071      | 6.929    | TIDAK COCOK |             
H11-29b | 59     | 59           | 0        | COCOK       |             
R11-19b | 18,7   | 18.7302      | 0.0302   | COCOK       |             
H11-09  | -28    | -16.4886     | 11.5114  | TIDAK COCOK |             
H11-29c | 52     | 52           | 0        | COCOK       |             
R11-19c | 16,5   | 16.5079      | 0.0079   | COCOK       |             
H11-10  | -31    | -16.39       | 14.61    | TIDAK COCOK |             
H11-05  | 56     | 56           | 0        | COCOK       |             
H11-06  | 89     | 88.8889      | -0.1111  | COCOK       |             
H11-07  | 11     | 11           | 0        | COCOK       |             
H11-07a | 3      | 3            | 0        | COCOK       |             
H11-07b | 5      | 5            | 0        | COCOK       |             
H11-07c | 3      | 3            | 0        | COCOK       |             
H11-12  | 0,002  | 0.0023       | 0.00026  | COCOK       |             
H11-13  | 0,053  | 0.0526       | -0.00038 | COCOK       |             
H11-13b | 0,030  | 0.0295       | -0.00054 | PEMBULATAN  |             
H11-11  | 0,028  | 0.0281       | 0.00011  | COCOK       |             
H11-23a | 0,458  | 0.458        | 0.00004  | COCOK       |             
H11-23b | 0,486  | 0.4862       | 0.00015  | COCOK       |             
H11-14  | 0,027  | 0.0271       | 0.00007  | COCOK       |             
H11-14w | 0,0625 | 0.0625       | 0        | COCOK       |             
H11-15  | 0,85   | 0.8539       | 0.0039   | COCOK       |             
H11-16  | 0,50   | 0.4986       | -0.0014  | COCOK       |             
H11-17  | 0,076  | 0.0759       | -0.00007 | COCOK       |             
H11-18  | 0,107  | 0.107        | -0.00001 | COCOK       |             
H11-20  | 0,634  | 0.6338       | -0.00017 | COCOK       |             
H11-19  | 0,675  | 0.6753       | 0.00030  | COCOK       |             
H11-22  | 0,295  | 0.2953       | 0.00033  | COCOK       |             
H11-21  | 0,307  | 0.3067       | -0.00027 | COCOK       |             
H11-25  | 30     | 30           | 0        | COCOK       |             
H11-24  | 9      | 9            | 0        | COCOK       |             
H11-28  | 6      | 11           | 5        | TIDAK COCOK |             
H11-27  | 4      | 0            | -4       | TIDAK COCOK |             
H11-26  | 0,03   | 0.0281       | -0.0019  | COCOK       |             
H11-30  | 0,675  | 0.6753       | 0.00030  | COCOK       |             
H11-31  | 0,027  | 0.0271       | 0.00007  | COCOK       |             
H11-32  | 0,076  | 0.0759       | -0.00007 | COCOK       |             
H11-33  | 89     | 88.8889      | -0.1111  | COCOK       |             
H13-05  | 512    | 512.4949     | 0.4949   | COCOK       |             
H13-04  | 354    | 354          | 0        | COCOK       |             
H13-01  | 36     | 36           | 0        | COCOK       |             
H13-02  | 16     | 16           | 0        | COCOK       |             
H13-03  | 0,50   | 0.516        | 0.016    | COCOK       | mode=sekitar
H13-19a | 0,64   | 0.6427       | 0.0027   | COCOK       |             
H13-19b | 0,65   | 0.6514       | 0.0014   | COCOK       |             
H13-19c | 0,67   | 0.67         | -0.00004 | COCOK       |             
H13-19d | 0,74   | 0.7426       | 0.0026   | COCOK       |             
H13-21  | 0,85   | 0.8467       | -0.0033  | COCOK       |             
H13-22  | 0,80   | 0.7978       | -0.0022  | COCOK       |             
H13-23a | 0,59   | 0.5921       | 0.0021   | COCOK       |             
H13-23b | 0,70   | 0.6986       | -0.0014  | COCOK       |             
H13-24a | 0,70   | 0.7038       | 0.0038   | COCOK       |             
H13-24b | 0,92   | 0.9222       | 0.0022   | COCOK       |             
H13-25a | 0,81   | 0.8082       | -0.0018  | COCOK       |             
H13-25b | 0,87   | 0.8649       | -0.0051  | PEMBULATAN  |             
H13-24c | 0,76   | 0.7593       | -0.00073 | COCOK       |             
H13-24d | 0,82   | 0.8145       | -0.0055  | PEMBULATAN  |             
H13-26a | 0,77   | 0.7741       | 0.0041   | COCOK       |             
H13-26b | 0,79   | 0.7851       | -0.0049  | COCOK       |             
H13-27a | 0,812  | 0.8125       | 0.00050  | COCOK       |             
H13-27b | 0,179  | 0.1792       | 0.00017  | COCOK       |             
H13-06  | 18     | 18           | 0        | COCOK       |             
H13-28  | 0      | 0            | 0        | COCOK       |             
H13-07  | 0,927  | 0.9272       | 0.00025  | COCOK       |             
H13-08  | 0,134  | 0.1345       | 0.00048  | COCOK       |             
H13-09  | 0,942  | 0.9418       | -0.00016 | COCOK       |             
H13-10  | 0,81   | 0.8131       | 0.0031   | COCOK       |             
H13-11  | 0,94   | 0.9418       | 0.0018   | COCOK       |             
H13-12  | 0,67   | 0.6649       | -0.0051  | PEMBULATAN  |             
H13-13  | 0,73   | 0.7256       | -0.0044  | COCOK       |             
H13-14  | 0,69   | 0.6937       | 0.0037   | COCOK       |             
H13-15  | 0,74   | 0.743        | 0.003    | COCOK       |             
H13-16  | 0,036  | 0.0362       | 0.00017  | COCOK       |             
H13-17  | 0,005  | 0.0053       | 0.00034  | COCOK       |             
H13-18  | 0,13   | 0.1345       | 0.0045   | COCOK       | mode=sekitar
H14-01  | 478    | 478.0167     | 0.0167   | COCOK       |             
H14-02  | 0      | 0            | 0        | COCOK       |             
H14-06  | 26548  | 26548        | 0        | COCOK       |             
H14-07  | 7108   | 7108         | 0        | COCOK       |             
H14-08  | 4635   | 4635         | 0        | COCOK       |             
H14-09  | 1992   | 1992         | 0        | COCOK       |             
H14-10  | 536    | 536          | 0        | COCOK       |             
H14-11  | 344    | 344          | 0        | COCOK       |             
H14-12  | 1621   | 1675         | 54       | TIDAK COCOK |             
H14-13  | 1526   | 1651         | 125      | TIDAK COCOK |             
H14-03  | 60     | 60           | 0        | COCOK       |             
H14-04  | 12     | 12           | 0        | COCOK       |             
H14-05  | 3      | 3            | 0        | COCOK       |             
H14-49a | 0,460  | 0.4601       | 0.00012  | COCOK       |             
H14-49b | 0,462  | 0.4624       | 0.00038  | COCOK       |             
H14-50  | 0,854  | 0.8539       | -0.00014 | COCOK       |             
H14-17  | 0,410  | 0.41         | -0.00004 | COCOK       |             
H14-18  | 0,510  | 0.5098       | -0.00018 | COCOK       |             
H14-19  | 0,010  | 0.01         | 0.00004  | COCOK       |             
H14-49c | 0,450  | 0.4497       | -0.00026 | COCOK       |             
H14-49d | 0,502  | 0.5024       | 0.00036  | COCOK       |             
H14-26  | 0,027  | 0.0271       | 0.00007  | COCOK       |             
H14-20  | 0,451  | 0.4507       | -0.00030 | COCOK       |             
H14-21  | 0,507  | 0.5066       | -0.00036 | COCOK       |             
H14-22  | 0,199  | 0.1993       | 0.00026  | COCOK       |             
H14-49e | 0,464  | 0.4643       | 0.00026  | COCOK       |             
H14-49f | 0,494  | 0.4937       | -0.00028 | COCOK       |             
H14-51  | 0,499  | 0.4986       | -0.00043 | COCOK       |             
H14-23  | 0,402  | 0.4022       | 0.00020  | COCOK       |             
H14-24  | 0,476  | 0.4758       | -0.00016 | COCOK       |             
H14-25  | 0,095  | 0.0951       | 0.00010  | COCOK       |             
H14-29  | 0,076  | 0.0759       | -0.00007 | COCOK       |             
H14-27  | 0,421  | 0.421        | -0.00005 | COCOK       |             
H14-28  | 0,497  | 0.4974       | 0.00043  | COCOK       |             
H14-14  | 0,0006 | 0.00062      | 0.00002  | COCOK       |             
H14-15  | 0,0015 | 0.0015       | 0.00003  | COCOK       |             
H14-16  | 1,13   | 1.1336       | 0.0036   | COCOK       |             
H14-31  | 0,634  | 0.6338       | -0.00017 | COCOK       |             
H14-33  | 0,675  | 0.6753       | 0.00030  | COCOK       |             
H14-35  | 0,693  | 0.6925       | -0.00048 | COCOK       |             
H14-36  | 0,791  | 0.7905       | -0.00048 | COCOK       |             
H14-32  | 0,295  | 0.2953       | 0.00033  | COCOK       |             
H14-34  | 0,307  | 0.3067       | -0.00027 | COCOK       |             
H14-37  | 0,356  | 0.3563       | 0.00033  | COCOK       |             
H14-38  | 0,483  | 0.4831       | 0.00013  | COCOK       |             
H14-39a | 48,3   | 48.254       | -0.046   | COCOK       |             
H14-39b | 31,1   | 31.1111      | 0.0111   | COCOK       |             
H14-39c | 17,8   | 17.7778      | -0.0222  | COCOK       |             
H14-39d | 45,7   | 45.7143      | 0.0143   | COCOK       |             
H14-39e | 18,7   | 18.7302      | 0.0302   | COCOK       |             
H14-39f | 16,5   | 16.5079      | 0.0079   | COCOK       |             
H14-39g | 11     | 11           | 0        | COCOK       |             
H14-39h | 0      | 0            | 0        | COCOK       |             
H14-39i | 4      | 4            | 0        | COCOK       |             
H14-40  | 610    | 610.2333     | 0.2333   | COCOK       |             
H14-41  | 478    | 478.0167     | 0.0167   | COCOK       |             
H14-42a | 324    | 323.83       | -0.17    | COCOK       |             
H14-42b | 1055   | 1054.78      | -0.22    | COCOK       |             
H14-42c | 1628   | 1627.67      | -0.33    | COCOK       |             
H14-42d | 345    | 344.91       | -0.09    | COCOK       |             
H14-42e | 922    | 921.99       | -0.01    | COCOK       |             
H14-42f | 1340   | 1339.5       | -0.5     | COCOK       |             
H14-42g | 186    | 186.41       | 0.41     | COCOK       |             
H14-42h | 219    | 219.3        | 0.3      | COCOK       |             
H14-42i | 241    | 240.66       | -0.34    | COCOK       |             
H14-42j | 202    | 202.3        | 0.3      | COCOK       |             
H14-42k | 222    | 222.17       | 0.17     | COCOK       |             
H14-42l | 220    | 220.32       | 0.32     | COCOK       |             
H14-43a | 75,5   | 93.1607      | 17.6607  | TIDAK COCOK |             
H14-43b | 75,1   | 98.0167      | 22.9167  | TIDAK COCOK |             
H14-43c | 93,2   | 75.4913      | -17.7087 | TIDAK COCOK |             
H14-43d | 98,0   | 75.094       | -22.906  | TIDAK COCOK |             
H14-44a | 0,089  | 0.0968       | 0.0078   | TIDAK COCOK |             
H14-44b | 0,088  | 0.0873       | -0.00073 | PEMBULATAN  |             
H14-44c | 0,097  | 0.0886       | -0.0084  | TIDAK COCOK |             
H14-44d | 0,087  | 0.0879       | 0.00087  | PEMBULATAN  |             
H14-45a | 2133   | 2132.95      | -0.05    | COCOK       |             
H14-45b | 1114   | 1113.76      | -0.24    | COCOK       |             
H14-45c | 1,7    | 1.7049       | 0.0049   | COCOK       |             
H14-45d | 6      | 6.7634       | 0.7634   | TIDAK COCOK |             
H14-30b | 0,028  | 0.0281       | 0.00011  | COCOK       |             
H14-30  | 0,077  | 0.0765       | -0.00052 | PEMBULATAN  |             
H14-36b | 0,675  | 0.6753       | 0.00030  | COCOK       |             
H14-38b | 0,307  | 0.3067       | -0.00027 | COCOK       |             
H14-46  | 0,21   | 0.206        | -0.004   | COCOK       |             
H14-47  | 0,53   | 0.5324       | 0.0024   | COCOK       |             
H14-48a | 0,80   | 0.8037       | 0.0037   | COCOK       |             
H14-48b | 0,87   | 0.8718       | 0.0018   | COCOK       |             
H14-48c | 0,65   | 0.6534       | 0.0034   | COCOK       |             
H14-48d | 0,63   | 0.6335       | 0.0035   | COCOK       |             
H04-01  | 0,860  | 0.86         | 0        | COCOK       |             
H04-02  | 0,899  | 0.899        | 0        | COCOK       |             
H04-03  | 0,880  | 0.88         | 0        | COCOK       |             
H04-04  | 0,980  | 0.98         | 0        | COCOK       |             
M-10a   | 10     | 11           | 1        | TIDAK COCOK |             
M-05    | 1      | 0            | -1       | TIDAK COCOK | mode=min    
M-01    | 11     | 11           | 0        | COCOK       |             
M-06    | 58     | 58.2683      | 0.2683   | COCOK       |             
M-07a   | 0,50   | 0.5019       | 0.0019   | COCOK       | mode=sekitar
M-07b   | 0,908  | 0.9083       | 0.00033  | COCOK       |             
M-07c   | 0,083  | 0.0833       | 0.00033  | COCOK       |             
M-01b   | 11     | 11           | 0        | COCOK       |             
M-02    | 0,028  | 0.0281       | 0.00011  | COCOK       |             
M-03a   | 0,675  | 0.6753       | 0.00030  | COCOK       |             
M-03b   | 0,634  | 0.6338       | -0.00017 | COCOK       |             
M-03c   | 9      | 9            | 0        | COCOK       |             
M-04a   | 0,307  | 0.3067       | -0.00027 | COCOK       |             
M-04b   | 0,295  | 0.2953       | 0.00033  | COCOK       |             
M-11a   | 0,902  | 0.902        | 0        | COCOK       |             
M-11b   | 0,846  | 0.846        | 0        | COCOK       |             
M-12a   | 0,920  | 0.92         | 0        | COCOK       |             
M-12b   | 0,800  | 0.8          | 0        | COCOK       |             
M-09a   | 0,429  | 0.4287       | -0.00030 | COCOK       |             
M-09b   | 0,433  | 0.4333       | 0.00030  | COCOK       |             
M-09c   | 0,81   | 0.8135       | 0.0035   | COCOK       |             
M-08    | 0,750  | 0.7502       | 0.00020  | COCOK       |             
P-01    | 10,2   | 10.1706      | -0.0294  | COCOK       |             
P-02    | 58     | 58.2683      | 0.2683   | COCOK       |             
P-07a   | 9,8    | 9.7819       | -0.0181  | COCOK       |             
P-07b   | 39,1   | 39.1278      | 0.0278   | COCOK       |             
P-07c   | 68,5   | 68.4736      | -0.0264  | COCOK       |             
P-08    | 26     | 26.3149      | 0.3149   | COCOK       |             
P-08c   | 98     | 97.8         | -0.2     | COCOK       |             
P-T1E   | 0,59   | 0.5945       | 0.0045   | COCOK       |             
P-T1J   | 0,67   | 0.6712       | 0.0012   | COCOK       |             
P-T2E   | 0,33   | 0.3283       | -0.0017  | COCOK       |             
P-T2J   | 0,35   | 0.3469       | -0.0031  | COCOK       |             
P-T3E   | 0,18   | 0.1836       | 0.0036   | COCOK       |             
P-T3J   | 0,20   | 0.1962       | -0.0038  | COCOK       |             
P-T4E   | 0,34   | 0.3386       | -0.0014  | COCOK       |             
P-T4J   | 0,43   | 0.4257       | -0.0043  | COCOK       |             
P-T5E   | 0,80   | 0.8037       | 0.0037   | COCOK       |             
P-T5J   | 0,87   | 0.8718       | 0.0018   | COCOK       |             
P-12    | 128    | 128          | 0        | COCOK       |             
P-17    | 0,027  | 0.0271       | 0.00007  | COCOK       |             
P-18    | 2,4    | 2.43         | 0.03     | COCOK       |             
P-19    | 60     | 60           | 0        | COCOK       |             

Baris ledger: 476 | dihitung ulang otomatis: 463 | tanpa fungsi hitung (diperiksa manual): 13
Status hitung ulang: COCOK=429, PEMBULATAN=11, TIDAK COCOK=23
TIDAK COCOK: R06-04, R10-01, R-60a, R-64, B-09c, B-09g, B-29, H11-08, H11-09, H11-10, H11-28, H11-27, H14-12, H14-13, H14-43a, H14-43b, H14-43c, H14-43d, H14-44a, H14-44c, H14-45d, M-10a, M-05
```

Kode keluar: 1
