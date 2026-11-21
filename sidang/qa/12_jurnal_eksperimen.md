# 12 — Jurnal eksperimen: perjalanan dari 2025 sampai versi terbaru

Riwayat penelitian deteksi fault 4 sensor kelembaban tanah (EDM-Fuzzy vs JSD-Fuzzy + ANN-LM)
dari commit pertama (2025-09-07) sampai 2026-11-21. Setiap fase memuat setup, apa yang
berubah, alasannya, dan dampaknya (sebelum → sesudah). Semua angka dikutip dari commit atau
file yang disebut; yang tidak punya bukti ditandai **"tidak ditemukan bukti"**.

Sumber: `git log --reverse` (114 commit), `git show <commit>:<path>`, output notebook,
`README.md`, `Bahan_Sidang.md`, `sidang/MAPPING.md`, `sidang/qa/*`, `hasil11/`, `hasil13/`,
`hasil14/`, `JSD_FUZZY_IMPROVEMENT.md`.

## Ringkasan satu layar

| | Versi awal (2025-12-14, `48966f8`) | Versi terbaru (2026-11-21) |
|---|---|---|
| Data | `tabel_sensor4_generated.csv`, 60.763 sampel, 4 sensor **dirata-rata** jadi satu sinyal | `data_sensor.csv` 281.721 baris 30 s → rata-rata 5 menit (28.173 baris), 4 kanal terpisah |
| Fault | drift rampa 0,15–0,2 per sampel (tidak fisis), bias/drift menandai seluruh deret | drift 1 poin/hari dalam episode 3 hari, spike/bias/hardware per sensor, label pasti |
| Fitur | 10 lag; entropi hanya dihitung untuk tabel CV, belum masuk classifier | 60 fitur entropi (4 sensor × τ = 1..15), EDM vs JSD beda hanya di ukuran jarak |
| Model | ANN 6–3, `scipy.least_squares` (Jacobian numerik), klasifikasi per sampel | ANN dengan LM asli (Jacobian analitik, redaman Nielsen), grid arsitektur di dalam fold |
| Evaluasi | split berurutan, tanpa CV, tanpa uji statistik | StratifiedGroupKFold blok waktu, uji-t + Wilcoxon + Nadeau–Bengio + Holm + bootstrap + purge + label acak |
| Hasil | F1 S1 0,99 · **S2 0** · S3 0,73 · S4 0,50 · S5 0,999 | NB11: JSD unggul 11/15 sel, F1 0,458 → 0,486 (Δ +0,028), p = 0,076 → "konsisten, bukan terbukti" |
| Sensor rusak | tidak bisa ditunjuk | sensor-ID F1 EDM 0,634 / JSD 0,675 |

Tiga titik balik terbesar: **(1)** Juli 2026 — definisi JSD diganti menjadi pengganti
langsung jarak Euclidean supaya perbandingan adil (sebelumnya 160 vs 40 fitur);
**(2)** Oktober 2026 — LM asli menggantikan `lbfgs`, lalu drift dibuat fisis (dulu 57,6
poin/hari), semua angka turun dan p naik dari 0,050 ke 0,076; **(3)** November 2026 —
ledger klaim, koreksi Holm, purge dan kontrol label acak membuat klaim dibatasi menjadi
"konsisten".

---

## Fase 0 — Repositori dan data awal (2025-09-07 → 2025-10-21)

- Commit `eb11117`, `1c0adf2`, `0210e30` (berkas audio, tidak terkait riset); `e47251e`
  (2025-10-21) menambah `tabel_sensor4_generated.csv`: 60.763 baris, kolom
  `timestamp,id,suhu,kelembaban1..4,ph1..4,moistanalog1..4,penyiraman,time_diff`.
  Baris pertama `1970-01-01 07:05:37` (artefak epoch), lalu 2025-09-14 14:50:07 sampai
  2025-10-17 12:24:49, interval ±30 s.
- Asal data dan arti "generated" tidak dijelaskan di commit (lihat Gap 2).

## Fase 1 — Versi awal `final_databaru_optimized.ipynb` (2025-12-14 → 12-15)

- Commit `48966f8` dan beberapa "add ipynb"/"remove" pada hari yang sama.
- **Setup:** empat sensor dirata-rata jadi `kelembaban_tanah_persen`. Simulator: drift
  `np.arange(len(x)) * intensity` (0,15–0,2 per sampel), spike periodik, bias konstan
  dengan mask seluruhnya True, hardware stuck/loss; fault S1 dipilih acak tanpa seed.
  Skenario S1 satu fault acak, S2 empat fault tunggal, S3 enam pasangan, S4 empat tripel,
  S5 empat sekaligus. ANN 1–2 lapis (6, 3, tanh, sigmoid) dengan
  `scipy.optimize.least_squares` (`max_nfev = 60`), biner per sampel, split berurutan.
  Grid search memakai `label_auto` (ambang level sinyal, bukan mask fault). Fungsi
  MSE/CMSE/FME/EDM/JSD ada untuk tabel CV ("Updated sesuai permintaan Bu Luh",
  skala 1, 5, 9, 13, 17; panjang 2000/4000/10000), belum menjadi fitur classifier.
- **Hasil:** grid search akurasi 0,980–0,999 (karena labelnya ambang level). Dengan label
  mask (sel 32): F1 uji S1 0,989 · S2 **0,000** · S3 0,727 · S4 0,496 · S5 0,999;
  waktu latih 484–849 s per skenario.
- **Cacat:** label grid search bukan fault; mask bias/drift = seluruh deret; drift tidak
  fisis; sensor tidak bisa dilokalisasi; S2 gagal total; entropi belum dipakai; tanpa CV,
  seed, atau uji statistik.

## Fase 2 — Revisi-2, deteksi per sensor (2025-12-14 → 12-28)

- Commit `ce0fb26`…`3b086db` (pesan "fix", "speed up", "update"), `7b9e2cf` (revisi-2-pisah),
  `4680b4d` (README, requirements, setup), `57fae25` (`EVALUASI_METODOLOGI.md`).
- **Berubah:** deteksi per sensor ("bukan berbasis sinyal gabungan"), split berurutan,
  ranking alarm terawal per sensor, fitur entropi 4 × S.
- **Dampak:** ANN multi-kelas pada nilai mentah akurasi 0,3026; grid `label_auto` turun ke
  0,82 (S1) / 0,95 (S2) / 0,93 (S5); per sensor: satu sensor F1 0,908, sensor lain 0,000;
  label drift 100 % positif.
- **Evaluasi diri** mencatat CV entropi gagal, label hanya dari sensor pertama, mapping
  indeks window salah, pengisian NaN ffill/0.

## Fase 3 — Notebook v6, EDM-Fuzzy berbasis window (2026-01-04 → 01-06)

- Commit `998fe20` (v6, menghapus revisi-2 dan evaluasi), `b1d084e`.
- **Setup:** DS = 2, WIN = 512, STRIDE = 256, 17 kelas, fault per sensor dengan parameter
  yang kelak dipakai NB11 (drift 0,02/sampel, spike 0,08·SD p = 0,015, bias 0,08,
  hardware 0,08/0,05), S = 10, m = 2, r = 0,2·SD; empat metode; sklearn `MLPClassifier`.
- **Hasil:** akurasi / macro-F1 EDM 0,355/0,328 · CMSE 0,344/0,310 · FME 0,361/0,335 ·
  **JSD 0,308/0,294 (terburuk)**. Biaya ekstraksi CMSE 5.088 s vs EDM 246 s.

**Jeda 2026-01-06 → 2026-06-26** tanpa commit riset (commit Maret hanya gambar tak terkait).

## Fase 4 — JSD "rich", perbaikan cache, Rencana Paper RQ1–RQ5 (2026-06-26 → 07-02)

- Commit `093d822`, `b3bb065`, `3398b3c`, `f7b37a0`, `b843da0`, `2eca409`, `d237006`.
- **Kenapa:** JSD lama hanya satu skalar per skala ("discarding the magnitude/location").
  `rich=True` mengeluarkan `[jsd, fe, mean_m, std_m]` per skala. Lalu ditemukan cache fitur
  dikunci nama metode saja, sehingga deskriptor rich tidak pernah terpakai dan
  perbandingan tidak adil (6 vs 10 skala).
- **Dampak:** benchmark 17 kelas JSD 0,341 → 0,421 akurasi. Setelah perbaikan cache
  (`b843da0`): F1 rata-rata EDM/JSD 0,465/0,541 → 0,454/0,655; JSD menang 4/5 → 5/5;
  p = 0,157 → 0,0046.
- **Masalah tersisa:** JSD rich 160 fitur vs EDM 40 fitur — tidak setara (diakui di Fase 7).

## Fase 5 — XGBoost, fitur hibrida, broker, scenario ladder (2026-07-03 → 07-15)

- Commit `a01012f`, `d84e3f8`, `c12f271`, `9b58bc4`.
- **Kenapa:** akurasi 17 kelas rendah (±0,41); entropi buta offset, jadi fitur level
  ditambahkan; ladder diselaraskan dengan skema pemrosesan klien.
- **Hasil:** multi-label JSD F1 0,85 / AUC 0,88; broker 4 kanal 0,85 vs fusi 0,80; ladder
  JSD > EDM 4/5 (0,80 vs 0,71). ANN "LM" masih `lbfgs` ("sklearn has no true LM").

## Fase 6 — Identifikasi sensor dua tahap; hasil tidak sah diarsipkan (2026-07-17 → 07-19)

- Commit `c4d913f`, `175333c`. **Kenapa:** permintaan pembimbing (WA 2026-07-17, voice note
  2026-07-19): akurasi per klasifikasi 17 kelas lalu identifikasi sensor yang salah.
- **Berubah:** injeksi sensor-selective; tahap 2 `MultiOutputClassifier`; kelas fault
  disubsampel 50:50 (tebakan mayoritas sebelumnya tampak 0,93).
- **Koreksi:** notebook per sensor lama dinyatakan **TIDAK SAH** — injeksi mengenai keempat
  sensor sehingga label kembar (prevalensi 0,882, ROC-AUC 0,50); F1 0,925 adalah artefak
  "selalu tebak fault" (`archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb`).

## Fase 7 — Data baru, flowchart pembimbing, grouped CV, JSD drop-in (2026-07-20 → 08-03)

- Commit `4e2288d`, `d324144`, `86e7f0f`, `1768d31`, `4d7a6ac`, `461b17e`, `46a4224`, dll.
- **Data:** `data_sensor.csv` 281.721 baris (2025-09-14 → 2025-12-21, 30 s; baris awal
  pecahan → hasil interpolasi), dirata-rata 5 menit, N = 200/700/1000.
- **NB07:** mengikuti flowchart pembimbing; EDM murni; StratifiedGroupKFold per blok waktu,
  grid search di dalam fold. `86e7f0f` membuang fitur waktu yang diam-diam ikut (208/448
  fitur, seharusnya 40/160).
- **NB08 — definisi JSD diganti** (`461b17e`): rich 160 vs 40 fitur "never structurally
  comparable". JSD menjadi pengganti langsung jarak Euclidean di similarity fuzzy
  (soft-binning Gaussian 8 bin). Softmax ditolak karena korelasi 0,98 dengan Euclidean;
  soft-binning 0,75. Hasil N = 200: JSD 0,624 vs EDM 0,541, 5/5, p = 0,028; JSD ±2× lebih
  lambat, jadi tidak ada klaim efisiensi. `46a4224` memperluas τ ke 1..15:
  0,627 vs 0,572, belum signifikan (p persis: tidak ditemukan bukti).

**Jeda 2026-08-03 → 2026-10-15.** NB09/NB10 dikerjakan (log 2026-09-29, 2026-10-11) tetapi
notebook, builder, dan `hasil09/`/`hasil10/` tidak pernah di-commit.

## Fase 8 — LM asli, Jacobian analitik; NB09/10/11 sebelum koreksi drift (2026-09-29 → 10-20)

- Commit `28f0885`, `131cdfb`, `39eeb86`, `19fc3e8`, `fa63f3f`.
- **Kenapa:** permintaan pembimbing — ulangi JSD vs EDM dengan **LM asli dari paper**,
  bukan `lbfgs`.
- **Setup NB11:** 60 fitur, `lm_fit` Gauss-Newton terdamping dengan Jacobian analitik
  ber-self-check, grid 8 arsitektur × relu/tanh, GroupKFold 5 luar / 2 dalam.
- **Hasil:** NB09 BR 0,504 · LM 0,504 · GA 0,335 · Adam 0,321. NB10 v3 pooled p = 0,048.
  NB11 v1: 11/15, +0,030, p = 0,0511, sensor-ID 0,770 vs 0,722, wall 525 menit.
  NB11 v2: 12/15, p = 0,0504, sensor-ID 0,765 vs 0,721, wall 588 menit.

## Fase 9 — Drift dibuat fisis; semua notebook dijalankan ulang (2026-10-22 → 10-23)

- Commit `9963cd2`, `4e1c56e`, `7892ed6`, `c9b05be`, `27737ee`.
- **Kenapa:** simulator lama menambah 0,2 poin per sampel = 57,6 poin/hari (+5.635 poin),
  melewati 100 % setelah ±22 jam — terdeteksi hanya karena artefak. Diganti
  `DRIFT_POINTS_PER_DAY = 1.0` dalam tiga episode 3 hari (10/40/70 % deret), clip 0–100.
- **Dampak:**

| Notebook | Sebelum | Sesudah |
|---|---|---|
| 02 broker 4 kanal vs rata-rata | 0,845 vs 0,797 | 0,821 vs 0,748 |
| 03 tingkat dimenangi JSD | 4/5 | 5/5 |
| 04 F1 sensor 17 kelas J/E | 0,899 / 0,860 | 0,902 / 0,846 |
| 08 F1 J/E | 0,627 / 0,572, tidak signifikan | 0,505 / 0,453, p = 0,035 |
| 09 BR/LM/GA/Adam | 0,504/0,504/0,335/0,321 | 0,429/0,433/0,306/0,268 |
| 10 pooled p | 0,048 | 0,052 (N = 700 p = 0,016) |
| 11 v2 → v3 | 12/15, p = 0,0504, sensor-ID 0,765/0,721 | 11/15, p = 0,076, sensor-ID 0,675/0,634 |

  Kelas drift jauh lebih sulit dideteksi; inilah sebab utama angka turun.

## Fase 10 — Edisi sidang, koreksi narasi ladder (2026-10-26 → 10-29)

- Commit `870dec1`, `37b07a9`, `97d8e31`, `ccf99b1`.
- Notebook sidang hanya membaca CSV ekspor dengan `assert`. Hasil sidang dibatasi ke ANN-LM.
  Ladder dikoreksi (17 kelas bukan S5; sensor-ID adalah RQ6 terpisah). Peringkat sensor
  yang salah diganti peringkat per skenario. Catatan nilai nol S4: 11.371 sampel (4,0 %),
  tiga blok macet 16–21 Des 2025.

## Fase 11 — Sensitivitas, intensitas, bentuk fault baru: NB12/13/14 (2026-11-03 → 11-12)

- **NB12** (`aec703a`): bias tak terlihat secara struktural (AUC ±0,50 untuk 0,08–5 poin);
  hardware (0,3; 0,2) TPR 0,908 / FPR 0,083; binning rekomendasi F1 validasi 0,805.
- **NB13** (`cd1417c`, `b012d1b`): hanya hardware lolos ambang deteksi mulai (0,3; 0,2);
  drift baru AUC 0,85 (J) / 0,80 (E) pada total 60 poin; spike JSD lebih peka
  (AUC 0,922 vs 0,815). Pada nilai penelitian TPR/FPR JSD 0,548/0,473.
- **NB14** (`ad56b3c`, `8a592e0`): pipeline NB11 dengan fault tingkat "sedang". JSD unggul
  12/15; F1 0,421 → 0,497, **p = 0,0006**, d = 1,13; sensor-ID 0,693 → 0,791. EDM menang di
  N700 S4, N700 S5, N1000 S3. Nilai "sedang" adalah asumsi (tab "Perubahan program" pada
  dokumen usulan tidak tersedia). Status: uji tambahan.

## Fase 12 — Data hipotetik berlabel, model tersimpan, ledger klaim, statistik (2026-11-20)

- **NB15** (permintaan dosen, chat 2026-11-18): basis program pertama (lag, tanpa entropi)
  dengan LM asli dan simulator NB14, split blok harian 3:1:1. v1 rata-rata F1 S1–S5
  0,90 · 0,85 · 0,83 · 0,88 · 0,96. v2 model gabungan 0,754 (4 sensor 0,616). v3 grid
  search: (16,8) F1 0,753, alarm palsu 0,8 %; 4 sensor (12,6) 0,640, alarm palsu 7,4 %.
- **NB15b:** inferensi portabel tanpa pelatihan, selisih 0 terhadap NB15.
- **NB16** (pembimbing, 2026-11-20: apakah model bisa disimpan): 10 model N = 700, 58 menit.
  Pickle sklearn 1.6.1 tidak terbaca di 1.9.1 → dikonversi ke `SavedLM` numpy (selisih 0,0).
- **Ledger klaim** (`eaf0743`): 26 klaim salah/meleset diperbaiki; angka arsip tak terbukti
  dibuang; `cek_klaim.py` → "GERBANG KLAIM: LULUS".
- **Statistik** (`1b57140`): NB11 N = 700 p = 0,027 → Holm 0,081; gabungan dz = 0,49,
  CI95 bootstrap [+0,0023; +0,0584]. NB14 CI [+0,0425; +0,1076].
- **Purge dan label acak** (`63e7aba`): purge mengubah F1 −0,072…+0,012; JSD ≥ EDM 4/5 → 5/5
  setelah purge; label acak F1 0,446–0,508 (tingkat tebakan).
- **Pustaka** (`e217535`): EDM-Fuzzy bersumber Zhou dkk. 2021 (IEEE TII); laju drift
  0,003–0,17 poin/hari diturunkan menjadi asumsi; JSD terbukti buta urutan.

## Fase 13 — Notebook sidang interaktif dan catatan internal (2026-11-20 → 11-21)

- Commit `e7389d3`…`edca7a9`. Widget hari/sensor/fault/intensitas dengan prediksi bobot
  numpy; 540 kombinasi < 1 s; Kaggle COMPLETE 23,6–24,5 s; widget terbukti berubah langsung
  di editor Kaggle (`sidang/bukti_kaggle/widget_editor_2026-11-21/`); Q&A T1–T57; catatan
  internal dipindah ke Artifact. Tidak ada angka model baru.

---

## Tabel ringkas sebelum → sesudah

| Titik balik | Metrik | Sebelum | Sesudah | Sumber |
|---|---|---|---|---|
| v6 → JSD rich | akurasi 17 kelas JSD | 0,308 (v6) / 0,341 | 0,421 | `998fe20`; `JSD_FUZZY_IMPROVEMENT.md` |
| Perbaikan cache | F1 E/J; p | 0,465/0,541; 0,157 | 0,454/0,655; 0,0046 | `b843da0` |
| JSD drop-in | F1 E/J; p | 0,454/0,655 (rich) | 0,541/0,624; 0,028 | `461b17e` |
| Per sensor → sensor-selective | F1 sensor | 0,925 (tidak sah) | 0,846/0,902 (NB04 v6) | `175333c`; README |
| lbfgs → LM asli | pooled p | NB08 0,035 | NB11 v3 0,076 | README |
| Koreksi drift | NB11 menang; p | 12/15; 0,0504 | 11/15; 0,076 | `19fc3e8`, `7892ed6` |
| Koreksi drift | NB11 sensor-ID E/J | 0,721/0,765 | 0,634/0,675 | idem |
| Koreksi drift | NB09 LM/BR | 0,504/0,504 | 0,433/0,429 | `4e1c56e` |
| Holm | NB11 N = 700 p | 0,027 | 0,081 | `statistik_sel.csv` |
| Fault baru NB14 | F1 E/J; p | 0,458/0,486; 0,076 | 0,421/0,497; 0,0006 | `hasil14/14_draf_laporan.md` |
| Purge N = 700 | JSD ≥ EDM | 4/5 | 5/5 | `efek_purge_permutasi.csv` |
| NB15 v2 → v3 | F1 4 sensor | 0,616 | 0,640 | `2b43aea`, `01cb43a` |
| Runtime NB11 | wall | 525 menit (v1) | 610 menit (v3) | `131cdfb`, Ringkasan v3 |

## Gap, inkonsistensi, dan yang belum selesai

1. **Jeda kosong:** 2026-01-06 → 06-26 (±5,5 bulan) dan 2026-08-03 → 10-15. NB09/NB10 dari
   jeda kedua tidak pernah di-commit; bukti NB10 v4 hanya di `sidang/bukti_kaggle/`.
2. **Asal-usul data belum dikonfirmasi:** apa yang "generated" pada
   `tabel_sensor4_generated.csv`, dan siapa yang menginterpolasi `data_sensor.csv`.
   Keduanya mulai di stempel yang sama (2025-09-14 14:50:07). Checklist pembimbing A1/A2.
3. **Paper ACS:** judul/DOI tidak ditemukan di Crossref/OpenAlex, padahal 15b menyebutnya
   terbit. Checklist A3.
4. **Sumber laju drift:** tidak ada; 1 poin/hari juga asumsi. Checklist A4.
5. **Arah dan signifikansi berganti-ganti:** JSD terburuk (Jan) → terbaik (Jun); p 0,157 →
   0,0046 → 0,028 → tidak signifikan → 0,035 → 0,048/0,052 → 0,0511/0,0504/0,076 → 0,0006
   (NB14). Beberapa sel NB11 negatif (N200 S1/S3, N1000 S2/S3).
6. **Definisi JSD berubah di tengah jalan** (rich → drop-in). README baris 01 masih
   menampilkan angka JSD rich dengan MLP sklearn yang tidak sebanding. Notebook 01–08 tidak
   memakai LM.
7. **Non-determinisme:** NB10 dan NB11 berprotokol sama tetapi p berbeda (0,052 vs 0,076);
   tujuh sel terpotong `MAX_TRAIN_S`; belum ada ulangan multi-seed.
8. **Cacat kode diketahui, belum di-run ulang:** bobot keluaran mulai nol (`lm_pack`),
   sensor-ID memakai split acak 70/30, langkah LM yang ditolak menghabiskan iterasi,
   hardware efektif 5 % + 3 %, spike NB11 periodik tanpa seed, kebocoran di batas grup
   (sebagian sudah diukur dengan purge).
9. **Banyak uji, sel tidak independen:** Wilcoxon n = 5 tidak bisa di bawah 0,0625; setelah
   Holm, N = 700 NB11 tidak signifikan.
10. **Klaim yang dibuang:** F1 0,925 per sensor, JSD rich, fitur hibrida, XGBoost/BR/GA dari
    edisi sidang, klaim efisiensi, "Xavier dengan bias nol", angka arsip tak terbukti,
    ringkasan `hasil04` v5.
11. **Label "Revisi pasca-sidang"** pada NB13/14: sidang yang dimaksud tidak terdokumentasi.
12. **Batas representasi:** bias konstan tak terlihat, drift realistis sulit, S2–S4 F1
    0,21–0,53, EDM lebih stabil, JSD buta urutan.
13. **Artefak di luar git:** `hasil*/exports/` di-gitignore; sebagian bukti hanya di Kaggle.
14. **Pesan commit 2025 tidak informatif** ("add", "fix", "wip"); alasan perubahan Desember
    2025 hanya bisa direkonstruksi dari diff.
