# 09 — Checklist tanda tangan pembimbing

Satu halaman untuk dibawa ke Bu Luh sebelum sidang. Setiap baris butuh
keputusan pembimbing (setuju / ubah / hapus), bukan hanya dibaca. Baris yang
belum disetujui **tidak** boleh muncul di slide.

Commit acuan: tulis hash `git log -1 --format=%h` saat checklist dicetak:
`________`.

## A. Fakta yang hanya bisa dipastikan pembimbing

| # | Pertanyaan | Kenapa penting | Jawaban pembimbing |
|---|---|---|---|
| A1 | Asal-usul `data_sensor.csv`: sensor apa, lokasi, siapa yang menginterpolasi ke grid 30 detik? (`02_kelemahan.md` B6) | Penguji hampir pasti bertanya "data asli atau buatan?" | |
| A2 | Apa yang "generated" pada `tabel_sensor4_generated.csv` (NB15)? | Disebut "data buatan" di chat 18/11/26 | |
| A3 | Judul dan DOI paper ACS (EDM-Fuzzy + ANN-LM). Apakah sama dengan Wardhani dkk. 2026 JITK, atau paper lain? (`07_pustaka_dan_kebaruan.md` (a)) | Belum ditemukan di Crossref/OpenAlex; tidak boleh disitasi tanpa judul pasti | |
| A4 | Laju drift 0,003–0,17 poin/hari: ada sumber? Kalau tidak, setuju ditulis "asumsi skenario"? | Tidak ditemukan sumber terbit (`07` (e)) | |

## B. Klaim yang akan diucapkan

| # | Klaim (persis seperti di slide) | Bukti | Setuju? |
|---|---|---|---|
| B1 | JSD-Fuzzy lebih tinggi di 11 dari 15 sel, rata-rata selisih F1 +0,028 (NB11) | `11_rq4_selisih.csv`; ledger | |
| B2 | Gabungan NB11 belum signifikan: p = 0,076 (Wilcoxon 0,107); disebut "konsisten", bukan "terbukti" | `statistik_sel.csv` | |
| B3 | N = 700: p = 0,027 tanpa koreksi, p = 0,081 setelah Holm (3 ukuran jendela) | `statistik_sel.csv` kolom `p_t_holm_3N` | |
| B4 | NB14 gabungan: p = 0,0006, d_z = 1,13, CI bootstrap 95 % [+0,043; +0,108] | `statistik_sel.csv` | |
| B5 | Sensor-ID JSD 0,675 vs EDM 0,634, dengan catatan split acak 70/30 (optimistis) | NB11 §12; `02` F1 | |
| B6 | Kebaruan = JSD sebagai jarak antar template di dalam fuzzy entropy; EDM-Fuzzy dari Zhou dkk. 2021 | `07` (b) | |
| B7 | Hasil purge dan kontrol label acak (angka dari `efek_purge_permutasi.csv`) | `hasil_uji/` | |

## C. Batasan yang akan diakui sendiri

| # | Batasan | Setuju diucapkan? |
|---|---|---|
| C1 | Gangguan disimulasikan, bukan kerusakan fisik | |
| C2 | JSD-Fuzzy buta urutan (template naik vs turun memberi JSD = 0) | |
| C3 | Sedikit kebocoran di batas grup CV (diukur dengan purge) | |
| C4 | Bobot awal tidak persis Xavier (`lm_pack`/`lm_unpack`), sama untuk kedua metode | |
| C5 | Bias konstan tidak terdeteksi karena template dipusatkan | |
| C6 | Hardware malfunction efektif 5 % hilang + 3 % macet | |

## D. Gerbang teknis (diisi kandidat sebelum bertemu pembimbing)

Tempel keluaran baris terakhir tiap perintah:

```
./venv/bin/python sidang/qa/cek_klaim.py                  -> GERBANG KLAIM: ______
./venv/bin/python -m pytest -q sidang/qa                  -> ___ passed
./venv/bin/python sidang/qa/jalankan_notebook.py sidang/04_Sidang_Interaktif.ipynb \
    --keluar sidang/bukti_demo --batas-sel 2 --batas-total 60   -> HASIL: ______
./venv/bin/python sidang/qa/cek_algoritma_tangan.py       -> semua [OK]? ______
```

## Tanda tangan

Pembimbing: ______________________  Tanggal: ____________

Kandidat:   ______________________  Tanggal: ____________
