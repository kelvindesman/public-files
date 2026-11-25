# 06 — Skenario mock sidang

Tujuannya menguji jawaban di depan orang lain, bukan membaca ulang catatan.
Bukti bahwa persiapan berhasil datang dari penguji tiruan, bukan dari diri
sendiri.

## Peran

- **Penguji 1 — metode** (Bu Luh atau teman satu lab): fokus validitas dan
  statistik (`02` bagian A, C; `05` §4).
- **Penguji 2 — algoritma & kode**: fokus LM, entropi, kode (`05` §1–2, §5;
  `03`, `04`). Boleh menunjuk baris kode acak di `deteksi_fault_core.py`.
- **Pencatat**: menandai setiap jawaban 0/1/2 (rubrik di bawah) dan mencatat
  waktu.

Penguji membaca `02`, `05`, dan `01_ringkasan_klaim.md`. Kandidat **tidak**
boleh membuka file jawaban selama sesi; hanya notebook/Kaggle untuk bukti.

## Alur (±60 menit, direkam)

| Menit | Kegiatan |
|---|---|
| 0–15 | Presentasi: masalah → data → fault → entropi EDM vs JSD → ANN-LM → hasil NB11 → uji tambahan NB13/14 → batasan |
| 15–30 | Ronde 1 (Penguji 1): 6 pertanyaan dari `05` §3–4 + 2 dari `02` |
| 30–45 | Ronde 2 (Penguji 2): tulis rumus LM dan FuzzyEn di papan; 2 baris kode acak "kenapa begini?" |
| 45–55 | Ronde gangguan (lihat bawah) |
| 55–60 | Umpan balik dan daftar jawaban yang gagal |

## Ronde gangguan (latihan saat sesuatu tidak berjalan)

1. **Demo Kaggle tidak terbuka / kernel error** → buka PNG dan CSV hasil yang
   sudah diekspor (`hasil11/`, `sidang/assets/`), katakan "hasil ini dari run
   versi X, log-nya tersimpan".
2. **Angka di layar beda dengan slide** → "Saya cek sumbernya" → buka
   `01_klaim_bukti.csv`, sebut angka yang berstatus COCOK.
3. **Penguji menyebut angka yang tidak ada** → jangan mengiyakan; "Angka yang
   saya punya adalah …, dari file …".
4. **Penguji menolak klaim 'konsisten'** → kalimat siap pakai no. 2
   (`Bahan_Sidang.md`) + Bonferroni (`02` C3).
5. **Model salah tebak saat demo** → `sidang/PANDUAN_DEMO.md` §7.

## Rubrik per jawaban

- **2** — benar, ≤ 30 detik, menyebut bukti, mengakui batasan bila relevan.
- **1** — benar tetapi bertele-tele, tanpa bukti, atau ragu-ragu.
- **0** — salah, mengarang angka, atau mengklaim lebih dari bukti.

Target lulus mock: rata-rata ≥ 1,6, **tidak ada nilai 0 untuk angka**. Ulangi
mock untuk semua pertanyaan bernilai 0–1 sampai bernilai 2.

## Setelah mock

- Tambahkan pertanyaan baru dari penguji tiruan ke `05_bank_pertanyaan.md`.
- Jalankan ulang `./venv/bin/python sidang/qa/cek_klaim.py` dan
  `./venv/bin/python sidang/qa/cek_algoritma_tangan.py` setiap kali ada angka
  atau kode yang berubah; keduanya harus lolos sebelum sidang.
