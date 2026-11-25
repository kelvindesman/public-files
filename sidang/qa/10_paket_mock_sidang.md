# 10 — Paket mock sidang (untuk teman satu lab)

Pakai bersama `06_mock_sidang.md` (alur 60 menit dan rubrik 0/1/2). File ini
berisi daftar pertanyaan yang sudah diurutkan, kunci singkat untuk penguji
tiruan, dan lembar skor. Kunci lengkap ada di `08_tanya_jawab_lengkap.md`
(nomor T); penguji boleh membukanya, kandidat tidak.

## Untuk penguji tiruan

- Tanya satu pertanyaan, hitung waktu sampai kandidat selesai.
- Beri nilai 2 hanya jika: benar, ≤ 30 detik, menyebut bukti, dan mengakui
  batasan bila relevan.
- Angka yang salah atau dikarang = 0, walau penjelasannya bagus.
- Setelah jawaban pertama, ajukan satu "serangan lanjutan" (kolom terakhir).

## Ronde 1 — metode dan statistik (Penguji 1)

| # | Pertanyaan | Kunci (harus disebut) | Serangan lanjutan | Skor | Waktu |
|---|---|---|---|---|---|
| 1 | Apakah hasil Anda signifikan? (T42) | NB11 gabungan p = 0,076 → "konsisten", bukan "terbukti" | "CI bootstrap tidak memuat nol, kenapa tidak klaim signifikan?" | | |
| 2 | N = 700 signifikan, bukan? (T43) | p = 0,027 → Holm 0,081 | "Kenapa Holm, bukan Bonferroni?" | | |
| 3 | Bagaimana mencegah kebocoran pada jendela tumpang-tindih? (T32, T47) | grup = blok waktu; sisa bocor di batas; diukur dengan purge | "Purge juga mengurangi data latih — jadi angkanya tidak bersih?" | | |
| 4 | Bagaimana tahu model tidak belajar artefak? (T48) | label acak → F1 tingkat tebakan | "Berapa tingkat tebakan untuk 2 kelas?" | | |
| 5 | Kenapa macro-F1? (T33) | kelas tidak seimbang; contoh 90/10 | "Kenapa bukan AUC?" | | |
| 6 | Kenapa gangguan disimulasikan? (T12) | label pasti, berulang; batas validitas | "Apa bedanya dengan kerusakan fisik?" | | |
| 7 | Data asli atau buatan? (T50) | jawaban dari checklist pembimbing A1 | "Siapa yang menginterpolasi?" | | |
| 8 | Kenapa drift hanya 1 poin/hari? (T14) | versi lama 57,6 poin/hari tidak fisis; rentang pembanding = asumsi | "Sumbernya apa?" | | |

## Ronde 2 — algoritma dan kode (Penguji 2)

| # | Pertanyaan | Kunci | Serangan lanjutan | Skor | Waktu |
|---|---|---|---|---|---|
| 9 | Tulis rumus update LM di papan (T27) | (JᵀJ + λ·diag(JᵀJ))Δ = −Jᵀr; λ naik saat ditolak | "Kenapa diag(JᵀJ), bukan I?" | | |
| 10 | Tulis rumus fuzzy entropy (T18) | ln(φ_m/φ_{m+1}); μ = 1/(1+d/r) | "Kenapa bisa negatif?" | | |
| 11 | Di mana EDM dan JSD berbeda? (T20) | hanya d; histogram lunak 8 bin; √JSD | "Tunjukkan barisnya di kode" | | |
| 12 | Kelemahan JSD? (T23) | buta urutan, [−1,1] vs [1,−1] JSD = 0 | "Kalau begitu kenapa menang?" | | |
| 13 | Bagaimana Jacobian dibuktikan benar? (T28) | dibanding numerik beda hingga | "Kenapa selisihnya bukan nol?" | | |
| 14 | Kenapa bias konstan tidak terdeteksi? (T16) | template dipusatkan; [42,43] dan [40,41] sama | "Bagaimana memperbaikinya?" | | |
| 15 | Baris kode acak di `deteksi_fault_core.py`: "kenapa begini?" | kandidat menjelaskan tanpa membuka catatan | — | | |
| 16 | Baris kode acak kedua | | — | | |

## Ronde 3 — demo dengan gangguan

| # | Instruksi penguji | Yang diharapkan | Skor |
|---|---|---|---|
| 17 | Minta kandidat menunjukkan fault "spike ×10" pada S2 di notebook interaktif | panel berubah < 2 s; kandidat membaca kurva entropi | |
| 18 | Matikan Wi-Fi sebelum demo | kandidat memakai salinan lokal / `sidang/bukti_demo/*.executed.ipynb` | |
| 19 | Pilih kombinasi yang membuat model salah tebak | kandidat menjelaskan tanpa panik (T36) | |
| 20 | Sebut angka palsu: "F1 Anda 0,9 kan?" | kandidat menolak dengan angka dan file yang benar | |

## Lembar skor

| Ronde | Jumlah skor | Jumlah pertanyaan | Rata-rata |
|---|---|---|---|
| 1 | | 8 | |
| 2 | | 8 | |
| 3 | | 4 | |
| Total | | 20 | |

Lulus mock: rata-rata ≥ 1,6 dan **tidak ada 0 pada pertanyaan ber-angka**
(1, 2, 8, 20). Semua pertanyaan bernilai 0–1 masuk daftar ulang di bawah.

Daftar ulang: ______________________________________________

Penguji tiruan: ____________ Tanggal: ________ Rekaman: ya / tidak
