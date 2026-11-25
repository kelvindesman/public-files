#!/usr/bin/env python3
"""Builder notebook sidang interaktif: sidang/04_Sidang_Interaktif.ipynb.

Satu notebook untuk dua pemakaian:
  * presentasi  — catatan internal disembunyikan (tombol "Mode presentasi" di sel
                  pertama, atau salinan `*_presentasi.ipynb` yang catatan internalnya
                  sudah dibuang lewat ekspor_notebook_sidang.py);
  * internal    — tiap bagian punya catatan lipat (<details>) berisi
                  penjelasan dari nol, tanya-jawab penguji, dan penunjuk bukti.

Aturan isi:
  * Tidak ada pelatihan model. Panel menghitung fault dan entropi langsung dengan
    deteksi_fault_core (fungsi penelitian yang sama) dari data 5 menit di
    sidang/assets/cache_interaktif.npz; model N = 200/700/1000 dibaca dari
    sidang/assets/model_sidang.npz (keduanya dibuat sidang/qa/buat_cache_interaktif.py).
  * Angka hasil dibaca dari file ekspor saat notebook dijalankan, tidak diketik di
    markdown. Catatan internal menunjuk id ledger (sidang/qa/01_klaim_bukti.csv).
  * Tanya-jawab di catatan internal diambil dari sidang/qa/08_tanya_jawab_lengkap.md
    (bagian bertanda `<!-- bagian: X -->`), jadi hanya ada satu sumber jawaban.

Jalankan: ./venv/bin/python build_sidang_interaktif_notebook.py
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "sidang" / "04_Sidang_Interaktif.ipynb"
QA = ROOT / "sidang" / "qa" / "08_tanya_jawab_lengkap.md"
TAG = "catatan-internal"

cells: list = []


def md(s: str):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s: str):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


def qa_bagian(bagian: str) -> str:
    if not QA.exists():
        return f"_Tanya-jawab bagian ini belum ditulis (`{QA.relative_to(ROOT)}`)._"
    t = QA.read_text(encoding="utf-8")
    m = re.search(rf"<!-- bagian: {re.escape(bagian)} -->(.*?)(?=<!-- bagian: |\Z)", t, re.S)
    if not m:
        return f"_Belum ada tanya-jawab bertanda `{bagian}` di `{QA.name}`._"
    return m.group(1).strip()


def internal(bagian: str, judul: str, penjelasan: str, bukti: str):
    """Sel markdown bertanda catatan-internal: lipat, hilang di mode/salinan presentasi."""
    isi = (f'<details class="{TAG}">\n<summary><b>Catatan internal — {judul}</b> '
           f'(klik untuk buka; tidak tampil saat presentasi)</summary>\n\n'
           f"#### Penjelasan\n\n{penjelasan.strip()}\n\n"
           f"#### Tanya-jawab penguji\n\n{qa_bagian(bagian)}\n\n"
           f"#### Bukti\n\n{bukti.strip()}\n\n</details>")
    c = nbf.v4.new_markdown_cell(isi)
    c.metadata["tags"] = [TAG]
    cells.append(c)


# ============================================================== 0. pembuka
md("""
# Deteksi Fault 4 Sensor Kelembaban Tanah: EDM-Fuzzy vs JSD-Fuzzy + ANN-LM

**Notebook sidang interaktif.** Semua angka dibaca dari hasil penelitian yang sudah
diekspor; tidak ada model yang dilatih ulang di sini, jadi tiap sel selesai dalam
hitungan detik.

Cara pakai: **Run All** sekali, lalu ubah pilihan di panel kontrol: panjang jendela
(N = 200/700/1000, bawaan 1000), hari mulai, sensor yang dirusak, kombinasi fault, dan
intensitas. Fault dan entropi dihitung langsung dengan fungsi penelitian yang sama;
grafik dan tebakan model ikut berubah.
""")

code(r"""
# === Setup 1/3: pustaka angka ===
import json, sys, time
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd
""")

code(r"""
# === Setup 2/3: grafik dan widget ===
import matplotlib.pyplot as plt
import ipywidgets as w
from IPython.display import display, HTML, Markdown
""")

code(r"""
# === Setup 3/3: lokasi file, mode presentasi ===

# Lokal: repo penelitian. Kaggle: file dataset `sidang-bahan` (datar) disusun ulang
# ke struktur folder repo di /kaggle/working/bahan memakai symlink.
BERKAS = {"cache_interaktif.npz": "sidang/assets", "model_sidang.npz": "sidang/assets",
          "deteksi_fault_core.py": ".",
          "11_rq5_f1_berpasangan.csv": "hasil11/exports", "11_sensor_ringkas.csv": "hasil11/exports",
          "statistik_sel.csv": "sidang/qa/hasil_uji", "statistik_lipatan.csv": "sidang/qa/hasil_uji",
          "efek_purge_permutasi.csv": "sidang/qa/hasil_uji"}
ROOT = next((p for p in [Path(".."), Path(".")]
             if (p / "sidang" / "assets" / "cache_interaktif.npz").exists()), None)
if ROOT is None:
    import glob
    ROOT = Path("/kaggle/working/bahan")
    for nama, folder in BERKAS.items():
        asal = glob.glob(f"/kaggle/input/**/{nama}", recursive=True)
        if asal:
            (ROOT / folder).mkdir(parents=True, exist_ok=True)
            tujuan = ROOT / folder / nama
            if not tujuan.exists():
                tujuan.symlink_to(asal[0])
C = np.load(ROOT / "sidang" / "assets" / "cache_interaktif.npz")
sys.path.insert(0, str(ROOT.resolve()))
import deteksi_fault_core as core   # rumus entropi dan simulator fault penelitian (notebook 11)
pd.set_option("display.precision", 3)
plt.rcParams.update({"figure.dpi": 90, "axes.spines.top": False, "axes.spines.right": False})

_gaya = w.Output()
def _mode(presentasi):
    with _gaya:
        _gaya.clear_output()
        if presentasi:
            display(HTML("<style>.catatan-internal{display:none!important}</style>"))
tombol_mode = w.ToggleButton(value=False, description="Mode presentasi", icon="eye-slash",
                             tooltip="Sembunyikan semua catatan internal")
tombol_mode.observe(lambda ch: _mode(ch["new"]), "value")
display(w.HBox([tombol_mode, _gaya]))
print("Folder kerja:", ROOT.resolve())
""")

internal("pembuka", "cara memakai notebook ini",
         """
- **Mode presentasi**: tekan tombol di atas. Semua kotak "Catatan internal" hilang
  dari layar. Kalau tombol tidak bekerja (mis. di VS Code atau tampilan statis
  Kaggle), buka salinan `04_Sidang_Interaktif_presentasi.ipynb` yang memang tidak
  berisi catatan internal.
- **Alur bicara** (±15 menit): masalah → data → fault → entropi (EDM vs JSD) →
  ANN-LM → tebakan model → hasil → statistik → kelemahan → penutup.
- **Aturan menjawab**: angka hanya boleh disebut kalau tampil di notebook atau
  berstatus COCOK di `sidang/qa/01_klaim_bukti.csv`. Kalau ragu: "saya tunjukkan
  di notebook", lalu buka bagian yang relevan.
""",
         "- Gerbang angka: `./venv/bin/python sidang/qa/cek_klaim.py` → `GERBANG KLAIM: LULUS`.\n"
         "- Gerbang notebook: `sidang/qa/jalankan_notebook.py` (waktu tiap sel dicatat).")

# ============================================================== 1. masalah
md("""
## 1. Masalahnya

Sensor kelembaban tanah di sistem irigasi bisa rusak diam-diam: nilainya bergeser,
melonjak, merayap naik, atau macet. Sistem yang percaya pada sensor rusak akan
menyiram terlalu banyak atau terlalu sedikit.

**Pertanyaan penelitian:** kalau sinyal 4 sensor dipotong per jendela waktu dan
tiap sensor diringkas dengan **entropi multiskala**, apakah mengganti ukuran
kemiripan **Euclidean (EDM-Fuzzy)** dengan **Jensen–Shannon Divergence
(JSD-Fuzzy)** membuat ANN-LM lebih baik mengenali (1) ada tidaknya fault,
(2) jenis/kombinasi fault, dan (3) sensor mana yang rusak?
""")

internal("masalah", "masalah dan kontribusi",
         """
Bayangkan empat termometer di satu ruangan. Kalau satu menunjukkan angka aneh,
kita curiga termometer itu rusak karena tiga lainnya "normal". Penelitian ini
melakukan hal yang sama untuk sensor kelembaban tanah, tetapi tidak membandingkan
angkanya langsung. Yang dibandingkan adalah **pola** sinyalnya: seberapa teratur
atau acak naik-turunnya. Ukuran keteraturan itu disebut **entropi**.

Entropi dihitung dengan cara membandingkan potongan-potongan kecil sinyal: "apakah
potongan ini mirip potongan itu?". Cara mengukur "mirip" itulah yang diganti:
- **EDM** (Euclidean): jarak garis lurus antar angka, seperti penggaris.
- **JSD**: tiap potongan diubah jadi histogram kecil, lalu dibandingkan bentuk
  histogramnya.

Kontribusinya: (1) JSD sebagai ukuran kemiripan di dalam fuzzy entropy multiskala,
(2) evaluasi berantai sampai identifikasi sensor, (3) pelaporan jujur kapan
keunggulannya signifikan dan kapan tidak. Rumusan kebaruan yang aman ada di
`sidang/qa/07_pustaka_dan_kebaruan.md`.
""",
         "- `Bahan_Sidang.md` Q1–Q5; `sidang/qa/07_pustaka_dan_kebaruan.md` (prior-art).")

# ============================================================== 2. data
md("""
## 2. Data: 4 sensor, ±98 hari

Sensor mencatat tiap 30 detik; broker menyinkronkan keempatnya lalu merata-rata
per 5 menit. Grafik di bawah: rata-rata per jam, seluruh periode.
""")

code(r"""
jam = C["ringkas_per_jam"]
hari = np.arange(len(jam)) / 24
fig, ax = plt.subplots(figsize=(11, 3))
for j in range(4):
    ax.plot(hari, jam[:, j], lw=0.8, label=f"S{j+1}")
for h0, h1 in core.drift_episode_days(int(C["n_sampel_5menit"])):
    ax.axvspan(h0, h1, color="0.85", zorder=0)
ax.set_xlabel("hari sejak awal rekaman"); ax.set_ylabel("kelembaban (%)")
ax.set_title("Data lapangan (abu-abu = tiga episode drift simulator, 3 hari per episode)")
ax.legend(ncol=4, fontsize=8); plt.tight_layout(); plt.show()
print(f"{int(C['n_sampel_5menit']):,} titik 5 menit = {int(C['n_sampel_5menit']) * 300 / 86400:.1f} hari")
""")

internal("data", "data, sinkronisasi, nilai nol S4",
         """
- **Broker** = penampung yang menerima data keempat sensor dan menyusunnya di satu
  sumbu waktu. Sensor tidak dirata-rata menjadi satu karena identitas tiap sensor
  diperlukan untuk menunjuk sensor yang rusak (notebook 02).
- **5 menit**: 10 titik 30 detik dirata-rata jadi satu. Derau berkurang, pola
  harian tetap.
- **Tidak ada baris dibuang karena nilainya aneh.** Nilai macet dan nol di S4
  dilaporkan, bukan dibuang, karena justru bisa gejala fault.
- Bagian grafik yang lurus sempurna di keempat sensor ada di file data asli,
  bukan hasil pengolahan.
""",
         "- Ledger: B-09a…g (nilai nol S4), R-ILU1…5 (mutu data), P-08 (segmen lurus).\n"
         "- Notebook: `Ilustrasi_Data_Akuisisi_dan_Fault.ipynb` §1–§2, `07` §3b.")

# ============================================================== kontrol bersama
code(r"""
# === Panel kontrol (dipakai bagian 3, 4, dan 6) ===
# Model tersimpan dimuat di sini supaya pilihan panjang jendela = N yang modelnya ada.
MZ = np.load(ROOT / "sidang" / "assets" / "model_sidang.npz")
META = {int(k): v for k, v in json.loads(str(MZ["_meta"])).items()}
X5 = np.asarray(C["X5"], float); PER_HARI = int(86400 / core.SAMPLING_SECONDS)
METODE = [str(m) for m in C["metode"]]
FAULT4 = ["drift", "spike", "bias", "hardware"]

pil_N = w.Dropdown(options=[(f"N = {n} ({n * 300 / 3600:.0f} jam)", n) for n in META], value=1000,
                   description="Panjang")
pil_hari = w.FloatSlider(value=10, min=0, max=round((len(X5) - max(META)) / PER_HARI, 1), step=0.5,
                         description="Mulai hari", continuous_update=False, readout_format=".1f")
pil_sensor = w.SelectMultiple(options=[(f"S{s+1}", s) for s in range(4)], value=(1,), rows=4,
                              description="Sensor rusak")
pil_fault = w.SelectMultiple(options=FAULT4, value=tuple(FAULT4), rows=4, description="Fault")
pil_int = w.FloatLogSlider(value=1, base=10, min=-1, max=2, step=0.1, description="Intensitas ×",
                           continuous_update=False, readout_format=".2g")
pil_lihat = w.Dropdown(options=[(f"S{s+1}", s) for s in range(4)], value=1, description="Gambar sensor")
PANEL = w.VBox([w.HBox([pil_N, pil_hari, pil_int]), w.HBox([pil_sensor, pil_fault, w.VBox([pil_lihat])])])
KONTROL = dict(n=pil_N, hari=pil_hari, sensor=pil_sensor, fault=pil_fault, k=pil_int, lihat=pil_lihat)


def _skala(fn, kw, k):
    # Intensitas mengalikan amplitudo drift/spike/bias; peluang hardware dikali min(k, 5), maks 0,6.
    kw = dict(kw)
    if fn is core.simulate_bias_fault:
        kw["bias"] *= k
    elif fn is core.simulate_hardware_fault:
        kw = {a: min(0.6, v * min(k, 5)) for a, v in kw.items()}
    else:
        kw["intensity"] *= k
    return kw


@lru_cache(maxsize=32)
def deret_rusak(sensor, fault, k):
    # Fault disuntikkan ke SELURUH deret lalu jendela dipotong, persis seperti data latih
    # (core.segment_and_label); urutan komponen = definisi kondisi di core.SCENARIOS.
    if not sensor or not fault:
        return X5, np.zeros(X5.shape, bool), "normal"
    kondisi = core.condition_for(fault)
    faults = [(fn, _skala(fn, kw, k)) for fn, kw in core.SCENARIOS[kondisi]]
    Y, M = core.inject_faults_multisensor(X5, faults, list(sensor), seed=2026)
    return Y, M, kondisi


@lru_cache(maxsize=2048)
def fitur_sensor(n, mulai, s, mi, kunci):
    # Entropi multiskala 1 sensor (15 skala). float32 seperti core.make_windows; sd = 7 + 19·s
    # seperti core.concat_multisensor_features untuk jendela pertama.
    Y = X5 if kunci is None else deret_rusak(*kunci)[0]
    x = Y[mulai:mulai + n, s].astype(np.float32).astype(float)
    return core.entropy_1d(x, METODE[mi], sd=7 + 19 * s)


def jendela(n, hari, sensor, fault, k, **_):
    sensor, fault = tuple(sorted(sensor)), tuple(f for f in FAULT4 if f in fault)
    mulai = min(int(round(hari * PER_HARI)), len(X5) - n)
    kunci = (sensor, fault, float(k)) if sensor and fault else None
    Y, M, kondisi = deret_rusak(*kunci) if kunci else (X5, np.zeros(X5.shape, bool), "normal")
    F = {mi: (np.concatenate([fitur_sensor(n, mulai, s, mi, None) for s in range(4)]),
              np.concatenate([fitur_sensor(n, mulai, s, mi, kunci if s in sensor else None) for s in range(4)]))
         for mi in range(len(METODE))}
    return dict(mulai=mulai, bersih=X5[mulai:mulai + n], rusak=Y[mulai:mulai + n], mask=M[mulai:mulai + n],
                kondisi=kondisi, sensor=sensor, fault=fault, F=F)

def hidupkan(fn, kontrol=None, panel=None):
    # Tampilkan hasil fn(**pilihan) sebagai output sel biasa (terlihat juga di tampilan
    # statis) lalu perbarui di tempat setiap kali pilihan panel berubah.
    kontrol = kontrol or KONTROL
    display(panel or PANEL)
    pegangan = []
    def render(*_):
        isi = fn(**{k: v.value for k, v in kontrol.items()})
        if not pegangan:
            pegangan.extend(display(x, display_id=True) for x in isi)
        else:
            for h, x in zip(pegangan, isi):
                h.update(x)
        plt.close("all")
    for v in kontrol.values():
        v.observe(render, "value")
    render()

display(PANEL)
""")

# ============================================================== 3. fault
md("""
## 3. Fault disuntikkan ke data asli

Data lapangan dianggap **bersih**; fault dibuat dengan simulator lalu disuntikkan
ke satu sensor. Empat jenis: **drift** (merayap naik pelan), **spike** (lonjakan
sesaat), **bias** (bergeser tetap), **hardware** (macet/nilai hilang); "semua" =
keempatnya sekaligus.
""")

code(r"""
FIG_F, AX_F = plt.subplots(1, 2, figsize=(11, 2.8), gridspec_kw={"width_ratios": [3, 1]})
L_asli, = AX_F[0].plot([], [], color="tab:blue", lw=1, label="asli")
L_rusak, = AX_F[0].plot([], [], color="tab:red", lw=0.9, alpha=0.8, label="diberi fault")
L_beda, = AX_F[1].plot([], [], color="tab:red", lw=0.8)
AX_F[0].legend(fontsize=8)
AX_F[1].set_title("selisih (fault − asli)", fontsize=9); AX_F[1].set_xlabel("jam")
FIG_F.tight_layout(); plt.close(FIG_F)

def gambar_fault(**p):
    # satu figur dipakai ulang; hanya data garis yang diganti (respons cepat)
    j, s, n = jendela(**p), p["lihat"], p["n"]
    t = np.arange(n) * 300 / 3600
    bersih, rusak = j["bersih"][:, s], j["rusak"][:, s]
    L_asli.set_data(t, bersih); L_rusak.set_data(t, rusak); L_beda.set_data(t, rusak - bersih)
    AX_F[0].set_ylabel(f"S{s+1} (%)")
    AX_F[0].set_xlabel(f"jam sejak hari {j['mulai'] / PER_HARI:.1f} (N = {n} = {n * 300 / 3600:.0f} jam)")
    for ax in AX_F:
        ax.relim(); ax.autoscale_view()
    porsi = " · ".join(f"S{q+1} **{j['mask'][:, q].mean():.0%}**" for q in range(4))
    return [FIG_F, Markdown(f"Kondisi simulator: **{j['kondisi']}** · porsi sampel yang kena fault di "
                            f"jendela ini: {porsi}")]

hidupkan(gambar_fault)
""")

internal("fault", "simulator fault",
         """
- Fault **disimulasikan** karena data lapangan tidak punya label "rusak" yang
  pasti. Dengan simulasi kita tahu persis kapan dan di sensor mana fault terjadi,
  jadi jawaban benarnya diketahui.
- Rumus singkat (lihat `deteksi_fault_core.py` §3):
  drift = tambah laju × waktu, hanya di 3 episode 3 hari lalu di-reset;
  spike = lonjakan berkala setinggi 0,08 × SD; bias = + 0,08 poin tetap;
  hardware = ±3 % sampel diganti nilai dari waktu lain (macet) dan 5 % hilang
  (NaN, lalu diisi nilai sebelumnya). Parameter tertulis 8 %, tetapi sampel yang
  hilang menimpa sebagian sampel macet (kelemahan F4).
- Intensitas di panel bisa digeser ×0,1 sampai ×100 hanya untuk **melihat**; model
  dilatih dengan ×1. Uji intensitas yang sebenarnya ada di notebook 13.
- Fault dan entropi di panel dihitung langsung dengan `deteksi_fault_core` (fungsi
  yang sama dengan notebook 11 dan model tersimpan): fault disuntikkan ke seluruh
  deret lalu jendela dipotong, persis seperti data latih.
- Pertanyaan jebakan: "bias 0,08 poin itu kecil sekali?" — ya, dan itu sebabnya
  bias hampir tidak terdeteksi (lihat bagian 4 dan kelemahan B2).
""",
         "- Ledger: R13-* (uji intensitas), B-11 (laju drift).\n"
         "- `sidang/qa/02_kelemahan.md` B1–B3, F4–F5; notebook 13 dan 14.")

# ============================================================== 4. entropi
md("""
## 4. Entropi multiskala: EDM-Fuzzy vs JSD-Fuzzy

Tiap sensor diringkas menjadi **15 angka** (satu per skala τ = 1..15): seberapa
tidak teratur polanya ketika sinyal dilihat makin kasar. Empat sensor × 15 = **60
fitur** per jendela. Garis putus-putus = setelah fault.
""")

code(r"""
T_SK = core.T_SCALES; _tau = np.arange(1, T_SK + 1)
FIG_E, AX_E = plt.subplots(1, 2, figsize=(11, 2.8), sharex=True)
L_E = [(ax.plot(_tau, _tau * 0.0, "o-", ms=3, label="asli")[0],
        ax.plot(_tau, _tau * 0.0, "s--", ms=3, label="fault")[0]) for ax in AX_E]
for ax in AX_E:
    ax.set_xlabel("skala τ"); ax.legend(fontsize=8)
AX_E[0].set_ylabel("entropi"); FIG_E.tight_layout(); plt.close(FIG_E)

def gambar_entropi(**p):
    j, s = jendela(**p), p["lihat"]
    beda = []
    for mi, ax in enumerate(AX_E):
        asli, rusak = (f[s * T_SK:(s + 1) * T_SK] for f in j["F"][mi])
        L_E[mi][0].set_ydata(asli); L_E[mi][1].set_ydata(rusak)
        ax.set_title(f"{METODE[mi]} — S{s+1}, N = {p['n']}", fontsize=10)
        ax.relim(); ax.autoscale_view()
        beda.append(np.abs(rusak - asli).mean())
    return [FIG_E, Markdown(f"Rata-rata perubahan entropi S{s+1} akibat fault: " +
                            " · ".join(f"{METODE[mi]} **{beda[mi]:.3f}**" for mi in range(2)))]

hidupkan(gambar_entropi)
""")

internal("entropi", "fuzzy entropy, EDM, JSD",
         """
Langkah fuzzy entropy untuk satu sensor pada satu skala (contoh angka lengkap di
`sidang/qa/03_algoritma_dengan_tangan.md`):
1. **Coarse-graining komposit**: rata-rata tiap τ titik berurutan (skala τ),
   diulang untuk τ titik awal yang berbeda lalu hasil entropinya dirata-rata.
2. **Embedding**: potong jadi vektor pendek panjang m = 2 (dan m + 1 = 3), kurangi
   rata-rata tiap vektor (*centering*). Akibatnya pergeseran tetap (bias) hilang.
3. **Kemiripan**: bandingkan tiap vektor dengan 128 vektor acuan acak, lalu
   μ = 1 / (1 + d / r).
   - EDM: d = jarak Euclidean antar vektor, r = 0,2 × SD sinyal.
   - JSD: tiap vektor diubah jadi histogram lunak 8 bin (kernel Gauss di rentang
     ±3 SD), d = √JSD antar histogram (jarak Jensen–Shannon), r = 0,2 × √ln 2.
4. **Entropi** = ln(rata-rata kemiripan m ÷ rata-rata kemiripan m+1).
   Sinyal teratur → pola panjang tetap mirip → entropi kecil.

Kenapa JSD bisa lebih peka: ia membandingkan **bentuk sebaran** nilai di dalam
potongan, bukan jarak titik ke titik, sehingga lonjakan dan macet mengubah bentuk
histogram walau jaraknya kecil. Harganya: ekstraksi JSD lebih lambat.
""",
         "- Kode: `deteksi_fault_core.py` `edm_fuzzy_entropy_1d`, `jsd_fuzzy_entropy_1d`, `pairwise_jsd`.\n"
         "- Pemeriksa contoh tangan: `sidang/qa/cek_algoritma_tangan.py`.\n"
         "- Ledger H14-42*, H14-45c/d (biaya ekstraksi).")

# ============================================================== 5. model
md("""
## 5. ANN dilatih dengan Levenberg–Marquardt (LM)

60 fitur → jaringan saraf kecil (1–2 lapisan tersembunyi) → kelas. Bobot dicari
dengan LM: langkah **(JᵀJ + λ·diag(JᵀJ))·Δ = −Jᵀr**. λ kecil ≈ Gauss–Newton
(cepat dekat minimum), λ besar ≈ turun gradien (hati-hati jauh dari minimum).
Coba geser λ pada contoh mini di bawah (satu neuron, dua bobot).
""")

code(r"""
rng_toy = np.random.default_rng(0)
x_toy = np.linspace(0, 4, 30); y_toy = 2.0 * np.tanh(0.8 * x_toy) + rng_toy.normal(0, 0.05, 30)

def lm_mini(lam0, adaptif, iters=25):
    th = np.array([0.2, 3.0]); lam = lam0; hist = []
    for _ in range(iters):
        a, b = th; z = np.tanh(b * x_toy); r = a * z - y_toy
        J = np.column_stack([z, a * x_toy * (1 - z**2)])
        A = J.T @ J; g = J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)) + 1e-12 * np.eye(2), -g)
        th_baru = th + step
        r_baru = th_baru[0] * np.tanh(th_baru[1] * x_toy) - y_toy
        if (r_baru**2).sum() < (r**2).sum():
            th = th_baru; lam = lam / 3 if adaptif else lam
        else:
            lam = lam * 2 if adaptif else lam
        hist.append(((th[0] * np.tanh(th[1] * x_toy) - y_toy) ** 2).mean())
    return th, hist

def gambar_lm(log_lambda, adaptif):
    th, hist = lm_mini(10.0 ** log_lambda, adaptif)
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.6))
    axs[0].semilogy(hist, "o-", ms=3); axs[0].set_xlabel("iterasi"); axs[0].set_ylabel("MSE")
    axs[0].set_title(f"λ awal = {10.0 ** log_lambda:g}, {'adaptif' if adaptif else 'tetap'}", fontsize=10)
    axs[1].plot(x_toy, y_toy, "k.", label="data")
    axs[1].plot(x_toy, th[0] * np.tanh(th[1] * x_toy), "r-", label="hasil fit"); axs[1].legend(fontsize=8)
    plt.tight_layout()
    return [fig]

K_LM = dict(log_lambda=w.FloatSlider(value=-2, min=-4, max=3, step=0.5, description="log10 λ"),
            adaptif=w.Checkbox(value=True, description="λ adaptif (seperti penelitian)"))
hidupkan(gambar_lm, K_LM, w.HBox(list(K_LM.values())))
""")

internal("model", "ANN-LM dan cara evaluasi",
         """
- **Kenapa LM**: untuk jaringan kecil dan data ratusan sampel, LM memakai
  informasi lengkung (JᵀJ) sehingga butuh jauh lebih sedikit iterasi daripada
  Adam/gradien biasa. Notebook 09 membandingkan LM, Bayesian regularization, GA,
  dan Adam: LM ≈ BR, keduanya jauh di atas GA dan Adam.
- **Implementasi sendiri** dengan Jacobian analitik (turunan dihitung rumus, bukan
  beda-hingga), aturan λ Nielsen. Klasifikasi dilakukan dengan target one-hot +
  MSE, lalu softmax untuk peluang.
- **Evaluasi**: 5 lipatan *StratifiedGroupKFold*; grup = blok waktu, supaya jendela
  dari waktu yang sama tidak ada di data latih dan uji sekaligus. Arsitektur dipilih
  di dalam lipatan latih saja (nested CV).
- Contoh mini di atas: λ tetap kecil kadang melompat terlalu jauh, λ tetap besar
  lambat; λ adaptif menggabungkan keduanya.
""",
         "- Kode: `deteksi_fault_core.py` `lm_fit`, `lm_residual_jacobian`, `inner_search_best`.\n"
         "- Ledger R09-* (perbandingan pelatih), B-29 (grid 8 arsitektur).\n"
         "- `sidang/qa/04_walkthrough_kode.md`, `05_bank_pertanyaan.md` §1.")

# ============================================================== 6. prediksi
md("""
## 6. Tebakan model tersimpan untuk jendela pilihan

Model tersimpan (`models/`, satu set per panjang jendela N = 200/700/1000) dipakai
langsung: **S1** menjawab "normal atau rusak", model skenario sesuai jumlah jenis
fault menjawab jenis/kombinasinya, dan model sensor-ID menunjuk sensor yang rusak
(skor > 0,5).
""")

code(r"""
# Bobot model N = 200/700/1000 dari models/ (diekspor ke numpy oleh sidang/qa/buat_cache_interaktif.py,
# prediksinya sudah dicek sama dengan model asli; selisih < 1e-9). MZ dan META dimuat di panel kontrol.

def maju(kunci, X, sensor=False):
    X = np.array(X, float, copy=True)
    nan = np.isnan(X)
    if nan.any():
        X[nan] = np.broadcast_to(MZ[kunci + "|fill"], X.shape)[nan]
    a = (X - MZ[kunci + "|mean"]) / MZ[kunci + "|scale"]
    n, act = int(MZ[kunci + "|n"]), str(MZ[kunci + "|act"])
    for i in range(n):
        a = a @ MZ[f"{kunci}|W{i}"] + MZ[f"{kunci}|b{i}"]
        if i < n - 1:
            a = np.maximum(a, 0.0) if act == "relu" else np.tanh(a)
    if sensor:
        return a
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

MODELS = {}
for k in sorted({"|".join(x.split("|")[:3]) for x in MZ.files if not x.startswith("_")}):
    nn, sc, met = k.split("|")
    MODELS[(int(nn[1:]), sc, met)] = {"class_names": [str(c) for c in MZ[k + "|kelas"]],
                                      "detektor": lambda X, k=k: maju(k + "|det", X),
                                      "sensor_id": (lambda X, k=k: maju(k + "|sen", X, sensor=True))
                                                   if k + "|sen|n" in MZ.files else None,
                                      "cv_f1_nb11": float(MZ[k + "|cv_f1_nb11"])}
SKEN_KELAS = {core.SCEN_SHORT[sc]: cls for sc, cls in core.LADDER.items()}
def kelas_benar(sc, kondisi):
    return next(nama for nama, kond in SKEN_KELAS[sc] if kondisi in kond or (sc == "S1" and kondisi != "normal"
                                                                                and nama == "faulty"))
for n, m in META.items():
    print(f"N = {n}: {sum(k[0] == n for k in MODELS)} model | dibuat {m['dibuat']} | commit {m['commit']}")
""")

code(r"""
def tebak(**p):
    j, n = jendela(**p), p["n"]
    sken = ["S1", "S2", "S3", "S4", "S5"][len(j["fault"])] if j["sensor"] else "S1"
    baris, sens = [], []
    for mi, met in enumerate(METODE):
        X2 = np.vstack(j["F"][mi])
        for sc in dict.fromkeys(["S1", sken]):
            b = MODELS[(n, sc, met)]; P = b["detektor"](X2)
            for wi, nama in enumerate(["asli", "diberi fault"]):
                kk = int(P[wi].argmax())
                baris.append({"Jendela": nama, "Model": sc, "Metode": met, "Tebakan": b["class_names"][kk],
                              "Yakin": f"{P[wi, kk]:.0%}",
                              "Benar": kelas_benar(sc, "normal" if wi == 0 else j["kondisi"]),
                              "F1 CV (NB11)": round(b["cv_f1_nb11"], 3)})
        if j["sensor"] and MODELS[(n, sken, met)]["sensor_id"] is not None:
            skor = MODELS[(n, sken, met)]["sensor_id"](j["F"][mi][1][None])[0]
            sens.append({"Metode": met, **{f"S{q+1}": round(float(v), 2) for q, v in enumerate(skor)},
                         "Ditebak rusak": ", ".join(f"S{q+1}" for q, v in enumerate(skor) if v > 0.5) or "-",
                         "Benar": ", ".join(f"S{q+1}" for q in j["sensor"])})
    isi = [Markdown(f"**N = {n}, hari {j['mulai'] / PER_HARI:.1f}, kondisi `{j['kondisi']}`, intensitas "
                    f"×{p['k']:.2g}.** Model {sken} dipilih sesuai jumlah jenis fault (1 → S2, 2 → S3, "
                    f"3 → S4, 4 → S5). Intensitas latih = ×1."),
           pd.DataFrame(baris)]
    if sens:
        isi += [Markdown("**Sensor mana yang rusak** (skor > 0,5 = rusak):"), pd.DataFrame(sens)]
    else:
        isi += [Markdown("Tidak ada sensor yang dirusak: hanya model S1 (normal vs rusak) yang relevan."),
                pd.DataFrame()]
    return isi

hidupkan(tebak)
""")

internal("prediksi", "membaca tebakan model",
         """
- Satu contoh bisa benar bisa salah. Peluang dekat 50 % artinya model ragu. Yang
  adil adalah angka CV (bagian 7), bukan satu contoh.
- Model dilatih pada **semua** jendela skenario lalu disimpan; akurasi pada data
  latih bukan ukuran generalisasi. Angka resmi = F1 CV notebook 11.
- Model hanya mengenal intensitas ×1. Pada ×10/×50 tebakan bisa aneh karena
  polanya di luar data latih — itu jawaban jujur kalau ditanya.
- Demo cadangan dengan banyak jendela acak: `sidang/03_Demo_Model_Tersimpan.ipynb`
  Langkah 6.
""",
         "- `models/meta.json` (arsitektur, cv_f1_nb11), `sidang/PANDUAN_DEMO.md`.\n"
         "- Ledger P-04…P-06 (bukti run demo `sidang/bukti_demo/`).")

# ============================================================== 7. hasil
md("""
## 7. Hasil utama (notebook 11, LM asli, 15 sel = 5 skenario × 3 panjang jendela)

Nilai = F1 makro rata-rata 5 lipatan CV. Hijau = JSD lebih tinggi.
""")

code(r"""
f1 = pd.read_csv(ROOT / "hasil11" / "exports" / "11_rq5_f1_berpasangan.csv")
f1 = f1[pd.to_numeric(f1["N_window"], errors="coerce").notna()].copy()
f1["N"] = f1["N_window"].astype(float).astype(int)
tabel = f1.pivot(index="Scenario", columns="N", values="Difference")
display(Markdown("**Selisih F1 (JSD − EDM)** per sel:"))
display(tabel.style.format("{:+.3f}").background_gradient(cmap="RdYlGn", vmin=-0.1, vmax=0.1))
r = f1[["EDM-Fuzzy F1", "JSD-Fuzzy F1"]].astype(float)
print(f"JSD menang {int((f1['Difference'].astype(float) > 0).sum())} dari {len(f1)} sel | "
      f"rata-rata F1 EDM {r['EDM-Fuzzy F1'].mean():.3f} vs JSD {r['JSD-Fuzzy F1'].mean():.3f}")
sid = pd.read_csv(ROOT / "hasil11" / "exports" / "11_sensor_ringkas.csv")
display(Markdown("**Identifikasi sensor** (dirantai dari prediksi out-of-fold):")); display(sid)
""")

internal("hasil", "cara membaca hasil",
         """
- **F1 makro**: rata-rata F1 semua kelas, sehingga kelas kecil dihitung sama
  penting. Akurasi bisa menipu kalau satu kelas dominan.
- S1 (normal vs rusak) dan S5 (empat fault sekaligus) paling mudah; S2–S4 sulit
  karena kelasnya banyak dan beberapa fault (bias, drift kecil) hampir tak terlihat.
- Kalimat aman: "JSD-Fuzzy unggul **konsisten** (menang di sebagian besar sel),
  tetapi pada protokol utama (NB11) keunggulan gabungannya **belum signifikan**;
  dengan fault bentuk episode (NB14) keunggulannya signifikan."
""",
         "- Ledger R11-01…R11-18, H14-01… (NB14).\n"
         "- Notebook 11 §10–§12, notebook 14; `hasil11/Ringkasan_untuk_laporan.md`.")

# ============================================================== 8. statistik
md("""
## 8. Seberapa kuat buktinya?

Uji berpasangan per notebook, ditambah ukuran efek (Cohen dz), interval
kepercayaan bootstrap 95 % untuk selisih rata-rata, dan koreksi Holm untuk tiga
panjang jendela.
""")

code(r"""
st = pd.read_csv(ROOT / "sidang" / "qa" / "hasil_uji" / "statistik_sel.csv")
kol = ["notebook", "N", "menang_JSD", "n_sel", "rerata_selisih", "p_t", "p_wilcoxon",
       "p_t_holm_3N", "cohen_dz", "ci95_bawah", "ci95_atas"]
display(st[kol].style.format({c: "{:.3f}" for c in kol[4:]}, na_rep="—"))
p_lip = ROOT / "sidang" / "qa" / "hasil_uji" / "statistik_lipatan.csv"
if p_lip.exists():
    display(Markdown("**Tingkat lipatan, N = 700** (arsitektur tetap; t terkoreksi Nadeau–Bengio):"))
    display(pd.read_csv(p_lip).round(3))
""")

internal("statistik", "statistik dan cara membela angka p",
         """
- **Uji-t berpasangan**: tiap sel punya dua angka (EDM, JSD) dari data dan lipatan
  yang sama, jadi yang diuji adalah selisihnya.
- **Wilcoxon** tidak mengandaikan selisih berdistribusi normal. Dengan 5 pasangan,
  p terkecil yang mungkin 0,0625, jadi Wilcoxon per-N tidak akan pernah < 0,05.
- **Holm**: tiga uji (N = 200/700/1000) sekaligus memperbesar peluang "kebetulan
  signifikan"; Holm menaikkan p untuk mengoreksinya.
- **Cohen dz** = rata-rata selisih ÷ simpangan baku selisih: ukuran efek yang
  tidak tergantung jumlah sel. ±0,2 kecil, ±0,5 sedang, ±0,8 besar.
- **Bootstrap CI**: ambil ulang 15 sel secara acak 10.000 kali; kalau rentang
  tidak memuat 0, selisihnya konsisten positif. Dengan hanya 15 sel, CI ini
  cenderung sempit; laporkan berdampingan dengan p, bukan pengganti.
- **Nadeau–Bengio**: lipatan CV berbagi data latih, sehingga uji-t biasa terlalu
  optimistis; koreksinya memperbesar variansi dengan faktor (1/k + n_uji/n_latih).
""",
         "- `sidang/qa/statistik_tambahan.py` + `test_statistik_tambahan.py` (6 uji, bersumber pustaka).\n"
         "- `sidang/qa/02_kelemahan.md` C1–C6, F6; ledger R11-03…05.")

# ============================================================== 9. kelemahan
md("""
## 9. Kelemahan yang kami ketahui (dan apa yang sudah diukur)
""")

code(r"""
p_ef = ROOT / "sidang" / "qa" / "hasil_uji" / "efek_purge_permutasi.csv"
if p_ef.exists():
    ef = pd.read_csv(p_ef)
    display(Markdown("**Purge** = buang jendela latih yang bertumpang-tindih waktu dengan jendela uji. "
                     "**Permutasi** = label latih diacak; F1 harus turun ke tingkat tebakan."))
    display(ef.round(3))
else:
    print("Hasil uji purge/permutasi belum tersedia.")
""")

md("""
| Jenis | Kelemahan | Dampak |
|---|---|---|
| Internal | Jendela bersebelahan berbagi setengah sampel di batas grup CV | skor absolut sedikit optimistis; berlaku sama untuk EDM dan JSD |
| Konstruk | Fault disimulasikan, bukan kerusakan nyata | hasil berlaku untuk bentuk fault yang dimodelkan |
| Konstruk | *Centering* menghapus bias tetap | bias tidak terdeteksi oleh kedua metode |
| Statistik | 15 sel tidak saling bebas; banyak uji | p gabungan dibaca hati-hati, dilengkapi Holm/dz/CI |
| Eksternal | Satu lokasi, 4 sensor, ±98 hari | perlu validasi di lokasi/sensor lain |
""")

internal("kelemahan", "menjawab serangan validitas",
         """
Pola jawaban tiap kelemahan: **akui → dampak → mitigasi**, dua kalimat. Contoh
untuk kebocoran CV: "Pengelompokan per blok waktu mencegah jendela yang sama bocor,
tetapi jendela di batas blok masih berbagi setengah sampelnya, jadi skor
absolutnya sedikit optimistis. Karena berlaku sama untuk EDM dan JSD, selisihnya
tetap sah, dan kami mengukurnya: lihat tabel purge di atas."
""",
         "- `sidang/qa/02_kelemahan.md` (daftar lengkap A1–F7 dengan file:baris).\n"
         "- `sidang/qa/uji_purge_permutasi_N700.py` → `hasil_uji/purge_permutasi_N700.csv`.")

# ============================================================== 10. penutup
md("""
## 10. Kesimpulan

1. Entropi multiskala 4 sensor + ANN-LM dapat mendeteksi fault dan menunjuk sensor
   yang rusak, paling baik untuk hardware/macet dan kombinasi banyak fault.
2. Mengganti Euclidean dengan **Jensen–Shannon Divergence** memberi keunggulan yang
   konsisten; kekuatan statistiknya bergantung pada bentuk fault (lihat bagian 8).
3. Bias tetap dan drift kecil tetap sulit: keterbatasan ini berasal dari
   *centering* di dalam entropi, bukan dari classifier.
""")

internal("penutup", "kalimat penutup dan langkah lanjut",
         """
Tiga kalimat siap pakai ada di `Bahan_Sidang.md` bagian "Kalimat siap pakai".
Langkah lanjut yang masuk akal: validasi di sensor/lokasi lain, fitur tanpa
centering untuk bias, jendela lebih pendek untuk pemantauan dekat real-time,
dan CV dengan purge sebagai protokol baku.
""",
         "- `Bahan_Sidang.md` Q44; `sidang/qa/06_mock_sidang.md` (latihan).")

# ============================================================== lampiran internal
# Semua catatan persiapan ikut di salinan internal (hilang di salinan presentasi).
md("""
<div class="catatan-internal">

## Lampiran internal — catatan persiapan lengkap

Semua bahan persiapan sidang, satu blok per file, salinan apa adanya dari repo saat
notebook dibangun. Hanya ada di salinan internal.

</div>
""")
cells[-1].metadata["tags"] = [TAG]

internal("data-hipotetik", "percobaan data hipotetik tanpa entropi (NB15)",
         """
NB15 terpisah dari perbandingan EDM vs JSD: fitur lag per sampel, tanpa entropi,
untuk sistem notifikasi. Penguji bisa bertanya tentang urutan penyisipan gangguan,
jumlah neuron output, dan ruang grid search.
""",
         "- `build_15_data_hipotetik_ann_lm_notebook.py`, `build_14_lengkap_fault_baru_notebook.py` (`NEW_SIM`).\n"
         "- `hasil15/data-hipotetik-berlabel-ann-lm.log`.")

LAMPIRAN = [
    ("Bahan_Sidang.md", "Bahan sidang Q1–Q48 dan kalimat siap pakai"),
    ("sidang/qa/08_tanya_jawab_lengkap.md", "Tanya-jawab lengkap tiga lapis T1–T57"),
    ("sidang/qa/05_bank_pertanyaan.md", "Bank pertanyaan menyerang P1–P33"),
    ("sidang/qa/02_kelemahan.md", "Kelemahan dan ancaman validitas"),
    ("sidang/qa/03_algoritma_dengan_tangan.md", "Algoritma dengan tangan (LM, EDM, JSD)"),
    ("sidang/qa/04_walkthrough_kode.md", "Walkthrough kode"),
    ("sidang/qa/07_pustaka_dan_kebaruan.md", "Pustaka dan posisi kebaruan"),
    ("sidang/qa/01_ringkasan_klaim.md", "Ringkasan audit klaim"),
    ("sidang/qa/06_mock_sidang.md", "Skenario mock sidang"),
    ("sidang/qa/10_paket_mock_sidang.md", "Paket mock sidang dan lembar skor"),
    ("sidang/qa/11_kuis_pemahaman.md", "Kuis pemahaman"),
    ("sidang/qa/12_jurnal_eksperimen.md", "Jurnal eksperimen 2025 sampai versi terbaru"),
    ("sidang/qa/09_checklist_pembimbing.md", "Checklist tanda tangan pembimbing"),
    ("sidang/PANDUAN_DEMO.md", "Panduan demo"),
    ("sidang/MAPPING.md", "Peta klaim ke notebook"),
]
for _rel, _judul in LAMPIRAN:
    _isi = (ROOT / _rel).read_text(encoding="utf-8").replace("<!-- bagian:", "<!-- (bagian):")
    cells.append(nbf.v4.new_markdown_cell(
        f'<details class="{TAG}">\n<summary><b>Lampiran — {_judul}</b> (`{_rel}`)</summary>\n\n'
        f"{_isi.strip()}\n\n</details>"))
    cells[-1].metadata["tags"] = [TAG]

# setiap bagian di 08 harus muncul di salah satu catatan internal
_bagian_qa = set(re.findall(r"<!-- bagian: (\S+) -->", QA.read_text(encoding="utf-8"))) - {"nama"}
_dipakai = set(re.findall(r'internal\("([^"]+)"', Path(__file__).read_text(encoding="utf-8")))
assert _bagian_qa <= _dipakai, f"bagian 08 belum masuk notebook: {sorted(_bagian_qa - _dipakai)}"


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    n_int = sum(1 for c in cells if TAG in c.metadata.get("tags", []))
    print(f"Tersimpan {OUT.relative_to(ROOT)}: {len(cells)} sel, {n_int} catatan internal")


if __name__ == "__main__":
    main()
