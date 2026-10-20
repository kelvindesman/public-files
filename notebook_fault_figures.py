"""Potongan sel bersama: tabel nasib 10 langkah + grafik asli vs fault.

Dipakai oleh tiga builder notebook supaya isinya persis sama:
  * build_cv_5skenario_notebook.py        (JSD Fuzzy CV 5skenario)
  * build_11_annlm_real_notebook.py       (JSD EDM ANN-LM Lengkap Q3)
  * build_ilustrasi_data_fault_notebook.py (Ilustrasi Data Akuisisi dan Fault)

Modul ini hanya menyimpan teks; notebook yang dihasilkan tetap mandiri.
"""

QC_STEPS_TITLE = "Nasib tiap langkah di dataset ini"


def qc_steps_md(heading, next_sec):
    return """## %s — %s

Pertanyaan pembimbing: *"missing value atau data di luar range diapain dulu di
tahap ini?"* — jawabannya ada di kolom **Tindakan**. Tidak ada baris yang
dibuang karena nilainya mencurigakan; yang dibuang hanya cacat struktural
(stempel waktu ganda), dan di dataset ini jumlahnya nol.

| # | Langkah | Hasil di data ini | Tindakan |
|---|---|---|---|
| 1 | Kolom wajib `kelembaban1..4` | 4/4 ada | lanjut, kalau tidak ada program berhenti |
| 2 | `datetime` UTC lalu diurutkan | 281.721 baris terurut | — |
| 3 | Stempel waktu ganda dibuang | **0** | tidak ada baris yang dibuang |
| 4 | Cek nilai kosong (NaN) | **0** per kolom | tidak ada yang perlu ditambal |
| 5 | Cek nilai di luar 0–100 %% | **0** | **dilaporkan, tidak dibuang** — bisa jadi gejala fault |
| 6 | Pembacaan macet ≥ 10 berturut | S1 556 · S2 654 · S3 659 · S4 657 kejadian | **dilaporkan, tidak dibuang** |
| 7 | Pencilan ekstrem \\|z\\| > 6 | S1 13 · S2 0 · S3 0 · S4 0 titik | **dilaporkan, tidak dibuang** |
| 8 | Rata-rata per 5 menit | 281.721 → **28.173** baris | agregasi, bukan pembuangan |
| 9 | Grid seragam + interpolasi waktu | 0 slot bolong → **0** isian | tidak terpakai di dataset ini |
| 10 | Sisa NaN ditambal median | 0 → **0** isian | tidak terpakai di dataset ini |

**Jadi secara eksplisit:** missing value, *kalau ada*, ditambal di langkah 9
(interpolasi waktu) lalu langkah 10 (median kolom). Nilai di luar 0–100 %% tidak
dibuang, hanya dicatat, karena pada penelitian ini pembacaan mustahil adalah
bukti fault, bukan sampah.

**Setelah injeksi fault** muncul NaN baru dari *hardware malfunction*
(`loss_prob = 5 %%`). NaN itu bukan kotoran data akuisisi: di §%s nilainya
diisi `ffill/bfill` lalu median supaya fitur bisa dihitung, sementara `mask`
fault tetap disimpan agar label per-sensor tidak berbohong.
""" % (heading, QC_STEPS_TITLE, next_sec)


FIGURES_TITLE = "Grafik sinyal asli vs setelah injeksi fault"


def figures_md(heading, next_sec):
    return """## %s — %s

Permintaan pembimbing: tampilkan **grafik data asli dan data setelah disisipi
fault untuk semua kelas**, cukup sampel. Dua bentuk:

1. **Lima panel vertikal** per sensor — meniru contoh pembimbing: Spike, Drift,
   drift + spike, drift + spike + bias, dan keempat gangguan sekaligus.
2. **Grid 16 panel** per sensor — Normal + 15 kombinasi fault, jadi seluruh
   kelas tercakup.

Garis **abu-abu** = sinyal asli (setelah pembersihan, sebelum injeksi).
Garis **merah** = setelah injeksi; pada *hardware malfunction* NaN dibiarkan
sebagai **celah**. Garis **oranye putus-putus** = setelah NaN ditambal
`ffill/bfill` + median, yaitu yang benar-benar dilihat model. Potongan yang
diplot 300 sampel = 1,04 hari pada laju 5 menit; seed tetap supaya gambar
reproducible. Nilai yang diplot juga diekspor ke CSV.

**Catatan skala.** Drift menambah 0,2 %% per sampel (≈ +60 %% dalam 300 sampel),
jadi paling kasatmata. Sebaliknya spike hanya ±0,08 × simpangan baku dan bias
hanya +0,08 %%, lebih kecil daripada ayunan alami sinyal; karena itu panel
spike/bias diberi kotak **selisih vs asli** (gangguan − asli) supaya besar
gangguan tetap terbaca. *Hardware malfunction* memakai 8 %% sampel macet dan
5 %% sampel hilang (NaN); rentang NaN diberi bayangan oranye.

Grafik ini hanya ilustrasi; ia tidak mengubah data yang dipakai §%s.
""" % (heading, FIGURES_TITLE, next_sec)


def figures_code(prefix, seed_base=0):
    return r'''# === Grafik: sinyal asli vs setelah injeksi fault (contoh untuk laporan) ===
FIG_START = 2000          # potongan 300 sampel = 1,04 hari pada laju 5 menit
FIG_LEN = 300
CASES_5 = [
    ("Spike", "spike"),
    ("Drift", "drift"),
    ("Kombinasi dua gangguan: drift + spike", "drift+spike"),
    ("Kombinasi tiga gangguan: drift + spike + bias", "drift+spike+bias"),
    ("Kombinasi empat gangguan: drift + spike + bias + hardware malfunction",
     "spike+bias+malfunc+drift"),
]
CASES_GRID = [
    ("Normal (tanpa gangguan)", None),
    ("spike", "spike"), ("drift", "drift"), ("bias", "bias"), ("hardware", "hardware"),
    ("drift + spike", "drift+spike"), ("drift + bias", "drift+bias"),
    ("drift + hardware", "drift+malfunc"), ("spike + bias", "spike+bias"),
    ("spike + hardware", "spike+malfunc"), ("bias + hardware", "bias+malfunc"),
    ("drift + spike + bias", "drift+spike+bias"),
    ("drift + spike + hardware", "spike+drift+malfunc"),
    ("drift + bias + hardware", "drift+bias+malfunc"),
    ("spike + bias + hardware", "spike+bias+malfunc"),
    ("drift + spike + bias + hardware", "spike+bias+malfunc+drift"),
]

def _inject_plot(x, key, seed):
    if key is None:
        return x.copy(), x.copy()
    y, _ = simulate_multiple_faults(x, SCENARIOS[key], seed=seed)
    yf = pd.DataFrame(y).ffill().bfill()
    yf = yf.fillna(yf.median(numeric_only=True)).to_numpy().ravel()
    return np.asarray(y, dtype=float), np.asarray(yf, dtype=float)

def _panel(ax, x, y_asli, y_inj, y_isi, judul, inset=False):
    ax.plot(x, y_asli, color="#8a8a8a", lw=2.0, alpha=0.75, label="Sinyal asli")
    ax.plot(x, y_inj, color="#c0392b", lw=1.0, label="Setelah disisipi gangguan")
    if np.isnan(y_inj).any():
        ax.plot(x, y_isi, color="#e67e22", lw=1.0, ls="--",
                label="Setelah NaN ditambal (ffill + median)")
        nan_idx = np.where(np.isnan(y_inj))[0]
        runs = np.split(nan_idx, np.where(np.diff(nan_idx) != 1)[0] + 1)
        for j, run in enumerate(runs):
            if len(run) == 0:
                continue
            ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color="#e67e22", alpha=0.18,
                       lw=0, label="Rentang NaN (data hilang)" if j == 0 else None)
    dif = y_inj - y_asli
    if inset and not np.isnan(dif).any() and np.nanmax(np.abs(dif)) < 1.0 and np.nanmax(np.abs(dif)) > 0:
        ki = int(np.nanargmax(np.abs(dif)))
        lo, hi = max(0, ki - 25), min(len(x), ki + 26)
        axi = ax.inset_axes([0.60, 0.52, 0.37, 0.43])
        axi.plot(x[lo:hi], dif[lo:hi], color="#c0392b", lw=1.2)
        axi.axhline(0, color="#8a8a8a", lw=0.8, ls=":")
        axi.tick_params(labelsize=5)
        axi.set_title("selisih vs asli (maks %.3g%%)" % np.nanmax(np.abs(dif)),
                      fontsize=6)
        axi.grid(alpha=0.2)
    ax.set_title(judul, fontsize=9)
    ax.margins(y=0.08)
    ax.grid(alpha=0.25)

def _catat(rows, sensor, kasus, x, asli, inj, isi):
    for xi, a, b, d in zip(x, asli, inj, isi):
        rows.append({"Sensor": sensor, "Kasus": kasus, "Indeks": int(xi),
                     "Asli": round(float(a), 4),
                     "Injeksi": (np.nan if np.isnan(b) else round(float(b), 4)),
                     "Terisi": round(float(d), 4)})

x_plot = np.arange(FIG_LEN)
plot_rows = []
for si, (c, sname) in enumerate(zip(cols, SENSORS)):
    seg = df_sync[c].iloc[FIG_START:FIG_START + FIG_LEN].to_numpy(dtype=float)
    if len(seg) < FIG_LEN:
        raise ValueError("Potongan data untuk gambar kurang dari 300 sampel.")

    # --- (1) lima panel vertikal, gaya contoh pembimbing ---
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 12.0), sharex=True)
    for ci, (ax, (judul, key)) in enumerate(zip(axes, CASES_5)):
        y_inj, y_isi = _inject_plot(seg, key, seed=__SEED__ + si * 100 + ci)
        _panel(ax, x_plot, seg, y_inj, y_isi, judul, inset=True)
        ax.set_ylabel("Kelembaban (%)", fontsize=8)
        _catat(plot_rows, sname, key, x_plot, seg, y_inj, y_isi)
    axes[-1].set_xlabel("Indeks sampel (5 menit)")
    h, l = [], []
    for _ax in axes:
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in l:
                h.append(_h); l.append(_l)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("Contoh sinyal asli vs setelah injeksi fault — " + sname,
                 y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("exports/__PREFIX__contoh_fault_vs_asli_" + sname + ".png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("[Tersimpan] exports/__PREFIX__contoh_fault_vs_asli_" + sname + ".png")

    # --- (2) grid 16 panel: normal + 15 kombinasi ---
    fig, axes = plt.subplots(4, 4, figsize=(14, 11), sharex=True)
    for ci, (ax, (judul, key)) in enumerate(zip(axes.ravel(), CASES_GRID)):
        y_inj, y_isi = _inject_plot(seg, key, seed=__SEED__ + 5000 + si * 100 + ci)
        _panel(ax, x_plot, seg, y_inj, y_isi, judul)
        ax.tick_params(labelsize=7)
        _catat(plot_rows, sname, key or "normal", x_plot, seg, y_inj, y_isi)
    for ax in axes[-1]:
        ax.set_xlabel("Indeks sampel (5 menit)", fontsize=8)
    h, l = [], []
    for _ax in axes.ravel():
        for _h, _l in zip(*_ax.get_legend_handles_labels()):
            if _l not in l:
                h.append(_h); l.append(_l)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("Semua kelas: sinyal asli vs setelah injeksi fault — " + sname,
                 y=0.998, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("exports/__PREFIX__grid_fault_vs_asli_" + sname + ".png",
                dpi=120, bbox_inches="tight")
    plt.show()
    print("[Tersimpan] exports/__PREFIX__grid_fault_vs_asli_" + sname + ".png")

plot_df = pd.DataFrame(plot_rows)
export_df(plot_df, "__PREFIX__contoh_fault_vs_asli")
print("Baris data plot:", len(plot_df), "| sensor:", sorted(plot_df.Sensor.unique()))
'''.replace("__PREFIX__", prefix).replace("__SEED__", str(seed_base))
