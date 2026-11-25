"""E2E notebook sidang interaktif: jalankan seperti di layar presentasi lalu gerakkan semua pilihan panel.

Jalankan: ./venv/bin/python -m pytest -q sidang/qa/test_notebook_sidang.py
"""
import json
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[2]
NB_INTERNAL = ROOT / "sidang" / "04_Sidang_Interaktif.ipynb"
NB_PRESENTASI = ROOT / "sidang" / "04_Sidang_Interaktif_presentasi.ipynb"
PY = sys.executable

# Dijalankan di kernel yang sama setelah semua sel: ubah tiap pilihan lewat widget
# (memicu callback seperti klik di layar) dan ukur waktu responsnya.
GERAK_PANEL = r"""
import time, json, itertools, traceback
err, lambat, n = [], [], 0
kombinasi_fault = [c for r in range(5) for c in itertools.combinations(FAULT4, r)]   # 16, termasuk kosong
sensor_pilihan = [(1,), (0, 3), (0, 1, 2), (0, 1, 2, 3), ()]
hari_pilihan = [0.0, 10.0, 39.5, 69.0, pil_hari.max]
for i, (nn, fault) in enumerate(itertools.product(META, kombinasi_fault)):
    p = dict(n=nn, hari=hari_pilihan[i % 5], sensor=sensor_pilihan[i % 5], fault=fault,
             k=[0.1, 1.0, 50.0][i % 3], lihat=i % 4)
    t0 = time.time()
    for fn in (gambar_fault, gambar_entropi, tebak):
        try:
            isi = fn(**p); plt.close("all")
        except Exception:
            err.append((fn.__name__, p, traceback.format_exc(limit=2)))
    dt = time.time() - t0; n += 1
    if dt > 4.0: lambat.append((nn, len(p["sensor"]), round(dt, 2)))
t0 = time.time()
pil_fault.value = ("spike", "bias"); pil_sensor.value = (0, 3); pil_int.value = 10.0
pil_hari.value = 40.0; pil_N.value = max(META); pil_lihat.value = 3
t_widget = time.time() - t0
print("HASIL_GERAK=" + json.dumps({"n": n, "n_N": len(META), "error": [str(e)[:400] for e in err[:3]],
                                   "n_error": len(err), "lambat": lambat[:5], "n_lambat": len(lambat),
                                   "t_widget_6_ubah": round(t_widget, 2)}))
"""


def _jalankan(nb_path, tambahan=None):
    nb = nbformat.read(nb_path, as_version=4)
    if tambahan:
        nb.cells.append(nbformat.v4.new_code_cell(tambahan))
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(nb_path.parent)}}).execute()
    return nb


# source: Kelvin, permintaan sidang 2026-11-20 — "notebook should be interactive fast using model"
def test_semua_pilihan_panel_jalan_dan_cepat():
    nb = _jalankan(NB_INTERNAL, GERAK_PANEL)
    teks = "".join(o.get("text", "") for o in nb.cells[-1].outputs if o.output_type == "stream")
    hasil = json.loads(teks.split("HASIL_GERAK=")[1].splitlines()[0])
    print(hasil)
    assert hasil["n_error"] == 0, hasil["error"]
    assert hasil["n"] == hasil["n_N"] * 16
    # satu perubahan panel = fault + entropi baru untuk semua sensor yang dirusak (N = 1000: ±0,4 s/sensor/metode)
    assert hasil["n_lambat"] == 0, hasil["lambat"]
    assert hasil["t_widget_6_ubah"] < 20.0


# source: Kelvin, chat 2026-11-21 — "make sure sidang-interaktif-deteksi-fault all adjustable for demo purpose";
# seed dropped from the panel: Kelvin chat 2026-11-25 — "hide seed from all surface so less question";
# default N = 1000: Kelvin chat 2026-11-21 pasted checklist "3. Model demo sidang (...), data 1000"
def test_panel_mencakup_semua_parameter_demo():
    nb = _jalankan(NB_INTERNAL, "print('BAWAAN_N=' + str(pil_N.value))\n"
                                "print('KONTROL=' + json.dumps(sorted(KONTROL)) + ' N=' + json.dumps(sorted(META)))")
    teks = "".join(o.get("text", "") for o in nb.cells[-1].outputs if o.output_type == "stream")
    assert json.loads(teks.split("KONTROL=")[1].split(" N=")[0]) == sorted(
        ["n", "hari", "sensor", "fault", "k", "lihat"])
    assert json.loads(teks.split(" N=")[1].splitlines()[0]) == [200, 700, 1000]
    assert teks.split("BAWAAN_N=")[1].splitlines()[0] == "1000"


# source: deteksi_fault_core.segment_and_label + simpan_model.py — the panel computes features live from
# sidang/assets/cache_interaktif.npz X5; normal windows must reproduce the stored training features exactly.
def test_fitur_langsung_sama_dengan_fitur_latih():
    sys.path.insert(0, str(ROOT))
    import deteksi_fault_core as core
    X5 = np.load(ROOT / "sidang" / "assets" / "cache_interaktif.npz")["X5"]
    for f in sorted((ROOT / "models").glob("fitur_N*.npz")):
        N = int(f.stem.split("_N")[1])
        d = np.load(f)
        idx = np.where(d["y_cond"] == 0)[0][:3]
        for i in idx:
            Wi = X5[d["starts"][i]:d["starts"][i] + N].astype(np.float32)
            for F, met in [(d["F_edm"], "EDM-Fuzzy"), (d["F_jsd"], "JSD-Fuzzy")]:
                # Training features were computed on Kaggle (x86_64), this check runs on the laptop (arm64):
                # float rounding moves a few JSD histogram bin edges (max seen 8.6e-4). A wrong seed or a
                # window shifted by one day differs by 0.019-0.039, so 2e-3 still catches real mismatches.
                assert np.abs(core._one_window(Wi, met, 7 + 1000 * i) - F[i]).max() < 2e-3, (N, i, met)


# source: Kelvin, permintaan sidang 2026-11-20 — "when presentation hide detailed note cheatsheet but for internal show notes"
def test_salinan_presentasi_tanpa_catatan_internal():
    r = subprocess.run([PY, str(ROOT / "sidang" / "qa" / "ekspor_notebook_sidang.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    internal = nbformat.read(NB_INTERNAL, as_version=4)
    pres = nbformat.read(NB_PRESENTASI, as_version=4)
    n_int = sum("catatan-internal" in c.metadata.get("tags", []) for c in internal.cells)
    assert n_int >= 10
    assert all('class="catatan-internal"' not in c.source for c in pres.cells)
    assert len(pres.cells) == len(internal.cells) - n_int
    nb = _jalankan(NB_PRESENTASI)
    errs = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errs


# source: sidang/qa/buat_cache_interaktif.py ekspor_model — notebook forward must equal saved SavedLM (diff < 1e-9);
# re-checked here on the stored training features models/fitur_N700.npz, independent of the cache.
def test_bobot_numpy_sama_dengan_model_tersimpan():
    sys.path.insert(0, str(ROOT / "models"))
    import joblib
    sys.path.insert(0, str(ROOT / "sidang" / "qa"))
    from buat_cache_interaktif import daftar_meta, maju
    import deteksi_fault_core as core
    semua = daftar_meta()
    assert 700 in semua
    for N, meta in semua.items():
        d = np.load(ROOT / "models" / f"fitur_N{N}.npz")
        for row in meta["models"]:
            b = joblib.load(ROOT / "models" / row["file"])
            X = d["F_edm"] if b["method"] == "EDM-Fuzzy" else d["F_jsd"]
            m = b["detektor"]
            lay = core.lm_unpack(m.theta, m._template())
            p = {"fill": m.fill, "mean": m.mean, "scale": m.scale, "act": m.activation,
                 "W": [W for W, _ in lay], "b": [bb for _, bb in lay]}
            assert np.abs(maju(p, X, "kelas") - m.predict_proba(X)).max() < 1e-9


# source: Kelvin, chat 2026-11-25 — "hide seed from all surface so less question"
def test_seed_tidak_terlihat_di_salinan_presentasi():
    subprocess.run([PY, str(ROOT / "sidang" / "qa" / "ekspor_notebook_sidang.py")], check=True, capture_output=True)
    pres = nbformat.read(NB_PRESENTASI, as_version=4)
    markdown = " ".join(c.source for c in pres.cells if c.cell_type == "markdown").lower()
    assert "seed" not in markdown
    kode = "\n".join(c.source for c in pres.cells if c.cell_type == "code")
    assert 'description="Seed"' not in kode and "pil_seed" not in kode
    # the one remaining fixed value lives inside the fault simulator call, not in the panel
    assert kode.lower().count("seed") == 1
