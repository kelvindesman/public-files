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
for hi, s, fi, ki in itertools.product(range(len(C["hari"])), range(4), range(len(C["fault"])), range(len(C["intensitas"]))):
    for fn in (gambar_fault, gambar_entropi, tebak):
        t0 = time.time()
        try:
            isi = fn(hi=hi, s=s, fi=fi, ki=ki); plt.close("all")
        except Exception:
            err.append((fn.__name__, hi, s, fi, ki, traceback.format_exc(limit=1)))
        dt = time.time() - t0; n += 1
        if dt > 1.0: lambat.append((fn.__name__, round(dt, 2)))
t0 = time.time(); pil_fault.value = 0; pil_sensor.value = 3; pil_int.value = 2; pil_hari.value = 1
t_widget = time.time() - t0
print("HASIL_GERAK=" + json.dumps({"n": n, "error": err[:3], "n_error": len(err), "lambat": lambat[:5],
                                   "n_lambat": len(lambat), "t_widget_4_ubah": round(t_widget, 2)}))
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
    assert hasil["n"] == 3 * 4 * 5 * 3 * 3
    assert hasil["n_lambat"] == 0, hasil["lambat"]
    assert hasil["t_widget_4_ubah"] < 4.0


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
    from buat_cache_interaktif import maju
    import deteksi_fault_core as core
    d = np.load(ROOT / "models" / "fitur_N700.npz")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    for row in meta["models"]:
        b = joblib.load(ROOT / "models" / row["file"])
        X = d["F_edm"] if b["method"] == "EDM-Fuzzy" else d["F_jsd"]
        m = b["detektor"]
        lay = core.lm_unpack(m.theta, m._template())
        p = {"fill": m.fill, "mean": m.mean, "scale": m.scale, "act": m.activation,
             "W": [W for W, _ in lay], "b": [bb for _, bb in lay]}
        assert np.abs(maju(p, X, "kelas") - m.predict_proba(X)).max() < 1e-9
