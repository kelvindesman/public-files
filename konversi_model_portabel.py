"""Ubah model hasil Kaggle v1 (pipeline sklearn di-pickle) ke format portabel SavedLM.

Model v1 dari kernel `simpan-model-final-deteksi-fault` disimpan dengan
scikit-learn 1.6.1 sehingga tidak bisa dibuka di versi lain. Skrip ini dijalankan
dengan scikit-learn 1.6.1 yang sama, membuka tiap bundle, mengganti pipeline
`detektor` dan `sensor_id` dengan `SavedLM` (angka identik), lalu menulis ulang.
Versi `simpan_model.py` sekarang langsung menyimpan SavedLM, jadi skrip ini hanya
dibutuhkan untuk model v1.

Pakai:
    uv run --no-project --with scikit-learn==1.6.1 --with numpy --with pandas \
        --with joblib python konversi_model_portabel.py models
"""
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deteksi_fault_core as core  # noqa: E402

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models")
Z = np.load(model_dir / "fitur_N700.npz")
print("scikit-learn", sklearn.__version__)
for f in sorted(model_dir.glob("model_N*.joblib")):
    b = joblib.load(f)
    det, sens = b["detektor"], b["sensor_id"]
    if isinstance(det, core.SavedLM):
        print(f.name, "sudah portabel")
        continue
    F = Z["F_jsd" if b["method"].startswith("JSD") else "F_edm"]
    new_det = core.SavedLM.from_pipeline(det, "kelas")
    d = float(np.abs(det.predict_proba(F) - new_det.predict_proba(F)).max())
    assert d < 1e-9, (f.name, d)
    b["detektor"] = new_det
    if sens is not None:
        new_sens = core.SavedLM.from_pipeline(sens, "sensor")
        ref = sens.named_steps["lm"].decision_function(sens[:-1].transform(F))
        ds = float(np.abs(ref - new_sens.decision_function(F)).max())
        assert ds < 1e-9, (f.name, ds)
        b["sensor_id"] = new_sens
    else:
        ds = 0.0
    joblib.dump(b, f, compress=3)
    print(f"{f.name}: dikonversi, selisih maks proba={d:.1e} sensor={ds:.1e}")
shutil.copy(HERE / "deteksi_fault_core.py", model_dir / "deteksi_fault_core.py")
print("deteksi_fault_core.py di", model_dir, "diperbarui")
