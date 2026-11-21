"""The exported defence notes must link every notebook reference to its Kaggle kernel."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# source: Kelvin chat 2026-11-22 — "apakah link kaggle juga terhubung di catatan claude sehingga dr catatan
# bisa membuktkan ke kodingan dengan cepat"
def test_semua_rujukan_notebook_di_catatan_tertaut_ke_kaggle(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "sidang/qa/ekspor_catatan_artifact.py"), str(tmp_path)],
                   check=True, capture_output=True)
    docs = json.loads((tmp_path / "catatan.json").read_text())["docs"]
    ids = {json.loads(f.read_text())["id"] for f in (ROOT / ".kaggle_push").glob("*/kernel-metadata.json")}
    sisa, n_link = [], 0
    for d in docs:
        t = d["md"]
        for kid in re.findall(r"\]\(https://www\.kaggle\.com/code/([\w-]+/[\w-]+)", t):
            n_link += 1
            assert kid in ids, kid
        t = re.sub(r"```.*?```", "", t, flags=re.S)
        t = re.sub(r"\[[^\]\n]*\]\([^)\s]*\)", "", t)
        t = re.sub(r"`[^`\n]*`", "", t)
        sisa += [(d["id"], r) for r in re.findall(r"\bNB ?\d\d[b]?\b|\b[Nn]otebook \d\d?\b", t)]
    assert not sisa, sisa[:10]
    assert n_link > 100
    peta = next(d for d in docs if d["id"] == "kaggle")["md"]
    assert all(f"https://www.kaggle.com/code/{i})" in peta for i in ids)
