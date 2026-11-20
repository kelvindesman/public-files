#!/usr/bin/env python3
"""ekspor_catatan_artifact.py — kumpulkan semua catatan internal notebook sidang ke satu halaman.

Sumber: sel bertanda `catatan-internal` di sidang/04_Sidang_Interaktif.ipynb (catatan per
bagian + lampiran semua file persiapan). Keluaran: folder dengan `index.html` dan
`catatan.json` (array {id, grup, judul, md}), siap dipublikasikan sebagai Claude Artifact.

Jalankan: ./venv/bin/python sidang/qa/ekspor_catatan_artifact.py <folder_keluar>
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "sidang" / "04_Sidang_Interaktif.ipynb"
TAG = "catatan-internal"
HTML = Path(__file__).with_name("catatan_artifact_template.html")


def bongkar(src: str) -> tuple[str, str, str] | None:
    """Ambil (grup, judul, isi markdown) dari satu sel catatan; None untuk sel judul lampiran."""
    m = re.match(r'<details class="catatan-internal">\s*<summary>(.*?)</summary>\s*(.*)</details>\s*$', src, re.S)
    if not m:
        return None
    ringkas, isi = m.group(1), m.group(2).strip()
    judul = re.sub(r"<[^>]+>", "", ringkas)
    judul = re.sub(r"\s*\(klik untuk buka[^)]*\)", "", judul).strip()
    if judul.startswith("Catatan internal — "):
        return "Catatan per bagian notebook", judul.removeprefix("Catatan internal — "), isi
    judul = judul.removeprefix("Lampiran — ")
    judul = re.sub(r"\s*\(`[^`]+`\)$", lambda x: "", judul)
    sumber = re.search(r"\(`([^`]+)`\)", ringkas)
    if sumber:
        isi = f"*Sumber: `{sumber.group(1)}`*\n\n{isi}"
    return "Lampiran: file persiapan", judul, isi


def main() -> int:
    keluar = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sidang" / "artifact_catatan"
    keluar.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(NB, as_version=4)
    docs = []
    for c in nb.cells:
        if TAG not in c.metadata.get("tags", []):
            continue
        b = bongkar(c.source)
        if b is None:
            continue
        grup, judul, isi = b
        docs.append({"id": f"c{len(docs) + 1:02d}", "grup": grup, "judul": judul, "md": isi})
    n_sel = sum(TAG in c.metadata.get("tags", []) for c in nb.cells)
    assert len(docs) == n_sel - 1, f"{len(docs)} dokumen dari {n_sel} sel catatan (1 sel judul)"
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cs"],
                            capture_output=True, text=True).stdout.strip()
    (keluar / "catatan.json").write_text(json.dumps({"commit": commit, "docs": docs}, ensure_ascii=False),
                                         encoding="utf-8")
    (keluar / "index.html").write_text(HTML.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(docs)} dokumen -> {keluar} (commit {commit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
