#!/usr/bin/env python3
"""ekspor_notebook_sidang.py — buat dua salinan notebook sidang interaktif.

  sidang/04_Sidang_Interaktif.ipynb             salinan internal (sumber, berisi catatan)
  sidang/04_Sidang_Interaktif_presentasi.ipynb  salinan presentasi: semua sel bertanda
                                                `catatan-internal` dibuang

Pembuangan memakai nbconvert TagRemovePreprocessor (bukan regex), jadi yang hilang
persis sel yang ditandai. Skrip gagal kalau salinan presentasi masih memuat blok
catatan internal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import TagRemovePreprocessor

ROOT = Path(__file__).resolve().parents[2]
SUMBER = ROOT / "sidang" / "04_Sidang_Interaktif.ipynb"
PRESENTASI = ROOT / "sidang" / "04_Sidang_Interaktif_presentasi.ipynb"
TAG = "catatan-internal"


def sisa_internal(nb) -> int:
    return sum(1 for c in nb.cells
               if TAG in c.metadata.get("tags", []) or
               (c.cell_type == "markdown" and f'class="{TAG}"' in c.source))


def main() -> int:
    nb = nbformat.read(SUMBER, as_version=4)
    n_awal = sisa_internal(nb)
    pp = TagRemovePreprocessor(remove_cell_tags={TAG}, enabled=True)
    nb2, _ = pp.preprocess(nb, {})
    sisa = sisa_internal(nb2)
    nbformat.write(nb2, PRESENTASI)
    print(f"internal: {n_awal} sel catatan | presentasi: {len(nb2.cells)} sel, "
          f"sisa catatan internal {sisa} -> {PRESENTASI.relative_to(ROOT)}")
    return 1 if sisa or n_awal == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
