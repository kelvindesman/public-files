#!/usr/bin/env python3
"""jalankan_notebook.py — jalankan notebook tanpa layar dan simpan buktinya.

Dipakai sebagai gerbang (gate) notebook sidang: notebook dijalankan dari awal
sampai akhir dengan nbclient, waktu tiap sel dicatat, dan salinan yang sudah
terisi output disimpan supaya angka yang dikutip di dokumen punya bukti lokal.

Pemakaian:
    ./venv/bin/python sidang/qa/jalankan_notebook.py NOTEBOOK.ipynb \
        --keluar sidang/bukti_demo [--batas-sel 2.0] [--batas-total 60]

Hasil di folder --keluar:
    <nama>.executed.ipynb   salinan berisi output
    <nama>.waktu.csv        indeks sel, jenis, detik, 80 karakter pertama sumber
Kode keluar 1 kalau ada sel error, atau kalau batas waktu diberikan dan dilampaui.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("notebook")
    ap.add_argument("--keluar", required=True)
    ap.add_argument("--batas-sel", type=float, default=None, help="detik maksimum per sel kode")
    ap.add_argument("--batas-total", type=float, default=None, help="detik maksimum seluruh notebook")
    ap.add_argument("--kernel", default="python3")
    a = ap.parse_args(argv)

    src = Path(a.notebook).resolve()
    out = Path(a.keluar).resolve()
    out.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(src, as_version=4)

    client = NotebookClient(nb, timeout=1800, kernel_name=a.kernel,
                            resources={"metadata": {"path": str(src.parent)}})
    waktu = []
    error = None
    t_all = time.time()
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            t0 = time.time()
            try:
                client.execute_cell(cell, i)
            except CellExecutionError as e:
                error = (i, str(e).splitlines()[-1] if str(e) else "error")
                waktu.append((i, time.time() - t0, cell.source[:80].replace("\n", " ")))
                break
            waktu.append((i, time.time() - t0, cell.source[:80].replace("\n", " ")))
    total = time.time() - t_all

    stem = src.stem
    nbformat.write(nb, out / f"{stem}.executed.ipynb")
    with open(out / f"{stem}.waktu.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sel", "detik", "sumber_awal"])
        for i, s, t in waktu:
            w.writerow([i, f"{s:.2f}", t])
        w.writerow(["TOTAL", f"{total:.2f}", ""])

    lambat = [(i, s) for i, s, _ in waktu if a.batas_sel is not None and s > a.batas_sel]
    print(f"{src.name}: {len(waktu)} sel kode, total {total:.1f} s, "
          f"sel terlama {max((s for _, s, _ in waktu), default=0):.1f} s")
    gagal = False
    if error:
        print(f"ERROR di sel {error[0]}: {error[1]}"); gagal = True
    if lambat:
        print("Melewati batas per sel: " + ", ".join(f"sel {i} ({s:.1f} s)" for i, s in lambat)); gagal = True
    if a.batas_total is not None and total > a.batas_total:
        print(f"Melewati batas total {a.batas_total} s"); gagal = True
    print("HASIL: " + ("GAGAL" if gagal else "LULUS"))
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
