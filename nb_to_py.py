#!/usr/bin/env python3
"""Ubah notebook jadi skrip .py (hanya sel kode) untuk smoke-test lokal.

Pemakaian: python nb_to_py.py <notebook.ipynb> <keluaran.py>
"""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
out = ['import matplotlib; matplotlib.use("Agg")\n',
       'def display(*args): pass\n',
       'def FileLink(*args, **kw): return None\n']
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    body = "".join(c["source"])
    body = body.replace("from IPython.display import FileLink, display", "pass")
    out.append(body.rstrip() + "\n\n\n")
open(dst, "w").write("".join(out))
print("wrote", dst)
