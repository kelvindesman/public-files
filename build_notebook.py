#!/usr/bin/env python3
"""Assemble 01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb from the two reference notebooks.

- Pipeline + rich JSD-Fuzzy + ANN: from archive/SUMBER_01b_Dev_JSD_Rich.ipynb (cells 0..66)
- RQ1-RQ5 + Paper Summary: from archive/SUMBER_01a_Paper_Q3_RQ1_RQ5.ipynb (cells 68..88)
- Export + Notes: from archive/SUMBER_01b_Dev_JSD_Rich.ipynb (cells 67..68)

Run: python3 build_notebook.py   (needs the classifier to allow python execution)
"""
import json, copy, os, sys

ROOT = '/Users/kelvin/apps/public-files/'
sys.path.insert(0, ROOT)  # patch_runtime lives next to this script
DEV = ROOT + 'archive/SUMBER_01b_Dev_JSD_Rich.ipynb'
PAPER = ROOT + 'archive/SUMBER_01a_Paper_Q3_RQ1_RQ5.ipynb'
OUT = ROOT + '01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb'

dev = json.load(open(DEV))
paper = json.load(open(PAPER))

def firstline(c):
    return ''.join(c['source']).split('\n')[0]

# Locate dev export cell (cells after it = export + notes)
dev_export_idx = None
for i, c in enumerate(dev['cells']):
    if firstline(c).startswith('# Export all key tables'):
        dev_export_idx = i
        break
assert dev_export_idx is not None, 'dev export cell not found'

# Locate paper RQ block start + export cell
paper_rq_start = None
paper_export_idx = None
for i, c in enumerate(paper['cells']):
    fl = firstline(c)
    if paper_rq_start is None and fl.startswith('# Lima Skenario Klasifikasi Paper'):
        paper_rq_start = i
    if fl.startswith('# Export all key tables'):
        paper_export_idx = i
assert paper_rq_start is not None, 'paper RQ start not found'
assert paper_export_idx is not None, 'paper export cell not found'

# RQ block = paper[paper_rq_start : paper_export_idx]  (excludes paper export/notes)
rq_block = paper['cells'][paper_rq_start:paper_export_idx]

# Assemble: dev pipeline (0..export-1) + RQ block + dev export + dev notes
new_cells = dev['cells'][:dev_export_idx] + rq_block + dev['cells'][dev_export_idx:]

# Patch title (cell 0) to reflect the merged paper notebook
new_cells[0] = copy.deepcopy(new_cells[0])
new_cells[0]['source'] = [
    "# JSD-Fuzzy + ANN — Paper Notebook (Merged)\n",
    "\n",
    "Notebook final gabungan untuk paper: deteksi fault 4 sensor kelembaban (17 skenario) dengan\n",
    "multiscale entropy + ANN (MLPClassifier, GridSearch).\n",
    "\n",
    "**Sumber merge:**\n",
    "- `archive/SUMBER_01b_Dev_JSD_Rich.ipynb` → pipeline lengkap + JSD-Fuzzy *rich* (`[jsd, fe, mean_m, std_m]` per skala)\n",
    "  + 4 metode entropy (EDM-Fuzzy, CMSE, FME, JSD-Fuzzy) + ANN + per-scenario + eksperimen S.\n",
    "- `archive/SUMBER_01a_Paper_Q3_RQ1_RQ5.ipynb` → RQ1–RQ5 (validasi feature matrix, stabilitas CV, separability,\n",
    "  5 skenario klasifikasi EDM-Fuzzy vs JSD-Fuzzy, paired t-test) + Paper Summary Table.\n",
    "\n",
    "Pipeline, data, config, dan seed identik dengan kedua referensi → angka head-to-head konsisten.\n",
    "FAST_MODE default. Target: Kaggle + lokal.\n",
]

nb_new = copy.deepcopy(dev)
nb_new['cells'] = new_cells

# Make the Paper Summary Excel export robust: wrap openpyxl ExcelWriter in
# try/except with CSV fallback so Run All does not abort if openpyxl is absent.
for c in nb_new['cells']:
    if c.get('cell_type') != 'code':
        continue
    src = ''.join(c['source'])
    if 'pd.ExcelWriter("exports/paper_summary_tables.xlsx", engine="openpyxl")' in src:
        old = (
            'with pd.ExcelWriter("exports/paper_summary_tables.xlsx", engine="openpyxl") as writer:\n'
            '    for sheet_name, df in summary_frames:\n'
            '        df.to_excel(writer, sheet_name=sheet_name, index=False)\n'
            '    if "rq2_by_scale" in globals():\n'
            '        rq2_by_scale.to_excel(writer, sheet_name="rq2_cv_by_scale")\n'
            '    if "rq3_table" in globals():\n'
            '        rq3_table.to_excel(writer, sheet_name="rq3_separability", index=False)\n'
            '\n'
            'print("\\n[Saved] exports/paper_summary_tables.xlsx")'
        )
        new = (
            'try:\n'
            '    with pd.ExcelWriter("exports/paper_summary_tables.xlsx", engine="openpyxl") as writer:\n'
            '        for sheet_name, df in summary_frames:\n'
            '            df.to_excel(writer, sheet_name=sheet_name, index=False)\n'
            '        if "rq2_by_scale" in globals():\n'
            '            rq2_by_scale.to_excel(writer, sheet_name="rq2_cv_by_scale")\n'
            '        if "rq3_table" in globals():\n'
            '            rq3_table.to_excel(writer, sheet_name="rq3_separability", index=False)\n'
            '    print("\\n[Saved] exports/paper_summary_tables.xlsx")\n'
            'except Exception as _e:\n'
            '    print(f"[skip] Excel export gagal ({_e}); menyimpan CSV sebagai fallback.")\n'
            '    for _sheet_name, _df in summary_frames:\n'
            '        _df.to_csv(f"exports/{_sheet_name}.csv", index=False)\n'
            '    if "rq2_by_scale" in globals():\n'
            '        rq2_by_scale.to_csv("exports/rq2_cv_by_scale.csv")\n'
            '    if "rq3_table" in globals():\n'
            '        rq3_table.to_csv("exports/rq3_separability.csv", index=False)'
        )
        if old in src:
            c['source'] = src.replace(old, new).splitlines(keepends=True)
            print('[patch] wrapped openpyxl ExcelWriter in try/except (CSV fallback)')

# Runtime fixes (vectorised SampEn, RUNTIME_PROFILE, wall-clock budget guard).
# Without these the merged notebook needs >12 h and Kaggle SIGKILLs it (exit 137).
from patch_runtime import apply_runtime_patches
apply_runtime_patches(nb_new)

# Clear outputs / execution_count for a clean Run All
for c in nb_new['cells']:
    if c.get('cell_type') == 'code':
        c['outputs'] = []
        c['execution_count'] = None

json.dump(nb_new, open(OUT, 'w'), ensure_ascii=False, indent=1)
print('WROTE', OUT)
print('  total cells        =', len(nb_new['cells']))
print('  dev_export_idx     =', dev_export_idx)
print('  paper_rq_start     =', paper_rq_start)
print('  paper_export_idx   =', paper_export_idx)
print('  rq_block cells     =', len(rq_block))
print('  pipeline cells     =', dev_export_idx)
print('  tail cells         =', len(dev['cells']) - dev_export_idx)
