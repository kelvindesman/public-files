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
# (id, judul, file markdown di sidang/qa) untuk catatan bergambar di luar notebook
TAMBAHAN = [("portabel", "dari eksperimen ke model portabel (bergambar)", "13_alur_model_portabel.md")]


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


PRONOMINA = r"saya|Saya|Anda|kami|Kami|kita|Kita|aku|Aku"

# Frasa yang perlu ditulis ulang utuh supaya tetap wajar tanpa kata ganti orang.
FRASA = [
    ("Bapak/Ibu bisa", "Penguji bisa"), ("Bu Luh", "pembimbing"),
    ("hasil Anda", "hasil penelitian ini"), ("F1 Anda", "F1-nya"),
    ("paper kami", "paper sebelumnya"), ("buatan Anda", "buatan sendiri"),
    ("skenario Anda", "skenarionya"), ("nested grouped CV Anda", "nested grouped CV yang dipakai"),
    ("persiapan saya", "persiapan sidang"),
    ("membuat Anda naik", "membuat posisi naik"), ("Anda mundur", "algoritma mundur"),
    ("Saat presentasi kita tidak melatih; kita **memuat**", "Saat presentasi tidak ada pelatihan; yang dilakukan hanya **memuat**"),
    ("seberapa sering kita melihat beda sebesar ini", "seberapa sering beda sebesar ini muncul"),
    ("kita curiga termometer itu rusak", "termometer itu patut dicurigai rusak"),
    ("kita tahu persis", "diketahui persis"), ("kami mengukurnya", "sudah diukur"),
    ("kita menyembunyikan kasus sulit", "kasus sulit ikut tersembunyi"),
    ("saya tidak boleh merusak sensor", "sensor sungguhan tidak boleh dirusak"),
    ("lalu menulis ulang forward pass", "lalu forward pass ditulis ulang"),
    ("Saran saya", "Sarannya:"), ("dan menyebut hasilnya deskriptif", "dan hasilnya disebut deskriptif"),
    ("Saya memakainya sebagai pembanding dan mengganti hanya", "Metode itu dipakai sebagai pembanding; yang diganti hanya"),
    ("saya tunjukkan", "akan ditunjukkan"), ("saya cek", "akan dicek"), ("Saya cek", "Akan dicek"),
    ("saya tidak menemukan", "belum ditemukan"), ("saya punya", "tersedia"),
    ("saya menyebutnya", "hasil ini disebut"), ("Saya memakainya", "Pustaka ini dipakai"),
    ("Saya menambahkan", "Ditambahkan"), ("Saya memakai", "Dipakai"),
    ("saya membaca", "yang dipakai adalah"), ("saya menekankan", "yang ditekankan adalah"),
]
# Kata kerja dasar yang dipasifkan: "saya tampilkan" -> "ditampilkan".
PASIF = (r"tampilkan|perbaiki|klaim|koreksi|tulis|laporkan|catat|uji|cek|tunjukkan|ukur|"
         r"rata-ratakan|ambil|beri|acak|baca|ralat|pakai|masukkan")


def gaya_netral(t: str) -> str:
    """Hapus semua kata ganti orang (saya/Anda/kami/kita/aku) dan sapaan Bapak/Ibu."""
    for lama, baru in FRASA:
        t = re.sub(r"\s+".join(map(re.escape, lama.split())), baru, t)

    def pasif(m: re.Match) -> str:
        kata = "di" + m.group(2)
        return kata.capitalize() if m.group(1)[0].isupper() else kata
    t = re.sub(rf"\b({PRONOMINA})\s+({PASIF})\b", pasif, t)
    # sisa subjek aktif -> "penelitian ini"; sisa kepemilikan ("Saran saya") -> dibuang
    t = re.sub(r"(^|[.!?:\"]\s*|\*\s*)(Saya|Kami|Kita|Aku)\s+(?=me|ber|men|tidak)",
               lambda m: m.group(1) + "Penelitian ini ", t, flags=re.M)
    t = re.sub(r"\b(saya|kami|kita|aku)\s+(?=me|ber|tidak)", "penelitian ini ", t)
    t = re.sub(rf"\s+\b({PRONOMINA})\b", "", t)
    return t


# ---------------------------------------------------------------- tautan bukti ke Kaggle
KAGGLE = "https://www.kaggle.com/code/"
NAMA_SIDANG = {"sidang_satu": "00_Sidang_Deteksi_Fault", "sidang_peta": "00_Peta_Program",
               "sidang_alur": "01_Alur_Sistem", "sidang_hasil": "02_Hasil",
               "sidang_demo": "03_Demo_Model_Tersimpan", "sidang_interaktif": "04_Sidang_Interaktif"}


def peta_kaggle() -> tuple[dict[str, dict], dict[str, dict]]:
    """({kunci NB: kernel}, {nama notebook sidang: kernel}) dari .kaggle_push/*/kernel-metadata.json."""
    nb, sidang = {}, {}
    for f in sorted((ROOT / ".kaggle_push").glob("*/kernel-metadata.json")):
        meta = json.loads(f.read_text())
        k = {"folder": f.parent.name, "id": meta["id"], "judul": meta["title"],
             "url": KAGGLE + meta["id"]}
        folder = f.parent.name
        if folder in NAMA_SIDANG:
            sidang[NAMA_SIDANG[folder]] = k
        elif re.fullmatch(r"\d\d[a-z]?", folder):
            nb[folder] = k
        elif folder == "sens12":
            nb["12"] = k
        else:
            k["lain"] = True
            nb.setdefault(folder, k)
    return nb, sidang


def tautkan(md: str, nb: dict, sidang: dict) -> str:
    """Ubah rujukan NBxx / "notebook xx" / `hasilxx/...` / `sidang/0x_*.ipynb` jadi tautan Kaggle.

    Blok kode, tautan yang sudah ada, dan isi span kode lain dibiarkan.
    """
    def kunci(no, b=""):
        return f"{int(no):02d}{(b or '').lower()}"

    def ganti_teks(t):
        def f(m):
            k = kunci(m.group(2) or m.group(4), m.group(3) or m.group(5))
            return f"[{m.group(0)}]({nb[k]['url']})" if k in nb else m.group(0)
        return re.sub(r"\b(NB ?(\d\d)([b])?|[Nn]otebook (\d\d?)([b])?)\b", f, t)

    def ganti_kode(span):
        isi = span[1:-1]
        m = re.match(r"hasil_?(\d\d)([b])?/", isi)
        if m and kunci(*m.groups()) in nb:
            return f"[{span}]({nb[kunci(*m.groups())]['url']}/output)"
        m = re.match(r"sidang/(0\d_[A-Za-z_]+?)(?:_presentasi)?\.ipynb$", isi)
        if m and m.group(1) in sidang:
            return f"[{span}]({sidang[m.group(1)]['url']})"
        return span

    keluar = []
    for i, blok in enumerate(re.split(r"(```.*?```)", md, flags=re.S)):
        if i % 2:
            keluar.append(blok); continue
        for j, bag in enumerate(re.split(r"(\[[^\]\n]*\]\([^)\s]*\))", blok)):
            if j % 2:
                keluar.append(bag); continue
            for h, pot in enumerate(re.split(r"(`[^`\n]+`)", bag)):
                keluar.append(ganti_kode(pot) if h % 2 else ganti_teks(pot))
    return "".join(keluar)


def dok_peta_kaggle(nb: dict, sidang: dict) -> str:
    baris = ["Semua notebook Kaggle (akun kelvindsmn, private). Rujukan NBxx, `hasilxx/...` dan "
             "`sidang/0x_*.ipynb` di catatan lain sudah tertaut ke halaman ini atau ke tab Output kernel. "
             "File `sidang/qa/hasil_uji/*` dihitung skrip lokal di `sidang/qa/`, tidak ada di Kaggle.", "",
             "## Notebook sidang", "", "| Notebook | Kaggle |", "|---|---|"]
    baris += [f"| `sidang/{n}.ipynb` | [{k['judul']}]({k['url']}) |" for n, k in sorted(sidang.items())]
    baris += ["", "## Notebook penelitian", "", "| NB | Kaggle | Keluaran (log, CSV) |", "|---|---|---|"]
    for key, k in sorted(nb.items()):
        baris.append(f"| {key} | [{k['judul']}]({k['url']}) | [Output]({k['url']}/output) |")
    baris += ["", "## Dataset", "",
              "- [sidang-bahan](https://www.kaggle.com/datasets/kelvindsmn/sidang-bahan): cache dan bobot model untuk notebook sidang",
              "- [model-deteksi-fault-tersimpan](https://www.kaggle.com/datasets/kelvindsmn/model-deteksi-fault-tersimpan): model tersimpan N = 700 (NB16)",
              "- [data-hipotetik-fault-sensor-berlabel](https://www.kaggle.com/datasets/kelvindsmn/data-hipotetik-fault-sensor-berlabel): data dan model NB15"]
    return "\n".join(baris)


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
        isi, judul = gaya_netral(isi), gaya_netral(judul)
        docs.append({"id": f"c{len(docs) + 1:02d}", "grup": grup, "judul": judul, "md": isi})
    n_nb = len(docs)
    # Catatan bergambar yang ditulis langsung di sidang/qa (bukan dari notebook): disisipkan
    # setelah catatan per bagian notebook terakhir.
    pos = max(i for i, d in enumerate(docs) if d["grup"] == "Catatan per bagian notebook") + 1
    for i, (did, judul, f) in enumerate(TAMBAHAN):
        docs.insert(pos + i, {"id": did, "grup": "Catatan per bagian notebook", "judul": judul,
                              "md": (Path(__file__).parent / f).read_text(encoding="utf-8")})
    sisa = [(d["id"], m.group(0)) for d in docs
            for m in re.finditer(r"\b(saya|Saya|Anda|kami|Kami|kita|Kita|aku|Aku|Bapak|Ibu)\b|\w+-?ku\b(?<!laku)(?<!baku)(?<!buku)", d["judul"] + " " + d["md"])]
    assert not sisa, f"masih ada kata ganti orang: {sisa[:10]}"
    n_sel = sum(TAG in c.metadata.get("tags", []) for c in nb.cells)
    assert n_nb == n_sel - 1, f"{n_nb} dokumen dari {n_sel} sel catatan (1 sel judul)"
    peta_nb, peta_sidang = peta_kaggle()
    for d in docs:
        d["md"] = tautkan(d["md"], peta_nb, peta_sidang)
    docs.append({"id": "kaggle", "grup": "Bukti di Kaggle", "judul": "Peta link Kaggle",
                 "md": tautkan(dok_peta_kaggle(peta_nb, peta_sidang), peta_nb, peta_sidang)})
    commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cs"],
                            capture_output=True, text=True).stdout.strip()
    (keluar / "catatan.json").write_text(json.dumps({"commit": commit, "docs": docs}, ensure_ascii=False),
                                         encoding="utf-8")
    (keluar / "index.html").write_text(HTML.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"{len(docs)} dokumen -> {keluar} (commit {commit})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
