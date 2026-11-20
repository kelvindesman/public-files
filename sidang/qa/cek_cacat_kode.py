"""Verifikasi cacat/keanehan kode yang dibahas di 04_walkthrough_kode.md.

Jalankan dari akar repo:
    nice -n 19 ./venv/bin/python sidang/qa/cek_cacat_kode.py

Semua contoh mainan (detik), tidak mengubah file apa pun.
"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import deteksi_fault_core as core  # noqa: E402

warnings.filterwarnings("ignore")
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def judul(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
judul("1. lm_pack (W semua lalu b semua) vs lm_unpack (W1,b1,W2,b2 berselang)")
tpl = core.lm_init_layers(2, [2], 2, seed=0)
th = core.lm_pack(tpl)
un = core.lm_unpack(th, tpl)
print("template (niat Xavier):")
for i, (W, b) in enumerate(tpl, 1):
    print(f"  W{i} =\n{W}\n  b{i} = {b}")
print("theta = lm_pack(template) =", th)
print("yang benar-benar dipakai jaringan (lm_unpack):")
for i, (W, b) in enumerate(un, 1):
    print(f"  W{i} =\n{W}\n  b{i} = {b}")


def pack_benar(layers):
    return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in layers])


assert np.allclose(core.lm_unpack(pack_benar(tpl), tpl)[1][0], tpl[1][0])
print("perbaikan pack_benar(): lm_unpack(pack_benar(t)) == t  -> OK")
# pack -> unpack -> pack bukan identitas
rt = core.lm_pack(core.lm_unpack(th, tpl))
print("lm_pack(lm_unpack(theta)) == theta ?", bool(np.allclose(rt, th)))

print("\nKonfigurasi nyata: n_in = 60 (4 sensor x 15 skala). Lapis 1 SELALU benar (W1 = niat,")
print("b1 terisi bobot lapis 2). Tabel di bawah: lapis >= 2 sesudah lm_unpack.")
print("out: 2 = S1/S5, 5 = S2/S4, 7 = S3, 4 = sensor-ID")
print(f"{'hidden':>9s} {'out':>3s} | {'lapis':>5s} {'%W nol':>7s} {'unit hulu W-keluar nol':>23s} "
      f"{'b != 0':>7s}")
for hidden in [(16,), (32,), (64,), (16, 8), (32, 8), (32, 16), (32, 32), (64, 16)]:
    for n_out in (2, 5, 7, 4):
        t = core.lm_init_layers(60, list(hidden), n_out, seed=42)
        u = core.lm_unpack(core.lm_pack(t), t)
        assert np.array_equal(u[0][0], t[0][0])
        for li, ((W0, b0), (W, b)) in enumerate(zip(t, u), 1):
            if li == 1:
                continue
            zero_pct = 100.0 * np.mean(W == 0)
            dead_in = int(np.sum(np.all(W == 0, axis=1)))
            print(f"{str(hidden):>9s} {n_out:3d} | {li:5d} {zero_pct:6.1f}% {dead_in:12d}/{W.shape[0]:<10d} "
                  f"{int(np.sum(b != 0)):4d}/{b.size:<3d}")

# ---------------------------------------------------------------------------
judul("2. Konsistensi: semua jalur prediksi memakai lm_unpack yang sama")
rng = np.random.default_rng(0)
Xs = rng.normal(size=(40, 6))
ys = (Xs[:, 0] + 0.5 * Xs[:, 1] > 0).astype(int) + (Xs[:, 2] > 1).astype(int)
pipe = core.make_lm_pipeline((5,), "tanh", iters=8, max_train_s=30)
pipe.fit(Xs, ys)
sv = core.SavedLM.from_pipeline(pipe, "kelas")
d1 = np.abs(pipe.predict_proba(Xs) - sv.predict_proba(Xs)).max()
print(f"Pipeline vs SavedLM predict_proba: maks selisih = {d1:.2e}")
assert d1 < 1e-12
Ysens = (rng.random((40, 4)) < 0.4).astype(float)
sp = core.make_sensor_pipeline((5,), "tanh", iters=5, max_train_s=30)
sp.fit(Xs, Ysens)
sv2 = core.SavedLM.from_pipeline(sp, "sensor")
d2 = np.abs(sp.decision_function(Xs) if hasattr(sp, "decision_function") else 0)
d2 = np.abs(sp.named_steps["lm"].decision_function(
    sp.named_steps["sc"].transform(sp.named_steps["imp"].transform(Xs))) - sv2.decision_function(Xs)).max()
print(f"Pipeline sensor vs SavedLM decision_function: maks selisih = {d2:.2e}")
assert d2 < 1e-12
# pembaca .npz notebook 15 (15_muat_model.prob_fault) membaca W1,b1,W2,b2 berselang = lm_unpack
lm = pipe.named_steps["lm"]
shapes = [W.shape for W, _ in lm.template_]
i, layers = 0, []
for (r_, c_) in shapes:
    W = lm.theta_[i:i + r_ * c_].reshape(r_, c_); i += r_ * c_
    b = lm.theta_[i:i + c_]; i += c_
    layers.append((W, b))
same = all(np.array_equal(a[0], b_[0]) and np.array_equal(a[1], b_[1])
           for a, b_ in zip(layers, core.lm_unpack(lm.theta_, lm.template_)))
print("Urutan baca prob_fault (notebook 15) == lm_unpack ?", same)
assert same

# ---------------------------------------------------------------------------
judul("3. Dampak init teracak: LM dengan pack asli vs pack_benar (data mainan)")
rng = np.random.default_rng(1)
Xd = rng.normal(size=(120, 10))
asli = core.lm_pack
for n_cls, hidden in ((3, [8]), (2, [8, 4])):
    yd = np.argmax(Xd[:, :n_cls] + 0.3 * rng.normal(size=(120, n_cls)), axis=1)
    Yd = np.eye(n_cls)[yd]
    for iters in (6, 15):
        for nama, fn in (("asli (teracak)", asli), ("perbaikan", pack_benar)):
            core.lm_pack = fn
            rows = []
            for s in range(5):
                r = core.lm_fit(Xd, Yd, 10, hidden, n_cls, iters=iters, seed=s)
                acc = float((core.lm_predict_proba(r["theta"], r["template"], Xd).argmax(1) == yd).mean())
                rows.append((r["hist"]["loss"][0], r["train_mse"], acc))
            a_ = np.array(rows)
            core.lm_pack = asli
            print(f"hidden={str(hidden):7s} kelas={n_cls} iter={iters:2d} {nama:15s}: MSE awal {a_[:, 0].mean():.4f} | "
                  f"MSE latih akhir {a_[:, 1].mean():.4f} (min {a_[:, 1].min():.4f}, maks {a_[:, 1].max():.4f}) "
                  f"| akurasi latih {a_[:, 2].mean():.3f}")
core.lm_pack = asli
# jaringan 2 lapis tersembunyi + 2 keluaran: W3 nol semua -> Jacobian lapis 1-2 nol di awal
t = core.lm_init_layers(10, [8, 4], 2, seed=0)
th = core.lm_pack(t)
J = core.lm_residual_jacobian(th, t, Xd[:20], np.eye(2)[np.zeros(20, int)], "tanh")
n12 = 10 * 8 + 8 + 8 * 4 + 4
print(f"[8,4]->2 di theta awal: kolom Jacobian lapis 1-2 yang tak nol = "
      f"{int(np.sum(np.abs(J[:, :n12]).sum(0) > 0))}/{n12}; lapis 3 = "
      f"{int(np.sum(np.abs(J[:, n12:]).sum(0) > 0))}/{J.shape[1] - n12}")

# ---------------------------------------------------------------------------
judul("4. Fault bias konstan tidak terlihat oleh kedua fitur entropi")
x = core.np.sin(np.arange(300) * 0.21) * 3 + 40 + np.random.default_rng(3).normal(0, 0.4, 300)
for meth, f in (("EDM", core.edm_fuzzy_entropy_1d), ("JSD", core.jsd_fuzzy_entropy_1d)):
    a = f(x, np.arange(1, 16), seed=5)
    b = f(x + 0.08, np.arange(1, 16), seed=5)
    c = f(x + 5.0, np.arange(1, 16), seed=5)
    print(f"{meth}: maks |F(x+0.08) - F(x)| = {np.nanmax(np.abs(a - b)):.2e} ; "
          f"maks |F(x+5) - F(x)| = {np.nanmax(np.abs(a - c)):.2e}")

# ---------------------------------------------------------------------------
judul("5. Grup CV = starts // WIN dengan stride WIN/2: tumpang-tindih antar grup")
for WIN in (200, 700, 1000):
    starts = np.arange(0, 20 * WIN, WIN // 2)
    groups = starts // WIN
    pairs = cross = 0
    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            ov = max(0, WIN - abs(starts[i] - starts[j]))
            if ov > 0:
                pairs += 1
                cross += groups[i] != groups[j]
    print(f"WIN={WIN}: pasangan window bertumpang-tindih = {pairs}, di grup BERBEDA = {cross} "
          f"({100 * cross / pairs:.0f}%), tumpang-tindih tiap pasangan itu = 50%")
starts = np.arange(0, 20 * 200, 100)
g2 = starts // (2 * 200)
print("contoh perbaikan grup = starts // (2*WIN): masih ada batas -> perlu 'purge' window di batas")

# ---------------------------------------------------------------------------
judul("6. simulate_hardware_fault: mask hilang (loss) selalu bagian dari mask stuck")
xx = np.linspace(20, 30, 100000)
y, msk = core.simulate_hardware_fault(xx.copy(), seed=1)
nan = np.isnan(y)
changed = (~nan) & (y != xx)
print(f"porsi NaN (loss) = {nan.mean():.4f} (parameter 0.05) ; porsi 'stuck' yang tersisa = "
      f"{(msk & ~nan).mean():.4f} (parameter 0.08 -> efektif ~0.03) ; total mask = {msk.mean():.4f}")

# ---------------------------------------------------------------------------
judul("7. Seed diabaikan oleh drift/spike/bias: pola sama untuk semua seed")
z = np.full(500, 30.0) + np.sin(np.arange(500))
s1 = core.simulate_spike_fault(z, **core._S, seed=1)[1]
s2 = core.simulate_spike_fault(z, **core._S, seed=999)[1]
print("posisi spike seed=1 :", np.flatnonzero(s1)[:8], "...")
print("posisi spike seed=999:", np.flatnonzero(s2)[:8], "... identik:", bool(np.array_equal(s1, s2)))
print(f"tinggi spike = 0.08 * sd(x) = {0.08 * np.nanstd(z):.4f} ; bandingkan r = 0.2 * sd = {0.2 * np.std(z, ddof=1):.4f}")

# ---------------------------------------------------------------------------
judul("8. Entropi JSD bisa negatif (phi_{m+1} > phi_m)")
x8 = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 4.0])
print("JSD-FuzzyEn deret 8 titik tau=1:", core.jsd_fuzzy_entropy_1d(x8, [1])[0])
print("\nSELESAI")
