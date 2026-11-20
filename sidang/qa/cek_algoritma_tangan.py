"""Cek hitungan tangan di 03_algoritma_dengan_tangan.md terhadap fungsi ASLI.

Jalankan dari akar repo:
    nice -n 19 ./venv/bin/python sidang/qa/cek_algoritma_tangan.py

Tiap bagian punya dua jalur:
  1. jalur "tangan": rumus ditulis ulang dengan loop skalar (modul math),
     tanpa memanggil fungsi inti;
  2. jalur "asli": fungsi dari deteksi_fault_core.py.
Keduanya harus sama (toleransi 1e-9), dan angka kunci juga harus sama
dengan konstanta HAND_* yang disalin ke file .md (toleransi 1e-9).
Semua contoh mainan: beberapa milidetik, aman untuk laptop 8 GB.
"""
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import deteksi_fault_core as core  # noqa: E402

TOL = 1e-9
np.set_printoptions(precision=6, suppress=True, linewidth=140)
N_CHECK = 0


def cek(nama, a, b, tol=TOL):
    global N_CHECK
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape, f"{nama}: bentuk beda {a.shape} vs {b.shape}"
    both_nan = np.isnan(a) & np.isnan(b)
    err = np.where(both_nan, 0.0, np.abs(a - b))
    assert np.all(np.isfinite(err)), f"{nama}: NaN di satu sisi saja"
    e = float(err.max()) if err.size else 0.0
    assert e <= tol, f"{nama}: selisih {e:.3e} > {tol}"
    N_CHECK += 1
    print(f"  [OK] {nama:<46s} maks selisih = {e:.1e}")


def tampil(nama, v):
    print(f"{nama} =\n{np.asarray(v)}")


# =====================================================================
# Konstanta yang tertulis di 03_algoritma_dengan_tangan.md (12 desimal)
# =====================================================================
HAND_LM = {
    "E0": 1.634917719156,        # E = 0.5 * sum(r^2) di theta awal
    "gnorm": 2.627734635861,     # ||J^T r|| iterasi 1
    "E1_try": 2.586706041095,    # E di theta + Delta iterasi 1 (ditolak)
    "rho1": -0.582166493645,     # rasio gain iterasi 1
    "E9_new": 0.194302037423,    # E percobaan iterasi 9 (diterima)
    "denom9": 1.381334783791,    # L(0) - L(Delta) iterasi 9
    "rho9": 0.911252346194,
    "fac9": 0.443564086033,      # max(1/3, 1 - (2 rho - 1)^3)
    "lam9_new": 0.082462781822,
}
HAND_EDM = {
    "r": 0.261861468283,
    "phi_m_tau1": 0.383517345294,
    "phi_m1_tau1": 0.331532298319,
    "FE_tau1": 0.145659613004,
    "phi_m_tau2": 0.437489954604,
}
HAND_JSD = {
    "h": 1.122263435499,
    "r_jsd": 0.166510922232,
    "jsd_pair_0_1": 0.011810952121,
    "phi_m_tau1": 0.774330360139,
    "phi_m1_tau1": 0.868786036155,
    "FE_tau1": -0.115098272073,
}

# =====================================================================
# (a) Satu iterasi LM, jaringan 2-2-2 (tanh, keluaran linear), 3 sampel
# =====================================================================
print("=" * 78)
print("(a) SATU ITERASI LEVENBERG-MARQUARDT, jaringan 2-[2]-2, 3 sampel")
print("=" * 78)
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])   # one-hot, kelas 0,1,0
SEED = 0
template = core.lm_init_layers(2, [2], 2, seed=SEED)
theta0 = core.lm_pack(template)
tampil("Xavier W1 (template)", template[0][0])
tampil("Xavier W2 (template)", template[1][0])
tampil("theta0 = lm_pack(template)", theta0)
L = core.lm_unpack(theta0, template)
W1, b1, W2, b2 = L[0][0], L[0][1], L[1][0], L[1][1]
tampil("W1 (hasil lm_unpack)", W1)
tampil("b1 (hasil lm_unpack)", b1)
tampil("W2 (hasil lm_unpack)", W2)
tampil("b2 (hasil lm_unpack)", b2)

# --- jalur tangan: maju ---
Hh = [[0.0, 0.0] for _ in range(3)]
Z1 = [[0.0, 0.0] for _ in range(3)]
O = [[0.0, 0.0] for _ in range(3)]
for i in range(3):
    for q in range(2):
        Z1[i][q] = X[i, 0] * W1[0, q] + X[i, 1] * W1[1, q] + b1[q]
        Hh[i][q] = math.tanh(Z1[i][q])
    for c in range(2):
        O[i][c] = Hh[i][0] * W2[0, c] + Hh[i][1] * W2[1, c] + b2[c]
r_hand = [O[i][c] - Y[i, c] for i in range(3) for c in range(2)]
tampil("z1 = X W1 + b1", Z1)
tampil("h = tanh(z1)", Hh)
tampil("keluaran o = h W2 + b2", O)
tampil("residual r = (o - Y).ravel()", r_hand)
cek("residual tangan vs lm_residuals", r_hand, core.lm_residuals(theta0, template, X, Y, "tanh"))
E0 = 0.5 * sum(v * v for v in r_hand)
print(f"E0 = 0.5 * sum(r^2) = {E0:.12f}")
cek("E0 vs konstanta .md", E0, HAND_LM["E0"])

# --- jalur tangan: Jacobian analitik (urutan kolom = lm_unpack: W1, b1, W2, b2) ---
J_hand = np.zeros((6, 12))
for i in range(3):
    for c in range(2):
        row = 2 * i + c
        col = 0
        for p in range(2):                 # dr/dW1[p,q] = x_ip * W2[q,c] * (1 - h_iq^2)
            for q in range(2):
                J_hand[row, col] = X[i, p] * W2[q, c] * (1 - Hh[i][q] ** 2)
                col += 1
        for q in range(2):                 # dr/db1[q] = W2[q,c] * (1 - h_iq^2)
            J_hand[row, col] = W2[q, c] * (1 - Hh[i][q] ** 2)
            col += 1
        for p in range(2):                 # dr/dW2[p,q] = h_ip * [q == c]
            for q in range(2):
                J_hand[row, col] = Hh[i][p] * (1.0 if q == c else 0.0)
                col += 1
        for q in range(2):                 # dr/db2[q] = [q == c]
            J_hand[row, col] = 1.0 if q == c else 0.0
            col += 1
J_real = core.lm_residual_jacobian(theta0, template, X, Y, "tanh")
tampil("J (6 x 12), kolom: W1_00 W1_01 W1_10 W1_11 b1_0 b1_1 W2_00 W2_01 W2_10 W2_11 b2_0 b2_1", J_hand)
cek("Jacobian tangan vs lm_residual_jacobian", J_hand, J_real)

r0 = np.array(r_hand)


def satu_iterasi(theta, lam, rinci):
    """Salinan langkah lm_fit, ditulis terbuka. Kembalikan theta, lam, catatan."""
    rr = core.lm_residuals(theta, template, X, Y, "tanh")
    E = 0.5 * float(rr @ rr)
    J = core.lm_residual_jacobian(theta, template, X, Y, "tanh")
    g = J.T @ rr
    A = J.T @ J
    d = np.where(np.diag(A) < 1e-12, 1e-12, np.diag(A))
    step = np.linalg.solve(A + lam * np.diag(d), -g)
    r_new = core.lm_residuals(theta + step, template, X, Y, "tanh")
    E_new = 0.5 * float(r_new @ r_new)
    denom = 0.5 * float(step @ (lam * d * step - g))
    rho = (E - E_new) / denom
    if rinci:
        tampil("   g = J^T r", g)
        print(f"   ||g|| = {np.linalg.norm(g):.12f}")
        tampil("   A = J^T J", A)
        tampil("   d = diag(A), dijepit >= 1e-12", d)
        tampil("   langkah Delta: (A + lam*diag(d)) Delta = -g", step)
        sisa = (A + lam * np.diag(d)) @ step + g
        print(f"   cek: maks |(A + lam D) Delta + g| = {np.abs(sisa).max():.2e}")
        tampil("   residual percobaan r(theta + Delta)", r_new)
        print(f"   E = {E:.12f}, E_baru = {E_new:.12f}")
        print(f"   L(0)-L(Delta) = 0.5 Delta^T (lam d*Delta - g) = {denom:.12f}")
        print(f"   rho = (E - E_baru)/(L(0)-L(Delta)) = {rho:.12f}")
    rec = {"lam": lam, "E": E, "E_new": E_new, "denom": denom, "rho": rho,
           "gnorm": float(np.linalg.norm(g)), "terima": E_new < E}
    if E_new < E:
        fac = max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3) if rho > 0 else 2.0
        rec["faktor"] = fac
        theta, lam = theta + step, lam * fac
        if rinci:
            print(f"   DITERIMA. faktor Nielsen = max(1/3, 1-(2rho-1)^3) = {fac:.12f}")
    else:
        rec["faktor"] = 2.0
        lam = lam * 2.0
        if rinci:
            print("   DITOLAK (E_baru >= E): theta tetap, lam dikali 2")
    lam = float(np.clip(lam, 1e-12, 1e12))
    rec["lam_baru"] = lam
    return theta, lam, rec


theta, lam = theta0.copy(), 1e-3
log = []
for it in range(1, 10):
    rinci = it in (1, 9)
    if rinci:
        print(f"--- iterasi {it} (rinci), lam = {lam:.12f} ---")
    theta, lam, rec = satu_iterasi(theta, lam, rinci)
    log.append(rec)
    fit = core.lm_fit(X, Y, 2, [2], 2, activation="tanh", iters=it, seed=SEED, lam0=1e-3)
    cek(f"theta sesudah iterasi {it} vs lm_fit(iters={it})", theta, fit["theta"])
    if it < 9:
        nxt = core.lm_fit(X, Y, 2, [2], 2, activation="tanh", iters=it + 1, seed=SEED, lam0=1e-3)
        cek(f"lam sesudah iterasi {it} vs hist['lam'][{it}]", lam, nxt["hist"]["lam"][it])
print("ringkas iterasi 1..9:")
print(" it   lam_dipakai        E            E_percobaan      L(0)-L(D)        rho          terima  lam_baru")
for it, rcd in enumerate(log, 1):
    print(f" {it:2d}  {rcd['lam']:.6f}  {rcd['E']:.9f}  {rcd['E_new']:.9f}  {rcd['denom']:.9f}  "
          f"{rcd['rho']:+.9f}  {'ya' if rcd['terima'] else 'tidak':5s}  {rcd['lam_baru']:.9f}")
cek("E0 vs konstanta .md", log[0]["E"], HAND_LM["E0"])
cek("||g|| it1 vs konstanta .md", log[0]["gnorm"], HAND_LM["gnorm"])
cek("E percobaan it1 vs konstanta .md", log[0]["E_new"], HAND_LM["E1_try"])
cek("rho it1 vs konstanta .md", log[0]["rho"], HAND_LM["rho1"])
cek("E percobaan it9 vs konstanta .md", log[8]["E_new"], HAND_LM["E9_new"])
cek("denom it9 vs konstanta .md", log[8]["denom"], HAND_LM["denom9"])
cek("rho it9 vs konstanta .md", log[8]["rho"], HAND_LM["rho9"])
cek("faktor it9 vs konstanta .md", log[8]["faktor"], HAND_LM["fac9"])
cek("lam baru it9 vs konstanta .md", log[8]["lam_baru"], HAND_LM["lam9_new"])
theta1 = theta
P = core.lm_predict_proba(theta1, template, X, "tanh")
tampil("softmax(keluaran) sesudah langkah diterima pertama (predict_proba)", P)
print(f"prediksi kelas = {P.argmax(axis=1)} (target {Y.argmax(axis=1)})")

# =====================================================================
# (b) Fuzzy entropy jarak Euclidean (EDM-Fuzzy) pada deret 8 titik
# =====================================================================
print()
print("=" * 78)
print("(b) EDM-FUZZY: deret 8 titik, m = 2, r = 0.2*sd")
print("=" * 78)
x = np.array([3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 4.0])
print("x =", x)
mean_x = sum(x) / len(x)
sd = math.sqrt(sum((v - mean_x) ** 2 for v in x) / (len(x) - 1))
r = 0.2 * sd
print(f"mean = {mean_x}, sd(ddof=1) = {sd:.12f}, r = 0.2*sd = {r:.12f}")
cek("r vs konstanta .md", r, HAND_EDM["r"])


def cg_tangan(xs, s, k):
    n = (len(xs) - k) // s
    return [sum(xs[k + j * s: k + (j + 1) * s]) / s for j in range(n)]


def embed_tangan(y, mm):
    out = []
    for i in range(len(y) - mm + 1):
        v = list(y[i:i + mm])
        mu = sum(v) / mm
        out.append([a - mu for a in v])
    return out


def phi_tangan(V, dist, r_):
    N = len(V)
    if N < 3:
        return float("nan"), None, None
    D = [[dist(V[i], V[j]) for j in range(N)] for i in range(N)]
    M = [[0.0 if i == j else 1.0 / (1.0 + D[i][j] / (r_ + core.EPS)) for j in range(N)] for i in range(N)]
    rows = [sum(M[i]) / (N - 1) for i in range(N)]
    return sum(rows) / N, np.array(D), np.array(M)


def euclid(a, b):
    return math.sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


ent_k = {}
for s in (1, 2):
    for k in range(s):
        y = cg_tangan(list(x), s, k)
        cek(f"coarse-grain tau={s} k={k}", y, core.coarse_grain_cmse(x, s, k))
        print(f"tau={s}, k={k}: y = {np.round(y, 6)}  (panjang {len(y)})")
        if len(y) < 4:
            print("   panjang < m+2 = 4 -> offset ini DILEWATI oleh kode")
            continue
        Vm = embed_tangan(y, 2)
        Vm1 = embed_tangan(y, 3)
        cek(f"embed m=2 tau={s} k={k}", Vm, core.embed_matrix_centered(np.array(y), 2))
        cek(f"embed m=3 tau={s} k={k}", Vm1, core.embed_matrix_centered(np.array(y), 3))
        tampil(f"   vektor m=2 terpusat (tau={s})", Vm)
        tampil(f"   vektor m=3 terpusat (tau={s})", Vm1)
        pm, Dm, Mm = phi_tangan(Vm, euclid, r)
        pm1, Dm1, Mm1 = phi_tangan(Vm1, euclid, r)
        if Dm is not None:
            tampil(f"   jarak Euclid m=2 (tau={s})", Dm)
            tampil(f"   keanggotaan mu = 1/(1+d/r), diagonal 0 (tau={s})", Mm)
            print(f"   rata-rata baris m=2 = {np.round(Mm.sum(axis=1) / (len(Vm) - 1), 6)}")
        if Dm1 is not None:
            tampil(f"   jarak Euclid m=3 (tau={s})", Dm1)
            tampil(f"   keanggotaan m=3 (tau={s})", Mm1)
            print(f"   rata-rata baris m=3 = {np.round(Mm1.sum(axis=1) / (len(Vm1) - 1), 6)}")
        cek(f"phi_m tau={s} k={k}", pm, core.phi_euclidean(np.array(Vm), r, 128, 0))
        cek(f"phi_m1 tau={s} k={k}", pm1, core.phi_euclidean(np.array(Vm1), r, 128, 0))
        print(f"   phi_m = {pm:.12f}, phi_m+1 = {pm1}")
        if pm > 0 and pm1 > 0 and math.isfinite(pm1):
            ent_k[(s, k)] = math.log(pm / pm1)
            print(f"   ln(phi_m/phi_m+1) = {ent_k[(s, k)]:.12f}")
        else:
            print("   phi_m+1 = NaN (hanya 2 vektor) -> tidak dihitung")
        if s == 1:
            cek("phi_m tau=1 vs konstanta .md", pm, HAND_EDM["phi_m_tau1"])
            cek("phi_m1 tau=1 vs konstanta .md", pm1, HAND_EDM["phi_m1_tau1"])
        if s == 2 and k == 0:
            cek("phi_m tau=2 vs konstanta .md", pm, HAND_EDM["phi_m_tau2"])
fe_real = core.edm_fuzzy_entropy_1d(x, [1, 2], m=2, r_ratio=0.2, n_ref=128, seed=0)
fe_hand = [ent_k.get((1, 0), np.nan), np.nan]
print(f"edm_fuzzy_entropy_1d(x, skala [1,2]) = {fe_real}")
cek("FE tau=1,2 tangan vs edm_fuzzy_entropy_1d", fe_hand, fe_real)
cek("FE tau=1 vs konstanta .md", fe_real[0], HAND_EDM["FE_tau1"])
Ftoy = np.array([[0.30, 1.0], [0.50, np.nan], [0.10, 3.0]])
tampil("sanitize(contoh matriks fitur dengan NaN)", core.sanitize(Ftoy))
cek("sanitize isi NaN dengan median kolom", core.sanitize(Ftoy), [[0.3, 1.0], [0.5, 2.0], [0.1, 3.0]])

# n_ref: subsampling titik acuan hanya aktif bila jumlah vektor > n_ref
Vbig = core.embed_matrix_centered(np.sin(np.arange(40) * 0.7), 2)
full = core.phi_euclidean(Vbig, 0.1, n_ref=128, seed=0)
sub = core.phi_euclidean(Vbig, 0.1, n_ref=10, seed=0)
print(f"n_ref: 39 vektor, phi penuh = {full:.6f}, phi dengan n_ref=10 = {sub:.6f} (perkiraan acak)")

# =====================================================================
# (c) JSD-Fuzzy pada deret yang sama
# =====================================================================
print()
print("=" * 78)
print("(c) JSD-FUZZY: deret yang sama, m = 2, 8 bin, h = jarak antar pusat")
print("=" * 78)
centers = [-3 * sd + i * (6 * sd) / 7 for i in range(8)]
h = centers[1] - centers[0]
r_jsd = 0.2 * math.sqrt(math.log(2))
print("pusat bin c =", np.round(centers, 6))
print(f"h = {h:.12f}, r_jsd = 0.2*sqrt(ln 2) = {r_jsd:.12f}")
cek("pusat bin vs np.linspace", centers, np.linspace(-3 * sd, 3 * sd, 8))
cek("h vs konstanta .md", h, HAND_JSD["h"])
cek("r_jsd vs konstanta .md", r_jsd, HAND_JSD["r_jsd"])


def dist_tangan(v):
    K = [sum(math.exp(-0.5 * ((vj - c) / (h + core.EPS)) ** 2) for vj in v) for c in centers]
    S = sum(K)
    return [kk / (S + 1e-300) for kk in K], K


def jsd_tangan(p, q):
    t = 0.0
    for a, b in zip(p, q):
        mm = 0.5 * (a + b)
        lm = math.log(mm + core.EPS)
        t += 0.5 * (a * (math.log(a + core.EPS) - lm) + b * (math.log(b + core.EPS) - lm))
    return max(t, 0.0)


y1 = list(x)
Vm = embed_tangan(y1, 2)
Vm1 = embed_tangan(y1, 3)
c_np = np.linspace(-3 * sd, 3 * sd, 8)
for mm, V in ((2, Vm), (3, Vm1)):
    Ps, Ks = zip(*[dist_tangan(v) for v in V])
    Ps = np.array(Ps)
    tampil(f"   kernel mentah K (m={mm}), baris = vektor", np.array(Ks))
    tampil(f"   distribusi P (m={mm}), tiap baris jumlah 1", Ps)
    cek(f"P tangan vs vectors_to_distributions m={mm}", Ps,
        core.vectors_to_distributions(np.array(V), c_np, h))
    Jm = np.array([[jsd_tangan(Ps[i], Ps[j]) for j in range(len(V))] for i in range(len(V))])
    cek(f"JSD tangan vs pairwise_jsd m={mm}", Jm, core.pairwise_jsd(Ps, Ps))
    tampil(f"   JSD (nat) m={mm}", Jm)
    tampil(f"   d = sqrt(JSD) m={mm}", np.sqrt(Jm))
    pj, _, Mj = phi_tangan([list(p) for p in Ps], lambda a, b: math.sqrt(jsd_tangan(a, b)), r_jsd)
    tampil(f"   keanggotaan mu = 1/(1+d/r_jsd) m={mm}", Mj)
    print(f"   rata-rata baris = {np.round(Mj.sum(axis=1) / (len(V) - 1), 6)}")
    cek(f"phi_jsd m={mm} tangan vs phi_jsd", pj, core.phi_jsd(np.array(V), c_np, h, r_jsd, 128, 0))
    print(f"   phi (m={mm}) = {pj:.12f}")
    if mm == 2:
        phi_j_m = pj
        P2 = Ps
        cek("JSD(P1,P2) vs konstanta .md", Jm[0, 1], HAND_JSD["jsd_pair_0_1"])
    else:
        phi_j_m1 = pj

# rincian satu pasangan: vektor 1 dan 2
p, q = P2[0], P2[1]
mmix = 0.5 * (p + q)
print("rincian pasangan v1, v2:")
tampil("   P1", p)
tampil("   P2", q)
tampil("   M = (P1+P2)/2", mmix)
kl1 = float(np.sum(p * (np.log(p + core.EPS) - np.log(mmix + core.EPS))))
kl2 = float(np.sum(q * (np.log(q + core.EPS) - np.log(mmix + core.EPS))))
print(f"   KL(P1||M) = {kl1:.12f}, KL(P2||M) = {kl2:.12f}, JSD = {0.5 * (kl1 + kl2):.12f}")
H = lambda v: -float(np.sum(v * np.log(v)))  # noqa: E731
print(f"   bentuk Lin 1991: H(M) - (H(P1)+H(P2))/2 = {H(mmix) - 0.5 * (H(p) + H(q)):.12f}")
fe_j = math.log(phi_j_m / phi_j_m1)
print(f"JSD-FuzzyEn tau=1 = ln({phi_j_m:.12f}/{phi_j_m1:.12f}) = {fe_j:.12f}")
cek("phi_jsd m vs konstanta .md", phi_j_m, HAND_JSD["phi_m_tau1"])
cek("phi_jsd m+1 vs konstanta .md", phi_j_m1, HAND_JSD["phi_m1_tau1"])
fj_real = core.jsd_fuzzy_entropy_1d(x, [1, 2], m=2, n_ref=128, seed=0)
print(f"jsd_fuzzy_entropy_1d(x, skala [1,2]) = {fj_real}")
cek("FE tau=1,2 tangan vs jsd_fuzzy_entropy_1d", [fe_j, np.nan], fj_real)
cek("JSD-FE tau=1 vs konstanta .md", fj_real[0], HAND_JSD["FE_tau1"])
# sifat: vektor terpusat [-a, a] dan [a, -a] -> distribusi sama -> JSD 0
pa = core.vectors_to_distributions(np.array([[-0.5, 0.5], [0.5, -0.5]]), c_np, h)
print(f"vektor naik [-0.5, 0.5] vs turun [0.5, -0.5]: JSD = {core.pairwise_jsd(pa, pa)[0, 1]:.3e}, "
      f"Euclid = {math.sqrt(2.0):.6f}")

# =====================================================================
# (d) Self-check Jacobian: analitik vs numerik (selisih tengah)
# =====================================================================
print()
print("=" * 78)
print("(d) SELF-CHECK JACOBIAN: analitik vs numerik (selisih tengah, eps = 1e-6)")
print("=" * 78)
for act, th_cek, nama_th in (("tanh", theta0, "theta awal"), ("relu", theta0, "theta awal"),
                             ("tanh", theta1, "theta sesudah iterasi 9"),
                             ("relu", theta1, "theta sesudah iterasi 9")):
    eps = 1e-6
    Jn = np.empty_like(J_real)
    for pidx in range(theta0.size):
        tp = th_cek.copy()
        tm = th_cek.copy()
        tp[pidx] += eps
        tm[pidx] -= eps
        Jn[:, pidx] = (core.lm_residuals(tp, template, X, Y, act)
                       - core.lm_residuals(tm, template, X, Y, act)) / (2 * eps)
    Ja = core.lm_residual_jacobian(th_cek, template, X, Y, act)
    err = np.abs(Ja - Jn).max()
    print(f"{act:4s} @ {nama_th:<24s}: maks |J_analitik - J_numerik| = {err:.3e}")
    if act == "tanh" and th_cek is theta1:
        tampil("   J analitik (tanh, theta sesudah iterasi 9)", Ja)
        tampil("   J numerik (tanh)", Jn)
    assert err < 1e-8, f"Jacobian {act} tidak cocok"
    N_CHECK += 1

print()
print(f"SEMUA {N_CHECK} CEK LOLOS (toleransi {TOL}).")
