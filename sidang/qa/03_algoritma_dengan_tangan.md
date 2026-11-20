# 03 — Algoritma dengan tangan (angka nyata dari kode)

Dokumen ini menghitung **dengan tangan** empat hal yang paling sering diminta
penguji: (a) satu iterasi Levenberg–Marquardt, (b) fuzzy entropy dengan jarak
Euclidean (EDM-Fuzzy), (c) fuzzy entropy dengan kemiripan Jensen–Shannon
(JSD-Fuzzy), dan (d) self-check Jacobian. Semua angka **bukan karangan**: tiap
angka dihitung ulang oleh `sidang/qa/cek_algoritma_tangan.py` lewat dua jalur
(rumus ditulis terbuka dengan loop skalar, dan fungsi asli di
`deteksi_fault_core.py`), lalu dicocokkan dengan toleransi $10^{-9}$. Keluaran
lengkap skrip ditempel di bagian akhir.

Cara menjalankan (dari akar repo):

```
nice -n 19 ./venv/bin/python sidang/qa/cek_algoritma_tangan.py
```

Angka di tabel dibulatkan 6 desimal supaya terbaca; angka kunci ditulis 12
desimal dan itulah yang dicek skrip.

---

## (a) Satu iterasi Levenberg–Marquardt

### a.1 Rumus dari paper dan rumus di kode

**Masalah.** Jaringan dilatih dengan meminimalkan jumlah kuadrat residual

$$E(\theta) = \tfrac12\, r(\theta)^\top r(\theta),\qquad r = \operatorname{vec}(O(\theta) - Y),$$

dengan $O$ keluaran jaringan dan $Y$ target one-hot (kode: `lm_residuals`,
`deteksi_fault_core.py:477-479`; $r$ diurutkan baris-per-baris:
$r = [r_{1,1}, r_{1,2}, r_{2,1}, \dots]$).

**Levenberg (1944)** menambahkan $\lambda I$ ke persamaan Gauss–Newton.
**Marquardt (1963)** mengganti $I$ dengan diagonal $J^\top J$ supaya langkah
tidak tergantung skala tiap parameter:

$$\big(J^\top J + \lambda\,\operatorname{diag}(J^\top J)\big)\,\Delta = -J^\top r .$$

Kode: `deteksi_fault_core.py:536-539`
(`d = np.diag(A)`, dijepit minimal $10^{-12}$, lalu
`np.linalg.solve(A + lam * np.diag(d), -g)`). Penerapan LM ke jaringan saraf
dengan Jacobian dari backpropagation mengikuti **Hagan & Menhaj (1994)**.

**Rasio gain dan aturan Nielsen** (Nielsen 1999; Madsen, Nielsen & Tingleff
2004, "Methods for Non-Linear Least Squares Problems", Alg. 3.16):

$$\rho = \frac{E(\theta) - E(\theta+\Delta)}{L(0) - L(\Delta)},\qquad
L(0)-L(\Delta) = \tfrac12\,\Delta^\top\big(\lambda D\Delta - g\big),\quad g = J^\top r,\; D=\operatorname{diag}(d).$$

Jika $\rho>0$: terima, $\lambda \leftarrow \lambda\cdot\max\!\big(\tfrac13,\,1-(2\rho-1)^3\big)$, $\nu\leftarrow 2$.
Jika $\rho\le 0$: tolak, $\lambda \leftarrow \lambda\,\nu$, $\nu \leftarrow 2\nu$.

Kode: `deteksi_fault_core.py:546-563`.

**Perbedaan kode vs buku (jujur):**

1. Uji terima di kode adalah $E_\text{baru} < E$ (baris 552), bukan $\rho>0$.
   Keduanya **setara** di sini, karena penyebut selalu positif:
   dari $(A+\lambda D)\Delta = -g$ diperoleh
   $\lambda D\Delta - g = A\Delta + 2\lambda D\Delta$, sehingga
   $L(0)-L(\Delta) = \tfrac12(\Delta^\top A\Delta + 2\lambda\Delta^\top D\Delta) > 0$.
   Cabang `else lam * 2.0` di baris 560 (saat $\rho\le 0$ tetapi galat turun)
   praktis tidak pernah terjadi.
2. Saat ditolak, kode **selalu** mengalikan $\lambda$ dengan 2 (baris 562).
   Nielsen menggandakan $\nu$ tiap penolakan beruntun (×2, ×4, ×8, …). Akibatnya
   penolakan beruntun lebih lama keluar dari daerah $\lambda$ kecil — terlihat di
   contoh di bawah: 7 penolakan berturut-turut (Nielsen asli butuh 3–4).
3. Skala $\operatorname{diag}(J^\top J)$ (Marquardt) dipadukan dengan aturan
   $\lambda$ Nielsen yang aslinya dirumuskan untuk $\mu I$. Ini kombinasi yang
   lazim (lihat Transtrum & Sethna 2012), tetapi bukan resep satu paper.
4. Pengaman tambahan: $d_j \ge 10^{-12}$ (kolom Jacobian nol), $\lambda \in
   [10^{-12}, 10^{12}]$, $\lambda\times10$ bila `solve` gagal/hasil tak hingga,
   berhenti bila $\lVert g\rVert < 10^{-10}$, bila penurunan relatif
   $\le 10^{-12}$, bila iterasi habis, atau bila waktu latih > `max_train_s`.
5. **Iterasi yang ditolak tetap dihitung sebagai iterasi** (loop `for k in
   range(iters)`), dan Jacobian dihitung ulang walau $\theta$ tidak berubah.
   Dengan `LM_ITERS_SEARCH = 6` saat grid search, model bisa saja selesai tanpa
   satu langkah pun diterima (contoh di bawah butuh 8 iterasi sampai langkah
   pertama diterima).
6. Lapisan keluaran **linear** dan dilatih dengan kuadrat galat terhadap one-hot
   (gaya Hagan–Menhaj). Softmax **hanya** dipasang saat prediksi
   (`lm_predict_proba`, baris 570-575) — tidak ikut dilatih, jadi "probabilitas"
   tidak terkalibrasi, tetapi argmax-nya sama dengan argmax keluaran linear.

### a.2 Data mainan

Jaringan 2 masukan – 2 tersembunyi (tanh) – 2 keluaran (linear), 3 sampel:

$$X=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\qquad
Y=\begin{bmatrix}1&0\\0&1\\1&0\end{bmatrix}\;(\text{kelas }0,1,0).$$

Bobot awal **persis** seperti `lm_fit(X, Y, 2, [2], 2, seed=0)`:
`lm_init_layers` (Xavier/Glorot 2010, $U(-\ell,\ell)$, $\ell=\sqrt{6/(n_\text{in}+n_\text{out})}=\sqrt{6/4}=1{,}224745$)
memberi

$$W_1^\text{niat}=\begin{bmatrix}0.335486&-0.563905\\-1.124381&-1.184261\end{bmatrix},\;
W_2^\text{niat}=\begin{bmatrix}0.767352&1.011041\\0.261203&0.562149\end{bmatrix},\; b_1=b_2=0.$$

`lm_pack` menyusun $\theta_0$ = [semua W, lalu semua b] =
`[0.335486, -0.563905, -1.124381, -1.184261, 0.767352, 1.011041, 0.261203, 0.562149, 0, 0, 0, 0]`.
`lm_unpack` membaca berselang $W_1, b_1, W_2, b_2$, sehingga jaringan yang
**sungguh dipakai** adalah

$$W_1=\begin{bmatrix}0.335486&-0.563905\\-1.124381&-1.184261\end{bmatrix},\;
b_1=[0.767352,\;1.011041],\;
W_2=\begin{bmatrix}0.261203&0.562149\\0&0\end{bmatrix},\;
b_2=[0,\;0].$$

Ini cacat `lm_pack`/`lm_unpack` (dibahas lengkap di `04_walkthrough_kode.md`,
bagian "Cacat yang diketahui"). Di contoh ini akibatnya terlihat jelas: baris
kedua $W_2$ nol, jadi unit tersembunyi ke-2 tidak berpengaruh ke keluaran pada
awal latihan.

Urutan parameter (kolom Jacobian) = urutan `lm_unpack`:
$[W_{1,00}, W_{1,01}, W_{1,10}, W_{1,11}, b_{1,0}, b_{1,1}, W_{2,00}, W_{2,01}, W_{2,10}, W_{2,11}, b_{2,0}, b_{2,1}]$.

### a.3 Maju (forward)

$z_1 = XW_1 + b_1$, $h=\tanh(z_1)$, $o = hW_2 + b_2$. Contoh sampel 1
($x=(1,0)$): $z_1 = (0.335486+0.767352,\; -0.563905+1.011041) = (1.102838,\;0.447135)$,
$h = (\tanh 1.102838, \tanh 0.447135) = (0.801516, 0.419542)$,
$o_1 = 0.801516\cdot0.261203 + 0.419542\cdot 0 = 0.209359$,
$o_2 = 0.801516\cdot0.562149 = 0.450572$.

| sampel | $z_1$ | $h=\tanh z_1$ | $o$ | $r = o - y$ |
|---|---|---|---|---|
| 1 | (1.102838, 0.447135) | (0.801516, 0.419542) | (0.209359, 0.450572) | (−0.790641, 0.450572) |
| 2 | (−0.357028, −0.173220) | (−0.342594, −0.171508) | (−0.089487, −0.192589) | (−0.089487, −1.192589) |
| 3 | (−0.021542, −0.737125) | (−0.021539, −0.627405) | (−0.005626, −0.012108) | (−1.005626, −0.012108) |

$$E_0 = \tfrac12\sum r^2 = \mathbf{1.634917719156}.$$

### a.4 Jacobian analitik lewat backpropagation

Karena keluaran linear, turunan residual $r_{ic}$ (sampel $i$, keluaran $c$):

$$\frac{\partial r_{ic}}{\partial b_{2,q}} = [q=c],\quad
\frac{\partial r_{ic}}{\partial W_{2,pq}} = h_{ip}[q=c],\quad
\frac{\partial r_{ic}}{\partial b_{1,q}} = W_{2,qc}\,(1-h_{iq}^2),\quad
\frac{\partial r_{ic}}{\partial W_{1,pq}} = x_{ip}\,W_{2,qc}\,(1-h_{iq}^2).$$

Di kode (`lm_residual_jacobian`, baris 482-508) ini dikerjakan sekaligus untuk
semua keluaran: $\delta$ awal = identitas $I_{n_\text{out}}$ per sampel (baris
487, karena $\partial o_c/\partial o_q = [q=c]$), blok $W$ =
$a_\text{prev}\otimes\delta$ (`einsum "ip,icq->icpq"`, baris 493), blok $b$ =
$\delta$, lalu $\delta \leftarrow (\delta W^\top)\odot(1-h^2)$ (baris 497-500;
untuk ReLU turunannya $[h>0]$).

Contoh satu angka: $J_{(1,1),W_{1,00}} = x_{10}\,W_{2,00}(1-h_{10}^2) =
1\cdot0.261203\cdot(1-0.801516^2) = 0.261203\cdot0.357572 = 0.093399$.

$J$ ($6\times12$, baris = $r_{11}, r_{12}, r_{21}, r_{22}, r_{31}, r_{32}$):

| | W1_00 | W1_01 | W1_10 | W1_11 | b1_0 | b1_1 | W2_00 | W2_01 | W2_10 | W2_11 | b2_0 | b2_1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r11 | 0.093399 | 0 | 0 | 0 | 0.093399 | 0 | 0.801516 | 0 | 0.419542 | 0 | 1 | 0 |
| r12 | 0.201009 | 0 | 0 | 0 | 0.201009 | 0 | 0 | 0.801516 | 0 | 0.419542 | 0 | 1 |
| r21 | 0 | 0 | 0.230546 | 0 | 0.230546 | 0 | −0.342594 | 0 | −0.171508 | 0 | 1 | 0 |
| r22 | 0 | 0 | 0.496170 | 0 | 0.496170 | 0 | 0 | −0.342594 | 0 | −0.171508 | 0 | 1 |
| r31 | 0.261082 | 0 | 0.261082 | 0 | 0.261082 | 0 | −0.021539 | 0 | −0.627405 | 0 | 1 | 0 |
| r32 | 0.561889 | 0 | 0.561889 | 0 | 0.561889 | 0 | 0 | −0.021539 | 0 | −0.627405 | 0 | 1 |

Kolom $W_{1,01}, W_{1,11}, b_{1,1}$ nol karena $W_{2,1\cdot}=0$ (akibat cacat
pack/unpack). Itu juga contoh nyata **mengapa** kode menjepit
$d_j=\max(d_j,10^{-12})$: tanpa itu baris/kolom nol membuat
$A+\lambda D$ singular.

### a.5 Iterasi 1 (ditolak)

$g = J^\top r$:
`[-0.252630, 0, -0.881712, 0, -0.864988, 0, -0.581394, 0.769975, 0.314576, 0.401169, -1.885754, -0.754125]`,
$\lVert g\rVert = \mathbf{2.627734635861}$.

$A = J^\top J$ (lengkap di keluaran skrip), diagonalnya
$d = $ `[0.433011, 1e-12, 0.683218, 1e-12, 0.732346, 1e-12, 0.760263, 0.760263, 0.599067, 0.599067, 3, 3]`.

Dengan $\lambda = 10^{-3}$ (`lam0`), selesaikan $(A + \lambda D)\Delta=-g$:
$\Delta$ = `[-0.781785, 0, 0.618051, 0, 0.597494, 0, 1.534301, -1.319635, -1.285292, 0.862243, 0.116816, 0.282593]`
(sisa persamaan $\le 2{,}2\times10^{-16}$).

Residual percobaan $r(\theta_0+\Delta)$ =
`[-0.120300, 0.095003, 1.586013, -1.392115, 0.624098, -0.554071]`,
$E_\text{baru} = \mathbf{2.586706041095} > E_0$.

$L(0)-L(\Delta) = \tfrac12\Delta^\top(\lambda D\Delta - g) = 1.634907423098$,
$\rho = (1.634917719156 - 2.586706041095)/1.634907423098 = \mathbf{-0.582166493645}$.

**Ditolak**: $\theta$ tetap, $\lambda \leftarrow 2\lambda = 0.002$. Langkah
Gauss–Newton hampir murni ($\lambda$ kecil) terlalu jauh untuk masalah
nonlinear ini; 6 residual vs 12 parameter juga berarti $J^\top J$ singular
(rank $\le 6$), hanya $\lambda D$ yang membuatnya dapat dibalik.

### a.6 Iterasi 2–9

| it | $\lambda$ dipakai | $E$ | $E$ percobaan | $L(0)-L(\Delta)$ | $\rho$ | terima | $\lambda$ baru |
|---|---|---|---|---|---|---|---|
| 1 | 0.001 | 1.634917719 | 2.586706041 | 1.634907423 | −0.582166 | tidak | 0.002 |
| 2 | 0.002 | 1.634917719 | 2.570539124 | 1.634876917 | −0.572289 | tidak | 0.004 |
| 3 | 0.004 | 1.634917719 | 2.538810320 | 1.634757500 | −0.552922 | tidak | 0.008 |
| 4 | 0.008 | 1.634917719 | 2.477672606 | 1.634299728 | −0.515667 | tidak | 0.016 |
| 5 | 0.016 | 1.634917719 | 2.363942083 | 1.632613779 | −0.446538 | tidak | 0.032 |
| 6 | 0.032 | 1.634917719 | 2.165689337 | 1.626843925 | −0.326258 | tidak | 0.064 |
| 7 | 0.064 | 1.634917719 | 1.856812411 | 1.609412045 | −0.137873 | tidak | 0.128 |
| 8 | 0.128 | 1.634917719 | 1.453046600 | 1.565691880 | +0.116160 | ya | 0.185909510 |
| 9 | 0.185910 | 1.453046600 | 0.194302037 | 1.381334784 | +0.911252 | ya | 0.082462782 |

Iterasi 8 diterima tetapi $\rho$ kecil: faktor Nielsen
$1-(2\cdot0.116160-1)^3 = 1-(-0.767680)^3 = 1.452418 > 1$, jadi $\lambda$
justru **naik** ($0.128\times1.452418 = 0.185910$) — model "kurang percaya"
pada pendekatan kuadratnya.

**Iterasi 9 rinci (diterima, $\rho$ besar):**
$\lVert g\rVert = 4.094388793528$;
$\Delta$ = `[0.280035, -0.557554, -0.171412, 0.682996, -0.107852, 0.304873, -0.111151, 0.208397, -0.026669, -0.189804, -0.062157, 0.127196]`;
$E = 1.453046600031 \to E_\text{baru} = \mathbf{0.194302037423}$;
$L(0)-L(\Delta) = \mathbf{1.381334783791}$;
$\rho = (1.453046600031-0.194302037423)/1.381334783791 = \mathbf{0.911252346194}$;
faktor $=\max(1/3,\,1-(2\cdot0.911252-1)^3) = \mathbf{0.443564086033}$;
$\lambda_\text{baru} = 0.185909509851\times0.443564086033 = \mathbf{0.082462781822}$.

Sesudah iterasi 9, `predict_proba` (softmax keluaran linear) =
`[[0.698230, 0.301770], [0.418494, 0.581506], [0.693228, 0.306772]]` → kelas
`[0, 1, 0]` = target.

Skrip mencocokkan $\theta$ sesudah tiap iterasi dengan `lm_fit(..., iters=k)`
dan $\lambda$ dengan `hist["lam"]` untuk $k=1..9$: selisih 0.

---

## (b) Fuzzy entropy dengan jarak Euclidean (EDM-Fuzzy)

### b.1 Rumus paper

**Costa et al. (2002, 2005) — multiskala (coarse-graining):**

$$y^{(\tau)}_j = \frac1\tau\sum_{i=(j-1)\tau+1}^{j\tau} x_i,\qquad j = 1..\lfloor N/\tau\rfloor,$$

dengan toleransi $r$ **tetap** dari deret asli untuk semua skala.

**Composite MSE (Wu et al. 2013)** — dipakai kode: untuk skala $\tau$ dibuat
$\tau$ deret bergeser $k=0..\tau-1$:
$y^{(\tau,k)}_j = \frac1\tau\sum_{i=0}^{\tau-1} x_{k+(j-1)\tau+i}$
(`coarse_grain_cmse`, baris 267-271), dan entropi skala $\tau$ = rata-rata
entropi tiap $k$ (baris 311-313).

**Chen et al. (2007), FuzzyEn:**

1. vektor $X^m_i = [u_i,\dots,u_{i+m-1}] - \bar u_i$ (dikurangi rata-ratanya sendiri), $i=1..N-m$;
2. jarak $d^m_{ij} = \max_k |X^m_i(k) - X^m_j(k)|$ (Chebyshev);
3. keanggotaan $D^m_{ij} = \exp\!\big(-(d^m_{ij})^n/r\big)$;
4. $\phi^m = \frac1{N-m}\sum_i \frac1{N-m-1}\sum_{j\ne i} D^m_{ij}$;
5. $\mathrm{FuzzyEn}(m,n,r) = \ln\phi^m - \ln\phi^{m+1}$.

### b.2 Rumus di kode dan perbedaannya

`edm_fuzzy_entropy_1d` (baris 300-314) + `phi_euclidean` (286-297):

$$d_{ij} = \lVert X_i - X_j\rVert_2,\qquad \mu_{ij} = \frac{1}{1 + d_{ij}/r},\qquad
\phi^m = \frac1{|R|}\sum_{i\in R}\frac1{N_m-1}\sum_{j\ne i}\mu_{ij},$$
$$\mathrm{FE}_\tau = \frac1{|K|}\sum_{k\in K}\ln\frac{\phi^m_k}{\phi^{m+1}_k},\qquad r = 0.2\cdot\operatorname{sd}(x,\text{ddof}=1).$$

| Hal | Chen 2007 / Costa | Kode | Baris |
|---|---|---|---|
| Jarak | Chebyshev (maks) | **Euclidean** (inilah "EDM") | 292-294 |
| Keanggotaan | $\exp(-d^n/r)$ | **$1/(1+d/r)$** (rasional, ekor lebih panjang, tak pernah 0) | 295 |
| Jumlah vektor | $N-m$ untuk $m$ **dan** $m+1$ | semua: $N-m+1$ untuk $m$, $N-m$ untuk $m+1$ | 277 |
| Titik acuan $i$ | semua | **subsampel acak `n_ref=128`** bila $N_m>128$; $j$ tetap semua | 281-283, 290-291 |
| Acuan $m$ vs $m+1$ | sama | **seed berbeda** (`seed+11s+k` vs `seed+17s+k`) | 309-310 |
| Coarse-graining | Costa, $k=0$ | composite, rata-rata $\ln$ rasio atas $k$ (CMSE, bukan RCMSE) | 304-313 |
| $r$ | tetap dari deret asli | sama (dihitung sekali dari jendela) | 302 |
| Panjang minimum | – | offset dilewati bila $\lvert y\rvert < m+2$; $\phi=$NaN bila vektor $<3$ | 307, 288 |

Diagonal $\mu_{ii}$ di-nol-kan (baris 296) supaya pasangan diri sendiri tidak
ikut dihitung — setara $j\ne i$ di Chen.

### b.3 Data mainan

$x = [3, 5, 4, 6, 5, 7, 6, 4]$ ($N=8$), $\bar x = 5$,
$\operatorname{sd} = \sqrt{12/7} = 1.309307341416$,
$r = 0.2\cdot\operatorname{sd} = \mathbf{0.261861468283}$.

### b.4 Skala $\tau = 1$

Coarse-graining $\tau=1$ = deret itu sendiri.

**Embedding $m=2$, dipusatkan** ($[a,b] - \frac{a+b}2$):

| $i$ | mentah | terpusat |
|---|---|---|
| 1 | [3, 5] | [−1, 1] |
| 2 | [5, 4] | [0.5, −0.5] |
| 3 | [4, 6] | [−1, 1] |
| 4 | [6, 5] | [0.5, −0.5] |
| 5 | [5, 7] | [−1, 1] |
| 6 | [7, 6] | [0.5, −0.5] |
| 7 | [6, 4] | [1, −1] |

Hanya ada tiga pola, jadi jaraknya hanya:
$\lVert[-1,1]-[0.5,-0.5]\rVert = \sqrt{1.5^2+1.5^2} = 2.121320$;
$\lVert[-1,1]-[1,-1]\rVert = \sqrt{8} = 2.828427$;
$\lVert[0.5,-0.5]-[1,-1]\rVert = \sqrt{0.5} = 0.707107$; pola sama: 0.

Keanggotaan $\mu = 1/(1+d/r)$:
$d=2.121320 \Rightarrow 1/(1+8.100925) = 0.109879$;
$d=2.828427 \Rightarrow 0.084737$;
$d=0.707107 \Rightarrow 0.270248$; $d=0 \Rightarrow 1$.

Matriks $\mu$ ($7\times7$, diagonal 0):

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | rata-rata baris $\sum_{j\ne i}\mu_{ij}/6$ |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.109879 | 1 | 0.109879 | 1 | 0.109879 | 0.084737 | 0.402396 |
| 2 | 0.109879 | 0 | 0.109879 | 1 | 0.109879 | 1 | 0.270248 | 0.433314 |
| 3 | 1 | 0.109879 | 0 | 0.109879 | 1 | 0.109879 | 0.084737 | 0.402396 |
| 4 | 0.109879 | 1 | 0.109879 | 0 | 0.109879 | 1 | 0.270248 | 0.433314 |
| 5 | 1 | 0.109879 | 1 | 0.109879 | 0 | 0.109879 | 0.084737 | 0.402396 |
| 6 | 0.109879 | 1 | 0.109879 | 1 | 0.109879 | 0 | 0.270248 | 0.433314 |
| 7 | 0.084737 | 0.270248 | 0.084737 | 0.270248 | 0.084737 | 0.270248 | 0 | 0.177492 |

$\phi^2 = (3\cdot0.402396 + 3\cdot0.433314 + 0.177492)/7 = \mathbf{0.383517345294}$.

**Embedding $m+1=3$, dipusatkan** (6 vektor):
$[-1,1,0]$ (i=1,3,5), $[0,-1,1]$ (i=2,4), $[1.333333, 0.333333, -1.666667]$ (i=6, dari [7,6,4]).
Jarak: $\lVert[-1,1,0]-[0,-1,1]\rVert=\sqrt6=2.449490$;
$\lVert[-1,1,0]-v_6\rVert = 2.943920$; $\lVert[0,-1,1]-v_6\rVert = 3.265986$.
$\mu$: 0.096580; 0.081684; 0.074227.
Rata-rata baris ($/5$): 0.454969 (i=1,3,5), 0.272793 (i=2,4), 0.078701 (i=6).

$\phi^3 = (3\cdot0.454969+2\cdot0.272793+0.078701)/6 = \mathbf{0.331532298319}$.

$$\mathrm{FE}_{\tau=1} = \ln\frac{0.383517345294}{0.331532298319} = \mathbf{0.145659613004}.$$

### b.5 Skala $\tau = 2$

- $k=0$: $y = [\tfrac{3+5}2, \tfrac{4+6}2, \tfrac{5+7}2, \tfrac{6+4}2] = [4, 5, 6, 5]$ (panjang 4 $\ge m+2$, dipakai).
  Vektor $m=2$ terpusat: $[-0.5,0.5], [-0.5,0.5], [0.5,-0.5]$; jarak $0, 1.414214$;
  $\mu$: $1$ dan $1/(1+1.414214/0.261861)=0.156235$;
  rata-rata baris $0.578117, 0.578117, 0.156235$;
  $\phi^2 = \mathbf{0.437489954604}$.
  Vektor $m=3$: hanya **2** vektor → `phi_euclidean` mengembalikan NaN (baris 288-289, butuh $\ge3$).
- $k=1$: $y = [4.5, 5.5, 6.5]$ (panjang 3 $< 4$) → **dilewati** (baris 307).

Tak ada $k$ yang sah → $\mathrm{FE}_{\tau=2} = $ NaN. Kode memberi
`edm_fuzzy_entropy_1d(x, [1,2]) = [0.145660, nan]`. NaN ini nanti diisi
median kolom oleh `sanitize` (baris 368-372; contoh skrip:
`[[0.3,1],[0.5,NaN],[0.1,3]] -> [[0.3,1],[0.5,2],[0.1,3]]`).

**Syarat panjang.** Skala $\tau$ sah bila $\lfloor (n-k)/\tau\rfloor \ge m+3 = 5$
untuk paling tidak satu $k$, yaitu $n \ge 5\tau$. Untuk $\tau=15$ perlu
$n\ge75$; jendela penelitian $n = 200/700/1000$ aman — NaN hanya muncul di contoh
mainan ini.

**`n_ref`.** Di contoh mainan jumlah vektor $<128$ sehingga semua dipakai.
Untuk jendela $N=200$, skala 1 punya 199 vektor, jadi hanya 128 titik acuan acak
yang dihitung barisnya (biaya $O(128\cdot N)$, bukan $O(N^2)$). Contoh di skrip:
39 vektor, $\phi$ penuh 0.331980 vs $\phi$ dengan `n_ref=10` 0.339608 —
perkiraan acak, bergantung `seed`.

---

## (c) Fuzzy entropy dengan kemiripan Jensen–Shannon (JSD-Fuzzy)

### c.1 Rumus

**Lin (1991):** untuk distribusi $P,Q$ dengan $M=\tfrac12(P+Q)$,

$$\mathrm{JS}(P,Q) = H(M) - \tfrac12H(P) - \tfrac12H(Q) = \tfrac12\mathrm{KL}(P\Vert M) + \tfrac12\mathrm{KL}(Q\Vert M),$$

dengan logaritma natural $0 \le \mathrm{JS} \le \ln 2$. $\sqrt{\mathrm{JS}}$
adalah metrik (Endres & Schindelin 2003).

**Di kode** (rancangan penelitian ini, bukan rumus satu paper):

1. `jsd_fuzzy_entropy_1d` (346-365): pusat bin
   $c_b = \text{linspace}(-3\,\mathrm{sd}, 3\,\mathrm{sd}, 8)$, lebar kernel
   $h = c_2-c_1 = 6\,\mathrm{sd}/7$, ambang $r_\text{JSD} = 0.2\sqrt{\ln2}$
   (20 % dari jarak JS maksimum $\sqrt{\ln 2}$).
2. `vectors_to_distributions` (317-321): tiap vektor terpusat $v$ (panjang $m$)
   diubah jadi histogram lunak
   $K_b = \sum_{t=1}^{m}\exp\!\big(-\tfrac12((v_t-c_b)/h)^2\big)$,
   $P_b = K_b/\sum_{b'}K_{b'}$.
3. `pairwise_jsd` (324-331): JS dengan $\ln$, $\epsilon=10^{-12}$ di dalam log.
4. `phi_jsd` (334-343): $d_{ij} = \sqrt{\mathrm{JS}(P_i,P_j)}$,
   $\mu_{ij} = 1/(1+d_{ij}/r_\text{JSD})$, lalu $\phi$ sama seperti EDM
   (diagonal 0, subsampel `n_ref`).
5. Entropi $= \ln(\phi^m/\phi^{m+1})$, coarse-graining dan syarat panjang sama
   dengan EDM.

**Perbedaan penting terhadap EDM / Chen:**

- Histogram **membuang urutan** nilai di dalam vektor: $[-a, a]$ (naik) dan
  $[a,-a]$ (turun) punya distribusi identik → JS $=0$, $\mu=1$. Skrip:
  JSD$([-0.5,0.5],[0.5,-0.5]) = 0$ sedangkan Euclid $=1.414214$.
- $\phi^{m+1}$ **bisa lebih besar** dari $\phi^m$, sehingga "entropi" JSD bisa
  **negatif** (terjadi di contoh ini). Itu bukan galat hitung; artinya
  histogram 3 titik lebih seragam antarvektor daripada histogram 2 titik.
- Pusat bin bergantung sd **jendela**, jadi skala histogram ikut menyesuaikan
  tiap jendela (sama seperti $r$ di EDM).
- Argumen `r_ratio` pada `jsd_fuzzy_entropy_1d` **tidak dipakai** (parameter mati).

### c.2 Hitungan

Deret sama: $\mathrm{sd}=1.309307341416$.
Pusat bin $c$ = `[-3.927922, -2.805659, -1.683395, -0.561132, 0.561132, 1.683395, 2.805659, 3.927922]`,
$h = \mathbf{1.122263435499}$, $r_\text{JSD} = 0.2\sqrt{0.693147} = \mathbf{0.166510922232}$.

**Vektor → distribusi ($m=2$).** Contoh $v_1=[-1,1]$, bin ke-4 ($c=-0.561132$):
$\exp(-\tfrac12((-1+0.561132)/1.122263)^2) + \exp(-\tfrac12((1+0.561132)/1.122263)^2)
= \exp(-0.076463) + \exp(-0.967519) = 0.926388 + 0.380025 = 1.306412$.

| pola | $K$ (8 bin) | $\sum K$ | $P$ |
|---|---|---|---|
| $[\mp1,\pm1]$ (i=1,3,5,7) | 0.033329, 0.277260, 0.888117, 1.306412, 1.306412, 0.888117, 0.277260, 0.033329 | 5.010236 | 0.006652, 0.055339, 0.177261, 0.260749, 0.260749, 0.177261, 0.055339, 0.006652 |
| $[\pm0.5,\mp0.5]$ (i=2,4,6) | 0.009837, 0.134248, 0.724213, 1.638054, 1.638054, 0.724213, 0.134248, 0.009837 | 5.012704 | 0.001962, 0.026782, 0.144475, 0.326781, 0.326781, 0.144475, 0.026782, 0.001962 |

Vektor 7 ($[1,-1]$) punya $P$ sama dengan vektor 1 — urutan hilang.

**JSD satu pasangan ($P_1$, $P_2$).**
$M = \tfrac12(P_1+P_2)$ = `[0.004307, 0.041060, 0.160868, 0.293765, 0.293765, 0.160868, 0.041060, 0.004307]`;
$\mathrm{KL}(P_1\Vert M) = 0.011040055742$, $\mathrm{KL}(P_2\Vert M) = 0.012581848500$;
$\mathrm{JS} = \tfrac12(0.011040055742+0.012581848500) = \mathbf{0.011810952121}$
(bentuk Lin $H(M)-\tfrac12(H(P_1)+H(P_2))$ memberi angka sama).
$d = \sqrt{0.011810952} = 0.108678$, $\mu = 1/(1+0.108678/0.166511) = 0.605078$.

Matriks $\mu$ ($m=2$): pasangan pola sama → 1, beda pola → 0.605078.
Rata-rata baris: 0.802539 (i=1,3,5,7; $=(3\cdot1+3\cdot0.605078)/6$),
0.736719 (i=2,4,6; $=(2\cdot1+4\cdot0.605078)/6$).

$\phi^2 = (4\cdot0.802539+3\cdot0.736719)/7 = \mathbf{0.774330360139}$.

**$m+1=3$.** Vektor $[-1,1,0]$ dan $[0,-1,1]$ berisi nilai yang sama
$\{-1,0,1\}$ → distribusi identik
`[0.004725, 0.042731, 0.161342, 0.291203, 0.291203, 0.161342, 0.042731, 0.004725]`;
$v_6=[1.333,0.333,-1.667]$ →
`[0.017603, 0.082431, 0.163345, 0.211073, 0.254277, 0.193102, 0.068174, 0.009994]`.
JS$(\cdot, v_6) = 0.011685$, $d=0.108097$, $\mu=0.606358$.
Rata-rata baris: 0.921272 (i=1..5; $=(4+0.606358)/5$), 0.606358 (i=6).

$\phi^3 = (5\cdot0.921272+0.606358)/6 = \mathbf{0.868786036155}$.

$$\mathrm{JSDFE}_{\tau=1} = \ln\frac{0.774330360139}{0.868786036155} = \mathbf{-0.115098272073}.$$

$\tau=2$: sama seperti EDM, $\phi^{m+1}$ NaN → `jsd_fuzzy_entropy_1d(x,[1,2]) = [-0.115098, nan]`.

**Bandingkan:** EDM melihat 3 pola berbeda pada $m=2$ (naik besar, turun kecil,
turun besar) dan memberi entropi positif 0.1457; JSD hanya melihat 2 pola
(amplitudo 1 dan 0.5, arah diabaikan), dan pada $m=3$ hampir semua vektor
dianggap kembar, sehingga entropinya negatif.

---

## (d) Self-check Jacobian: analitik vs numerik

**Ide.** Turunan analitik (backprop) mudah salah indeks. Cek dengan selisih
hingga terpusat:

$$J^\text{num}_{:,p} = \frac{r(\theta + \varepsilon e_p) - r(\theta - \varepsilon e_p)}{2\varepsilon},\quad \varepsilon=10^{-6},$$

galat pemotongan $O(\varepsilon^2)\approx10^{-12}$, galat pembulatan
$\approx 10^{-16}/\varepsilon = 10^{-10}$. Jadi selisih $\sim10^{-10}$ berarti
**cocok**.

Notebook 11 melakukan hal yang sama di sel §10a-check
(`build_11_annlm_real_notebook.py:1585-1599`) pada jaringan 60-[16,8]-3 dengan
selisih **maju** ($\varepsilon=10^{-6}$, mode `"numeric"`, baris 1469-1477) dan
ambang galat relatif $10^{-3}$; notebook 15 mengulanginya untuk [6] dan [6,3]
(`build_15_data_hipotetik_ann_lm_notebook.py:468-479`).

Hasil skrip pada jaringan mainan:

| aktivasi | titik $\theta$ | maks $\lvert J^\text{analitik}-J^\text{num}\rvert$ |
|---|---|---|
| tanh | awal | 8.1e-11 |
| relu | awal | 3.4e-11 |
| tanh | sesudah iterasi 9 | 1.2e-10 |
| relu | sesudah iterasi 9 | 9.6e-11 |

Di $\theta$ sesudah iterasi 9 semua kolom $W_1$ terisi (tak ada baris $W_2$ nol
lagi), jadi cek ini menguji semua jalur backprop, termasuk turunan
$(1-h^2)$ unit tersembunyi kedua.

---

## Daftar pustaka yang dirujuk

- K. Levenberg (1944), Quart. Appl. Math. 2:164-168.
- D. W. Marquardt (1963), J. SIAM 11(2):431-441 — skala $\operatorname{diag}(J^\top J)$.
- H. B. Nielsen (1999), "Damping parameter in Marquardt's method", IMM-REP-1999-05, DTU;
  K. Madsen, H. B. Nielsen, O. Tingleff (2004), "Methods for Non-Linear Least Squares Problems", DTU.
- M. T. Hagan, M. B. Menhaj (1994), IEEE Trans. Neural Networks 5(6):989-993.
- X. Glorot, Y. Bengio (2010), AISTATS — inisialisasi Xavier.
- W. Chen, Z. Wang, H. Xie, W. Yu (2007), IEEE TNSRE 15(2):266-272 — FuzzyEn.
- M. Costa, A. L. Goldberger, C.-K. Peng (2002), PRL 89:068102; (2005) PRE 71:021906 — MSE.
- S.-D. Wu et al. (2013), Entropy 15(3):1069-1084 — composite MSE.
- J. Lin (1991), IEEE Trans. Inf. Theory 37(1):145-151 — Jensen–Shannon divergence.
- D. M. Endres, J. E. Schindelin (2003), IEEE Trans. Inf. Theory 49(7):1858-1860 — $\sqrt{\mathrm{JS}}$ metrik.
- M. K. Transtrum, J. P. Sethna (2012), arXiv:1201.5885 — variasi skala/damping LM.

---

## Keluaran `cek_algoritma_tangan.py` (dijalankan 2026-11-20, numpy 2.5.3)

```
==============================================================================
(a) SATU ITERASI LEVENBERG-MARQUARDT, jaringan 2-[2]-2, 3 sampel
==============================================================================
Xavier W1 (template) =
[[ 0.335486 -0.563905]
 [-1.124381 -1.184261]]
Xavier W2 (template) =
[[0.767352 1.011041]
 [0.261203 0.562149]]
theta0 = lm_pack(template) =
[ 0.335486 -0.563905 -1.124381 -1.184261  0.767352  1.011041  0.261203  0.562149  0.        0.        0.        0.      ]
W1 (hasil lm_unpack) =
[[ 0.335486 -0.563905]
 [-1.124381 -1.184261]]
b1 (hasil lm_unpack) =
[0.767352 1.011041]
W2 (hasil lm_unpack) =
[[0.261203 0.562149]
 [0.       0.      ]]
b2 (hasil lm_unpack) =
[0. 0.]
z1 = X W1 + b1 =
[[ 1.102838  0.447135]
 [-0.357028 -0.17322 ]
 [-0.021542 -0.737125]]
h = tanh(z1) =
[[ 0.801516  0.419542]
 [-0.342594 -0.171508]
 [-0.021539 -0.627405]]
keluaran o = h W2 + b2 =
[[ 0.209359  0.450572]
 [-0.089487 -0.192589]
 [-0.005626 -0.012108]]
residual r = (o - Y).ravel() =
[-0.790641  0.450572 -0.089487 -1.192589 -1.005626 -0.012108]
  [OK] residual tangan vs lm_residuals                maks selisih = 0.0e+00
E0 = 0.5 * sum(r^2) = 1.634917719156
  [OK] E0 vs konstanta .md                            maks selisih = 2.8e-13
J (6 x 12), kolom: W1_00 W1_01 W1_10 W1_11 b1_0 b1_1 W2_00 W2_01 W2_10 W2_11 b2_0 b2_1 =
[[ 0.093399  0.        0.        0.        0.093399  0.        0.801516  0.        0.419542  0.        1.        0.      ]
 [ 0.201009  0.        0.        0.        0.201009  0.        0.        0.801516  0.        0.419542  0.        1.      ]
 [ 0.        0.        0.230546  0.        0.230546  0.       -0.342594 -0.       -0.171508 -0.        1.        0.      ]
 [ 0.        0.        0.49617   0.        0.49617   0.       -0.       -0.342594 -0.       -0.171508  0.        1.      ]
 [ 0.261082  0.        0.261082  0.        0.261082  0.       -0.021539 -0.       -0.627405 -0.        1.        0.      ]
 [ 0.561889  0.        0.561889  0.        0.561889  0.       -0.       -0.021539 -0.       -0.627405  0.        1.      ]]
  [OK] Jacobian tangan vs lm_residual_jacobian        maks selisih = 5.6e-17
--- iterasi 1 (rinci), lam = 0.001000000000 ---
   g = J^T r =
[-0.25263   0.       -0.881712  0.       -0.864988  0.       -0.581394  0.769975  0.314576  0.401169 -1.885754 -0.754125]
   ||g|| = 2.627734635861
   A = J^T J =
[[ 0.433011  0.        0.383883  0.        0.433011  0.        0.069237  0.149009 -0.12462  -0.2682    0.354481  0.762897]
 [ 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      ]
 [ 0.383883  0.        0.683218  0.        0.683218  0.       -0.084607 -0.182087 -0.203345 -0.437629  0.491628  1.058058]
 [ 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      ]
 [ 0.433011  0.        0.683218  0.        0.732346  0.       -0.009746 -0.020975 -0.16416  -0.353297  0.585027  1.259067]
 [ 0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.        0.      ]
 [ 0.069237  0.       -0.084607  0.       -0.009746  0.        0.760263  0.        0.408541  0.        0.437384  0.      ]
 [ 0.149009  0.       -0.182087  0.       -0.020975  0.        0.        0.760263  0.        0.408541  0.        0.437384]
 [-0.12462   0.       -0.203345  0.       -0.16416   0.        0.408541  0.        0.599067  0.       -0.379372  0.      ]
 [-0.2682    0.       -0.437629  0.       -0.353297  0.        0.        0.408541  0.        0.599067  0.       -0.379372]
 [ 0.354481  0.        0.491628  0.        0.585027  0.        0.437384  0.       -0.379372  0.        3.        0.      ]
 [ 0.762897  0.        1.058058  0.        1.259067  0.        0.        0.437384  0.       -0.379372  0.        3.      ]]
   d = diag(A), dijepit >= 1e-12 =
[0.433011 0.       0.683218 0.       0.732346 0.       0.760263 0.760263 0.599067 0.599067 3.       3.      ]
   langkah Delta: (A + lam*diag(d)) Delta = -g =
[-0.781785 -0.        0.618051  0.        0.597494  0.        1.534301 -1.319635 -1.285292  0.862243  0.116816  0.282593]
   cek: maks |(A + lam D) Delta + g| = 2.22e-16
   residual percobaan r(theta + Delta) =
[-0.1203    0.095003  1.586013 -1.392115  0.624098 -0.554071]
   E = 1.634917719156, E_baru = 2.586706041095
   L(0)-L(Delta) = 0.5 Delta^T (lam d*Delta - g) = 1.634907423098
   rho = (E - E_baru)/(L(0)-L(Delta)) = -0.582166493645
   DITOLAK (E_baru >= E): theta tetap, lam dikali 2
  [OK] theta sesudah iterasi 1 vs lm_fit(iters=1)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 1 vs hist['lam'][1]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 2 vs lm_fit(iters=2)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 2 vs hist['lam'][2]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 3 vs lm_fit(iters=3)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 3 vs hist['lam'][3]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 4 vs lm_fit(iters=4)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 4 vs hist['lam'][4]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 5 vs lm_fit(iters=5)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 5 vs hist['lam'][5]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 6 vs lm_fit(iters=6)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 6 vs hist['lam'][6]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 7 vs lm_fit(iters=7)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 7 vs hist['lam'][7]        maks selisih = 0.0e+00
  [OK] theta sesudah iterasi 8 vs lm_fit(iters=8)     maks selisih = 0.0e+00
  [OK] lam sesudah iterasi 8 vs hist['lam'][8]        maks selisih = 0.0e+00
--- iterasi 9 (rinci), lam = 0.185909509851 ---
   g = J^T r =
[ 0.368378 -0.16316   1.696101 -1.689594  1.644899 -1.615544  0.753384 -0.820597 -0.412135  0.428428  1.349901 -1.462166]
   ||g|| = 4.094388793528
   A = J^T J =
[[ 1.532505 -1.012648  1.244351 -0.61227   1.532505 -1.012648  0.856635 -0.288356 -0.449861  0.15143   1.565965 -0.527127]
 [-1.012648  0.91181  -0.61227   0.320315 -1.012648  0.91181  -0.712122  0.457536  0.027279 -0.017527 -1.1232    0.721653]
 [ 1.244351 -0.61227   2.003512 -1.37776   2.003512 -1.37776   0.959255 -0.322899 -0.804929  0.270951  1.882985 -0.63384 ]
 [-0.61227   0.320315 -1.37776   1.141006 -1.37776   1.141006 -0.66826   0.429355  0.429459 -0.275926 -1.23832   0.795617]
 [ 1.532505 -1.012648  2.003512 -1.37776   2.291665 -1.778137  1.356487 -0.456613 -0.591487  0.199103  2.391734 -0.805093]
 [-1.012648  0.91181  -1.37776   1.141006 -1.778137  1.732501 -1.173473  0.753953  0.157997 -0.101513 -1.885365  1.211341]
 [ 0.856635 -0.712122  0.959255 -0.66826   1.356487 -1.173473  1.164883  0.       -0.048871  0.        1.820658  0.      ]
 [-0.288356  0.457536 -0.322899  0.429355 -0.456613  0.753953  0.        1.164883  0.       -0.048871  0.        1.820658]
 [-0.449861  0.027279 -0.804929  0.429459 -0.591487  0.157997 -0.048871  0.        0.599067  0.       -0.379372  0.      ]
 [ 0.15143  -0.017527  0.270951 -0.275926  0.199103 -0.101513  0.       -0.048871  0.        0.599067  0.       -0.379372]
 [ 1.565965 -1.1232    1.882985 -1.23832   2.391734 -1.885365  1.820658  0.       -0.379372  0.        3.        0.      ]
 [-0.527127  0.721653 -0.63384   0.795617 -0.805093  1.211341  0.        1.820658  0.       -0.379372  0.        3.      ]]
   d = diag(A), dijepit >= 1e-12 =
[1.532505 0.91181  2.003512 1.141006 2.291665 1.732501 1.164883 1.164883 0.599067 0.599067 3.       3.      ]
   langkah Delta: (A + lam*diag(d)) Delta = -g =
[ 0.280035 -0.557554 -0.171412  0.682996 -0.107852  0.304873 -0.111151  0.208397 -0.026669 -0.189804 -0.062157  0.127196]
   cek: maks |(A + lam D) Delta + g| = 4.44e-16
   residual percobaan r(theta + Delta) =
[ 0.009152  0.170271  0.09392  -0.577119 -0.075423  0.109324]
   E = 1.453046600031, E_baru = 0.194302037423
   L(0)-L(Delta) = 0.5 Delta^T (lam d*Delta - g) = 1.381334783791
   rho = (E - E_baru)/(L(0)-L(Delta)) = 0.911252346194
   DITERIMA. faktor Nielsen = max(1/3, 1-(2rho-1)^3) = 0.443564086033
  [OK] theta sesudah iterasi 9 vs lm_fit(iters=9)     maks selisih = 0.0e+00
ringkas iterasi 1..9:
 it   lam_dipakai        E            E_percobaan      L(0)-L(D)        rho          terima  lam_baru
  1  0.001000  1.634917719  2.586706041  1.634907423  -0.582166494  tidak  0.002000000
  2  0.002000  1.634917719  2.570539124  1.634876917  -0.572288589  tidak  0.004000000
  3  0.004000  1.634917719  2.538810320  1.634757500  -0.552921519  tidak  0.008000000
  4  0.008000  1.634917719  2.477672606  1.634299728  -0.515667274  tidak  0.016000000
  5  0.016000  1.634917719  2.363942083  1.632613779  -0.446538167  tidak  0.032000000
  6  0.032000  1.634917719  2.165689337  1.626843925  -0.326258475  tidak  0.064000000
  7  0.064000  1.634917719  1.856812411  1.609412045  -0.137873140  tidak  0.128000000
  8  0.128000  1.634917719  1.453046600  1.565691880  +0.116160224  ya     0.185909510
  9  0.185910  1.453046600  0.194302037  1.381334784  +0.911252346  ya     0.082462782
  [OK] E0 vs konstanta .md                            maks selisih = 2.8e-13
  [OK] ||g|| it1 vs konstanta .md                     maks selisih = 4.6e-13
  [OK] E percobaan it1 vs konstanta .md               maks selisih = 2.3e-13
  [OK] rho it1 vs konstanta .md                       maks selisih = 2.3e-13
  [OK] E percobaan it9 vs konstanta .md               maks selisih = 2.8e-13
  [OK] denom it9 vs konstanta .md                     maks selisih = 2.2e-13
  [OK] rho it9 vs konstanta .md                       maks selisih = 4.2e-13
  [OK] faktor it9 vs konstanta .md                    maks selisih = 6.4e-14
  [OK] lam baru it9 vs konstanta .md                  maks selisih = 1.2e-13
softmax(keluaran) sesudah langkah diterima pertama (predict_proba) =
[[0.69823  0.30177 ]
 [0.418494 0.581506]
 [0.693228 0.306772]]
prediksi kelas = [0 1 0] (target [0 1 0])

==============================================================================
(b) EDM-FUZZY: deret 8 titik, m = 2, r = 0.2*sd
==============================================================================
x = [3. 5. 4. 6. 5. 7. 6. 4.]
mean = 5.0, sd(ddof=1) = 1.309307341416, r = 0.2*sd = 0.261861468283
  [OK] r vs konstanta .md                             maks selisih = 1.9e-13
  [OK] coarse-grain tau=1 k=0                         maks selisih = 0.0e+00
tau=1, k=0: y = [3. 5. 4. 6. 5. 7. 6. 4.]  (panjang 8)
  [OK] embed m=2 tau=1 k=0                            maks selisih = 0.0e+00
  [OK] embed m=3 tau=1 k=0                            maks selisih = 0.0e+00
   vektor m=2 terpusat (tau=1) =
[[-1.   1. ]
 [ 0.5 -0.5]
 [-1.   1. ]
 [ 0.5 -0.5]
 [-1.   1. ]
 [ 0.5 -0.5]
 [ 1.  -1. ]]
   vektor m=3 terpusat (tau=1) =
[[-1.        1.        0.      ]
 [ 0.       -1.        1.      ]
 [-1.        1.        0.      ]
 [ 0.       -1.        1.      ]
 [-1.        1.        0.      ]
 [ 1.333333  0.333333 -1.666667]]
   jarak Euclid m=2 (tau=1) =
[[0.       2.12132  0.       2.12132  0.       2.12132  2.828427]
 [2.12132  0.       2.12132  0.       2.12132  0.       0.707107]
 [0.       2.12132  0.       2.12132  0.       2.12132  2.828427]
 [2.12132  0.       2.12132  0.       2.12132  0.       0.707107]
 [0.       2.12132  0.       2.12132  0.       2.12132  2.828427]
 [2.12132  0.       2.12132  0.       2.12132  0.       0.707107]
 [2.828427 0.707107 2.828427 0.707107 2.828427 0.707107 0.      ]]
   keanggotaan mu = 1/(1+d/r), diagonal 0 (tau=1) =
[[0.       0.109879 1.       0.109879 1.       0.109879 0.084737]
 [0.109879 0.       0.109879 1.       0.109879 1.       0.270248]
 [1.       0.109879 0.       0.109879 1.       0.109879 0.084737]
 [0.109879 1.       0.109879 0.       0.109879 1.       0.270248]
 [1.       0.109879 1.       0.109879 0.       0.109879 0.084737]
 [0.109879 1.       0.109879 1.       0.109879 0.       0.270248]
 [0.084737 0.270248 0.084737 0.270248 0.084737 0.270248 0.      ]]
   rata-rata baris m=2 = [0.402396 0.433314 0.402396 0.433314 0.402396 0.433314 0.177492]
   jarak Euclid m=3 (tau=1) =
[[0.       2.44949  0.       2.44949  0.       2.94392 ]
 [2.44949  0.       2.44949  0.       2.44949  3.265986]
 [0.       2.44949  0.       2.44949  0.       2.94392 ]
 [2.44949  0.       2.44949  0.       2.44949  3.265986]
 [0.       2.44949  0.       2.44949  0.       2.94392 ]
 [2.94392  3.265986 2.94392  3.265986 2.94392  0.      ]]
   keanggotaan m=3 (tau=1) =
[[0.       0.09658  1.       0.09658  1.       0.081684]
 [0.09658  0.       0.09658  1.       0.09658  0.074227]
 [1.       0.09658  0.       0.09658  1.       0.081684]
 [0.09658  1.       0.09658  0.       0.09658  0.074227]
 [1.       0.09658  1.       0.09658  0.       0.081684]
 [0.081684 0.074227 0.081684 0.074227 0.081684 0.      ]]
   rata-rata baris m=3 = [0.454969 0.272793 0.454969 0.272793 0.454969 0.078701]
  [OK] phi_m tau=1 k=0                                maks selisih = 5.6e-17
  [OK] phi_m1 tau=1 k=0                               maks selisih = 0.0e+00
   phi_m = 0.383517345294, phi_m+1 = 0.33153229831940895
   ln(phi_m/phi_m+1) = 0.145659613004
  [OK] phi_m tau=1 vs konstanta .md                   maks selisih = 3.9e-13
  [OK] phi_m1 tau=1 vs konstanta .md                  maks selisih = 4.1e-13
  [OK] coarse-grain tau=2 k=0                         maks selisih = 0.0e+00
tau=2, k=0: y = [4. 5. 6. 5.]  (panjang 4)
  [OK] embed m=2 tau=2 k=0                            maks selisih = 0.0e+00
  [OK] embed m=3 tau=2 k=0                            maks selisih = 0.0e+00
   vektor m=2 terpusat (tau=2) =
[[-0.5  0.5]
 [-0.5  0.5]
 [ 0.5 -0.5]]
   vektor m=3 terpusat (tau=2) =
[[-1.        0.        1.      ]
 [-0.333333  0.666667 -0.333333]]
   jarak Euclid m=2 (tau=2) =
[[0.       0.       1.414214]
 [0.       0.       1.414214]
 [1.414214 1.414214 0.      ]]
   keanggotaan mu = 1/(1+d/r), diagonal 0 (tau=2) =
[[0.       1.       0.156235]
 [1.       0.       0.156235]
 [0.156235 0.156235 0.      ]]
   rata-rata baris m=2 = [0.578117 0.578117 0.156235]
  [OK] phi_m tau=2 k=0                                maks selisih = 0.0e+00
  [OK] phi_m1 tau=2 k=0                               maks selisih = 0.0e+00
   phi_m = 0.437489954604, phi_m+1 = nan
   phi_m+1 = NaN (hanya 2 vektor) -> tidak dihitung
  [OK] phi_m tau=2 vs konstanta .md                   maks selisih = 4.1e-14
  [OK] coarse-grain tau=2 k=1                         maks selisih = 0.0e+00
tau=2, k=1: y = [4.5 5.5 6.5]  (panjang 3)
   panjang < m+2 = 4 -> offset ini DILEWATI oleh kode
edm_fuzzy_entropy_1d(x, skala [1,2]) = [0.14566     nan]
  [OK] FE tau=1,2 tangan vs edm_fuzzy_entropy_1d      maks selisih = 1.9e-16
  [OK] FE tau=1 vs konstanta .md                      maks selisih = 1.6e-13
sanitize(contoh matriks fitur dengan NaN) =
[[0.3 1. ]
 [0.5 2. ]
 [0.1 3. ]]
  [OK] sanitize isi NaN dengan median kolom           maks selisih = 0.0e+00
n_ref: 39 vektor, phi penuh = 0.331980, phi dengan n_ref=10 = 0.339608 (perkiraan acak)

==============================================================================
(c) JSD-FUZZY: deret yang sama, m = 2, 8 bin, h = jarak antar pusat
==============================================================================
pusat bin c = [-3.927922 -2.805659 -1.683395 -0.561132  0.561132  1.683395  2.805659  3.927922]
h = 1.122263435499, r_jsd = 0.2*sqrt(ln 2) = 0.166510922232
  [OK] pusat bin vs np.linspace                       maks selisih = 8.9e-16
  [OK] h vs konstanta .md                             maks selisih = 3.9e-13
  [OK] r_jsd vs konstanta .md                         maks selisih = 4.6e-13
   kernel mentah K (m=2), baris = vektor =
[[0.033329 0.27726  0.888117 1.306412 1.306412 0.888117 0.27726  0.033329]
 [0.009837 0.134248 0.724213 1.638054 1.638054 0.724213 0.134248 0.009837]
 [0.033329 0.27726  0.888117 1.306412 1.306412 0.888117 0.27726  0.033329]
 [0.009837 0.134248 0.724213 1.638054 1.638054 0.724213 0.134248 0.009837]
 [0.033329 0.27726  0.888117 1.306412 1.306412 0.888117 0.27726  0.033329]
 [0.009837 0.134248 0.724213 1.638054 1.638054 0.724213 0.134248 0.009837]
 [0.033329 0.27726  0.888117 1.306412 1.306412 0.888117 0.27726  0.033329]]
   distribusi P (m=2), tiap baris jumlah 1 =
[[0.006652 0.055339 0.177261 0.260749 0.260749 0.177261 0.055339 0.006652]
 [0.001962 0.026782 0.144475 0.326781 0.326781 0.144475 0.026782 0.001962]
 [0.006652 0.055339 0.177261 0.260749 0.260749 0.177261 0.055339 0.006652]
 [0.001962 0.026782 0.144475 0.326781 0.326781 0.144475 0.026782 0.001962]
 [0.006652 0.055339 0.177261 0.260749 0.260749 0.177261 0.055339 0.006652]
 [0.001962 0.026782 0.144475 0.326781 0.326781 0.144475 0.026782 0.001962]
 [0.006652 0.055339 0.177261 0.260749 0.260749 0.177261 0.055339 0.006652]]
  [OK] P tangan vs vectors_to_distributions m=2       maks selisih = 1.4e-16
  [OK] JSD tangan vs pairwise_jsd m=2                 maks selisih = 1.7e-18
   JSD (nat) m=2 =
[[0.       0.011811 0.       0.011811 0.       0.011811 0.      ]
 [0.011811 0.       0.011811 0.       0.011811 0.       0.011811]
 [0.       0.011811 0.       0.011811 0.       0.011811 0.      ]
 [0.011811 0.       0.011811 0.       0.011811 0.       0.011811]
 [0.       0.011811 0.       0.011811 0.       0.011811 0.      ]
 [0.011811 0.       0.011811 0.       0.011811 0.       0.011811]
 [0.       0.011811 0.       0.011811 0.       0.011811 0.      ]]
   d = sqrt(JSD) m=2 =
[[0.       0.108678 0.       0.108678 0.       0.108678 0.      ]
 [0.108678 0.       0.108678 0.       0.108678 0.       0.108678]
 [0.       0.108678 0.       0.108678 0.       0.108678 0.      ]
 [0.108678 0.       0.108678 0.       0.108678 0.       0.108678]
 [0.       0.108678 0.       0.108678 0.       0.108678 0.      ]
 [0.108678 0.       0.108678 0.       0.108678 0.       0.108678]
 [0.       0.108678 0.       0.108678 0.       0.108678 0.      ]]
   keanggotaan mu = 1/(1+d/r_jsd) m=2 =
[[0.       0.605078 1.       0.605078 1.       0.605078 1.      ]
 [0.605078 0.       0.605078 1.       0.605078 1.       0.605078]
 [1.       0.605078 0.       0.605078 1.       0.605078 1.      ]
 [0.605078 1.       0.605078 0.       0.605078 1.       0.605078]
 [1.       0.605078 1.       0.605078 0.       0.605078 1.      ]
 [0.605078 1.       0.605078 1.       0.605078 0.       0.605078]
 [1.       0.605078 1.       0.605078 1.       0.605078 0.      ]]
   rata-rata baris = [0.802539 0.736719 0.802539 0.736719 0.802539 0.736719 0.802539]
  [OK] phi_jsd m=2 tangan vs phi_jsd                  maks selisih = 5.6e-16
   phi (m=2) = 0.774330360139
  [OK] JSD(P1,P2) vs konstanta .md                    maks selisih = 6.0e-14
   kernel mentah K (m=3), baris = vektor =
[[0.035516 0.321197 1.21277  2.188909 2.188909 1.21277  0.321197 0.035516]
 [0.035516 0.321197 1.21277  2.188909 2.188909 1.21277  0.321197 0.035516]
 [0.035516 0.321197 1.21277  2.188909 2.188909 1.21277  0.321197 0.035516]
 [0.035516 0.321197 1.21277  2.188909 2.188909 1.21277  0.321197 0.035516]
 [0.035516 0.321197 1.21277  2.188909 2.188909 1.21277  0.321197 0.035516]
 [0.132103 0.61861  1.225826 1.584007 1.908235 1.449143 0.511615 0.075003]]
   distribusi P (m=3), tiap baris jumlah 1 =
[[0.004725 0.042731 0.161342 0.291203 0.291203 0.161342 0.042731 0.004725]
 [0.004725 0.042731 0.161342 0.291203 0.291203 0.161342 0.042731 0.004725]
 [0.004725 0.042731 0.161342 0.291203 0.291203 0.161342 0.042731 0.004725]
 [0.004725 0.042731 0.161342 0.291203 0.291203 0.161342 0.042731 0.004725]
 [0.004725 0.042731 0.161342 0.291203 0.291203 0.161342 0.042731 0.004725]
 [0.017603 0.082431 0.163345 0.211073 0.254277 0.193102 0.068174 0.009994]]
  [OK] P tangan vs vectors_to_distributions m=3       maks selisih = 1.4e-16
  [OK] JSD tangan vs pairwise_jsd m=3                 maks selisih = 3.5e-18
   JSD (nat) m=3 =
[[0.       0.       0.       0.       0.       0.011685]
 [0.       0.       0.       0.       0.       0.011685]
 [0.       0.       0.       0.       0.       0.011685]
 [0.       0.       0.       0.       0.       0.011685]
 [0.       0.       0.       0.       0.       0.011685]
 [0.011685 0.011685 0.011685 0.011685 0.011685 0.      ]]
   d = sqrt(JSD) m=3 =
[[0.       0.       0.       0.       0.       0.108097]
 [0.       0.       0.       0.       0.       0.108097]
 [0.       0.       0.       0.       0.       0.108097]
 [0.       0.       0.       0.       0.       0.108097]
 [0.       0.       0.       0.       0.       0.108097]
 [0.108097 0.108097 0.108097 0.108097 0.108097 0.      ]]
   keanggotaan mu = 1/(1+d/r_jsd) m=3 =
[[0.       1.       1.       1.       1.       0.606358]
 [1.       0.       1.       1.       1.       0.606358]
 [1.       1.       0.       1.       1.       0.606358]
 [1.       1.       1.       0.       1.       0.606358]
 [1.       1.       1.       1.       0.       0.606358]
 [0.606358 0.606358 0.606358 0.606358 0.606358 0.      ]]
   rata-rata baris = [0.921272 0.921272 0.921272 0.921272 0.921272 0.606358]
  [OK] phi_jsd m=3 tangan vs phi_jsd                  maks selisih = 0.0e+00
   phi (m=3) = 0.868786036155
rincian pasangan v1, v2:
   P1 =
[0.006652 0.055339 0.177261 0.260749 0.260749 0.177261 0.055339 0.006652]
   P2 =
[0.001962 0.026782 0.144475 0.326781 0.326781 0.144475 0.026782 0.001962]
   M = (P1+P2)/2 =
[0.004307 0.04106  0.160868 0.293765 0.293765 0.160868 0.04106  0.004307]
   KL(P1||M) = 0.011040055742, KL(P2||M) = 0.012581848500, JSD = 0.011810952121
   bentuk Lin 1991: H(M) - (H(P1)+H(P2))/2 = 0.011810952121
JSD-FuzzyEn tau=1 = ln(0.774330360139/0.868786036155) = -0.115098272073
  [OK] phi_jsd m vs konstanta .md                     maks selisih = 2.9e-13
  [OK] phi_jsd m+1 vs konstanta .md                   maks selisih = 3.3e-13
jsd_fuzzy_entropy_1d(x, skala [1,2]) = [-0.115098       nan]
  [OK] FE tau=1,2 tangan vs jsd_fuzzy_entropy_1d      maks selisih = 6.2e-16
  [OK] JSD-FE tau=1 vs konstanta .md                  maks selisih = 2.8e-13
vektor naik [-0.5, 0.5] vs turun [0.5, -0.5]: JSD = 0.000e+00, Euclid = 1.414214

==============================================================================
(d) SELF-CHECK JACOBIAN: analitik vs numerik (selisih tengah, eps = 1e-6)
==============================================================================
tanh @ theta awal              : maks |J_analitik - J_numerik| = 8.098e-11
relu @ theta awal              : maks |J_analitik - J_numerik| = 3.393e-11
tanh @ theta sesudah iterasi 9 : maks |J_analitik - J_numerik| = 1.203e-10
   J analitik (tanh, theta sesudah iterasi 9) =
[[ 0.351901 -0.781988  0.       -0.        0.351901 -0.781988  0.839537  0.        0.19204   0.        1.        0.      ]
 [-0.067985  0.303119 -0.        0.       -0.067985  0.303119  0.        0.839537  0.        0.19204   0.        1.      ]
 [ 0.       -0.        1.002551 -0.445116  1.002551 -0.445116  0.39881   0.        0.672147  0.        1.        0.      ]
 [-0.        0.       -0.193686  0.172539 -0.193686  0.172539  0.        0.39881   0.        0.672147  0.        1.      ]
 [ 0.966404 -0.740056  0.966404 -0.740056  0.966404 -0.740056  0.435167  0.       -0.297532 -0.        1.        0.      ]
 [-0.186703  0.286865 -0.186703  0.286865 -0.186703  0.286865  0.        0.435167 -0.       -0.297532  0.        1.      ]]
   J numerik (tanh) =
[[ 0.351901 -0.781988  0.        0.        0.351901 -0.781988  0.839537  0.        0.19204   0.        1.        0.      ]
 [-0.067985  0.303119  0.        0.       -0.067985  0.303119  0.        0.839537  0.        0.19204   0.        1.      ]
 [ 0.        0.        1.002551 -0.445116  1.002551 -0.445116  0.39881   0.        0.672147  0.        1.        0.      ]
 [ 0.        0.       -0.193686  0.172539 -0.193686  0.172539  0.        0.39881   0.        0.672147  0.        1.      ]
 [ 0.966404 -0.740056  0.966404 -0.740056  0.966404 -0.740056  0.435167  0.       -0.297532  0.        1.        0.      ]
 [-0.186703  0.286865 -0.186703  0.286865 -0.186703  0.286865  0.        0.435167  0.       -0.297532  0.        1.      ]]
relu @ theta sesudah iterasi 9 : maks |J_analitik - J_numerik| = 9.629e-11

SEMUA 65 CEK LOLOS (toleransi 1e-09).
```
