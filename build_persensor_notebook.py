"""Build archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb — Cara A: label per-sensor multi-output.

Base: 02_Sensor_Digabung_Broker_Jangan_Dirata_rata.ipynb (broker 4ch vs fused) + scenario set
shared with 01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb. Labelling changed from OR-across-sensor
(1 label/window) to per-sensor (4 binary labels/window). 1 MultiOutputClassifier.
"""
import json

def md(src): return {"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

cells = []
cells.append(md("""# Per-Sensor Fault Detection — Broker + Multi-Output Label (Cara A)

Notebook turunan `02_Sensor_Digabung_Broker_Jangan_Dirata_rata.ipynb` + skenario dari
`01_Metode_Mana_Paling_Akurat_EDM_vs_JSD.ipynb`. Menjawab: **apakah bisa menentukan sensor mana yang fault?**

Perubahan dari notebook broker:
- **Labelling**: bukan OR-antar-sensor (1 label/window), tapi **per-sensor** → 1 window
  punya 4 label biner `[S1, S2, S3, S4]` (fault/normal per sensor).
- **Classifier**: 1 `MultiOutputClassifier(MLPClassifier)` → 1 window masuk, 4 keputusan
  keluar (1 per sensor). Tetap 1 keputusan gabungan via broker, tapi sekarang menunjuk
  sensor sumber.
- **Benchmark**: 4-channel (A) vs fused-1ch (B), dievaluasi per-sensor + agregat.

Pipeline lain (broker, fault injection, mask, windowing, entropy EDM/JSD-Fuzzy,
fitur time-domain, ANN grid) identik dengan notebook broker.
"""))

cells.append(code("""# === Runtime guard ===
import os, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
NOTEBOOK_START = time.time()
KAGGLE_TIME_BUDGET_H = float(os.environ.get("KAGGLE_TIME_BUDGET_H", 10.5))
def elapsed_s(): return time.time()-NOTEBOOK_START
def budget_ok(need_s=0.0, label=""):
    left = KAGGLE_TIME_BUDGET_H*3600.0 - elapsed_s()
    if left < need_s:
        print(f"[budget] SKIP {label}: {left/60:.1f} min left"); return False
    return True
def log_stage(x): print(f"[t+{elapsed_s()/60:5.1f} min] {x}", flush=True)
log_stage("guard armed")
"""))

cells.append(code("""# === Imports + config ===
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
import requests
from io import StringIO
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Parallel, delayed
from scipy import stats as _sstats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             roc_auc_score, hamming_loss)

RUNTIME_PROFILE = os.environ.get("RUNTIME_PROFILE","fast")
N_JOBS=-1
DS=4; WIN=256; STRIDE=128; MAX_PER_SCN=200; RANDOM_SEED=42
S=10; scales=np.arange(1,S+1); m=2; r_ratio=0.2; n_ref=128; jsd_bins=40
METHODS=["EDM-Fuzzy","JSD-Fuzzy"]
SENSORS=["S1","S2","S3","S4"]
os.makedirs("exports", exist_ok=True)
print("config: DS",DS,"WIN",WIN,"STRIDE",STRIDE,"MAX_PER_SCN",MAX_PER_SCN,"S",S,"| profile",RUNTIME_PROFILE)
"""))

cells.append(code("""# === Load data (4 sensor) — output broker = satu tabel gabungan ===
def load_default_data():
    url="https://raw.githubusercontent.com/vousmeevoyez/public-files/refs/heads/main/tabel_sensor4_generated.csv"
    r=requests.get(url); r.raise_for_status()
    return pd.read_csv(StringIO(r.text))
df=load_default_data()
cols=["kelembaban1","kelembaban2","kelembaban3","kelembaban4"]
X_df=pd.DataFrame(df[cols].to_numpy(dtype=float),columns=cols).ffill().bfill()
X_df=X_df.fillna(X_df.median(numeric_only=True))
X=X_df.to_numpy(); X_ds=X[::DS]
print("Combined table (broker output):",X.shape,"-> downsampled",X_ds.shape)
"""))

cells.append(code("""# === Fault simulators + skenario (identik notebook utama) ===
def simulate_drift_fault(x,intensity=0.02,seed=None):
    drift=np.arange(len(x))*intensity; return x+drift, np.abs(drift)>1e-6
def simulate_spike_fault(x,intensity=0.08,p=0.015,seed=None):
    tau=max(1,int(1.0/p)) if p>0 else len(x)
    spikes=(np.arange(len(x))%tau==0).astype(float)*(intensity*np.nanstd(x))
    return x+spikes, spikes!=0
def simulate_bias_fault(x,bias=0.08,seed=None):
    return x+bias, np.ones(len(x),bool)
def simulate_hardware_fault(x,stuck_prob=0.08,loss_prob=0.05,seed=None):
    rng=np.random.default_rng(seed); n=len(x); rv=rng.random(n); idx=rng.integers(n,size=n)
    m1=rv<stuck_prob; y=x.copy(); y[m1]=x[idx[m1]]; m2=rv<loss_prob; y[m2]=np.nan
    return y,(m1|m2)
def simulate_multiple_faults(x,faults,seed=None):
    y=x.copy(); m=np.zeros(len(x),bool)
    for f,kw in faults:
        y,mi=f(y,**kw,seed=seed); m|=mi
    return y,m
def simulate_choose_one(x,options,seed=None):
    rng=np.random.default_rng(seed); f,kw=options[rng.integers(len(options))]
    return f(x,**kw,seed=seed)
SCENARIOS={
 "faulty":[(simulate_choose_one,{"options":[(simulate_drift_fault,{"intensity":0.02}),(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_bias_fault,{"bias":0.08}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})]})],
 "drift":[(simulate_drift_fault,{"intensity":0.02})],
 "spike":[(simulate_spike_fault,{"intensity":0.08,"p":0.015})],
 "bias":[(simulate_bias_fault,{"bias":0.08})],
 "hardware":[(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "bias+malfunc":[(simulate_bias_fault,{"bias":0.08}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "spike+malfunc":[(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "spike+bias":[(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_bias_fault,{"bias":0.08})],
 "drift+malfunc":[(simulate_drift_fault,{"intensity":0.02}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "drift+bias":[(simulate_drift_fault,{"intensity":0.02}),(simulate_bias_fault,{"bias":0.08})],
 "drift+spike":[(simulate_drift_fault,{"intensity":0.02}),(simulate_spike_fault,{"intensity":0.08,"p":0.015})],
 "spike+bias+malfunc":[(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_bias_fault,{"bias":0.08}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "drift+bias+malfunc":[(simulate_drift_fault,{"intensity":0.02}),(simulate_bias_fault,{"bias":0.08}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "spike+drift+malfunc":[(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_drift_fault,{"intensity":0.02}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05})],
 "drift+spike+bias":[(simulate_drift_fault,{"intensity":0.02}),(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_bias_fault,{"bias":0.08})],
 "spike+bias+malfunc+drift":[(simulate_spike_fault,{"intensity":0.08,"p":0.015}),(simulate_bias_fault,{"bias":0.08}),(simulate_hardware_fault,{"stuck_prob":0.08,"loss_prob":0.05}),(simulate_drift_fault,{"intensity":0.02})],
}
print("skenario:",len(SCENARIOS))
"""))

cells.append(md("""## Windowing + labelling **per-sensor** (Cara A)

Bedanya dengan notebook broker:
- `inject_faults_multisensor` kembalikan mask per-sensor `M` (N, 4).
- `window_fault_label_per_sensor` hitung label per window **per sensor** → `(Nwin, 4)`.
- Window normal: keempat sensor label 0.
- Window fault skenario-k: tiap sensor diberi label 0/1 sesuai mask-nya sendiri
  (bukan OR). Jadi 1 window = vektor 4 label `[S1,S2,S3,S4]`.
"""))

cells.append(code("""# === Windowing + per-sensor label (N, WIN, 4) + (N,4) labels ===
def make_windows(Xn,win,stride):
    Xn=np.asarray(Xn,dtype=np.float32); N=Xn.shape[0]
    if N<win: return np.empty((0,win,Xn.shape[1]),np.float32), np.array([],int)
    view=sliding_window_view(Xn,window_shape=win,axis=0); starts=np.arange(0,N-win+1,stride,dtype=int)
    return view[starts], starts
def inject_faults_multisensor(Xin,scenario_faults,seed=0):
    rng=np.random.default_rng(seed); Y=Xin.copy(); M=np.zeros_like(Y,bool)
    for s in range(Y.shape[1]):
        y,mm=simulate_multiple_faults(Y[:,s],scenario_faults,seed=int(rng.integers(1e9))); Y[:,s]=y; M[:,s]=mm
    Ydf=pd.DataFrame(Y).ffill().bfill(); Ydf=Ydf.fillna(Ydf.median(numeric_only=True))
    return Ydf.to_numpy(), M
def window_fault_label_per_sensor(mask,win,stride,thr=0.02):
    # Label per window per sensor. mask:(T,4) -> labels:(Nwin,4) bool.
    T,ns=mask.shape
    if win>T: return np.zeros((0,ns),bool)
    Wm=sliding_window_view(mask,window_shape=win,axis=0)[::stride]      # (Nwin,4,win)
    ratio=Wm.mean(axis=2)                                               # (Nwin,4)
    return ratio>thr

datasets=[]; Ylst=[]; scen_id=[]
# normal scenario: semua sensor label 0
W0,_=make_windows(X_ds,WIN,STRIDE); datasets.append(W0)
Ylst.append(np.zeros((len(W0),4),int)); scen_id.append(np.zeros(len(W0),int))
for k,(name,faults) in enumerate(SCENARIOS.items(),start=1):
    Y,Mk=inject_faults_multisensor(X_ds,faults,seed=100+k)
    Lk=window_fault_label_per_sensor(Mk,WIN,STRIDE)                    # (Nwin,4)
    Wk,_=make_windows(Y,WIN,STRIDE)
    n=min(len(Wk),Lk.shape[0])
    Wk,Lk=Wk[:n],Lk[:n]
    datasets.append(Wk); Ylst.append(Lk.astype(int)); scen_id.append(np.full(n,k,int))
W_all=np.concatenate(datasets,axis=0); Y_all=np.concatenate(Ylst,axis=0); scn_all=np.concatenate(scen_id,axis=0)
if W_all.ndim==3 and W_all.shape[1]==4 and W_all.shape[2]==WIN: W_all=W_all.transpose(0,2,1)
def balanced_subsample_by_scn(Xw,Yw,scn,maxp,seed=0):
    rng=np.random.default_rng(seed); keep=[]
    for c in np.unique(scn):
        idx=np.where(scn==c)[0]
        if len(idx)>maxp: idx=rng.choice(idx,size=maxp,replace=False)
        keep.append(idx)
    keep=np.concatenate(keep); rng.shuffle(keep); return Xw[keep],Yw[keep],scn[keep]
W_s,Y_s,scn_s=balanced_subsample_by_scn(W_all,Y_all,scn_all,MAX_PER_SCN,RANDOM_SEED)
print("W_s",W_s.shape,"| Y_s",Y_s.shape,"| per-sensor prevalensi:",
      {SENSORS[j]:round(float(Y_s[:,j].mean()),3) for j in range(4)})
"""))

cells.append(code("""# === Entropy (EDM-Fuzzy + JSD-Fuzzy) — identik notebook utama ===
def coarse_grain_mean(x,s):
    n=(len(x)//s)*s
    return x[:n].reshape(-1,s).mean(axis=1) if n>0 else np.array([],float)
def embed_matrix(y,mm):
    L=len(y)
    return np.lib.stride_tricks.sliding_window_view(y,mm) if L>=mm else np.empty((0,mm),float)
def fuzzy_phi(V,r,n_ref=256,seed=0):
    rng=np.random.default_rng(seed); N=V.shape[0]
    if N<3: return np.nan
    ref=rng.choice(N,size=n_ref,replace=False) if N>n_ref else np.arange(N)
    A=V[ref]; a2=np.sum(A*A,1,keepdims=True); b2=np.sum(V*V,1,keepdims=True).T
    d2=np.maximum(a2+b2-2*(A@V.T),0.0); mu=1.0/(1.0+d2/(r*r+1e-24)); mu[np.arange(len(ref)),ref]=0.0
    return (mu.sum(1)/(N-1)).mean()
def fuzzy_similarity_samples(V,r,n_ref=256,seed=0):
    rng=np.random.default_rng(seed); N=V.shape[0]
    if N<3: return np.array([],float)
    ref=rng.choice(np.arange(N),size=n_ref,replace=False) if N>n_ref else np.arange(N)
    A=V[ref]; a2=np.sum(A*A,1,keepdims=True); b2=np.sum(V*V,1,keepdims=True).T
    d=np.sqrt(np.maximum(a2+b2-2*(A@V.T),0.0)); mu=1.0/(1.0+(d/(r+1e-12))**2)
    for ri,i in enumerate(ref): mu[ri,i]=np.nan
    return mu[~np.isnan(mu)].ravel()
def edm_fuzzy_entropy_1d(x,scales,m=2,r_ratio=0.2,n_ref=256,seed=0):
    out=[]
    for s in scales:
        y=coarse_grain_mean(x,s)
        if len(y)<(m+2): out.append(np.nan); continue
        r=r_ratio*np.std(y,ddof=1)
        pm=fuzzy_phi(embed_matrix(y,m),r,n_ref,seed+11*s); pm1=fuzzy_phi(embed_matrix(y,m+1),r,n_ref,seed+17*s)
        out.append(np.log(pm/pm1) if (pm and pm1 and pm>0 and pm1>0 and not np.isnan(pm) and not np.isnan(pm1)) else np.nan)
    return np.array(out,float)
def jsd_fuzzy_entropy_1d(x,scales,m=2,r_ratio=0.2,n_ref=256,seed=0,bins=20,rich=True):
    out=[]; per=4 if rich else 1; be=np.linspace(0,1,bins+1); eps=1e-12
    for s in scales:
        y=coarse_grain_mean(x,s)
        if len(y)<(m+2): out.extend([np.nan]*per); continue
        r=r_ratio*np.std(y,ddof=1)
        mu_m=fuzzy_similarity_samples(embed_matrix(y,m),r,n_ref,seed+11*s)
        mu_m1=fuzzy_similarity_samples(embed_matrix(y,m+1),r,n_ref,seed+17*s)
        if len(mu_m)==0 or len(mu_m1)==0: out.extend([np.nan]*per); continue
        p,_=np.histogram(mu_m,bins=be); q,_=np.histogram(mu_m1,bins=be); p=p.astype(float); q=q.astype(float)
        if p.sum()==0 or q.sum()==0: out.extend([np.nan]*per); continue
        p/=p.sum(); q/=q.sum(); mid=0.5*(p+q)
        jsd=0.5*(np.sum(p*np.log((p+eps)/(mid+eps)))+np.sum(q*np.log((q+eps)/(mid+eps))))
        if rich: out.extend([jsd,np.log((mu_m.mean()+eps)/(mu_m1.mean()+eps)),mu_m.mean(),mu_m.std()])
        else: out.append(jsd)
    return np.array(out,float)
def compute_features_entropy(W,scales,method,m=2,r_ratio=0.2,n_ref=256,jsd_bins=20,seed=0,n_jobs=-1):
    Nwin,win,ns=W.shape; mk=method.strip().lower()
    def ent(x,sd):
        if mk=='edm-fuzzy': return edm_fuzzy_entropy_1d(x,scales,m,r_ratio,n_ref,sd)
        if mk=='jsd-fuzzy': return jsd_fuzzy_entropy_1d(x,scales,m,r_ratio,n_ref,sd,jsd_bins)
        raise ValueError(method)
    def one(i): return np.concatenate([ent(W[i,:,s],seed+1000*i+19*s) for s in range(ns)])
    if n_jobs==1 or Nwin<=1: F=np.vstack([one(i) for i in range(Nwin)])
    else: F=np.vstack(Parallel(n_jobs=n_jobs,prefer='processes')(delayed(one)(i) for i in range(Nwin)))
    Fdf=pd.DataFrame(F); Fdf=Fdf.fillna(Fdf.median(numeric_only=True)).fillna(0.0)
    return Fdf.to_numpy()
print("entropy fns ready")
"""))

cells.append(code("""# === Time-domain (hybrid) features — any channel count ===
def compute_time_features(W):
    N,WINl,ns=W.shape; t=np.arange(WINl, dtype=float); tc=t-t.mean(); tv=(tc**2).mean()+1e-12; feats=[]
    for s in range(ns):
        x=np.asarray(W[:,:,s],float); mu=x.mean(1); sd=x.std(1); rms=np.sqrt((x**2).mean(1))
        ptp=x.max(1)-x.min(1); mad=np.abs(x-mu[:,None]).mean(1)
        sk=_sstats.skew(x,axis=1,bias=False); ku=_sstats.kurtosis(x,axis=1,bias=False)
        d=np.diff(x,axis=1); sm=np.abs(d).mean(1); smx=np.abs(d).max(1)
        tr=(x*tc[None,:]).mean(1)/tv; sc=np.sign(x-mu[:,None]); zcr=(np.abs(np.diff(sc,axis=1))>0).mean(1)
        xf=np.abs(np.fft.rfft(x-mu[:,None],axis=1)); pw=xf**2; half=max(1,pw.shape[1]//2)
        hf=pw[:,half:].sum(1)/(pw.sum(1)+1e-12)
        feats.extend([mu,sd,rms,ptp,mad,sk,ku,sm,smx,tr,zcr,hf])
    return np.nan_to_num(np.column_stack(feats),nan=0.0,posinf=0.0,neginf=0.0)
print("time fns ready")
"""))

cells.append(code("""# === Bangun fitur: (A) 4-channel vs (B) fused 1-channel ===
W_A = W_s                                  # (N, WIN, 4)
W_B = W_s.mean(axis=2, keepdims=True)       # (N, WIN, 1) — fused single stream
print("A (4-channel):", W_A.shape, "| B (fused 1-channel):", W_B.shape)

def build_hybrid(W, method):
    Fe = compute_features_entropy(W, scales, method, m, r_ratio, n_ref, jsd_bins, seed=7, n_jobs=N_JOBS)
    T  = compute_time_features(W)
    return np.hstack([Fe, T])

VARIANTS = {"A_4channel": W_A, "B_fused_1ch": W_B}
FEAT = {}
for meth in METHODS:
    for vname, Wv in VARIANTS.items():
        log_stage(f"features {meth} / {vname}")
        FEAT[(meth, vname)] = build_hybrid(Wv, meth)
        print(f"  {meth:10s} {vname:12s} -> {FEAT[(meth,vname)].shape}")
"""))

cells.append(md("""## Benchmark per-sensor (Cara A: 4 label `[S1,S2,S3,S4]`)

1 `MultiOutputClassifier(MLPClassifier)` dilatih per kombinasi (Method × Variant).
Output: 4 keputusan biner (1 per sensor) per window. Dievaluasi:
- **Per-sensor**: Accuracy, Precision, Recall, F1, ROC-AUC (1 vs rest per sensor).
- **Agregat multi-label**: Subset accuracy (keempat sensor benar), Hamming loss,
  macro-F1.
"""))

cells.append(code("""# === Benchmark per-sensor: MultiOutputClassifier, A(4ch) vs B(fused) ===
def detect_persensor(X, Y, seed=42):
    Xtr,Xte,Ytr,Yte=train_test_split(X,Y,test_size=0.25,random_state=seed)
    I=X.shape[1]
    base=MLPClassifier(max_iter=300,random_state=seed,early_stopping=True,n_iter_no_change=12)
    pipe=Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler()),
                   ("mo",MultiOutputClassifier(base))])
    grid={"mo__estimator__hidden_layer_sizes":[(max(16,I//2),),(I,),(max(16,I//2),max(8,I//4))],
          "mo__estimator__alpha":[1e-4,1e-3]}
    gs=GridSearchCV(pipe,grid,cv=3,scoring="f1_macro",n_jobs=N_JOBS); gs.fit(Xtr,Ytr)
    b=gs.best_estimator_; pred=b.predict(Xte)            # (nte,4)
    proba=np.column_stack([e.predict_proba(Xte)[:,1] for e in b.named_steps["mo"].estimators_])
    per={}
    for j,sname in enumerate(SENSORS):
        p,r,f,_=precision_recall_fscore_support(Yte[:,j],pred[:,j],average="binary",zero_division=0)
        try: auc=roc_auc_score(Yte[:,j],proba[:,j])
        except Exception: auc=np.nan
        per[sname]=dict(Acc=accuracy_score(Yte[:,j],pred[:,j]),Prec=p,Rec=r,F1=f,AUC=auc)
    return dict(pred=pred,proba=proba,Yte=Yte,per=per,
                subset_acc=accuracy_score(Yte,pred),
                hamming=hamming_loss(Yte,pred),
                macro_f1=f1_score(Yte,pred,average="macro",zero_division=0))

rows=[]
for meth in METHODS:
    for vname in VARIANTS:
        if not budget_ok(120,f"{meth}/{vname}"): continue
        Xfull=FEAT[(meth,vname)]
        # lewati sensor tanpa positif cukup
        if any(Y_s[:,j].sum()<10 for j in range(4)):
            print(f"[skip] {meth}/{vname}: sensor kurang dari 10 positif"); continue
        r=detect_persensor(Xfull,Y_s,RANDOM_SEED)
        for sname,m in r["per"].items():
            rows.append({"Method":meth,"Variant":vname,"Sensor":sname,
                         "n_features":Xfull.shape[1],
                         "Accuracy":round(m["Acc"],3),"Precision":round(m["Prec"],3),
                         "Recall":round(m["Rec"],3),"F1":round(m["F1"],3),
                         "ROC_AUC":round(m["AUC"],3) if not np.isnan(m["AUC"]) else np.nan})
        rows.append({"Method":meth,"Variant":vname,"Sensor":"SUBSET(all4)",
                     "n_features":Xfull.shape[1],
                     "Accuracy":round(r["subset_acc"],3),"Precision":np.nan,
                     "Recall":np.nan,"F1":round(r["macro_f1"],3),
                     "ROC_AUC":np.nan})
        log_stage(f"  {meth}/{vname} done | subset_acc={r['subset_acc']:.3f} macro_f1={r['macro_f1']:.3f}")
bench=pd.DataFrame(rows)
print("\\n=== Benchmark per-sensor: 4-channel (A) vs fused-1ch (B) ===")
print(bench.to_string(index=False))
bench.to_csv("exports/persensor_benchmark.csv",index=False)
print("\\n[Saved] exports/persensor_benchmark.csv")
summ=bench[bench.Sensor!="SUBSET(all4)"].groupby(["Method","Variant"])[["Accuracy","F1","ROC_AUC"]].mean().round(3)
print("\\n=== Rata-rata per-sensor (4 sensor) per (Method,Variant) ==="); print(summ.to_string())
summ.to_csv("exports/persensor_summary.csv")
display(bench)
"""))

cells.append(code("""# === Plot: F1 per-sensor, 4-channel vs fused (per metode) ===
fig,axes=plt.subplots(1,len(METHODS),figsize=(13,5),sharey=True)
if len(METHODS)==1: axes=[axes]
for ax,meth in zip(axes,METHODS):
    sub=bench[(bench.Method==meth)&(bench.Sensor!="SUBSET(all4)")]
    piv=sub.pivot(index="Sensor",columns="Variant",values="F1").reindex(SENSORS)
    piv.plot.bar(ax=ax,rot=0); ax.set_title(f"{meth} — F1 per sensor"); ax.set_ylabel("F1")
fig.tight_layout(); fig.savefig("exports/persensor_f1.png",dpi=120)
plt.show()
print("[Saved] exports/persensor_f1.png")
"""))

cells.append(md("""## Catatan: trade-off Cara A

- **Keuntungan**: output menunjuk **sensor mana yang fault** (4 label per window).
- **Biaya**: keluar dari arsitektur broker murni (1 keputusan sistem). Sekarang 4
  sub-keputusan per sensor, meski masih 1 fitur gabungan + 1 MultiOutputClassifier.
- **Label per-sensor tidak menunjukkan tipe fault** (drift/spike/bias/hardware).
  Untuk itu butuh Cara B (16 label = sensor × fault-type), yang lebih berat.
- Prevalensi per sensor bisa timpang karena fault di-inject random per sensor saat
  `inject_faults_multisensor` — lihat print prevalensi di cell windowing.
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9"}}, "nbformat": 4, "nbformat_minor": 5}

with open("/Users/kelvin/apps/public-files/archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb","w") as f:
    json.dump(nb, f, indent=1)
print("wrote archive/LAMA_TIDAK_VALID_Sensor_Mana_Label_Kembar.ipynb | cells:", len(cells))
