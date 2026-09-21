#!/usr/bin/env python3
"""
Re-analysis of Pathway II external validation (n=219) for the Life/MDPI revision.

Reproduces the six PCA/UMAP x {RF, XGBoost, LightGBM} pipelines from the public
notebook (same seed = 42, same hard-coded Youden thresholds) and adds the
statistics requested by the reviewers:

  * full operating-characteristic metrics (sensitivity, specificity, PPV, NPV,
    balanced accuracy, F1)
  * external-cohort ROC-AUC and Brier score
  * accuracy with 95% bootstrap confidence intervals
  * calibration slope and intercept (Cox calibration)
  * paired McNemar tests (UMAP vs PCA for each classifier, and all pairs)
  * verification that the six prediction vectors are distinct (the "161 hits"
    coincidence raised by Reviewer 3)
  * K-Means cluster validity: silhouette over k=2..8 and bootstrap ARI at k=5,
    confirming that the outcome `cardio` is NOT a clustering input.

Outputs (written next to this script):
  results.json        machine-readable summary of everything
  metrics_219.csv     the external-cohort metrics table
  results.txt         human-readable log

Usage:
  pip install -r requirements.txt
  python run_validation.py
"""
import warnings, json, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import xgboost as xgb
import umap.umap_ as umap
from sklearn.metrics import (roc_auc_score, confusion_matrix, brier_score_loss,
                             balanced_accuracy_score, f1_score, silhouette_score,
                             adjusted_rand_score)
from sklearn.cluster import KMeans
from scipy import stats
import os

RS = 42
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

# ---------- Load data exactly as the original notebook ----------
dfA = pd.read_csv(os.path.join(DATA, "cardio_train_1.csv"), sep=';')
dfA['age'] = dfA['age'] / 365
dfA['height'] = dfA['height'] / 100
dfA.drop('id', axis=1, inplace=True)

dfB = pd.read_csv(os.path.join(DATA, "3-Variables-to-cardiovascular-disease.csv"), sep=',')
dfB = dfB.dropna()
CardioB = dfB['cardio']
dfB["height"] = dfB["height"] / 100
dfB.rename(columns={'age_years': 'age'}, inplace=True)
dfB.drop("id ", axis=1, inplace=True)
dfB.drop('cardio', axis=1, inplace=True)
dfB.drop('bmi', axis=1, inplace=True)

biolog = {'ap_hi_min':60,'ap_hi_max':250,'height_min':1.00,'height_max':2.20,'weight_min':35,'weight_max':250}
dfA2 = dfA.copy(); dfA2.drop('ap_lo', axis=1, inplace=True)
mask = ((dfA2['ap_hi']>=biolog['ap_hi_min'])&(dfA2['ap_hi']<=biolog['ap_hi_max'])&
        (dfA2['height']>=biolog['height_min'])&(dfA2['height']<=biolog['height_max'])&
        (dfA2['weight']>=biolog['weight_min'])&(dfA2['weight']<=biolog['weight_max']))
dfA2_limpio = dfA2[mask].copy()
cardioA = dfA2_limpio['cardio']
dfA2_limpio.drop('cardio', axis=1, inplace=True)
log(f"Training cohort after biological filtering: {dfA2_limpio.shape[0]} records "
    f"(from {dfA.shape[0]}); external validation cohort: {dfB.shape[0]} patients, "
    f"{int(CardioB.sum())} CVD-positive.")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    dfA2_limpio, cardioA, test_size=0.2, random_state=RS, stratify=cardioA)
dfB = dfB[X_train_raw.columns]

esc = StandardScaler()
X_train_s = esc.fit_transform(X_train_raw)
X_test_s  = esc.transform(X_test_raw)
dfB_s     = esc.transform(dfB)

# ---------- Dimensionality reduction (fit once on train, transform the rest) ----------
pca = PCA(n_components=7, random_state=RS)
Xtr_pca = pca.fit_transform(X_train_s); B_pca = pca.transform(dfB_s)
log("Fitting UMAP (7D) on the training cohort ...")
um = umap.UMAP(n_components=7, n_neighbors=15, min_dist=0.1, random_state=RS)
Xtr_umap = um.fit_transform(X_train_s); B_umap = um.transform(dfB_s)
log("UMAP done.")

yb = CardioB.values.astype(int)

# ---------- Final models + hard-coded Youden thresholds (from the notebook) ----------
def mk_lgbm(md): return LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=md,
                                       random_state=RS, n_jobs=-1, verbose=-1)
configs = {
 "LightGBM+PCA":  (mk_lgbm(12), Xtr_pca, B_pca, 0.5214),
 "LightGBM+UMAP": (mk_lgbm(10), Xtr_umap, B_umap, 0.5239),
 "XGBoost+PCA":   (xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6,
                    random_state=RS, n_jobs=-1, eval_metric='logloss'), Xtr_pca, B_pca, 0.5027),
 "XGBoost+UMAP":  (xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                    random_state=RS, n_jobs=-1, eval_metric='logloss'), Xtr_umap, B_umap, 0.4958),
 "RandomForest+PCA":  (RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=4,
                        min_samples_split=2, random_state=RS, n_jobs=-1), Xtr_pca, B_pca, 0.5186),
 "RandomForest+UMAP": (RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_leaf=4,
                        min_samples_split=10, random_state=RS, n_jobs=-1), Xtr_umap, B_umap, 0.5421),
}

def bootstrap_acc_ci(y, yhat, n=3000, seed=42):
    rng = np.random.default_rng(seed); N=len(y); accs=np.empty(n)
    for i in range(n):
        idx = rng.integers(0, N, N); accs[i] = (yhat[idx]==y[idx]).mean()
    return np.percentile(accs, [2.5, 97.5]) * 100

def calibration_slope_intercept(y, p):
    eps = 1e-6; lp = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    # slope: coefficient of the linear predictor
    m = LogisticRegression(solver='lbfgs', C=1e6).fit(lp.reshape(-1,1), y)
    slope = float(m.coef_[0][0])
    # intercept (calibration-in-the-large): slope fixed at 1, fit intercept only via offset
    m2 = LogisticRegression(solver='lbfgs', C=1e6, fit_intercept=True)
    m2.fit(np.zeros((len(y),1)), y)  # intercept-only baseline is not what we want; use offset trick:
    # logistic with offset = lp: fit intercept minimizing deviance -> use statsmodels-free approx
    from scipy.optimize import minimize_scalar
    def nll(a):
        z = a + lp; pr = 1/(1+np.exp(-z)); pr = np.clip(pr, eps, 1-eps)
        return -np.sum(y*np.log(pr) + (1-y)*np.log(1-pr))
    a = minimize_scalar(nll).x
    return round(slope,3), round(float(a),3)

rows=[]; preds={}; probs={}
for name,(mdl,Xtr,Bx,thr) in configs.items():
    mdl.fit(Xtr, y_train)
    p = mdl.predict_proba(Bx)[:,1]; yhat=(p>=thr).astype(int)
    preds[name]=yhat; probs[name]=p
    tn,fp,fn,tp = confusion_matrix(yb,yhat).ravel()
    sens=tp/(tp+fn); spec=tn/(tn+fp); ppv=tp/(tp+fp) if (tp+fp) else float('nan')
    npv=tn/(tn+fn) if (tn+fn) else float('nan')
    lo,hi = bootstrap_acc_ci(yb, yhat)
    cslope, cint = calibration_slope_intercept(yb, p)
    rows.append(dict(model=name, hits=int((yhat==yb).sum()), n=len(yb),
        accuracy_pct=round((yhat==yb).mean()*100,2), acc_CI95=f"[{lo:.1f}, {hi:.1f}]",
        sensitivity_pct=round(sens*100,1), specificity_pct=round(spec*100,1),
        PPV_pct=round(ppv*100,1), NPV_pct=round(npv*100,1),
        balanced_acc_pct=round(balanced_accuracy_score(yb,yhat)*100,1),
        F1=round(f1_score(yb,yhat),3), AUC_external=round(roc_auc_score(yb,p),3),
        Brier=round(brier_score_loss(yb,p),3),
        calib_slope=cslope, calib_intercept=cint, threshold=thr))
    log(f"{name:18s} hits={rows[-1]['hits']}/{len(yb)} acc={rows[-1]['accuracy_pct']}% "
        f"CI={rows[-1]['acc_CI95']} sens={rows[-1]['sensitivity_pct']} spec={rows[-1]['specificity_pct']} "
        f"AUC={rows[-1]['AUC_external']} Brier={rows[-1]['Brier']} calib(slope,int)=({cslope},{cint})")

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(os.path.join(HERE,"metrics_219.csv"), index=False)

# ---------- McNemar (paired) ----------
def mcnemar(y,a,b):
    b01=int(((a==y)&(b!=y)).sum()); b10=int(((a!=y)&(b==y)).sum()); ndisc=b01+b10
    p = 1.0 if ndisc==0 else stats.binomtest(min(b01,b10), ndisc, 0.5, alternative='two-sided').pvalue
    return dict(a_right_b_wrong=b01, a_wrong_b_right=b10, p_value=round(float(p),4))
pairs=[("LightGBM+UMAP","LightGBM+PCA"),("XGBoost+UMAP","XGBoost+PCA"),
       ("RandomForest+UMAP","RandomForest+PCA")]
mcn={f"{a} vs {b}":mcnemar(yb,preds[a],preds[b]) for a,b in pairs}
log("\nPaired McNemar (UMAP vs PCA):")
for k,v in mcn.items(): log("  ", k, v)
mcn_all={f"{a} vs {b}":mcnemar(yb,preds[a],preds[b]) for a,b in itertools.combinations(preds,2)}

# ---------- 161-hits anomaly: are any prediction vectors identical? ----------
identical=[(a,b) for a,b in itertools.combinations(preds,2) if np.array_equal(preds[a],preds[b])]
log("\nIdentical prediction vectors:", identical if identical else "none (all six are distinct)")

# ---------- Clustering validity (cardio NOT a feature) ----------
log("\nClustering feature columns (note: 'cardio' is absent):", list(X_train_raw.columns))
sil_tr={k:round(silhouette_score(X_train_s, KMeans(k,random_state=RS,n_init=10).fit_predict(X_train_s)),3) for k in range(2,9)}
sil_b ={k:round(silhouette_score(dfB_s,   KMeans(k,random_state=RS,n_init=10).fit_predict(dfB_s)),3) for k in range(2,9)}
log("Silhouette (train) by k:", sil_tr)
log("Silhouette (valid) by k:", sil_b)
def boot_ari(X,k=5,n=200,seed=42):
    rng=np.random.default_rng(seed); base=KMeans(k,random_state=RS,n_init=10).fit_predict(X); a=[]
    for _ in range(n):
        idx=rng.integers(0,len(X),len(X))
        a.append(adjusted_rand_score(base[idx], KMeans(k,random_state=int(rng.integers(1e6)),n_init=5).fit_predict(X[idx])))
    return round(float(np.mean(a)),3), round(float(np.std(a)),3)
ari_b=boot_ari(dfB_s,5,200)
log(f"Bootstrap ARI k=5 (validation cohort): mean={ari_b[0]} sd={ari_b[1]}")

# ---------- Save ----------
out=dict(n_valid=int(len(yb)), n_positive=int(yb.sum()),
         training_records_after_filter=int(dfA2_limpio.shape[0]),
         metrics=rows, mcnemar_umap_vs_pca=mcn, mcnemar_all_pairs=mcn_all,
         identical_prediction_vectors=identical,
         silhouette_train=sil_tr, silhouette_valid=sil_b, bootstrap_ARI_k5_valid=ari_b,
         cluster_features=list(X_train_raw.columns), seed=RS)
json.dump(out, open(os.path.join(HERE,"results.json"),"w"), indent=2)
open(os.path.join(HERE,"results.txt"),"w").write("\n".join(LOG))
log("\nSaved results.json, metrics_219.csv, results.txt")
