"""
anemia_reanalysis.py
Reproduces all results in:
  Nithya S, Sasikala S. Oversampling leakage and threshold-derived labels inflate reported
  accuracy in CBC-based anemia classification: a data audit and corrected statistical benchmark.

Usage:  python anemia_reanalysis.py --csv diagnosed_cbc_data_v4.csv --seed 123 --out results
        (add --fast for a quick 5x2 smoke test)
Data:   Kaggle "Anemia Types Classification" (E. Aboelnaga), file diagnosed_cbc_data_v4.csv
"""
import argparse, json, os, platform, warnings
import numpy as np, pandas as pd
from scipy import stats
import sklearn, xgboost
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.metrics import (make_scorer, matthews_corrcoef, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

COLS = ["WBC", "LYMp", "NEUTp", "LYMn", "NEUTn", "RBC", "HGB", "HCT",
        "MCV", "MCH", "MCHC", "PLT", "PDW", "PCT"]
LABEL = "Diagnosis"
KEEP = ["WBC", "RBC", "HGB", "MCV", "MCH", "MCHC", "PLT"]
BOUNDS = {  # Table 1 of the paper; HGB and MCHC in g/dL as in the raw file (paper reports g/L = x10)
    "WBC": (0.1, 100), "LYMp": (0, 100), "NEUTp": (0, 100), "LYMn": (0, 50),
    "NEUTn": (0, 50), "RBC": (0.5, 8), "HGB": (1, 25), "HCT": (2, 70),
    "MCV": (40, 150), "MCH": (10, 50), "MCHC": (20, 40), "PLT": (5, 1000),
    "PDW": (0, 30), "PCT": (0, 1)}
K, R_FULL, SEED_B = 5, 10, 20260815
SCORING = {"accuracy": "accuracy", "f1_macro": "f1_macro",
           "balanced_accuracy": "balanced_accuracy",
           "mcc": make_scorer(matthews_corrcoef)}


# ---------------------------------------------------------------- audit
def audit_step2(df, share_min=0.25, rel_tol=1e-2):
    """Per feature: most frequent value, its share, column mean, and whether the
    mode equals the mean. Mean imputation leaves the column mean unchanged, so the
    check works whether or not the mean is computed with the imputed cells."""
    rows = []
    for c in COLS:
        s = df[c].dropna()
        mode = s.mode().iloc[0]
        share = (s == mode).mean()
        mean = s.mean()
        is_art = share >= share_min and abs(mode - mean) <= rel_tol * max(abs(mean), 1e-12)
        rows.append(dict(feature=c, mode_value=mode, mode_share=round(share, 4),
                         column_mean=mean, mode_equals_mean=is_art,
                         n_cells=int((df[c] == mode).sum()) if is_art else 0))
    return pd.DataFrame(rows)


def clean(df_raw, drop_dups=True, mask_imputed=True, range_check=True):
    df = df_raw.copy()
    log = {"n_raw": len(df)}
    if drop_dups:
        log["n_duplicates_removed"] = int(df.duplicated(subset=COLS + [LABEL]).sum())
        df = df.drop_duplicates(subset=COLS + [LABEL], keep="first").reset_index(drop=True)
    aud = audit_step2(df)
    imputed_flag = pd.Series(False, index=df.index)
    if mask_imputed:
        for _, r in aud[aud.mode_equals_mean].iterrows():
            m = df[r.feature] == r.mode_value
            imputed_flag |= m
            df.loc[m, r.feature] = np.nan
    log["cells_masked_step2"] = int(aud.n_cells.sum()) if mask_imputed else 0
    if range_check:
        n = 0
        for c, (lo, hi) in BOUNDS.items():
            m = (df[c] < lo) | (df[c] > hi)
            n += int(m.sum()); df.loc[m, c] = np.nan
        log["cells_masked_step3"] = n
    log["n_final"] = len(df)
    return df, aud, imputed_flag, log


# ---------------------------------------------------------------- models
def models(seed):
    pre = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    return {
        "LogisticRegression": Pipeline(pre + [("clf", LogisticRegression(max_iter=5000))]),
        "SVM_RBF": Pipeline(pre + [("clf", SVC())]),
        "RandomForest": Pipeline(pre + [("clf", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1))]),
        "XGBoost": Pipeline(pre + [("clf", XGBClassifier(n_estimators=300, objective="multi:softprob",
                                                          eval_metric="mlogloss", random_state=seed, n_jobs=-1))]),
        "KNN_k7": Pipeline(pre + [("clf", KNeighborsClassifier(n_neighbors=7))]),
    }


def record_params(out):
    info = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "numpy": np.__version__, "models": {}}
    for name, p in models(0).items():
        prm = p.named_steps["clf"].get_params()
        info["models"][name] = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                for k, v in prm.items()}
    json.dump(info, open(os.path.join(out, "hyperparameters_and_versions.json"), "w"), indent=1)


# ---------------------------------------------------------------- statistics
def corrected_ttest(a, b, k=K, r=None):
    """Nadeau-Bengio corrected repeated k-fold t-test (Bouckaert & Frank form).
    n_test/n_train = 1/(k-1) = 0.25 for 5-fold; df = k*r - 1."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1.0 / n + 1.0 / (k - 1)) * var)
    mean = d.mean()
    t = mean / se if se > 0 else np.inf * np.sign(mean)
    df_ = n - 1
    p = 2 * stats.t.sf(abs(t), df_)
    tc = stats.t.ppf(0.975, df_)
    dz = mean / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
    return dict(mean_diff=mean, ci_low=mean - tc * se, ci_high=mean + tc * se,
                t=t, df=df_, p_raw=p, d_z=dz)


def holm(p):
    p = np.asarray(p); m = len(p); order = np.argsort(p)
    adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def run_cv(X, y, seed, repeats, out, tag):
    cv = RepeatedStratifiedKFold(n_splits=K, n_repeats=repeats, random_state=seed)
    splits = list(cv.split(X, y))          # identical partitions for every model and metric
    scores = {}
    for name, pipe in models(seed).items():
        res = cross_validate(pipe, X, y, cv=splits, scoring=SCORING, n_jobs=1)
        scores[name] = {m: res["test_" + m] for m in SCORING}
    summ = pd.DataFrame({n: {m: f"{100*v.mean():.1f} \u00b1 {100*v.std(ddof=1):.1f}" for m, v in s.items()}
                         for n, s in scores.items()}).T
    summ.to_csv(os.path.join(out, f"cv_summary_{tag}.csv"))
    names = list(scores)
    tests = []
    for metric in SCORING:
        block = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rr = corrected_ttest(scores[names[i]][metric], scores[names[j]][metric], r=repeats)
                block.append(dict(metric=metric, A=names[i], B=names[j], **rr))
        adj = holm([b["p_raw"] for b in block])
        for b, pa in zip(block, adj):
            b["p_holm"] = pa; b["significant"] = pa < 0.05
        tests += block
    tdf = pd.DataFrame(tests)
    for c in ["mean_diff", "ci_low", "ci_high"]:
        tdf[c] = (100 * tdf[c]).round(2)
    tdf.to_csv(os.path.join(out, f"pairwise_tests_{tag}.csv"), index=False)
    print(f"\n== {tag} ==\n{summ}\n")
    print(tdf[tdf.metric.isin(["accuracy", "f1_macro"])]
          [["metric", "A", "B", "mean_diff", "ci_low", "ci_high", "d_z", "p_holm", "significant"]].to_string(index=False))
    return scores


# ---------------------------------------------------------------- ablation
def ablation(df_raw, y_enc_fn, seed, repeats, out):
    """Accuracy/macro-F1 of RF and XGB under the data conditions of earlier studies."""
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    cv = RepeatedStratifiedKFold(n_splits=K, n_repeats=repeats, random_state=seed)
    scen = {}
    raw14 = df_raw.copy()
    scen["S1 raw: duplicates kept, 14 features, imputed values kept"] = (raw14[COLS], raw14[LABEL])
    d = df_raw.drop_duplicates(subset=COLS + [LABEL])
    scen["S2 duplicates removed, 14 features, imputed values kept"] = (d[COLS], d[LABEL])
    c14, _, _, _ = clean(df_raw)
    scen["S3 cleaned, 14 features, artifacts masked + in-fold median"] = (c14[COLS], c14[LABEL])
    scen["S4 cleaned, 13 features (PDW kept, six dropped)"] = (c14[[c for c in KEEP] + ["PDW"]], c14[LABEL])
    scen["S5 cleaned, 7 features (paper setting)"] = (c14[KEEP], c14[LABEL])
    rows = []
    for sname, (X, ylab) in scen.items():
        y = y_enc_fn(ylab)
        for mname in ["RandomForest", "XGBoost"]:
            res = cross_validate(models(seed)[mname], X, y, cv=cv, scoring=["accuracy", "f1_macro"])
            rows.append(dict(scenario=sname, model=mname,
                             accuracy=100 * res["test_accuracy"].mean(),
                             f1_macro=100 * res["test_f1_macro"].mean()))
    # SMOTE before vs inside the split, on raw data (conditions described for [4])
    y = y_enc_fn(raw14[LABEL]); X = raw14[COLS].values
    kn = max(1, min(5, int(np.bincount(y).min()) - 1))
    Xs, ys = SMOTE(k_neighbors=kn, random_state=seed).fit_resample(X, y)
    for mname in ["RandomForest", "XGBoost"]:
        clf = models(seed)[mname].named_steps["clf"]
        res = cross_validate(clone(clf), Xs, ys, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append(dict(scenario="S6 raw + SMOTE BEFORE split (leaky)", model=mname,
                         accuracy=100 * res["test_accuracy"].mean(), f1_macro=100 * res["test_f1_macro"].mean()))
        pipe = ImbPipeline([("smote", SMOTE(k_neighbors=kn, random_state=seed)), ("clf", clone(clf))])
        res = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append(dict(scenario="S7 raw + SMOTE INSIDE training folds", model=mname,
                         accuracy=100 * res["test_accuracy"].mean(), f1_macro=100 * res["test_f1_macro"].mean()))
    a = pd.DataFrame(rows).round(2)
    a.to_csv(os.path.join(out, "ablation_data_quality.csv"), index=False)
    print("\n== Ablation ==\n", a.to_string(index=False))


# ---------------------------------------------------------------- other checks
def label_recoverability(X, y, classes, seed, out):
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    rows = []
    for depth in [2, 3, 4, 5, 6, None]:
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("clf", DecisionTreeClassifier(max_depth=depth, random_state=seed))])
        res = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        rows.append(dict(max_depth=depth, accuracy=100 * res["test_accuracy"].mean(),
                         f1_macro=100 * res["test_f1_macro"].mean()))
    pd.DataFrame(rows).round(2).to_csv(os.path.join(out, "shallow_tree_recoverability.csv"), index=False)
    print("\n== Shallow-tree label recoverability ==\n", pd.DataFrame(rows).round(2).to_string(index=False))
    pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", DecisionTreeClassifier(max_depth=5, random_state=seed))]).fit(X, y)
    rules = export_text(pipe.named_steps["clf"], feature_names=list(X.columns),
                        class_names=list(classes))
    open(os.path.join(out, "tree_depth5_rules.txt"), "w").write(rules)


def nested_tuning(X, y, seed, repeats, out):
    grids = {
        "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
        "SVM_RBF": {"clf__C": [0.1, 1, 10, 100, 1000], "clf__gamma": ["scale", 0.01, 0.1, 1]},
        "KNN_k7": {"clf__n_neighbors": [1, 3, 5, 7, 9, 15], "clf__weights": ["uniform", "distance"]},
    }
    outer = RepeatedStratifiedKFold(n_splits=K, n_repeats=repeats, random_state=seed)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    rows = []
    for name, grid in grids.items():
        gs = GridSearchCV(models(seed)[name], grid, cv=inner, scoring="f1_macro", n_jobs=-1)
        res = cross_validate(gs, X, y, cv=outer, scoring=["accuracy", "f1_macro"])
        rows.append(dict(model=name + " (nested-CV tuned)",
                         accuracy=f"{100*res['test_accuracy'].mean():.1f} \u00b1 {100*res['test_accuracy'].std(ddof=1):.1f}",
                         f1_macro=f"{100*res['test_f1_macro'].mean():.1f} \u00b1 {100*res['test_f1_macro'].std(ddof=1):.1f}"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "nested_cv_tuned_baselines.csv"), index=False)
    print("\n== Nested-CV tuned baselines ==\n", pd.DataFrame(rows).to_string(index=False))


def per_class(X, y, classes, seed, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
    for name in ["RandomForest", "XGBoost"]:
        pred = cross_val_predict(models(seed)[name], X, y, cv=cv)
        rep = pd.DataFrame(classification_report(y, pred, target_names=classes, output_dict=True, digits=3)).T
        rep.round(3).to_csv(os.path.join(out, f"per_class_{name}_seed{seed}.csv"))
        cm = confusion_matrix(y, pred)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out, f"confusion_{name}_seed{seed}.csv"))
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        fig.tight_layout(); fig.savefig(os.path.join(out, f"fig1_confusion_{name}.png"), dpi=300); plt.close(fig)


# ---------------------------------------------------------------- tree vs ensembles (Section 4.6)
def tree_vs_ensembles(X, y, out, seed=SEED_B, repeats=R_FULL):
    """Depth-5 decision tree vs Random Forest and XGBoost on the 50 primary partitions
    (seed 20260815), Nadeau-Bengio corrected t-test, Holm over the 4 comparisons.
    The three models are refitted here with random_state=0 and median imputation only,
    exactly as in the run reported in Section 4.6 (tree 98.5%, RF 98.4%, XGB 98.4%)."""
    cv = RepeatedStratifiedKFold(n_splits=K, n_repeats=repeats, random_state=seed)
    mdl = {"Tree_depth5": DecisionTreeClassifier(max_depth=5, random_state=0),
           "RandomForest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
           "XGBoost": XGBClassifier(n_estimators=300, eval_metric="mlogloss", random_state=0, n_jobs=-1)}
    sc, summ = {}, []
    for name, m in mdl.items():
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("clf", m)])
        r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
        sc[name] = {k: r[f"test_{k}"] * 100 for k in ["accuracy", "f1_macro"]}
        summ.append(dict(model=name, **{f"{k}_mean": v.mean() for k, v in sc[name].items()},
                         **{f"{k}_sd": v.std(ddof=1) for k, v in sc[name].items()}))
    rows = []
    for metric in ["accuracy", "f1_macro"]:
        for other in ["RandomForest", "XGBoost"]:
            res = corrected_ttest(sc["Tree_depth5"][metric], sc[other][metric], k=K)
            rows.append(dict(metric=metric, A="Tree_depth5", B=other, **res))
    tests = pd.DataFrame(rows); tests["p_holm"] = holm(tests.p_raw.values)
    tests["significant"] = tests.p_holm < 0.05
    pd.DataFrame(summ).round(2).to_csv(os.path.join(out, "tree_vs_ensembles_summary.csv"), index=False)
    tests.to_csv(os.path.join(out, "tree_vs_ensembles_tests.csv"), index=False)
    print("\n== Depth-5 tree vs ensembles (seed %d, 5x%d) ==" % (seed, repeats))
    print(pd.DataFrame(summ).round(1).to_string(index=False))
    print(tests[["metric", "A", "B", "mean_diff", "ci_low", "ci_high", "d_z", "p_holm", "significant"]].round(3).to_string(index=False))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="diagnosed_cbc_data_v4.csv")
    ap.add_argument("--seed", type=int, default=123, help="secondary seed (Supplementary Tables S2-S3; ablation, trees, tuning)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true")
    a, _ = ap.parse_known_args()   # parse_known_args also works inside Jupyter/Colab
    os.makedirs(a.out, exist_ok=True)
    reps = 2 if a.fast else R_FULL

    import hashlib
    print("Input file:", a.csv, "SHA-256:", hashlib.sha256(open(a.csv, "rb").read()).hexdigest())
    raw = pd.read_csv(a.csv)
    missing = [c for c in COLS + [LABEL] if c not in raw.columns]
    if missing:
        raise SystemExit(f"Columns not found: {missing}. Edit COLS/LABEL.")
    record_params(a.out)

    cleaned, aud, flag, log = clean(raw)
    aud.to_csv(os.path.join(a.out, "step2_audit_table.csv"), index=False)
    json.dump(log, open(os.path.join(a.out, "cleaning_log.json"), "w"), indent=1)
    print("Cleaning log:", log); print(aud.to_string(index=False))

    # A14: are imputed records concentrated in particular classes?
    ct = pd.crosstab(cleaned[LABEL], flag.rename("any_imputed_value"))
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    ct.to_csv(os.path.join(a.out, "imputed_flag_by_class.csv"))
    print("\n== Imputed-record flag by class ==\n", ct, f"\nchi2={chi2:.2f}, df={dof}, p={p:.3g}")
    n_all7 = int(cleaned[["LYMp", "NEUTp", "LYMn", "NEUTn", "HCT", "PCT"]].isna().all(axis=1).sum())
    print("Records with all six mean-imputed features missing:", n_all7)

    le = LabelEncoder().fit(raw[LABEL])
    enc = lambda s: le.transform(s)
    classes = list(le.classes_)
    X, y = cleaned[KEEP], enc(cleaned[LABEL])

    run_cv(X, y, a.seed, reps, a.out, f"seed{a.seed}")
    run_cv(X, y, SEED_B, reps, a.out, f"seed{SEED_B}")
    ablation(raw, enc, a.seed, reps, a.out)
    label_recoverability(X, y, classes, a.seed, a.out)
    nested_tuning(X, y, a.seed, reps, a.out)
    per_class(X, y, classes, a.seed, a.out)
    tree_vs_ensembles(X, y, a.out, seed=SEED_B, repeats=reps)
    print(f"\nAll outputs written to {a.out}/")


if __name__ == "__main__":
    main()
