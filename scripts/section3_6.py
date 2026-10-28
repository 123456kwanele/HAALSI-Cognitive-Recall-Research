"""Section 3.6: leakage-safe held-out SHAP for Ridge, RF, MLP and spline KAN.
Place in scripts/ beside section3_3.py. Run python scripts/section3_6_shap.py
Requires: numpy pandas scikit-learn matplotlib torch shap.
SHAP explains prediction of follow-up delayed recall, NOT clinical diagnoses.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
try:
    import shap
except ImportError as exc:
    raise SystemExit('Missing SHAP: run python -m pip install shap') from exc
from section3_3 import load_data, train_kan, predict as predict_kan, FEATURE_SETS, SEED

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'; FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)
FEATURE_SET_NAMES = ['B_TotalRecall', 'C_DecomposedSPC', 'D_TotalPlusShape']
BACKGROUND_N = 40
EXPLAIN_N = 50
SHAP_PERMUTATION_REPEATS = 5  # computational cost scales with this


def make_model(name):
    if name == 'Ridge':
        estimator = Ridge(alpha=10.0)
    elif name == 'RandomForest':
        estimator = RandomForestRegressor(n_estimators=200, max_depth=6,
            min_samples_leaf=5, n_jobs=1, random_state=SEED)
    elif name == 'MLP':
        estimator = MLPRegressor(hidden_layer_sizes=(16, 8), alpha=1.0,
            early_stopping=True, random_state=SEED, max_iter=1000)
    else:
        raise ValueError(name)
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler()), ('regressor', estimator)])


def run():
    print('SECTION 3.6: HELD-OUT SHAP INTERPRETABILITY')
    print('SHAP version:', shap.__version__)
    df = load_data()
    y = df['delayed_recall_followup'].to_numpy(float)
    train_idx, test_idx = next(RepeatedKFold(n_splits=5, n_repeats=3,
                                             random_state=SEED).split(df))
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(train_idx, size=min(BACKGROUND_N, len(train_idx)), replace=False)
    ex_idx = rng.choice(test_idx, size=min(EXPLAIN_N, len(test_idx)), replace=False)
    metadata = []
    importance = []
    all_shap = []
    for fs in FEATURE_SET_NAMES:
        features = FEATURE_SETS[fs]
        X = df[features]
        bg = X.iloc[bg_idx].reset_index(drop=True)
        explained = X.iloc[ex_idx].reset_index(drop=True)
        for name in ['Ridge', 'RandomForest', 'MLP', 'KAN']:
            print(f'Explaining {name} | {fs}', flush=True)
            if name == 'KAN':
                fitted = train_kan(X.iloc[train_idx], y[train_idx], SEED)
                kan, imp, scaler, ym, ys, best_epoch = fitted
                def model_predict(a):
                    frame = pd.DataFrame(np.asarray(a), columns=features)
                    return np.asarray(predict_kan(kan, imp, scaler, ym, ys, frame), dtype=float)
            else:
                model = make_model(name)
                model.fit(X.iloc[train_idx], y[train_idx])
                def model_predict(a):
                    frame = pd.DataFrame(np.asarray(a), columns=features)
                    return np.asarray(model.predict(frame), dtype=float)
            # Use the SAME model-agnostic SHAP algorithm for all model families.
            # The background consists ONLY of outer-training participants.
            masker = shap.maskers.Independent(bg, max_samples=len(bg))
            explainer = shap.Explainer(model_predict, masker, algorithm='permutation',
                                      feature_names=features)
            max_evals = (2 * len(features) + 1) * SHAP_PERMUTATION_REPEATS
            values = explainer(explained, max_evals=max_evals, silent=True)
            matrix = np.asarray(values.values, dtype=float)
            if matrix.ndim != 2 or matrix.shape != (len(explained), len(features)):
                raise RuntimeError(f'Unexpected SHAP shape: {matrix.shape}')
            test_pred = model_predict(X.iloc[test_idx].to_numpy())
            metadata.append(dict(Model=name, FeatureSet=fs, TrainN=len(train_idx),
                TestN=len(test_idx), ExplainedN=len(explained), BackgroundN=len(bg),
                HeldoutMAE=mean_absolute_error(y[test_idx], test_pred),
                HeldoutR2=r2_score(y[test_idx], test_pred),
                Note='Single held-out fold; descriptive SHAP; not full CV importance'))
            for j, feature in enumerate(features):
                importance.append(dict(Model=name, FeatureSet=fs, Feature=feature,
                    MeanAbsSHAP=float(np.mean(np.abs(matrix[:, j]))),
                    MeanSignedSHAP=float(np.mean(matrix[:, j]))))
            for i in range(len(explained)):
                for j, feature in enumerate(features):
                    all_shap.append(dict(Model=name, FeatureSet=fs,
                        prim_key=str(df.iloc[ex_idx[i]]['prim_key']), Feature=feature,
                        FeatureValue=float(explained.iloc[i, j]),
                        SHAPValue=float(matrix[i, j])))
            slug = f'section3_6_shap_{name}_{fs}'
            try:
                plt.figure(figsize=(9, 5.5))
                shap.summary_plot(matrix, explained, feature_names=features,
                                  show=False, plot_type='dot', max_display=12)
                plt.title(f'{name}: {fs} | held-out SHAP')
                plt.tight_layout()
                plt.savefig(FIG / f'{slug}_beeswarm.png', dpi=180, bbox_inches='tight')
                plt.close('all')
                plt.figure(figsize=(9, 5))
                shap.summary_plot(matrix, explained, feature_names=features,
                                  show=False, plot_type='bar', max_display=12)
                plt.title(f'{name}: {fs} | mean absolute SHAP')
                plt.tight_layout()
                plt.savefig(FIG / f'{slug}_bar.png', dpi=180, bbox_inches='tight')
                plt.close('all')
            except Exception as exc:
                plt.close('all')
                warnings.warn(f'Plot failed for {slug}: {exc}')
            print('  Held-out MAE:', round(metadata[-1]['HeldoutMAE'], 4))
    pd.DataFrame(metadata).to_csv(OUT / 'section3_6_shap_model_metadata.csv', index=False)
    summary = pd.DataFrame(importance).sort_values(['Model', 'FeatureSet', 'MeanAbsSHAP'],
                                                     ascending=[True, True, False])
    summary.to_csv(OUT / 'section3_6_shap_importance.csv', index=False)
    pd.DataFrame(all_shap).to_csv(OUT / 'section3_6_shap_individual_values.csv', index=False)
    print('\nSHAP IMPORTANCE\n', summary.to_string(index=False))
    print('\nSECTION 3.6 COMPLETE: 12 model/feature-set combinations')
    print('Important: SHAP associations are not causal; correlated features share attribution.')

if __name__ == '__main__':
    run()
