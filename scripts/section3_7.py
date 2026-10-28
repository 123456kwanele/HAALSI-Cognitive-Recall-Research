"""Section 3.7: direct proposal Model A vs Model B comparison with cognitive-score outcome.
A: T_i; B: P_i, M_i, R_i. Includes adjusted and education-blind versions.
Run: python scripts/section3_7_proposal_alignment.py
Dependencies: numpy pandas scikit-learn matplotlib scipy.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'HAALSI_Paired_Delayed_Recall_Analysis.csv'
OUT = ROOT / 'outputs'; FIG = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
FEATURE_SETS = {
    'A_TotalOnly': ['T_i_w1'],
    'B_DecomposedOnly': ['P_i_w1', 'M_i_w1', 'R_i_w1'],
    'A_Adjusted': ['age_baseline', 'education_level', 'delayed_recall_baseline', 'T_i_w1'],
    'B_Adjusted': ['age_baseline', 'education_level', 'delayed_recall_baseline', 'P_i_w1', 'M_i_w1', 'R_i_w1'],
    'A_AdjustedEducationBlind': ['age_baseline', 'delayed_recall_baseline', 'T_i_w1'],
    'B_AdjustedEducationBlind': ['age_baseline', 'delayed_recall_baseline', 'P_i_w1', 'M_i_w1', 'R_i_w1'],
}


def make_estimator(name):
    if name == 'Ridge':
        reg = Ridge(alpha=10.0)
    else:
        reg = RandomForestRegressor(n_estimators=200, max_depth=6,
                min_samples_leaf=5, random_state=SEED, n_jobs=1)
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scale', StandardScaler()), ('model', reg)])


def paired_bootstrap(predictions, model, a, b, n_boot=2000):
    """Participant-level paired bootstrap of absolute-error improvement A minus B."""
    subset = predictions[(predictions['Model'] == model) &
                         (predictions['FeatureSet'].isin([a, b]))]
    # Each person occurs in three repeats; first average their errors over repeats.
    pivot = subset.groupby(['prim_key', 'FeatureSet'])['AbsError'].mean().unstack()
    pivot = pivot[[a, b]].dropna()
    diff = (pivot[a] - pivot[b]).to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = diff[idx].mean(axis=1)
    return dict(Model=model, Baseline=a, Proposed=b, N=len(diff),
                Improvement_PositiveIsBetter=float(diff.mean()),
                CI95_Lower=float(np.quantile(boot, .025)),
                CI95_Upper=float(np.quantile(boot, .975)))


def run():
    print('SECTION 3.7: PROPOSAL-ALIGNED TOTAL VS DECOMPOSED SCORE REGRESSION')
    if not DATA.exists():
        raise FileNotFoundError(f'Cannot find {DATA}')
    df = pd.read_csv(DATA)
    needed = set(sum(FEATURE_SETS.values(), [])) | {'prim_key', 'education_level', 'delayed_recall_followup'}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    for c in needed - {'prim_key'}:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=list(needed - {'prim_key'})).reset_index(drop=True)
    if df['prim_key'].isna().any() or df['prim_key'].duplicated().any():
        raise ValueError('prim_key must be unique and nonmissing')
    if not np.allclose(df['P_i_w1'] + df['M_i_w1'] + df['R_i_w1'], df['T_i_w1']):
        raise ValueError('P + M + R != T; cannot compare feature sets')
    print('Participants:', len(df))
    y = df['delayed_recall_followup'].to_numpy(float)
    splits = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(df))
    metrics, predictions = [], []
    for model in ['Ridge', 'RandomForest']:
        for fs, features in FEATURE_SETS.items():
            X = df[features]
            for index, (train, test) in enumerate(splits):
                reg = make_estimator(model)
                reg.fit(X.iloc[train], y[train])
                pred = reg.predict(X.iloc[test]); truth = y[test]
                metrics.append(dict(Model=model, FeatureSet=fs, Repeat=index//5+1,
                    Fold=index%5+1, MAE=mean_absolute_error(truth, pred),
                    RMSE=np.sqrt(mean_squared_error(truth, pred)), R2=r2_score(truth, pred)))
                for j, k in enumerate(test):
                    predictions.append(dict(prim_key=df.iloc[k]['prim_key'],
                        EducationLevel=int(df.iloc[k]['education_level']), Model=model,
                        FeatureSet=fs, Repeat=index//5+1, Fold=index%5+1,
                        Actual=float(truth[j]), Predicted=float(pred[j]),
                        AbsError=float(abs(pred[j]-truth[j])),
                        SignedError=float(pred[j]-truth[j])))
            print('Finished', model, fs, flush=True)
    metrics = pd.DataFrame(metrics)
    preds = pd.DataFrame(predictions)
    summary = metrics.groupby(['Model', 'FeatureSet'], as_index=False).agg(
        MAE_mean=('MAE', 'mean'), MAE_sd=('MAE', 'std'),
        RMSE_mean=('RMSE', 'mean'), R2_mean=('R2', 'mean'))
    summary.to_csv(OUT / 'section3_7_model_comparison.csv', index=False)
    preds.to_csv(OUT / 'section3_7_oof_predictions.csv', index=False)
    subgroups = preds.groupby(['Model', 'FeatureSet', 'EducationLevel'], as_index=False).agg(
        N_Predictions=('AbsError', 'size'), MAE=('AbsError', 'mean'),
        MeanSignedError=('SignedError', 'mean'))
    subgroups.to_csv(OUT / 'section3_7_subgroup_comparison.csv', index=False)
    pairs = [('A_TotalOnly', 'B_DecomposedOnly'),
             ('A_Adjusted', 'B_Adjusted'),
             ('A_AdjustedEducationBlind', 'B_AdjustedEducationBlind')]
    boot = pd.DataFrame([paired_bootstrap(preds, model, a, b)
                         for model in ['Ridge', 'RandomForest'] for a, b in pairs])
    boot.to_csv(OUT / 'section3_7_paired_model_comparison_bootstrap.csv', index=False)
    rows = []
    for col in ['T_i_w1', 'P_i_w1', 'M_i_w1', 'R_i_w1']:
        rho, p = spearmanr(df['education_level'], df[col])
        rows.append(dict(Feature=col, N=len(df), SpearmanRho=float(rho), P=float(p)))
    pd.DataFrame(rows).to_csv(OUT / 'section3_7_education_spearman.csv', index=False)
    # Plot direct paired MAE comparison.
    fig, ax = plt.subplots(figsize=(11, 5))
    display = summary[summary['FeatureSet'].isin(['A_Adjusted', 'B_Adjusted'])]
    for model, group in display.groupby('Model'):
        x = 0 if model == 'Ridge' else 1
        a = float(group.loc[group.FeatureSet == 'A_Adjusted', 'MAE_mean'].iloc[0])
        b = float(group.loc[group.FeatureSet == 'B_Adjusted', 'MAE_mean'].iloc[0])
        ax.bar(x-0.18, a, width=.34, label='Total score' if x == 0 else None)
        ax.bar(x+0.18, b, width=.34, label='Decomposed SPC' if x == 0 else None)
    ax.set_xticks([0, 1], ['Ridge', 'Random Forest'])
    ax.set_ylabel('Repeated CV MAE (words)')
    ax.set_title('Proposal-aligned adjusted models: total vs decomposed recall')
    ax.legend(); fig.tight_layout()
    fig.savefig(FIG / 'section3_7_proposal_aligned_mae.png', dpi=180)
    plt.close(fig)
    print('\nMODEL COMPARISON\n', summary.to_string(index=False))
    print('\nPAIRED COMPARISON\n', boot.to_string(index=False))
    print('\nSECTION 3.7 COMPLETE')
    print('No clinical diagnostic labels were used. Accuracy/AUC/kappa not estimated.')
    print('T=P+M+R: decomposition redistributes information; it does not create new observations.')
    print('Education categories are ordinal categories, NOT years of schooling.')

if __name__ == '__main__':
    run()
