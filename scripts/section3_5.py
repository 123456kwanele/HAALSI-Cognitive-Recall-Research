"""Section 3.5: HAALSI longitudinal SPC dynamics, robustness, supervisor metrics.
Run: python scripts/section3_5.py
Dependencies: numpy pandas scipy matplotlib scikit-learn statsmodels (optional for adjusted models)
Uses existing Section 3.4 out-of-fold predictions; does NOT retrain or invent labels.
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
from scipy import stats
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             explained_variance_score)

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
OUT.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)
SEED = 42
BOOTSTRAPS = 2000
RNG = np.random.default_rng(SEED)


def save(df, name):
    df.to_csv(OUT / name, index=False)
    print(f'Saved {name}: {len(df)} rows')


def ci_mean(x, n=BOOTSTRAPS):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    means = np.array([np.mean(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def load_paired():
    path = DATA / 'HAALSI_Paired_Delayed_Recall_Analysis.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing paired dataset: {path}. Run scripts/02_understand_dataset.py first.')
    df = pd.read_csv(path)
    needed = ['prim_key', 'education_level', 'age_baseline',
              'P_i_w1', 'M_i_w1', 'R_i_w1', 'T_i_w1',
              'P_i_w2', 'M_i_w2', 'R_i_w2', 'T_i_w2',
              'delayed_recall_baseline', 'delayed_recall_followup']
    missing = sorted(set(needed) - set(df))
    if missing:
        raise ValueError(f'Paired dataset missing columns: {missing}')
    for c in needed:
        if c != 'prim_key':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=needed).copy()
    if df.prim_key.duplicated().any():
        raise ValueError('Duplicate participant IDs in paired data.')
    if not df.education_level.isin([1, 2, 3, 4]).all():
        raise ValueError('Education levels must be 1–4 (ordinal categories).')
    for w in ('w1', 'w2'):
        if not np.allclose(df[f'P_i_{w}'] + df[f'M_i_{w}'] + df[f'R_i_{w}'], df[f'T_i_{w}']):
            raise ValueError(f'P+M+R != T for {w}')
        df[f'primacy_rate_{w}'] = df[f'P_i_{w}'] / 3
        df[f'middle_rate_{w}'] = df[f'M_i_{w}'] / 4
        df[f'recency_rate_{w}'] = df[f'R_i_{w}'] / 3
        df[f'primacy_recency_gap_{w}'] = df[f'primacy_rate_{w}'] - df[f'recency_rate_{w}']
    for measure in ['P_i', 'M_i', 'R_i', 'T_i', 'primacy_rate', 'middle_rate',
                    'recency_rate', 'primacy_recency_gap']:
        df[f'delta_{measure}'] = df[f'{measure}_w2'] - df[f'{measure}_w1']
    df['delta_delayed_recall'] = df.delayed_recall_followup - df.delayed_recall_baseline
    return df.reset_index(drop=True)


def paired_dynamics(df):
    print('\n1. PAIRED LONGITUDINAL SPC DYNAMICS')
    metrics = [('Primacy rate', 'primacy_rate'), ('Middle rate', 'middle_rate'),
               ('Recency rate', 'recency_rate'), ('Primacy–recency rate gap', 'primacy_recency_gap'),
               ('Total immediate recall', 'T_i'), ('Delayed recall', 'delayed_recall')]
    rows = []
    for label, var in metrics:
        if var == 'delayed_recall':
            a, b = df.delayed_recall_baseline.to_numpy(), df.delayed_recall_followup.to_numpy()
            change = df.delta_delayed_recall.to_numpy()
        else:
            a, b = df[f'{var}_w1'].to_numpy(), df[f'{var}_w2'].to_numpy()
            change = df[f'delta_{var}'].to_numpy()
        try:
            wilcoxon_p = stats.wilcoxon(change, zero_method='wilcox').pvalue if np.any(change != 0) else 1.0
        except ValueError:
            wilcoxon_p = np.nan
        low, high = ci_mean(change)
        rows.append(dict(Measure=label, N=len(change), W1_Mean=np.mean(a), W2_Mean=np.mean(b),
                         Mean_Change=np.mean(change), CI95_Lower=low, CI95_Upper=high,
                         SD_Change=np.std(change, ddof=1), Wilcoxon_P=wilcoxon_p))
    results = pd.DataFrame(rows)
    save(results, 'section3_5_paired_spc_changes.csv')
    print(results.round(4).to_string(index=False))

    groups = []
    for edu, g in df.groupby('education_level'):
        for label, var in metrics:
            delta = g.delta_delayed_recall if var == 'delayed_recall' else g[f'delta_{var}']
            low, high = ci_mean(delta)
            groups.append(dict(EducationCategory=int(edu), N=len(g), Measure=label,
                               MeanChange=delta.mean(), CI95_Lower=low, CI95_Upper=high))
    groups = pd.DataFrame(groups)
    save(groups, 'section3_5_spc_change_by_education.csv')

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, var in [('Primacy', 'primacy_rate'), ('Middle', 'middle_rate'), ('Recency', 'recency_rate')]:
        ax.plot([1, 2], [df[f'{var}_w1'].mean(), df[f'{var}_w2'].mean()], 'o-', label=name)
    ax.set(xticks=[1, 2], xticklabels=['Wave 1', 'Wave 2'], ylabel='Mean proportion recalled',
           title=f'Serial-position recall rates (paired participants, N={len(df)})', ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'section3_5_paired_spc_trajectories.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, var in [('Primacy', 'primacy_rate'), ('Middle', 'middle_rate'), ('Recency', 'recency_rate')]:
        means = [g[f'delta_{var}'].mean() for _, g in df.groupby('education_level')]
        ax.plot(sorted(df.education_level.unique()), means, marker='o', label=name)
    ax.axhline(0, color='gray', lw=1, linestyle='--')
    ax.set(xlabel='Education category (ordinal, not years)', ylabel='Mean W2 − W1 rate',
           title='Changes in SPC components by education category', xticks=[1, 2, 3, 4])
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'section3_5_spc_changes_by_education.png', dpi=180)
    plt.close(fig)
    return results


def adjusted_analyses(df):
    print('\n2. EDUCATION × SPC-REGION LONGITUDINAL INTERACTION')
    if not HAS_STATSMODELS:
        print('statsmodels not installed; skipping adjusted regression. Install with python -m pip install statsmodels')
        return
    region_data = []
    for reg in ('primacy', 'middle', 'recency'):
        x = df[['prim_key', 'education_level', 'age_baseline', 'T_i_w1',
                'delayed_recall_baseline']].copy()
        x['Region'] = reg
        x['Change'] = df[f'delta_{reg}_rate'].to_numpy()
        region_data.append(x)
    long = pd.concat(region_data, ignore_index=True)
    # Participant-clustered covariance accounts for three within-person regions.
    fit = smf.ols('Change ~ C(Region) * C(education_level) + age_baseline + T_i_w1', data=long).fit(
        cov_type='cluster', cov_kwds={'groups': long.prim_key})
    terms = [t for t in fit.params.index if 'C(Region)' in t and ':C(education_level)' in t]
    rows = [{'Term': t, 'Coefficient': fit.params[t], 'SE_Cluster': fit.bse[t],
             'P_Value': fit.pvalues[t]} for t in fit.params.index]
    if terms:
        constraint = ', '.join(f'{t} = 0' for t in terms)
        try:
            test = fit.wald_test(constraint, scalar=True)
            rows.append({'Term': 'JOINT_REGION_X_EDUCATION', 'Coefficient': float(test.statistic),
                         'SE_Cluster': np.nan, 'P_Value': float(test.pvalue)})
            print(f'Joint Region × Education Wald statistic: {float(test.statistic):.4f}; p={float(test.pvalue):.5g}')
        except Exception as e:
            print('Joint Wald test could not be evaluated:', e)
    save(pd.DataFrame(rows), 'section3_5_region_education_interaction.csv')

    # Sensitivity of change in SPC shape to baseline age, baseline recall and education.
    # Association only: do not interpret adjustment as causal or practice-effect correction.
    for outcome in ('delta_primacy_recency_gap', 'delta_T_i', 'delta_delayed_recall'):
        model = smf.ols(f'{outcome} ~ C(education_level) + age_baseline + T_i_w1 + delayed_recall_baseline',
                        data=df).fit(cov_type='HC3')
        table = pd.DataFrame({'Term': model.params.index, 'Coefficient': model.params.values,
                              'SE_HC3': model.bse.values, 'P_Value': model.pvalues.values})
        save(table, f'section3_5_adjusted_{outcome}.csv')
        print(f'{outcome}: adjusted R²={model.rsquared_adj:.4f}')


def sensitivity(df):
    print('\n3. SENSITIVITY / ADMINISTRATION-ISSUE ANALYSIS')
    subsets = [('All paired', df), ('Age < 75', df[df.age_baseline < 75]),
               ('Age >= 75', df[df.age_baseline >= 75])]
    flag_cols = [c for c in ['recall_admin_issue_w1', 'recall_admin_issue_w2'] if c in df.columns]
    if flag_cols:
        clean = df.copy()
        for col in flag_cols:
            vals = pd.to_numeric(clean[col], errors='coerce')
            clean = clean[vals.eq(0)]
        subsets.append(('No recorded administration issue', clean))
    else:
        print('No administration-issue flags found; sensitivity subset unavailable.')
    rows = []
    for name, subset in subsets:
        if len(subset) < 5:
            continue
        for measure in ['delta_primacy_rate', 'delta_middle_rate', 'delta_recency_rate',
                        'delta_primacy_recency_gap', 'delta_T_i', 'delta_delayed_recall']:
            low, high = ci_mean(subset[measure])
            rows.append(dict(Subset=name, N=len(subset), Measure=measure,
                             MeanChange=subset[measure].mean(), CI95_Lower=low, CI95_Upper=high))
    res = pd.DataFrame(rows)
    save(res, 'section3_5_sensitivity_changes.csv')
    print(res.round(4).to_string(index=False))


def attrition_audit(paired):
    print('\n4. FOLLOW-UP SELECTION / ATTRITION AUDIT')
    panel_path = DATA / 'HAALSI_Narrowed_SPC_Dataset.csv'
    # The narrow file is not guaranteed to contain baseline age/education/status.
    # Prefer the enriched long panel from preprocessing.
    candidates = [DATA / 'HAALSI_Narrowed_SPC_Dataset.csv',
                  DATA / 'HAALSI_Longitudinal_SPC_Dataset.csv']
    panel_path = DATA / 'HAALSI_Narrowed_SPC_Dataset.csv'
    enriched = DATA / 'HAALSI_Narrowed_SPC_Dataset.csv'
    # Search long panel first, then wide; never equate 'not in paired outcome analysis' with lost to follow-up.
    long_candidates = [DATA / 'HAALSI_Narrowed_SPC_Dataset.csv']
    for path in long_candidates:
        if not path.exists():
            continue
        p = pd.read_csv(path)
        if not {'prim_key', 'age_baseline', 'education_level', 'wave'}.issubset(p.columns):
            continue
        w1 = p[p.wave.astype(str).str.lower().isin(['1', 'w1', 'wave 1'])].copy()
        if not len(w1):
            continue
        w1 = w1.drop_duplicates('prim_key')
        w1['IncludedInPairedAnalysis'] = w1.prim_key.isin(set(paired.prim_key))
        rows = []
        for included, g in w1.groupby('IncludedInPairedAnalysis'):
            rows.append(dict(IncludedInPairedAnalysis=bool(included), N=len(g),
                             MeanAge=pd.to_numeric(g.age_baseline, errors='coerce').mean(),
                             MeanEducationCategory=pd.to_numeric(g.education_level, errors='coerce').mean()))
        save(pd.DataFrame(rows), 'section3_5_baseline_selection_audit.csv')
        print('Baseline inclusion is NOT equivalent to verified follow-up attrition.')
        return
    print('Enriched long panel with age, education and wave unavailable; skipping selection audit.')


def supervisor_metrics():
    print('\n5. SUPERVISOR PERFORMANCE REPORT: OUT-OF-FOLD PREDICTIONS')
    path = OUT / 'section3_4_ablation_oof_predictions.csv'
    if not path.exists():
        print(f'No predictions at {path}. Run section3_4.py first; cannot calculate tolerance accuracy honestly.')
        return
    p = pd.read_csv(path)
    needed = {'prim_key', 'Model', 'FeatureSet', 'EducationMode', 'Repeat', 'Fold', 'y_true', 'y_pred'}
    if not needed.issubset(p.columns):
        raise ValueError(f'Prediction file missing columns: {sorted(needed - set(p.columns))}')
    if p.duplicated(['prim_key', 'Model', 'FeatureSet', 'EducationMode', 'Repeat']).any():
        raise ValueError('Duplicate out-of-fold participant/model/repeat rows. Review CV export.')
    p['y_true'] = pd.to_numeric(p.y_true, errors='coerce')
    p['y_pred'] = pd.to_numeric(p.y_pred, errors='coerce')
    if p[['y_true', 'y_pred']].isna().any().any():
        raise ValueError('Missing outcome/prediction values in OOF file.')
    fold_metrics = []
    for key, g in p.groupby(['Model', 'FeatureSet', 'EducationMode', 'Repeat', 'Fold']):
        y, pred = g.y_true.to_numpy(), g.y_pred.to_numpy()
        fold_metrics.append(dict(zip(['Model', 'FeatureSet', 'EducationMode', 'Repeat', 'Fold'], key),
                                 MAE=mean_absolute_error(y, pred),
                                 RMSE=np.sqrt(mean_squared_error(y, pred)), R2=r2_score(y, pred),
                                 ExplainedVariance=explained_variance_score(y, pred),
                                 MeanSignedError=np.mean(pred-y),
                                 Within1Word=np.mean(np.abs(pred-y) <= 1),
                                 Within2Words=np.mean(np.abs(pred-y) <= 2),
                                 RoundedExactMatch=np.mean(np.rint(pred) == y)))
    folds = pd.DataFrame(fold_metrics)
    save(folds, 'section3_5_supervisor_fold_metrics.csv')
    metrics = ['MAE', 'RMSE', 'R2', 'ExplainedVariance', 'MeanSignedError',
               'Within1Word', 'Within2Words', 'RoundedExactMatch']
    agg = {k: ['mean', 'std'] for k in metrics}
    summary = folds.groupby(['Model', 'FeatureSet', 'EducationMode']).agg(agg)
    summary.columns = ['_'.join(x) for x in summary.columns]
    summary = summary.reset_index().sort_values('MAE_mean')
    summary['Within1Word_pct'] = 100 * summary['Within1Word_mean']
    summary['Within2Words_pct'] = 100 * summary['Within2Words_mean']
    summary['RoundedExactMatch_pct'] = 100 * summary['RoundedExactMatch_mean']
    save(summary, 'section3_5_supervisor_model_scorecard.csv')
    print('\nSUPERVISOR SCORECARD (with education):')
    cols = ['Model', 'FeatureSet', 'MAE_mean', 'RMSE_mean', 'R2_mean',
            'ExplainedVariance_mean', 'Within1Word_pct', 'Within2Words_pct',
            'RoundedExactMatch_pct', 'MeanSignedError_mean']
    print(summary[summary.EducationMode == 'WithEducation'][cols].round(4).to_string(index=False))

    # Participant-level uncertainty: average three out-of-fold predictions per participant,
    # then resample people, never 15 correlated CV folds as independent observations.
    person = p.groupby(['prim_key', 'Model', 'FeatureSet', 'EducationMode'], as_index=False).agg(
        y_true=('y_true', 'first'), y_pred=('y_pred', 'mean'))
    person['AbsoluteError'] = np.abs(person.y_pred - person.y_true)
    person['Within1Word'] = (person.AbsoluteError <= 1).astype(int)
    person['Within2Words'] = (person.AbsoluteError <= 2).astype(int)
    save(person, 'section3_5_supervisor_participant_predictions.csv')
    cis = []
    for key, g in person.groupby(['Model', 'FeatureSet', 'EducationMode']):
        for metric in ['AbsoluteError', 'Within1Word', 'Within2Words']:
            low, high = ci_mean(g[metric].to_numpy())
            cis.append(dict(Model=key[0], FeatureSet=key[1], EducationMode=key[2],
                            Metric=metric, Estimate=g[metric].mean(), CI95_Lower=low, CI95_Upper=high,
                            Unit='proportion' if metric.startswith('Within') else 'words'))
    save(pd.DataFrame(cis), 'section3_5_supervisor_participant_bootstrap.csv')

    best = summary[(summary.EducationMode == 'WithEducation') &
                   (((summary.Model == 'KAN') & (summary.FeatureSet == 'C_DecomposedSPC')) |
                    ((summary.Model == 'Ridge') & (summary.FeatureSet == 'B_TotalRecall')))]
    if len(best):
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = best.Model + ' · ' + best.FeatureSet
        ax.barh(labels, best.MAE_mean)
        ax.set(xlabel='Repeated CV mean MAE (words)', title='KAN and Ridge longitudinal recall prediction')
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(FIG / 'section3_5_supervisor_best_model_mae.png', dpi=180)
        plt.close(fig)

    note = '''SUPERVISOR METRIC DEFINITIONS\n\nOutcome: continuous Wave 2 CERAD delayed-recall score (words).\nRepeated CV: 5 folds x 3 repeats. Each fold is evaluated on held-out participants.\nMAE: average absolute prediction error in words; lower is better.\nRMSE: root mean squared prediction error in words; lower is better.\nR2: held-out coefficient of determination; higher is better, negative values possible.\nExplained variance: 1 - Var(prediction error)/Var(observed); distinct from R2.\nMean signed error: prediction minus observed; positive=overprediction.\nWithin1Word/Within2Words: proportion of held-out predictions with absolute error <=1/<=2 words.\nRoundedExactMatch: exact match after rounding continuous predictions; descriptive only.\nThese tolerance measures are NOT clinical classification accuracy.\nClassification accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC,\nand diagnostic false-positive rates are NOT defined without validated categorical labels.\n\nCaution: CV repeats are correlated. Participant bootstrap is descriptive, not\nan external-validation CI. Scores are not independent prospective validation.\nNo claim of observed clinical dementia or confirmed cognitive decline.\n'''
    (OUT / 'section3_5_supervisor_metric_definitions.txt').write_text(note, encoding='utf-8')


def main():
    print('Matplotlib backend:', matplotlib.get_backend())
    print('\nSECTION 3.5: LONGITUDINAL SPC DYNAMICS + SUPERVISOR REPORTING')
    df = load_paired()
    print(f'Paired analysis participants: {len(df)}; education counts: {df.education_level.value_counts().sort_index().to_dict()}')
    paired_dynamics(df)
    adjusted_analyses(df)
    sensitivity(df)
    attrition_audit(df)
    supervisor_metrics()
    print('\nSECTION 3.5 COMPLETED')
    print('Tables:', OUT)
    print('Figures:', FIG)
    print('Interpret longitudinal change cautiously: practice, selection, aging and administration may contribute.')
    print('No classification accuracy or diagnostic outcomes were fabricated.')

if __name__ == '__main__':
    main()
