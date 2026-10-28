"""Section 3.4: education ablation and subgroup fairness for HAALSI recall.
Run: python scripts/section3_4.py
Dependencies: numpy pandas matplotlib scikit-learn torch
"""
import os
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import copy
import random
import warnings

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'data' / 'HAALSI_Paired_Delayed_Recall_Analysis.csv'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SPLITS = 5
N_REPEATS = 3
EPOCHS = 250
PATIENCE = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.003
GRID_SIZE = 6
DEGREE = 3
HIDDEN_UNITS = 6
FEATURE_SETS = {
    'A_Conventional': ['age_baseline', 'education_level', 'delayed_recall_baseline'],
    'B_TotalRecall': ['age_baseline', 'education_level', 'delayed_recall_baseline', 'T_i_w1'],
    'C_DecomposedSPC': ['age_baseline', 'education_level', 'delayed_recall_baseline', 'P_i_w1', 'M_i_w1', 'R_i_w1'],
    'D_TotalPlusShape': ['age_baseline', 'education_level', 'delayed_recall_baseline', 'T_i_w1', 'primacy_recency_contrast', 'middle_outer_contrast'],
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def bspline_basis(x, knots, degree):
    """Differentiable Cox-de Boor B-spline basis. x: [batch, inputs]."""
    x = x.unsqueeze(-1)
    t = knots
    basis = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
    # Include rightmost endpoint of knot domain.
    basis = torch.where((x == t[-1]) & (torch.arange(len(t)-1, device=x.device) == len(t)-2), torch.ones_like(basis), basis)
    for k in range(1, degree + 1):
        n = len(t) - k - 1
        left_denom = t[k:k+n] - t[:n]
        right_denom = t[k+1:k+1+n] - t[1:n+1]
        left = torch.where(left_denom > 0, (x - t[:n]) / torch.clamp(left_denom, min=1e-12), torch.zeros_like(x)) * basis[..., :n]
        right = torch.where(right_denom > 0, (t[k+1:k+1+n] - x) / torch.clamp(right_denom, min=1e-12), torch.zeros_like(x)) * basis[..., 1:n+1]
        basis = left + right
    return basis


class KANLayer(nn.Module):
    """Each output sums learned univariate B-spline edge functions and linear skip terms."""
    def __init__(self, in_features, out_features, grid_size=GRID_SIZE, degree=DEGREE):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        # Uniform knots with repeated endpoints; standardized inputs are clamped to [-3,3].
        interior = torch.linspace(-3, 3, grid_size + 1)
        knots = torch.cat([interior[:1].repeat(degree), interior, interior[-1:].repeat(degree)])
        self.register_buffer('knots', knots)
        n_basis = len(knots) - degree - 1
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, n_basis) * 0.025)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = x.clamp(-3, 3)
        basis = bspline_basis(x, self.knots, self.degree)
        spline = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        linear = torch.einsum('bi,oi->bo', x, self.base_weight)
        return spline + linear + self.bias

    @torch.no_grad()
    def edge_curve(self, input_index, output_index, grid):
        x = torch.as_tensor(grid, dtype=self.spline_weight.dtype).reshape(-1, 1)
        basis = bspline_basis(x, self.knots, self.degree).squeeze(1)
        return (basis @ self.spline_weight[output_index, input_index] + x.squeeze(1) * self.base_weight[output_index, input_index]).cpu().numpy()


class KANRegressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.first = KANLayer(n_features, HIDDEN_UNITS)
        self.second = KANLayer(HIDDEN_UNITS, 1)

    def forward(self, x):
        hidden = torch.tanh(self.first(x))
        return self.second(hidden).squeeze(-1)


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f'Missing input: {DATA_FILE}')
    df = pd.read_csv(DATA_FILE)
    required = {'prim_key', 'delayed_recall_followup', 'P_i_w1', 'M_i_w1', 'R_i_w1', 'T_i_w1'}
    required.update(sum(FEATURE_SETS.values(), []))
    required.discard('primacy_recency_contrast')
    required.discard('middle_outer_contrast')
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df['prim_key'].isna().any() or df['prim_key'].duplicated().any():
        raise ValueError('Each participant must have one unique nonmissing prim_key.')
    for col in required - {'prim_key'}:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=list(required - {'prim_key'})).copy()
    if not np.allclose(df['P_i_w1'] + df['M_i_w1'] + df['R_i_w1'], df['T_i_w1']):
        raise ValueError('Baseline P + M + R does not equal T.')
    if not df['education_level'].isin([1, 2, 3, 4]).all():
        raise ValueError('Unexpected education categories.')
    if not df['delayed_recall_followup'].between(0, 10).all():
        raise ValueError('Outcome outside 0–10.')
    df['primacy_recency_contrast'] = df['P_i_w1'] / 3 - df['R_i_w1'] / 3
    df['middle_outer_contrast'] = df['M_i_w1'] / 4 - (df['P_i_w1'] / 3 + df['R_i_w1'] / 3) / 2
    return df.reset_index(drop=True)


def train_kan(X_train, y_train, seed):
    """Validation split and preprocessing are strictly within the outer training fold."""
    seed_everything(seed)
    fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size=0.18, random_state=seed)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(imputer.fit_transform(X_train.iloc[fit_idx]))
    x_val = scaler.transform(imputer.transform(X_train.iloc[val_idx]))
    y_mean = float(np.mean(y_train[fit_idx]))
    y_std = float(np.std(y_train[fit_idx]))
    y_std = max(y_std, 1e-6)
    yy_fit = (y_train[fit_idx] - y_mean) / y_std
    yy_val = (y_train[val_idx] - y_mean) / y_std
    model = KANRegressor(X_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    x_fit = torch.tensor(x_fit, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    yy_fit = torch.tensor(yy_fit, dtype=torch.float32)
    yy_val = torch.tensor(yy_val, dtype=torch.float32)
    best_loss = float('inf')
    best_state = None
    stale = 0
    best_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(len(x_fit))
        for batch in order.split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = criterion(model(x_fit[batch]), yy_fit[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), yy_val).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, imputer, scaler, y_mean, y_std, best_epoch


def predict(model, imputer, scaler, y_mean, y_std, X):
    transformed = scaler.transform(imputer.transform(X))
    with torch.no_grad():
        pred = model(torch.tensor(transformed, dtype=torch.float32)).numpy()
    return pred * y_std + y_mean



# ============================================================
# SECTION 3.4: EDUCATION ABLATION AND PREDICTIVE FAIRNESS
# Uses the exact same outer CV splits as Sections 3.2 and 3.3.
# Within each fold, train WITH and WITHOUT education.
# Bootstrap resamples participants, not correlated CV rows.
# ============================================================
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

BOOTSTRAPS = 2000
ABLATION_FEATURE_SETS = FEATURE_SETS  # Four feature sets, no arbitrary diagnostic labels.


def ridge_model():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=10.0)),
    ])


def validate_and_load():
    df = load_data()
    if len(df) < N_SPLITS:
        raise ValueError('Insufficient complete observations.')
    print('Participants:', len(df), '| Education counts:',
          df.education_level.value_counts().sort_index().to_dict())
    return df


def train_ablation(df):
    y = df['delayed_recall_followup'].to_numpy(float)
    splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                               random_state=SEED).split(df))
    fold_rows, pred_rows = [], []
    total = 2 * 2 * len(ABLATION_FEATURE_SETS) * len(splits)
    done = 0
    for model_name in ['Ridge', 'KAN']:
        for fs, full_features in ABLATION_FEATURE_SETS.items():
            for education_mode in ['WithEducation', 'WithoutEducation']:
                features = ([x for x in full_features if x != 'education_level']
                            if education_mode == 'WithoutEducation' else list(full_features))
                X = df[features]
                for split_i, (tr, te) in enumerate(splits):
                    repeat, fold = divmod(split_i, N_SPLITS)
                    if model_name == 'Ridge':
                        fitted = ridge_model().fit(X.iloc[tr], y[tr])
                        predicted = fitted.predict(X.iloc[te])
                        epochs = np.nan
                    else:
                        trained = train_kan(X.iloc[tr], y[tr], SEED + split_i)
                        model, imp, scale, ym, ys, epochs = trained
                        predicted = predict(model, imp, scale, ym, ys, X.iloc[te])
                    true = y[te]
                    fold_rows.append({
                        'Model': model_name, 'FeatureSet': fs, 'EducationMode': education_mode,
                        'Repeat': repeat + 1, 'Fold': fold + 1, 'N_Test': len(te),
                        'MAE': mean_absolute_error(true, predicted),
                        'RMSE': np.sqrt(mean_squared_error(true, predicted)),
                        'R2': r2_score(true, predicted), 'BestEpoch': epochs,
                    })
                    for j, row_i in enumerate(te):
                        pred_rows.append({
                            'prim_key': df.iloc[row_i]['prim_key'],
                            'education_level': int(df.iloc[row_i]['education_level']),
                            'Model': model_name, 'FeatureSet': fs,
                            'EducationMode': education_mode, 'Repeat': repeat + 1,
                            'Fold': fold + 1, 'y_true': float(true[j]),
                            'y_pred': float(predicted[j]),
                            'error': float(predicted[j] - true[j]),
                            'absolute_error': float(abs(predicted[j] - true[j])),
                        })
                    done += 1
                    print(f'\rCompleted {done}/{total} model fits', end='', flush=True)
    print()
    folds, preds = pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)
    folds.to_csv(OUT / 'section3_4_ablation_fold_metrics.csv', index=False)
    preds.to_csv(OUT / 'section3_4_ablation_oof_predictions.csv', index=False)
    return folds, preds


def summarize(folds, preds):
    overall = folds.groupby(['Model', 'FeatureSet', 'EducationMode']).agg(
        MAE_mean=('MAE', 'mean'), MAE_std=('MAE', 'std'),
        RMSE_mean=('RMSE', 'mean'), R2_mean=('R2', 'mean')).reset_index()
    overall.to_csv(OUT / 'section3_4_ablation_performance.csv', index=False)
    print('\nOVERALL PERFORMANCE\n', overall.round(4).to_string(index=False))

    # Each person appears once in each repetition. Averaging absolute errors
    # gives a per-person repeated-CV MAE contribution, not 3 independent people.
    people = preds.groupby(['prim_key', 'education_level', 'Model',
                            'FeatureSet', 'EducationMode'], as_index=False).agg(
        y_true=('y_true', 'first'), y_pred=('y_pred', 'mean'),
        abs_error=('absolute_error', 'mean'), error=('error', 'mean'))
    people.to_csv(OUT / 'section3_4_participant_level_errors.csv', index=False)

    subgroup = people.groupby(['Model', 'FeatureSet', 'EducationMode',
                               'education_level'], as_index=False).agg(
        N=('prim_key', 'nunique'), MAE=('abs_error', 'mean'),
        MeanError=('error', 'mean'))
    subgroup.to_csv(OUT / 'section3_4_education_subgroup_errors.csv', index=False)
    print('\nSUBGROUP ERRORS\n', subgroup.round(4).to_string(index=False))
    return people, subgroup


def participant_bootstrap(people):
    # Paired comparisons on the same participants. Bootstrap the paired
    # participant error differences, preserving within-person dependence.
    rng = np.random.default_rng(SEED)
    comparisons = [
        ('Education ablation: without minus with', 'WithEducation', 'WithoutEducation'),
    ]
    rows = []
    for (model, fs), group in people.groupby(['Model', 'FeatureSet']):
        with_ed = group[group.EducationMode == 'WithEducation'].set_index('prim_key')
        without_ed = group[group.EducationMode == 'WithoutEducation'].set_index('prim_key')
        common = with_ed.index.intersection(without_ed.index)
        if len(common) == 0:
            continue
        a = with_ed.loc[common]
        b = without_ed.loc[common]
        if not np.array_equal(a.education_level.to_numpy(), b.education_level.to_numpy()):
            raise ValueError('Education groups misaligned during bootstrap.')
        delta = (b.abs_error - a.abs_error).to_numpy()
        for label, mask in [('All', np.ones(len(delta), dtype=bool))] + [
            (f'EducationCategory{k}', (a.education_level.to_numpy() == k))
            for k in [1, 2, 3, 4]
        ]:
            d = delta[mask]
            if len(d) < 2:
                continue
            indices = rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))
            means = d[indices].mean(axis=1)
            rows.append({
                'Model': model, 'FeatureSet': fs, 'Subgroup': label,
                'N': len(d), 'Comparison': comparisons[0][0],
                'DeltaMAE_WithoutMinusWith': float(d.mean()),
                'CI95_Lower': float(np.quantile(means, .025)),
                'CI95_Upper': float(np.quantile(means, .975)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'section3_4_education_ablation_bootstrap.csv', index=False)
    print('\nABLATION: POSITIVE DELTA MEANS REMOVING EDUCATION HURTS\n',
          result[result.Subgroup == 'All'].round(4).to_string(index=False))
    return result


def fairness_bootstrap(people):
    # Fairness is measured as differences in *predictive* errors.
    # Gap = max(group MAE) - min(group MAE); this is not clinical FPR.
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for (model, fs, mode), g in people.groupby(['Model', 'FeatureSet', 'EducationMode']):
        groups = {int(k): sub for k, sub in g.groupby('education_level')}
        if set(groups) != {1, 2, 3, 4}:
            continue
        boot_maes, boot_biases = {}, {}
        for k, sub in groups.items():
            ae = sub.abs_error.to_numpy()
            err = sub.error.to_numpy()
            idx = rng.integers(0, len(sub), size=(BOOTSTRAPS, len(sub)))
            boot_maes[k] = ae[idx].mean(axis=1)
            boot_biases[k] = err[idx].mean(axis=1)
        mae_by_group = np.array([groups[k].abs_error.mean() for k in [1,2,3,4]])
        bias_by_group = np.array([groups[k].error.mean() for k in [1,2,3,4]])
        mae_gap_boot = np.ptp(np.column_stack([boot_maes[k] for k in [1,2,3,4]]), axis=1)
        bias_gap_boot = np.ptp(np.column_stack([boot_biases[k] for k in [1,2,3,4]]), axis=1)
        rows.append({
            'Model': model, 'FeatureSet': fs, 'EducationMode': mode,
            'OverallMAE': g.abs_error.mean(),
            'MAE_GroupGap': np.ptp(mae_by_group),
            'MAE_Gap_CI95_Lower': np.quantile(mae_gap_boot, .025),
            'MAE_Gap_CI95_Upper': np.quantile(mae_gap_boot, .975),
            'SignedError_GroupGap': np.ptp(bias_by_group),
            'SignedError_Gap_CI95_Lower': np.quantile(bias_gap_boot, .025),
            'SignedError_Gap_CI95_Upper': np.quantile(bias_gap_boot, .975),
        })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'section3_4_fairness_gap_bootstrap.csv', index=False)
    print('\nPREDICTIVE FAIRNESS GAPS (smaller is better)\n',
          result.round(4).to_string(index=False))
    return result


def spc_fairness_comparison(people):
    # Paired comparison: does C or D improve subgroup errors vs B?
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for (model, mode), g in people.groupby(['Model', 'EducationMode']):
        base = g[g.FeatureSet == 'B_TotalRecall'].set_index('prim_key')
        for candidate in ['C_DecomposedSPC', 'D_TotalPlusShape']:
            alt = g[g.FeatureSet == candidate].set_index('prim_key')
            ids = base.index.intersection(alt.index)
            a, b = base.loc[ids], alt.loc[ids]
            delta = (a.abs_error - b.abs_error).to_numpy()
            for group_name, mask in [('All', np.ones(len(ids), bool))] + [
                (f'EducationCategory{k}', a.education_level.to_numpy() == k)
                for k in [1,2,3,4]
            ]:
                d = delta[mask]
                if len(d) < 2:
                    continue
                sample = rng.integers(0, len(d), size=(BOOTSTRAPS, len(d)))
                means = d[sample].mean(axis=1)
                rows.append({
                    'Model': model, 'EducationMode': mode,
                    'Comparison': f'{candidate} vs B_TotalRecall',
                    'Subgroup': group_name, 'N': len(d),
                    'MAE_Improvement_PositiveIsBetter': d.mean(),
                    'CI95_Lower': np.quantile(means, .025),
                    'CI95_Upper': np.quantile(means, .975),
                })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'section3_4_spc_subgroup_comparisons.csv', index=False)
    print('\nSPC VS TOTAL RECALL: POSITIVE MEANS SPC BETTER\n',
          result[result.Subgroup == 'All'].round(4).to_string(index=False))
    return result


def figures(subgroup, fairness):
    for model in ['Ridge', 'KAN']:
        sub = subgroup[(subgroup.Model == model) &
                       (subgroup.FeatureSet.isin(['B_TotalRecall', 'C_DecomposedSPC']))]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, mode in zip(axes, ['WithEducation', 'WithoutEducation']):
            d = sub[sub.EducationMode == mode]
            for fs, part in d.groupby('FeatureSet'):
                ax.plot(part.education_level, part.MAE, marker='o', label=fs)
            ax.set_title(mode)
            ax.set_xlabel('Education category (ordinal 1–4)')
            ax.set_xticks([1, 2, 3, 4])
            ax.grid(alpha=.25)
        axes[0].set_ylabel('Mean absolute prediction error (words)')
        axes[1].legend()
        fig.suptitle(f'{model}: subgroup predictive errors')
        fig.tight_layout()
        fig.savefig(FIG / f'section3_4_{model.lower()}_subgroup_mae.png', dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    sub = fairness[fairness.EducationMode == 'WithEducation'].copy()
    labels = sub.Model + ' / ' + sub.FeatureSet.str.replace('_', ' ', regex=False)
    ax.barh(labels, sub.MAE_GroupGap)
    ax.set_xlabel('Maximum minus minimum education-group MAE (words)')
    ax.set_title('Predictive error disparities (descriptive)')
    fig.tight_layout()
    fig.savefig(FIG / 'section3_4_education_mae_gaps.png', dpi=200)
    plt.close(fig)


def main():
    print('Matplotlib backend:', matplotlib.get_backend())
    print('\nSECTION 3.4: EDUCATION ABLATION AND PREDICTIVE FAIRNESS')
    df = validate_and_load()
    folds, preds = train_ablation(df)
    people, subgroup = summarize(folds, preds)
    participant_bootstrap(people)
    fairness = fairness_bootstrap(people)
    spc_fairness_comparison(people)
    figures(subgroup, fairness)
    print('\nSECTION 3.4 COMPLETED')
    print('Tables:', OUT, '\nFigures:', FIG)
    print('Note: No clinical dementia labels, false-positive rates, or causal claims.')
    print('Education categories 1–4 have deliberately neutral labels.')
    print('Repeated CV folds are correlated; participant bootstrap is descriptive.')
    print('The 4 education categories have unequal sample sizes; interpret small groups cautiously.')


if __name__ == '__main__':
    main()
