"""Section 3.3: spline-based KAN regression for longitudinal HAALSI recall.
Run: python scripts/section3_3.py
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


def run_cv(df):
    print('\nSECTION 3.3: SPLINE KAN LONGITUDINAL REGRESSION')
    print('Participants:', len(df), '| PyTorch:', torch.__version__)
    y = df['delayed_recall_followup'].to_numpy(dtype=float)
    splits = list(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(df))
    folds, predictions, models = [], [], {}
    for fs, features in FEATURE_SETS.items():
        X = df[features]
        for idx, (train_idx, test_idx) in enumerate(splits):
            repeat, fold = divmod(idx, N_SPLITS)
            trained = train_kan(X.iloc[train_idx], y[train_idx], SEED + idx)
            model, imp, scale, ym, ys, epochs = trained
            pred = predict(model, imp, scale, ym, ys, X.iloc[test_idx])
            truth = y[test_idx]
            folds.append({'Model': 'KAN', 'FeatureSet': fs, 'Repeat': repeat+1, 'Fold': fold+1,
                          'N_Train': len(train_idx), 'N_Test': len(test_idx),
                          'MAE': mean_absolute_error(truth, pred),
                          'RMSE': np.sqrt(mean_squared_error(truth, pred)),
                          'R2': r2_score(truth, pred),
                          'MeanError': float(np.mean(pred-truth)), 'BestEpoch': epochs})
            for j, row_idx in enumerate(test_idx):
                predictions.append({'prim_key': df.iloc[row_idx]['prim_key'], 'Model': 'KAN',
                                    'FeatureSet': fs, 'Repeat': repeat+1, 'Fold': fold+1,
                                    'education_level': df.iloc[row_idx]['education_level'],
                                    'y_true': truth[j], 'y_pred': pred[j],
                                    'error': pred[j]-truth[j], 'absolute_error': abs(pred[j]-truth[j])})
            if idx == 0:
                models[fs] = (trained, features, X.iloc[test_idx].copy(), truth.copy())
            print(f'{fs}: fold {idx+1}/{len(splits)} | MAE={folds[-1]["MAE"]:.3f} | epochs={epochs}', flush=True)
    fold_df = pd.DataFrame(folds)
    pred_df = pd.DataFrame(predictions)
    fold_df.to_csv(OUT / 'section3_3_kan_cv_fold_results.csv', index=False)
    pred_df.to_csv(OUT / 'section3_3_kan_cv_predictions.csv', index=False)
    summary = fold_df.groupby('FeatureSet').agg(MAE_mean=('MAE','mean'), MAE_std=('MAE','std'),
                    RMSE_mean=('RMSE','mean'), RMSE_std=('RMSE','std'),
                    R2_mean=('R2','mean'), R2_std=('R2','std'),
                    BestEpoch_mean=('BestEpoch','mean')).reset_index().sort_values('MAE_mean')
    summary.insert(0, 'Model', 'KAN')
    summary.to_csv(OUT / 'section3_3_kan_model_performance.csv', index=False)
    print('\nKAN MODEL PERFORMANCE\n', summary.round(4).to_string(index=False))
    return fold_df, pred_df, summary, models


def compare_ridge(kan_folds):
    path = OUT / 'section3_2_cv_fold_results.csv'
    if not path.exists():
        print('\nRidge fold comparison skipped: section3_2_cv_fold_results.csv not found.')
        return
    baseline = pd.read_csv(path)
    baseline = baseline[baseline['Model'] == 'Ridge']
    required = {'FeatureSet', 'Repeat', 'Fold', 'MAE', 'RMSE', 'R2'}
    if not required.issubset(baseline.columns):
        print('Ridge fold comparison skipped: columns missing.')
        return
    merged = kan_folds.merge(baseline, on=['FeatureSet','Repeat','Fold'], suffixes=('_KAN','_Ridge'), validate='one_to_one')
    if len(merged) != len(kan_folds):
        raise ValueError('Incomplete KAN-Ridge fold matching; check Section 3.2 CV settings.')
    merged['MAE_Improvement_KAN_over_Ridge'] = merged['MAE_Ridge'] - merged['MAE_KAN']
    merged.to_csv(OUT / 'section3_3_kan_vs_ridge_folds.csv', index=False)
    result = merged.groupby('FeatureSet').agg(Mean_MAE_Improvement=('MAE_Improvement_KAN_over_Ridge','mean'),
                     SD_Fold_Improvement=('MAE_Improvement_KAN_over_Ridge','std'),
                     Folds_KAN_Better=('MAE_Improvement_KAN_over_Ridge', lambda x: int((x>0).sum()))).reset_index()
    result.to_csv(OUT / 'section3_3_kan_vs_ridge_summary.csv', index=False)
    print('\nKAN vs RIDGE (positive means KAN lower MAE):\n', result.round(4).to_string(index=False))
    print('Note: repeated CV folds are correlated; these are descriptive comparisons, not independent significance tests.')


def subgroup_results(predictions):
    rows = []
    for (fs, edu), g in predictions.groupby(['FeatureSet', 'education_level']):
        per_person = g.groupby('prim_key').agg(y_true=('y_true','first'),
                      abs_error=('absolute_error','mean'), signed_error=('error','mean'),
                      y_pred=('y_pred','mean')).reset_index()
        rows.append({'Model':'KAN','FeatureSet':fs,'EducationLevel':edu,'N_Participants':len(per_person),
                     'MAE':per_person['abs_error'].mean(), 'MeanError':per_person['signed_error'].mean(),
                     'RMSE':np.sqrt(np.mean((per_person['y_pred']-per_person['y_true'])**2))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / 'section3_3_kan_education_subgroup_errors.csv', index=False)
    print('\nKAN EDUCATION SUBGROUP ERRORS\n', result.round(4).to_string(index=False))
    return result


def plot_interpretability(models, summary):
    # Interpret first-layer edge functions, not global marginal effects.
    fs = 'D_TotalPlusShape'
    (model, imputer, scaler, ym, ys, _), features, X_hold, y_hold = models[fs]
    grid = np.linspace(-3, 3, 150)
    fig, axes = plt.subplots(len(features), 1, figsize=(9, 2.6*len(features)), squeeze=False)
    for j, name in enumerate(features):
        ax = axes[j, 0]
        for hidden in range(HIDDEN_UNITS):
            curve = model.first.edge_curve(j, hidden, grid)
            ax.plot(grid, curve, linewidth=1.2, alpha=0.8, label=f'Hidden unit {hidden+1}')
        ax.set_title(f'KAN first-layer spline edges: {name}')
        ax.set_xlabel('Standardized baseline feature (clipped at ±3)')
        ax.set_ylabel('Edge contribution (hidden units)')
        if j == 0:
            ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'section3_3_kan_first_layer_splines.png', dpi=250)
    plt.close(fig)

    # Holdout permutation importance, on exactly one fold, exploratory.
    rng = np.random.default_rng(SEED)
    base_pred = predict(model, imputer, scaler, ym, ys, X_hold)
    base_mae = mean_absolute_error(y_hold, base_pred)
    importance_rows = []
    for feature in features:
        changes = []
        for _ in range(15):
            shuffled = X_hold.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            changes.append(mean_absolute_error(y_hold, predict(model, imputer, scaler, ym, ys, shuffled)) - base_mae)
        importance_rows.append({'FeatureSet':fs,'Feature':feature,'Importance_Mean':np.mean(changes),
                                'Importance_STD':np.std(changes, ddof=1)})
    imp_df = pd.DataFrame(importance_rows).sort_values('Importance_Mean')
    imp_df.to_csv(OUT / 'section3_3_kan_permutation_importance.csv', index=False)
    plt.figure(figsize=(9, 5))
    plt.barh(imp_df['Feature'], imp_df['Importance_Mean'], xerr=imp_df['Importance_STD'])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Held-out increase in MAE after permutation')
    plt.title('KAN permutation importance (one held-out fold)')
    plt.tight_layout()
    plt.savefig(FIG / 'section3_3_kan_permutation_importance.png', dpi=250)
    plt.close()
    print('\nKAN PERMUTATION IMPORTANCE (one fold)\n', imp_df.round(4).to_string(index=False))
    print('Spline edges are internal learned functions, not direct causal or population-level effects.')


def plot_model_comparison(kan_summary):
    path = OUT / 'section3_2_model_performance.csv'
    if not path.exists():
        return
    old = pd.read_csv(path)
    combined = pd.concat([old[['Model','FeatureSet','MAE_mean']], kan_summary[['Model','FeatureSet','MAE_mean']]], ignore_index=True)
    combined.to_csv(OUT / 'section3_3_all_model_performance.csv', index=False)
    pivot = combined.pivot(index='FeatureSet', columns='Model', values='MAE_mean')
    ax = pivot.plot(kind='bar', figsize=(11, 6))
    ax.set_ylabel('Mean repeated-CV MAE (lower is better)')
    ax.set_xlabel('Feature set')
    ax.set_title('KAN versus Section 3.2 baselines')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(FIG / 'section3_3_all_model_comparison.png', dpi=250)
    plt.close()


def main():
    seed_everything(SEED)
    print('Matplotlib backend:', matplotlib.get_backend())
    df = load_data()
    folds, predictions, summary, models = run_cv(df)
    compare_ridge(folds)
    subgroup_results(predictions)
    plot_interpretability(models, summary)
    plot_model_comparison(summary)
    print('\nSECTION 3.3 COMPLETED')
    print('Tables:', OUT)
    print('Figures:', FIG)
    print('Outcome: Wave 2 delayed recall; predictors: Wave 1 only.')
    print('No dementia or impairment classification performed.')


if __name__ == '__main__':
    main()
