
# ============================================================
# SECTION 3.2: LONGITUDINAL MODEL COMPARISON
# Corrected Windows-safe version
# ============================================================

from pathlib import Path
import os

# Set a non-interactive backend BEFORE importing pyplot.
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"

SEED = 42
N_SPLITS = 5
N_REPEATS = 3
PERMUTATION_REPEATS = 15

MODEL_NAMES = ["Ridge", "RandomForest", "MLP"]

FEATURE_SETS = {
    "A_Conventional": [
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
    ],
    "B_TotalRecall": [
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
        "T_i_w1",
    ],
    "C_DecomposedSPC": [
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
        "P_i_w1",
        "M_i_w1",
        "R_i_w1",
    ],
    "D_TotalPlusShape": [
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
        "T_i_w1",
        "primacy_recency_contrast",
        "middle_outer_contrast",
    ],
}

sns.set_theme(style="whitegrid")


def heading(text):
    print("\n" + "=" * 65)
    print(text)
    print("=" * 65)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(FIG / filename, dpi=300, bbox_inches="tight")
    plt.close("all")


def load_data():
    heading("1. LOADING LONGITUDINAL DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    required = [
        "prim_key",
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
        "delayed_recall_followup",
        "P_i_w1",
        "M_i_w1",
        "R_i_w1",
        "T_i_w1",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["prim_key"].isna().any():
        raise ValueError("Missing participant identifiers.")

    if df["prim_key"].duplicated().any():
        raise ValueError("Duplicate participant identifiers.")

    for col in required:
        if col != "prim_key":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "delayed_recall_followup",
            "delayed_recall_baseline",
            "age_baseline",
            "education_level",
            "P_i_w1",
            "M_i_w1",
            "R_i_w1",
            "T_i_w1",
        ]
    ).copy()

    if not np.allclose(
        df["P_i_w1"] + df["M_i_w1"] + df["R_i_w1"],
        df["T_i_w1"],
    ):
        raise ValueError("SPC components do not sum to total.")

    if not df["education_level"].isin([1, 2, 3, 4]).all():
        raise ValueError("Unexpected education categories.")

    if not df["delayed_recall_followup"].between(0, 10).all():
        raise ValueError("Invalid follow-up recall scores.")

    df["primacy_rate_w1"] = df["P_i_w1"] / 3
    df["middle_rate_w1"] = df["M_i_w1"] / 4
    df["recency_rate_w1"] = df["R_i_w1"] / 3

    df["primacy_recency_contrast"] = (
        df["primacy_rate_w1"] - df["recency_rate_w1"]
    )

    df["middle_outer_contrast"] = (
        df["middle_rate_w1"]
        - (
            df["primacy_rate_w1"]
            + df["recency_rate_w1"]
        ) / 2
    )

    df = df.reset_index(drop=True)

    print("Analysis-ready participants:", len(df))
    print("Unique participants:", df["prim_key"].nunique())
    print("\nFollow-up delayed recall:")
    print(df["delayed_recall_followup"].describe())

    return df


def build_model(name, seed):
    if name == "Ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])

    if name == "RandomForest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=5,
                max_features=1.0,
                random_state=seed,
                n_jobs=1,  # Avoid unnecessary worker threads.
            )),
        ])

    if name == "MLP":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(16, 8),
                activation="relu",
                solver="adam",
                alpha=1.0,
                learning_rate_init=0.001,
                max_iter=1200,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=40,
                random_state=seed,
            )),
        ])

    raise ValueError(f"Unknown model: {name}")


def metrics(y, predicted):
    return {
        "MAE": mean_absolute_error(y, predicted),
        "RMSE": np.sqrt(mean_squared_error(y, predicted)),
        "R2": r2_score(y, predicted),
        "MeanError": float(np.mean(predicted - y)),
    }


def cross_validation(df):
    heading("2. REPEATED CROSS-VALIDATION")

    y = df["delayed_recall_followup"].to_numpy(dtype=float)

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    splits = list(cv.split(df))

    fold_rows = []
    prediction_rows = []
    importance_models = {}

    total = len(MODEL_NAMES) * len(FEATURE_SETS) * len(splits)
    completed = 0

    for name in MODEL_NAMES:
        for feature_set, features in FEATURE_SETS.items():
            X = df[features]

            for fold_index, (train_idx, test_idx) in enumerate(splits):
                repeat = fold_index // N_SPLITS + 1
                fold = fold_index % N_SPLITS + 1

                model = build_model(name, SEED + fold_index)

                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        category=ConvergenceWarning,
                    )
                    model.fit(X_train, y_train)

                predicted = model.predict(X_test)

                fold_rows.append({
                    "Model": name,
                    "FeatureSet": feature_set,
                    "Repeat": repeat,
                    "Fold": fold,
                    "N_Train": len(train_idx),
                    "N_Test": len(test_idx),
                    **metrics(y_test, predicted),
                })

                for j, idx in enumerate(test_idx):
                    error = float(predicted[j] - y_test[j])

                    prediction_rows.append({
                        "prim_key": df.iloc[idx]["prim_key"],
                        "Model": name,
                        "FeatureSet": feature_set,
                        "Repeat": repeat,
                        "Fold": fold,
                        "education_level": df.iloc[idx]["education_level"],
                        "y_true": float(y_test[j]),
                        "y_pred": float(predicted[j]),
                        "error": error,
                        "absolute_error": abs(error),
                    })

                if fold_index == 0:
                    importance_models[(name, feature_set)] = {
                        "model": model,
                        "X_test": X_test.copy(),
                        "y_test": y_test.copy(),
                        "features": features,
                    }

                completed += 1
                print(
                    f"\rCompleted {completed}/{total} fits",
                    end="",
                    flush=True,
                )

    print()

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    folds.to_csv(
        OUT / "section3_2_cv_fold_results.csv",
        index=False,
    )

    predictions.to_csv(
        OUT / "section3_2_cv_predictions.csv",
        index=False,
    )

    return folds, predictions, importance_models


def summarize_models(folds):
    heading("3. MODEL PERFORMANCE SUMMARY")

    summary = (
        folds.groupby(["Model", "FeatureSet"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
            MeanError_mean=("MeanError", "mean"),
        )
        .reset_index()
        .sort_values(["MAE_mean", "RMSE_mean"])
    )

    summary.to_csv(
        OUT / "section3_2_model_performance.csv",
        index=False,
    )

    print(summary.round(4).to_string(index=False))

    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=summary,
        x="FeatureSet",
        y="MAE_mean",
        hue="Model",
        errorbar=None,
    )

    plt.title("Longitudinal Model Comparison")
    plt.xlabel("Feature set")
    plt.ylabel("Mean cross-validation MAE")
    plt.xticks(rotation=15, ha="right")

    save_figure("section3_2_model_comparison.png")

    return summary


def incremental_spc_value(folds):
    heading("4. INCREMENTAL VALUE OF SPC FEATURES")

    pivot = folds.pivot(
        index=["Model", "Repeat", "Fold"],
        columns="FeatureSet",
        values="MAE",
    )

    comparisons = [
        (
            "B_TotalRecall",
            "A_Conventional",
            "Total recall beyond conventional baseline",
        ),
        (
            "C_DecomposedSPC",
            "B_TotalRecall",
            "Decomposed SPC versus total recall",
        ),
        (
            "D_TotalPlusShape",
            "B_TotalRecall",
            "SPC shape beyond total recall",
        ),
    ]

    rows = []

    for name in MODEL_NAMES:
        part = pivot.loc[name]

        for proposed, baseline, description in comparisons:
            improvement = part[baseline] - part[proposed]

            rows.append({
                "Model": name,
                "Comparison": description,
                "Baseline": baseline,
                "Proposed": proposed,
                "Mean_MAE_Improvement": improvement.mean(),
                "SD_Fold_Improvement": improvement.std(),
                "Folds_Improved": int((improvement > 0).sum()),
                "Total_Folds": len(improvement),
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "section3_2_spc_incremental_value.csv",
        index=False,
    )

    print(result.round(4).to_string(index=False))
    print("\nPositive improvement means lower MAE.")
    print("Repeated folds are not independent significance tests.")


def education_subgroups(predictions):
    heading("5. EDUCATION SUBGROUP PERFORMANCE")

    rows = []

    grouped = predictions.groupby(
        ["Model", "FeatureSet", "education_level"],
        dropna=False,
    )

    for (name, feature_set, education), group in grouped:
        # Each participant is predicted once per repetition.
        per_person = (
            group.groupby("prim_key")
            .agg(
                actual=("y_true", "first"),
                predicted=("y_pred", "mean"),
                mean_absolute_error=("absolute_error", "mean"),
                mean_signed_error=("error", "mean"),
            )
            .reset_index()
        )

        rows.append({
            "Model": name,
            "FeatureSet": feature_set,
            "EducationLevel": education,
            "N_Participants": len(per_person),
            "MAE": per_person["mean_absolute_error"].mean(),
            "MeanError": per_person["mean_signed_error"].mean(),
            "RMSE": np.sqrt(
                np.mean(
                    (
                        per_person["predicted"]
                        - per_person["actual"]
                    ) ** 2
                )
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUT / "section3_2_education_subgroup_errors.csv",
        index=False,
    )

    print(result.round(4).to_string(index=False))

    plot_df = result[
        result["FeatureSet"] == "D_TotalPlusShape"
    ]

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=plot_df,
        x="EducationLevel",
        y="MAE",
        hue="Model",
        errorbar=None,
    )

    plt.title("Prediction Error by Education Category")
    plt.xlabel("Education category")
    plt.ylabel("Mean absolute error")

    save_figure("section3_2_education_subgroup_mae.png")


def permutation_analysis(importance_models):
    heading("6. PERMUTATION IMPORTANCE")

    all_results = []
    feature_set = "D_TotalPlusShape"

    for name in MODEL_NAMES:
        print(f"\nComputing importance for {name}...", flush=True)

        item = importance_models[(name, feature_set)]

        result = permutation_importance(
            estimator=item["model"],
            X=item["X_test"],
            y=item["y_test"],
            scoring="neg_mean_absolute_error",
            n_repeats=PERMUTATION_REPEATS,
            random_state=SEED,
            n_jobs=1,
        )

        importance = pd.DataFrame({
            "Model": name,
            "FeatureSet": feature_set,
            "Feature": item["features"],
            "Importance_Mean": result.importances_mean,
            "Importance_STD": result.importances_std,
        })

        all_results.append(importance)

        ordered = importance.sort_values("Importance_Mean")

        plt.figure(figsize=(9, 5))

        plt.barh(
            ordered["Feature"],
            ordered["Importance_Mean"],
            xerr=ordered["Importance_STD"],
        )

        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Increase in MAE after permutation")
        plt.title(f"Permutation Importance: {name}")

        save_figure(
            f"section3_2_importance_{name.lower()}.png"
        )

        print("Saved importance figure for", name)

    combined = pd.concat(all_results, ignore_index=True)

    combined.to_csv(
        OUT / "section3_2_permutation_importance.csv",
        index=False,
    )

    print("\nPermutation importance results:")
    print(combined.round(4).to_string(index=False))
    print("\nImportance is based on one held-out fold per model.")


def prediction_diagnostics(predictions, summary):
    heading("7. PREDICTION DIAGNOSTICS")

    best = summary.iloc[0]

    name = best["Model"]
    feature_set = best["FeatureSet"]

    selected = predictions[
        (predictions["Model"] == name)
        & (predictions["FeatureSet"] == feature_set)
    ]

    averaged = (
        selected.groupby("prim_key")
        .agg(
            actual=("y_true", "first"),
            predicted=("y_pred", "mean"),
        )
        .reset_index()
    )

    averaged["residual"] = (
        averaged["actual"] - averaged["predicted"]
    )

    averaged.to_csv(
        OUT / "section3_2_best_model_predictions.csv",
        index=False,
    )

    plt.figure(figsize=(7, 7))

    sns.scatterplot(
        data=averaged,
        x="actual",
        y="predicted",
        alpha=0.6,
    )

    plt.plot([0, 10], [0, 10], "k--")
    plt.xlabel("Observed Wave 2 delayed recall")
    plt.ylabel("Predicted Wave 2 delayed recall")
    plt.title(f"Out-of-Fold Predictions: {name}")

    save_figure("section3_2_best_model_predictions.png")

    plt.figure(figsize=(9, 5))

    sns.histplot(
        averaged["residual"],
        bins=20,
        kde=True,
    )

    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Observed minus predicted")
    plt.title("Prediction Residual Distribution")

    save_figure("section3_2_residual_distribution.png")

    print("\nBest model:")
    print(
        best[
            [
                "Model",
                "FeatureSet",
                "MAE_mean",
                "RMSE_mean",
                "R2_mean",
            ]
        ]
    )


def run_section_3_2():
    heading("SECTION 3.2: LONGITUDINAL MODEL COMPARISON")

    print("Matplotlib backend:", matplotlib.get_backend())

    df = load_data()

    folds, predictions, importance_models = cross_validation(df)

    summary = summarize_models(folds)

    incremental_spc_value(folds)

    education_subgroups(predictions)

    permutation_analysis(importance_models)

    prediction_diagnostics(predictions, summary)

    heading("SECTION 3.2 COMPLETED")

    print("Tables:", OUT)
    print("Figures:", FIG)
    print("\nNo diagnostic labels or thresholds were created.")
    print("All predictive features are baseline measurements.")
    print("All preprocessing was fitted within training folds.")


if __name__ == "__main__":
    run_section_3_2()
