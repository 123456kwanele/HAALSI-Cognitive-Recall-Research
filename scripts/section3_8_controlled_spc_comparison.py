
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"

SEED = 42
N_SPLITS = 5
N_REPEATS = 3
TARGET = "delayed_recall_followup"

FEATURE_SETS = {
    "A_TotalOnly": [
        "T_i_w1"
    ],
    "B_DecomposedOnly": [
        "P_i_w1", "M_i_w1", "R_i_w1"
    ],
    "C_TotalAdjusted": [
        "T_i_w1",
        "age_baseline",
        "education_level",
        "delayed_recall_baseline"
    ],
    "D_DecomposedAdjusted": [
        "P_i_w1", "M_i_w1", "R_i_w1",
        "age_baseline",
        "education_level",
        "delayed_recall_baseline"
    ]
}

COMPARISONS = [
    ("A_TotalOnly", "B_DecomposedOnly"),
    ("C_TotalAdjusted", "D_DecomposedAdjusted")
]

MODELS = [
    "Ridge",
    "RandomForest",
    "MLP_Original",
    "MLP_LBFGS"
]


# ============================================================
# DATA VALIDATION
# ============================================================

def load_data():

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing approved dataset: {SOURCE}\n"
            "Run python scripts/02_understand_dataset.py first."
        )

    df = pd.read_csv(SOURCE)

    required = sorted(set(
        ["prim_key", TARGET, "education_level"]
        + [
            col
            for columns in FEATURE_SETS.values()
            for col in columns
        ]
    ))

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["prim_key"].isna().any():
        raise ValueError("Missing participant IDs.")

    if df["prim_key"].duplicated().any():
        raise ValueError("Duplicate participants.")

    for col in required:
        if col != "prim_key":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Do not silently discard records or repair data.
    missing_values = df[required].isna().sum()
    if missing_values.any():
        raise ValueError(
            "Missing values in approved analysis cohort:\n"
            + missing_values[missing_values > 0].to_string()
        )

    if len(df) != 504:
        raise ValueError(
            f"Expected 504 participants; found {len(df)}."
        )

    if not df["education_level"].isin([1, 2, 3, 4]).all():
        raise ValueError("Unexpected education categories.")

    if not np.allclose(
        df["P_i_w1"] + df["M_i_w1"] + df["R_i_w1"],
        df["T_i_w1"]
    ):
        raise ValueError("SPC components do not sum to total.")

    if not df[TARGET].between(0, 10).all():
        raise ValueError("Invalid delayed-recall outcome.")

    if not df["T_i_w1"].between(0, 10).all():
        raise ValueError("Invalid immediate-recall total.")

    if not df["delayed_recall_baseline"].between(0, 10).all():
        raise ValueError("Invalid baseline delayed recall.")

    df = df.sort_values("prim_key").reset_index(drop=True)

    print("\nApproved source:", SOURCE.name)
    print("Participants:", len(df))
    print("Unique IDs:", df["prim_key"].nunique())
    print("Missing outcomes:", df[TARGET].isna().sum())

    print("\nEducation distribution:")
    print(df["education_level"].value_counts().sort_index())

    return df


# ============================================================
# PREPROCESSING AND MODELS
# ============================================================

def make_pipeline(features, model_name):

    categorical = [
        c for c in features if c == "education_level"
    ]

    numerical = [
        c for c in features if c != "education_level"
    ]

    transformers = []

    if numerical:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numerical
        ))

    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(
                categories=[[1, 2, 3, 4]],
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical
        ))

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    if model_name == "Ridge":
        model = Ridge(alpha=10.0)

    elif model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1
        )

    elif model_name == "MLP_Original":
        # Preserve the previously reported configuration.
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1.0,
            max_iter=600,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=SEED
        )

    elif model_name == "MLP_LBFGS":
        # Sensitivity analysis: no internal validation split.
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="lbfgs",
            alpha=0.1,
            max_iter=1000,
            max_fun=15000,
            random_state=SEED
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])


# ============================================================
# REPEATED CROSS-VALIDATION
# ============================================================

def run_cross_validation(df):

    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED
    )

    metrics = []
    predictions = []
    diagnostics = []

    y = df[TARGET].to_numpy(dtype=float)

    for split, (train_idx, test_idx) in enumerate(
        cv.split(df), start=1
    ):

        print(f"\nSplit {split}/{N_SPLITS * N_REPEATS}")

        y_train = y[train_idx]
        y_test = y[test_idx]

        for model_name in MODELS:
            for feature_set, features in FEATURE_SETS.items():

                pipeline = make_pipeline(features, model_name)

                X_train = df.iloc[train_idx][features]
                X_test = df.iloc[test_idx][features]

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    pipeline.fit(X_train, y_train)

                convergence_warnings = [
                    str(w.message)
                    for w in caught
                    if issubclass(w.category, ConvergenceWarning)
                ]

                pred = pipeline.predict(X_test)

                if not np.isfinite(pred).all():
                    raise ValueError(
                        f"Nonfinite predictions: "
                        f"{model_name}, {feature_set}, split {split}"
                    )

                mae = mean_absolute_error(y_test, pred)
                rmse = np.sqrt(mean_squared_error(y_test, pred))
                r2 = r2_score(y_test, pred)

                metrics.append({
                    "split": split,
                    "model": model_name,
                    "feature_set": feature_set,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2
                })

                fitted_model = pipeline.named_steps["model"]

                diagnostics.append({
                    "split": split,
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "prediction_min": float(pred.min()),
                    "prediction_max": float(pred.max()),
                    "prediction_mean": float(pred.mean()),
                    "observed_mean": float(y_test.mean()),
                    "n_iterations": getattr(
                        fitted_model, "n_iter_", np.nan
                    ),
                    "training_loss": getattr(
                        fitted_model, "loss_", np.nan
                    ),
                    "convergence_warning_count": len(
                        convergence_warnings
                    ),
                    "convergence_warning_text": " | ".join(
                        convergence_warnings
                    )
                })

                for position, row_index in enumerate(test_idx):
                    predictions.append({
                        "split": split,
                        "model": model_name,
                        "feature_set": feature_set,
                        "prim_key": df.iloc[row_index]["prim_key"],
                        "education_level": int(
                            df.iloc[row_index]["education_level"]
                        ),
                        "actual": float(y_test[position]),
                        "predicted": float(pred[position]),
                        "signed_error": float(
                            pred[position] - y_test[position]
                        ),
                        "absolute_error": float(
                            abs(pred[position] - y_test[position])
                        )
                    })

                print(
                    f"  {model_name:<15} "
                    f"{feature_set:<22} "
                    f"MAE={mae:.4f} R2={r2:.4f}"
                )

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(predictions),
        pd.DataFrame(diagnostics)
    )


# ============================================================
# PAIRED REPRESENTATION COMPARISONS
# ============================================================

def compare_representations(metrics):

    rows = []

    for model_name in MODELS:

        model_metrics = metrics[
            metrics["model"] == model_name
        ]

        for total_name, decomposed_name in COMPARISONS:

            total = (
                model_metrics[
                    model_metrics["feature_set"] == total_name
                ]
                .set_index("split")
                .sort_index()
            )

            decomposed = (
                model_metrics[
                    model_metrics["feature_set"] == decomposed_name
                ]
                .set_index("split")
                .sort_index()
            )

            for metric in ["mae", "rmse", "r2"]:

                delta = (
                    decomposed[metric] - total[metric]
                ).to_numpy()

                # For MAE/RMSE, negative means decomposed is better.
                # For R2, positive means decomposed is better.
                better = (
                    delta > 0 if metric == "r2"
                    else delta < 0
                )

                rows.append({
                    "model": model_name,
                    "comparison": (
                        f"{decomposed_name} minus {total_name}"
                    ),
                    "metric": metric,
                    "mean_total": total[metric].mean(),
                    "mean_decomposed": decomposed[metric].mean(),
                    "mean_difference": delta.mean(),
                    "difference_sd": delta.std(ddof=1),
                    "splits_decomposed_better": int(better.sum()),
                    "n_splits": len(delta)
                })

    return pd.DataFrame(rows)


# ============================================================
# EDUCATION SUBGROUP ANALYSIS
# ============================================================

def subgroup_analysis(predictions):

    rows = []

    grouped = predictions.groupby(
        ["model", "feature_set", "education_level"]
    )

    for (model, feature_set, education), group in grouped:

        rows.append({
            "model": model,
            "feature_set": feature_set,
            "education_level": education,
            "n_prediction_rows": len(group),
            "n_unique_participants": group["prim_key"].nunique(),
            "mae": group["absolute_error"].mean(),
            "mean_signed_error": group["signed_error"].mean()
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_results(summary, subgroup):

    for metric in ["mae", "rmse", "r2"]:

        pivot = summary.pivot(
            index="model",
            columns="feature_set",
            values=metric
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(12, 6),
            rot=0
        )

        ax.set_title(
            f"Section 3.8: Controlled SPC Comparison ({metric.upper()})"
        )
        ax.set_xlabel("Model")
        ax.set_ylabel(metric.upper())

        ax.legend(
            title="Feature set",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()
        plt.savefig(
            FIG / f"section3_8_{metric}_comparison.png",
            dpi=250,
            bbox_inches="tight"
        )
        plt.close()

    for model_name in MODELS:

        subset = subgroup[subgroup["model"] == model_name]

        pivot = subset.pivot(
            index="education_level",
            columns="feature_set",
            values="mae"
        )

        ax = pivot.plot(
            kind="bar",
            figsize=(11, 6),
            rot=0
        )

        ax.set_title(
            f"{model_name}: MAE by Education Category"
        )
        ax.set_xlabel("Education category")
        ax.set_ylabel("Mean absolute error")

        ax.legend(
            title="Feature set",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        plt.tight_layout()
        plt.savefig(
            FIG / f"section3_8_education_{model_name}.png",
            dpi=250,
            bbox_inches="tight"
        )
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SECTION 3.8: CONTROLLED SPC PREDICTIVE COMPARISON")
    print("Approved 02_understand_dataset.py cohort")
    print("=" * 70)

    df = load_data()

    metrics, predictions, diagnostics = run_cross_validation(df)

    comparison = compare_representations(metrics)
    subgroup = subgroup_analysis(predictions)

    summary = (
        metrics.groupby(["model", "feature_set"])[
            ["mae", "rmse", "r2"]
        ]
        .mean()
        .reset_index()
    )

    metrics.to_csv(
        OUT / "section3_8_cv_metrics.csv", index=False
    )
    predictions.to_csv(
        OUT / "section3_8_oof_predictions.csv", index=False
    )
    diagnostics.to_csv(
        OUT / "section3_8_model_diagnostics.csv", index=False
    )
    comparison.to_csv(
        OUT / "section3_8_paired_comparison.csv", index=False
    )
    subgroup.to_csv(
        OUT / "section3_8_education_subgroups.csv", index=False
    )
    summary.to_csv(
        OUT / "section3_8_model_summary.csv", index=False
    )

    plot_results(summary, subgroup)

    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE")
    print("=" * 70)
    print(summary.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("DECOMPOSED MINUS TOTAL: MAE")
    print("=" * 70)
    print(
        comparison[comparison["metric"] == "mae"]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("MLP DIAGNOSTICS")
    print("=" * 70)

    mlp_diag = diagnostics[
        diagnostics["model"].str.startswith("MLP")
    ]

    print(
        mlp_diag.groupby(["model", "feature_set"])
        .agg(
            mean_iterations=("n_iterations", "mean"),
            mean_training_loss=("training_loss", "mean"),
            convergence_warnings=(
                "convergence_warning_count", "sum"
            ),
            minimum_prediction=("prediction_min", "min"),
            maximum_prediction=("prediction_max", "max")
        )
        .round(4)
        .to_string()
    )

    print("\n" + "=" * 70)
    print("EDUCATION SUBGROUP RESULTS")
    print("=" * 70)
    print(subgroup.round(4).to_string(index=False))

    run_summary = {
        "source": SOURCE.name,
        "participants": len(df),
        "unique_participants": df["prim_key"].nunique(),
        "cv_folds": N_SPLITS,
        "cv_repeats": N_REPEATS,
        "model_configurations": MODELS,
        "feature_sets": list(FEATURE_SETS),
        "n_model_fits": len(metrics),
        "target": TARGET,
        "notes": [
            "No repaired dataset used.",
            "No outcome imputation.",
            "All preprocessing fitted inside training folds.",
            "MLP_LBFGS is a sensitivity analysis.",
            "Repeated CV predictions are not independent participants.",
            "No clinical diagnostic performance is estimated."
        ]
    }

    with open(
        OUT / "section3_8_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(run_summary, f, indent=2)

    print("\nSECTION 3.8 COMPLETED")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
