
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf


# ============================================================
# SECTION 3.9
# SAME TOTAL RECALL SCORE, DIFFERENT SERIAL POSITION CURVES
#
# Research question:
# Among participants with the same Wave 1 immediate-recall
# total, are differences in SPC shape associated with
# Wave 2 delayed recall?
#
# This is an observational, associational analysis.
# It is not a diagnostic classification analysis.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SOURCE = DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"

SEED = 42

REQUIRED = [
    "prim_key",
    "P_i_w1",
    "M_i_w1",
    "R_i_w1",
    "T_i_w1",
    "age_baseline",
    "education_level",
    "delayed_recall_baseline",
    "delayed_recall_followup",
]


# ============================================================
# 2. LOAD AND VALIDATE ORIGINAL DATA
# ============================================================

def load_data():

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Original paired dataset not found:\n{SOURCE}"
        )

    print(f"\nLoading: {SOURCE.name}")

    df = pd.read_csv(SOURCE)

    missing = [
        col for col in REQUIRED
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    if df["prim_key"].isna().any():
        raise ValueError("Missing participant IDs")

    if df["prim_key"].duplicated().any():
        raise ValueError(
            "Duplicate participant IDs in paired dataset"
        )

    numeric_cols = [
        col for col in REQUIRED
        if col != "prim_key"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    n_original = len(df)

    df = df.dropna(
        subset=REQUIRED
    ).copy()

    n_complete = len(df)

    print(f"Original participants: {n_original}")
    print(f"Complete-case participants: {n_complete}")
    print(f"Excluded for missing data: {n_original - n_complete}")

    if n_complete < 30:
        raise ValueError(
            "Insufficient complete-case participants"
        )

    # Validate recall components.
    limits = {
        "P_i_w1": 3,
        "M_i_w1": 4,
        "R_i_w1": 3,
        "T_i_w1": 10,
        "delayed_recall_baseline": 10,
        "delayed_recall_followup": 10,
    }

    for col, maximum in limits.items():

        valid_range = df[col].between(0, maximum)

        valid_integer = np.isclose(
            df[col],
            np.round(df[col])
        )

        if not (valid_range & valid_integer).all():
            raise ValueError(
                f"Invalid recall values in {col}"
            )

    if not np.allclose(
        df["P_i_w1"] +
        df["M_i_w1"] +
        df["R_i_w1"],
        df["T_i_w1"]
    ):
        raise ValueError(
            "SPC components do not sum to total recall"
        )

    if not df["education_level"].isin(
        [1, 2, 3, 4]
    ).all():
        raise ValueError(
            "Unexpected education categories"
        )

    if not df["age_baseline"].between(18, 120).all():
        raise ValueError(
            "Unexpected baseline age values"
        )

    return df


# ============================================================
# 3. DERIVE SPC VARIABLES
# ============================================================

def derive_spc_variables(df):

    df = df.copy()

    df["primacy_rate"] = df["P_i_w1"] / 3.0
    df["middle_rate"] = df["M_i_w1"] / 4.0
    df["recency_rate"] = df["R_i_w1"] / 3.0

    df["primacy_recency_gap"] = (
        df["primacy_rate"] -
        df["recency_rate"]
    )

    df["spc_pattern"] = (
        df["P_i_w1"].astype(int).astype(str)
        + "-"
        + df["M_i_w1"].astype(int).astype(str)
        + "-"
        + df["R_i_w1"].astype(int).astype(str)
    )

    df["shape_group"] = np.select(
        [
            df["primacy_recency_gap"] > 0,
            df["primacy_recency_gap"] < 0,
        ],
        [
            "Primacy-dominant",
            "Recency-dominant",
        ],
        default="Balanced"
    )

    df["total_score"] = df["T_i_w1"].astype(int)

    return df


# ============================================================
# 4. DESCRIPTIVE SPC PATTERN ANALYSIS
# ============================================================

def descriptive_analysis(df):

    pattern_summary = (
        df.groupby(
            [
                "total_score",
                "spc_pattern",
                "shape_group"
            ],
            observed=True
        )
        .agg(
            n=("prim_key", "size"),
            mean_followup=(
                "delayed_recall_followup",
                "mean"
            ),
            sd_followup=(
                "delayed_recall_followup",
                "std"
            ),
            mean_baseline=(
                "delayed_recall_baseline",
                "mean"
            ),
            mean_age=(
                "age_baseline",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            ["total_score", "n"],
            ascending=[True, False]
        )
    )

    pattern_summary.to_csv(
        OUT / "section3_9_same_total_patterns.csv",
        index=False
    )

    group_summary = (
        df.groupby(
            ["total_score", "shape_group"],
            observed=True
        )
        .agg(
            n=("prim_key", "size"),
            mean_followup=(
                "delayed_recall_followup",
                "mean"
            ),
            sd_followup=(
                "delayed_recall_followup",
                "std"
            ),
            mean_baseline=(
                "delayed_recall_baseline",
                "mean"
            ),
            mean_age=(
                "age_baseline",
                "mean"
            )
        )
        .reset_index()
    )

    group_summary.to_csv(
        OUT / "section3_9_shape_groups_by_total.csv",
        index=False
    )

    diversity = (
        df.groupby("total_score")
        .agg(
            n_participants=("prim_key", "size"),
            n_patterns=("spc_pattern", "nunique"),
            n_shape_groups=("shape_group", "nunique")
        )
        .reset_index()
    )

    diversity.to_csv(
        OUT / "section3_9_pattern_diversity.csv",
        index=False
    )

    print("\nSPC PATTERN DIVERSITY")
    print(diversity.to_string(index=False))

    return pattern_summary, group_summary, diversity


# ============================================================
# 5. EXACT-TOTAL MATCHED COMPARISONS
# ============================================================

def matched_total_comparisons(df):

    results = []

    for total, subset in df.groupby("total_score"):

        primacy = subset.loc[
            subset["shape_group"] == "Primacy-dominant"
        ]

        recency = subset.loc[
            subset["shape_group"] == "Recency-dominant"
        ]

        balanced = subset.loc[
            subset["shape_group"] == "Balanced"
        ]

        n_primacy = len(primacy)
        n_recency = len(recency)
        n_balanced = len(balanced)

        mean_primacy = (
            primacy["delayed_recall_followup"].mean()
            if n_primacy else np.nan
        )

        mean_recency = (
            recency["delayed_recall_followup"].mean()
            if n_recency else np.nan
        )

        difference = (
            mean_primacy - mean_recency
            if n_primacy and n_recency
            else np.nan
        )

        results.append({
            "total_score": total,
            "n_total": len(subset),
            "n_primacy": n_primacy,
            "n_recency": n_recency,
            "n_balanced": n_balanced,
            "mean_followup_primacy": mean_primacy,
            "mean_followup_recency": mean_recency,
            "primacy_minus_recency": difference,
            "adequate_descriptive_groups": (
                n_primacy >= 5 and n_recency >= 5
            )
        })

    comparison = pd.DataFrame(results)

    comparison.to_csv(
        OUT / "section3_9_exact_total_comparisons.csv",
        index=False
    )

    print("\nEXACT-TOTAL COMPARISONS")
    print(comparison.to_string(index=False))

    return comparison


# ============================================================
# 6. REGRESSION MODELS
# ============================================================

def regression_analysis(df):

    # Treat total score as categorical.
    # This compares SPC patterns while holding the exact
    # total immediate-recall score constant.
    #
    # P and R are included, while M is omitted because:
    #
    # T = P + M + R
    #
    # Including all components and total would introduce
    # exact linear dependence.
    #
    # HC1 robust standard errors avoid the HC3
    # divide-by-zero issue in sparse/high-leverage cells.

    formula_base = (
        "delayed_recall_followup ~ "
        "C(total_score) + "
        "delayed_recall_baseline + "
        "age_baseline + "
        "C(education_level)"
    )

    formula_shape = (
        formula_base +
        " + P_i_w1 + R_i_w1"
    )

    model_base = smf.ols(
        formula_base,
        data=df
    ).fit(cov_type="HC1")

    model_shape = smf.ols(
        formula_shape,
        data=df
    ).fit(cov_type="HC1")

    # Check that the design matrices are full rank.
    for name, model in [
        ("Base", model_base),
        ("Shape", model_shape)
    ]:

        rank = np.linalg.matrix_rank(model.model.exog)
        n_columns = model.model.exog.shape[1]

        if rank < n_columns:
            raise ValueError(
                f"{name} model is rank deficient: "
                f"rank={rank}, columns={n_columns}"
            )

    joint_test = model_shape.wald_test(
        "P_i_w1 = 0, R_i_w1 = 0",
        scalar=True
    )

    joint_statistic = float(joint_test.statistic)
    joint_pvalue = float(joint_test.pvalue)

    summary = pd.DataFrame([
        {
            "Model": "Total + covariates",
            "N": int(model_base.nobs),
            "R2": model_base.rsquared,
            "Adjusted_R2": model_base.rsquared_adj,
            "AIC": model_base.aic,
            "BIC": model_base.bic
        },
        {
            "Model": "Total + SPC shape + covariates",
            "N": int(model_shape.nobs),
            "R2": model_shape.rsquared,
            "Adjusted_R2": model_shape.rsquared_adj,
            "AIC": model_shape.aic,
            "BIC": model_shape.bic
        }
    ])

    summary.to_csv(
        OUT / "section3_9_adjusted_model_comparison.csv",
        index=False
    )

    confidence_intervals = model_shape.conf_int()

    coefficients = pd.DataFrame({
        "Term": model_shape.params.index,
        "Coefficient": model_shape.params.values,
        "Robust_SE": model_shape.bse.values,
        "P_value": model_shape.pvalues.values,
        "CI_lower": confidence_intervals.iloc[:, 0].values,
        "CI_upper": confidence_intervals.iloc[:, 1].values
    })

    coefficients.to_csv(
        OUT / "section3_9_adjusted_coefficients.csv",
        index=False
    )

    # Avoid statsmodels.summary() here.
    # The global F-test in that summary can fail
    # under robust covariance with sparse categories.
    report_path = OUT / "section3_9_analysis_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SECTION 3.9: SAME TOTAL SCORE, DIFFERENT SPC\n"
        )

        f.write("=" * 65 + "\n\n")

        f.write(f"Participants: {len(df)}\n")
        f.write("Covariance estimator: HC1\n\n")

        f.write("BASE MODEL\n")
        f.write(formula_base + "\n\n")
        f.write(
            summary.iloc[[0]].to_string(index=False)
        )

        f.write("\n\nSHAPE MODEL\n")
        f.write(formula_shape + "\n\n")
        f.write(
            summary.iloc[[1]].to_string(index=False)
        )

        f.write("\n\nSHAPE MODEL COEFFICIENTS\n")
        f.write(coefficients.to_string(index=False))

        f.write("\n\nJOINT ROBUST WALD TEST\n")
        f.write("Null hypothesis: P_i_w1 = R_i_w1 = 0\n")
        f.write(f"Statistic: {joint_statistic:.6f}\n")
        f.write(f"P-value: {joint_pvalue:.8g}\n")

        f.write(
            "\nInterpretation: This is an adjusted "
            "association test, not out-of-sample "
            "prediction or evidence of causation.\n"
        )

    print("\nADJUSTED MODEL COMPARISON")
    print(summary.to_string(index=False))

    print("\nJOINT SPC SHAPE TEST")
    print(f"Wald statistic: {joint_statistic:.4f}")
    print(f"P-value: {joint_pvalue:.6g}")

    return summary, coefficients, joint_pvalue


# ============================================================
# 7. FIGURE: FOLLOW-UP RECALL BY SPC GROUP
# ============================================================

def plot_shape_groups(group_summary):

    plt.figure(figsize=(10, 6))

    groups = [
        "Primacy-dominant",
        "Balanced",
        "Recency-dominant"
    ]

    for group in groups:

        part = group_summary.loc[
            (group_summary["shape_group"] == group)
            & (group_summary["n"] >= 5)
        ].sort_values("total_score")

        if part.empty:
            continue

        plt.plot(
            part["total_score"],
            part["mean_followup"],
            marker="o",
            linewidth=2,
            label=group
        )

    plt.xlabel(
        "Wave 1 total immediate-recall score"
    )

    plt.ylabel(
        "Mean Wave 2 delayed-recall score"
    )

    plt.title(
        "Follow-up Delayed Recall by SPC Shape\n"
        "Within Exact Baseline Total Scores"
    )

    plt.xticks(range(0, 11))
    plt.ylim(0, 10)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIG / "section3_9_same_total_spc.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. FIGURE: NUMBER OF DISTINCT SPC PATTERNS
# ============================================================

def plot_pattern_diversity(diversity):

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        diversity["total_score"],
        diversity["n_patterns"]
    )

    ax.set_xlabel(
        "Wave 1 total immediate-recall score"
    )

    ax.set_ylabel(
        "Number of distinct observed SPC patterns"
    )

    ax.set_title(
        "SPC Pattern Diversity at Identical Total Scores"
    )

    ax.set_xticks(range(0, 11))

    fig.tight_layout()

    fig.savefig(
        FIG / "section3_9_pattern_diversity.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SECTION 3.9: SAME TOTAL SCORE, DIFFERENT SPC")
    print("=" * 70)

    df = load_data()

    df = derive_spc_variables(df)

    pattern_summary, group_summary, diversity = (
        descriptive_analysis(df)
    )

    comparison = matched_total_comparisons(df)

    model_summary, coefficients, pvalue = (
        regression_analysis(df)
    )

    plot_shape_groups(group_summary)
    plot_pattern_diversity(diversity)

    print("\n" + "=" * 70)
    print("SECTION 3.9 COMPLETED")
    print("=" * 70)

    print(f"\nParticipants analysed: {len(df)}")
    print(f"Unique SPC patterns: {df['spc_pattern'].nunique()}")

    print("\nOutputs saved to:")
    print(OUT)

    print("\nFigures saved to:")
    print(FIG)

    print("\nInterpretation reminders:")
    print(
        "- Participants may share total recall but differ "
        "in SPC shape."
    )
    print(
        "- Descriptive subgroup differences are not "
        "necessarily statistically significant."
    )
    print(
        "- The adjusted regression tests association, "
        "not causation."
    )
    print(
        "- Results do not establish clinical dementia "
        "or confirmed cognitive decline."
    )


if __name__ == "__main__":
    main()
