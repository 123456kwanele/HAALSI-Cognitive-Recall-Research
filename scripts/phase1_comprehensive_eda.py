
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# HAALSI PHASE 1: COMPREHENSIVE EXPLORATORY DATA ANALYSIS
# Updated for Phase 2 longitudinal dataset preparation
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

PANEL_FILE = DATA / "HAALSI_Narrowed_SPC_Dataset.csv"
WIDE_FILE = DATA / "HAALSI_Longitudinal_SPC_Dataset.csv"
PAIRED_FILE = DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"

WORDS = [f"trial_word_{i}" for i in range(1, 11)]
SPC = ["P_i", "M_i", "R_i", "T_i"]

EDUCATION_LABELS = {
    1: "No formal education",
    2: "Some primary",
    3: "Some secondary",
    4: "Secondary or more",
}

sns.set_theme(style="whitegrid")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: missing required columns {missing}"
        )


def to_numeric_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def to_boolean(series):
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .eq("true")
    )


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(
        FIG / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


def load_datasets():
    print("\nLoading prepared HAALSI datasets...")

    for path in [PANEL_FILE, WIDE_FILE, PAIRED_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prepared dataset: {path}"
            )

    panel = pd.read_csv(PANEL_FILE)
    wide = pd.read_csv(WIDE_FILE)
    paired = pd.read_csv(PAIRED_FILE)

    require_columns(
        panel,
        [
            "prim_key",
            "wave",
            "recall_completed",
            "delayed_recall_completed",
            "delayed_recall_score",
            "mmse_fully_completed",
            "mmse_score",
            "age_baseline",
            "education_level",
            "recall_admin_issue",
        ] + WORDS + SPC,
        "Panel"
    )

    require_columns(
        wide,
        [
            "prim_key",
            "valid_delayed_pair",
            "delayed_recall_baseline",
            "delayed_recall_followup",
            "delayed_recall_change",
        ],
        "Wide"
    )

    require_columns(
        paired,
        [
            "prim_key",
            "education_level",
            "age_baseline",
            "delayed_recall_baseline",
            "delayed_recall_followup",
            "delayed_recall_change",
            "P_i_w1",
            "M_i_w1",
            "R_i_w1",
            "T_i_w1",
        ],
        "Paired"
    )

    panel = to_numeric_columns(
        panel,
        WORDS + SPC + [
            "wave",
            "age_baseline",
            "education_level",
            "mmse_score",
            "delayed_recall_score",
            "recall_admin_issue",
            "primacy_rate",
            "middle_rate",
            "recency_rate",
        ]
    )

    paired = to_numeric_columns(
        paired,
        [
            "age_baseline",
            "education_level",
            "delayed_recall_baseline",
            "delayed_recall_followup",
            "delayed_recall_change",
            "P_i_w1",
            "M_i_w1",
            "R_i_w1",
            "T_i_w1",
            "primacy_rate_w1",
            "middle_rate_w1",
            "recency_rate_w1",
        ]
    )

    for column in [
        "recall_completed",
        "delayed_recall_completed",
        "mmse_fully_completed",
    ]:
        panel[column] = to_boolean(panel[column])

    wide["valid_delayed_pair"] = to_boolean(
        wide["valid_delayed_pair"]
    )

    if panel.duplicated(["prim_key", "wave"]).any():
        raise ValueError(
            "Duplicate participant-wave observations."
        )

    if wide["prim_key"].duplicated().any():
        raise ValueError(
            "Duplicate participants in wide dataset."
        )

    if paired["prim_key"].duplicated().any():
        raise ValueError(
            "Duplicate participants in paired dataset."
        )

    print("Panel observations:", len(panel))
    print("Unique participants:", panel["prim_key"].nunique())
    print("Wide participants:", len(wide))
    print("Analysis-ready participants:", len(paired))

    return panel, wide, paired


# ============================================================
# 1. COHORT DESCRIPTION
# ============================================================

def cohort_description(panel, paired):
    print("\n" + "=" * 65)
    print("1. COHORT DESCRIPTION")
    print("=" * 65)

    summary_columns = [
        "age_baseline",
        "education_level",
        "mmse_score",
        "delayed_recall_score",
        "P_i",
        "M_i",
        "R_i",
        "T_i",
    ]

    summary = (
        panel.groupby("wave")[summary_columns]
        .agg(["count", "mean", "std", "median"])
    )

    summary.to_csv(
        OUT / "demographic_cognitive_summary.csv"
    )

    print("\nCohort summary:")
    print(summary.round(3))

    paired_summary = paired[
        [
            "age_baseline",
            "education_level",
            "delayed_recall_baseline",
            "delayed_recall_followup",
            "delayed_recall_change",
            "P_i_w1",
            "M_i_w1",
            "R_i_w1",
            "T_i_w1",
        ]
    ].describe().T

    paired_summary.to_csv(
        OUT / "paired_cohort_summary.csv"
    )

    print("\nPaired cohort summary:")
    print(paired_summary.round(3))


# ============================================================
# 2. MISSINGNESS AND ASSESSMENT COMPLETION
# ============================================================

def missingness_analysis(panel):
    print("\n" + "=" * 65)
    print("2. MISSINGNESS AND COMPLETION")
    print("=" * 65)

    columns = [
        "age_baseline",
        "education_level",
        "mmse_score",
        "delayed_recall_score",
        "P_i",
        "M_i",
        "R_i",
        "T_i",
    ]

    missingness = (
        panel.groupby("wave")[columns]
        .apply(
            lambda frame: frame.isna().mean() * 100
        )
    )

    missingness.to_csv(
        OUT / "missingness_by_wave.csv"
    )

    completion = (
        panel.groupby("wave")[
            [
                "recall_completed",
                "delayed_recall_completed",
                "mmse_fully_completed",
            ]
        ]
        .agg(["sum", "mean"])
    )

    completion.to_csv(
        OUT / "assessment_completion_by_wave.csv"
    )

    print("\nMissingness (%):")
    print(missingness.round(2))

    print("\nAssessment completion:")
    print(completion)

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        missingness,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Missing (%)"}
    )

    plt.title("Missingness by Assessment Wave")
    plt.ylabel("Wave")

    save_figure("missingness_heatmap.png")


# ============================================================
# 3. SERIAL-POSITION CURVES BY WAVE
# ============================================================

def serial_position_by_wave(panel):
    print("\n" + "=" * 65)
    print("3. SERIAL-POSITION CURVES BY WAVE")
    print("=" * 65)

    valid = panel.loc[
        panel["recall_completed"]
        & panel[WORDS].notna().all(axis=1)
    ].copy()

    for word in WORDS:
        if not valid[word].isin([0, 1]).all():
            raise ValueError(
                f"Invalid recall indicator in {word}"
            )

    probabilities = (
        valid.groupby("wave")[WORDS]
        .mean()
    )

    probabilities.to_csv(
        OUT / "word_recall_probabilities.csv"
    )

    plt.figure(figsize=(10, 6))

    for wave, row in probabilities.iterrows():
        plt.plot(
            range(1, 11),
            row.values * 100,
            marker="o",
            linewidth=2,
            label=f"Wave {int(wave)}"
        )

    plt.axvspan(
        0.5, 3.5,
        alpha=0.06,
        color="steelblue"
    )

    plt.axvspan(
        3.5, 7.5,
        alpha=0.04,
        color="gray"
    )

    plt.axvspan(
        7.5, 10.5,
        alpha=0.06,
        color="orange"
    )

    plt.xlabel("Serial presentation position")
    plt.ylabel("Recall probability (%)")
    plt.title("CERAD Trial 1: Serial Position by Wave")
    plt.xticks(range(1, 11))
    plt.ylim(0, 100)
    plt.legend()

    save_figure("spc_by_wave.png")

    print("\nItem recall probabilities:")
    print(probabilities.round(3))

    return valid


# ============================================================
# 4. SERIAL POSITION BY EDUCATION
# ============================================================

def serial_position_by_education(valid):
    print("\n" + "=" * 65)
    print("4. SERIAL POSITION BY EDUCATION")
    print("=" * 65)

    education = valid.dropna(
        subset=["education_level"]
    ).copy()

    education["education_group"] = (
        education["education_level"]
        .map(EDUCATION_LABELS)
    )

    education = education.dropna(
        subset=["education_group"]
    )

    # Use baseline only to avoid counting a participant
    # twice in the education-stratified descriptive curve.

    education = education.loc[
        education["wave"].eq(1)
    ].copy()

    probabilities = (
        education.groupby("education_group")[WORDS]
        .mean()
    )

    probabilities.to_csv(
        OUT / "spc_by_education.csv"
    )

    plt.figure(figsize=(10, 6))

    for label, row in probabilities.iterrows():
        plt.plot(
            range(1, 11),
            row.values * 100,
            marker="o",
            label=label
        )

    plt.xlabel("Serial presentation position")
    plt.ylabel("Recall probability (%)")
    plt.title(
        "Baseline Serial-Position Curves by Education"
    )
    plt.xticks(range(1, 11))
    plt.ylim(0, 100)
    plt.legend(title="Education")

    save_figure("spc_by_education.png")

    print("\nBaseline education group counts:")
    print(
        education["education_group"]
        .value_counts()
    )


# ============================================================
# 5. DELAYED RECALL DISTRIBUTIONS
# ============================================================

def delayed_recall_distributions(panel):
    print("\n" + "=" * 65)
    print("5. DELAYED RECALL DISTRIBUTIONS")
    print("=" * 65)

    valid = panel.loc[
        panel["delayed_recall_completed"]
        & panel["delayed_recall_score"].notna()
    ].copy()

    print(
        valid.groupby("wave")[
            "delayed_recall_score"
        ].describe()
    )

    plt.figure(figsize=(9, 5))

    sns.histplot(
        data=valid,
        x="delayed_recall_score",
        hue="wave",
        bins=np.arange(-0.5, 11.5, 1),
        element="step",
        stat="count",
        common_norm=False
    )

    plt.xlabel("CERAD delayed-recall score")
    plt.ylabel("Assessment count")
    plt.title("CERAD Delayed Recall by Wave")

    save_figure("delayed_recall_distribution.png")


# ============================================================
# 6. PAIRED DELAYED RECALL CHANGE
# ============================================================

def paired_delayed_recall_analysis(paired):
    print("\n" + "=" * 65)
    print("6. PAIRED DELAYED RECALL CHANGE")
    print("=" * 65)

    change = paired[
        "delayed_recall_change"
    ].dropna()

    print("\nPaired analysis participants:", len(paired))
    print("\nChange statistics:")
    print(change.describe())

    change.describe().to_csv(
        OUT / "delayed_recall_change_summary.csv"
    )

    plt.figure(figsize=(9, 5))

    sns.histplot(
        change,
        bins=np.arange(-7.5, 7.5, 1),
        discrete=False
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        label="No change"
    )

    plt.xlabel("Delayed recall change (W2 - W1)")
    plt.ylabel("Number of participants")
    plt.title(
        "Within-Participant Delayed-Recall Change"
    )
    plt.legend()

    save_figure("delayed_recall_change_distribution.png")

    plt.figure(figsize=(7, 7))

    sns.scatterplot(
        data=paired,
        x="delayed_recall_baseline",
        y="delayed_recall_followup",
        alpha=0.6
    )

    plt.plot(
        [0, 10],
        [0, 10],
        linestyle="--",
        color="black",
        label="No change"
    )

    plt.xlim(-0.5, 10.5)
    plt.ylim(-0.5, 10.5)
    plt.xlabel("Baseline delayed recall")
    plt.ylabel("Follow-up delayed recall")
    plt.title("Paired CERAD Delayed-Recall Scores")
    plt.legend()

    save_figure("paired_delayed_recall_scatter.png")


# ============================================================
# 7. BASELINE SPC VS FOLLOW-UP MEMORY
# ============================================================

def baseline_spc_followup_analysis(paired):
    print("\n" + "=" * 65)
    print("7. BASELINE SPC VS FOLLOW-UP MEMORY")
    print("=" * 65)

    predictors = [
        "P_i_w1",
        "M_i_w1",
        "R_i_w1",
        "T_i_w1",
        "age_baseline",
        "education_level",
        "delayed_recall_baseline",
    ]

    outcomes = [
        "delayed_recall_followup",
        "delayed_recall_change",
    ]

    records = []

    for predictor in predictors:
        for outcome in outcomes:

            subset = paired[
                [predictor, outcome]
            ].dropna()

            if (
                len(subset) < 3
                or subset[predictor].nunique() < 2
                or subset[outcome].nunique() < 2
            ):
                rho = np.nan
                n = len(subset)
            else:
                rho = subset[
                    [predictor, outcome]
                ].corr(
                    method="spearman"
                ).iloc[0, 1]

                n = len(subset)

            records.append({
                "predictor": predictor,
                "outcome": outcome,
                "n": n,
                "spearman_rho": rho,
            })

    correlations = pd.DataFrame(records)

    correlations.to_csv(
        OUT / "baseline_spc_followup_correlations.csv",
        index=False
    )

    print("\nLongitudinal exploratory correlations:")
    print(correlations.round(3))

    for predictor, label in [
        ("P_i_w1", "Baseline primacy"),
        ("M_i_w1", "Baseline middle"),
        ("R_i_w1", "Baseline recency"),
    ]:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=paired,
            x=predictor,
            y="delayed_recall_followup",
            color="lightsteelblue"
        )

        plt.xlabel(f"{label} score")
        plt.ylabel("Follow-up delayed recall")
        plt.title(
            f"{label} vs Follow-up Delayed Recall"
        )

        save_figure(
            f"{predictor}_vs_followup_delayed_recall.png"
        )


# ============================================================
# 8. MMSE: SECONDARY DESCRIPTIVE ANALYSIS
# ============================================================

def mmse_secondary_analysis(panel, wide):
    print("\n" + "=" * 65)
    print("8. MMSE SECONDARY ANALYSIS")
    print("=" * 65)

    valid = panel.loc[
        panel["mmse_fully_completed"]
        & panel["mmse_score"].notna()
    ].copy()

    print(
        "\nFully completed MMSE by wave:"
    )
    print(
        valid.groupby("wave")[
            "mmse_score"
        ].describe()
    )

    if not valid.empty:
        plt.figure(figsize=(9, 5))

        sns.histplot(
            data=valid,
            x="mmse_score",
            hue="wave",
            bins=np.arange(-0.5, 31.5, 1),
            element="step"
        )

        plt.xlabel("MMSE total score")
        plt.title(
            "Fully Completed MMSE Scores by Wave"
        )

        save_figure("mmse_distribution.png")

    change = pd.to_numeric(
        wide["mmse_change"],
        errors="coerce"
    ).dropna()

    print("\nValid paired MMSE changes:", len(change))

    if not change.empty:
        print(change.describe())

        plt.figure(figsize=(8, 5))

        sns.histplot(
            change,
            bins=15
        )

        plt.axvline(
            0,
            color="black",
            linestyle="--"
        )

        plt.xlabel("MMSE change (W2 - W1)")
        plt.title("Paired Fully Completed MMSE Change")

        save_figure("mmse_change_distribution.png")


# ============================================================
# 9. BASELINE SPEARMAN CORRELATIONS
# ============================================================

def baseline_correlations(panel):
    print("\n" + "=" * 65)
    print("9. BASELINE CORRELATION MATRIX")
    print("=" * 65)

    baseline = panel.loc[
        panel["wave"].eq(1)
    ].copy()

    columns = [
        "P_i",
        "M_i",
        "R_i",
        "T_i",
        "age_baseline",
        "education_level",
        "delayed_recall_score",
    ]

    correlation = baseline[columns].corr(
        method="spearman"
    )

    correlation.to_csv(
        OUT / "spearman_correlation_matrix.csv"
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        square=True
    )

    plt.title(
        "Baseline Spearman Correlation Matrix"
    )

    save_figure("correlation_heatmap.png")


# ============================================================
# 10. ATTRITION AND SELECTION CHECKS
# ============================================================

def selection_analysis(panel, wide):
    print("\n" + "=" * 65)
    print("10. ATTRITION AND SELECTION")
    print("=" * 65)

    baseline = panel.loc[
        panel["wave"].eq(1),
        [
            "prim_key",
            "age_baseline",
            "education_level",
            "T_i",
            "delayed_recall_score",
        ]
    ].copy()

    baseline = baseline.merge(
        wide[
            ["prim_key", "valid_delayed_pair"]
        ],
        on="prim_key",
        how="left",
        validate="one_to_one"
    )

    baseline["valid_delayed_pair"] = (
        baseline["valid_delayed_pair"]
        .fillna(False)
    )

    columns = [
        "age_baseline",
        "education_level",
        "T_i",
        "delayed_recall_score",
    ]

    summary = (
        baseline.groupby("valid_delayed_pair")[
            columns
        ]
        .agg(["count", "mean", "std", "median"])
    )

    summary.to_csv(
        OUT / "paired_selection_comparison.csv"
    )

    print("\nBaseline paired-selection comparison:")
    print(summary.round(3))

    print(
        "\nPaired outcome availability:"
    )
    print(
        baseline["valid_delayed_pair"]
        .value_counts()
    )


# ============================================================
# 11. ADMINISTRATION-QUALITY AUDIT
# ============================================================

def administration_quality(panel, paired):
    print("\n" + "=" * 65)
    print("11. ADMINISTRATION QUALITY")
    print("=" * 65)

    admin_summary = (
        panel.groupby(
            ["wave", "recall_admin_issue"],
            dropna=False
        )
        .size()
        .rename("n")
        .reset_index()
    )

    admin_summary.to_csv(
        OUT / "recall_administration_issues.csv",
        index=False
    )

    print("\nRecall administration issue codes:")
    print(admin_summary)

    if "recall_admin_issue_w1" in paired.columns:

        paired_admin = (
            paired["recall_admin_issue_w1"]
            .value_counts(dropna=False)
        )

        print(
            "\nPaired cohort baseline administration issues:"
        )
        print(paired_admin)


# ============================================================
# MAIN
# ============================================================

def run_comprehensive_eda():

    print("=" * 65)
    print("HAALSI PHASE 1: COMPREHENSIVE EDA")
    print("=" * 65)

    panel, wide, paired = load_datasets()

    cohort_description(panel, paired)

    missingness_analysis(panel)

    valid = serial_position_by_wave(panel)

    serial_position_by_education(valid)

    delayed_recall_distributions(panel)

    paired_delayed_recall_analysis(paired)

    baseline_spc_followup_analysis(paired)

    mmse_secondary_analysis(panel, wide)

    baseline_correlations(panel)

    selection_analysis(panel, wide)

    administration_quality(panel, paired)

    print("\n" + "=" * 65)
    print("COMPREHENSIVE EDA COMPLETED")
    print("=" * 65)

    print("\nFigures saved to:", FIG)
    print("Tables saved to:", OUT)

    print(
        "\nInterpretation: Observed score changes "
        "do not by themselves establish cognitive "
        "decline, improvement, or dementia status."
    )


if __name__ == "__main__":
    run_comprehensive_eda()
