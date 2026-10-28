
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr

import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# SECTION 3.1: EDUCATION AND SERIAL-POSITION EFFECTS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PANEL_PATH = DATA / "HAALSI_Narrowed_SPC_Dataset.csv"

WORDS = [f"trial_word_{i}" for i in range(1, 11)]
REGIONS = ["Primacy", "Middle", "Recency"]

EDU_LABELS = {
    1: "No formal education",
    2: "Some primary",
    3: "Some secondary",
    4: "Secondary or more",
}

sns.set_theme(style="whitegrid")


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def load_data():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)

    df = pd.read_csv(PANEL_PATH)

    required = [
        "prim_key",
        "wave",
        "education_level",
        "age_baseline",
        "recall_completed",
        "P_i",
        "M_i",
        "R_i",
        "T_i",
    ] + WORDS

    require_columns(df, required)

    numeric = [
        "wave",
        "education_level",
        "age_baseline",
        "P_i",
        "M_i",
        "R_i",
        "T_i",
    ] + WORDS

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df["recall_completed"] = (
        df["recall_completed"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    if df.duplicated(["prim_key", "wave"]).any():
        raise ValueError(
            "Duplicate participant-wave observations."
        )

    valid_edu = list(EDU_LABELS)

    unexpected = (
        df["education_level"].notna()
        & ~df["education_level"].isin(valid_edu)
    )

    if unexpected.any():
        raise ValueError(
            "Unexpected education codes: "
            f"{df.loc[unexpected, 'education_level'].unique()}"
        )

    for word in WORDS:
        invalid = (
            df[word].notna()
            & ~df[word].isin([0, 1])
        )

        if invalid.any():
            raise ValueError(
                f"Invalid binary recall values: {word}"
            )

    valid = df.loc[
        df["recall_completed"]
        & df[WORDS].notna().all(axis=1)
    ].copy()

    valid = valid.dropna(
        subset=["education_level", "age_baseline"]
    )

    valid["education_level"] = (
        valid["education_level"].astype(int)
    )

    for name, col, denominator in [
        ("primacy", "P_i", 3),
        ("middle", "M_i", 4),
        ("recency", "R_i", 3),
    ]:
        valid[f"{name}_rate"] = (
            valid[col] / denominator
        )

    valid["primacy_minus_recency"] = (
        valid["primacy_rate"]
        - valid["recency_rate"]
    )

    print("Panel observations:", len(df))
    print("Unique participants:", df["prim_key"].nunique())
    print("Valid analysis assessments:", len(valid))

    return df, valid


# ============================================================
# 1. SPEARMAN EDUCATION CORRELATIONS
# ============================================================

def education_correlations(valid):

    print("\n" + "=" * 65)
    print("1. SPEARMAN EDUCATION CORRELATIONS")
    print("=" * 65)

    features = [
        "T_i",
        "P_i",
        "M_i",
        "R_i",
        "primacy_rate",
        "middle_rate",
        "recency_rate",
        "primacy_minus_recency",
    ]

    results = []

    for wave in [1, 2]:

        subset = valid.loc[
            valid["wave"].eq(wave)
        ]

        for feature in features:

            clean = subset[
                ["education_level", feature]
            ].dropna()

            if (
                len(clean) >= 3
                and clean["education_level"].nunique() > 1
                and clean[feature].nunique() > 1
            ):
                rho, p_value = spearmanr(
                    clean["education_level"],
                    clean[feature]
                )
            else:
                rho, p_value = np.nan, np.nan

            results.append({
                "wave": wave,
                "component": feature,
                "n": len(clean),
                "spearman_rho": rho,
                "p_value": p_value,
            })

    result = pd.DataFrame(results)

    result.to_csv(
        OUT / "spearman_education_correlations.csv",
        index=False
    )

    print(result.round(4).to_string(index=False))

    return result


# ============================================================
# 2. EDUCATION-STRATIFIED SPC ANALYSIS
# ============================================================

def education_spc_analysis(valid):

    print("\n" + "=" * 65)
    print("2. EDUCATION-STRATIFIED SPC")
    print("=" * 65)

    baseline = valid.loc[
        valid["wave"].eq(1)
    ].copy()

    baseline["education_group"] = (
        baseline["education_level"].map(EDU_LABELS)
    )

    features = [
        "P_i",
        "M_i",
        "R_i",
        "T_i",
        "primacy_rate",
        "middle_rate",
        "recency_rate",
        "primacy_minus_recency",
    ]

    rows = []

    for level, group in baseline.groupby(
        "education_level"
    ):

        for feature in features:

            rows.append({
                "education_level": level,
                "education_group": EDU_LABELS[level],
                "component": feature,
                "n": group[feature].notna().sum(),
                "mean": group[feature].mean(),
                "std": group[feature].std(),
                "median": group[feature].median(),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUT / "education_spc_group_summary.csv",
        index=False
    )

    print("\nBaseline education counts:")
    print(
        baseline["education_group"].value_counts()
    )

    plot_df = summary.loc[
        summary["component"].isin([
            "primacy_rate",
            "middle_rate",
            "recency_rate",
        ])
    ].copy()

    plot_df["region"] = (
        plot_df["component"]
        .str.replace("_rate", "", regex=False)
        .str.title()
    )

    plot_df["region"] = pd.Categorical(
        plot_df["region"],
        categories=REGIONS,
        ordered=True
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=plot_df,
        x="region",
        y="mean",
        hue="education_group",
        hue_order=[
            EDU_LABELS[k] for k in sorted(EDU_LABELS)
        ],
        marker="o"
    )

    plt.ylim(0, 1)
    plt.xlabel("Serial-position region")
    plt.ylabel("Mean recall probability")
    plt.title(
        "Baseline SPC Components by Education"
    )

    plt.tight_layout()

    plt.savefig(
        FIG / "section3_1_education_spc_regions.png",
        dpi=300
    )
    plt.close()

    return baseline


# ============================================================
# 3. GEE EDUCATION × POSITION INTERACTION
# ============================================================

def gee_education_position(valid):

    print("\n" + "=" * 65)
    print("3. EDUCATION × POSITION GEE")
    print("=" * 65)

    long = valid.melt(
        id_vars=[
            "prim_key",
            "wave",
            "education_level",
            "age_baseline",
        ],
        value_vars=WORDS,
        var_name="word",
        value_name="recalled"
    )

    long["word_position"] = (
        long["word"]
        .str.extract(r"(\d+)$")[0]
        .astype(int)
    )

    long["position"] = np.select(
        [
            long["word_position"] <= 3,
            long["word_position"].between(4, 7),
            long["word_position"] >= 8,
        ],
        REGIONS,
        default="Unknown"
    )

    long["position"] = pd.Categorical(
        long["position"],
        categories=REGIONS
    )

    long["education_level"] = pd.Categorical(
        long["education_level"],
        categories=list(EDU_LABELS)
    )

    age_sd = long["age_baseline"].std()

    if not np.isfinite(age_sd) or age_sd <= 0:
        raise ValueError(
            "Insufficient age variation."
        )

    long["age_z"] = (
        long["age_baseline"]
        - long["age_baseline"].mean()
    ) / age_sd

    formula = (
        "recalled ~ "
        "C(position, Treatment(reference='Primacy'))"
        " * C(education_level, Treatment(reference=1))"
        " + age_z + C(wave)"
    )

    model = smf.gee(
        formula=formula,
        groups="prim_key",
        data=long,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    )

    result = model.fit(maxiter=200)

    print(result.summary())

    with open(
        OUT / "section3_1_gee_interaction.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(result.summary().as_text())

    coefficients = pd.DataFrame({
        "term": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "p_value": result.pvalues.values,
        "odds_ratio": np.exp(result.params.values),
    })

    coefficients.to_csv(
        OUT / "section3_1_gee_coefficients.csv",
        index=False
    )

    # Joint Wald test of all position × education
    # interactions rather than relying on isolated
    # coefficient p-values.

    interaction_terms = [
        name for name in result.params.index
        if ":C(education_level" in name
    ]

    if not interaction_terms:
        raise ValueError(
            "GEE interaction terms were not found."
        )

    restriction = np.zeros(
        (len(interaction_terms), len(result.params))
    )

    for i, term in enumerate(interaction_terms):
        j = result.params.index.get_loc(term)
        restriction[i, j] = 1

    joint_test = result.wald_test(
        restriction,
        scalar=True
    )

    joint_summary = pd.DataFrame([{
        "test": "Joint education-position interaction",
        "chi_square": float(joint_test.statistic),
        "df": len(interaction_terms),
        "p_value": float(joint_test.pvalue),
        "participants": long["prim_key"].nunique(),
        "word_observations": len(long),
    }])

    joint_summary.to_csv(
        OUT / "section3_1_gee_joint_interaction.csv",
        index=False
    )

    print("\nJoint interaction test:")
    print(joint_summary.to_string(index=False))

    return result


# ============================================================
# 4. BOOTSTRAP COMPARISON OF EDUCATION ASSOCIATIONS
# ============================================================

def compare_primacy_recency(baseline):

    print("\n" + "=" * 65)
    print("4. PRIMACY VS RECENCY EDUCATION ASSOCIATION")
    print("=" * 65)

    clean = baseline[
        [
            "education_level",
            "primacy_rate",
            "recency_rate",
        ]
    ].dropna().reset_index(drop=True)

    rho_p = spearmanr(
        clean["education_level"],
        clean["primacy_rate"]
    ).statistic

    rho_r = spearmanr(
        clean["education_level"],
        clean["recency_rate"]
    ).statistic

    observed_difference = rho_p - rho_r

    rng = np.random.default_rng(42)
    bootstrap_differences = []

    for _ in range(1000):

        indices = rng.integers(
            0, len(clean), size=len(clean)
        )

        sample = clean.iloc[indices]

        if any(
            sample[col].nunique() < 2
            for col in clean.columns
        ):
            continue

        boot_p = spearmanr(
            sample["education_level"],
            sample["primacy_rate"]
        ).statistic

        boot_r = spearmanr(
            sample["education_level"],
            sample["recency_rate"]
        ).statistic

        if np.isfinite(boot_p) and np.isfinite(boot_r):
            bootstrap_differences.append(
                boot_p - boot_r
            )

    if not bootstrap_differences:
        raise ValueError(
            "No valid bootstrap replicates."
        )

    lower, upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5]
    )

    result = pd.DataFrame([{
        "n": len(clean),
        "primacy_education_rho": rho_p,
        "recency_education_rho": rho_r,
        "rho_difference": observed_difference,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "valid_bootstrap_replicates": len(
            bootstrap_differences
        ),
    }])

    result.to_csv(
        OUT / "section3_1_primacy_recency_comparison.csv",
        index=False
    )

    print(result.round(4).to_string(index=False))

    print(
        "\nA confidence interval excluding zero "
        "supports a difference between associations. "
        "It does not establish recency neutrality."
    )


# ============================================================
# 5. LONGITUDINAL CHANGES IN SPC REGIONS
# ============================================================

def longitudinal_spc_changes(panel):

    print("\n" + "=" * 65)
    print("5. LONGITUDINAL SPC CHANGES")
    print("=" * 65)

    eligible = panel.loc[
        panel["recall_completed"]
        & panel[WORDS].notna().all(axis=1)
    ].copy()

    wide = eligible.pivot(
        index="prim_key",
        columns="wave",
        values=["P_i", "M_i", "R_i"]
    )

    required = [
        (region, wave)
        for region in ["P_i", "M_i", "R_i"]
        for wave in [1, 2]
    ]

    if not set(required).issubset(wide.columns):
        raise ValueError(
            "Wave 1 or Wave 2 SPC columns missing."
        )

    changes = pd.DataFrame(index=wide.index)

    for name, column, denominator in [
        ("primacy", "P_i", 3),
        ("middle", "M_i", 4),
        ("recency", "R_i", 3),
    ]:

        changes[f"{name}_rate_w1"] = (
            wide[(column, 1)] / denominator
        )

        changes[f"{name}_rate_w2"] = (
            wide[(column, 2)] / denominator
        )

        changes[f"delta_{name}_rate"] = (
            changes[f"{name}_rate_w2"]
            - changes[f"{name}_rate_w1"]
        )

    baseline_info = (
        panel.loc[
            panel["wave"].eq(1),
            [
                "prim_key",
                "education_level",
                "age_baseline",
            ]
        ]
        .set_index("prim_key")
    )

    changes = changes.join(
        baseline_info,
        how="left"
    )

    changes = changes.dropna().reset_index()

    changes.to_csv(
        OUT / "section3_1_paired_spc_changes.csv",
        index=False
    )

    change_columns = [
        "delta_primacy_rate",
        "delta_middle_rate",
        "delta_recency_rate",
    ]

    summary = (
        changes.groupby("education_level")[
            change_columns
        ]
        .agg(["count", "mean", "std"])
    )

    summary.to_csv(
        OUT / "section3_1_spc_change_by_education.csv"
    )

    print(
        "Paired immediate-recall participants:",
        len(changes)
    )

    print("\nChanges by education:")
    print(summary.round(4))

    print(
        "\nInterpretation: Changes in recall performance "
        "may reflect practice effects, differences in "
        "administration, selection, and other factors."
    )


# ============================================================
# MAIN
# ============================================================

def run_section_3_1():

    print("=" * 65)
    print("SECTION 3.1: EDUCATION AND SERIAL-POSITION BIAS")
    print("=" * 65)

    panel, valid = load_data()

    education_correlations(valid)

    baseline = education_spc_analysis(valid)

    gee_education_position(valid)

    compare_primacy_recency(baseline)

    longitudinal_spc_changes(panel)

    print("\n" + "=" * 65)
    print("SECTION 3.1 COMPLETED")
    print("=" * 65)

    print("Tables saved to:", OUT)
    print("Figures saved to:", FIG)

    print(
        "\nNo cognitive impairment labels were created. "
        "No diagnostic classifications were inferred."
    )


if __name__ == "__main__":
    run_section_3_1()
