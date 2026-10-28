
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

PANEL_FILE = DATA / "HAALSI_Narrowed_SPC_Dataset.csv"
PAIRED_FILE = DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"

WORDS = [f"trial_word_{i}" for i in range(1, 11)]

POSITIONS = ["Primacy", "Middle", "Recency"]

EDUCATION_LABELS = {
    1: "No formal education",
    2: "Some primary",
    3: "Some secondary",
    4: "Secondary or more",
}


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{label}: missing columns {missing}"
        )


def read_data():
    if not PANEL_FILE.exists():
        raise FileNotFoundError(PANEL_FILE)

    if not PAIRED_FILE.exists():
        raise FileNotFoundError(PAIRED_FILE)

    panel = pd.read_csv(PANEL_FILE)
    paired = pd.read_csv(PAIRED_FILE)

    required_panel = [
        "prim_key",
        "wave",
        "recall_completed",
        "age_baseline",
        "education_level",
    ] + WORDS

    required_paired = [
        "prim_key",
        "delayed_recall_baseline",
        "delayed_recall_followup",
        "delayed_recall_change",
        "P_i_w1",
        "M_i_w1",
        "R_i_w1",
    ]

    require_columns(
        panel, required_panel, "Panel"
    )

    require_columns(
        paired, required_paired, "Paired"
    )

    numeric_columns = (
        WORDS
        + ["wave", "age_baseline", "education_level"]
    )

    for column in numeric_columns:
        panel[column] = pd.to_numeric(
            panel[column],
            errors="coerce"
        )

    panel["recall_completed"] = (
        panel["recall_completed"]
        .astype("string")
        .str.lower()
        .eq("true")
    )

    if panel.duplicated(["prim_key", "wave"]).any():
        raise ValueError(
            "Duplicate participant-wave records detected."
        )

    if paired["prim_key"].duplicated().any():
        raise ValueError(
            "Duplicate participants in paired dataset."
        )

    for word in WORDS:
        invalid = (
            panel[word].notna()
            & ~panel[word].isin([0, 1])
        )

        if invalid.any():
            raise ValueError(
                f"{word}: values outside 0/1"
            )

    valid = panel.loc[
        panel["recall_completed"]
        & panel[WORDS].notna().all(axis=1)
    ].copy()

    valid["education_level"] = (
        valid["education_level"]
        .where(valid["education_level"].isin(
            EDUCATION_LABELS.keys()
        ))
    )

    valid["education_group"] = (
        valid["education_level"]
        .map(EDUCATION_LABELS)
    )

    print("\nPanel observations:", len(panel))
    print("Valid immediate-recall assessments:", len(valid))
    print("Paired analysis participants:", len(paired))

    return panel, paired, valid


# ============================================================
# 1. t-SNE: RECALL PATTERNS ONLY
# ============================================================

def run_tsne(valid):

    print("\n" + "=" * 65)
    print("1. t-SNE: IMMEDIATE RECALL PATTERNS")
    print("=" * 65)

    # Each participant contributes only their baseline
    # assessment. This avoids counting repeated observations
    # as independent points in the visualization.

    tsne_df = valid.loc[
        valid["wave"].eq(1)
    ].copy()

    if len(tsne_df) < 10:
        warnings.warn(
            "Too few baseline observations for t-SNE."
        )
        return

    # Education and age are intentionally excluded.
    # The embedding represents only word recall patterns.

    X = tsne_df[WORDS].to_numpy(dtype=float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    perplexity = min(
        30.0,
        max(5.0, (len(tsne_df) - 1) / 3)
    )

    perplexity = min(
        perplexity,
        len(tsne_df) - 1
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto"
    )

    embedding = tsne.fit_transform(X_scaled)

    tsne_df["tsne_1"] = embedding[:, 0]
    tsne_df["tsne_2"] = embedding[:, 1]

    tsne_df[
        [
            "prim_key",
            "wave",
            "education_level",
            "education_group",
            "tsne_1",
            "tsne_2",
        ]
    ].to_csv(
        OUT / "tsne_coordinates.csv",
        index=False
    )

    plt.figure(figsize=(10, 7))

    sns.scatterplot(
        data=tsne_df,
        x="tsne_1",
        y="tsne_2",
        hue="education_group",
        hue_order=[
            EDUCATION_LABELS[k]
            for k in sorted(EDUCATION_LABELS)
        ],
        palette="viridis",
        alpha=0.7,
        s=45
    )

    plt.title(
        "Baseline CERAD Recall Patterns by Education"
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(
        title="Education",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()

    plt.savefig(
        FIG / "tsne_spc_education.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(
        "t-SNE participants:",
        len(tsne_df)
    )
    print(
        "Saved:",
        FIG / "tsne_spc_education.png"
    )


# ============================================================
# 2. WORD-LEVEL LONGITUDINAL GEE
# ============================================================

def run_gee(valid):

    print("\n" + "=" * 65)
    print("2. GEE: EDUCATION × SERIAL POSITION")
    print("=" * 65)

    analysis = valid.dropna(
        subset=[
            "prim_key",
            "wave",
            "education_level",
            "age_baseline",
        ]
    ).copy()

    if analysis.empty:
        raise ValueError(
            "No complete observations for GEE."
        )

    # Keep documented education categories.
    # Do not assume the difference between adjacent
    # education categories is numerically equal.

    analysis["education_level"] = (
        analysis["education_level"].astype(int)
    )

    age_mean = analysis["age_baseline"].mean()
    age_sd = analysis["age_baseline"].std()

    if not np.isfinite(age_sd) or age_sd <= 0:
        raise ValueError(
            "Age has insufficient variation."
        )

    analysis["age_z"] = (
        analysis["age_baseline"] - age_mean
    ) / age_sd

    long = analysis.melt(
        id_vars=[
            "prim_key",
            "wave",
            "education_level",
            "age_z",
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
            long["word_position"].between(1, 3),
            long["word_position"].between(4, 7),
            long["word_position"].between(8, 10),
        ],
        POSITIONS,
        default="Unknown"
    )

    long = long.loc[
        long["position"].ne("Unknown")
    ].dropna(
        subset=[
            "recalled",
            "education_level",
            "age_z",
        ]
    ).copy()

    long["position"] = pd.Categorical(
        long["position"],
        categories=POSITIONS,
        ordered=True
    )

    long["education_level"] = pd.Categorical(
        long["education_level"],
        categories=sorted(EDUCATION_LABELS),
        ordered=True
    )

    # Cluster by participant, not participant-wave.
    # All words and both waves for a participant belong
    # to the same repeated-measures cluster.

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

    summary_path = (
        OUT / "position_education_interaction.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(result.summary().as_text())
        file.write(
            "\n\nInterpretation note:\n"
            "This is a population-averaged association "
            "model, not a predictive model for cognitive "
            "decline or a dementia diagnosis.\n"
        )

    coef = pd.DataFrame({
        "term": result.params.index,
        "coefficient": result.params.values,
        "std_error": result.bse.values,
        "p_value": result.pvalues.values,
        "odds_ratio": np.exp(result.params.values),
    })

    coef.to_csv(
        OUT / "gee_interaction_coefficients.csv",
        index=False
    )

    print(
        "\nGEE participants:",
        long["prim_key"].nunique()
    )
    print(
        "Participant-wave assessments:",
        analysis.shape[0]
    )
    print(
        "Word-level observations:",
        len(long)
    )

    return result, long


# ============================================================
# 3. PREDICTED PROBABILITIES
# ============================================================

def plot_gee_predictions(result, long):

    print("\n" + "=" * 65)
    print("3. GEE PREDICTED RECALL PROBABILITIES")
    print("=" * 65)

    observed_education = sorted(
        long["education_level"]
        .dropna()
        .astype(int)
        .unique()
    )

    prediction_rows = []

    for education in observed_education:
        for position in POSITIONS:

            prediction_rows.append({
                "position": position,
                "education_level": education,
                "age_z": 0.0,
                "wave": 1,
                "education_group": (
                    EDUCATION_LABELS[education]
                ),
            })

    prediction_df = pd.DataFrame(prediction_rows)

    prediction_df["position"] = pd.Categorical(
        prediction_df["position"],
        categories=POSITIONS,
        ordered=True
    )

    prediction_df["education_level"] = pd.Categorical(
        prediction_df["education_level"],
        categories=sorted(EDUCATION_LABELS),
        ordered=True
    )

    prediction_df["predicted_probability"] = (
        result.predict(prediction_df)
    )

    prediction_df.to_csv(
        OUT / "gee_predicted_probabilities.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=prediction_df,
        x="position",
        y="predicted_probability",
        hue="education_group",
        hue_order=[
            EDUCATION_LABELS[k]
            for k in observed_education
        ],
        marker="o",
        markersize=8
    )

    plt.ylim(0, 1)
    plt.xlabel("Serial-position region")
    plt.ylabel("Predicted recall probability")
    plt.title(
        "Education × Serial Position: "
        "Population-Averaged GEE Estimates"
    )
    plt.legend(
        title="Education",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )
    plt.tight_layout()

    plt.savefig(
        FIG / "gee_position_education.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(
        "Saved:",
        FIG / "gee_position_education.png"
    )


# ============================================================
# 4. PAIRED COHORT DESCRIPTIVE CHECK
# ============================================================

def paired_cohort_check(paired):

    print("\n" + "=" * 65)
    print("4. PAIRED DELAYED-RECALL COHORT")
    print("=" * 65)

    numeric_columns = [
        "delayed_recall_baseline",
        "delayed_recall_followup",
        "delayed_recall_change",
        "P_i_w1",
        "M_i_w1",
        "R_i_w1",
    ]

    for column in numeric_columns:
        paired[column] = pd.to_numeric(
            paired[column],
            errors="coerce"
        )

    print("\nParticipants:", len(paired))

    print(
        paired[numeric_columns].describe()
    )

    summary = paired[numeric_columns].describe().T

    summary.to_csv(
        OUT / "paired_delayed_recall_descriptives.csv"
    )

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=paired,
        x="delayed_recall_change",
        bins=np.arange(-7.5, 7.5, 1),
        color="steelblue"
    )

    plt.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Delayed recall change (Wave 2 - Wave 1)"
    )
    plt.ylabel("Number of participants")
    plt.title(
        "Distribution of Paired CERAD Delayed-Recall Change"
    )
    plt.tight_layout()

    plt.savefig(
        FIG / "paired_delayed_recall_change.png",
        dpi=300
    )
    plt.close()


# ============================================================
# MAIN
# ============================================================

def run_phase1_5_eda():

    print("=" * 65)
    print("PHASE 1.5: ADVANCED HAALSI EDA")
    print("=" * 65)

    panel, paired, valid = read_data()

    run_tsne(valid)

    result, long = run_gee(valid)

    plot_gee_predictions(result, long)

    paired_cohort_check(paired)

    print("\n" + "=" * 65)
    print("ADVANCED EDA COMPLETED")
    print("=" * 65)

    print(
        "This analysis does not establish "
        "causality or diagnose cognitive decline."
    )


if __name__ == "__main__":
    run_phase1_5_eda()
