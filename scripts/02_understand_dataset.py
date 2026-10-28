
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PHASE 2: HAALSI DATA PREPARATION
# Serial Position Curves and Longitudinal Delayed Recall
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

PARENT = DATA / "Longitudinal Public Release Dataset V5.dta"
W1 = DATA / "HAALSI_DEMENTIA_Respondent_W1.dta"
W2 = DATA / "HAALSI_DEMENTIA_Respondent_W2.dta"

WORDS = [f"trial_word_{i}" for i in range(1, 11)]


# ============================================================
# HELPERS
# ============================================================

def numeric(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.mask(x < 0)


def require(df, columns, name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: missing columns {missing}"
        )


def get_column(df, name):
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return numeric(df[name])


def validate_key(df, name):
    require(df, ["prim_key"], name)

    if df["prim_key"].isna().any():
        raise ValueError(f"{name}: missing prim_key")

    if df["prim_key"].duplicated().any():
        raise ValueError(f"{name}: duplicate prim_key")


def checked_score(df, name, completed, maximum, label):
    score = get_column(df, name)

    invalid = (
        score.notna()
        & ~score.between(0, maximum)
    )

    if invalid.any():
        raise ValueError(
            f"{label}: {invalid.sum()} invalid scores"
        )

    return score.where(completed)


# ============================================================
# PROCESS INDIVIDUAL DEMENTIA WAVE
# ============================================================

def process_wave(df, wave):

    prefix = "" if wave == 1 else "w2_"

    word_names = [
        f"{prefix}wr102as{i}"
        for i in range(1, 11)
    ]

    status_name = f"{prefix}wr_status"
    total_name = f"{prefix}wr_trial1"

    delayed_name = f"{prefix}dr_correct"
    delayed_status_name = f"{prefix}dr_status"

    mmse_name = f"{prefix}mmse_total"
    mmse_status_name = f"{prefix}mmse_status"

    admin_name = f"{prefix}wradmin"

    require(
        df,
        [
            "prim_key",
            status_name,
            total_name,
            delayed_name,
            delayed_status_name,
        ] + word_names,
        f"Wave {wave}"
    )

    result = pd.DataFrame(index=df.index)

    result["prim_key"] = df["prim_key"]
    result["wave"] = wave
    result["wave_source"] = f"Wave {wave}"

    # --------------------------------------------------------
    # Immediate recall assessment
    # --------------------------------------------------------

    result["recall_status"] = get_column(
        df, status_name
    )

    result["recall_completed"] = (
        result["recall_status"].eq(1)
    )

    result["recall_admin_issue"] = get_column(
        df, admin_name
    )

    # Suffixes 1-10 correspond to presentation positions,
    # based on the confirmed administration information.
    #
    # A missing word indicator is coded as not recalled
    # only for a completed assessment.

    for i, source in enumerate(word_names, start=1):

        raw = get_column(df, source)

        unexpected = (
            raw.notna()
            & raw.ne(1)
        )

        if unexpected.any():
            raise ValueError(
                f"{source}: unexpected indicator values"
            )

        result[f"trial_word_{i}"] = np.where(
            result["recall_completed"],
            raw.eq(1).astype(float),
            np.nan
        )

    # --------------------------------------------------------
    # Serial-position features
    # --------------------------------------------------------

    result["P_i"] = result[
        WORDS[:3]
    ].sum(axis=1, min_count=3)

    result["M_i"] = result[
        WORDS[3:7]
    ].sum(axis=1, min_count=4)

    result["R_i"] = result[
        WORDS[7:]
    ].sum(axis=1, min_count=3)

    result["T_i"] = result[
        WORDS
    ].sum(axis=1, min_count=10)

    # Normalize by number of positions in each region.
    result["primacy_rate"] = result["P_i"] / 3
    result["middle_rate"] = result["M_i"] / 4
    result["recency_rate"] = result["R_i"] / 3

    result["primacy_recency_difference"] = (
        result["primacy_rate"]
        - result["recency_rate"]
    )

    # --------------------------------------------------------
    # Verify immediate recall totals
    # --------------------------------------------------------

    result["recorded_trial1"] = checked_score(
        df,
        total_name,
        result["recall_completed"],
        10,
        f"Wave {wave} Trial 1"
    )

    mismatch = (
        result["recall_completed"]
        & result["recorded_trial1"].notna()
        & result["T_i"].ne(
            result["recorded_trial1"]
        )
    )

    missing_total = (
        result["recall_completed"]
        & result["recorded_trial1"].isna()
    )

    if mismatch.any() or missing_total.any():

        result.loc[
            mismatch | missing_total,
            [
                "prim_key",
                "T_i",
                "recorded_trial1",
            ]
        ].to_csv(
            OUT / f"wave{wave}_recall_mismatches.csv",
            index=False
        )

        raise ValueError(
            f"Wave {wave}: "
            f"{mismatch.sum()} mismatched totals; "
            f"{missing_total.sum()} missing totals"
        )

    # --------------------------------------------------------
    # CERAD delayed recall
    # --------------------------------------------------------

    result["delayed_recall_status"] = get_column(
        df,
        delayed_status_name
    )

    result["delayed_recall_completed"] = (
        result["delayed_recall_status"].eq(1)
    )

    result["delayed_recall_score"] = checked_score(
        df,
        delayed_name,
        result["delayed_recall_completed"],
        10,
        f"Wave {wave} delayed recall"
    )

    # --------------------------------------------------------
    # MMSE: secondary assessment
    # --------------------------------------------------------

    result["mmse_status"] = get_column(
        df,
        mmse_status_name
    )

    result["mmse_score"] = get_column(
        df,
        mmse_name
    )

    result["mmse_fully_completed"] = (
        result["mmse_status"].eq(1)
    )

    invalid_mmse = (
        result["mmse_score"].notna()
        & ~result["mmse_score"].between(0, 30)
    )

    if invalid_mmse.any():
        raise ValueError(
            f"Wave {wave}: MMSE scores outside 0-30"
        )

    print(f"\nWave {wave}")
    print("Participants:", len(result))
    print(
        "Immediate recall completed:",
        result["recall_completed"].sum()
    )
    print(
        "Valid delayed recall:",
        result["delayed_recall_score"].notna().sum()
    )
    print(
        "Fully completed MMSE:",
        result["mmse_fully_completed"].sum()
    )

    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("HAALSI PHASE 2: LONGITUDINAL DATA PREPARATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Load source data
    # --------------------------------------------------------

    for path in [PARENT, W1, W2]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}"
            )

    print("\nLoading datasets...")

    parent = pd.read_stata(
        PARENT,
        convert_categoricals=False
    )

    w1 = pd.read_stata(
        W1,
        convert_categoricals=False
    )

    w2 = pd.read_stata(
        W2,
        convert_categoricals=False
    )

    for name, df in [
        ("Parent", parent),
        ("W1", w1),
        ("W2", w2),
    ]:

        validate_key(df, name)

        print(name, df.shape)

    # --------------------------------------------------------
    # Baseline demographics
    # --------------------------------------------------------

    require(
        parent,
        ["w1c_bd_educ4"],
        "Parent demographics"
    )

    age_candidates = [
        "w1c_rage_calc",
        "w1bd002",
    ]

    age_col = next(
        (
            c for c in age_candidates
            if c in parent.columns
        ),
        None
    )

    if age_col is None:
        raise ValueError(
            "Baseline age variable not found."
        )

    demo = pd.DataFrame({
        "prim_key": parent["prim_key"],
        "education_level": get_column(
            parent, "w1c_bd_educ4"
        ),
        "age_baseline": get_column(
            parent, age_col
        ),
    })

    print("\nAge source:", age_col)

    # --------------------------------------------------------
    # Construct wave-level observations
    # --------------------------------------------------------

    wave1 = process_wave(w1, 1)
    wave2 = process_wave(w2, 2)

    panel = pd.concat(
        [wave1, wave2],
        ignore_index=True
    )

    panel = panel.merge(
        demo,
        on="prim_key",
        how="left",
        validate="many_to_one"
    )

    panel = panel.sort_values(
        ["prim_key", "wave"]
    ).reset_index(drop=True)

    panel_path = (
        DATA / "HAALSI_Narrowed_SPC_Dataset.csv"
    )

    panel.to_csv(
        panel_path,
        index=False
    )

    # --------------------------------------------------------
    # Create wide-format longitudinal dataset
    # --------------------------------------------------------

    wide_columns = [
        *WORDS,
        "P_i",
        "M_i",
        "R_i",
        "T_i",
        "primacy_rate",
        "middle_rate",
        "recency_rate",
        "primacy_recency_difference",
        "recorded_trial1",
        "recall_status",
        "recall_completed",
        "recall_admin_issue",
        "delayed_recall_status",
        "delayed_recall_completed",
        "delayed_recall_score",
        "mmse_score",
        "mmse_status",
        "mmse_fully_completed",
    ]

    wide = panel.pivot(
        index="prim_key",
        columns="wave",
        values=wide_columns
    )

    wide.columns = [
        f"{name}_w{wave}"
        for name, wave in wide.columns
    ]

    wide = wide.reset_index()

    wide = wide.merge(
        demo,
        on="prim_key",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Normalize pivoted column types
    # --------------------------------------------------------

    numeric_columns = [
        *WORDS,
        "P_i",
        "M_i",
        "R_i",
        "T_i",
        "primacy_rate",
        "middle_rate",
        "recency_rate",
        "primacy_recency_difference",
        "recorded_trial1",
        "recall_status",
        "recall_admin_issue",
        "delayed_recall_status",
        "delayed_recall_score",
        "mmse_score",
        "mmse_status",
    ]

    for name in numeric_columns:
        for wave in [1, 2]:

            column = f"{name}_w{wave}"

            wide[column] = pd.to_numeric(
                wide[column],
                errors="coerce"
            )

    for name in [
        "recall_completed",
        "delayed_recall_completed",
        "mmse_fully_completed",
    ]:

        for wave in [1, 2]:

            column = f"{name}_w{wave}"

            wide[column] = (
                wide[column]
                .fillna(False)
                .astype(bool)
            )

    # --------------------------------------------------------
    # Primary outcome: CERAD delayed recall
    # --------------------------------------------------------

    valid_delayed_pair = (
        wide["delayed_recall_completed_w1"]
        & wide["delayed_recall_completed_w2"]
        & wide["delayed_recall_score_w1"].notna()
        & wide["delayed_recall_score_w2"].notna()
    )

    wide["valid_delayed_pair"] = (
        valid_delayed_pair
    )

    wide["delayed_recall_baseline"] = (
        wide["delayed_recall_score_w1"]
    )

    wide["delayed_recall_followup"] = (
        wide["delayed_recall_score_w2"]
    )

    wide["delayed_recall_change"] = (
        wide["delayed_recall_followup"]
        - wide["delayed_recall_baseline"]
    ).where(valid_delayed_pair)

    # --------------------------------------------------------
    # Secondary outcome: MMSE
    # --------------------------------------------------------

    valid_mmse_pair = (
        wide["mmse_fully_completed_w1"]
        & wide["mmse_fully_completed_w2"]
        & wide["mmse_score_w1"].notna()
        & wide["mmse_score_w2"].notna()
    )

    wide["mmse_change"] = (
        wide["mmse_score_w2"]
        - wide["mmse_score_w1"]
    ).where(valid_mmse_pair)

    wide["mmse_followup"] = (
        wide["mmse_score_w2"]
    ).where(
        wide["mmse_fully_completed_w2"]
    )

    wide_path = (
        DATA / "HAALSI_Longitudinal_SPC_Dataset.csv"
    )

    wide.to_csv(
        wide_path,
        index=False
    )

    # --------------------------------------------------------
    # Construct primary analysis cohort
    # --------------------------------------------------------

    # Require:
    # - Valid delayed recall at both waves
    # - Completed baseline immediate recall
    # - Valid baseline SPC features
    # - Available baseline age and education
    #
    # Do not impute missing outcome values.
    # Retain administration issue flags for sensitivity
    # analyses.

    analysis_mask = (
        valid_delayed_pair
        & wide["recall_completed_w1"]
        & wide["T_i_w1"].notna()
        & wide["age_baseline"].notna()
        & wide["education_level"].notna()
    )

    analysis = wide.loc[
        analysis_mask
    ].copy()

    analysis_path = (
        DATA / "HAALSI_Paired_Delayed_Recall_Analysis.csv"
    )

    analysis.to_csv(
        analysis_path,
        index=False
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL DATA PREPARATION SUMMARY")
    print("=" * 65)

    print("\nPanel observations:", len(panel))

    print(
        "Unique participants:",
        panel["prim_key"].nunique()
    )

    print(
        "Wave 1 participants:",
        len(w1)
    )

    print(
        "Wave 2 participants:",
        len(w2)
    )

    matched = set(
        w1["prim_key"]
    ).intersection(
        set(w2["prim_key"])
    )

    print(
        "Matched participants:",
        len(matched)
    )

    print(
        "Valid paired delayed recall:",
        int(valid_delayed_pair.sum())
    )

    print(
        "Analysis-ready paired participants:",
        len(analysis)
    )

    print(
        "Valid paired fully completed MMSE:",
        int(valid_mmse_pair.sum())
    )

    print("\nDelayed recall change:")

    print(
        wide.loc[
            valid_delayed_pair,
            "delayed_recall_change"
        ].describe()
    )

    print("\nSaved datasets:")

    print(panel_path)
    print(wide_path)
    print(analysis_path)

    print("\nNo diagnostic labels created.")
    print("No missing outcomes imputed.")
    print("Data preparation completed.")


if __name__ == "__main__":
    main()
