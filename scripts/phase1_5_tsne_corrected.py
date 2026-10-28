
from pathlib import Path
import os
import json
import warnings

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"

FIG.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


ORIGINAL = DATA / "HAALSI_Narrowed_SPC_Dataset.csv"

WORDS = [f"trial_word_{i}" for i in range(1, 11)]

SEED = 42
PERPLEXITIES = [10, 20, 30, 50]

EDUCATION_LABELS = {
    1: "Education category 1",
    2: "Education category 2",
    3: "Education category 3",
    4: "Education category 4",
}


# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_data():

    path = ORIGINAL

    if not path.exists():
        raise FileNotFoundError(
            "Neither the repaired nor original panel CSV exists."
        )

    print(f"\nLoading: {path.name}")

    df = pd.read_csv(path)

    required = [
        "prim_key",
        "wave",
        "education_level",
        "recall_completed",
    ] + WORDS

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["wave"] = pd.to_numeric(df["wave"], errors="coerce")
    df["education_level"] = pd.to_numeric(
        df["education_level"], errors="coerce"
    )

    for col in WORDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["recall_completed"] = (
        df["recall_completed"]
        .astype(str)
        .str.lower()
        .isin(["true", "1", "1.0"])
    )

    baseline = df.loc[
        (df["wave"] == 1)
        & df["recall_completed"]
        & df[WORDS].notna().all(axis=1)
    ].copy()

    if baseline["prim_key"].duplicated().any():
        raise ValueError("Duplicate baseline participants")

    if not baseline[WORDS].isin([0, 1]).all().all():
        raise ValueError("Recall indicators must be binary")

    baseline["education_group"] = (
        baseline["education_level"].map(EDUCATION_LABELS)
    )

    print(f"Baseline participants: {len(baseline)}")
    print(
        "Education missing:",
        baseline["education_group"].isna().sum()
    )

    if len(baseline) < 10:
        raise ValueError("Insufficient baseline participants")

    return baseline


# ============================================================
# RECALL PATTERN AUDIT
# ============================================================

def audit_patterns(df):

    df = df.copy()

    df["pattern"] = (
        df[WORDS].astype(int).astype(str).agg("".join, axis=1)
    )

    frequency = df["pattern"].value_counts()

    df["pattern_frequency"] = (
        df["pattern"].map(frequency)
    )

    pattern_table = (
        df.groupby("pattern")
        .agg(
            frequency=("prim_key", "size"),
            education_categories=(
                "education_level",
                lambda x: x.dropna().nunique()
            ),
        )
        .reset_index()
        .sort_values("frequency", ascending=False)
    )

    pattern_table.to_csv(
        OUT / "tsne_pattern_audit.csv",
        index=False
    )

    print("\nRECALL PATTERN AUDIT")
    print(f"Unique patterns: {len(frequency)}")
    print(f"Repeated patterns: {(frequency > 1).sum()}")
    print(f"Most common pattern frequency: {frequency.max()}")
    print(
        "Participants sharing a pattern:",
        int(df["pattern_frequency"].gt(1).sum())
    )

    return df, pattern_table


# ============================================================
# T-SNE HELPER
# ============================================================

def fit_tsne(X, perplexity, seed=SEED):

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        metric="euclidean",
    )

    return model.fit_transform(X)


def save_embedding(df, embedding, filename):

    output = df[
        [
            "prim_key",
            "education_level",
            "education_group",
            "pattern",
            "pattern_frequency",
        ]
    ].copy()

    output["tsne_1"] = embedding[:, 0]
    output["tsne_2"] = embedding[:, 1]

    output.to_csv(OUT / filename, index=False)


# ============================================================
# PARTICIPANT-LEVEL T-SNE
# ============================================================

def participant_tsne(df):

    X = df[WORDS].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 11)
    )

    diagnostics = []

    colors = {
        1: "#377eb8",
        2: "#ff7f00",
        3: "#4daf4a",
        4: "#984ea3",
    }

    for ax, perplexity in zip(
        axes.flat, PERPLEXITIES
    ):

        if perplexity >= len(X):
            ax.set_visible(False)
            continue

        embedding = fit_tsne(X, perplexity)

        score = trustworthiness(
            X,
            embedding,
            n_neighbors=10,
            metric="euclidean",
        )

        diagnostics.append({
            "perplexity": perplexity,
            "trustworthiness_k10": float(score),
            "n_participants": len(df),
        })

        for education, label in EDUCATION_LABELS.items():

            mask = df["education_level"].eq(education).to_numpy()

            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=24,
                alpha=0.55,
                color=colors[education],
                label=label,
                edgecolors="none",
            )

        unknown = df["education_level"].isna().to_numpy()

        if unknown.any():
            ax.scatter(
                embedding[unknown, 0],
                embedding[unknown, 1],
                s=24,
                alpha=0.5,
                color="gray",
                label="Education missing",
            )

        ax.set_title(
            f"Perplexity {perplexity} | Trustworthiness {score:.3f}"
        )

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        save_embedding(
            df,
            embedding,
            f"tsne_participants_perplexity_{perplexity}.csv",
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        "CERAD Baseline Recall Patterns: t-SNE Sensitivity Analysis",
        fontsize=15,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    fig.savefig(
        FIG / "tsne_perplexity_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    pd.DataFrame(diagnostics).to_csv(
        OUT / "tsne_perplexity_diagnostics.csv",
        index=False,
    )

    return diagnostics


# ============================================================
# UNIQUE-PATTERN T-SNE
# ============================================================

def unique_pattern_tsne(df, pattern_table):

    unique = pattern_table.copy()

    X = np.array(
        [
            [int(character) for character in pattern]
            for pattern in unique["pattern"]
        ],
        dtype=float,
    )

    if len(X) <= 30:
        raise ValueError(
            "Not enough unique patterns for perplexity 30"
        )

    embedding = fit_tsne(
        X,
        perplexity=30,
    )

    unique["tsne_1"] = embedding[:, 0]
    unique["tsne_2"] = embedding[:, 1]

    unique.to_csv(
        OUT / "tsne_unique_patterns.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    sizes = (
        15 + 12 * np.sqrt(unique["frequency"].to_numpy())
    )

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=sizes,
        c=unique["frequency"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.25,
    )

    fig.colorbar(
        scatter,
        ax=ax,
        label="Participants sharing recall pattern",
    )

    ax.set_title(
        "Unique CERAD Recall Patterns (Bubble Size = Frequency)"
    )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    fig.tight_layout()

    fig.savefig(
        FIG / "tsne_unique_pattern_frequency.png",
        dpi=250,
    )

    plt.close(fig)


# ============================================================
# DIRECT RECALL PROFILE PLOT
# ============================================================

def recall_profile_plot(df):

    fig, ax = plt.subplots(figsize=(10, 6))

    for education, label in EDUCATION_LABELS.items():

        group = df.loc[
            df["education_level"].eq(education)
        ]

        if group.empty:
            continue

        proportions = group[WORDS].mean().to_numpy()

        ax.plot(
            range(1, 11),
            proportions,
            marker="o",
            linewidth=2,
            label=f"{label} (n={len(group)})",
        )

    ax.set_title(
        "Observed Baseline Recall Probability by Word Position"
    )

    ax.set_xlabel("CERAD Trial 1 Word Position")
    ax.set_ylabel("Proportion Recalled")

    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 1)

    ax.axvline(3.5, linestyle="--", color="gray", alpha=0.5)
    ax.axvline(7.5, linestyle="--", color="gray", alpha=0.5)

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    fig.savefig(
        FIG / "tsne_companion_recall_profiles.png",
        dpi=250,
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HAALSI: CORRECTED T-SNE AND RECALL PATTERN DIAGNOSTICS")
    print("=" * 70)

    df = load_data()

    df, patterns = audit_patterns(df)

    print("\nRunning participant-level t-SNE...")
    diagnostics = participant_tsne(df)

    print("\nRunning unique-pattern t-SNE...")
    unique_pattern_tsne(df, patterns)

    print("\nGenerating direct recall profiles...")
    recall_profile_plot(df)

    summary = {
        "baseline_participants": int(len(df)),
        "unique_patterns": int(len(patterns)),
        "largest_pattern_frequency": int(
            patterns["frequency"].max()
        ),
        "perplexity_diagnostics": diagnostics,
        "interpretation": (
            "t-SNE is exploratory. Islands are not diagnostic "
            "classes, and distances between distant clusters "
            "should not be interpreted literally."
        ),
    }

    with open(
        OUT / "tsne_diagnostic_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 70)
    print("T-SNE ANALYSIS COMPLETE")
    print("=" * 70)

    print(json.dumps(summary, indent=2))

    print("\nFigures saved to:", FIG)
    print("Diagnostic tables saved to:", OUT)


if __name__ == "__main__":
    main()
