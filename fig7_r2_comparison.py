"""
Figure 7: Left panel -- incremental R^2 (beyond city fixed effects)
contributed by income alone, ethnic diversity alone, and both jointly,
for three outcomes (mobility diversity, mentions diversity, mobility
degree). Right panel -- coefficient (+/- SE) for ethnic_diversity on
mentions diversity, estimated alone vs. jointly with income.

Uses `ethnic_diversity` (normalized Shannon entropy across all 8 Census
race/ethnicity categories -- see data_loader.py) as the single
composition predictor, in place of separate pct_black/pct_other_race
shares. That earlier two-term version left pct_white as an implicit
omitted reference category in every regression (with city FE, the
intercept absorbs whatever share pct_black + pct_other_race don't
cover); using a single entropy score removes that reference-category
framing entirely.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_cities
from stats_utils import incremental_r2, fit_coefs

INCOME_TERMS = ["Income_10k", "Income_10k_sq"]
ETH_TERMS = ["ethnic_diversity"]

OUTCOMES = [
    ("mobility_diversity", "Mobility\ndiversity"),
    ("mentions_diversity", "Mentions\ndiversity"),
    ("mobility_degree", "Mobility\ndegree"),
]


def left_panel(ax, df):
    specs = {"Income only": INCOME_TERMS, "Ethnic diversity only": ETH_TERMS,
             "Both": INCOME_TERMS + ETH_TERMS}
    width = 0.25
    x = np.arange(len(OUTCOMES))
    for i, (label, preds) in enumerate(specs.items()):
        vals = []
        for outcome, _ in OUTCOMES:
            inc_r2, _, _ = incremental_r2(df, outcome, preds)
            vals.append(inc_r2 * 100)
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in OUTCOMES])
    ax.set_ylabel("Incremental R\u00b2 (%) beyond city FE")
    ax.set_title("Explanatory contribution by predictor set")
    ax.legend(fontsize=8)


def right_panel(ax, df):
    alone, _ = fit_coefs(df, "mentions_diversity", ETH_TERMS, extra_terms=["C(city)"])
    joint, _ = fit_coefs(df, "mentions_diversity", ETH_TERMS + INCOME_TERMS, extra_terms=["C(city)"])

    x = np.array([0])
    width = 0.35

    alone_coef = [alone["ethnic_diversity"][0]]
    alone_se = [alone["ethnic_diversity"][1]]
    joint_coef = [joint["ethnic_diversity"][0]]
    joint_se = [joint["ethnic_diversity"][1]]

    ax.bar(x - width / 2, alone_coef, width, yerr=alone_se, label="Ethnic diversity alone",
           color="firebrick", capsize=4)
    ax.bar(x + width / 2, joint_coef, width, yerr=joint_se, label="Jointly with income",
           color="gray", capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["ethnic_diversity"])
    ax.set_xlim(-1, 1)
    ax.set_ylabel("Coefficient on mentions diversity")
    ax.set_title("Ethnic diversity coefficient, alone vs. jointly with income")
    ax.legend(fontsize=8)


def make_figure(df, out_path="output/figure7_joint_income_ethnicity_regression.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    left_panel(axes[0], df)
    right_panel(axes[1], df)
    fig.suptitle("Figure 7: Income vs. ethnic-diversity contribution, and joint-model attenuation",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_all_cities()
    make_figure(df)
