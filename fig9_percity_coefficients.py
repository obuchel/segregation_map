"""
Figure 9: Per-city comparison of the ethnic-diversity coefficient on
mentions diversity, alone vs. jointly with income. One panel per city;
each panel shows the ethnic_diversity coefficient (+/- SE) estimated
alone (red) and jointly with income (gray).

Uses `ethnic_diversity` (normalized Shannon entropy across all 8 Census
race/ethnicity categories) as the single composition predictor, in
place of separate pct_black/pct_other_race shares -- consistent with
Figures 7 and 8. That avoids leaving pct_white as an implicit omitted
reference category in any of these per-city regressions.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_cities, CITIES, CITY_DISPLAY
from stats_utils import fit_coefs

INCOME_TERMS = ["Income_10k", "Income_10k_sq"]
ETH_TERMS = ["ethnic_diversity"]
CITY_ORDER = [CITY_DISPLAY.get(c, c) for c in CITIES]


def make_figure(df, out_path="output/figure9_per_city_mentions_coefficients.png"):
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True)

    for ax, city in zip(axes, CITY_ORDER):
        sub = df[df["city"] == city]
        alone, _ = fit_coefs(sub, "mentions_diversity", ETH_TERMS)
        joint, _ = fit_coefs(sub, "mentions_diversity", ETH_TERMS + INCOME_TERMS)

        x = np.array([0])
        width = 0.35
        alone_coef = [alone["ethnic_diversity"][0]]
        alone_se = [alone["ethnic_diversity"][1]]
        joint_coef = [joint["ethnic_diversity"][0]]
        joint_se = [joint["ethnic_diversity"][1]]

        ax.bar(x - width / 2, alone_coef, width, yerr=alone_se, color="firebrick", capsize=3,
               label="Alone")
        ax.bar(x + width / 2, joint_coef, width, yerr=joint_se, color="gray", capsize=3,
               label="+ income")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["ethnic_\ndiversity"], fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_title(city, fontsize=11)

    axes[0].set_ylabel("Coefficient on mentions diversity")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Figure 9: Per-city ethnic-diversity coefficient on mentions diversity, alone vs. jointly with income",
                 fontsize=11.5, y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_all_cities()
    make_figure(df)
