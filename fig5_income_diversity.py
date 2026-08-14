"""
Figure 5: Per-city, per-network relationship between income and
connection diversity. Mean diversity (entropy) by within-city income
decile, 5 cities x {mobility, mentions} = 10 panels, with a fitted
quadratic (income + income^2) overlay and a checkmark/X marking
whether the curvature is a significant, negative (inverted-U) term.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_cities, CITIES, CITY_DISPLAY
from stats_utils import quadratic_fit

CITY_ORDER = [CITY_DISPLAY.get(c, c) for c in CITIES]
NETWORKS = [("mobility_diversity", "Mobility"), ("mentions_diversity", "Mentions")]


def make_figure(df, out_path="output/figure5_per_city_income_diversity.png"):
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True)

    for row, (col, label) in enumerate(NETWORKS):
        for c_i, city in enumerate(CITY_ORDER):
            ax = axes[row, c_i]
            sub = df[df["city"] == city]

            grp = sub.groupby("income_decile")[col].agg(["mean", "sem", "count"]).dropna()
            ax.errorbar(grp.index, grp["mean"], yerr=grp["sem"], fmt="o", color="black",
                        markersize=4, capsize=3, linewidth=1)

            # Quadratic fit line over the observed decile range
            model, is_inv_u, p = quadratic_fit(sub, "income_decile", col)
            xs = np.linspace(sub["income_decile"].min(), sub["income_decile"].max(), 100)
            pred = model.predict({"income_decile": xs, "_x2": xs ** 2})
            ax.plot(xs, pred, "--", color="gray", linewidth=1.5)

            mark = "\u2713" if is_inv_u else "\u2717"
            mark_color = "green" if is_inv_u else "red"
            ax.text(0.95, 0.92, mark, transform=ax.transAxes, ha="right", va="top",
                    fontsize=14, color=mark_color, fontweight="bold")

            if row == 0:
                ax.set_title(city, fontsize=11)
            if row == 1:
                ax.set_xlabel("Income decile")
            if c_i == 0:
                ax.set_ylabel(f"{label}\ndiversity (entropy)")
            ax.set_xticks(range(1, 11, 2))

    fig.suptitle("Figure 5: Income decile vs. connection diversity, by city and network",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_all_cities()
    make_figure(df)
