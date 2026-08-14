"""
Figure 6: Per-network relationship between percentage Black population
and connection diversity, pooled across all five cities. Mean diversity
(entropy) by within-city decile of pct_black, mobility (left) vs
mentions (right), with a fitted quadratic overlay.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_all_cities
from stats_utils import quadratic_fit

NETWORKS = [("mobility_diversity", "Mobility"), ("mentions_diversity", "Mentions")]


def make_figure(df, out_path="output/figure6_ethnicity_diversity_regression.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    for ax, (col, label) in zip(axes, NETWORKS):
        grp = df.groupby("pct_black_decile")[col].agg(["mean", "sem"]).dropna()
        ax.errorbar(grp.index, grp["mean"], yerr=grp["sem"], fmt="o", color="black",
                    markersize=5, capsize=3, linewidth=1.2)

        model, is_inv_u, p = quadratic_fit(df, "pct_black_decile", col)
        xs = np.linspace(df["pct_black_decile"].min(), df["pct_black_decile"].max(), 100)
        pred = model.predict({"pct_black_decile": xs, "_x2": xs ** 2})
        ax.plot(xs, pred, "--", color="gray", linewidth=1.8)

        sig = "p < 0.0001" if p < 0.0001 else f"p = {p:.3g}"
        mark = "\u2713" if is_inv_u else "\u2717"
        ax.text(0.95, 0.92, mark, transform=ax.transAxes, ha="right", va="top",
                fontsize=16, color="green" if is_inv_u else "red", fontweight="bold")
        ax.set_title(f"{label} network  ({sig})", fontsize=11)
        ax.set_xlabel("Within-city decile of % Black population")
        ax.set_ylabel("Diversity (entropy)")
        ax.set_xticks(range(1, 11, 2))

    fig.suptitle("Figure 6: % Black population decile vs. connection diversity (pooled, 5 cities)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    df = load_all_cities()
    make_figure(df)
