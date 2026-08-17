import json
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Same exclude list used in PublicationMapPage.jsx (bad/water tracts)
EXCLUDE = {
    "17031990000", "18089990000", "18091990000", "17097990000", "55059990000",
    "26163990100", "26099990100", "26147990000", "36005050400", "24029990000",
}

Q_COLORS = ["#ca0020", "#f4a582", "#f7f7f7", "#92c5de", "#0571b0"]  # RdBu, 5-class
Q_LABELS = ["Q1 (Lowest)", "Q2", "Q3", "Q4", "Q5 (Highest)"]

CITIES = [
    ("Philadelphia", "features_Philadelphia_diversity.json"),
    ("Dallas", "features_Dallas_diversity.json"),
    ("Detroit", "features_Detroit_diversity.json"),
    ("Chicago", "features_Chicago_diversity.json"),
    ("New York", "features_NewYork_diversity.json"),
]

DATA_DIR = "/mnt/user-data/uploads"


def load_city(fname):
    with open(f"{DATA_DIR}/{fname}") as f:
        d = json.load(f)
    feats = d["features"]
    # Match the map's EXCLUDE filtering
    feats = [f for f in feats if str(f["properties"].get("GEOID")) not in EXCLUDE]
    return feats


def compute_quintiles(feats):
    # Only rank tracts with a valid Income value, same as computeIncomeQuintiles() in the app
    valid = [f for f in feats if f["properties"].get("Income") not in (None,) and not (
        isinstance(f["properties"].get("Income"), float) and math.isnan(f["properties"]["Income"])
    )]
    valid.sort(key=lambda f: f["properties"]["Income"])
    n = len(valid)
    for i, f in enumerate(valid):
        f["properties"]["Income_Quantile"] = min(4, int((i / n) * 5))
    return valid  # only the ranked (valid-income) tracts matter downstream


def city_summary(feats):
    ranked = compute_quintiles(feats)
    by_q = {q: [] for q in range(5)}
    for f in ranked:
        by_q[f["properties"]["Income_Quantile"]].append(f)
    rows = []
    for q in range(5):
        group = by_q[q]
        pop = sum((f["properties"].get("Population") or 0) for f in group)
        rows.append({"q": q, "tracts": len(group), "pop": pop})
    total_pop = sum(r["pop"] for r in rows)
    running = 0
    for r in rows:
        running += r["pop"]
        r["cum_pct"] = (running / total_pop * 100) if total_pop else 0
    return rows


def draw_panel(ax, ax2, rows, title):
    x = np.arange(5)
    pops = [r["pop"] for r in rows]
    cum = [r["cum_pct"] for r in rows]
    tracts = [r["tracts"] for r in rows]

    bars = ax.bar(x, pops, color=Q_COLORS, edgecolor="#999", linewidth=0.6, width=0.65, zorder=2)

    pop_max = max(pops) if pops else 1
    y_top = math.ceil((pop_max * 1.28) / 100000) * 100000 or 100000
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Population (number of people)", color="#1f4e9c", fontsize=9.5)
    ax.tick_params(axis="y", colors="#1f4e9c", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=9)
    ax.yaxis.set_major_formatter(lambda v, pos: f"{int(v):,}")
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#1f4e9c")

    ax.set_xticks(x)
    ax.set_xticklabels(Q_LABELS, fontsize=8.8)
    ax.set_xlabel("Income Quintile", fontsize=9)

    # Bold population value label above each bar
    for xi, pop in zip(x, pops):
        ax.text(xi, pop + y_top * 0.03, f"{pop:,.0f}", ha="center", va="bottom",
                 fontsize=8.3, fontweight="bold", color="#222")

    # Tract-count callout boxes above the value labels
    for xi, pop, tr in zip(x, pops, tracts):
        ax.annotate(f"{tr} tracts", xy=(xi, pop + y_top * 0.13),
                    ha="center", va="center", fontsize=7.6, color="#333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999", lw=0.6))

    # Cumulative % line + markers on secondary axis
    ax2.plot(x, cum, color="#1f4e9c", marker="o", markersize=4, linewidth=1.8, zorder=3)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Cumulative population (%)", color="#1f4e9c", fontsize=9.5)
    ax2.tick_params(axis="y", colors="#1f4e9c", labelsize=8.5)
    ax2.yaxis.set_major_formatter(lambda v, pos: f"{int(v)}%")
    for spine in ["top"]:
        ax2.spines[spine].set_visible(False)
    ax2.spines["right"].set_color("#1f4e9c")

    for xi, c in zip(x, cum):
        va = "bottom" if c < 92 else "top"
        offset = 5 if va == "bottom" else -5
        ax2.text(xi, c + offset, f"{c:.1f}%", ha="center", va=va, fontsize=8,
                  fontweight="bold", color="#1f4e9c", clip_on=False)

    ax.set_title(f"{title} \u2013 Population by Income Quintile", fontsize=11.5, fontweight="bold")


def main():
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 10.2))
    fig.suptitle("Population by Income Quintile \u2013 Five Metro Areas", fontsize=20, fontweight="bold", y=0.98)

    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    for (city, fname), (r, c) in zip(CITIES, positions):
        ax = axes[r][c]
        ax2 = ax.twinx()
        feats = load_city(fname)
        rows = city_summary(feats)
        draw_panel(ax, ax2, rows, city)

    # Legend panel in the last (empty) cell
    legend_ax = axes[1][2]
    legend_ax.axis("off")
    pop_patch = mpatches.Patch(facecolor="#d1e5f0", edgecolor="#999", label="Population (number of people)")
    line_handle = plt.Line2D([0], [0], color="#1f4e9c", marker="o", markersize=6,
                              linewidth=2, label="Cumulative population (%)")
    legend_ax.legend(handles=[pop_patch, line_handle], loc="center", fontsize=13, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = "/mnt/user-data/outputs/all_cities_income_quintile.png"
    fig.savefig(out, dpi=170, facecolor="white")
    print("Saved:", out)


if __name__ == "__main__":
    main()
