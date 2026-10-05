"""All figures of the entry, drawn from the saved results (no solver, no model).

    .venv/bin/python make_figures.py            # writes figures/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).parent
WORK = ROOT / "work"
FIGURES = ROOT / "figures"
OVERLOAD = 1.001

GRIDS = {
    "IEEE 118": {"sweep": WORK / "hybrid_case118_ieee_base.parquet", "stem": WORK / "conformal_case118_ieee_base"},
    "Texas 2000": {"sweep": WORK / "texas/out/hybrid_texas_n2x450.parquet", "stem": WORK / "texas/out/conformal_texas_n2x450"},
}
# (surrogate, sigma) -> label, colour. Colour follows the surrogate in every figure.
SERIES = {
    ("genco_d", "shift"): ("GENCO", "#2a78d6"),
    ("dc_d", "shift"): ("DC, tuned", "#eb6834"),
    ("dc", "global"): ("DC, plain", "#8a8983"),
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "legend.frameon": False,
})


def percent(axis):
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGURES / name)


def budget():
    """Overloads found against the share of states sent to AC, riskiest states first."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (grid, paths) in zip(axes, GRIDS.items()):
        curves = pd.read_csv(f"{paths['stem']}_budget.csv")
        notes = []
        for (surrogate, sigma), (label, colour) in SERIES.items():
            c = curves[(curves.surrogate == surrogate) & (curves.sigma == sigma)]
            ax.plot(np.r_[0, c.ac_calls], np.r_[0, c.found], color=colour, label=label)
            done = c.ac_calls[c.found >= 1 - 1e-12].min()
            ax.plot(done, 1, "o", color=colour, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
            notes.append((done, label))
        # the two leading curves end close together: one shared label instead of two colliding ones
        (a, name_a), (b, name_b), (c_, name_c) = notes
        ax.annotate(f"{name_a} {a:.1%} · {name_b} {b:.1%}", (min(a, b), 1), xytext=(-4, 9), textcoords="offset points", ha="left", color=INK, fontsize=9)
        ax.annotate(f"{name_c} {c_:.0%}", (c_, 1), xytext=(4, -16), textcoords="offset points", ha="right", color=INK, fontsize=9)
        ax.set_title(grid)
        ax.set_xlabel("share of states sent to the AC solver")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1.12)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]), ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        percent(ax.xaxis), percent(ax.yaxis)
    axes[0].set_ylabel("real overloads found")
    axes[0].legend(loc="lower right")
    fig.suptitle("With a fixed AC budget, which states should it be spent on?", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.9, "Held-out outage sets. States ranked by how close the surrogate puts any line to its limit. Dots: budget at which every overload is found.",
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.8, wspace=0.14)
    save(fig, "budget.png")


def coverage():
    """Does the promised miss rate hold? Miss rate of real overloads against alpha over 100 random splits."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (grid, paths) in zip(axes, GRIDS.items()):
        splits = pd.read_csv(f"{paths['stem']}_splits.csv")
        ax.plot([0, 0.105], [0, 0.105], color=MUTED, linewidth=1, zorder=1)
        ax.annotate("promised", (0.1, 0.1), xytext=(-4, 6), textcoords="offset points", ha="right", color=MUTED, fontsize=9)
        for offset, ((surrogate, sigma), (label, colour)) in zip((-0.0012, 0.0012), list(SERIES.items())[:2]):
            g = splits[(splits.surrogate == surrogate) & (splits.sigma == sigma)].groupby("alpha").line_missed_share
            mean, low, high = g.mean(), g.quantile(0.1), g.quantile(0.9)
            ax.errorbar(mean.index + offset, mean, yerr=[mean - low, high - mean], fmt="o", color=colour, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=2, label=label, zorder=3)
        ax.set_title(grid)
        ax.set_xlabel("promised miss rate (alpha)")
        ax.set_xticks([0.01, 0.02, 0.05, 0.10])
        ax.set_xlim(0, 0.11), ax.set_ylim(0, 0.2)
        percent(ax.xaxis), percent(ax.yaxis)
    axes[0].set_ylabel("real overloads not flagged")
    axes[0].legend(loc="upper left")
    fig.suptitle("The promised miss rate holds on average", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.9, "100 random splits of the outage sets. Dot: mean; bar: 10th to 90th percentile. Texas has ~130 calibration overloads, so single splits scatter widely.",
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, "coverage.png")


def cost():
    """What the guarantee costs: AC runs on states that turn out to be secure."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    alphas = [0.10, 0.05, 0.02, 0.01]
    for ax, (grid, paths) in zip(axes, GRIDS.items()):
        splits = pd.read_csv(f"{paths['stem']}_splits.csv")
        for i, ((surrogate, sigma), (label, colour)) in enumerate(SERIES.items()):
            wasted = splits[(splits.surrogate == surrogate) & (splits.sigma == sigma)].groupby("alpha").wasted.mean().reindex(alphas)
            x = np.arange(len(alphas)) + (i - 1) * 0.26
            ax.bar(x, wasted, width=0.22, color=colour, label=label)
            for xi, value in zip(x, wasted):
                ax.annotate(f"{value:.0%}", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.set_title(grid)
        ax.set_xticks(np.arange(len(alphas)), [f"{a:.0%}" for a in alphas])
        ax.set_xlabel("promised miss rate (alpha)")
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
        percent(ax.yaxis)
    axes[0].set_ylabel("secure states sent to AC anyway")
    axes[1].legend(loc="upper left")
    fig.suptitle("What the guarantee costs: AC runs wasted on secure states", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.9, "Mean over 100 random splits of the outage sets. States that really have an overload always go to AC and are not counted here.",
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, "cost.png")


def scatter():
    """Predicted against true loading for the lines that are anywhere near their limit."""
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.6), sharex=True, sharey=True)
    for row, (grid, paths) in zip(axes, GRIDS.items()):
        data = pd.read_parquet(paths["sweep"], columns=["ac", "genco", "dc"])
        data = data[data.ac > 0.5]
        data = data.sample(min(len(data), 60000), random_state=0)
        for ax, (column, label, colour) in zip(row, (("genco", "GENCO", "#2a78d6"), ("dc", "DC, plain", "#8a8983"))):
            ax.plot([0.5, 1.5], [0.5, 1.5], color=MUTED, linewidth=1, zorder=1)
            ax.scatter(data.ac, data[column], s=4, color=colour, alpha=0.25, linewidths=0, rasterized=True)
            ax.axvline(1, color=INK, linewidth=0.8), ax.axhline(1, color=INK, linewidth=0.8)
            missed = ((data.ac > OVERLOAD) & (data[column] <= OVERLOAD)).sum()
            false = ((data.ac <= OVERLOAD) & (data[column] > OVERLOAD)).sum()
            ax.text(1.02, 0.56, f"missed: {missed}", color=INK, fontsize=9)
            ax.text(0.52, 1.42, f"false alarms: {false}", color=INK, fontsize=9)
            ax.set_title(f"{grid} · {label}")
            ax.set_xlim(0.5, 1.5), ax.set_ylim(0.5, 1.5)
            percent(ax.xaxis), percent(ax.yaxis)
    for ax in axes[1]:
        ax.set_xlabel("true loading (AC power flow)")
    for ax in axes[:, 0]:
        ax.set_ylabel("predicted loading")
    fig.suptitle("Line loadings near the limit: prediction against AC power flow", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.925, "Line readings above 50% loading from all outage states (up to 60,000 sampled per grid). Counts refer to the sampled points.", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.88, wspace=0.08, hspace=0.16)
    save(fig, "scatter.png")


def shift():
    """Error grows with how far the outage moved the line: the reason for the adaptive band."""
    edges = [-1, 0.1, 1, 5, 10, 20, 1e4]
    labels = ["<0.1", "0.1–1", "1–5", "5–10", "10–20", ">20"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (grid, paths) in zip(axes, GRIDS.items()):
        data = pd.read_parquet(paths["sweep"], columns=["load_scale", "outages", "line", "ac", "genco", "dc"])
        base = data[data.outages == ""].set_index(["load_scale", "line"])[["ac", "genco", "dc"]].add_suffix("0")
        data = data[data.outages != ""].join(base, on=["load_scale", "line"])
        bucket = pd.cut((data.genco - data.genco0).abs() * 100, edges, labels=labels)
        for i, (name, error, colour) in enumerate((
                ("GENCO", (data.ac0 + data.genco - data.genco0 - data.ac).abs(), "#2a78d6"),
                ("DC, tuned", (data.ac0 + data.dc - data.dc0 - data.ac).abs(), "#eb6834"))):
            p99 = error.groupby(bucket, observed=True).quantile(0.99) * 100
            x = np.arange(len(labels)) + (i - 0.5) * 0.36
            ax.bar(x, p99, width=0.32, color=colour, label=name)
            for xi, value in zip(x, p99):
                ax.annotate(f"{value:.1f}", (xi, value), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8)
        ax.set_title(grid)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("99th percentile loading error (points)")
    axes[0].legend(loc="upper left")
    fig.suptitle("Errors live where the outage moves the flow", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.9, "Both surrogates anchored on the AC base case. A line the outage leaves alone is predicted almost exactly, so its band can be narrow.",
             color=MUTED, fontsize=9)
    fig.supxlabel("how far the outage moved the line from its base-case loading (points of rating)", color=MUTED, fontsize=10, y=0.0)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, "shift.png")

GENCO, GENCO1, DC, DC1 = "#2a78d6", "#1baf7a", "#8a8983", "#eb6834"


def onestep():
    """The organisers' example setup (GOC 500): overloads found against scenarios sent to AC."""
    curves = pd.read_csv(WORK / "goc500/budget_newton_tiny_seed0.csv")
    lines = pd.read_parquet(WORK / "goc500/lines_newton_tiny_seed0.parquet", columns=["state", "ac", "genco", "genco1", "dc", "dc1"])
    n, real = lines.state.nunique(), lines.ac > OVERLOAD
    critical = lines.state[real].nunique() / n
    series = {"genco1": ("GENCO + one Newton step", GENCO1), "genco": ("GENCO", GENCO), "dc": ("DC", DC), "dc1": ("DC + one Newton step", DC1)}
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.axvline(critical, color=INK, linewidth=0.8)
    ax.annotate(f"{critical:.1%} of scenarios really have an overload", (critical, 0.05), xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=9)
    for column, (label, colour) in series.items():
        c = curves[curves.surrogate == column]
        ax.plot(np.r_[0, c.ac_calls], np.r_[0, c.found_lines], color=colour, label=label)
        # operating point of the plain rule "send to AC if any predicted loading is above the limit"
        sent = lines.state[lines[column] > OVERLOAD].unique()
        found = 1 - (real & ~lines.state.isin(sent)).sum() / real.sum()
        ax.plot(len(sent) / n, found, "o", color=colour, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        offset = {"genco1": (10, -4), "genco": (8, -12), "dc": (8, -12), "dc1": (-10, -18)}[column]
        ax.annotate(f"{len(sent) / n:.1%} to AC, {found:.1%} found", (len(sent) / n, found), xytext=offset, textcoords="offset points",
                    ha="right" if offset[0] < 0 else "left", color=INK, fontsize=9)
    ax.set_xlabel("share of scenarios sent to the AC solver")
    ax.set_ylabel("overloaded lines found")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1.05)
    percent(ax.xaxis), percent(ax.yaxis)
    ax.legend(loc="lower right")
    fig.suptitle("One Newton step from GENCO finds the overloads without wasted AC runs", x=0.08, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.08, 0.905, "GOC 500, 14,992 scenarios with up to five branches removed (the example entry's setup), released GENCO Tiny checkpoint.\n"
             "Curves: riskiest scenarios first. Dots: the plain rule, send a scenario to AC if any predicted loading exceeds the limit.", color=MUTED, fontsize=9, va="top")
    fig.subplots_adjust(top=0.8)
    save(fig, "onestep.png")


def newton():
    """Newton iterations to convergence by start point, and loading error after 0, 1, 2 steps."""
    grids = {"IEEE 118": WORK / "warmstart_case118", "Texas 2000": WORK / "texas/out/warmstart_texas"}
    starts = {"flat": "flat start", "dc": "DC angles", "prev": "pre-outage\nsolution", "genco": "GENCO"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (grid, stem) in zip(axes, grids.items()):
        states = pd.read_parquet(f"{stem}.parquet")
        states = states[states.set == "normal"]
        means = [states[f"it_{k}"].mean() for k in starts]
        bars = ax.bar(range(len(starts)), means, width=0.5, color=[DC, DC, DC, GENCO])
        for bar, value in zip(bars, means):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=9)
        ax.set_title(f"{grid} · {len(states)} states (N-0 to N-3)")
        ax.set_xticks(range(len(starts)), list(starts.values()))
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 5.2)
    axes[0].set_ylabel("Newton iterations to converge (mean)")
    fig.suptitle("Started from GENCO, Newton needs about half the iterations", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.9, "Plain full Newton, tolerance 1e-6 MW; every start reaches the same solution. Iterations only: GENCO inference itself is not free.", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.8, wspace=0.08)
    save(fig, "newton.png")


def voltage():
    """Voltage magnitudes: GENCO against AC power flow at PQ buses. DC power flow assumes 1.0 everywhere."""
    grids = {"IEEE 118": WORK / "voltage_case118.parquet", "Texas 2000": WORK / "texas/out/voltage_texas.parquet"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (grid, path) in zip(axes, grids.items()):
        data = pd.read_parquet(path, columns=["type", "vm_ac", "vm_genco", "vmin", "vmax"])
        data = data[data.type == "PQ"]
        low, high = data.vmin.iloc[0], data.vmax.iloc[0]
        outside = data[(data.vm_ac < low) | (data.vm_ac > high)]
        sample = pd.concat([data.sample(min(len(data), 40000), random_state=0), outside])
        span = [min(sample.vm_ac.min(), low) - 0.01, max(sample.vm_ac.max(), high) + 0.01]
        ax.plot(span, span, color=MUTED, linewidth=1, zorder=1)
        ax.plot(span, [1, 1], color=DC, linewidth=2, label="DC power flow (assumes 1.0)")
        ax.scatter(sample.vm_ac, sample.vm_genco, s=4, color=GENCO, alpha=0.3, linewidths=0, rasterized=True, label="GENCO")
        for limit in (low, high):
            ax.axvline(limit, color=INK, linewidth=0.8)
        ax.set_title(f"{grid} · limits {low:.2f}–{high:.2f} p.u.")
        ax.set_xlim(span), ax.set_ylim(span)
        ax.set_xlabel("true voltage magnitude (AC power flow, p.u.)")
    axes[0].set_ylabel("predicted voltage magnitude (p.u.)")
    legend = axes[1].legend(loc="lower right", markerscale=3)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    fig.suptitle("Voltage: the part of the answer DC power flow does not have", x=0.07, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.07, 0.91, "Load buses in all outage states (40,000 sampled per grid, plus every reading outside the limits). Vertical lines: voltage limits.", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.82, wspace=0.16)
    save(fig, "voltage.png")

def stress():
    """Does the promised miss rate survive a shift between calibration and use? IEEE 118, widths calibrated once."""
    fixed = pd.read_csv(WORK / "shift_case118_fixed.csv")
    fixed["shift"] = fixed["shift"].str.replace(r"^redispatch.*", "redispatch", regex=True)
    shifts = {"control": "same distribution (control)", "load": "unseen load levels", "pf Qd x1.2": "reactive demand ×1.2", "pf Qd x1.5": "reactive demand ×1.5",
              "pf Qd x0.7": "reactive demand ×0.7", "redispatch": "generation moved off the OPF dispatch", "n3": "three lines out (N-3)", "n4": "four lines out (N-4)"}
    series = {("genco_d", "shift"): ("GENCO, anchored", GENCO), ("genco", "shift"): ("GENCO, not anchored", GENCO1), ("dc_d", "shift"): ("DC, tuned", DC1)}
    alpha = 0.05
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.axvline(alpha, color=INK, linewidth=0.8)
    ax.annotate("promised: 5%", (alpha, -0.75), xytext=(5, 0), textcoords="offset points", color=MUTED, fontsize=9, va="center")
    names = [n for n in shifts if n in set(fixed["shift"])]
    for i, ((surrogate, sigma), (label, colour)) in enumerate(series.items()):
        rows = fixed[(fixed.surrogate == surrogate) & (fixed.sigma == sigma) & (fixed.alpha == alpha)].set_index("shift").reindex(names)
        y = np.arange(len(names)) + (i - 1) * 0.22
        ax.errorbar(rows.line_missed_share, y, xerr=[rows.line_missed_share - rows.ci_lo, rows.ci_hi - rows.line_missed_share], fmt="o", color=colour,
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, elinewidth=2, label=label)
    ax.set_yticks(range(len(names)), [shifts[n] for n in names])
    ax.set_ylim(len(names) - 0.4, -1.1)
    ax.set_xlim(0, 0.14)
    ax.set_xlabel("real overloads not flagged")
    ax.grid(axis="y", visible=False)
    percent(ax.xaxis)
    ax.legend(loc="upper right")
    fig.suptitle("Where the guarantee holds, and where it breaks", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "IEEE 118. Band widths calibrated once on N-1/N-2 outages at five load levels, then applied unchanged to shifted states.\n"
             "Dot: miss rate; bar: 90% interval over outage sets. Tuned DC stays below the line by sending far more secure states to AC.", color=MUTED, fontsize=9, va="top")
    fig.subplots_adjust(top=0.8, left=0.36)
    save(fig, "stress.png")

def workflow():
    """The three-stage workflow, with what happens to 1,000 scenarios of the GOC 500 set."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    lines = pd.read_parquet(WORK / "goc500/lines_newton_tiny_seed0.parquet", columns=["state", "ac", "genco1"])
    n = lines.state.nunique()
    real = lines.ac > OVERLOAD
    sent = lines.state[lines.genco1 > OVERLOAD].unique()
    to_ac = round(1000 * len(sent) / n)
    wasted = round(1000 * len(np.setdiff1d(sent, lines.state[real].unique())) / n)
    missed = (real & ~lines.state.isin(sent)).sum()

    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(0, 10), ax.set_ylim(0, 3.9)
    ax.axis("off")
    boxes = [
        (0.2, "1  GENCO", ["predicts voltages", "and line loadings"], GENCO),
        (3.6, "2  One Newton step", ["on the real equations, from", "GENCO's voltages; remaining", "mismatch is a self-check"], GENCO1),
        (7.0, "3  Full AC power flow", ["only if a line is at its limit", "or the self-check fails"], INK),
    ]
    for x, title, body, colour in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.5), 2.7, 1.6, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=SURFACE, edgecolor=colour, linewidth=2))
        ax.text(x + 0.15, 2.8, title, fontsize=11, fontweight="bold", color=INK, va="center")
        for i, line in enumerate(body):
            ax.text(x + 0.15, 2.4 - 0.27 * i, line, fontsize=9, color=MUTED, va="center")
    for x0, x1, label in ((2.95, 3.55, "1,000"), (6.35, 6.95, f"{to_ac}")):
        ax.add_patch(FancyArrowPatch((x0, 2.3), (x1, 2.3), arrowstyle="-|>", mutation_scale=14, color=INK, linewidth=1.5))
        ax.text((x0 + x1) / 2, 2.52, label, ha="center", fontsize=9, color=INK)
    ax.add_patch(FancyArrowPatch((4.95, 1.48), (4.95, 0.75), arrowstyle="-|>", mutation_scale=14, color=INK, linewidth=1.5))
    ax.text(5.1, 1.1, f"{1000 - to_ac}", fontsize=9, color=INK, va="center")
    ax.text(4.95, 0.5, "accepted as secure, no AC solve", ha="center", fontsize=10, color=INK)
    ax.text(8.35, 1.3, f"{to_ac - wasted} really have an overload,\n{wasted} turn out to be secure", ha="center", fontsize=9, color=MUTED, va="top", linespacing=1.4)
    fig.suptitle("Three stages, the expensive one last", x=0.03, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.03, 0.885, f"Per 1,000 scenarios of the GOC 500 set (up to five branches removed, released GENCO Tiny). "
             f"Of {int(real.sum()):,} overloaded lines in all {n:,} scenarios, {missed} sit in an accepted scenario.", color=MUTED, fontsize=9)
    save(fig, "workflow.png")

def scale():
    """The plain rule on three grid sizes: secure scenarios sent to AC, and overloaded lines missed."""
    grids = {"GOC 500": WORK / "goc500/lines_newton_tiny_seed0.parquet", "GOC 2000": WORK / "goc2000/lines_newton_tiny_seed0.parquet",
             "GOC 10000": WORK / "goc10000/lines_newton_tiny_seed0.parquet"}
    series = {"genco": ("GENCO", GENCO), "genco1": ("+1 step", GENCO1), "genco2": ("+2 steps", "#4a3aa7"), "dc": ("DC", DC)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    for ax, (grid, path) in zip(axes, grids.items()):
        lines = pd.read_parquet(path, columns=["state", "ac", *series])
        n, real = lines.state.nunique(), lines.ac > OVERLOAD
        critical = set(lines.state[real])
        for i, (column, (label, colour)) in enumerate(series.items()):
            sent = set(lines.state[(lines[column] > OVERLOAD) | lines[column].isna()])
            wasted = len(sent - critical) / (n - len(critical))
            missed = int((real & ~lines.state.isin(sent)).sum())
            ax.bar(i, wasted, width=0.6, color=colour)
            ax.annotate(f"{wasted:.1%}", (i, wasted), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=9)
            ax.text(i, -0.17, f"{missed} missed", ha="center", va="top", color=INK, fontsize=8.5)
        ax.set_title(grid, pad=22)
        ax.text(0, 1.02, f"{n:,} scenarios · {int(real.sum()):,} overloaded lines", transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
        ax.set_xticks(range(len(series)), [label for label, _ in series.values()], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="x", visible=False)
        percent(ax.yaxis)
    axes[0].set_ylabel("secure scenarios sent to AC anyway")
    fig.suptitle("From 500 to 10,000 buses: one Newton step, then two", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.915, "Up to five branches removed per scenario, released GENCO Tiny checkpoints. Rule: send a scenario to AC if any predicted loading exceeds the limit.\n"
             "Steps are Newton steps started from GENCO. Under each bar: overloaded lines left in scenarios that were not sent to AC.", color=MUTED, fontsize=9, va="top")
    fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.08)
    save(fig, "scale.png")

def starts():
    """One Newton step from three starts: overloaded lines missed, on four grids."""
    def frame(grid):
        if grid == "IEEE 118":
            d = pd.read_parquet(WORK / "zeroshot/lines_118.parquet", columns=["state", "ac", "prev1", "pg1", "hybpg1"])
            return d.rename(columns={"pg1": "genco1", "hybpg1": "hyb1"})
        folder = WORK / {"GOC 500": "goc500", "GOC 2000": "goc2000", "GOC 10000": "goc10000"}[grid]
        d = pd.read_parquet(folder / "lines_newton_tiny_seed0.parquet", columns=["state", "line", "ac", "genco1"])
        for name, column in (("prev", "prev1"), ("hybrid", "hyb1")):
            d = d.merge(pd.read_parquet(folder / f"lines_{name}_tiny_seed0.parquet", columns=["state", "line", column]), on=["state", "line"])
        return d
    series = {"prev1": ("pre-outage AC solution", DC), "genco1": ("GENCO", GENCO), "hyb1": ("pre-outage solution + GENCO's change", GENCO1)}
    grids = ["IEEE 118", "GOC 500", "GOC 2000", "GOC 10000"]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.4))
    for ax, grid in zip(axes, grids):
        d = frame(grid)
        n, real = d.state.nunique(), d.ac > OVERLOAD
        critical = set(d.state[real])
        shares = []
        for i, (column, (label, colour)) in enumerate(series.items()):
            sent = set(d.state[(d[column] > OVERLOAD) | d[column].isna()])
            missed = int((real & ~d.state.isin(sent)).sum())
            wasted = len(sent - critical) / max(n - len(critical), 1)
            share = missed / real.sum()
            shares.append(share)
            ax.bar(i, share, width=0.6, color=colour, label=label)
            ax.annotate(f"{missed}", (i, share), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=10)
            ax.text(i, -0.14, f"{wasted:.1%}", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK, fontsize=9)
        ax.set_title(grid, pad=22)
        ax.text(0, 1.02, f"{int(real.sum()):,} overloaded lines", transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
        ax.set_xticks([])
        ax.set_ylim(0, max(shares) * 1.25)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0].set_ylabel("overloaded lines missed")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), title="Newton step started from")
    fig.suptitle("Which start makes one Newton step enough?", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.915, "Up to five branches removed per scenario. A scenario goes to the full solver if any loading after the step exceeds the limit.\n"
             "Above each bar: lines missed. Below: secure scenarios sent to the full solver anyway. Each panel has its own scale.", color=MUTED, fontsize=9, va="top")
    fig.subplots_adjust(top=0.70, bottom=0.26, wspace=0.36, left=0.09)
    save(fig, "starts.png")


def finetune():
    """Fine-tuning the released checkpoint: lines missed after one Newton step, before and after, on three grids."""
    results = WORK / "finetune/results"
    grids = {"GOC 500": ("500", "ft_seed0"), "GOC 2000": ("2000", "ft2000_seed0"), "GOC 10000": ("10000", "ft10000_seed0")}
    grids = {g: v for g, v in grids.items() if (results / f"lines_hybrid_{v[0]}_{v[1]}.parquet").exists()}
    bars = (("prev1", "released", "pre-outage AC solution", DC, None), ("genco1", "released", "GENCO, released", GENCO, None),
            ("genco1", "tuned", "GENCO, fine-tuned", GENCO, "///"), ("hyb1", "released", "pre-outage + GENCO's change, released", GENCO1, None),
            ("hyb1", "tuned", "pre-outage + GENCO's change, fine-tuned", GENCO1, "///"))
    positions = [0, 1.3, 2.1, 3.4, 4.2]
    fig, axes = plt.subplots(1, len(grids), figsize=(4 * len(grids), 4.4), squeeze=False)
    for ax, (grid, (size, tag)) in zip(axes[0], grids.items()):
        columns = ["state", "ac", "genco", "prev1", "genco1", "hyb1"]
        frames = {"released": pd.read_parquet(results / f"lines_hybrid_{size}_released.parquet", columns=columns),
                  "tuned": pd.read_parquet(results / f"lines_hybrid_{size}_{tag}.parquet", columns=columns)}
        shares = []
        for x, (column, model, label, colour, hatch) in zip(positions, bars):
            d = frames[model]
            real = d.ac > OVERLOAD
            critical = set(d.state[real])
            sent = set(d.state[(d[column] > OVERLOAD) | d[column].isna()])
            missed = int((real & ~d.state.isin(sent)).sum())
            wasted = len(sent - critical) / (d.state.nunique() - len(critical))
            shares.append(missed / real.sum())
            ax.bar(x, shares[-1], width=0.7, color=colour, hatch=hatch, edgecolor=SURFACE, linewidth=0, label=label)
            ax.annotate(f"{missed}", (x, shares[-1]), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=10)
            ax.text(x, -0.03, f"{wasted:.1%}", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK, fontsize=8.5)
        errors = [np.median((d.genco - d.ac).abs()) * 100 for d in frames.values()]
        ax.set_title(grid, pad=22)
        ax.text(0, 1.02, f"{int(real.sum()):,} overloaded lines; raw error {errors[0]:.2f} → {errors[1]:.2f} pt", transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
        ax.set_xticks([])
        ax.set_ylim(0, max(max(shares), 1e-3) * 1.25)
        ax.grid(axis="x", visible=False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    axes[0][0].set_ylabel("overloaded lines missed")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), title="Newton step started from")
    fig.suptitle("Does fine-tuning make GENCO's predicted change more useful?", x=0.06, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.06, 0.915, "Released GENCO Tiny against the same checkpoint fine-tuned on new multi-outage scenarios of the same grid (hatched). Same held-out test scenarios.\n"
             "Above each bar: lines missed. Below: secure scenarios sent to the full solver anyway. 'Raw error' is GENCO's median loading error before any Newton step.", color=MUTED, fontsize=9, va="top")
    fig.subplots_adjust(top=0.70, bottom=0.24, wspace=0.3, left=0.09)
    save(fig, "finetune.png")


def fulln1():
    """Complete N-1 analysis on GOC 2000: what each workflow costs and what it misses."""
    summary = pd.read_csv(WORK / "fulln1/summary.csv")
    rows = summary[(summary.kind == "workflow") & (summary.bases == "test")].set_index("item")
    labels = {"W0": "Full AC for every outage", "W1": "One Newton step from the\npre-outage solution", "W3": "One step from GENCO",
              "W4": "One step from pre-outage\nsolution + GENCO's change", "W2": "GENCO alone", "W5": "DC sensitivities alone"}
    colours = {"W0": DC, "W1": DC, "W3": GENCO, "W4": GENCO1, "W2": GENCO, "W5": DC1}
    iteration_column = [c for c in rows.columns if "iter" in c and ("share" in c or "rel" in c or "vs" in c)]
    time_column = [c for c in rows.columns if "mps" in c.lower()]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    order = list(labels)
    panels = (("Newton iterations vs full AC", iteration_column[0], "{:.0%}", True),
              ("Critical outages missed", "missed_share_of_critical", "{:.1%}", True),
              ("Estimated seconds per 1,000 outages", time_column[0], "{:.1f}", False))
    for ax, (title, column, fmt, is_share) in zip(axes, panels):
        values = rows.loc[order, column].astype(float)
        ax.barh(range(len(order)), values, height=0.55, color=[colours[w] for w in order])
        for y, value in enumerate(values):
            ax.annotate(fmt.format(value), (value, y), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, values.max() * 1.22)
        ax.grid(axis="y", visible=False)
        if is_share:
            percent(ax.xaxis)
    axes[0].set_yticks(range(len(order)), [labels[w] for w in order], fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("A complete N-1 analysis: 57,384 outages on GOC 2000", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.9, "18 base cases, every non-islanding branch outage; 2.5% of outages are critical. Time includes GENCO inference (batched on the laptop GPU), unit costs measured on this laptop.",
             color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.78, left=0.2, wspace=0.12)
    save(fig, "fulln1.png")


if __name__ == "__main__":
    for figure in (starts, fulln1, finetune, workflow, onestep, scale, newton, voltage, stress, budget, coverage, cost, scatter, shift):
        figure()
