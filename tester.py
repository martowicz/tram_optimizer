import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
import pandas as pd
from data_generator import generate_dummy_data
from data_loader import load_data_as_matrices
from timetable_phase import generate_timetable
from annealing_helpers import build_problem_data, run_schedule_optimization

NUM_HOURS = 24
HOURS = list(range(NUM_HOURS))
NUM_RUNS = 5

# Get timetable df for given parameters
def get_result(C, R, alpha, beta, P, W):
    data = build_problem_data(P, C, R, alpha, beta)

    initial_temp = 10.0
    min_temp = 1e-3
    cooling_rate = 0.999
    max_iter = 1000
    log_every = 10

    results = run_schedule_optimization(data, initial_temp, min_temp, cooling_rate, max_iter, log_every)

    return generate_timetable(results, W)


def get_result_averaged(C, R, alpha, beta, P, W, n_runs=NUM_RUNS):
    """Run the optimisation n_runs times and return the mean trips-per-hour array."""
    runs = []
    for _ in range(n_runs):
        tdf = get_result(C, R, alpha, beta, P, W)
        runs.append(get_trips_per_hour(tdf))
    return np.mean(runs, axis=0)   # shape (24,)


# Save chart for given timetable
def save_chart(dir, timetable_df, C, R, alpha, beta):
    file_name = f"C{C}_R{R}_alpha{alpha}_beta{beta}.png"
    file_path = dir + file_name
    title = f"C = {C}, R = {R}, alpha = {alpha}, beta = {beta}"

    df = timetable_df.groupby("TripID").max().groupby("HourWindow").count()
    df = df.rename(columns={"Stop": "trips"})

    fig, ax = plt.subplots()
    df.plot.bar(y="trips", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("hour")
    ax.set_ylabel("trips in hour")
    plt.savefig(file_path)
    plt.close(fig)


def get_trips_per_hour(timetable_df):
    """Return a length-24 numpy array of trip counts, zero-filling missing hours."""
    series = (
        timetable_df.groupby("TripID")["HourWindow"]
        .first()
        .value_counts()
        .reindex(HOURS, fill_value=0)
        .sort_index()
    )
    return series.values.astype(float)


def make_comparisons(dir, results, cs, rs, alphas, betas):
    param_ranges = {"C": cs, "R": rs, "alpha": alphas, "beta": betas}
    baseline = {"C": cs[0], "R": rs[0], "alpha": alphas[0], "beta": betas[0]}

    def make_key(C, R, alpha, beta):
        return f"{C}_{R}_{alpha}_{beta}"

    def lookup(param_name, val):
        kwargs = {**baseline, param_name: val}
        return results.get(make_key(**kwargs))

    param_data = {}
    for param_name, param_values in param_ranges.items():
        entries = []
        for val in param_values:
            trips = lookup(param_name, val)   # already a numpy array now
            if trips is not None:
                entries.append((val, trips))
        param_data[param_name] = entries

    _plot_per_param_summary(dir, param_data)
    _plot_hourly_heatmaps(dir, param_data)
    _plot_violin(dir, param_data)
    _plot_parallel_coordinates(dir, param_data)
    _plot_correlation_heatmap(dir, results, cs, rs, alphas, betas)


# ── Plot 1: per-parameter summary (mean ± std, avg-min, avg-max) ──────────────
def _plot_per_param_summary(dir, param_data):
    for param_name, entries in param_data.items():
        valid_values, avg_means, avg_stds, avg_mins, avg_maxs = [], [], [], [], []
        for val, trips in entries:
            valid_values.append(val)
            avg_means.append(np.mean(trips))
            avg_stds.append(np.std(trips))
            avg_mins.append(np.min(trips))
            avg_maxs.append(np.max(trips))

        x = np.arange(len(valid_values))
        x_labels = [str(v) for v in valid_values]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.errorbar(x, avg_means, yerr=avg_stds, fmt='o-', capsize=5,
                    label="mean ± std", color="steelblue")
        ax.plot(x, avg_mins, 's--', color="seagreen", label="avg min")
        ax.plot(x, avg_maxs, 'D--', color="tomato",   label="avg max")
        ax.fill_between(x,
                        np.array(avg_means) - np.array(avg_stds),
                        np.array(avg_means) + np.array(avg_stds),
                        alpha=0.15, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(param_name)
        ax.set_ylabel("trips per hour")
        ax.set_title(f"Effect of {param_name} on trips-per-hour distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{dir}compare_{param_name}.png")
        plt.close(fig)


# ── Plot 3: hourly heatmap grids ──────────────────────────────────────────────
def _plot_hourly_heatmaps(dir, param_data):
    for param_name, entries in param_data.items():
        n_vals = len(entries)
        matrix = np.zeros((n_vals, NUM_HOURS))
        y_labels = []

        for i, (val, trips) in enumerate(entries):
            matrix[i] = trips
            y_labels.append(str(val))

        fig, ax = plt.subplots(figsize=(14, max(3, n_vals * 0.7 + 1)))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest",
                       extent=[-0.5, NUM_HOURS - 0.5, n_vals - 0.5, -0.5])
        plt.colorbar(im, ax=ax, label="trips in hour")
        ax.set_xticks(HOURS)
        ax.set_xticklabels(HOURS, fontsize=8)
        ax.set_yticks(range(n_vals))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("hour of day")
        ax.set_ylabel(param_name)
        ax.set_title(f"Trips per hour — effect of {param_name}")
        plt.tight_layout()
        plt.savefig(f"{dir}heatmap_{param_name}.png", dpi=120)
        plt.close(fig)


# ── Plot 4: violin plots ──────────────────────────────────────────────────────
def _plot_violin(dir, param_data):
    for param_name, entries in param_data.items():
        data_arrays = [trips for _, trips in entries]
        x_labels    = [str(val) for val, _ in entries]

        fig, ax = plt.subplots(figsize=(10, 5))
        parts = ax.violinplot(data_arrays, positions=range(len(entries)),
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)

        means = [np.mean(t) for t in data_arrays]
        ax.scatter(range(len(entries)), means, color="tomato", zorder=3,
                   label="mean", s=40)

        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels(x_labels)
        ax.set_xlabel(param_name)
        ax.set_ylabel("trips per hour")
        ax.set_title(f"Distribution of trips per hour — effect of {param_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{dir}violin_{param_name}.png", dpi=120)
        plt.close(fig)


# ── Plot 5: parallel coordinates ─────────────────────────────────────────────
def _plot_parallel_coordinates(dir, param_data):
    axes_names = ["C", "R", "alpha", "beta", "avg", "std", "min", "max"]
    colors = {"C": "steelblue", "R": "seagreen", "alpha": "tomato", "beta": "goldenrod"}

    baseline = {p: entries[0][0] for p, entries in param_data.items()}

    rows = []
    for param_name, entries in param_data.items():
        for val, trips in entries:
            pvals = {**baseline, param_name: val}
            row = [
                pvals["C"], pvals["R"], pvals["alpha"], pvals["beta"],
                np.mean(trips), np.std(trips), np.min(trips), np.max(trips),
            ]
            rows.append((param_name, row))

    rows_array = np.array([r for _, r in rows])
    col_min = rows_array.min(axis=0)
    col_max = rows_array.max(axis=0)
    col_range = np.where(col_max - col_min == 0, 1, col_max - col_min)
    rows_norm = (rows_array - col_min) / col_range

    n_axes = len(axes_names)
    x_positions = np.arange(n_axes)

    fig, ax = plt.subplots(figsize=(13, 6))
    plotted_labels = set()
    for i, (param_name, _) in enumerate(rows):
        y = rows_norm[i]
        label = param_name if param_name not in plotted_labels else "_nolegend_"
        plotted_labels.add(param_name)
        ax.plot(x_positions, y, color=colors[param_name],
                alpha=0.55, linewidth=1.4, label=label)

    for j, axis_name in enumerate(axes_names):
        ax.axvline(j, color="black", linewidth=0.8, alpha=0.4)
        ax.text(j, -0.08, f"{col_min[j]:.2g}", ha="center", va="top",
                fontsize=7, transform=ax.get_xaxis_transform())
        ax.text(j,  1.04, f"{col_max[j]:.2g}", ha="center", va="bottom",
                fontsize=7, transform=ax.get_xaxis_transform())

    ax.set_xticks(x_positions)
    ax.set_xticklabels(axes_names)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([])
    ax.set_title("Parallel coordinates — parameter sweeps")
    ax.legend(title="varied param", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{dir}parallel_coordinates.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Plot 6: correlation heatmap ───────────────────────────────────────────────
def _plot_correlation_heatmap(dir, results, cs, rs, alphas, betas):
    """
    4 rows (parameters) × 4 cols (output metrics): Pearson r between each
    parameter and each output metric. Parameter-to-parameter cells are omitted.
    """
    rows = []
    for C in cs:
        for R in rs:
            for alpha in alphas:
                for beta in betas:
                    key = f"{C}_{R}_{alpha}_{beta}"
                    trips = results.get(key)
                    if trips is None:
                        continue
                    rows.append({
                        "C": C, "R": R, "alpha": alpha, "beta": beta,
                        "avg":     np.mean(trips),
                        "std":     np.std(trips),
                        "avg_min": np.min(trips),
                        "avg_max": np.max(trips),
                    })

    if not rows:
        return

    df = pd.DataFrame(rows)
    params  = ["C", "R", "alpha", "beta"]
    metrics = ["avg", "std", "avg_min", "avg_max"]

    # Build the 4×4 cross-correlation matrix (params vs metrics only)
    mat = np.array([
        [df[p].corr(df[m]) for m in metrics]
        for p in params
    ])

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(params)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(params)

    for i in range(len(params)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if abs(mat[i, j]) < 0.7 else "white")

    ax.set_title("Parameter → output metric correlations (Pearson r)")
    plt.tight_layout()
    plt.savefig(f"{dir}correlation_matrix.png", dpi=120)
    plt.close(fig)


# Run tests for different parameters
def test_and_plot(cs, rs, alphas, betas, P, W):
    total = len(cs) * len(rs) * len(alphas) * len(betas)
    cnt = 0
    results = dict()
    for C in cs:
        for R in rs:
            for alpha in alphas:
                for beta in betas:
                    trips = get_result_averaged(C, R, alpha, beta, P, W)
                    results[f"{C}_{R}_{alpha}_{beta}"] = trips
                    cnt += 1
                    percent = cnt / total * 100
                    bar_len = 30
                    filled = int(bar_len * cnt / total)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    print(f"\r[{bar}] {percent:6.2f}% ({cnt}/{total})", end="", flush=True)
    print()
    dir = "test_plots/"
    os.makedirs(dir, exist_ok=True)
    make_comparisons(dir, results, cs, rs, alphas, betas)


def test_and_chart(cs, rs, alphas, betas, P, W):
    dir = "test_charts/"
    os.makedirs(dir, exist_ok=True)
    total = len(cs) * len(rs) * len(alphas) * len(betas)
    cnt = 0
    for C in cs:
        for R in rs:
            for alpha in alphas:
                for beta in betas:
                    trips = get_result(C, R, alpha, beta, P, W)
                    save_chart(dir, trips, C, R, alpha, beta)
                    cnt += 1
                    percent = cnt / total * 100
                    bar_len = 30
                    filled = int(bar_len * cnt / total)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    print(f"\r[{bar}] {percent:6.2f}% ({cnt}/{total})", end="", flush=True)
    print()

if __name__ == "__main__":
    num_stops = 5
    num_hours = 24
    generate_dummy_data(num_stops, num_hours)
    P, W = load_data_as_matrices("data/input/travel_times.csv", "data/input/demand.csv", num_stops, num_hours)

    cs = [30, 40, 50, 75, 100]
    rs = [15, 30, 45, 60, 90]
    alphas = [0.1, 1.0, 5.0, 10.0, 20.0]
    betas = [5.0, 10.0, 50.0, 75.0, 100.0]
    test_and_plot(cs, rs, alphas, betas, P, W)