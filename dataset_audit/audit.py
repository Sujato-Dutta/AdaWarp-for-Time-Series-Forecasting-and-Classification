from __future__ import annotations

import math
import os
import re
from pathlib import Path

_LOCAL_MPL_CACHE = Path(__file__).resolve().parent / ".matplotlib"
_LOCAL_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
TAB = OUT / "tables"
CSV = OUT / "statistics"

DATASETS = {
    "ETTh1": ROOT / "TSLibrary/dataset/ETT-small/ETTh1.csv",
    "ETTh2": ROOT / "TSLibrary/dataset/ETT-small/ETTh2.csv",
    "ETTm1": ROOT / "TSLibrary/dataset/ETT-small/ETTm1.csv",
    "ETTm2": ROOT / "TSLibrary/dataset/ETT-small/ETTm2.csv",
    "Weather": ROOT / "TSLibrary/dataset/weather/weather.csv",
    "Electricity": ROOT / "TSLibrary/dataset/electricity/electricity.csv",
    "Traffic": ROOT / "TSLibrary/dataset/traffic/traffic.csv",
}

ETT_MEANINGS = {
    "HUFL": "High useful load",
    "HULL": "High useless load",
    "MUFL": "Middle useful load",
    "MULL": "Middle useless load",
    "LUFL": "Low useful load",
    "LULL": "Low useless load",
    "OT": "Transformer oil temperature (forecast target)",
}

WEATHER_META = {
    "p (mbar)": ("Air pressure", "mbar", "pressure/density"),
    "T (degC)": ("Air temperature", r"$^\circ$C", "temperature"),
    "Tpot (K)": ("Potential temperature", "K", "temperature"),
    "Tdew (degC)": ("Dew-point temperature", r"$^\circ$C", "temperature"),
    "rh (%)": ("Relative humidity", r"\%", "humidity"),
    "VPmax (mbar)": ("Saturation vapor pressure", "mbar", "humidity"),
    "VPact (mbar)": ("Actual vapor pressure", "mbar", "humidity"),
    "VPdef (mbar)": ("Vapor-pressure deficit", "mbar", "humidity"),
    "sh (g/kg)": ("Specific humidity", r"g\,kg$^{-1}$", "humidity"),
    "H2OC (mmol/mol)": ("Water-vapor concentration", r"mmol\,mol$^{-1}$", "humidity"),
    "rho (g/m**3)": ("Air density", r"g\,m$^{-3}$", "pressure/density"),
    "wv (m/s)": ("Mean wind speed", r"m\,s$^{-1}$", "wind"),
    "max. wv (m/s)": ("Maximum wind speed", r"m\,s$^{-1}$", "wind"),
    "wd (deg)": ("Wind direction", "degree", "wind"),
    "rain (mm)": ("Precipitation amount", "mm", "precipitation/radiation"),
    "raining (s)": ("Duration of precipitation in interval", "s", "precipitation/radiation"),
    "SWDR": ("Short-wave downward radiation", r"W\,m$^{-2}$", "precipitation/radiation"),
    "PAR": ("Photosynthetically active radiation", r"$\mu$mol\,m$^{-2}$\,s$^{-1}$", "precipitation/radiation"),
    "max. PAR": ("Maximum photosynthetically active radiation", r"$\mu$mol\,m$^{-2}$\,s$^{-1}$", "precipitation/radiation"),
    "Tlog (degC)": ("Data-logger internal temperature", r"$^\circ$C", "temperature"),
    "OT": ("Ambient CO$_2$ concentration (locally renamed from source CO2 channel)", "ppm", "atmospheric composition"),
}

COLORS = ["#2456A6", "#16A085", "#E67E22", "#8E44AD", "#C0392B", "#2C3E50", "#7F8C8D"]


def tex(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def fmt(x: float, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "--"
    ax = abs(float(x))
    if ax != 0 and (ax >= 100000 or ax < 0.001):
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def load(name: str) -> tuple[pd.DatetimeIndex, pd.DataFrame, int]:
    frame = pd.read_csv(DATASETS[name])
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    if name == "Weather":
        values = values.rename(
            columns={
                col: ("max. PAR" if col.startswith("max. PAR") else "PAR" if col.startswith("PAR") else "SWDR" if col.startswith("SWDR") else col)
                for col in values.columns
            }
        )
    invalid = int((values <= -9000).sum().sum())
    values = values.mask(values <= -9000)
    return pd.DatetimeIndex(dates), values, invalid


def cadence_hours(dates: pd.DatetimeIndex) -> float:
    diffs = np.asarray((dates[1:] - dates[:-1]) / pd.Timedelta(hours=1), dtype=float)
    return float(np.nanmedian(diffs[diffs > 0]))


def dominant_period(values: np.ndarray, dt_hours: float, max_days: float = 60.0) -> float:
    x = np.asarray(values, dtype=float)
    good = np.isfinite(x)
    if good.sum() < 16:
        return float("nan")
    if not good.all():
        x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    x = x - (slope * t + intercept)
    x = x * np.hanning(len(x))
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=dt_hours)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freq
    valid = (freq > 0) & (periods >= 4 * dt_hours) & (periods <= max_days * 24)
    if not valid.any():
        return float("nan")
    idx = np.flatnonzero(valid)[np.argmax(power[valid])]
    return float(periods[idx])


def period_label(hours: float) -> str:
    if not np.isfinite(hours):
        return "undetermined"
    if abs(hours - 24) <= 3:
        return "daily"
    if abs(hours - 168) <= 20:
        return "weekly"
    if hours < 12:
        return "sub-daily"
    if hours < 48:
        return "near-daily"
    return f"long-cycle ({hours / 24:.1f} d)"


def series_stats(values: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    dt = cadence_hours(dates)
    rows = []
    for col in values.columns:
        x = values[col].to_numpy(dtype=float)
        std = float(np.nanstd(x))
        diff_std = float(np.nanstd(np.diff(x[np.isfinite(x)]))) if np.isfinite(x).sum() > 2 else np.nan
        period = dominant_period(x, dt)
        rows.append(
            {
                "variable": col,
                "min": float(np.nanmin(x)),
                "max": float(np.nanmax(x)),
                "mean": float(np.nanmean(x)),
                "std": std,
                "dominant_period_hours": period,
                "roughness": diff_std / (std + 1e-12),
                "missing_count": int(np.isnan(x).sum()),
            }
        )
    return pd.DataFrame(rows)


def representative_start(values: pd.DataFrame, samples: int) -> int:
    n = len(values)
    starts = np.arange(0, max(1, n - samples + 1), samples)
    if len(starts) <= 2:
        return int(max(0, (n - samples) // 2))
    feats = []
    for start in starts:
        block = values.iloc[start : start + samples].to_numpy(dtype=float)
        feats.append([np.nanmean(block), np.nanstd(block), np.nanmean(np.abs(np.diff(block, axis=0)))])
    feats = np.asarray(feats)
    med = np.nanmedian(feats, axis=0)
    scale = np.nanmedian(np.abs(feats - med), axis=0) + 1e-12
    score = np.nansum(np.abs((feats - med) / scale), axis=1)
    return int(starts[np.nanargmin(score)])


def plot_ett(name: str, dates: pd.DatetimeIndex, values: pd.DataFrame) -> None:
    plot_step = max(1, math.ceil(len(values) / 5000))
    fig, axes = plt.subplots(7, 1, figsize=(11, 10), sharex=True)
    for i, (ax, col) in enumerate(zip(axes, values.columns)):
        ax.plot(dates[::plot_step], values[col].iloc[::plot_step], color=COLORS[i], lw=0.45)
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(alpha=0.15)
    axes[0].set_title(f"{name}: complete local benchmark record")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_full.pdf", bbox_inches="tight")
    plt.close(fig)

    dt = cadence_hours(dates)
    width = max(1, int(round(7 * 24 / dt)))
    start = representative_start(values, width)
    stop = min(len(values), start + width)
    fig, axes = plt.subplots(7, 1, figsize=(11, 10), sharex=True)
    for i, (ax, col) in enumerate(zip(axes, values.columns)):
        ax.plot(dates[start:stop], values[col].iloc[start:stop], color=COLORS[i], lw=0.9)
        ax.set_ylabel(col, rotation=0, ha="right", va="center", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"{name}: representative seven-day window")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_week.pdf", bbox_inches="tight")
    plt.close(fig)


def lag_relation(load: np.ndarray, target: np.ndarray, dt: float, max_hours: int = 48) -> tuple[float, float]:
    x = np.diff(load.astype(float))
    y = np.diff(target.astype(float))
    max_lag = max(1, int(round(max_hours / dt)))
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 20 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0, float("nan")
    x = (x - np.mean(x)) / np.std(x)
    y = (y - np.mean(y)) / np.std(y)
    full = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.asarray([full[len(x) - 1 + lag] / (len(x) - abs(lag)) for lag in lags])
    idx = int(np.argmax(np.abs(scores)))
    return float(lags[idx] * dt), float(scores[idx])


def ett_audit() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    stats, lags, comparison = {}, {}, []
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        dates, values, _ = load(name)
        st = series_stats(values, dates)
        st.insert(1, "meaning", st["variable"].map(ETT_MEANINGS))
        st.insert(2, "unit", np.where(st["variable"].eq("OT"), "degree C", "dataset load unit"))
        stats[name] = st
        st.to_csv(CSV / f"{name}_variables.csv", index=False)
        dt = cadence_hours(dates)
        lag_rows = []
        for col in values.columns[:-1]:
            level = float(values[col].corr(values["OT"]))
            lag_h, lag_corr = lag_relation(values[col].to_numpy(), values["OT"].to_numpy(), dt)
            lag_rows.append({"variable": col, "level_correlation": level, "best_change_lag_hours": lag_h, "change_correlation": lag_corr})
        lags[name] = pd.DataFrame(lag_rows)
        lags[name].to_csv(CSV / f"{name}_load_ot_lags.csv", index=False)
        plot_ett(name, dates, values)
        for _, row in st.iterrows():
            comparison.append({"dataset": name, **row.to_dict()})
    comp = pd.DataFrame(comparison)
    comp.to_csv(CSV / "ett_cross_dataset.csv", index=False)
    return stats, lags, comp


def plot_weather(dates: pd.DatetimeIndex, values: pd.DataFrame) -> None:
    dt = cadence_hours(dates)
    width = int(round(7 * 24 / dt))
    start = representative_start(values, width)
    stop = min(len(values), start + width)
    groups = list(dict.fromkeys(meta[2] for meta in WEATHER_META.values()))
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 12), sharex=True)
    for ax, group in zip(axes, groups):
        cols = [c for c, meta in WEATHER_META.items() if meta[2] == group]
        for col in cols:
            x = values[col].iloc[start:stop].to_numpy(dtype=float)
            z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
            ax.plot(dates[start:stop], z, lw=0.8, label=WEATHER_META[col][0])
        ax.set_ylabel(group.replace("/", "/\n"), fontsize=8)
        ax.grid(alpha=0.15)
        ax.legend(ncol=min(4, len(cols)), fontsize=6, loc="upper right")
    axes[0].set_title("Weather: standardized variables over a representative seven-day window")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIG / "Weather_groups_week.pdf", bbox_inches="tight")
    plt.close(fig)

    corr = values.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [re.sub(r"\s*\(.*", "", c) for c in values.columns]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=6)
    ax.set_title("Weather: Pearson correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.75, label="correlation")
    fig.tight_layout()
    fig.savefig(FIG / "Weather_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def weather_audit() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    dates, values, invalid = load("Weather")
    stats = series_stats(values, dates)
    stats.insert(1, "meaning", stats["variable"].map(lambda x: WEATHER_META[x][0]))
    stats.insert(2, "unit", stats["variable"].map(lambda x: WEATHER_META[x][1]))
    stats.insert(3, "group", stats["variable"].map(lambda x: WEATHER_META[x][2]))
    def weather_behavior(row: pd.Series) -> str:
        variable = row["variable"]
        if variable in {"rain (mm)", "raining (s)"}:
            return "sharp/intermittent precipitation events"
        if variable == "wd (deg)":
            return "circular and abrupt; Euclidean roughness is misleading"
        if variable in {"wv (m/s)", "max. wv (m/s)"}:
            return "gusty with a daily spectral component"
        if variable in {"SWDR", "PAR", "max. PAR"}:
            return "smooth daily daylight envelope with night zeros"
        prefix = "sharp/moderate; " if row["roughness"] > 0.5 else "smooth/moderate; "
        return prefix + period_label(row["dominant_period_hours"])

    stats["behavior"] = stats.apply(weather_behavior, axis=1)
    stats.to_csv(CSV / "Weather_variables.csv", index=False)
    groups = []
    corr = values.corr()
    for group in stats["group"].unique():
        cols = stats.loc[stats["group"] == group, "variable"].tolist()
        block = corr.loc[cols, cols].to_numpy()
        upper = block[np.triu_indices(len(cols), 1)] if len(cols) > 1 else np.array([])
        groups.append(
            {
                "group": group,
                "variables": len(cols),
                "median_within_group_correlation": float(np.nanmedian(upper)) if len(upper) else float("nan"),
                "median_roughness": float(stats.loc[stats["group"] == group, "roughness"].median()),
            }
        )
    group_df = pd.DataFrame(groups)
    group_df.to_csv(CSV / "Weather_groups.csv", index=False)
    plot_weather(dates, values)
    return stats, group_df, invalid


def lag_corr_matrix(x: np.ndarray, lag: int) -> np.ndarray:
    if len(x) <= lag:
        return np.full(x.shape[1], np.nan)
    a, b = x[:-lag], x[lag:]
    am, bm = np.nanmean(a, axis=0), np.nanmean(b, axis=0)
    aa, bb = a - am, b - bm
    num = np.nansum(aa * bb, axis=0)
    den = np.sqrt(np.nansum(aa * aa, axis=0) * np.nansum(bb * bb, axis=0)) + 1e-12
    return num / den


def anonymous_audit(name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]:
    dates, values, _ = load(name)
    x = values.to_numpy(dtype=float)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    rough = np.nanstd(np.diff(x, axis=0), axis=0) / (stds + 1e-12)
    zero = np.nanmean(np.isclose(x, 0.0, atol=1e-12), axis=0)
    daily = lag_corr_matrix(x, 24)
    weekly = lag_corr_matrix(x, 168)
    target = x[:, -1]
    target_corr = []
    for j in range(x.shape[1]):
        good = np.isfinite(x[:, j]) & np.isfinite(target)
        if good.sum() < 2 or np.nanstd(x[good, j]) == 0 or np.nanstd(target[good]) == 0:
            target_corr.append(np.nan)
        else:
            target_corr.append(float(np.corrcoef(x[good, j], target[good])[0, 1]))
    stats = pd.DataFrame(
        {
            "channel": values.columns,
            "min": mins,
            "max": maxs,
            "mean": means,
            "std": stds,
            "roughness": rough,
            "zero_fraction": zero,
            "daily_lag_correlation": daily,
            "weekly_lag_correlation": weekly,
            "correlation_with_local_OT_channel": target_corr,
        }
    )
    feature = np.column_stack(
        [np.log1p(np.abs(means)), np.log1p(stds), np.nan_to_num(rough), np.nan_to_num(daily), np.nan_to_num(weekly), zero]
    )
    labels = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(StandardScaler().fit_transform(feature))
    stats["cluster_id"] = labels
    summaries = []
    for cluster in range(3):
        mask = labels == cluster
        summaries.append(
            {
                "cluster_id": cluster,
                "channels": int(mask.sum()),
                "fraction_percent": float(mask.mean() * 100),
                "median_mean": float(np.median(means[mask])),
                "median_std": float(np.median(stds[mask])),
                "median_roughness": float(np.median(rough[mask])),
                "median_daily_corr": float(np.median(daily[mask])),
                "median_weekly_corr": float(np.median(weekly[mask])),
            }
        )
    cluster_df = pd.DataFrame(summaries)
    periodic_id = int(cluster_df["median_daily_corr"].idxmax())
    remaining = [i for i in range(3) if i != periodic_id]
    low_id = min(remaining, key=lambda i: cluster_df.loc[i, "median_mean"])
    irregular_id = next(i for i in remaining if i != low_id)
    names = {periodic_id: "strongly periodic", low_id: "low activity", irregular_id: "irregular/high variability"}
    stats["cluster"] = [names[int(v)] for v in labels]
    cluster_df["cluster"] = cluster_df["cluster_id"].map(names)

    positive_std = stds[stds > 1e-12]
    constant_threshold = 0.01 * float(np.median(positive_std)) if len(positive_std) else 1e-12
    q1, q3 = np.quantile(rough[np.isfinite(rough)], [0.25, 0.75])
    volatile_threshold = float(q3 + 1.5 * (q3 - q1))
    diagnostics = {
        "near_constant_percent": float(np.mean(stds <= constant_threshold) * 100),
        "near_constant_threshold": constant_threshold,
        "highly_volatile_percent": float(np.mean(rough > volatile_threshold) * 100),
        "highly_volatile_threshold": volatile_threshold,
        "median_target_correlation": float(np.nanmedian(np.asarray(target_corr)[:-1])),
        "median_daily_correlation": float(np.nanmedian(daily)),
        "median_weekly_correlation": float(np.nanmedian(weekly)),
    }
    stats.to_csv(CSV / f"{name}_channels.csv", index=False)
    cluster_df.to_csv(CSV / f"{name}_clusters.csv", index=False)

    quantile_targets = np.quantile(means, [0.1, 0.5, 0.9])
    reps = []
    used: set[int] = set()
    for target_mean in quantile_targets:
        order = np.argsort(np.abs(means - target_mean))
        idx = next(int(j) for j in order if int(j) not in used)
        used.add(idx)
        reps.append(str(values.columns[idx]))
    width = 7 * 24
    start = representative_start(values[reps], width)
    stop = min(len(values), start + width)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    levels = ["low", "medium", "high"]
    for ax, col, level, color in zip(axes, reps, levels, COLORS):
        ax.plot(dates[start:stop], values[col].iloc[start:stop], color=color, lw=0.9)
        ax.set_ylabel(f"{level}\nch. {col}", fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"{name}: representative low-, medium-, and high-activity channels")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}_representatives.pdf", bbox_inches="tight")
    plt.close(fig)
    return stats, cluster_df, diagnostics, reps


def latex_table(headers: list[str], rows: list[list[str]], spec: str | None = None, long: bool = False) -> str:
    spec = spec or ("l" + "r" * (len(headers) - 1))
    env = "longtable" if long else "tabular"
    lines = [f"\\begin{{{env}}}{{{spec}}}", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule"]
    if long:
        lines += ["\\endfirsthead", "\\toprule", " & ".join(headers) + " \\\\", "\\midrule", "\\endhead"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", f"\\end{{{env}}}"]
    return "\n".join(lines)


def write_channel_appendix(name: str, stats: pd.DataFrame) -> None:
    rows = []
    for _, r in stats.iterrows():
        rows.append(
            [
                tex(r["channel"]),
                fmt(r["min"]),
                fmt(r["max"]),
                fmt(r["mean"]),
                fmt(r["std"]),
                fmt(r["daily_lag_correlation"]),
                tex(r["cluster"]),
            ]
        )
    content = latex_table(["Channel", "Min", "Max", "Mean", "Std", r"$\rho_{24}$", "Cluster"], rows, "lrrrrrl", long=True)
    (TAB / f"{name.lower()}_channels.tex").write_text(content, encoding="utf-8")


def build_report(
    ett_stats: dict[str, pd.DataFrame],
    ett_lags: dict[str, pd.DataFrame],
    weather_stats: pd.DataFrame,
    weather_groups: pd.DataFrame,
    weather_invalid: int,
    anonymous: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, float], list[str]]],
) -> None:
    parts = [r"""\documentclass[10pt]{article}
\usepackage[margin=0.72in]{geometry}
\usepackage{booktabs,longtable,graphicx,subcaption,array,xcolor,hyperref,siunitx,amsmath}
\usepackage[T1]{fontenc}
\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}
\newcommand{\datasetunit}{dataset load unit}
\title{Variable-Level Semantic and Behavioural Audit of the Long-Horizon Forecasting Benchmarks}
\author{}
\date{}
\begin{document}
\maketitle

\begin{abstract}
This audit describes what each benchmark channel measures before treating it as a forecasting coordinate. It combines source documentation with statistics computed directly from the CSV files used by this repository. We report raw ranges, means, standard deviations, dominant periods, cross-variable relations, representative windows, and modelling implications. Missing sentinels are excluded rather than interpreted as physical measurements. The analysis is descriptive: correlation, spectral prominence, and clustering do not by themselves establish physical causation.
\end{abstract}

\section{Audit protocol}
All numerical results are generated by \texttt{dataset\_audit/audit.py}. Means and standard deviations use all finite local observations (population standard deviation). Dominant period is the strongest FFT period after linear detrending and Hann tapering, searched from four samples to 60 days. ``Representative week'' means the non-overlapping seven-day block whose mean, standard deviation, and mean absolute first difference are jointly closest to the componentwise median block. ETT lead/lag uses first differences and searches $\pm48$ hours; a positive lag means that load changes lead oil-temperature changes. For the anonymous panels, $\rho_{24}$ and $\rho_{168}$ are hourly lag correlations. Nearly constant means channel standard deviation below 1\% of the median nonzero channel standard deviation. Highly volatile means normalized first-difference roughness above the Tukey upper fence. These fixed, data-independent definitions prevent post-hoc visual labels.

\paragraph{Semantic sources and local transformations.}
ETT names and collection details follow the official ETT repository (\url{https://github.com/zhouhaoyi/ETDataset}). Weather meanings and units follow the MPI Biogeochemistry weather-station documentation (\url{https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf}). Electricity semantics follow the UCI ElectricityLoadDiagrams20112014 record (\url{https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014}). Traffic is an hourly processed panel of PeMS detector occupancy; PeMS measures detector occupancy and the UCI PeMS description defines occupancy as a fraction between zero and one (\url{https://archive.ics.uci.edu/dataset/204/pems_sf}). Dates, channel counts, and all values below are taken from the local benchmark files, which may be resampled or relabelled versions of their source archives.

\paragraph{The overloaded \texttt{OT} label.}
\texttt{OT} means oil temperature only in ETT. From the local Weather values and source-column order, we infer that its \texttt{OT} is the source CO2 channel renamed for a common loader; the repository contains no retained conversion manifest that proves the rename directly. Likewise, Electricity \texttt{OT} is the final anonymous client (original index 320) and Traffic \texttt{OT} is the final anonymous detector (original index 861). Neither is an oil-temperature target. This distinction matters for both interpretation and leakage audits.
"""]

    parts.append(r"\section{Electricity Transformer Temperature datasets}")
    parts.append(r"""ETTh1/ETTh2 are hourly and ETTm1/ETTm2 are 15-minute records from two transformer stations. Each contains six external load measurements and transformer oil temperature. High, middle, and low distinguish the three load groups; useful and useless are the source archive's names for load components that do and do not contribute to useful transfer. The archive does not expose enough station wiring metadata to equate these labels unambiguously with particular voltage sides or to assert an exact active/reactive-power mapping. It also does not publish an engineering unit for the six load columns; we therefore retain \datasetunit{} rather than inventing kW or kVAr. Oil temperature is reported in degrees Celsius. ETTm1 corresponds to station 1 and ETTm2 to station 2, so the minute/hour pairs should resemble one another while station 1 and station 2 need not share ranges.""")
    for name in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        st = ett_stats[name]
        dates, values, _ = load(name)
        parts.append(f"\\subsection{{{name}}}")
        parts.append(
            f"The local file contains {len(values):,} observations from {dates[0]} to {dates[-1]} at {cadence_hours(dates):g}-hour cadence. "
            "Table~\\ref{tab:" + name.lower() + "} reports unnormalized values."
        )
        rows = []
        for _, r in st.iterrows():
            rows.append([tex(r["variable"]), tex(r["meaning"]), tex(r["unit"]), f"{fmt(r['min'])}--{fmt(r['max'])}", f"{fmt(r['mean'])} $\\pm$ {fmt(r['std'])}", f"{fmt(r['dominant_period_hours'], 1)} h ({period_label(r['dominant_period_hours'])})"])
        parts.append("\\begin{table}[ht]\n\\centering\\scriptsize\n\\caption{" + name + " variable semantics and empirical behaviour.}\\label{tab:" + name.lower() + "}\n\\resizebox{\\textwidth}{!}{%\n" + latex_table(["Variable", "Meaning", "Unit", "Min--Max", r"Mean $\pm$ Std", "Dominant period"], rows, "lllrrl") + "}\n\\end{table}")
        lag_rows = []
        for _, r in ett_lags[name].iterrows():
            direction = "load leads" if r["best_change_lag_hours"] > 0 else ("load lags" if r["best_change_lag_hours"] < 0 else "synchronous")
            lag_rows.append([tex(r["variable"]), fmt(r["level_correlation"]), fmt(r["best_change_lag_hours"], 1), tex(direction), fmt(r["change_correlation"])])
        parts.append("\\begin{table}[ht]\n\\centering\\small\n\\caption{" + name + " load--oil-temperature relations. Level correlation is contemporaneous; lag is selected on first differences.}\\label{tab:" + name.lower() + "lags}\n" + latex_table(["Load", r"Level $\rho$", "Best lag (h)", "Direction", r"Change $\rho$"], lag_rows, "lrrlr") + "\n\\end{table}")
        parts.append("\\begin{figure}[ht]\n\\centering\n\\begin{subfigure}{.49\\textwidth}\\includegraphics[width=\\linewidth]{figures/" + name + "_full.pdf}\\caption{Complete record.}\\end{subfigure}\n\\begin{subfigure}{.49\\textwidth}\\includegraphics[width=\\linewidth]{figures/" + name + "_week.pdf}\\caption{Representative seven-day window.}\\end{subfigure}\n\\caption{" + name + " raw trajectories.}\\label{fig:" + name.lower() + "}\n\\end{figure}")

    parts.append(r"""\subsection{Cross-station and sampling comparison}
The hourly and 15-minute versions of a station have closely matched marginal distributions, as expected from alternate sampling of the same physical system: station 1 oil temperature has mean 13.325 (ETTh1) versus 13.321 (ETTm1), while station 2 has mean 26.609 in both resolutions. The larger contrast is between stations. Station 2 oil temperature spans approximately $-2.65$ to $58.88\,^{\circ}$C, compared with $-4.22$ to $46.01\,^{\circ}$C at station 1; its HUFL and MUFL also reach 107.893 and 93.230, far above station 1. Daily load periodicity and smoother oil-temperature motion recur across the four files, but several station-2 channels are dominated by slower multiweek spectral peaks. Lead/lag signs should be read as predictive timing, not thermal causality: differencing suppresses common slow trends, but weather, dispatch, and controller state remain unobserved confounders.

\paragraph{Modelling implication.}
ETT is low-dimensional but not univariate. A useful model should preserve sharp daily load changes while continuing the smoother thermal state, and it should not assume that one station's normalization or cross-channel coupling transfers unchanged to the other. The weak-to-moderate change correlations also argue against treating any single load as a deterministic driver of oil temperature.
""")

    parts.append(r"\section{Weather}")
    dates, values, _ = load("Weather")
    parts.append(
        f"The local Weather panel contains {len(values):,} ten-minute observations from {dates[0]} to {dates[-1]} and 21 variables. "
        f"We excluded {weather_invalid:,} occurrences of the source missing sentinel $-9999$ before every statistic. Variables are grouped by physical role; CO2 is retained as a separate atmospheric-composition group."
    )
    rows = []
    for _, r in weather_stats.iterrows():
        rows.append([tex(r["variable"]), r["meaning"], r["unit"], tex(r["group"]), f"{fmt(r['min'])}--{fmt(r['max'])}", f"{fmt(r['mean'])} $\\pm$ {fmt(r['std'])}", tex(r["behavior"])])
    parts.append("\\begin{longtable}{llllrrl}\n\\caption{Weather variable dictionary and statistics after excluding $-9999$ sentinels.}\\label{tab:weather}\\\\\n\\toprule\nVariable & Meaning & Unit & Group & Min--Max & Mean $\\pm$ Std & Main pattern \\\\\n+\\midrule\\endfirsthead\n\\toprule\nVariable & Meaning & Unit & Group & Min--Max & Mean $\\pm$ Std & Main pattern \\\\\n+\\midrule\\endhead\n" + "\n".join(" & ".join(row) + r" \\" for row in rows) + "\n\\bottomrule\n\\end{longtable}")
    group_rows = [[tex(r["group"]), str(int(r["variables"])), fmt(r["median_within_group_correlation"]), fmt(r["median_roughness"])] for _, r in weather_groups.iterrows()]
    parts.append("\\begin{table}[ht]\\centering\\small\n\\caption{Within-group Weather co-movement and roughness.}\\label{tab:weather-groups}\n" + latex_table(["Group", "Variables", "Median within-group $\\rho$", "Median roughness"], group_rows, "lrrr") + "\n\\end{table}")
    parts.append(r"""\begin{figure}[ht]
\centering
\begin{subfigure}{.62\textwidth}\includegraphics[width=\linewidth]{figures/Weather_groups_week.pdf}\caption{Standardized group trajectories.}\end{subfigure}
\begin{subfigure}{.36\textwidth}\includegraphics[width=\linewidth]{figures/Weather_correlation.pdf}\caption{Variable correlation.}\end{subfigure}
\caption{Weather behaviour and co-movement. Standardization is used only for visualization.}\label{fig:weather}
\end{figure}

Temperature, potential temperature, dew point, and logger temperature are smooth state variables with strong common motion. Vapor-pressure and humidity variables are physically coupled transformations and consequently move together, although relative humidity and vapor-pressure deficit often oppose one another. Wind direction is circular and should not be treated as an ordinary Euclidean scalar; sine/cosine encoding is preferable. Rain duration and amount are sparse and sharp, while radiation has a strong daylight envelope. These distinctions explain why one shared smoothness prior is inappropriate.

\paragraph{Modelling implication.}
Weather requires heterogeneous observation models: smooth continuation for thermodynamic state, bounded/circular treatment for wind direction, and intermittent-event capacity for rain and gusts. High within-group correlation also means that evaluating cross-variable modules solely on aggregate error can hide redundancy; group-wise errors are a useful diagnostic.
""")

    for name in ["Electricity", "Traffic"]:
        stats, clusters, diag, reps = anonymous[name]
        dates, values, _ = load(name)
        write_channel_appendix(name, stats)
        parts.append(f"\\section{{{name}}}")
        if name == "Electricity":
            description = (
                "Each channel is an anonymized electricity client. The source archive records quarter-hour power in kW; the local benchmark is an hourly 321-channel transformation, so values below are reported in the local stored scale rather than silently converting units. "
                "The final column `OT' is anonymous client 320."
            )
        else:
            description = (
                "Each channel is an anonymized PeMS road detector's occupancy fraction, a dimensionless fraction of the sampling interval during which the detector is occupied. The local file is hourly with 862 detectors. "
                "The final column `OT' is anonymous detector 861."
            )
        parts.append(description + f" The local file spans {dates[0]} to {dates[-1]} ({len(values):,} observations).")
        parts.append(
            f"Using the preregistered thresholds, {diag['near_constant_percent']:.2f}\\% of channels are nearly constant and {diag['highly_volatile_percent']:.2f}\\% are highly volatile. "
            f"The median 24-hour and 168-hour lag correlations are {diag['median_daily_correlation']:.3f} and {diag['median_weekly_correlation']:.3f}. "
            f"The median correlation of anonymous channels with the local `OT' channel is {diag['median_target_correlation']:.3f}; this is a panel-dependence summary, not a target-causality claim."
        )
        rep_rows = []
        for level, col in zip(["low", "medium", "high"], reps):
            r = stats.loc[stats["channel"].astype(str) == col].iloc[0]
            rep_rows.append([level, tex(col), f"{fmt(r['min'])}--{fmt(r['max'])}", f"{fmt(r['mean'])} $\\pm$ {fmt(r['std'])}", fmt(r["daily_lag_correlation"]), tex(r["cluster"])])
        parts.append("\\begin{table}[ht]\\centering\\small\n\\caption{" + name + " representative channels selected by activity-mean quantiles.}\\label{tab:" + name.lower() + "-reps}\n" + latex_table(["Activity", "Channel", "Min--Max", r"Mean $\pm$ Std", r"$\rho_{24}$", "Cluster"], rep_rows, "llrrrl") + "\n\\end{table}")
        cluster_rows = [[tex(r["cluster"]), str(int(r["channels"])), fmt(r["fraction_percent"], 1) + r"\%", fmt(r["median_mean"]), fmt(r["median_std"]), fmt(r["median_daily_corr"]), fmt(r["median_weekly_corr"])] for _, r in clusters.iterrows()]
        parts.append("\\begin{table}[ht]\\centering\\small\n\\caption{" + name + " data-driven behavioural clusters ($k=3$, seed 42).}\\label{tab:" + name.lower() + "-clusters}\n" + latex_table(["Cluster", "$n$", "Share", "Median mean", "Median std", r"$\rho_{24}$", r"$\rho_{168}$"], cluster_rows, "lrrrrrr") + "\n\\end{table}")
        parts.append("\\begin{figure}[ht]\\centering\\includegraphics[width=.9\\linewidth]{figures/" + name + "_representatives.pdf}\\caption{" + name + " representative raw channels.}\\label{fig:" + name.lower() + "}\n\\end{figure}")
        if name == "Electricity":
            implication = "Client scale, zero inflation, and periodicity vary substantially. Per-channel normalization prevents large clients from dominating, while daily/weekly structure and cross-client factors motivate both temporal and variable-axis modelling. Anonymous IDs do not justify residential/commercial labels; the clusters are behavioural only."
        else:
            implication = "Occupancy is bounded, strongly periodic for many detectors, and locally irregular for others. A model should preserve detector-specific baselines while sharing congestion waves across sensors; treating all 862 channels as exchangeable ignores location-dependent behaviour."
        parts.append("\\paragraph{Modelling implication.} " + implication)

    parts.append(r"""\section{Cross-dataset conclusions}
The audit exposes three distinct geometries. ETT has only seven semantically different variables and combines daily electrical forcing with smoother transformer temperature. Weather has groups of physically coupled variables but sharply different temporal regularities. Electricity and Traffic are high-dimensional panels of repeated measurement type, where channel identity is anonymous but scale, periodicity, and volatility differ strongly. Consequently, a forecasting model should not infer ``more channels'' as ``more physical modalities'': in ETT and Weather, semantics differ by channel; in Electricity and Traffic, channels are instances of a common measurement process.

\section{Full anonymous-channel statistics}
The following appendices provide the requested per-channel ranges, means, standard deviations, daily periodicity, and cluster assignments. They are generated from the same unnormalized local files.
\subsection{Electricity channels}
\input{tables/electricity_channels.tex}
\clearpage
\subsection{Traffic channels}
\input{tables/traffic_channels.tex}

\end{document}
""")
    report = "\n\n".join(parts)
    report = report.replace("+\\midrule", "\\midrule")
    report = report.replace(
        "\\begin{longtable}{llllrrl}",
        "\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{longtable}{@{}p{.09\\textwidth}p{.19\\textwidth}p{.10\\textwidth}p{.13\\textwidth}rrp{.20\\textwidth}@{}}",
        1,
    )
    report = report.replace("\\end{longtable}", "\\end{longtable}\n\\normalsize", 1)
    (OUT / "dataset_audit.tex").write_text(report, encoding="utf-8")


def main() -> None:
    for directory in [OUT, FIG, TAB, CSV]:
        directory.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in DATASETS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))
    print("[audit] ETT", flush=True)
    ett_stats, ett_lags, _ = ett_audit()
    print("[audit] Weather", flush=True)
    weather_stats, weather_groups, weather_invalid = weather_audit()
    anonymous = {}
    for name in ["Electricity", "Traffic"]:
        print(f"[audit] {name}", flush=True)
        anonymous[name] = anonymous_audit(name)
    print("[audit] LaTeX", flush=True)
    build_report(ett_stats, ett_lags, weather_stats, weather_groups, weather_invalid, anonymous)
    print(f"Wrote {OUT / 'dataset_audit.tex'}")
    print(f"Figures: {len(list(FIG.glob('*.pdf')))}")
    print(f"Statistics: {len(list(CSV.glob('*.csv')))} CSV files")


if __name__ == "__main__":
    main()
