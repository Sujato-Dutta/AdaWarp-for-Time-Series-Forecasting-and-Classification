"""Generate publication-ready SVG figures without Matplotlib."""

from __future__ import annotations

from html import escape
import math
from pathlib import Path

from common import DATASETS, HORIZONS, METHODS, RESULTS, read_csv


GREEN = (15, 118, 87)
RED = (180, 35, 24)
WHITE = (250, 250, 249)
INK = "#17202A"
MUTED = "#667085"
GRID = "#D0D5DD"
DATASET_COLORS = {
    "ETTh1": "#7F56D9",
    "ETTh2": "#2E90FA",
    "ETTm2": "#06AED4",
    "Weather": "#12B76A",
    "Electricity": "#F79009",
    "Traffic": "#D92D20",
}


def _blend(start: tuple[int, int, int], end: tuple[int, int, int], amount: float) -> str:
    amount = min(1.0, max(0.0, amount))
    rgb = tuple(round(start[index] * (1 - amount) + end[index] * amount) for index in range(3))
    return "#" + "".join(f"{value:02X}" for value in rgb)


def diverging(value: float, limit: float) -> str:
    if value >= 0:
        return _blend(WHITE, GREEN, abs(value) / max(limit, 1e-9))
    return _blend(WHITE, RED, abs(value) / max(limit, 1e-9))


class SVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            f"<title>{escape(title)}</title>",
            '<rect width="100%" height="100%" fill="#FFFFFF"/>',
            '<style>text{font-family:Arial,Helvetica,sans-serif} .title{font-size:24px;font-weight:700;fill:#17202A}.panel{font-size:17px;font-weight:700;fill:#17202A}.label{font-size:13px;fill:#344054}.small{font-size:11px;fill:#667085}.value{font-size:12px;font-weight:600}.axis{stroke:#98A2B3;stroke-width:1}.grid{stroke:#EAECF0;stroke-width:1}</style>',
        ]

    def rect(self, x, y, width, height, fill="none", stroke="none", radius=0, stroke_width=1):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')

    def line(self, x1, y1, x2, y2, stroke=INK, width=1, dash=None):
        dash_text = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_text}/>')

    def circle(self, x, y, radius, fill, stroke="#FFFFFF", stroke_width=1):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')

    def polyline(self, points, stroke, width=2, fill="none", opacity=1.0):
        encoded = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(f'<polyline points="{encoded}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>')

    def text(self, x, y, content, css="label", anchor="start", fill=None, rotate=None):
        fill_text = f' fill="{fill}"' if fill else ""
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.parts.append(f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}"{fill_text}{transform}>{escape(str(content))}</text>')

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.parts + ["</svg>"]), encoding="utf-8")


def relative_heatmaps() -> None:
    rows = read_csv(RESULTS / "statistics" / "patchtst_relative_gains.csv")
    lookup = {(row["metric"], row["dataset"], int(row["horizon"])): float(row["relative_improvement_pct"]) for row in rows}
    values = list(lookup.values())
    limit = max(abs(value) for value in values)
    svg = SVG(1280, 570, "MVPF relative improvement over PatchTST")
    svg.text(640, 35, "MVPF relative improvement over PatchTST", "title", "middle")
    svg.text(640, 58, "Positive values (green) favor MVPF; negative values (red) favor PatchTST", "small", "middle")
    cell_w, cell_h = 104, 58
    for panel, metric in enumerate(("mse", "mae")):
        origin_x = 150 + panel * 620
        origin_y = 125
        svg.text(origin_x + 2 * cell_w, 91, metric.upper(), "panel", "middle")
        for column, horizon in enumerate(HORIZONS):
            svg.text(origin_x + column * cell_w + cell_w / 2, 112, horizon, "label", "middle")
        for row_index, dataset in enumerate(DATASETS):
            y = origin_y + row_index * cell_h
            svg.text(origin_x - 14, y + 35, dataset, "label", "end")
            for column, horizon in enumerate(HORIZONS):
                value = lookup[(metric, dataset, horizon)]
                x = origin_x + column * cell_w
                fill = diverging(value, limit)
                svg.rect(x, y, cell_w - 4, cell_h - 4, fill, "#FFFFFF", 4)
                text_fill = "#FFFFFF" if abs(value) / limit > 0.52 else INK
                svg.text(x + (cell_w - 4) / 2, y + 33, f"{value:+.1f}%", "value", "middle", text_fill)
    legend_x, legend_y = 505, 505
    for index in range(21):
        fraction = index / 20
        value = -limit + 2 * limit * fraction
        svg.rect(legend_x + index * 13, legend_y, 14, 16, diverging(value, limit))
    svg.text(legend_x, legend_y + 34, f"-{limit:.1f}%", "small", "middle")
    svg.text(legend_x + 130, legend_y + 34, "0", "small", "middle")
    svg.text(legend_x + 260, legend_y + 34, f"+{limit:.1f}%", "small", "middle")
    svg.save(RESULTS / "figures" / "patchtst_relative_heatmaps.svg")


def average_rank_figure() -> None:
    rows = read_csv(RESULTS / "statistics" / "average_ranks.csv")
    lookup = {row["method"]: row for row in rows}
    svg = SVG(1280, 620, "Average ranks across 24 LTSF tasks")
    svg.text(640, 36, "Average rank across 24 dataset-horizon tasks", "title", "middle")
    svg.text(640, 58, "Lower rank is better; Nemenyi CD at alpha=0.05 shown for context", "small", "middle")
    critical_difference = 3.031 * math.sqrt(len(METHODS) * (len(METHODS) + 1) / (6 * 24))
    for panel, metric in enumerate(("mse", "mae")):
        x0 = 115 + panel * 620
        y0 = 112
        plot_w = 480
        svg.text(x0 + plot_w / 2, 88, metric.upper(), "panel", "middle")
        for rank in range(1, 9):
            x = x0 + (rank - 1) * plot_w / 7
            svg.line(x, y0, x, y0 + 430, "#EAECF0")
            svg.text(x, y0 - 12, rank, "small", "middle")
        sorted_methods = sorted(METHODS, key=lambda method: float(lookup[method][f"{metric}_average_rank"]))
        for row_index, method in enumerate(sorted_methods):
            rank = float(lookup[method][f"{metric}_average_rank"])
            y = y0 + 40 + row_index * 46
            x = x0 + (rank - 1) * plot_w / 7
            color = "#0F7657" if method == "MVPF" else "#475467"
            svg.line(x0, y, x, y, "#D0D5DD", 2)
            svg.circle(x, y, 7 if method == "MVPF" else 5, color)
            svg.text(x + 12, y + 4, f"{method} ({rank:.2f})", "label", "start", color)
        cd_x1 = x0
        cd_x2 = x0 + critical_difference * plot_w / 7
        cd_y = y0 + 420
        svg.line(cd_x1, cd_y, cd_x2, cd_y, "#344054", 3)
        svg.line(cd_x1, cd_y - 6, cd_x1, cd_y + 6, "#344054", 2)
        svg.line(cd_x2, cd_y - 6, cd_x2, cd_y + 6, "#344054", 2)
        svg.text((cd_x1 + cd_x2) / 2, cd_y + 22, f"CD = {critical_difference:.2f}", "small", "middle")
    svg.save(RESULTS / "figures" / "average_rank_cd.svg")


def horizon_figure() -> None:
    rows = read_csv(RESULTS / "statistics" / "patchtst_relative_gains.csv")
    lookup = {(row["metric"], row["dataset"], int(row["horizon"])): float(row["relative_improvement_pct"]) for row in rows}
    all_values = list(lookup.values())
    lower = min(-5.0, math.floor(min(all_values) / 5) * 5)
    upper = max(15.0, math.ceil(max(all_values) / 5) * 5)
    svg = SVG(1280, 560, "Horizon robustness relative to PatchTST")
    svg.text(640, 36, "Horizon robustness relative to PatchTST", "title", "middle")
    svg.text(640, 58, "Dataset points with the cross-dataset median in black", "small", "middle")
    for panel, metric in enumerate(("mse", "mae")):
        x0 = 90 + panel * 620
        y0 = 105
        width, height = 500, 350
        svg.text(x0 + width / 2, 88, metric.upper(), "panel", "middle")
        def x_pos(index): return x0 + index * width / 3
        def y_pos(value): return y0 + height - (value - lower) * height / (upper - lower)
        for tick in range(int(lower), int(upper) + 1, 10):
            y = y_pos(tick)
            svg.line(x0, y, x0 + width, y, "#EAECF0")
            svg.text(x0 - 10, y + 4, f"{tick}%", "small", "end")
        svg.line(x0, y_pos(0), x0 + width, y_pos(0), "#98A2B3", 1.5)
        for index, horizon in enumerate(HORIZONS):
            svg.text(x_pos(index), y0 + height + 25, horizon, "label", "middle")
        for dataset in DATASETS:
            points = [(x_pos(index), y_pos(lookup[(metric, dataset, horizon)])) for index, horizon in enumerate(HORIZONS)]
            svg.polyline(points, DATASET_COLORS[dataset], 1.5, opacity=0.55)
            for x, y in points:
                svg.circle(x, y, 4, DATASET_COLORS[dataset], stroke_width=0)
        medians = []
        for index, horizon in enumerate(HORIZONS):
            values = sorted(lookup[(metric, dataset, horizon)] for dataset in DATASETS)
            median = 0.5 * (values[2] + values[3])
            medians.append((x_pos(index), y_pos(median)))
        svg.polyline(medians, "#101828", 4)
        for x, y in medians:
            svg.circle(x, y, 6, "#101828")
    legend_y = 525
    for index, dataset in enumerate(DATASETS):
        x = 235 + index * 145
        svg.circle(x, legend_y - 4, 5, DATASET_COLORS[dataset], stroke_width=0)
        svg.text(x + 10, legend_y, dataset, "small")
    svg.save(RESULTS / "figures" / "horizon_robustness.svg")


def ablation_heatmaps() -> None:
    rows = read_csv(RESULTS / "ablations" / "ablation_task_deltas.csv")
    lookup = {
        (row["ablation"], row["dataset"], int(row["horizon"]), metric): float(row[f"{metric}_relative_degradation_pct"])
        for row in rows
        for metric in ("mse", "mae")
    }
    datasets = ("ETTh1", "ETTh2", "Weather", "Electricity", "Traffic")
    groups = [
        (("no_linear_field", "single_scale_16"), "core_ablation_heatmaps.svg", "Core MVPF ablations"),
        (("no_adaptive_shifts", "no_trend_residuals"), "diagnostic_ablation_heatmaps.svg", "Diagnostic MVPF ablations"),
    ]
    labels = {
        "no_linear_field": "No linear field",
        "single_scale_16": "Single scale (16)",
        "no_adaptive_shifts": "No adaptive shifts",
        "no_trend_residuals": "No trend/residual",
    }
    for ablations, filename, title in groups:
        selected_values = [lookup[(ablation, dataset, horizon, metric)] for ablation in ablations for dataset in datasets for horizon in HORIZONS for metric in ("mse", "mae")]
        limit = max(1.0, max(abs(value) for value in selected_values))
        svg = SVG(1430, 760, title)
        svg.text(715, 34, title, "title", "middle")
        svg.text(715, 56, "Positive values mean the ablation is worse than full MVPF", "small", "middle")
        cell_w, cell_h = 78, 46
        for ablation_index, ablation in enumerate(ablations):
            for metric_index, metric in enumerate(("mse", "mae")):
                panel = ablation_index * 2 + metric_index
                x0 = 120 + (panel % 2) * 700
                y0 = 125 + (panel // 2) * 315
                svg.text(x0 + 2 * cell_w, y0 - 48, f"{labels[ablation]} - {metric.upper()}", "panel", "middle")
                for column, horizon in enumerate(HORIZONS):
                    svg.text(x0 + column * cell_w + cell_w / 2, y0 - 18, horizon, "small", "middle")
                for row_index, dataset in enumerate(datasets):
                    y = y0 + row_index * cell_h
                    svg.text(x0 - 10, y + 28, dataset, "small", "end")
                    for column, horizon in enumerate(HORIZONS):
                        value = lookup[(ablation, dataset, horizon, metric)]
                        x = x0 + column * cell_w
                        svg.rect(x, y, cell_w - 3, cell_h - 3, diverging(value, limit), "#FFFFFF", 3)
                        text_fill = "#FFFFFF" if abs(value) / limit > 0.55 else INK
                        svg.text(x + (cell_w - 3) / 2, y + 27, f"{value:+.1f}", "small", "middle", text_fill)
        svg.save(RESULTS / "figures" / filename)


def main() -> None:
    relative_heatmaps()
    average_rank_figure()
    horizon_figure()
    ablation_heatmaps()
    print("wrote 5 SVG figures to mvpf-tests/results/figures")


if __name__ == "__main__":
    main()

