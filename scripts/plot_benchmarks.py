"""benchmarks/results.json -> the plots (light and dark) and benchmarks/README.md, so every printed number comes from
the measured file. Test numbers come from each model's own release record: releases/v1.json for den and
benchmarks/kev-4b-card.json (copied from Kev's model card, with its sources) for Kev-4B.

    uv run python -m scripts.plot_benchmarks
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

OUT = Path("benchmarks")
RELEASE = Path("releases/v1.json")
DEN, KEV = "den-v1", "kev-4b"
NAMES = {DEN: "den v1 (yours)", KEV: "Kev-4B (Kev 1.0)"}
PANELS = {
    "transfer": ("Out of domain", "QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval"),
    "breadth": ("Held-out public datasets", "breadth-v1, 14 datasets"),
    "skills": ("Reasoning skills", "hard-v1: policies, probability, dates"),
    "devtools": ("Developer tooling", "code review, commits, flaky tests, safety"),
    "documents": ("Long documents", "CFPB complaints, up to ~7k tokens"),
    "core": ("Typed decisions", "decision-v7"),
    "probes": ("Dates and unknowable items", "probes"),
    "semif": ("Authored if-then rules", "SemIf"),
    "binding": ("Entity binding", "binding"),
}


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    surface: str
    ink: str
    muted: str
    grid: str
    neutral: str
    series: dict[str, str]


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984", {DEN: "#2a78d6", KEV: "#eb6834"})
DARK = Theme("dark", "#1a1a19", "#ffffff", "#c3c2b7", "#383835", "#8a8984", {DEN: "#3987e5", KEV: "#d95926"})


def short(panel: str) -> str:
    return Path(panel).stem


def figure(theme: Theme, title: str, subtitle: str, height: float) -> tuple[Figure, Axes]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": theme.grid})
    fig, ax = plt.subplots(figsize=(12, height), dpi=150)
    fig.patch.set_facecolor(theme.surface)
    ax.set_facecolor(theme.surface)
    fig.text(0.02, 0.97, title, fontsize=19, color=theme.ink, va="top", weight="bold")
    fig.text(0.02, 0.97 - 0.42 / height, subtitle, fontsize=11, color=theme.muted, va="top")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=theme.muted, length=0)
    return fig, ax


def legend(fig: Figure, theme: Theme, height: float) -> None:
    handles = [Rectangle((0, 0), 1, 1, color=theme.series[m]) for m in (DEN, KEV)]
    fig.legend(handles, [NAMES[DEN], NAMES[KEV]], loc="upper right", bbox_to_anchor=(0.98, 0.97 - 0.25 / height),
               frameon=False, ncol=2, labelcolor=theme.ink, fontsize=11)  # fmt: skip


def bars(theme: Theme, rows: Sequence[tuple[str, str, dict[str, float]]], title: str, subtitle: str, note: str,
         path: Path) -> None:  # fmt: skip
    """Horizontal grouped bars, one group per panel, den above Kev, each value printed at its bar's end."""
    height = 1.6 + 0.62 * len(rows)
    fig, ax = figure(theme, title, subtitle, height)
    legend(fig, theme, height)
    bar = 0.34
    for i, (label, sub, values) in enumerate(rows):
        y = len(rows) - 1 - i
        for j, model in enumerate((DEN, KEV)):
            if (v := values.get(model)) is None:
                continue
            top = y + (0.5 - j) * (bar + 0.04)
            ax.barh(top, v, height=bar, color=theme.series[model], linewidth=0)
            ax.text(v + 0.006, top, f"{v:.1%}", va="center", fontsize=10, color=theme.ink)
        ax.text(-0.01, y + 0.08, label, ha="right", va="center", fontsize=11, color=theme.ink, transform=ax.transData)
        ax.text(-0.01, y - 0.28, sub, ha="right", va="center", fontsize=8.5, color=theme.muted)
    ax.set_xlim(0, 1.06)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.30, right=0.97, top=1 - 1.15 / height, bottom=0.55 / height + 0.25 / height)
    fig.text(0.02, 0.02, note, fontsize=8.5, color=theme.muted, va="bottom")
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)


def deltas(theme: Theme, rows: Sequence[tuple[str, dict[str, float]]], path: Path) -> None:
    """den minus Kev, accuracy in points, with each 95% paired interval; colored by which model the interval favors."""
    height = 1.8 + 0.5 * len(rows)
    subtitle = "Points, with 95% intervals from resampling whole records (2,000 resamples); same questions on both"
    fig, ax = figure(theme, "Accuracy difference, den v1 minus Kev-4B", subtitle, height)
    for i, (label, d) in enumerate(rows):
        y = len(rows) - 1 - i
        diff, low, high = (100 * d[k] for k in ("diff", "low", "high"))
        color = theme.series[DEN] if low > 0 else theme.series[KEV] if high < 0 else theme.neutral
        ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot([diff], [y], "o", color=color, markersize=8, markeredgecolor=theme.surface, markeredgewidth=2)
        bold = "bold" if label.startswith("All") else "normal"
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=11, color=theme.ink, weight=bold,
                transform=ax.get_yaxis_transform())  # fmt: skip
        ax.text(1.01, y, f"{diff:+.1f}  [{low:+.1f}, {high:+.1f}]", ha="left", va="center", fontsize=10,
                color=theme.ink, transform=ax.get_yaxis_transform())  # fmt: skip
    span = max(6.0, *(abs(100 * d[k]) + 1 for _, d in rows for k in ("low", "high")))
    ax.set_xlim(-span, span)
    ax.axvline(0, color=theme.muted, linewidth=1)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.grid(axis="x", color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("← Kev-4B better      points      den v1 better →", color=theme.muted)
    fig.subplots_adjust(left=0.27, right=0.80, top=1 - 1.2 / height, bottom=0.75 / height)
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)


def reliability(theme: Theme, curves: dict[str, list[list[float]]], eces: dict[str, float], path: Path) -> None:
    """Accuracy against confidence, pooled over every panel: on the diagonal, a stated 80% is right 80% of the time."""
    fig, ax = figure(
        theme,
        "Calibration, every dev panel pooled",
        "Accuracy against stated confidence (10 bins); the dashed diagonal is perfect calibration",
        6.4,
    )
    ax.plot([0, 1], [0, 1], linestyle=(0, (4, 4)), color=theme.muted, linewidth=1)
    for model in (DEN, KEV):
        points = [(c / n, r / n) for c, r, n in curves[model] if n >= 20]
        label = f"{NAMES[model]}  ECE {eces[model]:.1%}"
        ax.plot(*zip(*points, strict=True), color=theme.series[model], linewidth=2, marker="o", markersize=8,
                markeredgecolor=theme.surface, markeredgewidth=2, label=label)  # fmt: skip
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.set_yticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.spines["left"].set_visible(True)
    ax.set_xlabel("stated confidence", color=theme.muted)
    ax.set_ylabel("accuracy", color=theme.muted)
    ax.grid(color=theme.grid, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=theme.ink, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.12)
    fig.savefig(path, facecolor=theme.surface)
    plt.close(fig)


def pooled_curve(results: dict[str, Any]) -> list[list[float]]:
    cells = [[0.0, 0.0, 0.0] for _ in range(10)]
    for r in results.values():
        for cell, add in zip(cells, r["reliability"], strict=True):
            for k in range(3):
                cell[k] += add[k]
    return cells


def curve_ece(cells: list[list[float]]) -> float:
    n = sum(c[2] for c in cells)
    return sum(abs(c[0] - c[1]) for c in cells) / n if n else 0.0


def pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def num(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def temperature(models: dict[str, Any]) -> float:
    """A server's calibration temperature from its /v1/models: Kev's on the model, den's under `data`."""
    entry = models["models"][0]
    return float(entry["temperature"] if "temperature" in entry else models["data"][0]["temperature"])


def picture(name: str, alt: str) -> str:
    return (f'<picture><source media="(prefers-color-scheme: dark)" srcset="{name}-dark.png">'
            f'<img src="{name}.png" alt="{alt}" width="100%"></picture>')  # fmt: skip


def write_readme(out: Path, report: dict[str, Any], tests: list[tuple[str, str, dict[str, float]]],
                 pooled: dict[str, float], eces: dict[str, float]) -> None:  # fmt: skip
    res, pair = report["results"], report["paired_accuracy"][f"{DEN} vs {KEV}"]
    panels = list(report["files"])
    wins = sum(pair[p]["low"] > 0 for p in panels)
    losses = sum(pair[p]["high"] < 0 for p in panels)
    servers = report["servers"]
    kev_model = servers[KEV]["models"]["models"][0]
    den_model = servers[DEN]["models"]["models"][0]
    kev_run = kev_model.get("run", kev_model["name"])
    lines = [
        "# Benchmarks: den v1 vs Kev-4B",
        "",
        "Both are 4B System One models on `Qwen/Qwen3.5-4B-Base`, LoRA rank 16, trained on the same data: "
        "Kev's four-stage recipe (`core` → `dates` → `documents` → `skills`+`devtools`) and nothing else. "
        "What differs is the model: "
        "den's set head (span pooling, cross-option attention, per-option prior) and prompt layout against Kev's "
        "pointer head and template. Every panel below is held out from both.",
        "",
        f"**On dev, pooled over {pooled['questions']:,.0f} questions, den v1 is {100 * pooled['diff']:+.1f} points "
        f"[{100 * pooled['low']:+.1f}, {100 * pooled['high']:+.1f}] against Kev-4B.** It is ahead with a 95% interval "
        f"clear of zero on {wins} of {len(panels)} panels and behind on {losses}.",
        "",
        picture("accuracy", "Accuracy per dev panel, den v1 and Kev-4B"),
        "",
        picture("delta", "Paired accuracy difference per panel with 95% intervals"),
        "",
        "## Dev panels (measured here)",
        "",
        "Each model ran behind its own server (`den serve`; Kev's `kev.serve` from "
        "[jaredpalmer/kev](https://github.com/jaredpalmer/kev)) on the same machine "
        f"({report['machine']}, MLX, bf16), at its own shipped temperature (den T = "
        f"{temperature(servers[DEN]['models']):.2f}, Kev T = {temperature(servers[KEV]['models']):.2f}). "
        "`scripts/run_benchmarks.py` sent every record of each pinned dev file under `data/clean/` to both servers, "
        "and `den/metrics.py` scored both. Accuracy counts every scored question, including Kev's permuted "
        "and contrastive variants. Unknowable items are counted apart (last column).",
        "",
        "| Panel | Questions | den acc | Kev acc | Δ pts [95% CI] | den NLL / Brier / ECE | Kev NLL / Brier / ECE "
        "| Coverage at ≤5% error, den / Kev |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for p in panels:
        d, k, x = res[DEN][p], res[KEV][p], pair[p]
        label = PANELS[short(p)][0]
        lines.append(
            f"| {label} (`{p}`) | {d['questions']:,} | {pct(d['accuracy'])} | {pct(k['accuracy'])} | "
            f"{100 * x['diff']:+.1f} [{100 * x['low']:+.1f}, {100 * x['high']:+.1f}] | "
            f"{num(d['nll'])} / {num(d['brier'])} / {pct(d['ece'])} | "
            f"{num(k['nll'])} / {num(k['brier'])} / {pct(k['ece'])} | "
            f"{pct(d['coverage_at_5%_error'])} / {pct(k['coverage_at_5%_error'])} |"
        )
    lines += [
        f"| **All dev panels, pooled** | {pooled['questions']:,.0f} | | | **{100 * pooled['diff']:+.1f} "
        f"[{100 * pooled['low']:+.1f}, {100 * pooled['high']:+.1f}]** | ECE {pct(eces[DEN])} | "
        f"ECE {pct(eces[KEV])} | |",
        "",
        "Δ is den minus Kev over the same questions. The interval resamples whole records, so questions that share a "
        "state move together (`metrics.paired_bootstrap`). Coverage is the share of questions that can be answered, "
        "most confident first, while keeping the error rate at 5% or below.",
        "",
        "### Kev's clean slices and robustness checks",
        "",
        "Kev's cards report suites with variants on their clean rows only. These columns use the same rows "
        "(`metrics.robustness`). A permutation flip is a changed answer after the options were reordered (lower is "
        "better). A contrastive pair is two states that differ only in the deciding fact, and it counts when both "
        "siblings are right (higher is better). Unknowable items have the deciding sentence removed, so a confident "
        "answer is a guess (lower is better; the mean top probability is in brackets). Kev's transfer and probes "
        "suites share the same 36 permuted rows and 64 contrastive pairs, so those columns repeat across the two.",
        "",
        "| Panel | Clean questions | den clean acc | Kev clean acc | den / Kev permutation flips | "
        "den / Kev contrastive pairs both right | den / Kev unknowable answered at p ≥ 0.9 |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for p in panels:
        d, k = res[DEN][p].get("robustness", {}), res[KEV][p].get("robustness", {})
        dc, kc = d.get("clean"), k.get("clean")
        perm = [r.get("permutation", {}).get("flip_rate") for r in (d, k)]
        flip = [r.get("paired_flip", {}).get("both_correct_rate") for r in (d, k)]
        unk = [r.get("unknowable") for r in (d, k)]
        guess = [f"{pct(u['share_at_0_9'])} ({u['mean_max_p']:.2f})" if u else "-" for u in unk]
        if not any((dc, perm[0], flip[0], unk[0])):
            continue
        lines.append(
            f"| {PANELS[short(p)][0]} | {dc['questions'] if dc else '-'} | {pct(dc and dc['accuracy'])} | "
            f"{pct(kc and kc['accuracy'])} | {pct(perm[0])} / {pct(perm[1])} | {pct(flip[0])} / {pct(flip[1])} | "
            f"{guess[0]} / {guess[1]} |"
        )
    lines += [
        "",
        "## Calibration",
        "",
        picture("calibration", "Reliability diagram, every dev panel pooled"),
        "",
        "Both models ship one fitted temperature. A well-calibrated model's 80% answers are right about 80% of the "
        'time, which is what makes thresholds such as "automate above 0.9, otherwise send to a person" safe.',
        "",
        "## Test (each model's own release numbers)",
        "",
        "Test partitions are read once per release candidate (hard rule 1 in `CLAUDE.md`), so they were **not** re-run "
        "here. den's numbers come from `releases/v1.json` (CUDA bf16, den's evaluation). Kev's come from its model "
        "card (fp32, Kev's evaluation), copied with their sources into `kev-4b-card.json`. The rows and question "
        "counts match: where Kev reports clean rows, den's clean slice of the same file is shown. They were measured "
        "by different code on different hardware, so no interval is given.",
        "",
        picture("test", "Test accuracy from each model's own release record"),
        "",
        "| Panel | Questions (den / Kev) | den v1 | Kev-4B | Difference |",
        "|---|---|---:|---:|---:|",
    ]
    for label, counts, v in tests:
        lines.append(f"| {label} | {counts} | {pct(v[DEN])} | {pct(v[KEV])} | {100 * (v[DEN] - v[KEV]):+.1f} pts |")
    lines += [
        "",
        "## Speed (this machine, one request at a time)",
        "",
        "Median latency per request as the client saw it, over the same records. Kev's server caches the state, "
        "so the questions after the first reuse it. den reads each question with the state alone, as it was trained.",
        "",
        "| Panel | den p50 / p95 ms | Kev p50 / p95 ms |",
        "|---|---:|---:|",
    ]
    for p in panels:
        d, k = res[DEN][p]["latency_ms"], res[KEV][p]["latency_ms"]
        lines.append(f"| {PANELS[short(p)][0]} | {d['request_p50']:.0f} / {d['request_p95']:.0f} | "
                     f"{k['request_p50']:.0f} / {k['request_p95']:.0f} |")  # fmt: skip
    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run den serve --run hf:ramput-labs/den-qwen3.5-4b@v1                                    # :8009",
        "git clone https://github.com/jaredpalmer/kev && cd kev",
        "git checkout fe64b1274ea7f80d4095866df90666abb03e9cf6",
        f"uv sync --extra serve && uv run --extra serve python -m kev.serve --run {kev_run}   # :8008",
        "uv run python -m scripts.run_benchmarks --model kev-4b=http://127.0.0.1:8008 "
        "--model den-v1=http://127.0.0.1:8009 --baseline kev-4b",
        "uv run python -m scripts.plot_benchmarks",
        "```",
        "",
        f"Measured {report['created']} at commit `{report['commit']}`. den served `{den_model['name']}` "
        f"({den_model['description']}). Kev served `{kev_run}`. `results.json` holds every metric, the "
        "sha256 of every scored file and both servers' `/v1/models`. `predictions/` holds each question's "
        "probabilities from both models.",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="plot_benchmarks", description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    report = json.loads((args.out / "results.json").read_text(encoding="utf-8"))
    card = json.loads((args.out / "kev-4b-card.json").read_text(encoding="utf-8"))
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    res, pair = report["results"], report["paired_accuracy"][f"{DEN} vs {KEV}"]
    panels = list(report["files"])

    dev_rows = [(*PANELS[short(x)], {m: res[m][x]["accuracy"] for m in (DEN, KEV)}) for x in panels]
    delta_rows = [(PANELS[short(x)][0], pair[x]) for x in panels] + [("All panels, pooled", pair["pooled"])]
    curves = {m: pooled_curve(res[m]) for m in (DEN, KEV)}
    eces = {m: curve_ece(curves[m]) for m in (DEN, KEV)}
    tests: list[tuple[str, str, dict[str, float]]] = []
    for file, kev in card["test"].items():
        den = release["test"][file]
        if kev["slice"] == "clean":
            den = den["robustness"]["clean"]
        label = PANELS[short(file)][0] + (" (clean)" if kev["slice"] == "clean" else "")
        tests.append(
            (label, f"{den['questions']:,} / {kev['questions']:,}", {DEN: den["accuracy"], KEV: kev["accuracy"]})
        )

    for theme in (LIGHT, DARK):
        suffix = "" if theme is LIGHT else "-dark"
        bars(theme, dev_rows, "den v1 vs Kev-4B: accuracy on held-out dev panels",
             "Same rows, same machine, each model through its own server at its own temperature",
             f"Dev partitions under data/clean, never trained on by either model. {report['machine']}, MLX bf16. "
             "Every scored question, variants included.",
             args.out / f"accuracy{suffix}.png")  # fmt: skip
        deltas(theme, delta_rows, args.out / f"delta{suffix}.png")
        reliability(theme, curves, eces, args.out / f"calibration{suffix}.png")
        bars(theme, [(label, f"{counts} questions", v) for label, counts, v in tests],
             "Test: each model's own release numbers",
             "Read once per release: den from releases/v1.json, Kev from its model card (kev-4b-card.json)",
             "Not re-run here (test is read once per release candidate); measured by different code and hardware, "
             "so no paired interval.",
             args.out / f"test{suffix}.png")  # fmt: skip
    write_readme(args.out, report, tests, pair["pooled"], eces)
    print(f"wrote {args.out}/README.md and {len(list(args.out.glob('*.png')))} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
