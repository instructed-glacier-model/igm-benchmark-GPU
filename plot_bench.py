#!/usr/bin/env python3
"""
Compare GPUs: ice-flow time per step and TensorFlow memory peak vs grid spacing, one color per GPU, and a
table at one grid spacing.

  ./plot_bench.py                                   # all results/*/ -> figures/bench_compare.png
  ./plot_bench.py results/rtx_4090 results/my_gpu --dx 400 -o cmp.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CASES = ["online_f32", "offline_f32"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def short(name: str) -> str:
    return re.sub(r"\s+(Workstation|Server) Edition$", "", name)


def name(label: str, meta: dict) -> str:
    n = short(meta.get("gpu_name", label))
    return f"{n}, {meta['variant']}" if meta.get("variant") else n


def load(d: Path):
    rows = list(csv.DictReader((d / "results.csv").open()))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    try:
        meta = json.loads((d / "meta.json").read_text())
    except Exception:
        meta = {}
    return rows, meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="*", type=Path, help="results/<label> folders (default: all)")
    p.add_argument("-o", "--out", type=Path, default=HERE / "figures" / "bench_compare.png")
    p.add_argument("--dx", type=float, default=200.0, help="grid spacing of the printed table (m)")
    args = p.parse_args()
    dirs = args.dirs or sorted(d for d in (HERE / "results").iterdir()
                               if (d / "results.csv").exists() and not d.name.endswith("_quick"))
    if not dirs:
        raise SystemExit("no results/*/results.csv found")
    if len(dirs) > len(COLORS):
        raise SystemExit(f"at most {len(COLORS)} GPUs per figure; pass a subset of folders")
    data = [(d.name, *load(d)) for d in dirs]
    dx_ticks = sorted({r["dx_m"] for _, rows, _ in data for r in rows if r["status"] == "ok"}, reverse=True)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK})
    fig, axes = plt.subplots(2, len(CASES), figsize=(8, 6.2), sharex=True, sharey="row", constrained_layout=True)
    for j, case in enumerate(CASES):
        for i, (key, ylab, scale) in enumerate([("iceflow_s", "ice-flow time per step [ms]", 1e3),
                                                 ("tf_peak_gib", "GPU memory peak (TF) [GiB]", 1.0)]):
            ax = axes[i, j]
            for c, (label, rows, meta) in zip(COLORS, data):
                pts = sorted((r["dx_m"], r[key] * scale) for r in rows
                             if r["case"] == case and r["status"] == "ok" and r[key] == r[key])
                if pts:
                    ax.plot(*zip(*pts), "-o", color=c, lw=2, ms=6, mec="white", mew=1.5,
                            label=name(label, meta) if j == 0 and i == 0 else None)
            ax.set_xscale("log")
            ax.set_xticks(dx_ticks, [f"{d:g}" for d in dx_ticks])
            ax.minorticks_off()
            ax.set_xlim(dx_ticks[0] * 1.2, dx_ticks[-1] / 1.2)      # coarse left, fine right
            ax.set_yscale("log")
            ax.grid(True, which="major", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(case.replace("_", "  "), fontweight="bold")
            if i == 1:
                ax.set_xlabel("grid spacing dx [m]")
            if j == 0:
                ax.set_ylabel(ylab)
    fig.legend(loc="outside upper center", ncol=min(3, len(data)), frameon=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    print(f"\nat dx = {args.dx:.0f} m: ice-flow time per step [ms] / TF memory peak [GiB]")
    print(f"{'GPU':<34}" + "".join(f"{c:>22}" for c in CASES))
    for label, rows, meta in data:
        cells = []
        for case in CASES:
            r = next((r for r in rows if r["case"] == case and r["dx_m"] == args.dx), None)
            cells.append("-" if r is None else r["status"] if r["status"] != "ok"
                         else f"{r['iceflow_s'] * 1e3:.1f} / {r['tf_peak_gib']:.2f}")
        print(f"{name(label, meta)[:33]:<34}" + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
