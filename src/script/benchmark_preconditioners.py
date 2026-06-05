#!/usr/bin/env python3
"""Build, run, and summarize the heterogeneous diffusion benchmark."""

from __future__ import annotations

import csv
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
PLOTS = RESULT / "plots"
EXECUTABLE = ROOT / "build" / "elliptic"
SETUP = ROOT / "setup_modules.sh"
CSV_FILE = RESULT / "benchmark_results.csv"

MPI_PROCS = [1, 2, 4]
PRECONDITIONERS = ["none", "jacobi", "ssor", "ilu", "amg"]

RESULT_RE = re.compile(
    r"^result\s+(\S+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)"
)


def run_shell(command: str, log_name: str, cwd: Path = ROOT) -> int:
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (RESULT / log_name).write_text(completed.stdout, encoding="utf-8")
    return completed.returncode


def shell_output(command: str) -> str:
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout.strip()


def build_code() -> int:
    return run_shell(f"source {SETUP} && cmake --build build", "build.log")


def mpi_launcher() -> str:
    return shell_output(
        f"source {SETUP} && "
        "for cmd in mpirun mpiexec srun; do command -v $cmd && break; done"
    )


def run_experiment(np: int, launcher: str) -> int:
    if np == 1:
        command = f"source {SETUP} && {EXECUTABLE}"
    elif launcher:
        flag = "-n" if Path(launcher).name == "srun" else "-np"
        command = f"source {SETUP} && {launcher} {flag} {np} {EXECUTABLE}"
    else:
        (RESULT / f"run_np{np}.log").write_text(
            "No MPI launcher found after sourcing setup_modules.sh.\n",
            encoding="utf-8",
        )
        return 127

    return run_shell(command, f"run_np{np}.log", RESULT)


def parse_run_log(np: int, return_code: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current = {
        "status": "ok",
        "returncode": str(return_code),
        "mpi_procs": str(np),
        "p": "",
        "refinements": "",
        "balls": "",
        "cells": "",
        "dofs": "",
        "case_setup_s": "",
        "assembly_s": "",
    }

    for line in (RESULT / f"run_np{np}.log").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("mpi_procs "):
            current["mpi_procs"] = line.split()[1]
        elif line.startswith("p "):
            current["p"] = line.split()[1]
        elif line.startswith("refinements "):
            current["refinements"] = line.split()[1]
        elif line.startswith("balls "):
            current["balls"] = line.split()[1]
        elif line.startswith("cells "):
            current["cells"] = line.split()[1]
        elif line.startswith("dofs "):
            current["dofs"] = line.split()[1]
        elif line.startswith("case_setup "):
            current["case_setup_s"] = line.split()[1]
        elif line.startswith("assembly "):
            current["assembly_s"] = line.split()[1]
        elif match := RESULT_RE.match(line):
            rows.append(
                current
                | {
                    "preconditioner": match.group(1),
                    "iterations": match.group(2),
                    "pc_setup_s": match.group(3),
                    "solve_s": match.group(4),
                    "total_s": match.group(5),
                }
            )

    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    columns = [
        "status",
        "returncode",
        "mpi_procs",
        "p",
        "refinements",
        "balls",
        "cells",
        "dofs",
        "case_setup_s",
        "assembly_s",
        "preconditioner",
        "iterations",
        "pc_setup_s",
        "solve_s",
        "total_s",
    ]
    with CSV_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def row_for(rows: list[dict[str, str]], np: str, p: str, ref: str, prec: str):
    for row in rows:
        if (
            row["mpi_procs"] == np
            and row["p"] == p
            and row["refinements"] == ref
            and row["preconditioner"] == prec
        ):
            return row
    return None


def svg_start(title: str, width: int = 860, height: int = 520) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#1f2933}.title{font-size:20px;font-weight:700}.grid{stroke:#d9e2ec}.axis{stroke:#243b53;stroke-width:1.3}.label{fill:#52606d;font-size:12px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text class="title" x="70" y="34">{title}</text>',
    ]


def write_svg(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines + ["</svg>\n"]), encoding="utf-8")
    return str(path.relative_to(RESULT))


def x_pos(value: float, xmin: float, xmax: float, left: int, right: int) -> float:
    return left + (value - xmin) * (right - left) / (xmax - xmin)


def y_pos(value: float, ymin: float, ymax: float, top: int, bottom: int, log: bool) -> float:
    if log:
        value = math.log10(max(value, 1e-12))
        ymin = math.log10(max(ymin, 1e-12))
        ymax = math.log10(max(ymax, 1e-12))
    return bottom - (value - ymin) * (bottom - top) / (ymax - ymin)


def draw_axes(lines: list[str], ylabel: str, left: int, right: int, top: int, bottom: int) -> None:
    for i in range(5):
        y = top + i * (bottom - top) / 4
        lines.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    lines.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left}" y2="{top}"/>')
    lines.append(f'<text class="label" transform="translate(24 310) rotate(-90)">{ylabel}</text>')


def draw_y_labels(
    lines: list[str],
    labels: list[tuple[str, float]],
    ymin: float,
    ymax: float,
    top: int,
    bottom: int,
    log: bool,
) -> None:
    for text, value in labels:
        y = y_pos(value, ymin, ymax, top, bottom, log)
        lines.append(f'<text class="label" x="42" y="{y + 4:.1f}">{text}</text>')


def plot_iterations_vs_p(rows: list[dict[str, str]]) -> str:
    path = PLOTS / "iterations_vs_heterogeneity_ref5_np1.svg"
    left, right, top, bottom = 80, 650, 60, 430
    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    lines = svg_start("Iterations vs heterogeneity, ref=5, np=1")
    draw_axes(lines, "CG iterations, log scale", left, right, top, bottom)
    draw_y_labels(
        lines,
        [("1", 1), ("10", 10), ("100", 100), ("1k", 1000), ("10k", 10000)],
        1,
        20000,
        top,
        bottom,
        True,
    )

    for index, prec in enumerate(PRECONDITIONERS):
        points = []
        for p_value in ["1", "3", "5"]:
            row = row_for(rows, "1", p_value, "5", prec)
            if row:
                x = x_pos(float(p_value), 1, 5, left, right)
                y = y_pos(float(row["iterations"]), 1, 20000, top, bottom, True)
                points.append((x, y))
        lines.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
            f'fill="none" stroke="{colors[index]}" stroke-width="2.5"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[index]}"/>')
        lines.append(f'<text x="690" y="{78 + 24 * index}" fill="{colors[index]}">{prec}</text>')

    for p_value in [1, 3, 5]:
        x = x_pos(p_value, 1, 5, left, right)
        lines.append(f'<text class="label" x="{x-5:.1f}" y="454">{p_value}</text>')
    lines.append('<text class="label" x="335" y="492">heterogeneity exponent p</text>')
    return write_svg(path, lines)


def plot_total_time(rows: list[dict[str, str]]) -> str:
    path = PLOTS / "total_time_p5_ref5_np1.svg"
    left, right, top, bottom = 80, 760, 60, 430
    lines = svg_start("Total time by preconditioner, p=5, ref=5, np=1")
    draw_axes(lines, "seconds, log scale", left, right, top, bottom)
    draw_y_labels(
        lines,
        [("0.05", 0.05), ("0.1", 0.1), ("1", 1), ("5", 5)],
        0.05,
        8.0,
        top,
        bottom,
        True,
    )
    slot = (right - left) / len(PRECONDITIONERS)
    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#ea580c"]

    for index, prec in enumerate(PRECONDITIONERS):
        row = row_for(rows, "1", "5", "5", prec)
        if not row:
            continue
        value = float(row["total_s"])
        x = left + slot * index + slot * 0.22
        y = y_pos(value, 0.05, 8.0, top, bottom, True)
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.56:.1f}" height="{bottom-y:.1f}" fill="{colors[index]}"/>')
        lines.append(f'<text class="label" x="{x:.1f}" y="454">{prec}</text>')
        lines.append(f'<text class="label" x="{x:.1f}" y="{y-8:.1f}">{value:.3g}s</text>')
    return write_svg(path, lines)


def plot_strong_scaling(rows: list[dict[str, str]]) -> str:
    path = PLOTS / "strong_scaling_p5_ref5.svg"
    left, right, top, bottom = 80, 650, 60, 430
    colors = ["#1d4ed8", "#dc2626", "#059669", "#7c3aed", "#ea580c"]
    lines = svg_start("Strong scaling, p=5, ref=5")
    draw_axes(lines, "speedup", left, right, top, bottom)
    draw_y_labels(
        lines,
        [("0", 0), ("1", 1), ("2", 2), ("3", 3), ("4", 4)],
        0,
        4,
        top,
        bottom,
        False,
    )
    ideal_points = [
        (x_pos(float(np), 1, 4, left, right), y_pos(float(np), 0, 4, top, bottom, False))
        for np in [1, 2, 4]
    ]
    lines.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in ideal_points)}" '
        f'fill="none" stroke="#64748b" stroke-width="2" stroke-dasharray="6,5"/>'
    )
    lines.append('<text x="690" y="198" fill="#64748b">ideal</text>')

    for index, prec in enumerate(PRECONDITIONERS):
        serial = row_for(rows, "1", "5", "5", prec)
        if not serial:
            continue
        serial_time = float(serial["total_s"])
        points = []
        for np in ["1", "2", "4"]:
            row = row_for(rows, np, "5", "5", prec)
            if row:
                speedup = serial_time / float(row["total_s"])
                points.append((x_pos(float(np), 1, 4, left, right), y_pos(speedup, 0, 4, top, bottom, False)))
        lines.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
            f'fill="none" stroke="{colors[index]}" stroke-width="2.5"/>'
        )
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[index]}"/>')
        lines.append(f'<text x="690" y="{78 + 24 * index}" fill="{colors[index]}">{prec}</text>')

    for np in [1, 2, 4]:
        x = x_pos(np, 1, 4, left, right)
        lines.append(f'<text class="label" x="{x-5:.1f}" y="454">{np}</text>')
    lines.append('<text class="label" x="335" y="492">MPI processes</text>')
    return write_svg(path, lines)


def plot_setup_solve(rows: list[dict[str, str]]) -> str:
    path = PLOTS / "setup_solve_p5_ref5_np1.svg"
    left, right, top, bottom = 80, 720, 60, 430
    lines = svg_start("Setup and solve split, preconditioned methods")
    draw_axes(lines, "seconds", left, right, top, bottom)
    y_max = 0.28
    draw_y_labels(lines, [("0", 0), ("0.07", 0.07), ("0.14", 0.14), ("0.21", 0.21), ("0.28", 0.28)], 0, y_max, top, bottom, False)
    methods = ["jacobi", "ssor", "ilu", "amg"]
    slot = (right - left) / len(methods)
    max_total = 0.28

    for index, prec in enumerate(methods):
        row = row_for(rows, "1", "5", "5", prec)
        if not row:
            continue
        setup = float(row["pc_setup_s"])
        solve = float(row["solve_s"])
        total = setup + solve
        setup_h = setup * (bottom - top) / max_total
        solve_h = solve * (bottom - top) / max_total
        x = left + slot * index + slot * 0.25
        width = slot * 0.50
        lines.append(f'<rect x="{x:.1f}" y="{bottom-solve_h:.1f}" width="{width:.1f}" height="{solve_h:.1f}" fill="#1d4ed8"/>')
        lines.append(f'<rect x="{x:.1f}" y="{bottom-solve_h-setup_h:.1f}" width="{width:.1f}" height="{setup_h:.1f}" fill="#ea580c"/>')
        lines.append(f'<text class="label" x="{x:.1f}" y="454">{prec}</text>')
        lines.append(f'<text class="label" x="{x:.1f}" y="{bottom-solve_h-setup_h-8:.1f}">{total:.3g}s</text>')

    lines.append('<rect x="750" y="72" width="14" height="14" fill="#ea580c"/><text x="772" y="84">setup</text>')
    lines.append('<rect x="750" y="98" width="14" height="14" fill="#1d4ed8"/><text x="772" y="110">solve</text>')
    return write_svg(path, lines)


def write_plots(rows: list[dict[str, str]]) -> list[str]:
    PLOTS.mkdir(parents=True, exist_ok=True)
    if not rows:
        return []
    return [
        plot_iterations_vs_p(rows),
        plot_total_time(rows),
        plot_strong_scaling(rows),
        plot_setup_solve(rows),
    ]


def main() -> int:
    RESULT.mkdir(parents=True, exist_ok=True)
    build_status = build_code()
    launcher = mpi_launcher()
    run_codes: dict[int, int] = {}
    rows: list[dict[str, str]] = []

    if build_status == 0:
        for np in MPI_PROCS:
            run_codes[np] = run_experiment(np, launcher)
            rows.extend(parse_run_log(np, run_codes[np]))

    write_csv(rows)
    plots = write_plots(rows)

    print(f"wrote {CSV_FILE}")
    for plot in plots:
        print(f"wrote {RESULT / plot}")

    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
