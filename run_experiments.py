#!/usr/bin/env python
"""Run the TL-FT-Pareto-DRLA experiment suite.

Examples
--------
Quick plumbing check (a few minutes, numbers are meaningless)::

    python run_experiments.py --budget smoke --scenarios BASE_N8

Full paper protocol, all 13 scenarios, in one process::

    python run_experiments.py --budget paper

Full paper protocol split into five independent parts (run them in parallel,
e.g. in five terminals, all with the same --output folder)::

    python run_experiments.py --budget paper --part A
    ...
    python run_experiments.py --budget paper --part E

Progress and final Excel report::

    python run_experiments.py --budget paper --status
    python run_experiments.py --budget paper --report

Every finished training/evaluation unit is written to disk immediately, so an
interrupted run continues where it stopped when the same command is repeated.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Five independent parts of the paper run (scenario ids; None = all seeds of the budget).
PARTS = {
    "A": (["BASE_N8", "WL_BOOKINFO", "WL_SPARSE30", "WL_TOPOLOGY", "WL_HEAVY"], None),
    "B": (["SC_N16_HOM", "SC_N16_HET", "CP_0p05", "CP_0p10", "CP_0p15", "CP_1p00"], None),
    "C": (["SC_N32_HOM"], "first"),
    "D": (["SC_N32_HET"], "first"),
    "E": (["SC_N32_HOM", "SC_N32_HET"], "rest"),
}


def _split_seeds(seeds, which):
    k = math.ceil(len(seeds) * 0.6)
    return list(seeds[:k]) if which == "first" else list(seeds[k:])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TL-FT-Pareto-DRLA experiment suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-dir", default=str(HERE / "data"), help="folder with the .xlsx workload files")
    p.add_argument("--output", default=str(HERE / "results"), help="output folder (shared by all parts)")
    p.add_argument(
        "--budget",
        default="paper",
        choices=["smoke", "pilot", "reduced", "paper"],
        help="smoke: plumbing test; pilot: development budget; paper: protocol of the paper",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--part", choices=sorted(PARTS), help="run one of the five parts of the paper run")
    g.add_argument(
        "--scenarios",
        default="",
        help="comma-separated scenario ids or groups "
        "(base, workload, scalability, capacity); empty = all 13 scenarios",
    )
    p.add_argument(
        "--seeds", default="", help="comma-separated training seeds (default: all seeds of the budget)"
    )
    p.add_argument("--report", action="store_true", help="only build the Excel report from existing results")
    p.add_argument("--status", action="store_true", help="only print the progress of the run")
    p.add_argument("--threads", type=int, default=1, help="PyTorch threads per process (default 1)")
    p.add_argument("--no-ablations", action="store_true", help="skip the ablation variants in BASE_N8")
    p.add_argument(
        "--no-specialists",
        action="store_true",
        help="skip the eight additional fixed-preference specialists in BASE_N8",
    )
    return p


def status(M, out_root: Path, budget: str, scenarios) -> None:
    """Count finished evaluation units per scenario."""
    b = M.BUDGETS[budget]
    root = out_root / f"results_{budget}" / "markers"
    done = {p.stem for p in root.glob("*__eval.json")} if root.exists() else set()
    learned = [s["method"] for s in M._suite_method_specs("v3", False)]
    others = (["xu_modrl"] if b.include_xu else []) + list(b.heuristics)
    print(f"\nProgress ({out_root / f'results_{budget}'})")
    total_all = done_all = 0
    for sc in scenarios:
        expected = [f"{sc.scenario_id}__{m}__seed{s}__eval" for s in b.seeds for m in learned + others]
        n = sum(1 for k in expected if k in done)
        total_all += len(expected)
        done_all += n
        bar = "#" * int(20 * n / max(1, len(expected)))
        print(f"  {sc.scenario_id:<12s} [{bar:<20s}] {n:>3d}/{len(expected)}")
    print(
        f"  {'total':<12s} {done_all}/{total_all} main evaluations "
        "(ablations, specialists and lookahead variants are not counted)\n"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["TLFT_TORCH_THREADS"] = str(max(1, args.threads))
    sys.path.insert(0, str(HERE))
    import tl_ft_pareto_drla as M

    src = M._suite_find_file(args.data_dir, M.SUITE_SOURCE)
    tgt = M._suite_find_file(args.data_dir, M.SUITE_TARGET)
    cfg = M.base_config(src, tgt, args.output)
    all_sc = M.default_scenarios(args.data_dir)

    if args.part:
        scen, seed_rule = PARTS[args.part]
        seeds = _split_seeds(M.BUDGETS[args.budget].seeds, seed_rule) if seed_rule else None
        if seeds is not None and not seeds:
            print(f"Part {args.part}: nothing to do for budget '{args.budget}'.")
            return 0
    else:
        scen = [s.strip() for s in args.scenarios.split(",") if s.strip()]
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()] or None
    selected = [s for s in all_sc if not scen or s.scenario_id in scen or s.group in scen]

    if args.status:
        status(M, Path(args.output), args.budget, selected)
        return 0
    t0 = time.time()
    root = M.run_suite(
        cfg,
        budget=args.budget,
        input_dir=args.data_dir,
        scenarios=scen,
        seeds=seeds,
        excel_only=args.report,
        method_set="v3",
        ablation_scenarios=[] if args.no_ablations else ["BASE_N8"],
        specialist_scenarios=[] if args.no_specialists else ["BASE_N8"],
        make_report=args.report or not args.part,
    )
    if args.part:
        print(
            f"\nPart {args.part} finished in {(time.time() - t0) / 3600:.2f} h. "
            "When all parts are finished, build the report with --report."
        )
    print(f"Results: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
