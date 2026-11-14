"""Additional experiments built on top of a finished paper run.

Experiment 1 -- contribution of preference conditioning (scenario BASE_N8):
    full_v3_ref                     reported method (checkpoint of the main run, re-evaluated)
    abl_rec_uniform                 same checkpoint; recovery ranks destinations with fixed
                                    uniform weights instead of the operator preference p
    abl_rec_restore_only            same checkpoint; recovery ranks destinations only by the
                                    projected restore downtime ("fastest restore first")
    abl_critic_no_pref              critic without preference input (constant input); p is still
                                    used for scalarisation, lookahead and recovery; retrained with
                                    the identical source -> transfer -> target pipeline
    abl_critic_no_pref_rec_uniform  both changes

Experiment 2 -- zero-shot transfer across infrastructure sizes:
    zs_v3                the 8-node BASE_N8 policies applied without any training to the 16- and
                         32-node scenarios (same test traces as the main run)
    zs_v3_no_lookahead   the same without the value-guided lookahead
    fs_v3                optional few-shot fine-tuning (--fewshot > 0)

The main results folder is only read; new results are written to
    <output>/results_<budget>/extra_experiments/
Every finished unit writes a marker, so an interrupted part continues when restarted.

Usage::

    python extra_experiments.py --part A        # experiment 1
    python extra_experiments.py --part B        # zero-shot, 16 nodes (hom. + het.)
    python extra_experiments.py --part C        # zero-shot, 32 nodes hom.
    python extra_experiments.py --part D        # zero-shot, 32 nodes het.
    python extra_experiments.py --status
    python extra_experiments.py --report        # Excel workbook, summary and figures
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Defaults (overridden by the command-line arguments)
MAIN_RESULTS_ROOT = str(HERE / "results")  # --output of run_experiments.py
INPUT_DIR = str(HERE / "data")
BUDGET = "paper"
ZS_PARETO = True  # zero-shot: also run the 60-preference sweep (hypervolume, steerability)
FEWSHOT_FRACTION = 0.0  # > 0: additionally fine-tune the 8-node policy with this fraction of the episodes
SEEDS = None  # None = all seeds of the budget
os.environ.setdefault("TLFT_TORCH_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import tl_ft_pareto_drla as M  # noqa: E402

EXTRA_VERSION = "extra-experiments-1.0"
ABL_SCENARIO = "BASE_N8"
ZS_SCENARIOS_ALL = ("SC_N16_HOM", "SC_N16_HET", "SC_N32_HOM", "SC_N32_HET")
PARTS = {
    "A": ("ablation", (ABL_SCENARIO,)),
    "B": ("zeroshot", ("SC_N16_HOM", "SC_N16_HET")),
    "C": ("zeroshot", ("SC_N32_HOM",)),
    "D": ("zeroshot", ("SC_N32_HET",)),
}
ABL_METHODS = [
    "full_v3_ref",
    "abl_rec_uniform",
    "abl_rec_restore_only",
    "abl_critic_no_pref",
    "abl_critic_no_pref_rec_uniform",
]
ABL_LABEL = {
    "full_v3_ref": "Full method (reported)",
    "abl_rec_uniform": "Recovery with fixed uniform ranking",
    "abl_rec_restore_only": "Recovery with restore-time-only ranking",
    "abl_critic_no_pref": "Critic without preference conditioning",
    "abl_critic_no_pref_rec_uniform": "No preference in critic and recovery",
}
ABL_SHORT = {
    "full_v3_ref": "Full\nmethod",
    "abl_rec_uniform": "Recovery:\nuniform rank",
    "abl_rec_restore_only": "Recovery:\nrestore-only",
    "abl_critic_no_pref": "Critic\nwithout p",
    "abl_critic_no_pref_rec_uniform": "No p in critic\nor recovery",
}
ZS_LABEL = {
    "zs_v3": "Zero-shot from 8 nodes (no training)",
    "zs_v3_no_lookahead": "Zero-shot from 8 nodes, no lookahead",
    "fs_v3": "Few-shot from 8 nodes",
    "full_v3": "Proposed, trained on this scenario",
    "no_tl_v3": "No-TL, trained on this scenario",
    "xu_modrl": "Xu-MODRL (adapted)",
    "k8s": "K8s+mask",
    "greedy_latency": "Greedy min-latency",
    "greedy_load": "Greedy least-loaded",
    "random": "Random feasible",
}
ZS_COMPARE = ["full_v3", "no_tl_v3", "xu_modrl", "k8s", "greedy_latency", "greedy_load", "random"]
TRACE_METRICS = ["mean_objective", "latency_obj", "energy_obj", "load_obj", "availability_loss_obj"]
METRIC_LABEL = {
    "mean_objective": "Mean normalised objective",
    "latency_obj": "Delay objective",
    "energy_obj": "Energy objective",
    "load_obj": "Load-variance objective",
    "availability_loss_obj": "Resilience risk C_R",
    "fault_tolerance": "Fault tolerance",
    "valid_rate": "Valid rate",
    "hypervolume": "Hypervolume",
}


# -----------------------------------------------------------------------------
# A. preference-free critic (ablation option) and recovery-ranking modes
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class XOptions(M.MethodV2Options):
    pref_conditioning: bool = True

    def tag(self) -> str:
        t = super().tag()
        return t if self.pref_conditioning else t + "pc0"


_ORIG_ENCODE_T = M.FTNodeWiseHyperDQN.encode_t


def _encode_t_ablation(self, node_x, node_adj, ms_x, ms_adj, ctx, preference):
    if not getattr(self.v2, "pref_conditioning", True):
        preference = torch.full_like(preference, 1.0 / preference.shape[1])
    return _ORIG_ENCODE_T(self, node_x, node_adj, ms_x, ms_adj, ctx, preference)


M.FTNodeWiseHyperDQN.encode_t = _encode_t_ablation

ABL_NO_PREF = XOptions(**{**dataclasses.asdict(M.V3_FULL), "pref_conditioning": False})

_REC_MODE = "preference"  # "preference" (method) | "uniform" | "restore_only"
_ORIG_REC_SCORE = M.FaultTolerantDRLAEnv._recovery_candidate_score


def _recovery_score_ablation(self, tmp, instance_idx, old_node_idx, candidate):
    if _REC_MODE == "preference":
        return _ORIG_REC_SCORE(self, tmp, instance_idx, old_node_idx, candidate)
    self._recovery_fast_score_calls += 1
    c = self._candidate_components(tmp, instance_idx, old_node_idx, candidate, include_expected_risk=False)
    if _REC_MODE == "uniform":
        return float(0.25 * (c["latency"] + c["energy"] + c["load"] + c["resilience"]))
    if _REC_MODE == "restore_only":
        return float(c["resilience"] + 1e-6 * c["restore_time"])
    raise ValueError(_REC_MODE)


M.FaultTolerantDRLAEnv._recovery_candidate_score = _recovery_score_ablation


class recovery_mode:
    def __init__(self, mode: str):
        self.mode, self.prev = mode, None

    def __enter__(self):
        global _REC_MODE
        self.prev, _REC_MODE = _REC_MODE, self.mode

    def __exit__(self, *exc):
        global _REC_MODE
        _REC_MODE = self.prev
        return False


# -----------------------------------------------------------------------------
# B. paths, checks and shared helpers
# -----------------------------------------------------------------------------
def _budget():
    return M.BUDGETS[BUDGET]


def _seeds() -> List[int]:
    return [int(s) for s in (SEEDS if SEEDS else _budget().seeds)]


def main_paths() -> Dict[str, Path]:
    return M._suite_paths(Path(MAIN_RESULTS_ROOT) / f"results_{BUDGET}")


def extra_paths() -> Dict[str, Path]:
    main_root = Path(MAIN_RESULTS_ROOT) / f"results_{BUDGET}"
    root = main_root / "extra_experiments"
    p = {
        "root": root,
        "raw": root / "raw",
        "markers": root / "markers",
        "checkpoints": root / "checkpoints",
        "plots": root / "plots",
        "report": root / "report",
        "traces": main_root / "traces",
    }  # SAME locked trace banks as the main run
    for k, v in p.items():
        if k != "traces":
            v.mkdir(parents=True, exist_ok=True)
    return p


def check_main_run(scenarios: Sequence[str]) -> List[str]:
    """Return a list of problems (empty = everything needed from the main run exists)."""
    root = Path(MAIN_RESULTS_ROOT) / f"results_{BUDGET}"
    probs = []
    if not root.is_dir():
        return [
            f"Main results folder not found: {root}\n   -> --output must be the output folder "
            f"of run_experiments.py (it contains results_{BUDGET})."
        ]
    for sid in scenarios:
        if not (root / "traces" / sid / "target_test.csv").exists():
            probs.append(f"missing test traces of {sid}: {root / 'traces' / sid / 'target_test.csv'}")
    for s in _seeds():
        ck = root / "checkpoints" / f"BASE_N8__full_v3__seed{s}_best.pt"
        if not ck.exists():
            probs.append(f"missing 8-node checkpoint (will be RE-TRAINED, ~1 h): {ck.name}")
    return probs


def _scenario(sid: str) -> M.Scenario:
    for sc in M.default_scenarios(INPUT_DIR):
        if sc.scenario_id == sid:
            return sc
    raise KeyError(sid)


def _setup(sid: str):
    b = _budget()
    sc = _scenario(sid)
    mp = main_paths()
    base = M.base_config(
        M._suite_find_file(INPUT_DIR, M.SUITE_SOURCE),
        M._suite_find_file(INPUT_DIR, M.SUITE_TARGET),
        MAIN_RESULTS_ROOT,
    )
    cfg = M._suite_config(base, sc, b, mp["root"])
    cfg.validate()
    src, tgt = M.build_scenario_specs(sc)
    traces = M._suite_scenario_traces(mp, sc, cfg, b, src, tgt)
    return sc, cfg, b, src, tgt, traces


def _train_with_source(paths, sc, cfg, src, tgt, traces, seed: int, opts, method: str):
    """Identical pipeline to the main run: source pre-training -> transfer -> target training."""
    tag = opts.tag()
    skey = f"{M._suite_source_dir(src)}__source_{tag}__seed{seed}"
    with M.v2_method(opts):
        tkey = f"{sc.scenario_id}__{method}__seed{seed}"
        if (
            M._suite_done(paths, tkey + "__train") is not None
            and (paths["checkpoints"] / f"{tkey}_best.pt").exists()
        ):
            agent, info = M._suite_train_dqn(
                paths,
                tkey,
                tgt,
                cfg,
                seed,
                cfg.episodes,
                traces["validation"],
                label="full",
                selection_preferences=M.fixed_weight_bank(),
                training_traces=traces["training"],
            )
            return agent, info
        print(f"  [TRAIN] source pre-training ({method}) seed={seed}", flush=True)
        s_agent, s_info = M._suite_train_dqn(
            paths,
            skey,
            src,
            cfg,
            seed,
            cfg.source_episodes,
            traces["source_validation"],
            label="source_pretrain",
            selection_preferences=M.fixed_weight_bank(),
            training_traces=traces["source_training"],
        )
        tf = s_agent.extract_selective_transfer(opts.transfer_mode)
        del s_agent
        gc.collect()
        print(f"  [TRAIN] target training ({method}) seed={seed}", flush=True)
        agent, info = M._suite_train_dqn(
            paths,
            tkey,
            tgt,
            cfg,
            seed,
            cfg.episodes,
            traces["validation"],
            label="full",
            initial_transfer=tf,
            selection_preferences=M.fixed_weight_bank(),
            training_traces=traces["training"],
        )
        info = {**info, "source_runtime_sec": s_info.get("train_runtime_sec", np.nan)}
    return agent, info


def _base8_checkpoint(seed: int) -> Path:
    """8-node full_v3 checkpoint of the main run (re-trained into the main folder if missing)."""
    mp = main_paths()
    ck = mp["checkpoints"] / f"BASE_N8__full_v3__seed{seed}_best.pt"
    if ck.exists():
        return ck
    warnings.warn(f"8-node checkpoint missing, re-training it: {ck}")
    sc, cfg, b, src, tgt, traces = _setup("BASE_N8")
    agent, _info = _train_with_source(mp, sc, cfg, src, tgt, traces, seed, M.V3_FULL, "full_v3")
    del agent
    gc.collect()
    return ck


def _load_v3(spec, cfg, ckpt: Path, opts=None):
    with M.v2_method(opts or M.V3_FULL):
        return M._agent_from_best_checkpoint(spec, cfg, ckpt, use_gnn=True)


def _base8_train_info(seed: int) -> Dict:
    d = M._suite_done(main_paths(), f"BASE_N8__full_v3__seed{seed}__train") or {}
    return {
        "train_runtime_sec": d.get("train_runtime_sec", np.nan),
        "parameter_count": d.get("parameter_count", np.nan),
        "best_episode": d.get("best_episode", np.nan),
    }


# -----------------------------------------------------------------------------
# C. evaluation (same protocol as the main suite + trace-level sweep rows)
# -----------------------------------------------------------------------------
def _evaluate(
    paths,
    sc,
    cfg,
    b,
    tgt,
    traces,
    seed,
    method,
    label,
    agent,
    info,
    *,
    pareto: bool,
    fault: bool,
    extra: Optional[Dict] = None,
) -> None:
    key = f"{sc.scenario_id}__{method}__seed{seed}__eval"
    if M._suite_done(paths, key) is not None:
        print(f"  [SKIP] {method} seed={seed} already evaluated", flush=True)
        return
    print(
        f"  [EVAL] {method} seed={seed} (recovery ranking: {_REC_MODE}, "
        f"lookahead={int(getattr(agent, 'eval_lookahead', 0) or 0)}, sweep={pareto})",
        flush=True,
    )
    t_start = time.perf_counter()
    test = traces["test"]
    base = {
        "scenario_id": sc.scenario_id,
        "group": sc.group,
        "method": method,
        "algorithm": label,
        "training_seed": seed,
        "recovery_ranking": _REC_MODE,
        **(extra or {}),
    }
    ep = M.UNIFORM_PREFERENCE
    t0 = time.perf_counter()
    agg, raw = M.evaluate_agent_on_traces(agent, tgt, cfg, test, ep, "full", "reactive_migration")
    eval_sec = time.perf_counter() - t0
    row = {
        **base,
        **M._suite_metrics(agg),
        "nodes": len(tgt.nodes),
        "instances": len(tgt.instances),
        "train_runtime_sec": info.get("train_runtime_sec", np.nan),
        "source_runtime_sec": info.get("source_runtime_sec", np.nan),
        "best_episode": info.get("best_episode", np.nan),
        "parameter_count": info.get("parameter_count", np.nan),
        "eval_runtime_sec": eval_sec,
        "ms_per_decision_incl_env_recovery": eval_sec * 1000.0 / max(1, len(test) * len(tgt.instances)),
        "eval_lookahead": int(getattr(agent, "eval_lookahead", 0) or 0),
        "test_traces": len(test),
    }
    trace_rows = M._suite_trace_rows(raw, base)
    cand_rows, steer_rows, sweep_rows = [], [], []
    if pareto:
        prefs = M.make_preference_set(max(b.pareto_preference_count, 1), cfg.preference_seed)
        fdf, _s, ftr = M.preference_fidelity_audit_detailed(
            agent, tgt, cfg, test, "full", "reactive_migration", prefs
        )
        for _, r in fdf.iterrows():
            vec = [float(x) for x in str(r["objective_vector"]).split()]
            rec = {
                **base,
                "audit_kind": r["audit_kind"],
                "preference_id": r["preference_id"],
                "controlled_objective": r.get("controlled_objective", ""),
                "controlled_weight": r.get("controlled_weight", np.nan),
                "w_latency": r["requested_w_latency"],
                "w_energy": r["requested_w_energy"],
                "w_load": r["requested_w_load"],
                "w_fault_tolerance": r["requested_w_fault_tolerance"],
                "latency_obj": vec[0],
                "energy_obj": vec[1],
                "load_obj": vec[2],
                "availability_loss_obj": vec[3],
                "valid_rate": r["valid_rate"],
                "latency": r["latency"],
                "energy": r["energy"],
                "load_variance": r["load_variance"],
                "fault_tolerance": r["fault_tolerance"],
                "mean_objective": r["mean_objective"],
            }
            (cand_rows if r["audit_kind"] == "pareto" else steer_rows).append(rec)
        for _, r in ftr.iterrows():
            w = np.asarray(
                [
                    r["requested_w_latency"],
                    r["requested_w_energy"],
                    r["requested_w_load"],
                    r["requested_w_fault_tolerance"],
                ],
                dtype=float,
            )
            f = np.asarray(
                [r["latency_obj"], r["energy_obj"], r["load_obj"], r["availability_loss_obj"]], dtype=float
            )
            sweep_rows.append(
                {
                    "scenario_id": sc.scenario_id,
                    "method": method,
                    "training_seed": seed,
                    "audit_kind": r["audit_kind"],
                    "preference_id": r["preference_id"],
                    "controlled_objective": r["controlled_objective"],
                    "controlled_weight": r["controlled_weight"],
                    "trace_id": r["trace_id"],
                    "valid": bool(r["valid"]),
                    "scalarized_cost": float(np.dot(w, f)),
                    "latency_obj": f[0],
                    "energy_obj": f[1],
                    "load_obj": f[2],
                    "availability_loss_obj": f[3],
                    "fault_tolerance": r["fault_tolerance"],
                    "recovery_attempts": int(r.get("recovery_attempts", 0) or 0),
                    "migrated_instances": int(r.get("migrated_instances", 0) or 0),
                }
            )
        pts = np.asarray(
            [
                [c["latency_obj"], c["energy_obj"], c["load_obj"], c["availability_loss_obj"]]
                for c in cand_rows
                if float(c["valid_rate"]) >= 1.0 - 1e-12
            ]
        ).reshape(-1, 4)
        row["hypervolume"], row["pareto_size"] = M._suite_hv_of_points(pts)
        row["pareto_candidates_complete"] = int(len(pts))
        pr = [s for s in sweep_rows if s["audit_kind"] == "pareto"]
        row["pref_avg_scalarized_cost"] = float(np.mean([s["scalarized_cost"] for s in pr])) if pr else np.nan
        row["hv_basis"] = f"{len(prefs)}-preference sweep"
    else:
        pt = (
            np.asarray(
                [[row["latency_obj"], row["energy_obj"], row["load_obj"], row["availability_loss_obj"]]]
            )
            if row["valid_rate"] >= 1.0 - 1e-12
            else np.empty((0, 4))
        )
        row["hypervolume"], row["pareto_size"] = M._suite_hv_of_points(pt)
        row["hv_basis"] = "single operating point"
    fault_rows = []
    if fault:
        for k in b.fault_cardinalities:
            ftr_k = M._suite_fault_traces(
                tgt, k, b.fault_sets_per_k, 7000 + M._suite_scenario_seed(sc.scenario_id)
            )
            fagg, _ = M.evaluate_agent_on_traces(
                agent, tgt, cfg, ftr_k, M.UNIFORM_PREFERENCE, "full", "reactive_migration"
            )
            fault_rows.append(
                {
                    **base,
                    "failed_node_count": int(k),
                    "node_sets_evaluated": len(ftr_k),
                    "nodes": len(tgt.nodes),
                    **M._suite_metrics(fagg),
                }
            )
    row["total_eval_wallclock_sec"] = time.perf_counter() - t_start
    M._suite_append(paths, "seed_results.csv", [row])
    M._suite_append(paths, "trace_results.csv", trace_rows)
    M._suite_append(paths, "pareto_candidates.csv", cand_rows)
    M._suite_append(paths, "steerability_controlled.csv", steer_rows)
    M._suite_append(paths, "sweep_traces.csv", sweep_rows)
    M._suite_append(paths, "fault_cardinality.csv", fault_rows)
    M._suite_mark(paths, key, {"row": {k: M._suite_cell(v) for k, v in row.items()}})
    print(
        f"         done in {row['total_eval_wallclock_sec'] / 60:.1f} min: mean objective "
        f"{row['mean_objective']:.4f}, valid {row['valid_rate']:.3f}, HV {row['hypervolume']:.4f}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# D. experiment drivers
# -----------------------------------------------------------------------------
def run_ablation(seeds: Optional[Sequence[int]] = None) -> None:
    xp = extra_paths()
    sc, cfg, b, src, tgt, traces = _setup(ABL_SCENARIO)
    for seed in seeds or _seeds():
        seed = int(seed)
        print("-" * 78 + f"\n[ABLATION] {ABL_SCENARIO} seed={seed}", flush=True)
        todo = [m for m in ABL_METHODS if M._suite_done(xp, f"{ABL_SCENARIO}__{m}__seed{seed}__eval") is None]
        if not todo:
            print("  all variants complete", flush=True)
            continue
        if any(m in todo for m in ABL_METHODS[:3]):
            agent = _load_v3(tgt, cfg, _base8_checkpoint(seed))
            agent.eval_lookahead = int(M.V3_FULL.lookahead_k)
            info = _base8_train_info(seed)
            for m, mode in (
                ("full_v3_ref", "preference"),
                ("abl_rec_uniform", "uniform"),
                ("abl_rec_restore_only", "restore_only"),
            ):
                with M.v2_method(M.V3_FULL), recovery_mode(mode):
                    _evaluate(
                        xp,
                        sc,
                        cfg,
                        b,
                        tgt,
                        traces,
                        seed,
                        m,
                        ABL_LABEL[m],
                        agent,
                        info,
                        pareto=True,
                        fault=False,
                    )
            del agent
            gc.collect()
        if any(m in todo for m in ABL_METHODS[3:]):
            agent, info = _train_with_source(
                xp, sc, cfg, src, tgt, traces, seed, ABL_NO_PREF, "abl_critic_no_pref"
            )
            agent.eval_lookahead = int(M.V3_FULL.lookahead_k)
            for m, mode in (
                ("abl_critic_no_pref", "preference"),
                ("abl_critic_no_pref_rec_uniform", "uniform"),
            ):
                with M.v2_method(ABL_NO_PREF), recovery_mode(mode):
                    _evaluate(
                        xp,
                        sc,
                        cfg,
                        b,
                        tgt,
                        traces,
                        seed,
                        m,
                        ABL_LABEL[m],
                        agent,
                        info,
                        pareto=True,
                        fault=False,
                    )
            del agent
            gc.collect()


def run_zero_shot(scenarios: Sequence[str] = ZS_SCENARIOS_ALL, seeds: Optional[Sequence[int]] = None) -> None:
    xp = extra_paths()
    for sid in scenarios:
        sc, cfg, b, src, tgt, traces = _setup(sid)
        print(
            "=" * 78 + f"\n[ZERO-SHOT] 8 nodes -> {sid}: {len(tgt.nodes)} nodes, "
            f"{len(tgt.instances)} instances, {len(traces['test'])} test traces",
            flush=True,
        )
        for seed in seeds or _seeds():
            seed = int(seed)
            todo = [
                m
                for m in ("zs_v3", "zs_v3_no_lookahead")
                if M._suite_done(xp, f"{sid}__{m}__seed{seed}__eval") is None
            ]
            fs_method = "fs_v3"
            want_fs = (
                FEWSHOT_FRACTION > 0 and M._suite_done(xp, f"{sid}__{fs_method}__seed{seed}__eval") is None
            )
            if not todo and not want_fs:
                print(f"  [SKIP] seed={seed} complete", flush=True)
                continue
            info = {
                **_base8_train_info(seed),
                "train_runtime_sec": 0.0,
                "base8_train_runtime_sec": _base8_train_info(seed)["train_runtime_sec"],
            }
            agent = _load_v3(tgt, cfg, _base8_checkpoint(seed))
            src_info = {"source_model": f"BASE_N8__full_v3__seed{seed}", "source_nodes": 8}
            with M.v2_method(M.V3_FULL), recovery_mode("preference"):
                agent.eval_lookahead = int(M.V3_FULL.lookahead_k)
                _evaluate(
                    xp,
                    sc,
                    cfg,
                    b,
                    tgt,
                    traces,
                    seed,
                    "zs_v3",
                    ZS_LABEL["zs_v3"],
                    agent,
                    info,
                    pareto=bool(ZS_PARETO),
                    fault=True,
                    extra=src_info,
                )
                agent.eval_lookahead = 0
                _evaluate(
                    xp,
                    sc,
                    cfg,
                    b,
                    tgt,
                    traces,
                    seed,
                    "zs_v3_no_lookahead",
                    ZS_LABEL["zs_v3_no_lookahead"],
                    agent,
                    info,
                    pareto=False,
                    fault=False,
                    extra=src_info,
                )
                if want_fs:
                    eps = max(1, int(round(FEWSHOT_FRACTION * cfg.episodes)))
                    f_cfg = dataclasses.replace(
                        cfg, checkpoint_min_episode=max(1, min(cfg.checkpoint_min_episode, eps // 2))
                    )
                    print(f"  [TRAIN] few-shot fine-tuning {eps} episodes seed={seed}", flush=True)
                    f_agent, f_info = M._suite_train_dqn(
                        xp,
                        f"{sid}__{fs_method}__seed{seed}",
                        tgt,
                        f_cfg,
                        seed,
                        eps,
                        traces["validation"],
                        label="full",
                        initial_transfer=agent.extract_selective_transfer("v2_full"),
                        selection_preferences=M.fixed_weight_bank(),
                        training_traces=traces["training"],
                    )
                    f_agent.eval_lookahead = int(M.V3_FULL.lookahead_k)
                    _evaluate(
                        xp,
                        sc,
                        cfg,
                        b,
                        tgt,
                        traces,
                        seed,
                        fs_method,
                        ZS_LABEL["fs_v3"],
                        f_agent,
                        {**f_info, "fewshot_episodes": eps},
                        pareto=bool(ZS_PARETO),
                        fault=False,
                        extra={**src_info, "fewshot_episodes": eps},
                    )
                    del f_agent
            del agent
            gc.collect()


def run_part(part: str) -> None:
    part = part.upper()
    kind, scen = PARTS[part]
    print("=" * 78)
    print(f"EXTRA EXPERIMENTS -- PART {part} ({kind}: {', '.join(scen)})   [{EXTRA_VERSION}]")
    print(f"main results : {Path(MAIN_RESULTS_ROOT) / f'results_{BUDGET}'}")
    print(f"new results  : {Path(MAIN_RESULTS_ROOT) / f'results_{BUDGET}' / 'extra_experiments'}")
    print(f"seeds        : {_seeds()}")
    print("=" * 78, flush=True)
    probs = check_main_run(scen if kind == "zeroshot" else (ABL_SCENARIO,))
    fatal = [p for p in probs if not p.startswith("missing 8-node checkpoint")]
    for p in probs:
        print("  ! " + p, flush=True)
    if fatal:
        raise SystemExit("\nSTOP: fix the problems above (usually --output, --data-dir or --budget).")
    try:
        torch.set_num_threads(max(1, int(os.environ.get("TLFT_TORCH_THREADS", "1"))))
    except Exception:
        pass
    t0 = time.time()
    if kind == "ablation":
        run_ablation()
    else:
        run_zero_shot(scen)
    hours = (time.time() - t0) / 3600.0
    done = extra_paths()["root"] / f"EXTRA_PART_{part}_DONE.txt"
    done.write_text(
        f"part {part} finished in {hours:.2f} h ({time.strftime('%Y-%m-%d %H:%M')})\n", encoding="utf-8"
    )
    print(
        f"\n[PART {part} FINISHED in {hours:.2f} h] -> {done}\nWhen all parts are finished run --report",
        flush=True,
    )


# -----------------------------------------------------------------------------
# E. status
# -----------------------------------------------------------------------------
def status() -> pd.DataFrame:
    main_root = Path(MAIN_RESULTS_ROOT) / f"results_{BUDGET}"
    if not main_root.is_dir():
        print(
            f"\n!! Main results folder not found: {main_root}\n"
            "   --output must be the output folder of run_experiments.py."
        )
        return pd.DataFrame()
    print(f"main results folder found: {main_root}")
    xp = extra_paths()
    rows = []
    for part, (kind, scen) in PARTS.items():
        methods = (
            ABL_METHODS
            if kind == "ablation"
            else (["zs_v3", "zs_v3_no_lookahead"] + (["fs_v3"] if FEWSHOT_FRACTION > 0 else []))
        )
        for sid in scen:
            for s in _seeds():
                for m in methods:
                    d = M._suite_done(xp, f"{sid}__{m}__seed{s}__eval")
                    rows.append(
                        {"part": part, "scenario": sid, "seed": s, "method": m, "done": d is not None}
                    )
    df = pd.DataFrame(rows)
    print("\nEXTRA EXPERIMENTS -- progress")
    for part, g in df.groupby("part"):
        mark = "DONE" if (xp["root"] / f"EXTRA_PART_{part}_DONE.txt").exists() else ""
        print(f"  part {part}: {int(g.done.sum())}/{len(g)} evaluations finished {mark}")
    return df


# -----------------------------------------------------------------------------
# F. report: statistics, figures, Excel
# -----------------------------------------------------------------------------
def _ci(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return M._suite_ci_half(v) if len(v) >= 2 else 0.0


def _fmt(v, d=4) -> str:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return "-"
    return f"{v.mean():.{d}f} ± {_ci(v):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def _paired_trace_p(a: pd.DataFrame, bdf: pd.DataFrame, metric: str, keys=("training_seed", "trace_id")):
    keys = list(keys)
    m = a[keys + [metric]].merge(bdf[keys + [metric]], on=keys, suffixes=("_a", "_b"))
    d = pd.to_numeric(m[f"{metric}_a"], errors="coerce") - pd.to_numeric(m[f"{metric}_b"], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(d) < 2:
        return np.nan, int(len(d)), np.nan
    return M._suite_signflip_p(d), int(len(d)), float(d.mean())


def _load(paths, name, keys):
    return M._suite_read(paths["raw"], name, keys)


def _ablation_tables(xp):
    seed = _load(xp, "seed_results.csv", ["scenario_id", "method", "training_seed"])
    trace = _load(xp, "trace_results.csv", ["scenario_id", "method", "training_seed", "trace_id"])
    sweep = _load(
        xp, "sweep_traces.csv", ["scenario_id", "method", "training_seed", "preference_id", "trace_id"]
    )
    steer = _load(
        xp, "steerability_controlled.csv", ["scenario_id", "method", "training_seed", "preference_id"]
    )
    out = {}
    if seed.empty:
        return out
    seed = seed[seed.method.isin(ABL_METHODS)]
    trace = trace[trace.method.isin(ABL_METHODS)] if len(trace) else trace
    sweep = sweep[sweep.method.isin(ABL_METHODS)] if len(sweep) else sweep
    steer = steer[steer.method.isin(ABL_METHODS)] if len(steer) else steer
    ref = "full_v3_ref"
    # 1) balanced operating point
    rows = []
    for m in [x for x in ABL_METHODS if x in set(seed.method)]:
        s = seed[seed.method == m]
        r = {"method": m, "variant": ABL_LABEL[m], "seeds": int(len(s))}
        for k in (
            "mean_objective",
            "latency_obj",
            "energy_obj",
            "load_obj",
            "availability_loss_obj",
            "fault_tolerance",
            "valid_rate",
        ):
            r[METRIC_LABEL.get(k, k)] = _fmt(pd.to_numeric(s[k], errors="coerce"))
            r["_" + k] = float(pd.to_numeric(s[k], errors="coerce").mean())
        rows.append(r)
    bal = pd.DataFrame(rows)
    if len(bal) and ref in set(bal.method):
        refv = bal.set_index("method")["_mean_objective"][ref]
        bal["Δ mean objective vs full (%)"] = (bal["_mean_objective"] - refv) / abs(refv) * 100.0
    out["A_Balanced"] = bal[[c for c in bal.columns if not c.startswith("_")]] if len(bal) else bal
    # 2) preference sweep
    rows = []
    sw_tr = pd.DataFrame()
    if len(sweep):
        sp = sweep[sweep.audit_kind == "pareto"].copy()
        sp["scalarized_cost"] = pd.to_numeric(sp["scalarized_cost"], errors="coerce")
        sp["recovery_attempts"] = pd.to_numeric(sp["recovery_attempts"], errors="coerce").fillna(0)
        sw_tr = sp.groupby(["method", "training_seed", "trace_id"], as_index=False).agg(
            scalarized_cost=("scalarized_cost", "mean"), recovery_attempts=("recovery_attempts", "sum")
        )
    for m in [x for x in ABL_METHODS if x in set(seed.method)]:
        s = seed[seed.method == m]
        r = {
            "method": m,
            "variant": ABL_LABEL[m],
            "seeds": int(len(s)),
            "Hypervolume (60-pref sweep) ↑": _fmt(pd.to_numeric(s.get("hypervolume"), errors="coerce")),
            "Pareto size": _fmt(pd.to_numeric(s.get("pareto_size"), errors="coerce"), 1),
            "Preference-averaged cost p·F ↓": _fmt(
                pd.to_numeric(s.get("pref_avg_scalarized_cost"), errors="coerce")
            ),
            "_hv": float(pd.to_numeric(s.get("hypervolume"), errors="coerce").mean()),
            "_cost": float(pd.to_numeric(s.get("pref_avg_scalarized_cost"), errors="coerce").mean()),
        }
        rows.append(r)
    sw = pd.DataFrame(rows)
    if len(sw) and ref in set(sw.method):
        hv0 = sw.set_index("method")["_hv"][ref]
        c0 = sw.set_index("method")["_cost"][ref]
        sw["Δ HV vs full (%)"] = (sw["_hv"] - hv0) / abs(hv0) * 100.0 if hv0 else np.nan
        sw["Δ cost vs full (%)"] = (sw["_cost"] - c0) / abs(c0) * 100.0 if c0 else np.nan
    out["A_Preference_Sweep"] = sw[[c for c in sw.columns if not c.startswith("_")]] if len(sw) else sw
    # 3) paired tests (full vs each variant), Holm within each metric family
    tests = []
    if ref in set(seed.method):
        fam = {}
        for m in [x for x in ABL_METHODS if x != ref and x in set(seed.method)]:
            for metric in TRACE_METRICS:
                if len(trace):
                    p, n, d = _paired_trace_p(trace[trace.method == m], trace[trace.method == ref], metric)
                    fam.setdefault(("balanced", metric), []).append(
                        {
                            "test": "balanced point p=uniform, pairs=(seed, test trace)",
                            "metric": METRIC_LABEL[metric],
                            "variant": ABL_LABEL[m],
                            "method": m,
                            "n_pairs": n,
                            "mean_diff_variant_minus_full": d,
                            "p_signflip": p,
                        }
                    )
            if len(sw_tr):
                a, r0 = sw_tr[sw_tr.method == m], sw_tr[sw_tr.method == ref]
                p, n, d = _paired_trace_p(a, r0, "scalarized_cost")
                fam.setdefault(("sweep", "all"), []).append(
                    {
                        "test": "60-pref sweep, cost averaged over prefs, pairs=(seed, test trace)",
                        "metric": "Preference-averaged cost p·F",
                        "variant": ABL_LABEL[m],
                        "method": m,
                        "n_pairs": n,
                        "mean_diff_variant_minus_full": d,
                        "p_signflip": p,
                    }
                )
                rec_keys = pd.concat([a, r0])
                rec_keys = rec_keys[rec_keys.recovery_attempts > 0][
                    ["training_seed", "trace_id"]
                ].drop_duplicates()
                p, n, d = _paired_trace_p(a.merge(rec_keys), r0.merge(rec_keys), "scalarized_cost")
                fam.setdefault(("sweep", "recovery"), []).append(
                    {
                        "test": "60-pref sweep, ONLY traces with a recovery, pairs=(seed, test trace)",
                        "metric": "Preference-averaged cost p·F",
                        "variant": ABL_LABEL[m],
                        "method": m,
                        "n_pairs": n,
                        "mean_diff_variant_minus_full": d,
                        "p_signflip": p,
                    }
                )
            hv = seed[seed.method.isin([m, ref])][["method", "training_seed", "hypervolume"]]
            pv = hv.pivot_table(
                index="training_seed", columns="method", values="hypervolume", aggfunc="mean"
            ).dropna()
            if len(pv) >= 2 and m in pv and ref in pv:
                d = (pv[m] - pv[ref]).to_numpy(float)
                fam.setdefault(("hv", "seed"), []).append(
                    {
                        "test": "hypervolume, pairs=seeds (exact sign-flip; 5 seeds -> min p = 0.0625)",
                        "metric": "Hypervolume",
                        "variant": ABL_LABEL[m],
                        "method": m,
                        "n_pairs": int(len(d)),
                        "mean_diff_variant_minus_full": float(d.mean()),
                        "p_signflip": M._suite_signflip_p(d),
                        "seeds_variant_lower": int(np.sum(d < 0)),
                    }
                )
        for _k, rows_ in fam.items():
            adj = M._holm_adjust([r["p_signflip"] for r in rows_])
            for r, a_ in zip(rows_, adj):
                r["p_holm"] = a_
                tests.append(r)
    out["A_Tests"] = pd.DataFrame(tests)
    # 4) steerability
    st = M._suite_steer(steer) if len(steer) else pd.DataFrame()
    if len(st):
        st["variant"] = st["method"].map(ABL_LABEL)
        st = st[
            [
                "variant",
                "method",
                "controlled_objective",
                "value_at_w_min",
                "value_at_w_max",
                "endpoint_improvement_pct",
                "rho_bar",
                "seeds_in_expected_direction",
                "n_seeds",
                "p_holm_two_sided",
                "p_holm_one_sided",
            ]
        ]
    out["A_Steerability"] = st
    # 5) reproducibility check vs the main run
    mp = main_paths()
    ms = _load(mp, "seed_results.csv", ["scenario_id", "method", "training_seed"])
    rr = []
    if len(ms):
        a = ms[(ms.scenario_id == ABL_SCENARIO) & (ms.method == "full_v3")]
        bb = seed[seed.method == ref]
        for _, r in bb.iterrows():
            o = a[a.training_seed == r.training_seed]
            if len(o):
                o = o.iloc[-1]
                rr.append(
                    {
                        "seed": r.training_seed,
                        "main_run_mean_objective": o.mean_objective,
                        "re_evaluated_mean_objective": r.mean_objective,
                        "abs_difference": abs(float(o.mean_objective) - float(r.mean_objective)),
                        "main_run_HV": o.get("hypervolume"),
                        "re_evaluated_HV": r.get("hypervolume"),
                    }
                )
    out["A_Reproducibility"] = pd.DataFrame(rr)
    out["_seed"], out["_steer"] = seed, steer
    return out


def _zs_tables(xp):
    out = {}
    seed = _load(xp, "seed_results.csv", ["scenario_id", "method", "training_seed"])
    trace = _load(xp, "trace_results.csv", ["scenario_id", "method", "training_seed", "trace_id"])
    steer = _load(
        xp, "steerability_controlled.csv", ["scenario_id", "method", "training_seed", "preference_id"]
    )
    fault = _load(
        xp, "fault_cardinality.csv", ["scenario_id", "method", "training_seed", "failed_node_count"]
    )
    if seed.empty:
        return out
    zsm = ["zs_v3", "zs_v3_no_lookahead", "fs_v3"]
    seed = seed[seed.method.isin(zsm)]
    if seed.empty:
        return out
    mp = main_paths()
    ms = _load(mp, "seed_results.csv", ["scenario_id", "method", "training_seed"])
    mt = _load(mp, "trace_results.csv", ["scenario_id", "method", "training_seed", "trace_id"])
    mst = _load(
        mp, "steerability_controlled.csv", ["scenario_id", "method", "training_seed", "preference_id"]
    )
    mf = _load(mp, "fault_cardinality.csv", ["scenario_id", "method", "training_seed", "failed_node_count"])
    scen = [s for s in ZS_SCENARIOS_ALL if s in set(seed.scenario_id)]
    ms = ms[ms.scenario_id.isin(scen) & ms.method.isin(ZS_COMPARE)] if len(ms) else ms
    mt = mt[mt.scenario_id.isin(scen) & mt.method.isin(ZS_COMPARE)] if len(mt) else mt
    allseed = pd.concat([seed, ms], ignore_index=True)
    alltrace = pd.concat([trace[trace.method.isin(zsm)] if len(trace) else trace, mt], ignore_index=True)
    order = zsm + ZS_COMPARE
    rows = []
    for sid in scen:
        for m in [x for x in order if x in set(allseed[allseed.scenario_id == sid].method)]:
            s = allseed[(allseed.scenario_id == sid) & (allseed.method == m)]
            tr = pd.to_numeric(s.get("train_runtime_sec"), errors="coerce").fillna(0) + pd.to_numeric(
                s.get("source_runtime_sec"), errors="coerce"
            ).fillna(0)
            if m in ("zs_v3", "zs_v3_no_lookahead"):
                tr = tr * 0.0
            rows.append(
                {
                    "scenario_id": sid,
                    "method": m,
                    "algorithm": ZS_LABEL.get(m, m),
                    "seeds": int(len(s)),
                    "Mean objective ↓": _fmt(pd.to_numeric(s.get("mean_objective"), errors="coerce")),
                    "C_R ↓": _fmt(pd.to_numeric(s.get("availability_loss_obj"), errors="coerce")),
                    "Fault tolerance ↑": _fmt(pd.to_numeric(s.get("fault_tolerance"), errors="coerce")),
                    "Valid rate ↑": _fmt(pd.to_numeric(s.get("valid_rate"), errors="coerce"), 3),
                    "Hypervolume ↑": (
                        _fmt(pd.to_numeric(s.get("hypervolume"), errors="coerce"))
                        if "sweep" in str(s.get("hv_basis", pd.Series([""])).iloc[0])
                        else "single point"
                    ),
                    "Training on this scenario (h)": float(tr.mean() / 3600.0),
                    "_obj": float(pd.to_numeric(s.get("mean_objective"), errors="coerce").mean()),
                }
            )
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["rank_by_mean_objective"] = (
            tab[~tab.method.isin(["zs_v3_no_lookahead", "fs_v3"])]
            .groupby("scenario_id")["_obj"]
            .rank(method="min")
        )
        gaps = []
        for sid, g in tab.groupby("scenario_id"):
            ref = g[g.method == "full_v3"]["_obj"]
            for i, r in g.iterrows():
                gaps.append((i, (r["_obj"] - ref.iloc[0]) / abs(ref.iloc[0]) * 100.0 if len(ref) else np.nan))
        for i, v in gaps:
            tab.loc[i, "Δ vs trained proposed (%)"] = v
    out["Z_ZeroShot"] = tab[[c for c in tab.columns if not c.startswith("_")]] if len(tab) else tab
    tests = []
    for sid in scen:
        for zm in [x for x in zsm if x in set(seed[seed.scenario_id == sid].method)]:
            for metric in TRACE_METRICS:
                fam = []
                a = alltrace[(alltrace.scenario_id == sid) & (alltrace.method == zm)]
                for o in ZS_COMPARE:
                    bdf = alltrace[(alltrace.scenario_id == sid) & (alltrace.method == o)]
                    if bdf.empty or a.empty:
                        continue
                    p, n, d = _paired_trace_p(a, bdf, metric)
                    fam.append(
                        {
                            "scenario_id": sid,
                            "zero_shot_method": zm,
                            "metric": METRIC_LABEL[metric],
                            "other": ZS_LABEL.get(o, o),
                            "n_pairs": n,
                            "mean_diff_zs_minus_other": d,
                            "zs_better": (d < 0) if np.isfinite(d) else None,
                            "p_signflip": p,
                        }
                    )
                adj = M._holm_adjust([r["p_signflip"] for r in fam]) if fam else []
                for r, a_ in zip(fam, adj):
                    r["p_holm"] = a_
                    tests.append(r)
    out["Z_Tests"] = pd.DataFrame(tests)
    st_all = pd.concat(
        [
            steer[steer.method.isin(zsm)] if len(steer) else steer,
            mst[mst.scenario_id.isin(scen) & (mst.method == "full_v3")] if len(mst) else mst,
        ],
        ignore_index=True,
    )
    st = M._suite_steer(st_all) if len(st_all) else pd.DataFrame()
    if len(st):
        st["algorithm"] = st["method"].map(lambda m: ZS_LABEL.get(m, m))
        st = st[
            [
                "scenario_id",
                "algorithm",
                "method",
                "controlled_objective",
                "endpoint_improvement_pct",
                "rho_bar",
                "seeds_in_expected_direction",
                "n_seeds",
                "p_holm_two_sided",
            ]
        ]
    out["Z_Steerability"] = st
    fr = pd.concat(
        [
            fault[fault.method.isin(zsm)] if len(fault) else fault,
            mf[mf.scenario_id.isin(scen) & mf.method.isin(["full_v3", "no_tl_v3", "k8s"])] if len(mf) else mf,
        ],
        ignore_index=True,
    )
    if len(fr):
        fr = fr.groupby(["scenario_id", "method", "failed_node_count"], as_index=False).agg(
            valid_rate=("valid_rate", "mean"),
            fault_tolerance=("fault_tolerance", "mean"),
            mean_objective=("mean_objective", "mean"),
        )
        fr["algorithm"] = fr["method"].map(lambda m: ZS_LABEL.get(m, m))
    out["Z_Faults"] = fr
    out["_tab"], out["_seed"] = tab, allseed
    return out


def _figures(xp, A, Z) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    col = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7b5cc4"]
    s = A.get("_seed")
    if s is not None and len(s):
        ms = [m for m in ABL_METHODS if m in set(s.method)]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (k, title) in zip(
            axes,
            (
                ("mean_objective", "Balanced objective ↓"),
                ("pref_avg_scalarized_cost", "Preference-averaged cost p·F ↓"),
                ("hypervolume", "Hypervolume ↑"),
            ),
        ):
            if k not in s.columns:
                ax.set_visible(False)
                continue
            vals = [pd.to_numeric(s[s.method == m][k], errors="coerce").dropna().to_numpy() for m in ms]
            ax.bar(
                range(len(ms)),
                [v.mean() if len(v) else np.nan for v in vals],
                yerr=[_ci(v) for v in vals],
                color=col[: len(ms)],
                capsize=3,
            )
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels([ABL_SHORT[m] for m in ms], rotation=0, fontsize=6.5)
            ax.set_title(title)
            lo = np.nanmin([v.mean() - _ci(v) for v in vals if len(v)]) if any(len(v) for v in vals) else 0
            hi = np.nanmax([v.mean() + _ci(v) for v in vals if len(v)]) if any(len(v) for v in vals) else 1
            ax.set_ylim(max(0, lo - 0.15 * (hi - lo + 1e-9)), hi + 0.1 * (hi - lo + 1e-9))
        fig.tight_layout()
        p = xp["plots"] / "X1_ablation_core_novelty.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figs.append(p)
    st = A.get("A_Steerability")
    if st is not None and len(st):
        pv = st.pivot_table(index="variant", columns="controlled_objective", values="rho_bar")
        pv = pv.reindex([ABL_LABEL[m] for m in ABL_METHODS if ABL_LABEL[m] in pv.index])
        cols = [c for c in ("latency", "energy", "load", "fault_tolerance") if c in pv.columns]
        pv = pv[cols]
        sign = np.array([[-1 if c != "fault_tolerance" else 1 for c in cols]])
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        im = ax.imshow(pv.to_numpy() * sign, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(pv.index)))
        ax.set_yticklabels(pv.index, fontsize=7)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="steering strength (+ = correct direction)")
        ax.set_title("Steerability (mean seed-wise Spearman ρ)")
        fig.tight_layout()
        p = xp["plots"] / "X2_ablation_steerability.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figs.append(p)
    tab = Z.get("_tab")
    if tab is not None and len(tab):
        scen = list(dict.fromkeys(tab.scenario_id))
        ms = [
            m
            for m in ["zs_v3", "fs_v3", "full_v3", "no_tl_v3", "greedy_latency", "xu_modrl", "k8s"]
            if m in set(tab.method)
        ]
        s = Z["_seed"]
        fig, ax = plt.subplots(figsize=(10, 3.6))
        w = 0.8 / max(1, len(ms))
        pal = {
            "zs_v3": "#e34948",
            "fs_v3": "#eda100",
            "full_v3": "#2a78d6",
            "no_tl_v3": "#eb6834",
            "greedy_latency": "#7b5cc4",
            "xu_modrl": "#1baf7a",
            "k8s": "#8c8c8c",
        }
        for j, m in enumerate(ms):
            vals = [
                pd.to_numeric(
                    s[(s.scenario_id == sid) & (s.method == m)]["mean_objective"], errors="coerce"
                ).dropna()
                for sid in scen
            ]
            ax.bar(
                np.arange(len(scen)) + (j - (len(ms) - 1) / 2) * w,
                [v.mean() if len(v) else np.nan for v in vals],
                w,
                yerr=[_ci(v) for v in vals],
                label=ZS_LABEL.get(m, m),
                color=pal.get(m, "#555"),
                capsize=2,
            )
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels(scen)
        ax.set_ylabel("Mean normalised objective ↓")
        ax.legend(fontsize=7, ncol=4, frameon=False)
        ax.set_title("Zero-shot transfer of the 8-node policy vs. methods trained on each scale")
        fig.tight_layout()
        p = xp["plots"] / "X3_zero_shot_scalability.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        figs.append(p)
    return figs


def _summary_text(A, Z) -> str:
    L = [
        f"EXTRA EXPERIMENTS SUMMARY ({EXTRA_VERSION}, budget={BUDGET}, {time.strftime('%Y-%m-%d %H:%M')})",
        "",
    ]
    bal, sw, te = A.get("A_Balanced"), A.get("A_Preference_Sweep"), A.get("A_Tests")
    if sw is not None and len(sw):
        L.append("[Experiment 1] Contribution of preference conditioning (BASE_N8)")
        for _, r in sw.iterrows():
            L.append(
                f"  {r['variant']:<42s} HV {r['Hypervolume (60-pref sweep) ↑']:<18s} "
                f"pref-avg cost {r['Preference-averaged cost p·F ↓']}"
            )
        if bal is not None and len(bal):
            for _, r in bal.iterrows():
                L.append(f"  {r['variant']:<42s} balanced objective {r['Mean normalised objective']}")
        if te is not None and len(te):
            for _, r in te[
                te.metric.isin(["Preference-averaged cost p·F", "Mean normalised objective"])
            ].iterrows():
                L.append(
                    f"    test [{r['test'][:40]}] {r['variant']}: diff={r['mean_diff_variant_minus_full']:+.4f} "
                    f"Holm p={r['p_holm']:.3g} (n={r['n_pairs']})"
                )
        L.append("")
    tab = Z.get("Z_ZeroShot")
    if tab is not None and len(tab):
        L.append("[Experiment 2] Zero-shot size transfer (8-node policy, no training on the target scale)")
        for _, r in tab.iterrows():
            L.append(
                f"  {r['scenario_id']:<11s} {r['algorithm']:<40s} obj {r['Mean objective ↓']:<18s} "
                f"HV {r['Hypervolume ↑']:<18s} train {r['Training on this scenario (h)']:.2f} h"
            )
    return "\n".join(L)


def build_extra_report() -> Path:
    xp = extra_paths()
    A = _ablation_tables(xp)
    Z = _zs_tables(xp)
    figs = _figures(xp, A, Z)
    readme = pd.DataFrame(
        {
            "item": [
                "Version",
                "Main results",
                "Experiment 1",
                "",
                "",
                "",
                "",
                "Experiment 2",
                "",
                "",
                "Statistics",
                "",
            ],
            "explanation": [
                EXTRA_VERSION,
                str(Path(MAIN_RESULTS_ROOT) / f"results_{BUDGET}"),
                "Core-novelty ablation on BASE_N8 (sheets A_*), same locked test traces and seeds as the main run.",
                "Recovery ablations re-use the reported checkpoint and change ONLY the ranking of recovery destinations.",
                "Critic-without-preference is retrained with the identical pipeline; its critic receives a constant "
                "preference input while p is still used for scalarisation, lookahead and recovery.",
                "Preference-averaged cost = mean over the 60 sweep preferences and test traces of p·F (lower is better).",
                "A_Reproducibility compares the re-evaluated reported checkpoint with the numbers of the main run.",
                "Zero-shot transfer (sheets Z_*): BASE_N8 policies (8 nodes, 20 instances) applied without any training "
                "to 16/32-node scenarios (40/80 instances), same test traces as the main run.",
                "Other methods in Z_ZeroShot are read from the main run (trained on each scenario).",
                "Training on this scenario = target + source training time spent on that scenario (0 for zero-shot).",
                "Paired sign-flip permutation tests over (seed, test trace) pairs, Holm-adjusted within each family.",
                "HV is compared over seeds (5 pairs; exact sign-flip, minimum attainable p = 0.0625).",
            ],
        }
    )
    sheets = [("README", readme)]
    for k in (
        "A_Balanced",
        "A_Preference_Sweep",
        "A_Tests",
        "A_Steerability",
        "A_Reproducibility",
        "Z_ZeroShot",
        "Z_Tests",
        "Z_Steerability",
        "Z_Faults",
    ):
        if k in A and isinstance(A[k], pd.DataFrame):
            sheets.append((k, A[k]))
        elif k in Z and isinstance(Z[k], pd.DataFrame):
            sheets.append((k, Z[k]))
    raw_seed = _load(xp, "seed_results.csv", ["scenario_id", "method", "training_seed"])
    sheets.append(("Raw_Seed_Results", raw_seed))
    xlsx = xp["report"] / "extra_results.xlsx"
    tmp = xlsx.with_name("extra_results.tmp.xlsx")
    with pd.ExcelWriter(tmp, engine="openpyxl") as w:
        for name, df in sheets:
            (df if len(df) else pd.DataFrame({"note": ["no results yet"]})).to_excel(
                w, sheet_name=name[:31], index=False
            )
            ws = w.sheets[name[:31]]
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
            ws.freeze_panes = "A2"
        if figs:
            from openpyxl.drawing.image import Image as XLImage

            ws = w.book.create_sheet("Figures")
            row = 1
            for f in figs:
                img = XLImage(str(f))
                img.width, img.height = img.width * 0.35, img.height * 0.35
                ws.add_image(img, f"A{row}")
                row += int(img.height / 18) + 3
    os.replace(tmp, xlsx)
    txt = _summary_text(A, Z)
    (xp["report"] / "extra_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nExcel : {xlsx}\nText  : {xp['report'] / 'extra_summary.txt'}\nFigures: {xp['plots']}")
    return xlsx


def main(argv=None) -> int:
    global MAIN_RESULTS_ROOT, INPUT_DIR, BUDGET, ZS_PARETO, FEWSHOT_FRACTION, SEEDS
    p = argparse.ArgumentParser(
        description="Conditioning ablation and zero-shot size transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--output", default=MAIN_RESULTS_ROOT, help="output folder of run_experiments.py")
    p.add_argument("--data-dir", default=INPUT_DIR, help="folder with the .xlsx workload files")
    p.add_argument("--budget", default=BUDGET, choices=["smoke", "pilot", "reduced", "paper"])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--part", choices=sorted(PARTS))
    g.add_argument("--status", action="store_true")
    g.add_argument("--report", action="store_true")
    p.add_argument("--seeds", default="", help="comma-separated seeds (default: all seeds of the budget)")
    p.add_argument("--no-sweep", action="store_true", help="zero-shot: skip the 60-preference sweep")
    p.add_argument("--fewshot", type=float, default=0.0, help="zero-shot: few-shot fraction, e.g. 0.1")
    a = p.parse_args(argv)
    MAIN_RESULTS_ROOT, INPUT_DIR, BUDGET = a.output, a.data_dir, a.budget
    ZS_PARETO, FEWSHOT_FRACTION = not a.no_sweep, float(a.fewshot)
    SEEDS = [int(x) for x in a.seeds.split(",") if x.strip()] or None
    if a.status:
        status()
    elif a.report:
        status()
        build_extra_report()
    else:
        run_part(a.part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
