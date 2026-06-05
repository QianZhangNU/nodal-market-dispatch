"""
Iterative N-1 contingency screening wrapper for the SCED solver.

Algorithm (cutting-plane / lazy-constraint generation):
  1. Build the base LP sections [A]-[E] once via _build_base_lp.
  2. Solve the base SCED (no contingency constraints, active_ctg_pairs = ∅).
  3. Post-contingency screening: compute all post-contingency flows in one
     batched matrix multiply, then flag pairs where
       |f_ctg[l]| / contingency_limit[l] > violation_threshold.
  4. Add newly flagged pairs to active_ctg_pairs and re-solve by appending
     section [F] to the cached base LP — no LP reconstruction.
  5. Repeat until no new violations or max_iter reached.

Performance notes
-----------------
- screen_n1_violations uses a batched (n_ctg, n_line, n_bus) @ (n_bus,)
  matmul instead of a Python loop over contingencies and lines.
  For C contingencies and L lines this cuts C×L Python iterations to one
  numpy BLAS call.

- solve_hourly_sced_n1 calls _build_base_lp once and _solve_with_ctg on
  every iteration.  Iterations 2+ skip the full LP reconstruction
  (line/GTC/ramp row building), saving roughly one _build_base_lp worth
  of time per re-solve.

- Pass contingency_ptdf_tensor + ctg_id_list (from NetworkModel attributes)
  to avoid np.stack overhead inside screen_n1_violations.
"""

import numpy as np
import pandas as pd
from typing import Optional


def screen_n1_violations(
    dispatch: dict,
    effective_load: pd.Series,
    generators: dict,
    bus_list: list,
    line_list: list,
    lines: dict,
    contingency_ptdf: Optional[dict],
    base_line_flows: dict,
    violation_threshold: float = 0.95,
    prefilter_ratio: float = 0.0,
    contingency_ptdf_tensor: Optional[np.ndarray] = None,
    ctg_id_list: Optional[list] = None,
) -> set:
    """Return (ctg_id, line_id) pairs whose post-contingency flow exceeds the limit.

    Uses a single batched matrix multiply — no Python loop over contingencies
    or lines.

    Parameters
    ----------
    dispatch : {gen_id: MW}
    effective_load : pd.Series index=bus  (load minus battery injection)
    generators : {gen_id: {bus, ...}}
    bus_list : ordered bus IDs
    line_list : ordered line IDs (matches PTDF rows)
    lines : {line_id: {contingency_limit, flow_limit, ...}}
    contingency_ptdf : {ctg_id: np.ndarray (n_line, n_bus)}.
        Ignored when contingency_ptdf_tensor + ctg_id_list are supplied.
    base_line_flows : {line_id: MW} from the SCED result (used by prefilter)
    violation_threshold : flag pair when |f_ctg| / contingency_limit > this.
        Default 0.95.
    prefilter_ratio : skip a line entirely when its BASE-CASE flow ratio
        |f_base| / contingency_limit < prefilter_ratio.  0.0 = off.
        E.g. 0.3 skips lines below 30 % of their contingency limit in the
        base case — a cheap heuristic that avoids computing contingency flows
        for obviously safe lines.
    contingency_ptdf_tensor : pre-stacked (n_ctg, n_line, n_bus) array
        from NetworkModel.contingency_ptdf_tensor.  Eliminates np.stack cost.
    ctg_id_list : contingency IDs matching tensor row order
        (NetworkModel.ctg_id_list).

    Returns
    -------
    set of (ctg_id, line_id) tuples
    """
    # ── Net injection per bus ──────────────────────────────────────────────
    gen_at_bus: dict = {}
    for g, gd in generators.items():
        gen_at_bus.setdefault(gd["bus"], []).append(g)

    net_inj = np.array([
        sum(dispatch.get(g, 0.0) for g in gen_at_bus.get(b, []))
        - float(effective_load.get(b, 0.0))
        for b in bus_list
    ])

    # ── Resolve contingency tensor (stack once if not pre-built) ──────────
    if contingency_ptdf_tensor is not None and ctg_id_list is not None:
        ctg_tensor = contingency_ptdf_tensor   # (n_ctg, n_line, n_bus)
        ctg_ids    = ctg_id_list
    elif contingency_ptdf:
        ctg_ids    = list(contingency_ptdf.keys())
        ctg_tensor = np.stack([contingency_ptdf[c] for c in ctg_ids])
    else:
        return set()

    if len(ctg_ids) == 0:
        return set()

    # ── Contingency limits and valid-line mask ────────────────────────────
    ctg_limits = np.array([lines[lid].get("contingency_limit") or 0.0
                            for lid in line_list])
    valid = ctg_limits > 0                          # (n_line,)

    if prefilter_ratio > 0.0 and np.any(valid):
        base_flows = np.array([abs(base_line_flows.get(lid, 0.0)) for lid in line_list])
        valid = valid & (base_flows / np.where(valid, ctg_limits, 1.0) >= prefilter_ratio)

    if not np.any(valid):
        return set()

    # ── Batched matmul: all post-contingency flows at once ────────────────
    # (n_ctg, n_line, n_bus) @ (n_bus,)  →  (n_ctg, n_line)
    all_ctg_flows = ctg_tensor @ net_inj

    # ── Vectorised violation check ────────────────────────────────────────
    limits_safe = np.where(valid, ctg_limits, 1.0)
    ratios   = np.abs(all_ctg_flows) / limits_safe          # (n_ctg, n_line)
    violated = (ratios > violation_threshold) & valid        # broadcast valid

    ci, li = np.where(violated)
    return {(ctg_ids[i], line_list[j]) for i, j in zip(ci, li)}


def solve_hourly_sced_n1(
    generators: dict,
    gen_list: list,
    bus_list: list,
    bus_idx: dict,
    lines: dict,
    line_list: list,
    ptdf: np.ndarray,
    settlement_points: dict,
    contingency_ptdf: Optional[dict] = None,
    gtcs: Optional[dict] = None,
    hour: Optional[pd.Timestamp] = None,
    u_fixed: Optional[dict] = None,
    p_prev: Optional[dict] = None,
    load_h: Optional[pd.Series] = None,
    pmax_h: Optional[pd.Series] = None,
    cost_h: Optional[pd.Series] = None,
    battery_injection_h: Optional[pd.Series] = None,
    use_network: bool = True,
    use_ramp: bool = True,
    use_gtc: bool = True,
    # N-1 parameters
    violation_threshold: float = 0.95,
    prefilter_ratio: float = 0.0,
    ctg_violation_cost: float = 1000.0,
    max_iter: int = 3,
    # Pre-built tensor from NetworkModel (avoids np.stack per screening call)
    contingency_ptdf_tensor: Optional[np.ndarray] = None,
    ctg_id_list: Optional[list] = None,
) -> dict:
    """Solve SCED with iterative N-1 contingency constraint screening.

    Key performance properties vs. calling solve_hourly_sced in a loop:

    1. Base LP cached — _build_base_lp (sections [A]-[E]) is called once.
       Each re-solve only appends [F] rows and runs HiGHS; the line/GTC/ramp
       row construction is not repeated.

    2. Vectorised screening — screen_n1_violations uses a batched matmul
       instead of a Python loop over C×L (contingency, line) pairs.

    3. Pre-built tensor — pass contingency_ptdf_tensor + ctg_id_list from
       NetworkModel to skip np.stack inside each screening call.

    Parameters (N-1 specific)
    --------------------------
    violation_threshold : float
        Flag (ctg_id, line_id) when |f_ctg| / contingency_limit > this.
        Default 0.95.
    prefilter_ratio : float
        Skip screening a line when |f_base| / contingency_limit < this.
        0.0 (default) checks all lines.  0.3–0.5 is a practical fast filter.
    ctg_violation_cost : float
        $/MWh soft-constraint penalty per MW of contingency violation.
        Default 1000.
    max_iter : int
        Maximum LP re-solves after the base case.  Default 3.
    contingency_ptdf_tensor : np.ndarray, optional
        Pre-stacked (n_ctg, n_line, n_bus) — from NetworkModel.contingency_ptdf_tensor.
    ctg_id_list : list, optional
        Contingency IDs matching tensor rows — from NetworkModel.ctg_id_list.

    Returns
    -------
    dict — same keys as solve_hourly_sced, plus:
        n1_active_pairs : set of (ctg_id, line_id) enforced in the final solve
        n1_iterations   : total LP solves performed (base + re-solves)
        ctg_line_duals  : {(ctg_id, line_id): shadow_price $/MWh}
    """
    from solvers.sced.solver import _build_base_lp, _solve_with_ctg

    gtcs = gtcs or {}

    # Effective load computed once — same formula as inside solve_hourly_sced
    if battery_injection_h is not None:
        effective_load = load_h - battery_injection_h.reindex(bus_list).fillna(0)
    else:
        effective_load = load_h

    # Stack contingency PTDF tensor once here if not pre-built by NetworkModel
    if contingency_ptdf_tensor is None and contingency_ptdf:
        _ids = list(contingency_ptdf.keys())
        if _ids:
            contingency_ptdf_tensor = np.stack([contingency_ptdf[c] for c in _ids])
            ctg_id_list = _ids

    # ── Build base LP once ────────────────────────────────────────────────
    cache = _build_base_lp(
        generators, gen_list, bus_list, lines, line_list, ptdf,
        gtcs, u_fixed, p_prev, load_h, pmax_h, cost_h, battery_injection_h,
        use_network, use_ramp, use_gtc,
    )

    active_ctg_pairs: set = set()
    result: dict = {}
    iteration = 0

    for iteration in range(max_iter + 1):
        # ── Solve (append [F] to cached base, no full LP rebuild) ─────────
        result = _solve_with_ctg(
            cache=cache,
            active_ctg_pairs=active_ctg_pairs,
            contingency_ptdf=contingency_ptdf or {},
            ctg_violation_cost=ctg_violation_cost,
            lines=lines, line_list=line_list, bus_list=bus_list,
            gen_list=gen_list, settlement_points=settlement_points,
            ptdf=ptdf, hour=hour, gtcs=gtcs,
            use_network=use_network, battery_injection_h=battery_injection_h,
        )

        if result.get("status") != "optimal":
            break

        if not contingency_ptdf or not use_network:
            break   # nothing to screen

        # ── Vectorised screening ──────────────────────────────────────────
        new_violations = screen_n1_violations(
            dispatch=result["dispatch"],
            effective_load=effective_load,
            generators=generators,
            bus_list=bus_list,
            line_list=line_list,
            lines=lines,
            contingency_ptdf=contingency_ptdf,
            base_line_flows=result.get("line_flows", {}),
            violation_threshold=violation_threshold,
            prefilter_ratio=prefilter_ratio,
            contingency_ptdf_tensor=contingency_ptdf_tensor,
            ctg_id_list=ctg_id_list,
        )
        new_violations -= active_ctg_pairs
        if not new_violations:
            break   # converged
        active_ctg_pairs |= new_violations

    result["n1_active_pairs"] = active_ctg_pairs
    result["n1_iterations"]   = iteration + 1
    return result
