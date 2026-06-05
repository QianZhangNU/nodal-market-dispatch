"""
Hourly SCED Solver (single-hour LP)
===================================
This module implements a topology-agnostic Security-Constrained Economic
Dispatch (SCED) model for one operating hour. Unit commitment is fixed from
SCUC, while dispatch is re-optimized against load, generator availability,
ramp limits, optional DC network constraints, optional GTC constraints, and
exogenous battery net injection.

The battery profile is treated as fixed input: battery generation reduces
effective load and battery charging increases effective load. This model does
not co-optimize battery dispatch.

Optimization problem solved by solve_hourly_sced
================================================

Sets:
  G: generators, B: buses, L: transmission lines, K: GTC constraints.

Inputs:
  u[g]       : fixed commitment from SCUC
  p_prev[g]  : previous-hour dispatch, used only when ramping is enabled
  D[b]       : effective load = load[b] - battery_injection[b]
  Pmax[g]    : hour-specific generator availability
  c[g]       : energy offer / marginal cost
  PTDF[l,b]  : DC shift factor for line l at bus b

Decision variables:
  p[g] >= 0       : generator dispatch (MW)
  shed >= 0       : load-shedding slack (MW)
  s[l] >= 0       : line overload slack only for lines with ocost > 0

Objective:
  minimize
      sum_g c[g] * p[g]
    + SHED_PRICE * shed
    + sum_{l with ocost} ocost[l] * s[l]

Subject to:
  [A] System balance:
      sum_g p[g] + shed = sum_b D[b]

  [B] Generator dispatch bounds:
      Pmin[g] * u[g] <= p[g] <= Pmax[g] * u[g]

  [C] DC line limits when use_network=True:
      f[l] = sum_b PTDF[l,b] * (sum_{g at b} p[g] - D[b])
      -flow_limit[l] <= f[l] <= flow_limit[l]                       no ocost
      -flow_limit[l] - s[l] <= f[l] <= flow_limit[l] + s[l]          with ocost

  [D] GTC limits when use_gtc=True:
      q[k] = sum_{l in k} sign[k,l] * f[l]
      -gtc_limit[k] <= q[k] <= gtc_limit[k]

  [E] Ramp limits when use_ramp=True and u[g] = 1:
      p[g] <= Pmax[g]                      if starting from offline
      p[g] <= p_prev[g] + ramp_up[g]       otherwise
      p[g] >= p_prev[g] - ramp_down[g]     if previously online

LMP extraction:
  The balance dual gives the energy component. Line and GTC duals are mapped
  back to buses with PTDFs, then settlement-point LMPs are aggregated from
  bus LMPs.

Shadow price / dual variable sign convention
=============================================

scipy linprog returns ineqlin.marginals[i] = dCost/db_i for each row of
A_ub @ x <= b_ub.  This value is non-positive when the constraint is binding
(relaxing a binding constraint saves cost in a minimization problem).

We convert to the economic convention by negating, so that shadow prices are
non-negative when the associated constraint is binding and costly:

  mu_plus  = -ineq_duals[idx_up]  >= 0   (shadow price of forward limit)
  mu_minus = -ineq_duals[idx_lo]  >= 0   (shadow price of backward limit)
  mu_l = mu_plus - mu_minus

Sign meaning for line_duals[l]:
  mu_l > 0  : forward limit binding (flow = +F_max, from_bus → to_bus)
  mu_l < 0  : backward limit binding (flow = -F_max, from_bus ← to_bus)
  mu_l = 0  : line is uncongested
  |mu_l|    : $/MWh opportunity cost = (expensive constrained gen) - (cheap remote gen)

LMP decomposition:
  LMP[b] = lam + cong[b]
  cong[b] = -sum_l  mu_l * PTDF[l, b]
           - sum_k  mu_k * sum_{l in k} sign[k,l] * PTDF[l, b]

The minus sign in cong[b] is the key identity.  Example:
  Forward congestion (mu_l > 0): cheap gen at from_bus trying to reach load at to_bus.
  PTDF[l, to_bus] < 0 (injecting locally at to_bus reduces forward flow, relieves
  congestion).  Therefore cong[to_bus] = -mu_l * PTDF[l, to_bus] > 0, so
  LMP[to_bus] = lam + cong > lam: the load pocket pays a congestion premium.

Two-bus worked example (from_bus=1 slack, to_bus=2 load):
  PTDF[L1, bus1] = 0    (slack bus is always zero by PTDF construction)
  PTDF[L1, bus2] = -1   (injecting at bus2 reduces forward flow on L1 by 1 MW)

  Case A — forward congestion (cheap gen at bus1, load at bus2, line at +F_max):
    mu_L1 = expensive_cost - cheap_cost  > 0  (e.g. +50 $/MWh)
    lam   = cheap_cost                        (e.g.  30 $/MWh)
    LMP[bus1] = lam - mu_L1 * 0  = lam                    =  30 $/MWh (source)
    LMP[bus2] = lam - mu_L1 * (-1) = lam + mu_L1          =  80 $/MWh (load pocket)

  Case B — backward congestion (cheap gen at bus2, load at bus1, line at -F_max):
    mu_L1 = -(expensive_cost - cheap_cost)  < 0  (e.g. -50 $/MWh)
    lam   = expensive_cost                        (e.g.  80 $/MWh)
    LMP[bus1] = lam - mu_L1 * 0  = lam                    =  80 $/MWh (load pocket)
    LMP[bus2] = lam - mu_L1 * (-1) = lam + mu_L1          =  30 $/MWh (gen pocket)

  Case C — no congestion:
    mu_L1 = 0
    LMP[bus1] = LMP[bus2] = lam (uniform energy price)

Internal architecture
=====================
The LP is split into two stages for performance:

  _build_base_lp()   — builds sections [A]-[E] once per hour.
                       Returns a _BaseLPCache namedtuple (immutable numpy arrays).

  _solve_with_ctg()  — appends section [F] (active N-1 contingency rows) to
                       the cached base, calls HiGHS, and extracts all results.

solve_hourly_sced() calls both in sequence (single-hour, backward-compatible).
solve_hourly_sced_n1() (in n1_screening.py) calls _build_base_lp once and
_solve_with_ctg on every iteration, avoiding redundant LP reconstruction.
"""

import numpy as np
import pandas as pd
from collections import namedtuple
from scipy.optimize import linprog


# ── Base LP cache ─────────────────────────────────────────────────────────────
# Holds the immutable LP components built from sections [A]-[E].
# The N-1 wrapper reuses this across iterations; only section [F] varies.

_BaseLPCache = namedtuple("_BaseLPCache", [
    "c", "A_ub", "b_ub",               # cost vector and inequality system
    "A_eq", "b_eq",                     # equality (power balance)
    "bounds",                           # variable bounds (list of tuples)
    "n_vars", "SHED_IDX",              # variable count and shed index
    "g_idx", "gen_at_bus",             # generator index maps
    "effective_load", "load_arr",       # load data
    "line_row_index", "gtc_row_index", # row index maps for dual extraction
])


def _build_base_lp(
    generators, gen_list, bus_list, lines, line_list, ptdf,
    gtcs, u_fixed, p_prev, load_h, pmax_h, cost_h, battery_injection_h,
    use_network, use_ramp, use_gtc,
) -> _BaseLPCache:
    """Build LP sections [A]-[E] without contingency constraints.

    Returns an immutable _BaseLPCache. Call once per operating hour and
    reuse across N-1 iterations by appending section [F] in _solve_with_ctg.
    """
    n_gen = len(gen_list)
    g_idx = {g: i for i, g in enumerate(gen_list)}
    gen_at_bus = {b: [g for g in gen_list if generators[g]["bus"] == b]
                  for b in bus_list}

    if battery_injection_h is not None:
        effective_load = load_h - battery_injection_h.reindex(bus_list).fillna(0)
    else:
        effective_load = load_h

    load_arr = effective_load.reindex(bus_list).values   # (n_bus,) — computed once

    SHED_PRICE = 5000.0
    n_vars = n_gen + 1
    SHED_IDX = n_gen
    total_load = float(effective_load.sum())

    c = np.zeros(n_vars)
    for i, g in enumerate(gen_list):
        c[i] = float(cost_h[g])
    c[SHED_IDX] = SHED_PRICE

    # [A] Power balance
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_gen] = 1.0
    A_eq[0, SHED_IDX] = 1.0
    b_eq = np.array([total_load])

    # [B] Bounds
    bounds = []
    for g in gen_list:
        gd = generators[g]
        u = u_fixed[g]
        bounds.append((gd["Pmin"] * u, float(pmax_h[g]) * u))
    bounds.append((0.0, max(total_load, 0.01)))

    A_ub_rows = []
    b_ub_rows = []
    line_row_index = {}

    # [C] Line constraints (with optional per-line OCOST slack)
    if use_network:
        lines_with_ocost = [l for l in line_list if lines[l].get("ocost", 0) > 0]
        n_slack = len(lines_with_ocost)
        if n_slack > 0:
            c = np.concatenate([c, [lines[l]["ocost"] for l in lines_with_ocost]])
            for _ in range(n_slack):
                bounds.append((0.0, None))
            A_eq = np.hstack([A_eq, np.zeros((1, n_slack))])
            slack_idx = {l: n_gen + 1 + i for i, l in enumerate(lines_with_ocost)}
            n_vars += n_slack
        else:
            slack_idx = {}

        for k, line_id in enumerate(line_list):
            f_max = lines[line_id]["flow_limit"]
            row = np.zeros(n_vars)
            load_offset = 0.0
            for b_idx, b in enumerate(bus_list):
                ptdf_kb = ptdf[k, b_idx]
                load_offset += ptdf_kb * load_arr[b_idx]
                for g in gen_at_bus[b]:
                    row[g_idx[g]] += ptdf_kb
            if line_id in slack_idx:
                row_pos = row.copy(); row_pos[slack_idx[line_id]] = -1.0
                row_neg = -row.copy(); row_neg[slack_idx[line_id]] = -1.0
                A_ub_rows.append(row_pos)
                b_ub_rows.append(f_max + load_offset)
                A_ub_rows.append(row_neg)
                b_ub_rows.append(f_max - load_offset)
            else:
                A_ub_rows.append(row.copy())
                b_ub_rows.append(f_max + load_offset)
                A_ub_rows.append(-row.copy())
                b_ub_rows.append(f_max - load_offset)
            line_row_index[line_id] = (len(A_ub_rows) - 2, len(A_ub_rows) - 1)

    # [D] GTC constraints
    gtc_row_index = {}
    if use_gtc and gtcs:
        for gtc_id, gtc in gtcs.items():
            gtc_limit = gtc["flow_limit"]
            row = np.zeros(n_vars)
            load_offset_total = 0.0
            for line_id, sign in gtc["monitored_lines"]:
                if line_id not in line_list:
                    continue
                k = line_list.index(line_id)
                for b_idx, b in enumerate(bus_list):
                    ptdf_kb = ptdf[k, b_idx]
                    load_offset_total += sign * ptdf_kb * load_arr[b_idx]
                    for g in gen_at_bus[b]:
                        row[g_idx[g]] += sign * ptdf_kb
            A_ub_rows.append(row.copy())
            b_ub_rows.append(gtc_limit + load_offset_total)
            A_ub_rows.append(-row.copy())
            b_ub_rows.append(gtc_limit - load_offset_total)
            gtc_row_index[gtc_id] = (len(A_ub_rows) - 2, len(A_ub_rows) - 1)

    # [E] Ramp constraints
    if use_ramp:
        for g in gen_list:
            gd = generators[g]
            u_now = u_fixed[g]
            if u_now == 0:
                continue
            pp = p_prev[g]
            u_prev = 1 if pp > 1e-3 else 0
            su_flag = (u_prev == 0 and u_now == 1)
            RU, RD = gd["ramp_up"], gd["ramp_down"]
            ramp_up_lim = gd["Pmax"] if su_flag else (pp + RU)
            row = np.zeros(n_vars)
            row[g_idx[g]] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(ramp_up_lim)
            if not su_flag and u_prev == 1:
                row2 = np.zeros(n_vars)
                row2[g_idx[g]] = -1.0
                A_ub_rows.append(row2)
                b_ub_rows.append(RD - pp)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    return _BaseLPCache(
        c=c, A_ub=A_ub, b_ub=b_ub,
        A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        n_vars=n_vars, SHED_IDX=SHED_IDX,
        g_idx=g_idx, gen_at_bus=gen_at_bus,
        effective_load=effective_load, load_arr=load_arr,
        line_row_index=line_row_index, gtc_row_index=gtc_row_index,
    )


def _solve_with_ctg(
    cache: _BaseLPCache,
    active_ctg_pairs: set,
    contingency_ptdf: dict,
    ctg_violation_cost: float,
    lines: dict,
    line_list: list,
    bus_list: list,
    gen_list: list,
    settlement_points: dict,
    ptdf: np.ndarray,
    hour,
    gtcs: dict,
    use_network: bool,
    battery_injection_h,
) -> dict:
    """Append [F] contingency rows to a cached base LP and solve.

    Does not modify the cache — creates new arrays by concatenation/padding,
    so the same cache can be safely reused across N-1 iterations.
    """
    # Start from cached components (no mutation)
    c      = cache.c
    A_ub   = cache.A_ub
    b_ub   = cache.b_ub
    A_eq   = cache.A_eq
    bounds = cache.bounds       # list — reassigned (not mutated) if ctg slacks added
    n_vars = cache.n_vars
    ctg_row_index_n1 = {}

    # [F] Active N-1 contingency constraints (soft, penalised) ─────────────────
    if active_ctg_pairs and contingency_ptdf and use_network:
        valid_ctg_list = sorted(
            (cid, lid)
            for cid, lid in active_ctg_pairs
            if cid in contingency_ptdf and lid in line_list
        )
        n_ctg = len(valid_ctg_list)
        if n_ctg > 0:
            c      = np.concatenate([c, [ctg_violation_cost] * n_ctg])
            bounds = list(bounds) + [(0.0, None)] * n_ctg
            A_eq   = np.hstack([A_eq, np.zeros((1, n_ctg))])
            n_new  = n_vars + n_ctg

            line_idx_map  = {lid: k for k, lid in enumerate(line_list)}
            base_n_rows   = cache.A_ub.shape[0] if cache.A_ub is not None else 0
            ctg_A, ctg_b  = [], []

            for i, (cid, lid) in enumerate(valid_ctg_list):
                k         = line_idx_map[lid]
                ctg_limit = lines[lid].get("contingency_limit") or lines[lid]["flow_limit"]
                ptdf_row  = contingency_ptdf[cid][k, :]
                slack_col = n_vars + i

                row = np.zeros(n_new)
                load_offset = 0.0
                for b_idx, b in enumerate(bus_list):
                    p_kb = ptdf_row[b_idx]
                    load_offset += p_kb * cache.load_arr[b_idx]
                    for g in cache.gen_at_bus[b]:
                        row[cache.g_idx[g]] += p_kb

                row_pos = row.copy(); row_pos[slack_col] = -1.0
                row_neg = -row.copy(); row_neg[slack_col] = -1.0
                ctg_A.extend([row_pos, row_neg])
                ctg_b.extend([ctg_limit + load_offset, ctg_limit - load_offset])
                ctg_row_index_n1[(cid, lid)] = (base_n_rows + 2*i, base_n_rows + 2*i + 1)

            ctg_A_arr = np.array(ctg_A)
            ctg_b_arr = np.array(ctg_b)
            if cache.A_ub is not None:
                # np.pad avoids the per-row list-comprehension copy
                A_ub = np.vstack([np.pad(cache.A_ub, ((0, 0), (0, n_ctg))), ctg_A_arr])
                b_ub = np.concatenate([cache.b_ub, ctg_b_arr])
            else:
                A_ub = ctg_A_arr
                b_ub = ctg_b_arr
            n_vars = n_new

    # ── Solve ────────────────────────────────────────────────────────────────
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=cache.b_eq,
                  bounds=bounds, method="highs", options={"disp": False})

    if res.status != 0:
        return {"status": res.message, "hour": hour}

    # ── Extract dispatch & duals ─────────────────────────────────────────────
    g_idx       = cache.g_idx
    gen_at_bus  = cache.gen_at_bus
    SHED_IDX    = cache.SHED_IDX

    dispatch     = {g: float(res.x[g_idx[g]]) for g in gen_list}
    load_shed_mw = float(res.x[SHED_IDX])
    lam          = float(res.eqlin.marginals[0])
    ineq_duals   = res.ineqlin.marginals if A_ub is not None else np.array([])

    def _dual(idx_up, idx_lo):
        return (-ineq_duals[idx_up]) - (-ineq_duals[idx_lo])

    line_duals = {lid: _dual(u, l) for lid, (u, l) in cache.line_row_index.items()}
    gtc_duals  = {gid: _dual(u, l) for gid, (u, l) in cache.gtc_row_index.items()}
    ctg_line_duals = {pair: _dual(u, l) for pair, (u, l) in ctg_row_index_n1.items()}

    # ── Line flows (vectorised) ───────────────────────────────────────────────
    line_flows = {}
    if use_network:
        gen_inj = np.array([sum(dispatch[g] for g in gen_at_bus[b]) for b in bus_list])
        net_inj = gen_inj - cache.load_arr
        flows   = ptdf @ net_inj          # single matrix-vector product
        line_flows = {lid: float(flows[k]) for k, lid in enumerate(line_list)}

    binding_lines = [l for l, mu in line_duals.items() if abs(mu) > 0.01]
    binding_gtcs  = [g for g, mu in gtc_duals.items()  if abs(mu) > 0.01]

    # ── LMP at each bus (vectorised) ─────────────────────────────────────────
    # cong[b] = -Σ_l mu_l * PTDF[l,b]  →  cong = -(mu_arr @ ptdf)
    mu_line_arr = np.array([line_duals.get(lid, 0.0) for lid in line_list])
    cong_arr    = -(mu_line_arr @ ptdf)   # (n_bus,)

    # GTC component (typically few GTCs — kept as loop)
    for gid, mu_gtc in gtc_duals.items():
        for lid, sign in gtcs[gid]["monitored_lines"]:
            if lid in line_list:
                k = line_list.index(lid)
                cong_arr -= mu_gtc * sign * ptdf[k, :]

    bus_lmp  = {b: lam + float(cong_arr[i]) for i, b in enumerate(bus_list)}
    bus_cong = {b: float(cong_arr[i])        for i, b in enumerate(bus_list)}

    # ── Settlement-point LMP aggregation ─────────────────────────────────────
    sp_lmp = {}
    for sp_name, sp in settlement_points.items():
        t = sp["type"]
        if t in ("resource_node", "load_zone"):
            sp_lmp[sp_name] = bus_lmp[sp["bus"]]
        elif t == "load_zone_weighted":
            sp_lmp[sp_name] = float(sum(bus_lmp[b] * w for b, w in sp["member_buses"].items()))
        elif t == "hub":
            sp_lmp[sp_name] = float(np.mean([bus_lmp[b] for b in sp["member_buses"]]))

    return {
        "status"              : "optimal",
        "hour"                : hour,
        "dispatch"            : dispatch,
        "load_shed_MW"        : load_shed_mw,
        "battery_inj_total_MW": float(battery_injection_h.sum()) if battery_injection_h is not None else 0.0,
        "bus_lmp"             : bus_lmp,
        "sp_lmp"              : sp_lmp,
        "lmp_energy"          : lam,
        "lmp_cong_bus"        : bus_cong,
        "line_duals"          : line_duals,
        "gtc_duals"           : gtc_duals,
        "ctg_line_duals"      : ctg_line_duals,
        "line_flows"          : line_flows,
        "binding_lines"       : binding_lines,
        "binding_gtcs"        : binding_gtcs,
        "obj"                 : float(res.fun),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def solve_hourly_sced(
    generators: dict,
    gen_list: list,
    bus_list: list,
    bus_idx: dict,
    lines: dict,
    line_list: list,
    ptdf: np.ndarray,
    settlement_points: dict,
    gtcs: dict = None,
    hour: pd.Timestamp = None,
    u_fixed: dict = None,
    p_prev: dict = None,
    load_h: pd.Series = None,
    pmax_h: pd.Series = None,
    cost_h: pd.Series = None,
    battery_injection_h: pd.Series = None,
    use_network: bool = True,
    use_ramp: bool = True,
    use_gtc: bool = True,
    contingency_ptdf: dict = None,
    active_ctg_pairs: set = None,
    ctg_violation_cost: float = 1000.0,
) -> dict:
    """Solve one hour of SCED and return dispatch + LMPs + duals.

    Thin wrapper: builds the base LP once via _build_base_lp, then appends
    any active N-1 contingency rows and solves via _solve_with_ctg.
    For iterative N-1 use (multiple re-solves per hour), prefer
    solve_hourly_sced_n1 which reuses the base LP across iterations.
    """
    gtcs = gtcs or {}
    cache = _build_base_lp(
        generators, gen_list, bus_list, lines, line_list, ptdf,
        gtcs, u_fixed, p_prev, load_h, pmax_h, cost_h, battery_injection_h,
        use_network, use_ramp, use_gtc,
    )
    return _solve_with_ctg(
        cache=cache,
        active_ctg_pairs=active_ctg_pairs or set(),
        contingency_ptdf=contingency_ptdf or {},
        ctg_violation_cost=ctg_violation_cost,
        lines=lines, line_list=line_list, bus_list=bus_list,
        gen_list=gen_list, settlement_points=settlement_points,
        ptdf=ptdf, hour=hour, gtcs=gtcs,
        use_network=use_network, battery_injection_h=battery_injection_h,
    )
