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
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def solve_hourly_sced(
    generators: dict,                  # {gen_id: {bus, Pmin, Pmax, ...}}
    gen_list: list,                    # ordered generator IDs
    bus_list: list,                    # ordered bus IDs
    bus_idx: dict,                     # {bus_id: index}
    lines: dict,                       # {line_id: {from_bus, to_bus, flow_limit, ocost, ...}}
    line_list: list,                   # ordered line IDs
    ptdf: np.ndarray,                  # [n_lines, n_buses]
    settlement_points: dict,           # {sp_name: {type, bus/member_buses, ...}}
    gtcs: dict = None,                 # {gtc_id: {monitored_lines, flow_limit, ...}}
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
) -> dict:
    
    if gtcs is None:
        gtcs = {}

    n_gen = len(gen_list)
    n_bus = len(bus_list)
    n_line = len(line_list)
    g_idx = {g: i for i, g in enumerate(gen_list)}
    gen_at_bus = {b: [g for g in gen_list if generators[g]["bus"] == b]
                 for b in bus_list}

    # ── Apply battery: effective_load = load - battery_injection ─────────────
    if battery_injection_h is not None:
        effective_load = load_h - battery_injection_h.reindex(bus_list).fillna(0)
    else:
        effective_load = load_h

    # ── Build LP ─────────────────────────────────────────────────────────────
    SHED_PRICE = 5000.0
    n_vars = n_gen + 1
    SHED_IDX = n_gen

    c = np.zeros(n_vars)
    for i, g in enumerate(gen_list):
        c[i] = float(cost_h[g])
    c[SHED_IDX] = SHED_PRICE

    # [A] Equality: power balance Σ p_g + shed = total_effective_load
    total_load = float(effective_load.sum())
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_gen] = 1.0
    A_eq[0, SHED_IDX] = 1.0
    b_eq = np.array([total_load])

    # [B] Bounds
    bounds = []
    for g in gen_list:
        gd = generators[g]
        u = u_fixed[g]
        pmax = float(pmax_h[g])
        bounds.append((gd["Pmin"] * u, pmax * u))
    bounds.append((0.0, max(total_load, 0.01)))

    A_ub_rows = []
    b_ub_rows = []
    line_row_index = {}

    # [C] Line constraints with OCOST soft constraint ──────────────────
    # For lines with OCOST defined, allow overload at penalty price OCOST_l
    # Adds N_LINES_WITH_OCOST extra slack variables.
    # μ_l is bounded above by OCOST_l (allows binding at controlled price).
    if use_network:
        load_arr = effective_load.reindex(bus_list).values
        # Extend variable vector with line slacks if any line has OCOST
        lines_with_ocost = [l for l in line_list if lines[l].get("ocost", 0) > 0]
        n_slack = len(lines_with_ocost)
        if n_slack > 0:
            # Extend cost vector and bounds
            c = np.concatenate([c, np.array([lines[l]["ocost"] for l in lines_with_ocost])])
            for _ in range(n_slack):
                bounds.append((0.0, None))   # slack ≥ 0, no upper bound
            # Extend equality constraint matrix (slacks don't enter power balance)
            A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0], n_slack))])
            slack_idx = {l: n_gen + 1 + i for i, l in enumerate(lines_with_ocost)}
            n_vars = n_vars + n_slack
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

            # If this line has OCOST, subtract slack (allow overload at penalty)
            if line_id in slack_idx:
                row_pos = row.copy()
                row_pos[slack_idx[line_id]] = -1.0   # f_l - s_l ≤ F_max
                row_neg = -row.copy()
                row_neg[slack_idx[line_id]] = -1.0   # -f_l - s_l ≤ F_max (reverse direction)
                A_ub_rows.append(row_pos)
                b_ub_rows.append(f_max + load_offset)
                A_ub_rows.append(row_neg)
                b_ub_rows.append(f_max - load_offset)
            else:
                # Hard constraint (no OCOST defined)
                A_ub_rows.append(row.copy())
                b_ub_rows.append(f_max + load_offset)
                A_ub_rows.append(-row.copy())
                b_ub_rows.append(f_max - load_offset)
            line_row_index[line_id] = (len(A_ub_rows) - 2, len(A_ub_rows) - 1)

    # [D] GTC constraints ─────────────────────────────────────────────────────
    # GTC: Σ_l (sign_l · flow_l) ≤ gtc_limit
    # Substituting flow_l = Σ_b ptdf[l,b] · (gen[b] - load[b]):
    # Σ_l sign_l · Σ_b ptdf[l,b] · (gen[b] - load[b]) ≤ gtc_limit
    # The LHS in terms of decision vars:
    # Σ_g (Σ_l sign_l · ptdf[l, bus(g)]) · p[g]  ≤  gtc_limit + Σ_l sign_l · load_offset_l
    gtc_row_index = {}
    if use_gtc:
        load_arr = effective_load.reindex(bus_list).values
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

            # Forward: Σ sign·flow ≤ limit
            A_ub_rows.append(row.copy())
            b_ub_rows.append(gtc_limit + load_offset_total)
            # Reverse: -Σ sign·flow ≤ limit
            A_ub_rows.append(-row.copy())
            b_ub_rows.append(gtc_limit - load_offset_total)
            gtc_row_index[gtc_id] = (len(A_ub_rows) - 2, len(A_ub_rows) - 1)

    # [E] Ramp constraints ─────────────────────────────────────────────────────
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
            Pmax = gd["Pmax"]

            ramp_up_lim = Pmax if su_flag else (pp + RU)
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

    # ── Solve ────────────────────────────────────────────────────────────────
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs", options={"disp": False})

    if res.status != 0:
        return {"status": res.message, "hour": hour}

    # ── Extract dispatch & duals ─────────────────────────────────────────────
    dispatch = {g: float(res.x[g_idx[g]]) for g in gen_list}
    load_shed_mw = float(res.x[SHED_IDX])
    lam = float(res.eqlin.marginals[0])
    ineq_duals = res.ineqlin.marginals if A_ub is not None else np.array([])

    # Line shadow prices
    line_duals = {}
    for line_id, (idx_up, idx_lo) in line_row_index.items():
        mu_plus  = -ineq_duals[idx_up]
        mu_minus = -ineq_duals[idx_lo]
        line_duals[line_id] = mu_plus - mu_minus

    # GTC shadow prices (NEW)
    gtc_duals = {}
    for gtc_id, (idx_up, idx_lo) in gtc_row_index.items():
        mu_plus  = -ineq_duals[idx_up]
        mu_minus = -ineq_duals[idx_lo]
        gtc_duals[gtc_id] = mu_plus - mu_minus

    # Line flows
    line_flows = {}
    if use_network:
        load_arr = effective_load.reindex(bus_list).values
        for k, line_id in enumerate(line_list):
            flow = 0.0
            for b_idx, b in enumerate(bus_list):
                net_b = sum(dispatch[g] for g in gen_at_bus[b]) - load_arr[b_idx]
                flow += ptdf[k, b_idx] * net_b
            line_flows[line_id] = flow

    binding_lines = [l for l, mu in line_duals.items() if abs(mu) > 0.01]
    binding_gtcs  = [g for g, mu in gtc_duals.items()  if abs(mu) > 0.01]

    # ── LMP at each bus ──────────────────────────────────────────────────────
    # LMP includes line congestion AND GTC components
    bus_lmp = {}
    bus_cong = {}
    for b_idx, b in enumerate(bus_list):
        cong = 0.0
        # Line component: LMP_b = lambda - Σ mu_l * PTDF[l,b]
        # The minus sign ensures load-pocket buses (where injecting locally
        # relieves congestion) have HIGHER LMPs, not lower.
        for k, line_id in enumerate(line_list):
            cong -= line_duals.get(line_id, 0) * ptdf[k, b_idx]
        # GTC component
        for gtc_id, mu_gtc in gtc_duals.items():
            gtc = gtcs[gtc_id]
            for line_id, sign in gtc["monitored_lines"]:
                if line_id in line_list:
                    k = line_list.index(line_id)
                    cong -= mu_gtc * sign * ptdf[k, b_idx]
        bus_lmp[b]  = lam + cong
        bus_cong[b] = cong

    # SP aggregation
    sp_lmp = {}
    for sp_name, sp in settlement_points.items():
        if sp["type"] == "resource_node":
            sp_lmp[sp_name] = bus_lmp[sp["bus"]]
        elif sp["type"] == "load_zone":
            # Single-bus load zone (legacy)
            sp_lmp[sp_name] = bus_lmp[sp["bus"]]
        elif sp["type"] == "load_zone_weighted":
            # Multi-bus weighted load zone (structure 5: enables LZ vs HB basis)
            members = sp["member_buses"]   # {bus: weight}
            sp_lmp[sp_name] = float(sum(bus_lmp[b] * w for b, w in members.items()))
        elif sp["type"] == "hub":
            # Hub = simple arithmetic mean of gen-side buses
            members = sp["member_buses"]
            sp_lmp[sp_name] = float(np.mean([bus_lmp[b] for b in members]))

    return {
        "status"        : "optimal",
        "hour"          : hour,
        "dispatch"      : dispatch,
        "load_shed_MW"  : load_shed_mw,
        "battery_inj_total_MW": (
            float(battery_injection_h.sum()) if battery_injection_h is not None else 0.0
        ),
        "bus_lmp"       : bus_lmp,
        "sp_lmp"        : sp_lmp,
        "lmp_energy"    : lam,
        "lmp_cong_bus"  : bus_cong,
        "line_duals"    : line_duals,
        "gtc_duals"     : gtc_duals,
        "line_flows"    : line_flows,
        "binding_lines" : binding_lines,
        "binding_gtcs"  : binding_gtcs,
        "obj"           : float(res.fun),
    }
