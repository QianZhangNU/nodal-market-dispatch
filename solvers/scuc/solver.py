"""
Daily SCUC Solver  (24-hour MIP)
==================================
Solves a single-day SCUC problem given:
  - System state at start of day (which units are on/off, output level)
  - Forecasted load + renewable Pmax(t) for next 24 hours
  - Generator parameters (cost, ramp, min-up/down, etc.)
  - Network constraints (ptdf, line limits, contingencies)

Returns:
  - Commitment schedule u[g, h]  for h ∈ {1..24}
  - Initial dispatch p[g, h] (will be updated by hourly SCED)
  - Total expected cost
  - End-of-day state for next-day initialization

Variable layout (flattened for scipy.optimize.milp):
  [u_g0_h0, u_g0_h1, ..., u_gN_h23,  ← n_gen × 24  binary commitment
   v_g0_h0, ...                        ← n_gen × 24  binary startup
   w_g0_h0, ...                        ← n_gen × 24  binary shutdown
   p_g0_h0, ...]                       ← n_gen × 24  continuous dispatch
"""

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csc_matrix
import time




# ─────────────────────────────────────────────────────────────────────────────
#  Helper indexing
# ─────────────────────────────────────────────────────────────────────────────


def _build_offsets(n_gen: int, n_hours: int = 24):
    """Variable offsets in the flattened decision vector."""
    N_block = n_gen * n_hours
    return {
        "u": 0,                      # commitment
        "v": N_block,                # startup
        "w": 2 * N_block,            # shutdown
        "p": 3 * N_block,            # dispatch
        "block_size": N_block,
        "total":      4 * N_block,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Build & solve daily SCUC
# ─────────────────────────────────────────────────────────────────────────────
def solve_daily_scuc(
    generators: dict,                  # {gen_id: {bus, Pmin, Pmax, cost_b, ...}}
    gen_list: list,                    # ordered generator IDs
    bus_list: list,                    # ordered bus IDs
    bus_idx: dict,                     # {bus_id: index}
    lines: dict,                       # {line_id: {from_bus, to_bus, flow_limit, ...}}
    ptdf: np.ndarray,                  # [n_lines, n_buses]
    day_start: pd.Timestamp,
    load_24h: pd.DataFrame,
    pmax_24h: pd.DataFrame,
    gen_cost_24h: pd.DataFrame,
    initial_state: dict,
    use_network: bool = True,
    reserve_pct: float = 0.07,
    mip_gap: float = 0.005,
    time_limit_sec: float = 60.0,
    verbose: bool = False,
    scuc_line_penalty: float = 5000.0
) -> dict:
    """
    Solve 24-hour SCUC for one day.
    When custom_lines is provided, uses those instead of global LINES dict
    (enables monthly topology variation with outages and seasonal ratings).

    Returns dict with:
        u_schedule  : DataFrame [24, n_gen]  binary commitment
        p_dispatch  : DataFrame [24, n_gen]  continuous dispatch (MW)
        startup     : DataFrame [24, n_gen]  startup events
        shutdown    : DataFrame [24, n_gen]  shutdown events
        total_cost  : float
        end_state   : dict for next-day init
        solve_time  : float (seconds)
        status      : str
    """
    lines_used     = lines
    line_list_used = list(lines_used.keys())
    n_line_used    = len(line_list_used)

    n_gen = len(gen_list)
    n_bus = len(bus_list)
    g_idx = {g: i for i, g in enumerate(gen_list)}

    n_h = 24
    off = _build_offsets(n_gen, n_h)
    n_vars = off["total"]

    hours = pd.date_range(day_start, periods=n_h, freq="h")

    # ── 1. Objective coefficients ────────────────────────────────────────────
    c = np.zeros(n_vars)
    for h in range(n_h):
        for g in gen_list:
            gd = generators[g]
            cost_b_h = float(gen_cost_24h.iloc[h][g])
            # Variable cost: per MW
            c[off["p"] + g_idx[g] * n_h + h] = cost_b_h
            # No-load cost
            c[off["u"] + g_idx[g] * n_h + h] = gd["cost_c"]
            # Startup cost
            c[off["v"] + g_idx[g] * n_h + h] = gd["su_cost"]
            # Shutdown cost
            c[off["w"] + g_idx[g] * n_h + h] = gd["sd_cost"]

    # ── 2. Build constraint matrix in COO format (faster than per-row append) ─
    # We use lil_matrix then convert to CSC at the end
    rows_data = []   # list of (col_indices, values, lb, ub)

    def add(cols, vals, lb, ub):
        rows_data.append((np.array(cols, dtype=int),
                          np.array(vals, dtype=float), lb, ub))

    def idx(var: str, g: str, h: int) -> int:
        return off[var] + g_idx[g] * n_h + h

    # ── [A] Logical linking: v - w = u(t) - u(t-1) ──────────────────────────
    for g in gen_list:
        u0 = initial_state[g]["u"]
        for h in range(n_h):
            cols = [idx("v", g, h), idx("w", g, h), idx("u", g, h)]
            vals = [1.0, -1.0, -1.0]
            if h == 0:
                rhs = -u0
            else:
                cols.append(idx("u", g, h-1))
                vals.append(1.0)
                rhs = 0.0
            add(cols, vals, rhs, rhs)

            # v + w ≤ 1 (no simultaneous start+stop)
            add([idx("v", g, h), idx("w", g, h)], [1.0, 1.0], -np.inf, 1.0)

    # ── [B] Output limits: Pmin·u ≤ p ≤ Pmax(t)·u ───────────────────────────
    for g in gen_list:
        gd = generators[g]
        for h in range(n_h):
            pmax_h = float(pmax_24h.iloc[h][g])
            # p ≥ Pmin·u  ⇔  -p + Pmin·u ≤ 0
            add([idx("p", g, h), idx("u", g, h)],
                [-1.0, gd["Pmin"]], -np.inf, 0.0)
            # p ≤ Pmax(h)·u  ⇔  p - Pmax(h)·u ≤ 0
            add([idx("p", g, h), idx("u", g, h)],
                [1.0, -pmax_h], -np.inf, 0.0)

    # ── [C] Ramp constraints ────────────────────────────────────────────────
    for g in gen_list:
        gd = generators[g]
        RU, RD = gd["ramp_up"], gd["ramp_down"]
        Pmax = gd["Pmax"]
        p0 = initial_state[g]["p"]
        u0 = initial_state[g]["u"]

        for h in range(n_h):
            if h == 0:
                # Ramp up: p[0] - p0 ≤ RU·u0 + Pmax·v[0]
                add([idx("p", g, 0), idx("v", g, 0)],
                    [1.0, -Pmax], -np.inf, RU * u0 + p0)
                # Ramp down: p0 - p[0] ≤ RD·u[0] + Pmax·w[0]
                add([idx("p", g, 0), idx("u", g, 0), idx("w", g, 0)],
                    [-1.0, -RD, -Pmax], -np.inf, -p0)
            else:
                add([idx("p", g, h), idx("p", g, h-1),
                     idx("u", g, h-1), idx("v", g, h)],
                    [1.0, -1.0, -RU, -Pmax], -np.inf, 0.0)
                add([idx("p", g, h-1), idx("p", g, h),
                     idx("u", g, h), idx("w", g, h)],
                    [1.0, -1.0, -RD, -Pmax], -np.inf, 0.0)

    # ── [D] Min up/down (tight Rajan-Takriti formulation) ───────────────────
    for g in gen_list:
        gd = generators[g]
        UT = min(gd["min_up"], n_h)
        DT = min(gd["min_down"], n_h)

        # Initial state enforcement
        u0 = initial_state[g]["u"]
        ut0 = initial_state[g]["up_time"]
        dt0 = initial_state[g]["down_time"]

        if u0 == 1 and gd["min_up"] > ut0:
            must_on = min(gd["min_up"] - ut0, n_h)
            for h in range(must_on):
                add([idx("u", g, h)], [1.0], 1.0, 1.0)
        elif u0 == 0 and gd["min_down"] > dt0:
            must_off = min(gd["min_down"] - dt0, n_h)
            for h in range(must_off):
                add([idx("u", g, h)], [1.0], 0.0, 0.0)

        # Min up: Σ_{τ=h-UT+1..h} v[g,τ] ≤ u[g,h]
        for h in range(UT - 1, n_h):
            cols = [idx("v", g, tau) for tau in range(h - UT + 1, h + 1)]
            cols.append(idx("u", g, h))
            vals = [1.0] * UT + [-1.0]
            add(cols, vals, -np.inf, 0.0)

        # Min down: Σ_{τ=h-DT+1..h} w[g,τ] ≤ 1 - u[g,h]
        for h in range(DT - 1, n_h):
            cols = [idx("w", g, tau) for tau in range(h - DT + 1, h + 1)]
            cols.append(idx("u", g, h))
            vals = [1.0] * DT + [1.0]
            add(cols, vals, -np.inf, 1.0)

    # ── [E] System power balance: Σ_g p[g,h] + load_shed[h] = Σ_b D[b,h] ────
    # Add load-shedding slack: a "virtual generator" that can serve any load
    # at a high penalty price ($9000/MWh ≈ ERCOT VOLL). This ensures feasibility
    # during extreme stress and gives meaningful scarcity prices.
    # Layout: append load_shed[h] as last n_h variables
    n_existing_vars = n_vars
    n_vars_with_shed = n_vars + n_h
    SHED_PRICE = 9000.0   # $/MWh — value of lost load proxy

    # Extend objective
    c_ext = np.zeros(n_vars_with_shed)
    c_ext[:n_vars] = c
    c_ext[n_vars:] = SHED_PRICE
    c = c_ext
    n_vars = n_vars_with_shed

    # Update offsets to include load_shed
    off["shed"] = n_existing_vars

    def shed_idx(h):
        return off["shed"] + h

    # Update existing power balance constraints to include load_shed
    # Append load_shed coefficient to each balance row
    # (We'll rebuild balance after collecting all rows below)

    for h in range(n_h):
        total_load = float(load_24h.iloc[h].sum())
        cols = [idx("p", g, h) for g in gen_list] + [shed_idx(h)]
        vals = [1.0] * n_gen + [1.0]
        add(cols, vals, total_load, total_load)

    # ── [F] Spinning reserve: Σ_g (Pmax(h)·u - p) ≥ reserve ──────────────────
    for h in range(n_h):
        total_load = float(load_24h.iloc[h].sum())
        reserve_req = reserve_pct * total_load
        cols = []
        vals = []
        rhs_const = 0.0
        for g in gen_list:
            pmax_h = float(pmax_24h.iloc[h][g])
            cols.append(idx("u", g, h)); vals.append(pmax_h)
            cols.append(idx("p", g, h)); vals.append(-1.0)
        # Σ pmax·u - Σ p ≥ reserve_req
        add(cols, vals, reserve_req, np.inf)

    # ── [G] DC line flow limits (ptdf) with high-penalty slack ──────────────
    # SCUC uses HARD line limits + slack variable per line per hour with HIGH
    # penalty ($5000/MW). This is purely for SCUC feasibility (commitment
    # decision robustness), NOT for shadow price extraction. SCED uses its own
    # separate OCOST values to determine actual market congestion prices.
    #
    # Why this works:
    #   - SCUC slack penalty $5000 >> any realistic dispatch savings
    #     → SCUC will only "use" slack if absolutely needed for feasibility
    #   - SCUC dispatch decisions (which units commit) are robust
    #   - SCED then re-solves with hard limits or true OCOST for accurate prices
    SCUC_LINE_PENALTY = 5000.0   # $/MW — high enough to deter unnecessary use

    if use_network:
        # Add line slack variables: n_lines × n_hours new vars
        n_existing_vars = n_vars
        n_line_slacks = n_line_used * n_h
        n_vars_new = n_vars + n_line_slacks

        # Extend objective
        c_ext = np.zeros(n_vars_new)
        c_ext[:n_vars] = c
        c_ext[n_vars:] = SCUC_LINE_PENALTY
        c = c_ext
        n_vars = n_vars_new
        off["line_slack"] = n_existing_vars

        def line_slack_idx(line_k, h):
            return off["line_slack"] + line_k * n_h + h

        # Build line constraints with slack: |f_l| - slack_l ≤ f_max
        for h in range(n_h):
            load_h = load_24h.iloc[h].values
            for k, line_id in enumerate(line_list_used):
                f_max = lines_used[line_id]["flow_limit"]

                cols, vals = [], []
                load_offset = 0.0
                for b_idx, b in enumerate(bus_list):
                    ptdf_kb = ptdf[k, b_idx]
                    load_offset += ptdf_kb * load_h[b_idx]
                    for g in gen_list:
                        if generators[g]["bus"] == b:
                            cols.append(idx("p", g, h))
                            vals.append(ptdf_kb)

                # Forward: Σ ptdf·p_g - load_offset - slack ≤ f_max
                cols_pos = cols + [line_slack_idx(k, h)]
                vals_pos = vals + [-1.0]
                add(cols_pos, vals_pos, -np.inf, f_max + load_offset)

                # Reverse: -(Σ ptdf·p_g - load_offset) - slack ≤ f_max
                cols_neg = cols + [line_slack_idx(k, h)]
                vals_neg = [-v for v in vals] + [-1.0]
                add(cols_neg, vals_neg, -np.inf, f_max - load_offset)

    # ── Convert to sparse matrix ─────────────────────────────────────────────
    n_rows = len(rows_data)
    A = lil_matrix((n_rows, n_vars))
    lb_vec = np.zeros(n_rows)
    ub_vec = np.zeros(n_rows)
    for i, (cols, vals, lb, ub) in enumerate(rows_data):
        A[i, cols] = vals
        lb_vec[i] = lb
        ub_vec[i] = ub
    A = csc_matrix(A)

    # ── Variable bounds ──────────────────────────────────────────────────────
    var_lb = np.zeros(n_vars)
    var_ub = np.ones(n_vars)
    for h in range(n_h):
        for g in gen_list:
            var_ub[idx("p", g, h)] = generators[g]["Pmax"]
        # Load shedding: 0 ≤ shed[h] ≤ total_load (could shed all if needed)
        max_shed = float(load_24h.iloc[h].sum())
        var_ub[shed_idx(h)] = max_shed
    # Line slack variables (if line constraints with slack are present)
    if "line_slack" in off:
        for k in range(n_line_used):
            for h in range(n_h):
                var_ub[line_slack_idx(k, h)] = np.inf   # unbounded above
    bounds = Bounds(lb=var_lb, ub=var_ub)

    # ── Integrality: u, v, w binary; p, shed continuous ──────────────────────
    integrality = np.zeros(n_vars)
    integrality[off["u"]:off["u"] + 3 * off["block_size"]] = 1

    # ── Solve ────────────────────────────────────────────────────────────────
    if verbose:
        print(f"  SCUC: {n_vars} vars, {n_rows} constraints. Solving...")
    t0 = time.time()
    res = milp(
        c=c,
        constraints=LinearConstraint(A, lb_vec, ub_vec),
        integrality=integrality,
        bounds=bounds,
        options={"disp": False, "mip_rel_gap": mip_gap,
                 "time_limit": time_limit_sec, "presolve": True},
    )
    solve_time = time.time() - t0

    if not res.success:
        return {"status": res.message, "solve_time": solve_time}

    x = res.x

    # ── Extract results ──────────────────────────────────────────────────────
    u_sched = pd.DataFrame(0, index=hours, columns=gen_list, dtype=int)
    v_sched = pd.DataFrame(0, index=hours, columns=gen_list, dtype=int)
    w_sched = pd.DataFrame(0, index=hours, columns=gen_list, dtype=int)
    p_disp  = pd.DataFrame(0.0, index=hours, columns=gen_list, dtype=float)

    col_idx = {g: i for i, g in enumerate(gen_list)}
    for g in gen_list:
        gi = col_idx[g]
        for h in range(n_h):
            u_sched.iat[h, gi] = int(round(x[idx("u", g, h)]))
            v_sched.iat[h, gi] = int(round(x[idx("v", g, h)]))
            w_sched.iat[h, gi] = int(round(x[idx("w", g, h)]))
            p_disp.iat[h, gi]  = float(x[idx("p", g, h)])

    # ── End-of-day state for next-day initialization ─────────────────────────
    end_state = {}
    for g in gen_list:
        u_end = int(u_sched.iloc[-1][g])
        p_end = float(p_disp.iloc[-1][g])

        # Compute up_time / down_time at end of day
        u_series = u_sched[g].values
        if u_end == 1:
            # Count consecutive 1's at end
            up_time = 1
            for h in range(n_h - 2, -1, -1):
                if u_series[h] == 1:
                    up_time += 1
                else:
                    break
            up_time += initial_state[g]["up_time"] if u_series[0] == 1 and \
                       initial_state[g]["u"] == 1 else 0
            down_time = 0
        else:
            down_time = 1
            for h in range(n_h - 2, -1, -1):
                if u_series[h] == 0:
                    down_time += 1
                else:
                    break
            down_time += initial_state[g]["down_time"] if u_series[0] == 0 and \
                         initial_state[g]["u"] == 0 else 0
            up_time = 0

        end_state[g] = {"u": u_end, "p": p_end,
                         "up_time": up_time, "down_time": down_time}

    return {
        "status"     : "optimal",
        "u_schedule" : u_sched,
        "p_dispatch" : p_disp,
        "startup"    : v_sched,
        "shutdown"   : w_sched,
        "total_cost" : float(res.fun),
        "end_state"  : end_state,
        "solve_time" : solve_time,
    }


def build_initial_state(generators: dict) -> dict:
    """Build initial state from generator init_* fields."""
    return {
        g: {
            "u":         gd["init_status"],
            "p":         gd["init_gen"],
            "up_time":   gd["init_up_time"],
            "down_time": gd["init_down_time"],
        }
        for g, gd in generators.items()
    }
