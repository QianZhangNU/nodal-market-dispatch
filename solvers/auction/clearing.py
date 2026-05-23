"""
CRR Auction Clearing Solver (topology-agnostic)
===============================================
Social-welfare-maximizing LP with Simultaneous Feasibility Test (SFT).

Supports four bid types:

  BUY OBL
      Buyer acquires an obligation CRR. It is settled at DART, so payoff can
      be positive or negative. It uses SFT capacity in both directions.

  BUY OPT
      Buyer acquires an option CRR. It is settled at max(DART, 0), so payoff
      is never negative. It uses only positive source-to-sink SFT exposure.

  SELL OBL
      Seller returns/offers obligation CRR capacity and releases signed SFT
      capacity in both directions.

  SELL OPT
      Seller returns/offers option CRR capacity and releases only the positive
      source-to-sink option exposure.

At a high level, the LP chooses awarded MW for each bid/offer to maximize
buyer value minus seller reservation cost, while keeping all awarded CRRs
simultaneously feasible on the transmission network.

Optimization problem solved by clear_auction
============================================

Sets:
  I: CRR bid/offer rows, L: transmission lines, B: buses.

Decision variables:
  x[i] >= 0: awarded MW for bid/offer row i.

Bid/path data:
  side[i] in {BUY, SELL}
  type[i] in {OBL, OPT}
  source[i]: source settlement point name for row i
  sink[i]: sink settlement point name for row i
  price[i]: buyer bid price or seller reservation price ($/MW)
  cap[i]: maximum bid/offer MW

Settlement-point injection vectors:
  e_src[i,b] = injection weight at bus b for source[i]
  e_snk[i,b] = injection weight at bus b for sink[i]

  For a resource node or single-bus load zone, the vector has 1.0 at its bus.
  For a weighted load zone, the vector uses its member-bus weights.
  For a hub, the vector averages equally across its member buses.

Source-to-sink path exposure  (result: matrix of shape n_lines × n_bids)
  Index key for this block:
    l ∈ L : transmission line index (row)
    i ∈ I : bid/offer index — one CRR instrument with a source→sink path (column)
    b ∈ B : bus index

  net_inj[i,b]   = injection weight at bus b when 1 MW flows on bid-i path
                 = e_snk[i,b] - e_src[i,b]
                   (positive at sink buses, negative at source buses)

  path_ptdf[l,i] = DC power flow on line l induced by 1 MW awarded to bid i
                 = sum_b PTDF[l,b] * net_inj[i,b]
                 = ptdf[l,:] @ (snk_inj[i] - src_inj[i])
                   (positive = flow in line's from→to direction)
                   Lines l and bids i are independent dimensions; every
                   (l, i) entry quantifies how much of line l's capacity is
                   consumed per MW of bid i.

  ptdf_obl[l,i]  = path_ptdf[l,i]
                   OBL uses signed exposure: the CRR consumes capacity in the
                   forward direction and releases it in the reverse direction.

  ptdf_opt[l,i]  = max(path_ptdf[l,i], 0)
                   OPT uses only forward (positive) exposure: the holder
                   exercises only when flow goes source→sink (DART > 0), so
                   reverse capacity is never consumed.

Objective:
  maximize
      sum_{i: BUY}  price[i] * x[i]
    - sum_{i: SELL} price[i] * x[i]

  scipy.linprog minimizes the equivalent negative objective:
     -sum_{BUY} price[i] * x[i] + sum_{SELL} price[i] * x[i].

Bounds:
  0 <= x[i] <= cap[i]

SFT constraints for each line l:
  Let F_l = capacity_factor * flow_limit[l].
  Let base[l] be signed flow from pre-existing baseload CRRs.

  Forward:
      sum_i a_fwd[l,i] * x[i] <= F_l - base[l]

  Reverse:
      sum_i a_rev[l,i] * x[i] <= F_l + base[l]

  Column coefficients:
      BUY  OBL: a_fwd = +ptdf_obl, a_rev = -ptdf_obl
      SELL OBL: a_fwd = -ptdf_obl, a_rev = +ptdf_obl
      BUY  OPT: a_fwd = +ptdf_opt, a_rev = 0
      SELL OPT: a_fwd = -ptdf_opt, a_rev = 0

Dual prices and ACPs:
  mu_fwd[l] and mu_rev[l] are the nonnegative SFT shadow prices from
  the forward and reverse line constraints.

  bus_shadow[b] = sum_l (mu_fwd[l] - mu_rev[l]) * PTDF[l,b]
  sp_shadow[s]  = inj(s) @ bus_shadow

  OBL ACP(i) = sp_shadow[sink[i]] - sp_shadow[source[i]]
  OPT ACP(i) = sum_l mu_fwd[l] * ptdf_opt[l,i]
"""


import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Any, Optional, Tuple


def _settlement_point_injection(
    sp_name: str,
    settlement_points: dict,
    bus_list: list,
    bus_idx: dict,
) -> np.ndarray:
    """Build a 1-MW injection vector for a settlement point."""
    sp = settlement_points[sp_name]
    n_bus = len(bus_list)
    inj = np.zeros(n_bus)

    if sp["type"] in ("resource_node", "load_zone"):
        inj[bus_idx[sp["bus"]]] = 1.0
    elif sp["type"] == "load_zone_weighted":
        for b, w in sp["member_buses"].items():
            inj[bus_idx[b]] = w
    elif sp["type"] == "hub":
        members = sp["member_buses"]
        for b in members:
            inj[bus_idx[b]] = 1.0 / len(members)
    return inj


def clear_auction(
    bids_df: pd.DataFrame,
    ptdf: np.ndarray,
    lines: Dict[str, dict],
    line_list: List[str],
    bus_list: List[Any],
    bus_idx: Dict[Any, int],
    settlement_points: Dict[str, dict],
    capacity_factor: float = 0.90,
    baseload_crrs: Optional[List[Tuple]] = None,
) -> dict:
    """
    Clear a CRR auction with OBL/OPT distinction and buy/sell separation.

    Parameters
    ----------
    bids_df : DataFrame
        Required columns:
            agent_id  : str
            source    : str (settlement point name)
            sink      : str (settlement point name)
            side      : str ('BUY' or 'SELL')
            crr_type  : str ('OBL' or 'OPT')
            bid_price : float ($/MWh)
            mw        : float (MW quantity)

    baseload_crrs : list of tuples
        Pre-existing CRRs consuming SFT capacity.
        Each tuple: (source, sink, mw) or (source, sink, mw, crr_type)

    Returns
    -------
    dict with:
        awarded       : DataFrame (input + mw_awarded, clearing_price)
        sp_shadow     : {sp: shadow_price}
        acp_by_path   : {(src, sink, crr_type): clearing_price}
        line_duals    : {line_id: {forward, reverse, net}}
        binding_lines : list
        total_welfare : float
        summary       : dict (MW totals by side × crr_type)
        status        : str
    """
    n_bids = len(bids_df)
    n_lines = len(line_list)

    if n_bids == 0:
        return {"status": "no_bids", "awarded": pd.DataFrame()}

    # Backward compatibility: default crr_type to OBL if missing
    bids = bids_df.reset_index(drop=True).copy()
    if "crr_type" not in bids.columns:
        bids["crr_type"] = "OBL"

    # ── 1. Build PATH_PTDF for each bid ──────────────────────────────────
    # OBL uses signed source-to-sink path PTDF. OPT uses only positive
    # source-to-sink capacity exposure.
    ptdf_obl = np.zeros((n_lines, n_bids))
    ptdf_opt = np.zeros((n_lines, n_bids))
    for k, row in bids.iterrows():
        snk_inj = _settlement_point_injection(row["sink"], settlement_points,
                                               bus_list, bus_idx)
        src_inj = _settlement_point_injection(row["source"], settlement_points,
                                               bus_list, bus_idx)
        ptdf_obl[:, k] = ptdf @ (snk_inj - src_inj)
        ptdf_opt[:, k] = np.maximum(ptdf_obl[:, k], 0.0)

    # ── 2. Objective: maximize social welfare ────────────────────────────
    c = np.zeros(n_bids)
    for k, row in bids.iterrows():
        if row["side"] == "BUY":
            c[k] = -row["bid_price"]
        else:  # SELL
            c[k] = +row["bid_price"]

    # ── 3. SFT constraints ───────────────────────────────────────────────
    A_ub_rows, b_ub_rows = [], []
    line_row_idx = {}

    # Baseload impact
    baseload_flow = np.zeros(n_lines)
    if baseload_crrs:
        for item in baseload_crrs:
            if len(item) == 3:
                src, snk, mw = item
                bl_type = "OBL"
            else:
                src, snk, mw, bl_type = item
            snk_inj = _settlement_point_injection(snk, settlement_points,
                                                   bus_list, bus_idx)
            src_inj = _settlement_point_injection(src, settlement_points,
                                                   bus_list, bus_idx)
            bl_ptdf = ptdf @ (snk_inj - src_inj)
            baseload_flow += mw * bl_ptdf

    for l_idx, line_id in enumerate(line_list):
        f_max = lines[line_id]["flow_limit"]
        eff_limit = capacity_factor * f_max

        row_fwd = np.zeros(n_bids)
        row_rev = np.zeros(n_bids)

        for k, bid in bids.iterrows():
            pp_obl = ptdf_obl[l_idx, k]
            pp_opt = ptdf_opt[l_idx, k]
            side = bid["side"]
            ctype = bid["crr_type"]

            if side == "BUY":
                if ctype == "OBL":
                    # OBL: capacity used in both directions
                    row_fwd[k] = pp_obl
                    row_rev[k] = -pp_obl
                else:  # OPT
                    # OPT: capacity used only in positive flow direction
                    # Option holder exercises only when DART > 0 → flow source→sink
                    row_fwd[k] = pp_opt
                    row_rev[k] = 0.0
            else:  # SELL
                if ctype == "OBL":
                    # Selling OBL releases capacity in both directions
                    row_fwd[k] = -pp_obl
                    row_rev[k] = pp_obl
                else:  # OPT
                    # Selling OPT releases capacity in positive direction only
                    row_fwd[k] = -pp_opt
                    row_rev[k] = 0.0

        A_ub_rows.append(row_fwd)
        b_ub_rows.append(eff_limit - baseload_flow[l_idx])
        A_ub_rows.append(row_rev)
        b_ub_rows.append(eff_limit + baseload_flow[l_idx])

        line_row_idx[line_id] = (len(A_ub_rows) - 2, len(A_ub_rows) - 1)

    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)

    # ── 4. Bounds ────────────────────────────────────────────────────────
    bounds = [(0, float(row["mw"])) for _, row in bids.iterrows()]

    # ── 5. Solve ─────────────────────────────────────────────────────────
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method="highs", options={"disp": False})

    if res.status != 0:
        return {"status": res.message}

    # ── 6. Extract results ───────────────────────────────────────────────
    awarded_df = bids.copy()
    awarded_df["mw_awarded"] = res.x

    # Line duals (forward and reverse separated)
    ineq_duals = res.ineqlin.marginals
    line_duals = {}
    for line_id, (idx_fwd, idx_rev) in line_row_idx.items():
        mu_fwd = -ineq_duals[idx_fwd]
        mu_rev = -ineq_duals[idx_rev]
        line_duals[line_id] = {
            "forward": mu_fwd,
            "reverse": mu_rev,
            "net": mu_fwd - mu_rev,
        }

    binding_lines = [l for l, d in line_duals.items()
                     if abs(d["forward"]) > 0.01 or abs(d["reverse"]) > 0.01]

    # Bus shadows
    bus_shadow = {}
    for b_i, b in enumerate(bus_list):
        s = 0.0
        for l_idx, line_id in enumerate(line_list):
            s += line_duals[line_id]["net"] * ptdf[l_idx, b_i]
        bus_shadow[b] = s

    # SP shadows
    sp_shadow = {}
    for sp_name in settlement_points:
        inj = _settlement_point_injection(sp_name, settlement_points,
                                           bus_list, bus_idx)
        sp_shadow[sp_name] = float(np.dot(inj, [bus_shadow[b] for b in bus_list]))

    # ── 7. Clearing prices by path and type ──────────────────────────────
    acp_by_path = {}
    for _, row in awarded_df.iterrows():
        key = (row["source"], row["sink"], row["crr_type"])
        if key not in acp_by_path:
            obl_acp = sp_shadow.get(row["sink"], 0) - sp_shadow.get(row["source"], 0)
            if row["crr_type"] == "OBL":
                # OBL: can be negative (holder pays when DART < 0)
                acp_by_path[key] = obl_acp
            else:  # OPT
                # OPT: price positive SFT exposure directly from constraint duals.
                # This is not the same as clipping the OBL ACP at zero.
                opt_ptdf = ptdf_opt[:, int(row.name)]
                opt_acp = 0.0
                for l_idx, line_id in enumerate(line_list):
                    opt_acp += line_duals[line_id]["forward"] * opt_ptdf[l_idx]
                acp_by_path[key] = opt_acp

    awarded_df["clearing_price"] = awarded_df.apply(
        lambda r: acp_by_path.get((r["source"], r["sink"], r["crr_type"]), 0),
        axis=1,
    )

    # ── 8. Summary ───────────────────────────────────────────────────────
    summary = {}
    for side in ["BUY", "SELL"]:
        for ctype in ["OBL", "OPT"]:
            mask = (awarded_df["side"] == side) & (awarded_df["crr_type"] == ctype)
            subset = awarded_df[mask]
            key = f"{side.lower()}_{ctype.lower()}"
            summary[f"{key}_mw"] = float(subset["mw_awarded"].sum())
            summary[f"{key}_n"] = int((subset["mw_awarded"] > 0.01).sum())

    return {
        "status":        "optimal",
        "awarded":       awarded_df,
        "sp_shadow":     sp_shadow,
        "bus_shadow":    bus_shadow,
        "acp_by_path":   acp_by_path,
        "line_duals":    line_duals,
        "binding_lines": binding_lines,
        "total_welfare": float(-res.fun),
        "summary":       summary,
    }


def clear_auction_vectorize(
    bids_df: pd.DataFrame,
    ptdf: np.ndarray,
    lines: Dict[str, dict],
    line_list: List[str],
    bus_list: List[Any],
    bus_idx: Dict[Any, int],
    settlement_points: Dict[str, dict],
    capacity_factor: float = 0.90,
    baseload_crrs: Optional[List[Tuple]] = None,
) -> dict:
    """
    Vectorized CRR auction clearing — identical results to clear_auction, lower runtime.

    Key differences from clear_auction:
      - path_ptdf built via one batched matmul  ptdf @ net_inj.T  (BLAS-3)
        instead of n_bids separate matrix-vector products (n_bids × BLAS-1).
      - SFT constraint matrix assembled with NumPy broadcasting, eliminating
        the O(n_lines × n_bids) nested Python loop.
      - Bus shadow prices computed as  mu_net @ ptdf  (one matmul).
      - SP shadow prices computed as  sp_inj_mat @ bus_shadow  (one matmul).
      - OPT ACPs for all bids computed as  mu_fwd @ ptdf_opt  (one matmul).
    """
    n_bids  = len(bids_df)
    n_lines = len(line_list)
    n_bus   = len(bus_list)

    if n_bids == 0:
        return {"status": "no_bids", "awarded": pd.DataFrame()}

    bids = bids_df.reset_index(drop=True).copy()
    if "crr_type" not in bids.columns:
        bids["crr_type"] = "OBL"

    source_names = bids["source"].tolist()
    sink_names   = bids["sink"].tolist()

    # ── 1. Build injection matrices, then path_ptdf in one batched matmul ────
    # snk_mat / src_mat : (n_bids, n_bus) — one row per bid
    snk_mat = np.zeros((n_bids, n_bus))
    src_mat = np.zeros((n_bids, n_bus))
    for k in range(n_bids):
        snk_mat[k] = _settlement_point_injection(
            sink_names[k],   settlement_points, bus_list, bus_idx)
        src_mat[k] = _settlement_point_injection(
            source_names[k], settlement_points, bus_list, bus_idx)

    net_inj  = snk_mat - src_mat          # (n_bids, n_bus)
    ptdf_obl = ptdf @ net_inj.T           # (n_lines, n_bids) — single BLAS-3 call
    ptdf_opt = np.maximum(ptdf_obl, 0.0)  # (n_lines, n_bids)

    # ── 2. Objective ──────────────────────────────────────────────────────────
    is_buy = bids["side"].values == "BUY"          # (n_bids,) bool
    prices = bids["bid_price"].values.astype(float)
    c = np.where(is_buy, -prices, prices)          # linprog minimises

    # ── 3. Baseload flows ─────────────────────────────────────────────────────
    baseload_flow = np.zeros(n_lines)
    if baseload_crrs:
        for item in baseload_crrs:
            src, snk, mw = item[:3]
            snk_inj = _settlement_point_injection(snk, settlement_points, bus_list, bus_idx)
            src_inj = _settlement_point_injection(src, settlement_points, bus_list, bus_idx)
            baseload_flow += mw * (ptdf @ (snk_inj - src_inj))

    # ── 4. SFT constraint matrix — fully vectorized ───────────────────────────
    # For each bid i and line l, the forward / reverse capacity consumed:
    #   BUY  OBL: A_fwd = +ptdf_obl,  A_rev = -ptdf_obl
    #   SELL OBL: A_fwd = -ptdf_obl,  A_rev = +ptdf_obl
    #   BUY  OPT: A_fwd = +ptdf_opt,  A_rev = 0
    #   SELL OPT: A_fwd = -ptdf_opt,  A_rev = 0
    is_obl = bids["crr_type"].values == "OBL"      # (n_bids,) bool
    sign   = np.where(is_buy, 1.0, -1.0)           # (n_bids,)

    # Select ptdf_obl or ptdf_opt per column, then scale by sign
    A_fwd = np.where(is_obl, ptdf_obl, ptdf_opt) * sign   # (n_lines, n_bids)
    A_rev = np.where(is_obl, -ptdf_obl * sign, 0.0)        # (n_lines, n_bids)

    # Interleave rows: [line0_fwd, line0_rev, line1_fwd, line1_rev, ...]
    A_ub = np.empty((2 * n_lines, n_bids))
    A_ub[0::2] = A_fwd
    A_ub[1::2] = A_rev

    eff_limits = capacity_factor * np.array(
        [lines[l]["flow_limit"] for l in line_list])  # (n_lines,)
    b_ub = np.empty(2 * n_lines)
    b_ub[0::2] = eff_limits - baseload_flow   # forward: F_l - base[l]
    b_ub[1::2] = eff_limits + baseload_flow   # reverse: F_l + base[l]

    # Row indices match clear_auction: fwd = 2*l, rev = 2*l+1
    line_row_idx = {lid: (2 * li, 2 * li + 1) for li, lid in enumerate(line_list)}

    # ── 5. Bounds ─────────────────────────────────────────────────────────────
    bounds = list(zip(np.zeros(n_bids), bids["mw"].values.astype(float)))

    # ── 6. Solve ──────────────────────────────────────────────────────────────
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                  method="highs", options={"disp": False})

    if res.status != 0:
        return {"status": res.message}

    # ── 7. Duals — vectorized extraction ─────────────────────────────────────
    awarded_df = bids.copy()
    awarded_df["mw_awarded"] = res.x

    ineq_duals  = res.ineqlin.marginals
    mu_fwd_arr  = -ineq_duals[0::2]           # (n_lines,)  forward constraint duals
    mu_rev_arr  = -ineq_duals[1::2]           # (n_lines,)  reverse constraint duals
    mu_net_arr  = mu_fwd_arr - mu_rev_arr     # (n_lines,)

    line_duals = {
        lid: {"forward": float(mu_fwd_arr[li]),
              "reverse": float(mu_rev_arr[li]),
              "net":     float(mu_net_arr[li])}
        for li, lid in enumerate(line_list)
    }
    binding_lines = [l for l, d in line_duals.items()
                     if abs(d["forward"]) > 0.01 or abs(d["reverse"]) > 0.01]

    # ── 8. Bus and SP shadows — one matmul each ───────────────────────────────
    # bus_shadow[b] = Σ_l mu_net[l] * PTDF[l,b]
    bus_shadow_arr = mu_net_arr @ ptdf                         # (n_bus,)
    bus_shadow = {b: float(bus_shadow_arr[bi])
                  for bi, b in enumerate(bus_list)}

    # SP shadows: batch all injection vectors into one matrix multiply
    sp_names   = list(settlement_points.keys())
    sp_inj_mat = np.zeros((len(sp_names), n_bus))
    for si, sp_name in enumerate(sp_names):
        sp_inj_mat[si] = _settlement_point_injection(
            sp_name, settlement_points, bus_list, bus_idx)
    sp_shadow_arr = sp_inj_mat @ bus_shadow_arr                # (n_sp,)
    sp_shadow = {sp_name: float(sp_shadow_arr[si])
                 for si, sp_name in enumerate(sp_names)}

    # ── 9. Clearing prices by path and type ───────────────────────────────────
    # OPT ACP for all bids in one matmul: mu_fwd @ ptdf_opt → (n_bids,)
    opt_acp_per_bid = mu_fwd_arr @ ptdf_opt                    # (n_bids,)

    acp_by_path = {}
    for k in range(n_bids):
        key = (source_names[k], sink_names[k], bids.at[k, "crr_type"])
        if key not in acp_by_path:
            if key[2] == "OBL":
                acp_by_path[key] = (sp_shadow.get(key[1], 0.0)
                                    - sp_shadow.get(key[0], 0.0))
            else:
                acp_by_path[key] = float(opt_acp_per_bid[k])

    awarded_df["clearing_price"] = awarded_df.apply(
        lambda r: acp_by_path.get((r["source"], r["sink"], r["crr_type"]), 0.0),
        axis=1,
    )

    # ── 10. Summary ───────────────────────────────────────────────────────────
    summary = {}
    for side in ["BUY", "SELL"]:
        for ctype in ["OBL", "OPT"]:
            mask   = (awarded_df["side"] == side) & (awarded_df["crr_type"] == ctype)
            subset = awarded_df[mask]
            key    = f"{side.lower()}_{ctype.lower()}"
            summary[f"{key}_mw"] = float(subset["mw_awarded"].sum())
            summary[f"{key}_n"]  = int((subset["mw_awarded"] > 0.01).sum())

    return {
        "status":        "optimal",
        "awarded":       awarded_df,
        "sp_shadow":     sp_shadow,
        "bus_shadow":    bus_shadow,
        "acp_by_path":   acp_by_path,
        "line_duals":    line_duals,
        "binding_lines": binding_lines,
        "total_welfare": float(-res.fun),
        "summary":       summary,
    }
