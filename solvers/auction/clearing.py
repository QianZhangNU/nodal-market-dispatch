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

Source-to-sink path exposure:
  net_inj[i,b] = e_snk[i,b] - e_src[i,b]
  path_ptdf[l,i] = sum_b PTDF[l,b] * net_inj[i,b]
  ptdf_obl[l,i] = path_ptdf[l,i]
  ptdf_opt[l,i] = max(path_ptdf[l,i], 0)

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
