"""
CRR Auction Solver Benchmark
=============================
Compares clear_auction (loop-based) vs clear_auction_vectorize (NumPy-batched)
across synthetic problem sizes that span this 17-bus example up to near-ERCOT scale.

Usage
-----
    python examples/ercot_17bus/benchmark_auction.py

Two tables are printed:
  1. Scaling comparison — total wall-clock time across five problem sizes.
  2. Step-by-step breakdown — per-phase timing at the "Large ISO" configuration
     (200 buses / 300 lines / 1000 bids) to isolate where time is saved.

Background on the vectorization
---------------------------------
Original clear_auction has three Python-level bottlenecks:

  Step 1  path_ptdf build  — n_bids serial  ptdf @ vec  calls (BLAS-1 loop).
                             Vectorized: one  ptdf @ net_inj.T  (BLAS-3).

  Step 2  SFT assembly     — O(n_lines × n_bids) nested Python loop over
                             bids.iterrows().  Vectorized: NumPy broadcast +
                             interleaved row assignment, no Python loop.

  Step 3  Bus shadow       — O(n_lines × n_buses) Python loop.
                             Vectorized: mu_net @ ptdf  (one matmul).

After vectorization the LP solve (HiGHS) becomes the sole bottleneck and
accounts for ~98 % of remaining runtime.  It cannot be reduced further without
switching to a sparse commercial solver (Gurobi / CPLEX) or decomposing the
problem by time period.
"""

import os
import sys
# Ensure repo root is on the path whether run as a script or as a module
_repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
if _repo_root not in sys.path:
    sys.path.insert(0, os.path.normpath(_repo_root))

import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from solvers.auction.clearing import (
    clear_auction,
    clear_auction_vectorize,
    _settlement_point_injection,
)


# ── Synthetic problem generator ───────────────────────────────────────────────

def make_problem(n_bus: int, n_line: int, n_bid: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    buses     = ["B" + str(i) for i in range(n_bus)]
    bus_idx   = {b: i for i, b in enumerate(buses)}
    line_list = ["L" + str(i) for i in range(n_line)]
    lines     = {l: {"flow_limit": 200.0} for l in line_list}
    ptdf      = rng.standard_normal((n_line, n_bus))
    sps       = {b: {"type": "resource_node", "bus": b} for b in buses}
    bids = pd.DataFrame({
        "agent_id":  ["A" + str(i) for i in range(n_bid)],
        "source":    list(rng.choice(buses, n_bid)),
        "sink":      list(rng.choice(buses, n_bid)),
        "side":      list(rng.choice(["BUY", "SELL"], n_bid, p=[0.7, 0.3])),
        "crr_type":  list(rng.choice(["OBL", "OPT"],  n_bid, p=[0.6, 0.4])),
        "bid_price": list(rng.uniform(1, 50, n_bid)),
        "mw":        list(rng.uniform(10, 100, n_bid)),
    }).reset_index(drop=True)
    return bids, ptdf, lines, line_list, buses, bus_idx, sps


def best_of(fn, args, reps=3):
    """Return (best_time_sec, result) over `reps` runs."""
    best_t, best_r = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        r  = fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best_t:
            best_t, best_r = elapsed, r
    return best_t, best_r


# ── Table 1: scaling comparison ───────────────────────────────────────────────

CONFIGS = [
    ( 17,   17,   50, "17-bus (this example)"),
    ( 50,   75,  200, "Small ISO"),
    (100,  150,  500, "Medium ISO"),
    (200,  300, 1000, "Large ISO"),
    (400,  600, 2000, "Near-ERCOT (base, no N-1)"),
]


def run_scaling_comparison(reps=3):
    print()
    print("=" * 72)
    print("  Table 1 — Scaling comparison (best of {:d} runs)".format(reps))
    print("=" * 72)
    print("{:<28} {:>5} {:>5} {:>5} {:>10} {:>10} {:>9} {:>6}".format(
        "Config", "Buses", "Lines", "Bids", "Orig (s)", "Vect (s)", "Speedup", "Match"))
    print("-" * 72)

    for n_bus, n_line, n_bid, label in CONFIGS:
        args = make_problem(n_bus, n_line, n_bid)
        # warm-up (JIT, BLAS cache)
        clear_auction(*args)
        clear_auction_vectorize(*args)

        t_orig, r1 = best_of(clear_auction,           args, reps)
        t_vect, r2 = best_of(clear_auction_vectorize, args, reps)

        welfare_match = abs(r1["total_welfare"] - r2["total_welfare"]) < 1e-4
        print("{:<28} {:>5d} {:>5d} {:>5d} {:>10.3f} {:>10.3f} {:>8.1f}x {:>6}".format(
            label, n_bus, n_line, n_bid,
            t_orig, t_vect, t_orig / t_vect, str(welfare_match)))

    print()


# ── Table 2: per-step breakdown ───────────────────────────────────────────────

BREAKDOWN_CONFIG = (200, 300, 1000, "Large ISO (200 buses / 300 lines / 1000 bids)")


def run_step_breakdown():
    n_bus, n_line, n_bid, label = BREAKDOWN_CONFIG
    bids, ptdf, lines_d, line_list, buses, bus_idx, sps = make_problem(n_bus, n_line, n_bid)

    # ── Original: instrument each step individually ───────────────────────────

    # Step 1 orig: serial matmuls
    t0 = time.perf_counter()
    ptdf_obl_o = np.zeros((n_line, n_bid))
    ptdf_opt_o = np.zeros((n_line, n_bid))
    for k, row in bids.iterrows():
        snk = _settlement_point_injection(row["sink"],   sps, buses, bus_idx)
        src = _settlement_point_injection(row["source"], sps, buses, bus_idx)
        ptdf_obl_o[:, k] = ptdf @ (snk - src)
        ptdf_opt_o[:, k] = np.maximum(ptdf_obl_o[:, k], 0.0)
    t_path_orig = time.perf_counter() - t0

    # Step 2 orig: nested Python loop for SFT
    t0 = time.perf_counter()
    a_rows, b_rows = [], []
    for l_idx in range(n_line):
        row_fwd = np.zeros(n_bid)
        row_rev = np.zeros(n_bid)
        for k, bid in bids.iterrows():
            pp_obl = ptdf_obl_o[l_idx, k]
            pp_opt = ptdf_opt_o[l_idx, k]
            if bid["side"] == "BUY":
                if bid["crr_type"] == "OBL":
                    row_fwd[k] = pp_obl;  row_rev[k] = -pp_obl
                else:
                    row_fwd[k] = pp_opt;  row_rev[k] = 0.0
            else:
                if bid["crr_type"] == "OBL":
                    row_fwd[k] = -pp_obl; row_rev[k] = pp_obl
                else:
                    row_fwd[k] = -pp_opt; row_rev[k] = 0.0
        a_rows.append(row_fwd); b_rows.append(200.0)
        a_rows.append(row_rev); b_rows.append(200.0)
    A_ub_o = np.array(a_rows)
    b_ub_o = np.array(b_rows)
    t_sft_orig = time.perf_counter() - t0

    # Step 3: LP solve (same problem; timed once, used as reference for both)
    c      = np.where(bids["side"].values == "BUY",
                      -bids["bid_price"].values, bids["bid_price"].values)
    bounds = list(zip(np.zeros(n_bid), bids["mw"].values))

    t0 = time.perf_counter()
    res = linprog(c=c, A_ub=A_ub_o, b_ub=b_ub_o, bounds=bounds,
                  method="highs", options={"disp": False})
    t_lp = time.perf_counter() - t0

    ineq   = res.ineqlin.marginals
    mu_net = -(ineq[0::2] - ineq[1::2])   # (n_line,)

    # Step 4 orig: nested bus shadow loop
    t0 = time.perf_counter()
    for b_i in range(n_bus):
        _v = sum(mu_net[l] * ptdf[l, b_i] for l in range(n_line))
    t_shadow_orig = time.perf_counter() - t0

    # ── Vectorized: instrument each step individually ─────────────────────────

    # Step 1 vect: batch injection matrix + single BLAS-3 matmul
    t0 = time.perf_counter()
    snk_mat = np.zeros((n_bid, n_bus))
    src_mat = np.zeros((n_bid, n_bus))
    for k in range(n_bid):
        snk_mat[k] = _settlement_point_injection(bids.at[k, "sink"],   sps, buses, bus_idx)
        src_mat[k] = _settlement_point_injection(bids.at[k, "source"], sps, buses, bus_idx)
    ptdf_obl_v = ptdf @ (snk_mat - src_mat).T
    ptdf_opt_v = np.maximum(ptdf_obl_v, 0.0)
    t_path_vect = time.perf_counter() - t0

    # Step 2 vect: broadcast + interleaved slice assignment
    t0 = time.perf_counter()
    is_buy = bids["side"].values == "BUY"
    is_obl = bids["crr_type"].values == "OBL"
    sign   = np.where(is_buy, 1.0, -1.0)
    A_fwd  = np.where(is_obl, ptdf_obl_v, ptdf_opt_v) * sign
    A_rev  = np.where(is_obl, -ptdf_obl_v * sign, 0.0)
    A_ub_v = np.empty((2 * n_line, n_bid))
    A_ub_v[0::2] = A_fwd
    A_ub_v[1::2] = A_rev
    b_ub_v = np.full(2 * n_line, 200.0)
    t_sft_vect = time.perf_counter() - t0

    # Step 3 vect: LP solve (same A_ub structure, re-timed for fairness)
    t0 = time.perf_counter()
    linprog(c=c, A_ub=A_ub_v, b_ub=b_ub_v, bounds=bounds,
            method="highs", options={"disp": False})
    t_lp2 = time.perf_counter() - t0

    # Step 4 vect: single matmul
    t0 = time.perf_counter()
    _bus_shadow_v = mu_net @ ptdf
    t_shadow_vect = time.perf_counter() - t0

    # ── Print ─────────────────────────────────────────────────────────────────
    steps = [
        ("1. path_ptdf build",   t_path_orig,   t_path_vect),
        ("2. SFT assembly",      t_sft_orig,    t_sft_vect),
        ("3. LP solve (HiGHS)",  t_lp,          t_lp2),
        ("4. Bus shadow prices", t_shadow_orig, t_shadow_vect),
    ]
    total_o = sum(r[1] for r in steps)
    total_v = sum(r[2] for r in steps)

    print("=" * 72)
    print("  Table 2 — Per-step breakdown: {}".format(label))
    print("=" * 72)
    print("{:<28} {:>10} {:>10} {:>9} {:>12}".format(
        "Step", "Orig (s)", "Vect (s)", "Speedup", "% orig total"))
    print("-" * 72)
    for name, to, tv in steps:
        print("{:<28} {:>10.3f} {:>10.3f} {:>8.1f}x {:>11.1f}%".format(
            name, to, tv, to / tv, 100.0 * to / total_o))
    print("-" * 72)
    print("{:<28} {:>10.3f} {:>10.3f} {:>8.1f}x".format(
        "TOTAL", total_o, total_v, total_o / total_v))
    print()
    print("  LP solve share of vectorized total: {:.1f}%".format(
        100.0 * t_lp2 / total_v))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_scaling_comparison(reps=3)
    run_step_breakdown()
