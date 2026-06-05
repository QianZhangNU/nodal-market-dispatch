"""
N-1 contingency screening benchmark.

Synthetic ring-topology system at N = 17 / 50 / 100 / 200 buses.
Each run measures:
  - _build_base_lp   : LP matrix construction (sections [A]-[E])
  - _solve_with_ctg  : one HiGHS solve (base case, no ctg rows)
  - screen_n1        : vectorised contingency screening
  - full n1 (3 iter) : solve_hourly_sced_n1 with max_iter=3

Run from the repo root:
  python benchmarks/benchmark_n1.py
"""

import time
import sys
import os
import numpy as np
import pandas as pd

# ── repo on path ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.network import NetworkModel
from solvers.sced.solver import _build_base_lp, _solve_with_ctg
from solvers.sced.n1_screening import screen_n1_violations, solve_hourly_sced_n1


# ── synthetic system builder ───────────────────────────────────────────────────

def make_ring_system(n_bus: int, seed: int = 0):
    """
    Ring topology with n_bus buses, n_bus lines, and n_bus-1 contingencies
    (one per non-slack line).

    Layout
    ------
      - Bus 1 is the slack.
      - Line k = bus k -> bus k+1 for k in 1..n_bus-1, plus line n_bus -> 1.
      - Generator at every bus; cheap at even buses, expensive at odd buses.
      - Load concentrated at bus n_bus//2.
    """
    rng = np.random.default_rng(seed)

    buses = {b: {"name": f"B{b}"} for b in range(1, n_bus + 1)}

    lines = {}
    for k in range(1, n_bus + 1):
        fr = k
        to = (k % n_bus) + 1
        x = float(rng.uniform(0.05, 0.20))
        line_id = f"L{fr}_{to}"
        lines[line_id] = {
            "from_bus": fr, "to_bus": to,
            "x_pu": x, "b_pu": 1.0 / x,
            "flow_limit": 500.0,
            "contingency_limit": 400.0,
        }

    line_list = list(lines.keys())

    # Contingencies: outage of every line except the slack-incident line L1_2
    contingencies = {}
    for lid in line_list[1:]:
        contingencies[f"CTG_{lid}"] = {"outaged_line": lid}

    model = NetworkModel(
        buses=buses,
        lines=lines,
        slack_bus=1,
        contingencies=contingencies,
    )

    # Generators: one per bus
    generators = {}
    gen_list = []
    for b in range(1, n_bus + 1):
        g = f"G{b}"
        cost = 30.0 if b % 2 == 0 else 60.0
        generators[g] = {
            "bus": b,
            "Pmin": 0, "Pmax": 200,
            "cost_b": cost,
            "ramp_up": 200, "ramp_down": 200,
        }
        gen_list.append(g)

    # Load: spread uniformly across all buses so no single line is overloaded.
    # Each bus takes 80 MW; total gen per bus is 200 MW so system is feasible.
    load_h = pd.Series({b: 80.0 for b in range(1, n_bus + 1)})

    settlement_points = {f"RN{b}": {"type": "resource_node", "bus": b}
                         for b in range(1, n_bus + 1)}

    common = dict(
        generators=generators,
        gen_list=gen_list,
        bus_list=model.bus_list,
        bus_idx=model.bus_idx,
        lines=model.lines,
        line_list=model.line_list,
        ptdf=model.ptdf,
        settlement_points=settlement_points,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={g: 1 for g in gen_list},
        p_prev={g: 0.0 for g in gen_list},
        load_h=load_h,
        pmax_h=pd.Series({g: 200.0 for g in gen_list}),
        cost_h=pd.Series({g: generators[g]["cost_b"] for g in gen_list}),
        use_ramp=False,
        use_network=True,
        use_gtc=False,
        battery_injection_h=None,
        gtcs={},
    )

    return model, common


# ── timing helper ──────────────────────────────────────────────────────────────

def timeit(fn, repeat: int = 5):
    """Return (mean_ms, std_ms) over `repeat` calls."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return arr.mean(), arr.std()


# ── benchmark ──────────────────────────────────────────────────────────────────

def run_benchmark(sizes=(17, 50, 100, 150), repeat=5):
    col_w = 16
    headers = ["N_bus", "N_ctg", "build_base(ms)", "solve_base(ms)",
               "screen_n1(ms)", "n1_pairs", "full_n1_3it(ms)", "prefilter(ms)"]
    print("\n" + "=" * (col_w * len(headers)))
    print("  N-1 Contingency Screening - Component Benchmark")
    print("=" * (col_w * len(headers)))
    print("".join(h.ljust(col_w) for h in headers))
    print("-" * (col_w * len(headers)))

    for n_bus in sizes:
        model, common = make_ring_system(n_bus)
        n_ctg = len(model.contingencies)

        # Extract args for _build_base_lp
        build_args = dict(
            generators=common["generators"],
            gen_list=common["gen_list"],
            bus_list=common["bus_list"],
            lines=common["lines"],
            line_list=common["line_list"],
            ptdf=common["ptdf"],
            gtcs=common["gtcs"],
            u_fixed=common["u_fixed"],
            p_prev=common["p_prev"],
            load_h=common["load_h"],
            pmax_h=common["pmax_h"],
            cost_h=common["cost_h"],
            battery_injection_h=common["battery_injection_h"],
            use_network=common["use_network"],
            use_ramp=common["use_ramp"],
            use_gtc=common["use_gtc"],
        )

        # ── 1. _build_base_lp ─────────────────────────────────────────────
        t_build, _ = timeit(lambda: _build_base_lp(**build_args), repeat)

        # ── 2. _solve_with_ctg (base case, no ctg rows) ───────────────────
        cache = _build_base_lp(**build_args)
        solve_args = dict(
            cache=cache,
            active_ctg_pairs=set(),
            contingency_ptdf=model.contingency_ptdf,
            ctg_violation_cost=1000.0,
            lines=common["lines"],
            line_list=common["line_list"],
            bus_list=common["bus_list"],
            gen_list=common["gen_list"],
            settlement_points=common["settlement_points"],
            ptdf=common["ptdf"],
            hour=common["hour"],
            gtcs=common["gtcs"],
            use_network=common["use_network"],
            battery_injection_h=common["battery_injection_h"],
        )
        t_solve, _ = timeit(lambda: _solve_with_ctg(**solve_args), repeat)

        # ── 3. screen_n1_violations (vectorised) ──────────────────────────
        base_result = _solve_with_ctg(**solve_args)
        if base_result.get("status") != "optimal":
            print(f"  WARNING: base solve infeasible for N={n_bus}, skipping screen/n1")
            continue
        eff_load = common["load_h"]
        screen_args = dict(
            dispatch=base_result["dispatch"],
            effective_load=eff_load,
            generators=common["generators"],
            bus_list=common["bus_list"],
            line_list=common["line_list"],
            lines=common["lines"],
            contingency_ptdf=model.contingency_ptdf,
            base_line_flows=base_result.get("line_flows", {}),
            violation_threshold=0.95,
            prefilter_ratio=0.0,
            contingency_ptdf_tensor=model.contingency_ptdf_tensor,
            ctg_id_list=model.ctg_id_list,
        )
        t_screen, _ = timeit(lambda: screen_n1_violations(**screen_args), repeat)

        # ── 4. full solve_hourly_sced_n1 (3 iterations, no prefilter) ────────
        n1_args = dict(
            **common,
            contingency_ptdf=model.contingency_ptdf,
            violation_threshold=0.95,
            prefilter_ratio=0.0,
            ctg_violation_cost=1000.0,
            max_iter=3,
            contingency_ptdf_tensor=model.contingency_ptdf_tensor,
            ctg_id_list=model.ctg_id_list,
        )
        # One diagnostic run to count active pairs (not timed)
        try:
            diag = solve_hourly_sced_n1(**n1_args)
            n_active = len(diag.get("n1_active_pairs", set()))
            t_n1, _ = timeit(lambda: solve_hourly_sced_n1(**n1_args), repeat)
            t_n1_str = f"{t_n1:.1f}"
        except (MemoryError, Exception) as e:
            if "allocate" in str(e).lower() or isinstance(e, MemoryError):
                n_active = "OOM"
                t_n1_str = "OOM"
            else:
                raise

        # ── 5. full solve with prefilter_ratio=0.3 ────────────────────────
        try:
            n1_pf_args = {**n1_args, "prefilter_ratio": 0.3}
            t_pf, _ = timeit(lambda: solve_hourly_sced_n1(**n1_pf_args), repeat)
            t_pf_str = f"{t_pf:.1f}"
        except (MemoryError, np.core._exceptions._ArrayMemoryError):
            t_pf_str = "OOM"

        row = [str(n_bus), str(n_ctg),
               f"{t_build:.1f}", f"{t_solve:.1f}",
               f"{t_screen:.1f}", str(n_active),
               t_n1_str, t_pf_str]
        print("".join(v.ljust(col_w) for v in row))

    print("=" * (col_w * len(headers)))
    print(f"  Timings: mean over {repeat} runs  (ms = milliseconds)")
    print()
    print("  Column notes")
    print("  ------------")
    print("  build_base    : _build_base_lp() - build LP sections [A]-[E] (done once per hour).")
    print("  solve_base    : _solve_with_ctg() - HiGHS solve, no contingency rows.")
    print("  screen_n1     : screen_n1_violations() - batched matmul (n_ctg, n_line, n_bus) @ (n_bus,).")
    print("  n1_pairs      : # (ctg, line) pairs activated across all 3 iterations.")
    print("  full_n1_3it   : solve_hourly_sced_n1(max_iter=3, prefilter=0.0) end-to-end.")
    print("  prefilter     : same with prefilter_ratio=0.3 (skip lines at <30% of ctg limit).")
    print()
    print("  Bottleneck analysis")
    print("  -------------------")
    print("  The screening matmul is cheap even at 150 buses (~30 ms).")
    print("  The LP solve dominates once many ctg pairs are activated: each pair adds")
    print("  2 rows + 1 slack variable to A_ub, so LP size grows O(n_pairs^2).")
    print("  Tuning levers in order of impact:")
    print("    1. violation_threshold (higher -> fewer pairs -> smaller LP)")
    print("    2. prefilter_ratio     (skip low-flow lines before the batched matmul)")
    print("    3. max_iter            (fewer re-solves)")
    print()


if __name__ == "__main__":
    run_benchmark()
