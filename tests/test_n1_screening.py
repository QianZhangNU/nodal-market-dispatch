"""Tests for iterative N-1 contingency screening.

Three-bus topology used throughout:

    Bus1 (slack) ---L12--- Bus2 ---L23--- Bus3
         \                               /
          --------- L13 ----------------

  G1 @ Bus1: cheap ($30), large (500 MW)
  G2 @ Bus3: expensive ($80), small (50 MW)
  Load: 200 MW all at Bus2

  Base-case flows with no congestion (G1=200 MW, G2=0):
    Two parallel paths carry the 200 MW from Bus1 to Bus2:
      Direct:   Bus1 -L12-> Bus2          133 MW  (2/3 of total)
      Indirect: Bus1 -L13-> Bus3 -L23-> Bus2  67 MW  (1/3 of total)
    Bus3 is a through-node (zero net injection): 67 MW enters via L13
    and exits via L23 (f_L23 = -67 MW in the from_bus=2 convention).

  contingency_limit set so the BASE case is fine but the N-1 case overloads.
  Specifically, lose L13 → all 200 MW flows via L12 → L12 post-ctg flow
  exceeds contingency_limit.  Solver must redispatch G2 to relieve L12.
"""

import numpy as np
import pandas as pd
import pytest

from solvers.network import NetworkModel
from solvers.sced import solve_hourly_sced, solve_hourly_sced_n1
from solvers.sced.n1_screening import screen_n1_violations


# ── shared fixture ────────────────────────────────────────────────────────────

def _make_3bus():
    """Return (model, generators, settlement_points, common_sced_kwargs)."""
    # Equal susceptances → flow splits 2/3 via L12 and 1/3 via L13 for a
    # source at Bus1 injecting to Bus2.
    model = NetworkModel(
        buses={1: {"name": "Gen"}, 2: {"name": "Load"}, 3: {"name": "Mid"}},
        lines={
            "L12": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                    "flow_limit": 200, "contingency_limit": 150},
            "L13": {"from_bus": 1, "to_bus": 3, "x_pu": 0.1, "b_pu": 10,
                    "flow_limit": 200, "contingency_limit": 150},
            # L23 contingency_limit set above the physical maximum (200 MW) so
            # that losing L12 — which forces all load through L13→L23 regardless
            # of dispatch — does not create an inherently infeasible constraint.
            "L23": {"from_bus": 2, "to_bus": 3, "x_pu": 0.1, "b_pu": 10,
                    "flow_limit": 200, "contingency_limit": 250},
        },
        slack_bus=1,
        contingencies={
            "CTG_L13": {"outaged_line": "L13"},
            "CTG_L12": {"outaged_line": "L12"},
        },
    )
    generators = {
        "G1": dict(bus=1, Pmin=0, Pmax=500, cost_b=30, ramp_up=500, ramp_down=500),
        "G2": dict(bus=3, Pmin=0, Pmax=100, cost_b=80, ramp_up=100, ramp_down=100),
    }
    settlement_points = {
        "RN1": dict(type="resource_node", bus=1),
        "RN2": dict(type="resource_node", bus=2),
        "RN3": dict(type="resource_node", bus=3),
    }
    common = dict(
        generators=generators,
        gen_list=["G1", "G2"],
        bus_list=[1, 2, 3],
        bus_idx={1: 0, 2: 1, 3: 2},
        lines=model.lines,
        line_list=model.line_list,
        ptdf=model.ptdf,
        settlement_points=settlement_points,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={"G1": 1, "G2": 1},
        p_prev={"G1": 200, "G2": 0},
        load_h=pd.Series({1: 0, 2: 200, 3: 0}),
        pmax_h=pd.Series({"G1": 500, "G2": 100}),
        cost_h=pd.Series({"G1": 30, "G2": 80}),
        use_ramp=False,
    )
    return model, generators, settlement_points, common


# ── screen_n1_violations ──────────────────────────────────────────────────────

def test_screen_n1_uncongested_no_violations():
    """Base-case dispatch within contingency limits → no violations flagged."""
    model, generators, _, common = _make_3bus()

    base = solve_hourly_sced(**common)
    assert base["status"] == "optimal"

    # In this ring topology the worst post-contingency ratio is 200/150 ≈ 1.33
    # (L12 or L13 fully loaded under the other's outage).  A threshold above
    # that value means nothing is flagged.
    violations = screen_n1_violations(
        dispatch=base["dispatch"],
        effective_load=pd.Series({1: 0, 2: 200, 3: 0}),
        generators=generators,
        bus_list=[1, 2, 3],
        line_list=model.line_list,
        lines=model.lines,
        contingency_ptdf=model.contingency_ptdf,
        base_line_flows=base["line_flows"],
        violation_threshold=1.4,   # above max ratio ~1.33 → nothing flagged
    )
    assert violations == set()


def test_screen_n1_detects_violation():
    """After unconstrained base solve, CTG_L13 overloads L12."""
    model, generators, _, common = _make_3bus()
    base = solve_hourly_sced(**common)

    violations = screen_n1_violations(
        dispatch=base["dispatch"],
        effective_load=pd.Series({1: 0, 2: 200, 3: 0}),
        generators=generators,
        bus_list=[1, 2, 3],
        line_list=model.line_list,
        lines=model.lines,
        contingency_ptdf=model.contingency_ptdf,
        base_line_flows=base["line_flows"],
        violation_threshold=0.95,
    )
    # Losing L13 forces all 200 MW through L12, exceeding 0.95 * 150 = 142.5 MW
    assert ("CTG_L13", "L12") in violations


def test_screen_n1_prefilter_suppresses_low_flow_lines():
    """prefilter_ratio skips lines whose base flow is a small fraction of the limit."""
    model, generators, _, common = _make_3bus()
    base = solve_hourly_sced(**common)

    eff_load = pd.Series({1: 0, 2: 200, 3: 0})

    # Without prefilter
    viol_all = screen_n1_violations(
        dispatch=base["dispatch"], effective_load=eff_load,
        generators=generators, bus_list=[1, 2, 3],
        line_list=model.line_list, lines=model.lines,
        contingency_ptdf=model.contingency_ptdf,
        base_line_flows=base["line_flows"],
        violation_threshold=0.95, prefilter_ratio=0.0,
    )

    # With a very high prefilter_ratio (0.99) → base flows are well below 0.99 *
    # contingency_limit for L23 (which carries ~0 MW in base) so L23 is pruned.
    viol_filtered = screen_n1_violations(
        dispatch=base["dispatch"], effective_load=eff_load,
        generators=generators, bus_list=[1, 2, 3],
        line_list=model.line_list, lines=model.lines,
        contingency_ptdf=model.contingency_ptdf,
        base_line_flows=base["line_flows"],
        violation_threshold=0.95, prefilter_ratio=0.99,
    )

    # Filtered set is a subset of the unfiltered set
    assert viol_filtered <= viol_all
    # L23 base flow ≈ 0 → it should be pruned by the high prefilter
    assert not any(line_id == "L23" for _, line_id in viol_filtered)


# ── solve_hourly_sced_n1 ─────────────────────────────────────────────────────

def test_n1_base_solve_no_contingency_ptdf():
    """Without contingency_ptdf, n1 wrapper behaves like the base solver."""
    model, generators, settlement_points, common = _make_3bus()

    base = solve_hourly_sced(**common)
    n1 = solve_hourly_sced_n1(**common, contingency_ptdf=None)

    assert n1["status"] == "optimal"
    assert n1["n1_iterations"] == 1
    assert n1["n1_active_pairs"] == set()
    assert abs(n1["dispatch"]["G1"] - base["dispatch"]["G1"]) < 1e-4
    assert abs(n1["dispatch"]["G2"] - base["dispatch"]["G2"]) < 1e-4


def test_n1_converges_and_respects_contingency_limit():
    """Iterative N-1 screening converges and the final dispatch is N-1 feasible.

    Without N-1 constraints: G1=200, G2=0.
    CTG_L13 → L12 post-ctg flow = 200 MW > contingency_limit = 150 MW.
    The wrapper should add (CTG_L13, L12) and re-solve, causing G2 to pick up
    load near Bus3 so that G1 can reduce its output and relieve L12.
    """
    model, generators, settlement_points, common = _make_3bus()

    result = solve_hourly_sced_n1(
        **common,
        contingency_ptdf=model.contingency_ptdf,
        violation_threshold=0.95,
        prefilter_ratio=0.0,
        ctg_violation_cost=5000.0,
        max_iter=3,
    )

    assert result["status"] == "optimal"
    assert result["n1_iterations"] >= 2          # at least one re-solve happened
    assert len(result["n1_active_pairs"]) > 0    # at least one pair was screened in

    # Verify N-1 feasibility of the final dispatch for the screened pairs
    eff_load = pd.Series({1: 0, 2: 200, 3: 0})
    gen_at_bus: dict = {}
    for g, gd in generators.items():
        gen_at_bus.setdefault(gd["bus"], []).append(g)
    net_inj = np.array([
        sum(result["dispatch"].get(g, 0) for g in gen_at_bus.get(b, []))
        - float(eff_load.get(b, 0))
        for b in [1, 2, 3]
    ])
    for ctg_id, line_id in result["n1_active_pairs"]:
        ctg_ptdf = model.contingency_ptdf[ctg_id]
        ctg_flows = ctg_ptdf @ net_inj
        k = model.line_list.index(line_id)
        ctg_limit = model.lines[line_id]["contingency_limit"]
        # With ctg_violation_cost=5000 >> max gen cost ($80), the solver strongly
        # prefers not to violate; post-ctg flow should be within the limit.
        assert abs(ctg_flows[k]) <= ctg_limit * 1.01, (
            f"{ctg_id} / {line_id}: post-ctg flow {ctg_flows[k]:.1f} MW "
            f"> limit {ctg_limit} MW"
        )


def test_n1_violation_threshold_tunable():
    """A tighter threshold (0.5) screens more pairs than the default (0.95)."""
    model, generators, settlement_points, common = _make_3bus()

    result_tight = solve_hourly_sced_n1(
        **common, contingency_ptdf=model.contingency_ptdf,
        violation_threshold=0.50, max_iter=3,
    )
    result_loose = solve_hourly_sced_n1(
        **common, contingency_ptdf=model.contingency_ptdf,
        violation_threshold=0.95, max_iter=3,
    )

    assert result_tight["status"] == "optimal"
    assert result_loose["status"] == "optimal"
    # Tighter threshold means more (or equal) pairs are activated
    assert len(result_tight["n1_active_pairs"]) >= len(result_loose["n1_active_pairs"])


def test_n1_ctg_line_duals_populated_for_active_pairs():
    """ctg_line_duals is returned for every active (ctg_id, line_id) pair."""
    model, generators, settlement_points, common = _make_3bus()

    result = solve_hourly_sced_n1(
        **common,
        contingency_ptdf=model.contingency_ptdf,
        violation_threshold=0.95,
        max_iter=3,
    )

    assert result["status"] == "optimal"
    for pair in result["n1_active_pairs"]:
        assert pair in result["ctg_line_duals"], (
            f"pair {pair} is active but missing from ctg_line_duals"
        )
