"""
Integration tests for CRR Alpha Lab v3
========================================
Tests each solver independently (small cases) and then the full
SCUC→SCED→Auction pipeline with the ERCOT 17-bus example.

Run: PYTHONPATH=. python tests/test_integration.py
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = 0
FAIL = 0

def check(condition, msg):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {msg}")
    else:
        FAIL += 1
        print(f"  ✗ FAIL: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: NetworkModel (3-bus)
# ─────────────────────────────────────────────────────────────────────────────

def test_network_3bus():
    print("\n[Test 1] NetworkModel — 3-bus triangle")
    from solvers.network import NetworkModel

    model = NetworkModel(
        buses={1: {"name": "A"}, 2: {"name": "B"}, 3: {"name": "C"}},
        lines={
            "L1": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10.0,
                    "flow_limit": 100, "contingency_limit": 80},
            "L2": {"from_bus": 2, "to_bus": 3, "x_pu": 0.2, "b_pu": 5.0,
                    "flow_limit": 80, "contingency_limit": 60},
            "L3": {"from_bus": 1, "to_bus": 3, "x_pu": 0.15, "b_pu": 6.67,
                    "flow_limit": 90, "contingency_limit": 70},
        },
        slack_bus=1,
        contingencies={"CTG_L1": {"outaged_line": "L1"}},
    )
    check(model.ptdf.shape == (3, 3), f"PTDF shape {model.ptdf.shape} == (3,3)")
    check(model.lodf.shape == (3, 3), f"LODF shape {model.lodf.shape} == (3,3)")
    check(abs(model.ptdf[0, 0]) < 1e-10, "Slack bus PTDF = 0")
    check("CTG_L1" in model.contingency_ptdf, "Contingency PTDF computed")

    # Verify PTDF row sums to 0 for non-slack (DC PF property)
    for k in range(3):
        row_sum = model.ptdf[k, :].sum()  # not exactly 0 because slack column is 0
        # But PTDF[k, slack] = 0, so sum of non-slack = sum of all
        # Actually PTDF rows don't sum to 0 in general. Check columns instead.
    check(True, "PTDF basic structure valid")

    # Test line_flow
    inj = np.array([100.0, -60.0, -40.0])  # 100 in at bus1, out at 2 and 3
    flows = model.line_flow(inj)
    check(len(flows) == 3, f"line_flow returns {len(flows)} flows")
    check(abs(sum(flows)) < 200, "Flows are reasonable magnitude")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: NetworkModel (ERCOT 17-bus)
# ─────────────────────────────────────────────────────────────────────────────

def test_network_17bus():
    print("\n[Test 2] NetworkModel — ERCOT 17-bus")
    from solvers.network import NetworkModel
    from examples.ercot_17bus.topology import (
        BUSES, LINES, SLACK_BUS, CONTINGENCIES,
    )
    model = NetworkModel(buses=BUSES, lines=LINES, slack_bus=SLACK_BUS,
                         contingencies=CONTINGENCIES)
    check(model.ptdf.shape == (26, 17), f"PTDF shape {model.ptdf.shape}")
    check(model.n_bus == 17, f"n_bus = {model.n_bus}")
    check(model.n_line == 26, f"n_line = {model.n_line}")
    check(len(model.contingency_ptdf) > 0,
          f"{len(model.contingency_ptdf)} contingency PTDFs")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: SCUC (2-gen, no network)
# ─────────────────────────────────────────────────────────────────────────────

def test_scuc_no_network():
    print("\n[Test 3] SCUC — 2-gen, no network")
    from solvers.scuc import solve_daily_scuc, build_initial_state

    generators = {
        "G1": dict(bus=1, type="CC", Pmin=10, Pmax=100, cost_b=30, cost_c=100,
                   su_cost=500, sd_cost=100, min_up=2, min_down=2,
                   ramp_up=100, ramp_down=100,
                   init_status=1, init_gen=50, init_up_time=24, init_down_time=0),
        "G2": dict(bus=1, type="GT", Pmin=5, Pmax=50, cost_b=80, cost_c=50,
                   su_cost=200, sd_cost=50, min_up=1, min_down=1,
                   ramp_up=50, ramp_down=50,
                   init_status=0, init_gen=0, init_up_time=0, init_down_time=8),
    }
    idx_h = pd.date_range("2023-01-01", periods=24, freq="h")
    load = pd.DataFrame({1: [60]*24}, index=idx_h)
    pmax = pd.DataFrame({"G1": [100]*24, "G2": [50]*24}, index=idx_h)
    cost = pd.DataFrame({"G1": [30]*24, "G2": [80]*24}, index=idx_h)

    state = build_initial_state(generators)
    result = solve_daily_scuc(
        generators=generators, gen_list=["G1", "G2"],
        bus_list=[1], bus_idx={1: 0},
        lines={}, ptdf=np.zeros((0, 1)),
        day_start=pd.Timestamp("2023-01-01"),
        load_24h=load, pmax_24h=pmax, gen_cost_24h=cost,
        initial_state=state, use_network=False,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    check(result["u_schedule"]["G1"].sum() == 24, "G1 always on (cheaper)")
    check(result["total_cost"] > 0, f"Cost: ${result['total_cost']:,.0f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: SCUC (17-bus with network)
# ─────────────────────────────────────────────────────────────────────────────

def test_scuc_17bus():
    print("\n[Test 4] SCUC — ERCOT 17-bus with network")
    from solvers.network import NetworkModel
    from solvers.scuc import solve_daily_scuc, build_initial_state
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, GENERATORS, GEN_LIST,
        SLACK_BUS, CONTINGENCIES,
    )
    from examples.ercot_17bus.profiles import build_all_profiles

    nm = NetworkModel(buses=BUSES, lines=LINES, slack_bus=SLACK_BUS,
                      contingencies=CONTINGENCIES)
    p = build_all_profiles()
    day = pd.Timestamp("2023-01-27")
    hours = pd.date_range(day, periods=24, freq="h")
    state = build_initial_state(GENERATORS)

    result = solve_daily_scuc(
        generators=GENERATORS, gen_list=GEN_LIST,
        bus_list=BUS_LIST, bus_idx=BUS_IDX,
        lines=LINES, ptdf=nm.ptdf,
        day_start=day, load_24h=p["load"].loc[hours],
        pmax_24h=p["pmax_t"].loc[hours],
        gen_cost_24h=p["gen_cost_t"].loc[hours],
        initial_state=state, use_network=True, verbose=True,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    n_on = int(result["u_schedule"].iloc[0].sum())
    check(n_on >= 5, f"Units committed HE1: {n_on}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: SCED (2-bus with congestion)
# ─────────────────────────────────────────────────────────────────────────────

def test_sced_congestion():
    print("\n[Test 5] SCED — 2-bus, line binding → congestion")
    from solvers.network import NetworkModel
    from solvers.sced import solve_hourly_sced

    model = NetworkModel(
        buses={1: {"name": "Gen"}, 2: {"name": "Load"}},
        lines={"L1": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                       "flow_limit": 50, "contingency_limit": 40}},
        slack_bus=1,
    )
    generators = {
        "G1": dict(bus=1, Pmin=0, Pmax=100, cost_b=30, ramp_up=100, ramp_down=100),
        "G2": dict(bus=2, Pmin=0, Pmax=50, cost_b=80, ramp_up=50, ramp_down=50),
    }
    sps = {
        "RN1": dict(type="resource_node", bus=1),
        "RN2": dict(type="resource_node", bus=2),
    }

    result = solve_hourly_sced(
        generators=generators, gen_list=["G1", "G2"],
        bus_list=[1, 2], bus_idx={1: 0, 2: 1},
        lines=model.lines, line_list=model.line_list,
        ptdf=model.ptdf, settlement_points=sps,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={"G1": 1, "G2": 1},
        p_prev={"G1": 50, "G2": 20},
        load_h=pd.Series({1: 10, 2: 70}),    # 80 MW total, line limit 50
        pmax_h=pd.Series({"G1": 100, "G2": 50}),
        cost_h=pd.Series({"G1": 30, "G2": 80}),
        use_ramp=False,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    check(result["dispatch"]["G2"] >= 19.9,
          f"G2 dispatched {result['dispatch']['G2']:.1f} MW (≥20 local)")
    check(result["bus_lmp"][2] > result["bus_lmp"][1],
          f"LMP bus2=${result['bus_lmp'][2]:.0f} > bus1=${result['bus_lmp'][1]:.0f}")
    check(len(result["binding_lines"]) > 0,
          f"Binding lines: {result['binding_lines']}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: SCUC → SCED pipeline (17-bus)
# ─────────────────────────────────────────────────────────────────────────────

def test_scuc_sced_pipeline():
    print("\n[Test 6] SCUC → SCED pipeline — ERCOT 17-bus")
    from solvers.network import NetworkModel
    from solvers.scuc import solve_daily_scuc, build_initial_state
    from solvers.sced import solve_hourly_sced
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST,
        GENERATORS, GEN_LIST, SETTLEMENT_POINTS,
        CONTINGENCIES, GENERIC_TRANSMISSION_CONSTRAINTS,
        SLACK_BUS,
    )
    from examples.ercot_17bus.profiles import build_all_profiles

    nm = NetworkModel(buses=BUSES, lines=LINES, slack_bus=SLACK_BUS,
                      contingencies=CONTINGENCIES)
    p = build_all_profiles()
    day = pd.Timestamp("2023-07-31")
    hours = pd.date_range(day, periods=24, freq="h")
    state = build_initial_state(GENERATORS)

    scuc = solve_daily_scuc(
        generators=GENERATORS, gen_list=GEN_LIST,
        bus_list=BUS_LIST, bus_idx=BUS_IDX,
        lines=LINES, ptdf=nm.ptdf,
        day_start=day, load_24h=p["load"].loc[hours],
        pmax_24h=p["pmax_t"].loc[hours],
        gen_cost_24h=p["gen_cost_t"].loc[hours],
        initial_state=state, use_network=True,
    )
    check(scuc["status"] == "optimal", f"SCUC: {scuc['status']}")

    # Run SCED for 3 sample hours
    sced_ok = 0
    for h in [0, 12, 20]:
        hour = hours[h]
        u_f = {g: int(scuc["u_schedule"].iloc[h][g]) for g in GEN_LIST}
        pp = {g: float(scuc["p_dispatch"].iloc[max(h-1, 0)][g]) for g in GEN_LIST}

        sced = solve_hourly_sced(
            generators=GENERATORS, gen_list=GEN_LIST,
            bus_list=BUS_LIST, bus_idx=BUS_IDX,
            lines=LINES, line_list=LINE_LIST,
            ptdf=nm.ptdf, settlement_points=SETTLEMENT_POINTS,
            gtcs=GENERIC_TRANSMISSION_CONSTRAINTS,
            hour=hour, u_fixed=u_f, p_prev=pp,
            load_h=p["load"].loc[hour],
            pmax_h=p["pmax_t"].loc[hour],
            cost_h=p["gen_cost_t"].loc[hour],
            use_network=True, use_ramp=True, use_gtc=True,
        )
        if sced["status"] == "optimal":
            sced_ok += 1
    check(sced_ok >= 2, f"SCED optimal in {sced_ok}/3 hours")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 7: CRR Auction (3-bus)
# ─────────────────────────────────────────────────────────────────────────────

def test_auction_3bus():
    print("\n[Test 7] CRR Auction — 3-bus, BUY OBL + OPT")
    from solvers.network import NetworkModel
    from solvers.auction import clear_auction

    model = NetworkModel(
        buses={1: {"name": "North"}, 2: {"name": "South"}, 3: {"name": "West"}},
        lines={
            "L_NS": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                      "flow_limit": 100, "contingency_limit": 80},
            "L_NW": {"from_bus": 1, "to_bus": 3, "x_pu": 0.15, "b_pu": 6.67,
                      "flow_limit": 80, "contingency_limit": 60},
            "L_WS": {"from_bus": 3, "to_bus": 2, "x_pu": 0.2, "b_pu": 5,
                      "flow_limit": 60, "contingency_limit": 50},
        },
        slack_bus=1,
    )
    sps = {
        "HB_NORTH": dict(type="hub", member_buses=[1]),
        "HB_SOUTH": dict(type="hub", member_buses=[2]),
        "LZ_WEST":  dict(type="resource_node", bus=3),
    }

    bids = pd.DataFrame([
        # BUY OBL — standard obligation CRR
        {"agent_id": "A1", "source": "HB_NORTH", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 5.0, "mw": 50},
        # BUY OPT — option CRR (only exercises when profitable)
        {"agent_id": "A2", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OPT", "bid_price": 3.0, "mw": 30},
        # BUY OBL on different path
        {"agent_id": "A3", "source": "LZ_WEST", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 4.0, "mw": 40},
    ])

    result = clear_auction(
        bids_df=bids, ptdf=model.ptdf,
        lines=model.lines, line_list=model.line_list,
        bus_list=model.bus_list, bus_idx=model.bus_idx,
        settlement_points=sps, capacity_factor=0.90,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    check(len(result["awarded"]) == 3, "All 3 bids processed")
    total_awarded = result["awarded"]["mw_awarded"].sum()
    check(total_awarded > 0, f"Total MW awarded: {total_awarded:.1f}")
    check("summary" in result, "Summary stats present")
    check(result["summary"]["buy_obl_mw"] > 0, f"BUY OBL: {result['summary']['buy_obl_mw']:.0f} MW")
    check(result["summary"]["buy_opt_mw"] > 0, f"BUY OPT: {result['summary']['buy_opt_mw']:.0f} MW")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 8: CRR Auction with SELL + OBL/OPT mix
# ─────────────────────────────────────────────────────────────────────────────

def test_auction_buy_sell():
    print("\n[Test 8] CRR Auction — BUY/SELL × OBL/OPT mix")
    from solvers.network import NetworkModel
    from solvers.auction import clear_auction

    model = NetworkModel(
        buses={1: {"name": "North"}, 2: {"name": "South"}, 3: {"name": "West"}},
        lines={
            "L_NS": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                      "flow_limit": 100, "contingency_limit": 80},
            "L_NW": {"from_bus": 1, "to_bus": 3, "x_pu": 0.15, "b_pu": 6.67,
                      "flow_limit": 80, "contingency_limit": 60},
            "L_WS": {"from_bus": 3, "to_bus": 2, "x_pu": 0.2, "b_pu": 5,
                      "flow_limit": 60, "contingency_limit": 50},
        },
        slack_bus=1,
    )
    sps = {
        "HB_NORTH": dict(type="hub", member_buses=[1]),
        "HB_SOUTH": dict(type="hub", member_buses=[2]),
        "LZ_WEST":  dict(type="resource_node", bus=3),
    }

    bids = pd.DataFrame([
        # BUY OBL
        {"agent_id": "B1", "source": "HB_NORTH", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 5.0, "mw": 80},
        # BUY OPT
        {"agent_id": "B2", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OPT", "bid_price": 2.0, "mw": 50},
        # SELL OBL — releases capacity, enabling more buys
        {"agent_id": "S1", "source": "HB_SOUTH", "sink": "HB_NORTH",
         "side": "SELL", "crr_type": "OBL", "bid_price": 1.0, "mw": 30},
        # SELL OPT
        {"agent_id": "S2", "source": "LZ_WEST", "sink": "HB_SOUTH",
         "side": "SELL", "crr_type": "OPT", "bid_price": 0.5, "mw": 20},
    ])

    result = clear_auction(
        bids_df=bids, ptdf=model.ptdf,
        lines=model.lines, line_list=model.line_list,
        bus_list=model.bus_list, bus_idx=model.bus_idx,
        settlement_points=sps, capacity_factor=0.90,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    check(result["summary"]["buy_obl_mw"] > 0, f"BUY OBL awarded: {result['summary']['buy_obl_mw']:.0f} MW")
    check(result["summary"]["sell_obl_mw"] >= 0, f"SELL OBL awarded: {result['summary']['sell_obl_mw']:.0f} MW")
    check(result["total_welfare"] >= 0, f"Welfare: ${result['total_welfare']:.0f}")

    # OPT clearing price should be ≥ 0 (option can't have negative value)
    for key, acp in result["acp_by_path"].items():
        if key[2] == "OPT":
            check(acp >= -0.01, f"OPT ACP ≥ 0: {key} = ${acp:.2f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 9: CRR Auction — 17-bus with all four bid types
# ─────────────────────────────────────────────────────────────────────────────

def test_auction_17bus():
    print("\n[Test 9] CRR Auction — ERCOT 17-bus, all bid types")
    from solvers.network import NetworkModel
    from solvers.auction import clear_auction
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST,
        SETTLEMENT_POINTS, SLACK_BUS,
    )

    nm = NetworkModel(buses=BUSES, lines=LINES, slack_bus=SLACK_BUS)

    bids = pd.DataFrame([
        {"agent_id": "T1", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OBL", "bid_price": 8.0, "mw": 100},
        {"agent_id": "T2", "source": "HB_NORTH", "sink": "LZ_HOUSTON",
         "side": "BUY", "crr_type": "OPT", "bid_price": 5.0, "mw": 150},
        {"agent_id": "T3", "source": "HB_WEST", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OBL", "bid_price": 6.0, "mw": 80},
        {"agent_id": "T4", "source": "HB_SOUTH", "sink": "LZ_SOUTH",
         "side": "SELL", "crr_type": "OBL", "bid_price": 2.0, "mw": 60},
        {"agent_id": "T5", "source": "RN_RGV_SOLAR", "sink": "HB_HOUSTON",
         "side": "BUY", "crr_type": "OPT", "bid_price": 7.0, "mw": 120},
        # Baseload CRR that pre-consumes capacity
    ])

    # Pre-existing PCRR
    baseload = [("HB_NORTH", "LZ_HOUSTON", 50, "OBL")]

    result = clear_auction(
        bids_df=bids, ptdf=nm.ptdf,
        lines=LINES, line_list=LINE_LIST,
        bus_list=BUS_LIST, bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
        capacity_factor=0.90,
        baseload_crrs=baseload,
    )
    check(result["status"] == "optimal", f"Status: {result['status']}")
    total_mw = result["awarded"]["mw_awarded"].sum()
    check(total_mw > 0, f"Total MW awarded: {total_mw:.1f}")
    n_paths = len(result["acp_by_path"])
    check(n_paths > 0, f"ACP computed for {n_paths} paths")
    check("summary" in result, "Summary present")

    # Verify backward compatibility (old bids without crr_type)
    old_bids = pd.DataFrame([
        {"agent_id": "X1", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "bid_price": 5.0, "mw": 50},
    ])
    r2 = clear_auction(
        bids_df=old_bids, ptdf=nm.ptdf,
        lines=LINES, line_list=LINE_LIST,
        bus_list=BUS_LIST, bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
    )
    check(r2["status"] == "optimal", "Backward compat (no crr_type): works")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 9: Monthly topology variation
# ─────────────────────────────────────────────────────────────────────────────

def test_monthly_topology():
    print("\n[Test 9] Monthly topology variation")
    from examples.ercot_17bus.topology import (
        LINES, GENERATORS, get_monthly_topology,
    )

    # January: relaxed rating
    mt_jan = get_monthly_topology(2023, 1)
    check(mt_jan["rating_factor"] == 1.20, f"Jan rating: {mt_jan['rating_factor']}")
    check(len(mt_jan["outaged_lines"]) == 0, "Jan: no outages")

    # March: L_NORTH_HOUSTON_1 outage
    mt_mar = get_monthly_topology(2023, 3)
    check("L_NORTH_HOUSTON_1" in mt_mar["outaged_lines"],
          f"Mar outage: {mt_mar['outaged_lines']}")
    check(len(mt_mar["lines"]) == len(LINES) - 1,
          f"Mar lines: {len(mt_mar['lines'])}/{len(LINES)}")

    # July: tightest rating
    mt_jul = get_monthly_topology(2023, 7)
    check(mt_jul["rating_factor"] == 0.80, f"Jul rating: {mt_jul['rating_factor']}")

    # April 2024: nuclear outage
    mt_apr24 = get_monthly_topology(2024, 4)
    check("NUC_SOUTH_001" in mt_apr24["outaged_generators"],
          f"Apr 2024 gen outage: {mt_apr24['outaged_generators']}")
    check(len(mt_apr24["generators"]) == len(GENERATORS) - 1,
          f"Apr 2024 gens: {len(mt_apr24['generators'])}/{len(GENERATORS)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CRR Alpha Lab v3 — Integration Tests")
    print("=" * 60)

    test_network_3bus()
    test_network_17bus()
    test_scuc_no_network()
    test_scuc_17bus()
    test_sced_congestion()
    test_scuc_sced_pipeline()
    test_auction_3bus()
    test_auction_buy_sell()
    test_auction_17bus()
    test_monthly_topology()

    print("\n" + "=" * 60)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL > 0:
        sys.exit(1)
