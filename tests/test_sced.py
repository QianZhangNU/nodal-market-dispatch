"""SCED solver tests."""

import pandas as pd


def test_sced_two_bus_congestion_dispatch_and_lmp():
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
    settlement_points = {
        "RN1": dict(type="resource_node", bus=1),
        "RN2": dict(type="resource_node", bus=2),
    }

    result = solve_hourly_sced(
        generators=generators,
        gen_list=["G1", "G2"],
        bus_list=[1, 2],
        bus_idx={1: 0, 2: 1},
        lines=model.lines,
        line_list=model.line_list,
        ptdf=model.ptdf,
        settlement_points=settlement_points,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={"G1": 1, "G2": 1},
        p_prev={"G1": 50, "G2": 20},
        load_h=pd.Series({1: 10, 2: 70}),
        pmax_h=pd.Series({"G1": 100, "G2": 50}),
        cost_h=pd.Series({"G1": 30, "G2": 80}),
        use_ramp=False,
    )

    assert result["status"] == "optimal"
    assert result["dispatch"]["G1"] == 60
    assert result["dispatch"]["G2"] == 20
    assert result["bus_lmp"][2] > result["bus_lmp"][1]
    assert result["binding_lines"] == ["L1"]


def test_shadow_price_sign_uncongested():
    """Uncongested: line dual is zero and LMP is uniform across buses.

    Setup: G1 (bus1, $30) can serve all 80 MW of load with the line limit set
    to 500 MW, so no congestion occurs.  The energy price lam equals G1's cost
    and the congestion component is zero everywhere.

    Sign-convention check:
      mu_L1 = 0  =>  lmp_cong_bus = 0 at every bus
      LMP[bus1] = LMP[bus2] = lam = 30 $/MWh
    """
    from solvers.network import NetworkModel
    from solvers.sced import solve_hourly_sced

    model = NetworkModel(
        buses={1: {"name": "Gen"}, 2: {"name": "Load"}},
        lines={"L1": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                      "flow_limit": 500, "contingency_limit": 400}},
        slack_bus=1,
    )
    generators = {
        "G1": dict(bus=1, Pmin=0, Pmax=100, cost_b=30, ramp_up=100, ramp_down=100),
        "G2": dict(bus=2, Pmin=0, Pmax=50, cost_b=80, ramp_up=50, ramp_down=50),
    }
    settlement_points = {
        "RN1": dict(type="resource_node", bus=1),
        "RN2": dict(type="resource_node", bus=2),
    }

    result = solve_hourly_sced(
        generators=generators,
        gen_list=["G1", "G2"],
        bus_list=[1, 2],
        bus_idx={1: 0, 2: 1},
        lines=model.lines,
        line_list=model.line_list,
        ptdf=model.ptdf,
        settlement_points=settlement_points,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={"G1": 1, "G2": 1},
        p_prev={"G1": 50, "G2": 0},
        load_h=pd.Series({1: 10, 2: 70}),
        pmax_h=pd.Series({"G1": 100, "G2": 50}),
        cost_h=pd.Series({"G1": 30, "G2": 80}),
        use_ramp=False,
    )

    assert result["status"] == "optimal"
    # Cheap G1 serves all load; G2 is idle
    assert abs(result["dispatch"]["G1"] - 80) < 1e-4
    assert abs(result["dispatch"]["G2"] - 0) < 1e-4

    # No congestion: line dual is zero
    assert abs(result["line_duals"]["L1"]) < 1e-4

    # Energy price equals G1's marginal cost; no congestion component anywhere
    assert abs(result["lmp_energy"] - 30) < 1e-4
    assert abs(result["lmp_cong_bus"][1]) < 1e-4
    assert abs(result["lmp_cong_bus"][2]) < 1e-4

    # Uniform LMP across buses
    assert abs(result["bus_lmp"][1] - 30) < 1e-4
    assert abs(result["bus_lmp"][2] - 30) < 1e-4


def test_shadow_price_sign_backward_congestion():
    """Backward congestion: line dual is negative and LMP[gen pocket] < lam.

    Setup:
      G2 (bus2, $30/MWh) is cheap; G1 (bus1, $80/MWh) is expensive.
      All 70 MW of load sits at bus1.  Without the line limit, G2 would serve
      all load (backward flow bus2 → bus1 = 70 MW).  With flow_limit=50 MW,
      G2 can export at most 50 MW backward, so G1 must cover the remaining 20 MW.

    Network (slack = bus1, PTDF[L1, bus1] = 0, PTDF[L1, bus2] = -1):
      The backward constraint row is  -PTDF[L1,bus2] * G2 <= F_max
      i.e.  G2 <= 50, which binds at G2 = 50.

    Sign-convention check:
      mu_L1 = -(G1_cost - G2_cost) = -(80 - 30) = -50 $/MWh  (negative = backward binding)
      lam   = G1_cost = 80 $/MWh  (marginal unit at the load-pocket slack bus)
      cong[bus1] = -mu_L1 * PTDF[L1, bus1] = -(-50) * 0  = 0
      cong[bus2] = -mu_L1 * PTDF[L1, bus2] = -(-50) * (-1) = -50
      LMP[bus1]  = 80 + 0   = 80 $/MWh  (load pocket, sets the energy price)
      LMP[bus2]  = 80 + (-50) = 30 $/MWh  (gen pocket, depressed by export constraint)
    """
    from solvers.network import NetworkModel
    from solvers.sced import solve_hourly_sced

    model = NetworkModel(
        buses={1: {"name": "Load"}, 2: {"name": "Gen"}},
        lines={"L1": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                      "flow_limit": 50, "contingency_limit": 40}},
        slack_bus=1,
    )
    generators = {
        "G1": dict(bus=1, Pmin=0, Pmax=50, cost_b=80, ramp_up=50, ramp_down=50),
        "G2": dict(bus=2, Pmin=0, Pmax=100, cost_b=30, ramp_up=100, ramp_down=100),
    }
    settlement_points = {
        "RN1": dict(type="resource_node", bus=1),
        "RN2": dict(type="resource_node", bus=2),
    }

    result = solve_hourly_sced(
        generators=generators,
        gen_list=["G1", "G2"],
        bus_list=[1, 2],
        bus_idx={1: 0, 2: 1},
        lines=model.lines,
        line_list=model.line_list,
        ptdf=model.ptdf,
        settlement_points=settlement_points,
        hour=pd.Timestamp("2023-01-01"),
        u_fixed={"G1": 1, "G2": 1},
        p_prev={"G1": 0, "G2": 50},
        load_h=pd.Series({1: 70, 2: 0}),
        pmax_h=pd.Series({"G1": 50, "G2": 100}),
        cost_h=pd.Series({"G1": 80, "G2": 30}),
        use_ramp=False,
    )

    assert result["status"] == "optimal"
    # G2 is capped by backward flow limit; G1 covers the shortfall
    assert abs(result["dispatch"]["G1"] - 20) < 1e-4
    assert abs(result["dispatch"]["G2"] - 50) < 1e-4

    # Backward limit binding => mu_L1 < 0
    assert result["line_duals"]["L1"] < 0
    assert abs(result["line_duals"]["L1"] - (-50)) < 1e-4

    # Energy price = expensive local gen G1's cost
    assert abs(result["lmp_energy"] - 80) < 1e-4

    # Congestion components match the worked example in the docstring
    assert abs(result["lmp_cong_bus"][1] - 0) < 1e-4
    assert abs(result["lmp_cong_bus"][2] - (-50)) < 1e-4

    # LMPs: load pocket (bus1) is expensive, gen pocket (bus2) is cheap
    assert abs(result["bus_lmp"][1] - 80) < 1e-4
    assert abs(result["bus_lmp"][2] - 30) < 1e-4
    assert result["bus_lmp"][1] > result["bus_lmp"][2]
