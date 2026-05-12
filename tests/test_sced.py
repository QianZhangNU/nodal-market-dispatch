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
