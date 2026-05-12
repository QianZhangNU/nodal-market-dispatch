"""SCUC solver tests."""

import numpy as np
import pandas as pd


def test_scuc_two_generator_no_network():
    from solvers.scuc import build_initial_state, solve_daily_scuc

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
    hours = pd.date_range("2023-01-01", periods=24, freq="h")
    load = pd.DataFrame({1: [60] * 24}, index=hours)
    pmax = pd.DataFrame({"G1": [100] * 24, "G2": [50] * 24}, index=hours)
    cost = pd.DataFrame({"G1": [30] * 24, "G2": [80] * 24}, index=hours)

    result = solve_daily_scuc(
        generators=generators,
        gen_list=["G1", "G2"],
        bus_list=[1],
        bus_idx={1: 0},
        lines={},
        ptdf=np.zeros((0, 1)),
        day_start=hours[0],
        load_24h=load,
        pmax_24h=pmax,
        gen_cost_24h=cost,
        initial_state=build_initial_state(generators),
        use_network=False,
    )

    assert result["status"] == "optimal"
    assert result["u_schedule"]["G1"].sum() == 24
    assert result["total_cost"] > 0


def test_scuc_ercot_17bus_network_smoke():
    from solvers.network import NetworkModel
    from solvers.scuc import build_initial_state, solve_daily_scuc
    from examples.ercot_17bus.profiles import build_all_profiles
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, GENERATORS, GEN_LIST,
        SLACK_BUS, CONTINGENCIES,
    )

    network = NetworkModel(
        buses=BUSES,
        lines=LINES,
        slack_bus=SLACK_BUS,
        contingencies=CONTINGENCIES,
    )
    profiles = build_all_profiles()
    day = pd.Timestamp("2023-01-27")
    hours = pd.date_range(day, periods=24, freq="h")

    result = solve_daily_scuc(
        generators=GENERATORS,
        gen_list=GEN_LIST,
        bus_list=BUS_LIST,
        bus_idx=BUS_IDX,
        lines=LINES,
        ptdf=network.ptdf,
        day_start=day,
        load_24h=profiles["load"].loc[hours],
        pmax_24h=profiles["pmax_t"].loc[hours],
        gen_cost_24h=profiles["gen_cost_t"].loc[hours],
        initial_state=build_initial_state(GENERATORS),
        use_network=True,
    )

    assert result["status"] == "optimal"
    assert int(result["u_schedule"].iloc[0].sum()) >= 5
