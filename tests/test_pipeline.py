"""Cross-solver pipeline smoke tests."""

import pandas as pd


def test_scuc_to_sced_pipeline_ercot_17bus_sample_hours():
    from solvers.network import NetworkModel
    from solvers.scuc import build_initial_state, solve_daily_scuc
    from solvers.sced import solve_hourly_sced
    from examples.ercot_17bus.profiles import build_all_profiles
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST,
        GENERATORS, GEN_LIST, SETTLEMENT_POINTS,
        CONTINGENCIES, GENERIC_TRANSMISSION_CONSTRAINTS, SLACK_BUS,
    )

    network = NetworkModel(
        buses=BUSES,
        lines=LINES,
        slack_bus=SLACK_BUS,
        contingencies=CONTINGENCIES,
    )
    profiles = build_all_profiles()
    day = pd.Timestamp("2023-07-31")
    hours = pd.date_range(day, periods=24, freq="h")

    scuc = solve_daily_scuc(
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
    assert scuc["status"] == "optimal"

    sced_ok = 0
    for h in [0, 12, 20]:
        hour = hours[h]
        u_fixed = {g: int(scuc["u_schedule"].iloc[h][g]) for g in GEN_LIST}
        p_prev = {g: float(scuc["p_dispatch"].iloc[max(h - 1, 0)][g]) for g in GEN_LIST}
        sced = solve_hourly_sced(
            generators=GENERATORS,
            gen_list=GEN_LIST,
            bus_list=BUS_LIST,
            bus_idx=BUS_IDX,
            lines=LINES,
            line_list=LINE_LIST,
            ptdf=network.ptdf,
            settlement_points=SETTLEMENT_POINTS,
            gtcs=GENERIC_TRANSMISSION_CONSTRAINTS,
            hour=hour,
            u_fixed=u_fixed,
            p_prev=p_prev,
            load_h=profiles["load"].loc[hour],
            pmax_h=profiles["pmax_t"].loc[hour],
            cost_h=profiles["gen_cost_t"].loc[hour],
            use_network=True,
            use_ramp=True,
            use_gtc=True,
        )
        sced_ok += int(sced["status"] == "optimal")

    assert sced_ok >= 2
