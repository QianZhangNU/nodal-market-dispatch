"""
DAM Simulation Runner — ERCOT 17-Bus Example
==============================================
Wires topology-agnostic solvers with the ERCOT 17-bus example data.
Demonstrates the clean separation: solvers know nothing about ERCOT.
"""

import numpy as np
import pandas as pd
import time
import pickle
import os
from dataclasses import dataclass, field

from solvers.network import NetworkModel
from solvers.scuc import solve_daily_scuc, build_initial_state
from solvers.sced import solve_hourly_sced

from examples.ercot_17bus.topology import (
    BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST,
    GENERATORS, GEN_LIST, SETTLEMENT_POINTS, SP_LIST,
    CONTINGENCIES, GENERIC_TRANSMISSION_CONSTRAINTS, GTC_LIST,
    SLACK_BUS, get_monthly_topology,
)
from examples.ercot_17bus.profiles import build_all_profiles
from examples.ercot_17bus.battery import generate_battery_profile


@dataclass
class SimulationResult:
    sp_lmp:      pd.DataFrame = None
    dispatch:    pd.DataFrame = None
    line_flows:  pd.DataFrame = None
    line_duals:  pd.DataFrame = None
    gtc_duals:   pd.DataFrame = None
    daily_stats: list = field(default_factory=list)
    metadata:    dict = field(default_factory=dict)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def run_dam_simulation(fast_mode=True, verbose=True, save_path=None):
    t0 = time.time()
    if verbose:
        print("Generating profiles...")
    profiles = build_all_profiles()
    battery_inj, _ = generate_battery_profile()

    # Select days (stratified sampling for fast mode)
    all_days = pd.date_range(
        profiles["load"].index[0].normalize(),
        profiles["load"].index[-1].normalize(), freq="D")
    if fast_mode:
        sys_load = profiles["load"].sum(axis=1)
        daily_max = sys_load.resample("D").max()
        days = set()
        for (yr, mo), grp in daily_max.groupby([daily_max.index.year, daily_max.index.month]):
            if mo % 2 == 1:
                days.add(grp.idxmax())
        days = sorted(days)
    else:
        days = list(all_days)

    n_days = len(days)
    if verbose:
        print(f"DAM Simulation [{'FAST' if fast_mode else 'FULL'}]: "
              f"{n_days} days, {len(BUS_LIST)} buses, {len(LINE_LIST)} lines, "
              f"{len(GTC_LIST)} GTCs")

    sp_records, dispatch_records = [], []
    flows_records, duals_records, gtc_records = [], [], []
    daily_stats = []

    state = build_initial_state(GENERATORS)
    current_month = None
    nm = None
    m_lines = m_gens = m_gtcs = m_llist = None

    for day_idx, day in enumerate(days):
        day_start = pd.Timestamp(day)
        day_hours = pd.date_range(day_start, periods=24, freq="h")

        # Monthly topology switch
        mk = (day_start.year, day_start.month)
        if mk != current_month:
            current_month = mk
            mt = get_monthly_topology(*mk)
            m_lines, m_gens, m_gtcs = mt["lines"], mt["generators"], mt["gtcs"]
            m_llist = list(m_lines.keys())
            nm = NetworkModel(buses=BUSES, lines=m_lines, slack_bus=SLACK_BUS,
                              contingencies=mt["contingencies"])
            if verbose:
                out = ""
                if mt["outaged_lines"]:
                    out += f" outL={mt['outaged_lines']}"
                if mt["outaged_generators"]:
                    out += f" outG={mt['outaged_generators']}"
                print(f"  [{mk[0]}-{mk[1]:02d}] rating={mt['rating_factor']:.2f}x "
                      f"lines={len(m_lines)}/{len(LINES)} "
                      f"gens={len(m_gens)}/{len(GENERATORS)}{out}")

        load_24h = profiles["load"].loc[day_hours]
        pmax_24h = profiles["pmax_t"].loc[day_hours].copy()
        gen_cost_24h = profiles["gen_cost_t"].loc[day_hours]
        bat_24h = battery_inj.loc[day_hours]

        for g in GEN_LIST:
            if g not in m_gens:
                pmax_24h[g] = 0.0

        eff_load = load_24h - bat_24h.reindex(columns=load_24h.columns, fill_value=0)

        # SCUC
        scuc = solve_daily_scuc(
            generators=GENERATORS, gen_list=GEN_LIST,
            bus_list=BUS_LIST, bus_idx=BUS_IDX,
            lines=m_lines, ptdf=nm.ptdf,
            day_start=day_start, load_24h=eff_load,
            pmax_24h=pmax_24h, gen_cost_24h=gen_cost_24h,
            initial_state=state, use_network=True,
            mip_gap=0.005, time_limit_sec=30.0)

        if scuc["status"] != "optimal":
            if verbose:
                print(f"  [{day.date()}] SCUC failed: {scuc['status']}")
            continue

        # SCED hourly
        n_fail = 0
        prev = {g: state[g]["p"] for g in GEN_LIST}
        for h_idx, hour in enumerate(day_hours):
            u_f = {g: int(scuc["u_schedule"].iloc[h_idx][g]) for g in GEN_LIST}
            sced = solve_hourly_sced(
                generators=GENERATORS, gen_list=GEN_LIST,
                bus_list=BUS_LIST, bus_idx=BUS_IDX,
                lines=m_lines, line_list=m_llist, ptdf=nm.ptdf,
                settlement_points=SETTLEMENT_POINTS,
                gtcs=m_gtcs,
                hour=hour, u_fixed=u_f, p_prev=prev,
                load_h=load_24h.iloc[h_idx],
                pmax_h=pmax_24h.iloc[h_idx],
                cost_h=gen_cost_24h.iloc[h_idx],
                battery_injection_h=bat_24h.iloc[h_idx],
                use_network=True, use_ramp=True, use_gtc=True)

            if sced["status"] != "optimal":
                n_fail += 1
                continue

            sp_records.append({"hour": hour, **sced["sp_lmp"]})
            dispatch_records.append({"hour": hour, **sced["dispatch"]})
            flows_records.append({"hour": hour, **sced["line_flows"]})
            duals_records.append({"hour": hour, **sced["line_duals"]})
            gtc_records.append({"hour": hour, **sced["gtc_duals"]})
            prev = sced["dispatch"]

        state = scuc["end_state"]
        daily_stats.append({"date": day.date(), "scuc_cost": scuc["total_cost"],
                            "n_sced_fail": n_fail})

        if verbose and (day_idx + 1) % 3 == 0:
            elapsed = time.time() - t0
            print(f"  [{day_idx+1}/{n_days}] {day.date()} "
                  f"ETA={elapsed/(day_idx+1)*(n_days-day_idx-1)/60:.1f}m")

    result = SimulationResult(
        sp_lmp=pd.DataFrame(sp_records).set_index("hour") if sp_records else pd.DataFrame(),
        dispatch=pd.DataFrame(dispatch_records).set_index("hour") if dispatch_records else pd.DataFrame(),
        line_flows=pd.DataFrame(flows_records).set_index("hour") if flows_records else pd.DataFrame(),
        line_duals=pd.DataFrame(duals_records).set_index("hour") if duals_records else pd.DataFrame(),
        gtc_duals=pd.DataFrame(gtc_records).set_index("hour") if gtc_records else pd.DataFrame(),
        daily_stats=daily_stats,
        metadata={"fast_mode": fast_mode, "n_days": n_days,
                  "n_hours": len(sp_records), "time_sec": time.time()-t0},
    )

    total = time.time() - t0
    if verbose:
        print(f"DONE. {total/60:.1f} min, {len(sp_records)} hours")
        if not result.sp_lmp.empty and "HB_BUSAVG" in result.sp_lmp.columns:
            print(f"  Avg system LMP: ${result.sp_lmp['HB_BUSAVG'].mean():.2f}/MWh")

    if save_path:
        result.save(save_path)
        if verbose:
            print(f"  Saved: {save_path}")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("=" * 60)
    print("  CRR Alpha Lab v3 — ERCOT 17-Bus DAM Simulation")
    print("=" * 60)
    result = run_dam_simulation(fast_mode=True, verbose=True,
                                save_path="results/ercot_17bus/dam_simulation.pkl")
