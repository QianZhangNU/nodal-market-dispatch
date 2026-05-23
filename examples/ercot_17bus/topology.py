"""
ERCOT-Flavored 17-Bus Test System
================================
This file defines a small ERCOT-style test system for studying nodal price
formation, congestion, CRR path exposure, and dispatch behavior. It is not a
full ERCOT network model. Instead, it is a deliberately compact 17-bus system
with enough structure to reproduce several important market mechanisms in a
transparent and inspectable way.

Design goals
------------
1. Keep the model small enough for fast SCUC/SCED and hand inspection.
   The system uses representative buses for major ERCOT areas.

2. Preserve nodal behavior.
   Hubs, load zones, resource nodes, local load pockets, and constrained
   interfaces are modeled separately so the example can produce hub-zone basis,
   congestion components, and source-sink CRR exposure.

3. Make congestion drivers explicit.
   Constraints are named and shaped so users can trace price separation back to
   a physical modeling assumption: a line limit, a GTC, an outage, a local load
   pocket, or a generation pocket.

Main modeled congestion structures
----------------------------------
West export congestion:
  GTC_WEST_EXPORT limits total export from West/Panhandle toward the rest of
  ERCOT. This represents broad West-area export congestion. It affects West
  prices versus the rest of the system, but it is not the direct driver of
  HB_WEST versus LZ_WEST separation.

West intrazonal basis:
  HB_WEST is represented by gen-side buses 1 and 2, while LZ_WEST is represented
  by the West load node at bus 13. The two parallel West intrazonal lines from
  bus 2 to bus 13 create the local interface behind which LZ_WEST can separate
  from HB_WEST. The expensive West peaker at bus 13 provides a local marginal
  resource when that interface is tight.

Houston import congestion:
  GTC_HOUSTON_IMPORT and individual Houston import lines model load-pocket
  congestion into the Houston area. This creates price separation when Houston
  load is high or import paths are limited.

South Texas/RGV export congestion:
  The RGV solar pocket, Laredo solar node, and GTC_SOUTH_TEXAS_EXPORT create a
  generation-pocket structure similar to real South Texas export constraints.
  This lets the example produce solar-driven congestion and curtailment risk.

Storage behavior:
  Four battery units are represented as exogenous net-injection profiles rather
  than co-optimized SCUC/SCED resources. They charge during midday solar-heavy
  hours and discharge during evening peak hours, allowing the example to show
  how storage can reshape net load and congestion patterns.

Monthly topology variation:
  get_monthly_topology() applies seasonal ratings, line outages, generator
  outages, and GTC limit adjustments. This supports scenario analysis for CRR
  and congestion studies without changing the base topology definitions.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  Time horizon
# ─────────────────────────────────────────────────────────────────────────────
SIM_START = pd.Timestamp("2023-01-01")
SIM_END   = pd.Timestamp("2024-12-31 23:00")
HOURS = pd.date_range(SIM_START, SIM_END, freq="h")
N_HOURS = len(HOURS)
BASE_MVA = 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  Bus Definitions  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
BUSES = {
    1:  dict(name="PANHANDLE_345",     region="Panhandle",   kV=345, lat=35.20, lng=-101.83),
    2:  dict(name="WEST_GEN_345",      region="West",        kV=345, lat=31.84, lng=-102.37),  # generator/hub node
    13: dict(name="WEST_LOAD_138",     region="West",        kV=138, lat=31.50, lng=-102.90),  # load node (Odessa/Midland)
    3:  dict(name="NORTH_345",         region="North",       kV=345, lat=32.78, lng=-96.80),
    4:  dict(name="NORTH_CENTRAL_345", region="NorthCentral",kV=345, lat=32.34, lng=-95.30),
    5:  dict(name="HOUSTON_NORTH_345", region="Houston",     kV=345, lat=30.21, lng=-95.62),
    6:  dict(name="HOUSTON_LOAD_138",  region="Houston",     kV=138, lat=29.76, lng=-95.37),
    7:  dict(name="HOUSTON_GEN_138",   region="Houston",     kV=138, lat=29.50, lng=-95.05),
    8:  dict(name="SOUTH_345",         region="South",       kV=345, lat=27.80, lng=-97.40),
    9:  dict(name="SOUTHERN_138",      region="Southern",    kV=138, lat=26.20, lng=-98.10),
    10: dict(name="AUSTIN_138",        region="South",       kV=138, lat=30.27, lng=-97.74),
    11: dict(name="WACO_138",          region="NorthCentral",kV=138, lat=31.55, lng=-97.15),
    12: dict(name="VICTORIA_138",      region="South",       kV=138, lat=28.81, lng=-96.99),

    # ── NEW: South Texas RGV solar pocket (mirrors NELRIO + NE_LOB GTC region)
    14: dict(name="RGV_SOLAR_HUB_138", region="South_RGV",   kV=138, lat=26.40, lng=-98.50),  # Magic Valley solar集群
    15: dict(name="RGV_LOAD_HARLNG",   region="South_RGV",   kV=138, lat=26.19, lng=-97.70),  # Harlingen 负荷
    16: dict(name="RGV_LOAD_BRWNSV",   region="South_RGV",   kV=138, lat=25.90, lng=-97.50),  # Brownsville 负荷
    17: dict(name="LAREDO_SOLAR_138",  region="South_RGV",   kV=138, lat=27.51, lng=-99.51),  # Laredo solar
}
N_BUS = len(BUSES)
BUS_LIST = list(BUSES.keys())
BUS_IDX  = {b: i for i, b in enumerate(BUS_LIST)}
SLACK_BUS = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Generator Fleet  +  Battery Storage  (v2: added 4 batteries)
# ─────────────────────────────────────────────────────────────────────────────
# Standard generators (same as v1)
GENERATORS = {
    # ── Wind (4 units, 1.8 GW) ────────────────────────────────────────────────
    "WND_PANHNDL_001": dict(
        bus=1, type="WIND", region="Panhandle",
        Pmin=0, Pmax=600, cost_b=0.5, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=600, ramp_down=600,
        init_status=1, init_gen=200, init_up_time=24, init_down_time=0,
    ),
    "WND_PANHNDL_002": dict(
        bus=1, type="WIND", region="Panhandle",
        Pmin=0, Pmax=400, cost_b=0.5, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=400, ramp_down=400,
        init_status=1, init_gen=150, init_up_time=24, init_down_time=0,
    ),
    "WND_WEST_001": dict(
        bus=2, type="WIND", region="West",
        Pmin=0, Pmax=900, cost_b=0.5, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=900, ramp_down=900,
        init_status=1, init_gen=300, init_up_time=24, init_down_time=0,
    ),
    "WND_SOUTH_001": dict(
        bus=8, type="WIND", region="South",
        Pmin=0, Pmax=300, cost_b=0.5, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=300, ramp_down=300,
        init_status=1, init_gen=100, init_up_time=24, init_down_time=0,
    ),

    # ── Solar (3 units, 900 MW) ───────────────────────────────────────────────
    "SOL_WEST_001": dict(
        bus=2, type="SOLAR", region="West",
        Pmin=0, Pmax=400, cost_b=0.0, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=400, ramp_down=400,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=12,
    ),
    # ── South Texas RGV Solar Cluster (3 units, 2.0 GW total) ───────────────
    # Mirrors real ERCOT Magic Valley solar concentration that drives
    # NELRIO + NE_LOB GTC binding. With 2 GW vs 3 GW system peak load,
    # creates classic generation-pocket congestion when sun is up.
    "SOL_RGV_001": dict(
        bus=14, type="SOLAR", region="South_RGV",
        Pmin=0, Pmax=900, cost_b=0.0, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=900, ramp_down=900,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=12,
    ),
    "SOL_RGV_002": dict(
        bus=14, type="SOLAR", region="South_RGV",
        Pmin=0, Pmax=600, cost_b=0.0, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=600, ramp_down=600,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=12,
    ),
    "SOL_LAREDO_001": dict(
        bus=17, type="SOLAR", region="South_RGV",
        Pmin=0, Pmax=500, cost_b=0.0, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=500, ramp_down=500,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=12,
    ),
    "SOL_AUSTIN_001": dict(
        bus=10, type="SOLAR", region="South",
        Pmin=0, Pmax=200, cost_b=0.0, cost_c=0,
        su_cost=0, sd_cost=0, min_up=1, min_down=1,
        ramp_up=200, ramp_down=200,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=12,
    ),

    # ── Combined Cycle Gas (5 units, 4.55 GW) ─────────────────────────────────
    "CC_HOUSTON_001": dict(
        bus=7, type="COMBINED_CYCLE", region="Houston",
        Pmin=200, Pmax=1100, cost_b=32.0, cost_c=400,
        su_cost=8000, sd_cost=2000, min_up=6, min_down=4,
        ramp_up=400, ramp_down=400,
        init_status=1, init_gen=600, init_up_time=24, init_down_time=0,
    ),
    "CC_HOUSTON_002": dict(
        bus=7, type="COMBINED_CYCLE", region="Houston",
        Pmin=150, Pmax=800, cost_b=34.0, cost_c=350,
        su_cost=6000, sd_cost=1500, min_up=6, min_down=4,
        ramp_up=320, ramp_down=320,
        init_status=1, init_gen=400, init_up_time=12, init_down_time=0,
    ),
    "CC_SOUTH_001": dict(
        bus=8, type="COMBINED_CYCLE", region="South",
        Pmin=120, Pmax=700, cost_b=35.0, cost_c=300,
        su_cost=5000, sd_cost=1200, min_up=6, min_down=4,
        ramp_up=280, ramp_down=280,
        init_status=1, init_gen=350, init_up_time=12, init_down_time=0,
    ),
    "CC_AUSTIN_001": dict(
        bus=10, type="COMBINED_CYCLE", region="South",
        Pmin=100, Pmax=550, cost_b=36.0, cost_c=250,
        su_cost=4000, sd_cost=1000, min_up=4, min_down=3,
        ramp_up=220, ramp_down=220,
        # init_status=0: keeps sum(Pmin of committed units)=2220 MW < 2253 MW
        # minimum system load (Jan 27), preventing overgeneration infeasibility
        # that would force nuclear offline on day 1.
        init_status=0, init_gen=0, init_up_time=0, init_down_time=4,
    ),
    "CC_NORTH_001": dict(
        bus=3, type="COMBINED_CYCLE", region="North",
        Pmin=150, Pmax=900, cost_b=33.0, cost_c=380,
        su_cost=7000, sd_cost=1800, min_up=6, min_down=4,
        ramp_up=350, ramp_down=350,
        init_status=1, init_gen=500, init_up_time=24, init_down_time=0,
    ),

    # ── Gas Turbine peakers (3 units, 550 MW) ─────────────────────────────────
    "GT_HOUSTON_001": dict(
        bus=6, type="GAS_TURBINE", region="Houston",
        Pmin=30, Pmax=200, cost_b=80.0, cost_c=200,
        su_cost=3000, sd_cost=500, min_up=2, min_down=2,
        ramp_up=200, ramp_down=200,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=8,
    ),
    "GT_NORTH_001": dict(
        bus=3, type="GAS_TURBINE", region="North",
        Pmin=30, Pmax=200, cost_b=85.0, cost_c=200,
        su_cost=3000, sd_cost=500, min_up=2, min_down=2,
        ramp_up=200, ramp_down=200,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=8,
    ),
    "GT_AUSTIN_001": dict(
        bus=10, type="GAS_TURBINE", region="South",
        Pmin=20, Pmax=150, cost_b=90.0, cost_c=150,
        su_cost=2000, sd_cost=400, min_up=2, min_down=2,
        ramp_up=150, ramp_down=150,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=8,
    ),
    # West Texas local peaker at bus 13 (load node).
    # Expensive small unit. When intrazonal lines bind and West cannot import
    # enough from bus 2, this unit must start, naturally pushing LMP at bus 13
    # ABOVE bus 2. This produces stable LZ_WEST > HB_WEST basis.
    "GT_WEST_PEAKER": dict(
        bus=13, type="GAS_TURBINE", region="West",
        Pmin=10, Pmax=120, cost_b=110.0, cost_c=80,
        su_cost=1500, sd_cost=300, min_up=2, min_down=2,
        ramp_up=120, ramp_down=120,
        init_status=0, init_gen=0, init_up_time=0, init_down_time=8,
    ),

    # ── Coal (2 units, 1.25 GW) ───────────────────────────────────────────────
    "COAL_NCNTRL_001": dict(
        bus=4, type="COAL", region="NorthCentral",
        Pmin=300, Pmax=750, cost_b=28.0, cost_c=800,
        su_cost=20000, sd_cost=5000, min_up=24, min_down=24,
        ramp_up=120, ramp_down=120,
        init_status=1, init_gen=550, init_up_time=72, init_down_time=0,
    ),
    "COAL_NCNTRL_002": dict(
        bus=4, type="COAL", region="NorthCentral",
        Pmin=200, Pmax=500, cost_b=29.0, cost_c=600,
        su_cost=15000, sd_cost=4000, min_up=24, min_down=24,
        ramp_up=80, ramp_down=80,
        init_status=1, init_gen=400, init_up_time=72, init_down_time=0,
    ),

    # ── Nuclear (1 unit, 1.2 GW) ──────────────────────────────────────────────
    "NUC_SOUTH_001": dict(
        bus=8, type="NUCLEAR", region="South",
        Pmin=1100, Pmax=1200, cost_b=8.0, cost_c=2000,
        su_cost=100000, sd_cost=50000,
        # min_up=17520: once committed, nuclear cannot decommit for 2 full years.
        # This covers the entire 2023-2024 simulation span. Using 8760 (1 year)
        # caused the must-on constraint to expire mid-simulation (up_time exceeds
        # min_up after accumulated elapsed hours), allowing economic decommit.
        # Plants cycle only for planned refueling (modeled via planned outage in
        # get_monthly_topology). After an outage the restart is handled in run_dam.
        min_up=17520, min_down=168,
        ramp_up=50, ramp_down=50,
        init_status=1, init_gen=1180, init_up_time=720, init_down_time=0,
    ),
}

GEN_LIST = list(GENERATORS.keys())
N_GEN = len(GEN_LIST)


# ─────────────────────────────────────────────────────────────────────────────
#  *** v2 ADDITION: Battery Storage ***
# ─────────────────────────────────────────────────────────────────────────────
# Battery is modeled as a NET INJECTION profile (not a SCUC variable) for
# simplicity. This reflects how batteries actually operate in 2025 ERCOT:
# they're price-takers that arbitrage based on forecasted prices, so
# dispatch is largely deterministic given LMP forecasts.
#
# Each battery has:
#   bus           : connection bus
#   power_mw      : max charge / discharge rate
#   energy_mwh    : storage capacity (typically 4-hour duration)
#   region        : geographic placement
#
# Operating profile (built in profiles.py):
#   HE 11-15: charge at full power (negative injection = act as load)
#   HE 17-20: discharge at full power (positive injection = act as gen)
#   Other hours: idle
BATTERIES = {
    "BAT_HOUSTON_001": dict(
        bus=6, region="Houston",
        power_mw=200, energy_mwh=800,
        description="Houston load center BESS — peak shave",
    ),
    "BAT_NORTH_001": dict(
        bus=3, region="North",
        power_mw=150, energy_mwh=600,
        description="DFW BESS — frequency + peak shave",
    ),
    "BAT_WEST_001": dict(
        bus=13, region="West",
        power_mw=150, energy_mwh=600,
        description="West Texas BESS at load node — solar firming + arbitrage",
    ),
    "BAT_SOUTH_001": dict(
        bus=10, region="South",
        power_mw=100, energy_mwh=400,
        description="South Texas BESS — net load flattening",
    ),
}
BATTERY_LIST = list(BATTERIES.keys())
N_BATTERY = len(BATTERY_LIST)


# ─────────────────────────────────────────────────────────────────────────────
#  Transmission Lines (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
LINES = {
    "L_PANHNDL_NORTH_1": dict(
        from_bus=1, to_bus=3, kV=345, x_pu=0.025, b_pu=40.0,
        flow_limit=600, contingency_limit=480,
        ocost=200,
        description="Panhandle wind export to DFW (CREZ-equivalent)",
    ),
    "L_PANHNDL_NORTH_2": dict(
        from_bus=1, to_bus=3, kV=345, x_pu=0.025, b_pu=40.0,
        flow_limit=600, contingency_limit=480,
        ocost=200,
        description="Panhandle wind export, parallel circuit",
    ),
    # Two parallel West intrazonal lines with UNEQUAL impedances + OCOST.
    # These lines, not the West Export GTC, are the direct driver of
    # HB_WEST vs LZ_WEST separation: HB_WEST averages gen-side buses 1/2,
    # while LZ_WEST is the load node at bus 13 behind this local interface.
    # OCOST allows controlled overload (caps mu at OCOST) so SCED produces
    # bounded LMP differentials reflecting realistic intrazonal premium.
    # SCUC's slack mechanism handles infeasibility separately.
    "L_WEST_INTRAZONAL_1": dict(
        from_bus=2, to_bus=13, kV=138, x_pu=0.080, b_pu=12.5,
        flow_limit=300, contingency_limit=240,
        ocost=15,    # When binds, mu capped at $15/MW -> realistic LZ_WEST premium
        description="West intrazonal main (lower x). OCOST=$15 caps mu at realistic level.",
    ),
    "L_WEST_INTRAZONAL_2": dict(
        from_bus=2, to_bus=13, kV=138, x_pu=0.180, b_pu=5.6,
        flow_limit=250, contingency_limit=200,
        ocost=15,
        description="West intrazonal parallel (higher x).",
    ),
    "L_WEST_NORTH": dict(
        from_bus=2, to_bus=3, kV=345, x_pu=0.030, b_pu=33.3,
        flow_limit=900, contingency_limit=700,
        ocost=250,
        description="West Texas to DFW",
    ),
    "L_WEST_SOUTH": dict(
        from_bus=2, to_bus=8, kV=345, x_pu=0.045, b_pu=22.2,
        flow_limit=600, contingency_limit=450,
        ocost=200,
        description="Permian to South Texas",
    ),
    "L_NORTH_NCNTRL": dict(
        from_bus=3, to_bus=4, kV=345, x_pu=0.020, b_pu=50.0,
        flow_limit=1200, contingency_limit=950,
        ocost=150,
        description="DFW to coal plants",
    ),
    "L_NCNTRL_HOUSTON": dict(
        from_bus=4, to_bus=5, kV=345, x_pu=0.035, b_pu=28.6,
        flow_limit=700, contingency_limit=560,
        ocost=250,
        description="East Texas → Houston North (tightened: load pocket import)",
    ),
    "L_NORTH_HOUSTON_1": dict(
        from_bus=3, to_bus=5, kV=345, x_pu=0.035, b_pu=28.6,
        flow_limit=600, contingency_limit=480,
        ocost=250,
        description="DFW→Houston main 345kV path (lower impedance, takes more flow)",
    ),
    "L_NORTH_HOUSTON_2": dict(
        from_bus=3, to_bus=5, kV=345, x_pu=0.055, b_pu=18.2,
        flow_limit=600, contingency_limit=480,
        ocost=250,
        description="DFW→Houston parallel 345kV (higher impedance, takes less flow). "
                    "Different impedances create unequal flow split — main line binds first, "
                    "parallel line cannot fully substitute → stable LMP basis emerges.",
    ),
    "L_HOUNORTH_HOULOAD": dict(
        from_bus=5, to_bus=6, kV=345, x_pu=0.015, b_pu=66.7,
        flow_limit=1100, contingency_limit=900,
        ocost=500,
        description="Houston import to load center (tightened: binds during peak)",
    ),
    "L_SOUTH_HOUSTON": dict(
        from_bus=8, to_bus=5, kV=345, x_pu=0.038, b_pu=26.3,
        flow_limit=600, contingency_limit=480,
        ocost=250,
        description="South Texas → Houston (alternate path, tightened)",
    ),
    "L_SOUTH_SOUTHERN": dict(
        from_bus=8, to_bus=9, kV=345, x_pu=0.050, b_pu=20.0,
        flow_limit=500, contingency_limit=380,
        ocost=100,
        description="Corpus to RGV/border",
    ),
    "L_HOULOAD_HOUGEN": dict(
        from_bus=6, to_bus=7, kV=138, x_pu=0.060, b_pu=16.7,
        flow_limit=600, contingency_limit=450,
        ocost=150,
        description="Houston load to coastal gen",
    ),
    "L_NCNTRL_WACO": dict(
        from_bus=4, to_bus=11, kV=138, x_pu=0.080, b_pu=12.5,
        flow_limit=400, contingency_limit=300,
        ocost=100,
        description="Coal to Waco",
    ),
    "L_WACO_AUSTIN": dict(
        from_bus=11, to_bus=10, kV=138, x_pu=0.090, b_pu=11.1,
        flow_limit=350, contingency_limit=270,
        ocost=50,
        description="Waco to Austin",
    ),
    "L_AUSTIN_VICTORIA": dict(
        from_bus=10, to_bus=12, kV=138, x_pu=0.100, b_pu=10.0,
        flow_limit=350, contingency_limit=270,
        ocost=50,
        description="Austin to South interconnect",
    ),
    "L_VICTORIA_HOULOAD": dict(
        from_bus=12, to_bus=6, kV=138, x_pu=0.080, b_pu=12.5,
        flow_limit=400, contingency_limit=300,
        ocost=200,
        description="Victoria to Houston load",
    ),
    "L_VICTORIA_SOUTH": dict(
        from_bus=12, to_bus=8, kV=138, x_pu=0.075, b_pu=13.3,
        flow_limit=450, contingency_limit=350,
        ocost=100,
        description="Victoria to Corpus",
    ),

    # ── South Texas RGV solar pocket lines ──────────────────────────────────
    # Export bottlenecks: RGV→SOUTH_345 main outflow paths
    "L_RGV_TO_SOUTH": dict(
        from_bus=14, to_bus=8, kV=345, x_pu=0.040, b_pu=25.0,
        flow_limit=900, contingency_limit=720,
        ocost=200,
        description="RGV solar hub → SOUTH_345 main export (NELRIO equivalent)",
    ),
    "L_RGV_TO_LAREDO": dict(
        from_bus=14, to_bus=17, kV=138, x_pu=0.060, b_pu=16.7,
        flow_limit=500, contingency_limit=400,
        ocost=30,
        description="RGV → Laredo cross-tie",
    ),
    "L_LAREDO_TO_SOUTH": dict(
        from_bus=17, to_bus=8, kV=345, x_pu=0.045, b_pu=22.2,
        flow_limit=700, contingency_limit=560,
        ocost=200,
        description="Laredo solar → SOUTH_345 export (NE_LOB equivalent)",
    ),
    # Internal RGV mesh: solar hub → load nodes (multiple parallel paths)
    "L_RGV_TO_HARLNG": dict(
        from_bus=14, to_bus=15, kV=138, x_pu=0.080, b_pu=12.5,
        flow_limit=350, contingency_limit=280,
        ocost=30,
        description="RGV solar hub → Harlingen load (intrazonal)",
    ),
    "L_RGV_TO_BRWNSV": dict(
        from_bus=14, to_bus=16, kV=138, x_pu=0.100, b_pu=10.0,
        flow_limit=300, contingency_limit=240,
        ocost=30,
        description="RGV solar hub → Brownsville load (intrazonal)",
    ),
    "L_HARLNG_TO_BRWNSV": dict(
        from_bus=15, to_bus=16, kV=138, x_pu=0.150, b_pu=6.7,
        flow_limit=200, contingency_limit=160,
        ocost=30,
        description="Harlingen ↔ Brownsville cross-tie (creates mesh)",
    ),
    # Connect old SOUTHERN_138 (bus 9) into the mesh as well
    "L_SOUTHERN_HARLNG": dict(
        from_bus=9, to_bus=15, kV=138, x_pu=0.090, b_pu=11.1,
        flow_limit=300, contingency_limit=240,
        ocost=30,
        description="SOUTHERN → Harlingen (RGV mesh)",
    ),
}

LINE_LIST = list(LINES.keys())
N_LINE = len(LINE_LIST)


# ─────────────────────────────────────────────────────────────────────────────
#  *** v2 ADDITION: Generic Transmission Constraints (GTC) ***
# ─────────────────────────────────────────────────────────────────────────────
# A GTC limits the SUM of flows on multiple monitored lines, not any single
# line individually. This is what creates persistent multi-month congestion
# in real ERCOT — e.g., WEST TEXAS EXPORT GTC was binding 9 of 11 months
# in 2025 ($86M congestion rent).
#
# GTC effect: "Σ (sign * line_flow) ≤ gtc_limit"
#
# In LP formulation this becomes a single linear constraint with shadow price
# that contributes to LMP at every bus via:
#    LMP_GTC_component[bus] = μ_GTC * Σ_l (sign_l * PTDF[l, bus])
#
# The "sign" on each line indicates direction:
#   +1 = flow from-→to direction counts toward the export
#   -1 = flow from-→to direction counts AGAINST the export (opposite direction)
GENERIC_TRANSMISSION_CONSTRAINTS = {
    "GTC_WEST_EXPORT": dict(
        monitored_lines=[
            ("L_WEST_NORTH",        +1),
            ("L_WEST_SOUTH",        +1),
            ("L_PANHNDL_NORTH_1",   +1),
            ("L_PANHNDL_NORTH_2",   +1),
        ],
        flow_limit=900,             # tighter — binds ~30% of hours, mirrors real WESTEX
        contingency_limit=720,
        description=(
            "WEST_EXPORT_GTC — limits TOTAL simultaneous power export from "
            "West/Panhandle to the rest of ERCOT. Mirrors real WEST TEXAS "
            "EXPORT GTC (NPRR1146)."
        ),
    ),
    "GTC_HOUSTON_IMPORT": dict(
        monitored_lines=[
            ("L_NORTH_HOUSTON_1",   +1),
            ("L_NORTH_HOUSTON_2",   +1),
            ("L_NCNTRL_HOUSTON",    +1),
            ("L_SOUTH_HOUSTON",     +1),
            ("L_VICTORIA_HOULOAD",  +1),
        ],
        flow_limit=1700,            # tightened from 2200 — binds during peak load
        contingency_limit=1300,
        description=(
            "HOUSTON_IMPORT_GTC — limits TOTAL simultaneous power import "
            "into Houston load center."
        ),
    ),
    # ── NEW: South Texas Export GTC (mirrors NELRIO + NE_LOB) ───────────────
    "GTC_SOUTH_TEXAS_EXPORT": dict(
        monitored_lines=[
            ("L_RGV_TO_SOUTH",      +1),  # main RGV outflow
            ("L_LAREDO_TO_SOUTH",   +1),  # Laredo outflow
        ],
        # Total RGV solar capacity: 2.0 GW + nuclear share + RGV local load 0.7 GW
        # Net export when sun is up: 2.0 - 0.7 = 1.3 GW typical, peak 1.8 GW
        # GTC at 1100 MW binds during HE 11-15 of sunny days
        flow_limit=1100,
        contingency_limit=880,
        description=(
            "SOUTH_TEXAS_EXPORT_GTC — limits TOTAL simultaneous power export "
            "from RGV solar pocket to rest of ERCOT. Mirrors real "
            "NELSON_SHARPE_RIO_HONDO_GTC and NORTH_EDINBURG_LOBO_GTC (Top 3 "
            "and Top 7 RT congestion constraints in 2025)."
        ),
    ),
}
GTC_LIST = list(GENERIC_TRANSMISSION_CONSTRAINTS.keys())
N_GTC = len(GTC_LIST)


# ─────────────────────────────────────────────────────────────────────────────
#  Settlement Points (v2: added LZ_HOUSTON_138 split for more granular CRRs)
# ─────────────────────────────────────────────────────────────────────────────
SETTLEMENT_POINTS = {
    # Resource Nodes
    "RN_PANHNDL":      dict(type="resource_node", bus=1,  description="Panhandle wind hub"),
    "RN_WEST":         dict(type="resource_node", bus=2,  description="West Texas wind+solar (gen-side)"),
    "RN_NORTH_GAS":    dict(type="resource_node", bus=3,  description="North CC + GT"),
    "RN_NCNTRL_COAL":  dict(type="resource_node", bus=4,  description="North Central coal"),
    "RN_HOU_GEN":      dict(type="resource_node", bus=7,  description="Houston coastal CC"),
    "RN_HOU_PEAK":     dict(type="resource_node", bus=6,  description="Houston peakers"),
    "RN_SOUTH_GEN":    dict(type="resource_node", bus=8,  description="South CC + nuclear"),
    "RN_AUSTIN_GEN":   dict(type="resource_node", bus=10, description="Austin gas (Sand Hill)"),
    "RN_RGV_SOLAR":    dict(type="resource_node", bus=14, description="RGV solar hub (Magic Valley)"),
    "RN_LAREDO_SOLAR": dict(type="resource_node", bus=17, description="Laredo solar"),

    # Load Zones — STRUCTURE 5: weighted average over multiple load buses
    # Real ERCOT LZs aggregate hundreds of load nodes; we use 2-3 each
    # to produce stable LZ vs HB basis from intrazonal congestion.
    "LZ_NORTH":   dict(type="load_zone_weighted",
                       member_buses={3: 0.85, 4: 0.15},
                       description="DFW + East TX weighted load zone"),
    "LZ_HOUSTON": dict(type="load_zone_weighted",
                       member_buses={6: 0.80, 5: 0.10, 12: 0.10},
                       description="Houston load + Victoria weighted"),
    "LZ_SOUTH":   dict(type="load_zone_weighted",
                       member_buses={10: 0.45, 15: 0.20, 16: 0.20, 9: 0.15},
                       description="Austin/SAT + RGV (Harlingen/Brownsville) weighted"),
    "LZ_WEST":    dict(type="load_zone_weighted",
                       member_buses={13: 1.00},
                       description="West Texas (Odessa/Midland) — single load node"),

    # Hubs — gen-side nodes only (this is what makes LZ vs HB different)
    "HB_NORTH":        dict(type="hub", bus=None, member_buses=[3, 4],
                            description="ERCOT North Hub (gen-side 345kV)"),
    "HB_HOUSTON":      dict(type="hub", bus=None, member_buses=[5, 7],
                            description="ERCOT Houston Hub (gen-side, excludes load bus 6)"),
    "HB_SOUTH":        dict(type="hub", bus=None, member_buses=[8, 14, 17],
                            description="ERCOT South Hub (gen-side: SOUTH + RGV solar + Laredo)"),
    "HB_WEST":         dict(type="hub", bus=None, member_buses=[1, 2],
                            description="ERCOT West Hub (gen-side 345kV: Panhandle + West)"),
    "HB_BUSAVG":       dict(type="hub", bus=None,
                            member_buses=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                            description="ERCOT-wide hub average"),
}
SP_LIST = list(SETTLEMENT_POINTS.keys())
N_SP = len(SP_LIST)


# ─────────────────────────────────────────────────────────────────────────────
#  Contingency List
# ─────────────────────────────────────────────────────────────────────────────
CONTINGENCIES = {
    "CTG_PANHNDL_1":   dict(outaged_line="L_PANHNDL_NORTH_1",
                             description="Loss of one Panhandle export circuit"),
    "CTG_NORTH_HOU_1": dict(outaged_line="L_NORTH_HOUSTON_1",
                             description="Loss of one DFW→Houston circuit"),
    "CTG_NCNTRL_HOU":  dict(outaged_line="L_NCNTRL_HOUSTON",
                             description="Loss of East TX→Houston path"),
    "CTG_WEST_NORTH":  dict(outaged_line="L_WEST_NORTH",
                             description="Loss of West TX→DFW backbone"),
    "CTG_NUC_OUTAGE":  dict(outaged_gen="NUC_SOUTH_001",
                             description="Nuclear unit refueling outage"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Load profile parameters
# ─────────────────────────────────────────────────────────────────────────────
ZONE_PEAK_LOAD = {
    "LZ_NORTH":   1600,
    "LZ_HOUSTON": 1800,
    "LZ_SOUTH":   1100,
    "LZ_WEST":     350,
}

BUS_LOAD_FRACTION = {
    # West region (load pocket on bus 13) — increased to push intrazonal binding
    1: 0.01, 2: 0.00, 13: 0.12,
    # North region
    3: 0.23, 4: 0.05,
    # Houston region (load pocket on bus 6)
    5: 0.06, 6: 0.23, 7: 0.02, 12: 0.04,
    # South region (Austin/SAT)
    10: 0.12, 11: 0.04,
    # South region (RGV — Magic Valley load pocket)
    8: 0.03, 9: 0.02, 15: 0.06, 16: 0.06,
    # Solar/gen-only buses
    14: 0.00, 17: 0.00,
}
_total_frac = sum(BUS_LOAD_FRACTION.values())
BUS_LOAD_FRACTION = {b: f/_total_frac for b, f in BUS_LOAD_FRACTION.items()}

ERCOT_TOTAL_PEAK = sum(ZONE_PEAK_LOAD.values()) + 700


# ─────────────────────────────────────────────────────────────────────────────
#  Seasonal Topology Variation (CRR-style model)
# ─────────────────────────────────────────────────────────────────────────────
# In real ERCOT, the CRR Network Model changes month-to-month:
#   - Line ratings differ by season (summer lower, winter higher)
#   - Planned outages remove lines/generators for specific months
#   - GTC limits may be adjusted seasonally
#   - Contingency list updates when topology changes
#
# This module transforms the static SSWG-style topology into a
# time-varying CRR-style model by applying monthly overlays.

import copy

# ── Seasonal line rating multipliers ─────────────────────────────────────────
# Summer: higher ambient temperature → lower thermal rating
# Winter: lower temperature → higher thermal rating
# Shoulder: transition months
SEASONAL_RATING_FACTORS = {
    # month: multiplier applied to flow_limit
    1: 1.20,   # Jan: cold, high rating
    2: 1.15,   # Feb
    3: 1.05,   # Mar: shoulder
    4: 1.00,   # Apr: nominal
    5: 0.95,   # May: warming
    6: 0.85,   # Jun: hot → tight
    7: 0.80,   # Jul: peak heat → tightest
    8: 0.82,   # Aug: still very hot
    9: 0.90,   # Sep: cooling
    10: 1.00,  # Oct: nominal
    11: 1.10,  # Nov: cooling
    12: 1.20,  # Dec: cold, high rating
}

# ── Planned outages ──────────────────────────────────────────────────────────
# Each outage removes a line or generator for a date range.
# Status: APPROVED → enters CRR model; SUBMITTED → does not.
PLANNED_OUTAGES = [
    dict(
        element_type="line",
        element_id="L_NORTH_HOUSTON_1",
        start_month=3, start_year=2023,
        end_month=3,   end_year=2023,
        status="APPROVED",
        reason="Transformer replacement on DFW→Houston 345kV #1",
    ),
    dict(
        element_type="line",
        element_id="L_PANHNDL_NORTH_1",
        start_month=10, start_year=2023,
        end_month=10,   end_year=2023,
        status="APPROVED",
        reason="CREZ line upgrade",
    ),
    dict(
        element_type="generator",
        element_id="NUC_SOUTH_001",
        start_month=4, start_year=2024,
        end_month=5,   end_year=2024,
        status="SCHEDULED",
        reason="Nuclear refueling outage (18-month cycle)",
    ),
    dict(
        element_type="line",
        element_id="L_WEST_INTRAZONAL_1",
        start_month=9, start_year=2023,
        end_month=9,   end_year=2023,
        status="APPROVED",
        reason="138kV breaker maintenance",
    ),
    dict(
        element_type="line",
        element_id="L_RGV_TO_SOUTH",
        start_month=11, start_year=2024,
        end_month=11,   end_year=2024,
        status="SUBMITTED",    # NOT approved → does NOT enter CRR model
        reason="Proposed 345kV reconductoring",
    ),
]

# ── Seasonal GTC limit adjustments ───────────────────────────────────────────
# Some GTCs have seasonally varying limits (e.g., WESTEX tighter in spring)
SEASONAL_GTC_FACTORS = {
    "GTC_WEST_EXPORT": {
        3: 0.85, 4: 0.85, 5: 0.90,   # Spring: tighter (high wind, low load)
        7: 1.10, 8: 1.10,             # Summer: slightly relaxed (high load pulls)
    },
    "GTC_SOUTH_TEXAS_EXPORT": {
        3: 0.80, 4: 0.80, 5: 0.85,   # Spring: very tight (solar + wind both high)
        6: 0.90,
    },
}


def get_monthly_topology(year: int, month: int) -> dict:
    """
    Generate a month-specific topology overlay (CRR-style model).

    Returns a dict with:
      - lines: dict of line_id → modified line params (with seasonal ratings)
      - generators: dict of gen_id → modified gen params
      - gtcs: dict of gtc_id → modified GTC params
      - outaged_lines: list of line_ids removed this month
      - outaged_generators: list of gen_ids removed this month
      - contingencies: updated contingency dict (minus outaged elements)
      - rating_factor: the seasonal multiplier applied

    The caller (dam_simulator_v2) uses this to modify PTDF, line limits,
    and generator availability for each simulated month.
    """
    rating_factor = SEASONAL_RATING_FACTORS.get(month, 1.0)

    # ── Apply seasonal line ratings ───────────────────────────────────────
    lines_monthly = {}
    for lid, ldata in LINES.items():
        lmod = copy.deepcopy(ldata)
        lmod["flow_limit"] = int(ldata["flow_limit"] * rating_factor)
        lmod["contingency_limit"] = int(ldata["contingency_limit"] * rating_factor)
        lines_monthly[lid] = lmod

    # ── Apply planned outages (APPROVED/SCHEDULED only) ───────────────────
    outaged_lines = []
    outaged_generators = []
    for outage in PLANNED_OUTAGES:
        if outage["status"] not in ("APPROVED", "SCHEDULED"):
            continue
        # Check if this month falls within the outage window
        outage_start = outage["start_year"] * 12 + outage["start_month"]
        outage_end   = outage["end_year"]   * 12 + outage["end_month"]
        current      = year * 12 + month
        if outage_start <= current <= outage_end:
            if outage["element_type"] == "line":
                outaged_lines.append(outage["element_id"])
            elif outage["element_type"] == "generator":
                outaged_generators.append(outage["element_id"])

    # Remove outaged lines from monthly topology
    for lid in outaged_lines:
        if lid in lines_monthly:
            del lines_monthly[lid]

    # ── Apply seasonal GTC adjustments ────────────────────────────────────
    gtcs_monthly = {}
    for gtc_id, gtc_data in GENERIC_TRANSMISSION_CONSTRAINTS.items():
        gmod = copy.deepcopy(gtc_data)
        # Apply seasonal factor if defined
        seasonal_factors = SEASONAL_GTC_FACTORS.get(gtc_id, {})
        gtc_factor = seasonal_factors.get(month, 1.0)
        gmod["flow_limit"] = int(gtc_data["flow_limit"] * gtc_factor)
        # Remove monitored lines that are outaged
        gmod["monitored_lines"] = [
            (lid, sign) for lid, sign in gmod["monitored_lines"]
            if lid not in outaged_lines
        ]
        gtcs_monthly[gtc_id] = gmod

    # ── Update contingency list ───────────────────────────────────────────
    contingencies_monthly = {}
    for ctg_id, ctg_data in CONTINGENCIES.items():
        if "outaged_line" in ctg_data:
            if ctg_data["outaged_line"] in outaged_lines:
                continue  # Can't have contingency on already-outaged line
            if ctg_data["outaged_line"] not in lines_monthly:
                continue
        if "outaged_gen" in ctg_data:
            if ctg_data["outaged_gen"] in outaged_generators:
                continue
        contingencies_monthly[ctg_id] = ctg_data

    # ── Modify generator availability ─────────────────────────────────────
    generators_monthly = {}
    for gid, gdata in GENERATORS.items():
        if gid in outaged_generators:
            continue
        generators_monthly[gid] = copy.deepcopy(gdata)

    return {
        "year":                year,
        "month":               month,
        "rating_factor":       rating_factor,
        "lines":               lines_monthly,
        "generators":          generators_monthly,
        "gtcs":                gtcs_monthly,
        "outaged_lines":       outaged_lines,
        "outaged_generators":  outaged_generators,
        "contingencies":       contingencies_monthly,
    }


def print_monthly_topology_summary(year: int, month: int):
    """Print a human-readable summary of monthly topology changes."""
    mt = get_monthly_topology(year, month)
    import calendar
    mname = calendar.month_abbr[month]

    print(f"=== {year}-{mname} Topology (CRR-style) ===")
    print(f"  Rating factor: {mt['rating_factor']:.2f}x "
          f"({'tighter' if mt['rating_factor'] < 1 else 'relaxed'})")
    print(f"  Lines available: {len(mt['lines'])}/{len(LINES)}")
    print(f"  Generators available: {len(mt['generators'])}/{len(GENERATORS)}")
    print(f"  Contingencies: {len(mt['contingencies'])}/{len(CONTINGENCIES)}")

    if mt["outaged_lines"]:
        print(f"  *** Outaged lines: {mt['outaged_lines']}")
    if mt["outaged_generators"]:
        print(f"  *** Outaged generators: {mt['outaged_generators']}")

    # Show GTC changes
    for gtc_id, gtc in mt["gtcs"].items():
        orig = GENERIC_TRANSMISSION_CONSTRAINTS[gtc_id]["flow_limit"]
        new  = gtc["flow_limit"]
        if new != orig:
            print(f"  GTC {gtc_id}: {orig} → {new} MW "
                  f"({'tighter' if new < orig else 'relaxed'})")

    # Show a few key line limit changes
    sample_lines = ["L_NORTH_HOUSTON_1", "L_WEST_INTRAZONAL_1",
                    "L_SOUTH_HOUSTON", "L_RGV_TO_SOUTH"]
    for lid in sample_lines:
        if lid in mt["lines"]:
            orig = LINES[lid]["flow_limit"]
            new  = mt["lines"][lid]["flow_limit"]
            print(f"  {lid}: {orig} → {new} MW")
        elif lid in LINES:
            print(f"  {lid}: OUTAGED this month")
