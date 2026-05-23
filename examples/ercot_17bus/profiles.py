"""
8760-Hour Profile Generator for ERCOT-Flavored System
======================================================
Generates realistic hourly profiles for:
  - Load (each bus)
  - Wind capacity factor (Panhandle, West, South — different regimes)
  - Solar capacity factor (only daytime, peak at noon)
  - Gas prices (monthly, with winter spikes)

Profile shapes are calibrated to match ERCOT statistics:
  - Summer cooling load: dominant in July-August
  - Winter morning ramp: 6-9am during cold snaps
  - Wind: anti-correlated with load (high overnight, low peak)
  - Solar: capacity factor ~25% annually, peak at solar noon

Output: pandas DataFrame indexed by hourly timestamp.
"""

import numpy as np
import pandas as pd
from .topology import (
    HOURS, N_HOURS, BUSES, BUS_LIST, ZONE_PEAK_LOAD,
    BUS_LOAD_FRACTION, ERCOT_TOTAL_PEAK, GENERATORS, GEN_LIST
)


def _hour_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract calendar features used by all profile generators."""
    df = pd.DataFrame(index=timestamps)
    df["hour"]      = timestamps.hour
    df["dow"]       = timestamps.dayofweek
    df["month"]     = timestamps.month
    df["doy"]       = timestamps.dayofyear
    df["is_weekend"]= (df["dow"] >= 5).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Load Profile
# ─────────────────────────────────────────────────────────────────────────────
def generate_load_profile(seed: int = 42) -> pd.DataFrame:
    """
    Generate ERCOT-style hourly load for each bus.

    Components:
      base_factor:    annual seasonal envelope (summer dominant)
      diurnal:        morning + afternoon peak shape
      weekend_factor: ~10-15% lower
      noise:          AR(1) for realistic short-term variability
      heat_wave:      occasional extreme summer days (~5x/year)
      cold_snap:      occasional extreme winter days (~3x/year)

    Returns
    -------
    DataFrame [N_HOURS x N_BUS] with bus IDs as columns
    """
    rng = np.random.default_rng(seed)
    feats = _hour_features(HOURS)

    # ── 1. Annual seasonal envelope (cooling-dominated) ──────────────────────
    # ERCOT: summer peak (Aug) ≈ 1.0, winter (Jan) ≈ 0.65, spring ≈ 0.55
    doy = feats["doy"].values
    seasonal = (
        0.55                                                # base level (spring)
        + 0.45 * np.exp(-((doy - 220) / 50)**2)            # summer Gaussian peak (Aug 8 = doy 220)
        + 0.18 * np.exp(-((doy - 15)  / 30)**2)            # winter morning peak boost
        + 0.18 * np.exp(-((doy - 380) / 30)**2)            # year-end winter (wraparound)
    )

    # ── 2. Diurnal shape (varies by season) ──────────────────────────────────
    hour = feats["hour"].values
    month = feats["month"].values

    # Summer shape: late afternoon peak (HE 17-18), morning shoulder
    summer_diurnal = (
        0.65 + 0.35 * np.sin(np.pi * np.clip((hour - 6) / 14, 0, 1))**1.5
    )
    summer_diurnal[(hour >= 14) & (hour <= 19)] *= 1.10   # afternoon AC peak boost

    # Winter shape: morning peak (HE 7-8), evening peak (HE 18-20)
    winter_diurnal = 0.65 + 0.20 * (
        np.exp(-((hour - 7)  / 2.5)**2)                    # morning ramp
        + np.exp(-((hour - 19) / 3.0)**2)                  # evening ramp
    )

    # Blend by season
    summer_weight = np.exp(-((doy - 220) / 80)**2)
    winter_weight = 1 - summer_weight
    diurnal = summer_weight * summer_diurnal + winter_weight * winter_diurnal

    # ── 3. Weekend factor ────────────────────────────────────────────────────
    weekend_factor = np.where(feats["is_weekend"].values == 1, 0.88, 1.0)

    # ── 4. Random noise (AR(1) for autocorrelation) ──────────────────────────
    noise = np.zeros(N_HOURS)
    eps = rng.normal(0, 0.025, N_HOURS)
    for t in range(1, N_HOURS):
        noise[t] = 0.8 * noise[t-1] + eps[t]

    # ── 5. Heat waves (extreme summer days) ──────────────────────────────────
    heat_wave_boost = np.zeros(N_HOURS)
    n_summers = (HOURS[-1].year - HOURS[0].year + 1)
    for yr in range(HOURS[0].year, HOURS[-1].year + 1):
        # 4-6 heat wave events per year, each lasting 2-4 days
        n_events = rng.integers(4, 7)
        for _ in range(n_events):
            doy_event = rng.integers(170, 250)
            duration  = rng.integers(2, 5)
            magnitude = rng.uniform(0.05, 0.12)            # 5-12% load boost (was 10-25%)
            start = pd.Timestamp(year=yr, month=1, day=1) + pd.Timedelta(days=int(doy_event))
            for d in range(duration):
                day = start + pd.Timedelta(days=d)
                if day in HOURS:
                    mask = (HOURS.date == day.date())
                    afternoon_mask = mask & (HOURS.hour >= 14) & (HOURS.hour <= 19)
                    heat_wave_boost[afternoon_mask] += magnitude

    # ── 6. Cold snaps (extreme winter, e.g., Uri 2021) ───────────────────────
    cold_snap_boost = np.zeros(N_HOURS)
    for yr in range(HOURS[0].year, HOURS[-1].year + 1):
        n_events = rng.integers(2, 4)
        for _ in range(n_events):
            doy_event = rng.choice([rng.integers(1, 50), rng.integers(335, 366)])
            duration  = rng.integers(2, 5)
            magnitude = rng.uniform(0.08, 0.18)            # 8-18% boost (was 15-35%)
            start = pd.Timestamp(year=yr, month=1, day=1) + pd.Timedelta(days=int(doy_event))
            for d in range(duration):
                day = start + pd.Timedelta(days=d)
                if day in HOURS:
                    mask = (HOURS.date == day.date())
                    morning_mask = mask & (HOURS.hour >= 5) & (HOURS.hour <= 10)
                    cold_snap_boost[morning_mask] += magnitude

    # ── Combine all components ───────────────────────────────────────────────
    system_load_pu = seasonal * diurnal * weekend_factor * (1 + noise) \
                     * (1 + heat_wave_boost + cold_snap_boost)
    system_load_mw = system_load_pu * ERCOT_TOTAL_PEAK

    # ── Distribute to buses ──────────────────────────────────────────────────
    bus_load = pd.DataFrame(index=HOURS, columns=BUS_LIST, dtype=float)
    for b in BUS_LIST:
        bus_load[b] = system_load_mw * BUS_LOAD_FRACTION[b]

    return bus_load


# ─────────────────────────────────────────────────────────────────────────────
#  Wind Capacity Factor Profile
# ─────────────────────────────────────────────────────────────────────────────
def generate_wind_profile(seed: int = 43) -> pd.DataFrame:
    """
    Generate hourly wind capacity factor (0-1) by region.
    
    ERCOT wind characteristics:
      - Annual CF ~38% in Panhandle, ~32% in West/South
      - Anti-correlated with load (high overnight, lull in afternoon)
      - High auto-correlation (multi-day calm or windy spells)
      - Inversely correlated with summer (seasonal lull in July-Aug)

    Returns
    -------
    DataFrame indexed by hour, columns = ['Panhandle', 'West', 'South']
    """
    rng = np.random.default_rng(seed)
    feats = _hour_features(HOURS)
    n = N_HOURS

    profiles = pd.DataFrame(index=HOURS)

    for region, base_cf in [("Panhandle", 0.38), ("West", 0.32), ("South", 0.30)]:
        # Diurnal: low in afternoon, high overnight
        hour = feats["hour"].values
        diurnal = 1.0 - 0.15 * np.sin(np.pi * (hour - 4) / 12)

        # Seasonal: lull in summer, peak in spring (Mar-May) and fall
        doy = feats["doy"].values
        seasonal = (
            1.0
            - 0.20 * np.exp(-((doy - 220) / 60)**2)        # summer lull
            + 0.10 * np.exp(-((doy - 100) / 40)**2)        # spring peak
            + 0.08 * np.exp(-((doy - 290) / 40)**2)        # fall peak
        )

        # AR(1) regional noise (multi-day calm/windy spells)
        noise = np.zeros(n)
        eps = rng.normal(0, 0.18, n)
        for t in range(1, n):
            noise[t] = 0.95 * noise[t-1] + eps[t]          # high persistence

        # Combine — each region has independent noise
        cf = base_cf * diurnal * seasonal + noise * base_cf
        cf = np.clip(cf, 0, 1)
        profiles[region] = cf

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
#  Solar Capacity Factor Profile
# ─────────────────────────────────────────────────────────────────────────────
def generate_solar_profile(seed: int = 44) -> pd.DataFrame:
    """
    Solar capacity factor: zero at night, bell curve during daylight.
    Peak ~80% at solar noon on clear summer day.
    Annual CF ~25%.
    """
    rng = np.random.default_rng(seed)
    feats = _hour_features(HOURS)

    profiles = pd.DataFrame(index=HOURS)

    for region in ["West", "South"]:
        hour = feats["hour"].values
        doy = feats["doy"].values

        # Daylight hours (CDT): roughly HE 7 to HE 20.
        # Use a single sine bell over the daylight window. Avoid cos(...)**2
        # here: squared cosine has multiple lobes over this interval and can
        # create artificial morning/noon/evening solar peaks.
        daylight_start = 7.0
        daylight_end = 20.0
        daylight_span = daylight_end - daylight_start
        solar_phase = (hour - daylight_start) / daylight_span
        clear_sky = 0.85 * np.sin(np.pi * np.clip(solar_phase, 0, 1))
        clear_sky[(hour < daylight_start) | (hour > daylight_end)] = 0

        # Seasonal: longer + brighter in summer
        seasonal = 1.0 + 0.15 * np.sin((doy - 80) / 365 * 2 * np.pi)

        # Cloud noise: occasional cloudy days
        cloud_noise = rng.uniform(0.6, 1.0, N_HOURS)
        # Persist clouds for several hours
        for t in range(1, N_HOURS):
            if rng.uniform() < 0.7:
                cloud_noise[t] = 0.6 * cloud_noise[t-1] + 0.4 * cloud_noise[t]

        cf = clear_sky * seasonal * cloud_noise
        cf = np.clip(cf, 0, 1)
        profiles[region] = cf

    return profiles


# ─────────────────────────────────────────────────────────────────────────────
#  Apply renewable profiles to specific generators
# ─────────────────────────────────────────────────────────────────────────────
def apply_renewable_profiles(load_df: pd.DataFrame,
                               wind_df: pd.DataFrame,
                               solar_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each renewable generator, multiply Pmax by hourly capacity factor.
    Returns a DataFrame [N_HOURS x N_GEN] of effective Pmax(t).

    Region aliasing: South_RGV (Magic Valley) reuses the South solar profile.
    """
    # Region aliases: map new region names to existing profile columns
    region_alias = {"South_RGV": "South"}

    pmax_t = pd.DataFrame(index=load_df.index, columns=GEN_LIST, dtype=float)

    for g, gd in GENERATORS.items():
        gen_region = region_alias.get(gd["region"], gd["region"])
        if gd["type"] == "WIND":
            cf = wind_df[gen_region]
            pmax_t[g] = gd["Pmax"] * cf
        elif gd["type"] == "SOLAR":
            cf = solar_df[gen_region]
            pmax_t[g] = gd["Pmax"] * cf
        else:
            pmax_t[g] = gd["Pmax"]

    return pmax_t


# ─────────────────────────────────────────────────────────────────────────────
#  Gas Price Profile (monthly, drives marginal cost shifts)
# ─────────────────────────────────────────────────────────────────────────────
def generate_gas_price_profile(seed: int = 45) -> pd.Series:
    """
    Monthly Henry Hub gas price [$/MMBtu], drives gas plant marginal cost.
    Baseline ~$3.50/MMBtu, with winter spikes to $6-12.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(HOURS[0], HOURS[-1], freq="MS")
    prices = []
    for m in months:
        base = 3.50
        if m.month in [12, 1, 2]:
            base += rng.uniform(1.5, 4.0)                  # winter premium
        elif m.month in [6, 7, 8]:
            base += rng.uniform(0.5, 1.5)                  # summer cooling demand
        # Random variation
        base += rng.normal(0, 0.4)
        prices.append(max(2.0, base))

    monthly_price = pd.Series(prices, index=months)
    # Forward-fill to hourly
    hourly_price = monthly_price.reindex(HOURS, method="ffill")
    return hourly_price


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level: build all profiles in one call
# ─────────────────────────────────────────────────────────────────────────────
def build_all_profiles(seed: int = 42) -> dict:
    """
    Generate all 8760-hour profiles needed for SCED simulation.

    Returns
    -------
    dict with keys:
        load        : DataFrame [N_HOURS x N_BUS]
        wind        : DataFrame [N_HOURS x 3 regions]
        solar       : DataFrame [N_HOURS x 2 regions]
        pmax_t      : DataFrame [N_HOURS x N_GEN]  effective Pmax per hour
        gas_price   : Series [N_HOURS]             $/MMBtu
        gen_cost_t  : DataFrame [N_HOURS x N_GEN]  marginal cost per hour
                                                    (gas plants vary with gas price)
    """
    print("Generating profiles...")
    load_df  = generate_load_profile(seed=seed)
    wind_df  = generate_wind_profile(seed=seed+1)
    solar_df = generate_solar_profile(seed=seed+2)
    gas_p    = generate_gas_price_profile(seed=seed+3)

    pmax_t   = apply_renewable_profiles(load_df, wind_df, solar_df)

    # ── Time-varying gen cost (gas plants modulate with gas price) ───────────
    gen_cost_t = pd.DataFrame(index=HOURS, columns=GEN_LIST, dtype=float)
    for g, gd in GENERATORS.items():
        cb = gd["cost_b"]
        if gd["type"] in ("COMBINED_CYCLE", "GAS_TURBINE"):
            # Gas cost scales with gas price (heat rate ~7-12 MMBtu/MWh embedded in cost_b)
            heat_rate = 7.5 if gd["type"] == "COMBINED_CYCLE" else 11.0
            gen_cost_t[g] = heat_rate * gas_p
        else:
            gen_cost_t[g] = cb   # constant for renewable/coal/nuclear

    return {
        "load"      : load_df,
        "wind"      : wind_df,
        "solar"     : solar_df,
        "pmax_t"    : pmax_t,
        "gas_price" : gas_p,
        "gen_cost_t": gen_cost_t,
    }


if __name__ == "__main__":
    profiles = build_all_profiles()
    print(f"\nProfile dimensions:")
    print(f"  Load shape:      {profiles['load'].shape}")
    print(f"  Wind shape:      {profiles['wind'].shape}")
    print(f"  Solar shape:     {profiles['solar'].shape}")
    print(f"  Pmax(t) shape:   {profiles['pmax_t'].shape}")
    print(f"  Gas price range: ${profiles['gas_price'].min():.2f} - ${profiles['gas_price'].max():.2f}")
    print(f"  System peak load: {profiles['load'].sum(axis=1).max():.0f} MW")
    print(f"  System min load:  {profiles['load'].sum(axis=1).min():.0f} MW")
    print(f"  Annual energy:    {profiles['load'].sum().sum()/1e6:.2f} TWh")
