"""
Battery Dispatch Profile Generator (v2)
=========================================
Generates a deterministic battery dispatch profile that mirrors real
2025 ERCOT battery operating patterns:

  - HE 11-15: charge at full power (negative net injection = consumes load)
  - HE 17-20: discharge at full power (positive net injection = generation)
  - Other hours: idle (zero net injection)

This is a price-taker model: batteries arbitrage based on EXPECTED LMP
shape. In a more sophisticated version they would optimize against
forecasted LMPs, but the deterministic charge/discharge pattern captures
the dominant market impact.

The result is a per-bus, per-hour MW adjustment that gets ADDED to load
(positive battery generation = subtracted from load → load decreases).
"""

import numpy as np
import pandas as pd

from .topology import (
    HOURS, N_HOURS, BUSES, BUS_LIST, BATTERIES, BATTERY_LIST
)


def generate_battery_profile(
    seed: int = 50,
    charge_hours: tuple  = (11, 12, 13, 14, 15),
    discharge_hours: tuple = (17, 18, 19, 20),
    cycle_efficiency: float = 0.85,
) -> pd.DataFrame:
    """
    Generate hourly battery NET INJECTION at each bus.

    Convention:
        + value = battery generating (acts as supply, decreases net load)
        - value = battery charging  (acts as load, increases net load)

    Returns DataFrame [N_HOURS x N_BUS] with battery contribution (MW).
    """
    rng = np.random.default_rng(seed)

    # Per-battery hourly profile
    bat_profile = pd.DataFrame(0.0, index=HOURS, columns=BATTERY_LIST)

    for bat_id, bat in BATTERIES.items():
        power = bat["power_mw"]
        energy = bat["energy_mwh"]

        # Calculate charge and discharge MW
        # Energy budget: must charge enough to discharge for full duration
        discharge_dur = len(discharge_hours)
        charge_dur    = len(charge_hours)

        # Discharge: deliver energy_mwh × eff over discharge_dur hours
        discharge_mw = min(power, energy * cycle_efficiency / discharge_dur)
        # Charge: must replenish energy_mwh / eff over charge_dur hours
        charge_mw    = min(power, energy / charge_dur)

        for h_idx, ts in enumerate(HOURS):
            hour = ts.hour + 1   # HE convention 1-24

            if hour in discharge_hours:
                # Add some variability — not always at full power
                # Higher discharge in summer, less in shoulder months
                month_factor = 1.0 + 0.20 * np.exp(-((ts.dayofyear - 220) / 70)**2)
                noise        = rng.uniform(0.85, 1.0)
                bat_profile.iat[h_idx, BATTERY_LIST.index(bat_id)] = (
                    discharge_mw * month_factor * noise
                )
            elif hour in charge_hours:
                # Charge most aggressively in summer (more solar)
                month_factor = 1.0 + 0.15 * np.exp(-((ts.dayofyear - 200) / 80)**2)
                noise        = rng.uniform(0.90, 1.0)
                bat_profile.iat[h_idx, BATTERY_LIST.index(bat_id)] = (
                    -charge_mw * month_factor * noise
                )
            # else: 0 (idle)

    # Aggregate to bus level
    bus_battery_injection = pd.DataFrame(0.0, index=HOURS, columns=BUS_LIST)
    for bat_id, bat in BATTERIES.items():
        bus = bat["bus"]
        bus_battery_injection[bus] += bat_profile[bat_id]

    return bus_battery_injection, bat_profile


if __name__ == "__main__":
    bus_inj, bat_prof = generate_battery_profile()
    print(f"Bus battery injection shape: {bus_inj.shape}")
    print(f"Battery unit profile shape: {bat_prof.shape}")
    print()
    print("Per-battery summary:")
    print(f"  Total annual discharge: {bat_prof[bat_prof > 0].sum().sum() / 1000:.0f} GWh")
    print(f"  Total annual charge:    {-bat_prof[bat_prof < 0].sum().sum() / 1000:.0f} GWh")
    print()
    print("Sample day (Jul 15, 2024):")
    sample = bat_prof.loc["2024-07-15 11:00":"2024-07-15 21:00"]
    print(sample.round(1).to_string())
    print()
    print("System impact (sum across all batteries) by HE:")
    by_hour = bat_prof.sum(axis=1).groupby(bat_prof.index.hour).mean()
    for h in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
        bar_len = abs(by_hour.get(h, 0)) / 10
        sign = "▲" if by_hour.get(h, 0) > 0 else "▼" if by_hour.get(h, 0) < 0 else "·"
        print(f"  HE {h+1:2d}: {by_hour.get(h, 0):+6.1f} MW  {sign * int(bar_len)}")
