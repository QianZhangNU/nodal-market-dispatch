"""
CRR Auction Example -- ERCOT 17-Bus System
==========================================
Demonstrates a realistic Congestion Revenue Right (CRR) auction workflow:

  Round 1 -- Annual CRR Auction (July peak topology)
      Allocates the bulk of network capacity to hedging participants.
      Summer model: tight line ratings (0.80x), peak-load topology.
      Capacity factor: 75% of SFT -- reflecting ERCOT practice of holding
      back 25% for monthly auctions and reliability buffer.

  Round 2 -- Monthly Short-Term CRR Auction (January, residual capacity)
      Uses winter topology (relaxed ratings, 1.20x) and treats the Annual
      awards as baseload CRRs consuming SFT. Only residual capacity is sold.
      Capacity factor: 90% of adjusted SFT limits.

Market Participant Archetypes
-----------------------------
  LSE_NORTH    DFW-area load-serving entity.  Buys OBL to hedge congestion cost
               of serving load that imports from West wind and North CC gas.
               Exposed to: RN_PANHNDL < LZ_NORTH when GTC_WEST_EXPORT binds.

  LSE_HOUSTON  Houston-area LSE, largest congestion exposure in the system.
               GTC_HOUSTON_IMPORT + local load-pocket lines -> HB_HOUSTON <
               LZ_HOUSTON during peak hours.  Buys heavily into Houston.

  LSE_SOUTH    South Texas + RGV load zone.  Buys options to hedge solar export
               constraints that can depress RGV prices below LZ_SOUTH.

  WIND_WEST    West Texas wind developer.  BUYs OBL to hedge locational discount
               when GTC_WEST_EXPORT pins RN_PANHNDL below system average.

  SOLAR_RGV    RGV + Laredo solar project.  Prefers OPT over OBL to avoid
               negative settlement risk in off-peak hours (no export congestion).

  CRR_FUND_A   Financial market maker.  Buys a portfolio of OPT on the highest-
               expected-congestion paths; SELLs back excess OBL capacity to earn
               auction clearing proceeds.

  UTIL_NCNTRL  North Central coal utility.  Returns excess OBL capacity from a
               prior period via SELL, releasing SFT headroom for others.

Path Congestion Rationale (summer peak, GTC binding)
-----------------------------------------------------
  RN_PANHNDL -> LZ_NORTH    West export GTC -> Panhandle prices cheap vs DFW
  RN_PANHNDL -> LZ_HOUSTON  Full West->Houston path: GTC + Houston import
  HB_HOUSTON -> LZ_HOUSTON  Houston load-pocket intrazonal premium
  HB_NORTH   -> LZ_HOUSTON  DFW backbone -> Houston import limit
  HB_WEST    -> LZ_WEST     West intrazonal: gen-bus vs Odessa/Midland load
  RN_RGV_SOLAR-> LZ_SOUTH   RGV solar pocket: GTC_SOUTH_TEXAS_EXPORT binding
  RN_RGV_SOLAR-> HB_SOUTH   RGV export option: solar curtailment exposure
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Allow `python run_auction.py` from the examples directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from solvers.network import NetworkModel
from solvers.auction import clear_auction
from examples.ercot_17bus.topology import (
    BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST, SETTLEMENT_POINTS,
    SLACK_BUS, get_monthly_topology,
)


# -----------------------------------------------------------------------------
#  Reference LMPs
#  These are illustrative monthly-average settlement-point LMPs ($/MWh)
#  derived from the DAM simulation for typical summer-peak and winter days.
#  They are used only for post-auction hedge-value analysis, not for clearing.
# -----------------------------------------------------------------------------

REF_LMP_JULY = {
    # West/Panhandle: export-constrained when GTC_WEST_EXPORT binds
    "RN_PANHNDL":     22.0,
    "RN_WEST":        24.0,
    "HB_WEST":        23.0,
    "LZ_WEST":        31.0,   # Odessa/Midland load pocket premium (+8)
    # North: DFW area, moderate import premium
    "RN_NORTH_GAS":   34.0,
    "RN_NCNTRL_COAL": 29.0,
    "HB_NORTH":       33.0,
    "LZ_NORTH":       35.0,
    # Houston: load pocket, largest premium
    "RN_HOU_GEN":     37.0,
    "RN_HOU_PEAK":    42.0,
    "HB_HOUSTON":     38.0,
    "LZ_HOUSTON":     51.0,   # Houston load pocket premium (+13)
    # South: Austin + San Antonio
    "RN_SOUTH_GEN":   31.0,
    "RN_AUSTIN_GEN":  33.0,
    "HB_SOUTH":       28.0,
    "LZ_SOUTH":       34.0,
    # RGV solar pocket: export constrained during HE 11-15
    "RN_RGV_SOLAR":    4.0,   # Export pocket, GTC binding (-24 vs LZ_HOUSTON!)
    "RN_LAREDO_SOLAR": 6.0,
    # System hub
    "HB_BUSAVG":      32.0,
}

REF_LMP_JANUARY = {
    "RN_PANHNDL":     35.0,   # Wind strong, but cold snap spikes gas
    "RN_WEST":        36.0,
    "HB_WEST":        35.5,
    "LZ_WEST":        38.0,   # Smaller intrazonal spread in winter
    "RN_NORTH_GAS":   42.0,
    "RN_NCNTRL_COAL": 38.0,
    "HB_NORTH":       41.0,
    "LZ_NORTH":       43.0,
    "RN_HOU_GEN":     44.0,
    "RN_HOU_PEAK":    48.0,
    "HB_HOUSTON":     44.0,
    "LZ_HOUSTON":     52.0,   # Houston pocket still premium (+8)
    "RN_SOUTH_GEN":   40.0,
    "RN_AUSTIN_GEN":  41.0,
    "HB_SOUTH":       39.0,
    "LZ_SOUTH":       43.0,
    "RN_RGV_SOLAR":   38.0,   # Winter: minimal solar, no export constraint
    "RN_LAREDO_SOLAR":37.0,
    "HB_BUSAVG":      40.0,
}


# -----------------------------------------------------------------------------
#  Bid Definitions
#
#  Each row: (agent_id, source_sp, sink_sp, side, crr_type, bid_$/MWh, MW)
#  Prices represent agents' private valuation of the CRR path's expected value.
#  Buyers who bid ABOVE the ACP will be awarded; sellers below ACP will sell.
# -----------------------------------------------------------------------------

def _bids(rows):
    """Convert list-of-tuples to auction DataFrame."""
    return pd.DataFrame(rows, columns=[
        "agent_id", "source", "sink", "side", "crr_type", "bid_price", "mw"])


# -- Annual Auction bids (summer valuation) -----------------------------------
ANNUAL_BIDS = _bids([
    # agent         source           sink            side   type   $/MWh   MW
    # -- LSE_NORTH: DFW load hedge ------------------------------------------
    # Primary hedge: Panhandle wind export path.  When GTC_WEST_EXPORT binds,
    # RN_PANHNDL << LZ_NORTH.  OBL settles at (LZ_NORTH - RN_PANHNDL) > 0.
    ("LSE_NORTH", "RN_PANHNDL",    "LZ_NORTH",   "BUY", "OBL", 12.0, 200),
    # Secondary hedge: West hub basis vs North load zone.
    ("LSE_NORTH", "HB_WEST",       "LZ_NORTH",   "BUY", "OBL",  7.0, 100),
    # Opportunistic: South Texas export option to North (unlikely but cheap).
    ("LSE_NORTH", "RN_RGV_SOLAR",  "LZ_NORTH",   "BUY", "OPT",  2.0,  50),

    # -- LSE_HOUSTON: Houston load hedge, largest exposure ------------------
    # Core position: gen-side hub vs load zone (intrazonal pocket premium).
    # Houston peakers at bus 6 -> LZ_HOUSTON can be $10-20 above HB_HOUSTON.
    ("LSE_HOUSTON", "HB_HOUSTON",  "LZ_HOUSTON", "BUY", "OBL", 16.0, 300),
    # Cross-path: North backbone import route.  GTC_HOUSTON_IMPORT binding.
    ("LSE_HOUSTON", "HB_NORTH",    "LZ_HOUSTON", "BUY", "OBL", 10.0, 150),
    # Wind-to-load: captures full West->Houston path spread.
    ("LSE_HOUSTON", "RN_PANHNDL",  "LZ_HOUSTON", "BUY", "OBL", 14.0, 120),

    # -- LSE_SOUTH: South Texas and RGV load zone ---------------------------
    # Local gen hedge: South CC to South load zone (small spread).
    ("LSE_SOUTH", "RN_SOUTH_GEN",  "LZ_SOUTH",   "BUY", "OBL",  5.0, 100),
    # RGV solar option: when GTC_SOUTH_TEXAS_EXPORT binds, RGV prices crash
    # below LZ_SOUTH.  OPT only pays when spread > 0 (avoids reverse exposure).
    ("LSE_SOUTH", "RN_RGV_SOLAR",  "LZ_SOUTH",   "BUY", "OPT", 10.0, 150),

    # -- WIND_WEST: West Texas wind developer -------------------------------
    # Hedge locational discount: when export binds, RN_PANHNDL < HB_BUSAVG.
    # OBL(RN_PANHNDL -> HB_BUSAVG) pays (HB_BUSAVG - RN_PANHNDL) > 0.
    ("WIND_WEST", "RN_PANHNDL",    "HB_BUSAVG",  "BUY", "OBL",  9.0, 250),
    # West intrazonal option: upside if gen bus > load bus (reverse pocket).
    ("WIND_WEST", "RN_WEST",       "HB_WEST",    "BUY", "OPT",  3.0, 100),

    # -- SOLAR_RGV: RGV solar project (Pmax 2.0 GW) ------------------------
    # OPT preferred over OBL: avoids negative settlement in off-peak hours
    # when there is no export constraint and RGV prices equal the system.
    # When GTC_SOUTH_TEXAS_EXPORT binds during HE 11-15, this OPT pays out.
    ("SOLAR_RGV", "RN_RGV_SOLAR",  "HB_SOUTH",   "BUY", "OPT", 13.0, 350),
    # Laredo solar complementary position (same export pocket, smaller unit).
    ("SOLAR_RGV", "RN_LAREDO_SOLAR","HB_SOUTH",  "BUY", "OPT",  8.0, 100),

    # -- CRR_FUND_A: Financial market maker --------------------------------
    # Cross-system option: West hub to Houston load zone -- pure spread trade.
    ("CRR_FUND_A", "HB_WEST",      "LZ_HOUSTON", "BUY", "OPT",  6.0, 100),
    # Wind-to-load option strip (OPT avoids paying when spread flips).
    ("CRR_FUND_A", "RN_PANHNDL",   "LZ_HOUSTON", "BUY", "OPT",  5.0,  80),
    # SELL: return system-wide OBL capacity held from prior period.
    # Reservation price $3/MWh; earns ACP if ACP > $3.
    ("CRR_FUND_A", "HB_BUSAVG",    "LZ_HOUSTON", "SELL","OBL",  3.0, 120),

    # -- UTIL_NCNTRL: North Central coal utility, capacity return -----------
    # SELL: returns excess annual OBL (not needed after fuel cost savings).
    ("UTIL_NCNTRL", "RN_NCNTRL_COAL","LZ_NORTH",  "SELL","OBL",  1.5, 200),
    ("UTIL_NCNTRL", "RN_NCNTRL_COAL","LZ_HOUSTON","SELL","OBL",  1.0, 100),
])

# -- Monthly (Short-Term CRR) bids for January ---------------------------------
# Lower valuations: winter has wider line ratings and less solar congestion.
# These bids compete only for residual capacity after annual awards.
MONTHLY_BIDS_JAN = _bids([
    # agent         source           sink            side   type   $/MWh   MW
    ("LSE_NORTH",   "RN_PANHNDL",    "LZ_NORTH",   "BUY", "OBL",  6.0, 100),
    ("LSE_NORTH",   "HB_WEST",       "LZ_NORTH",   "BUY", "OBL",  3.5,  60),
    ("LSE_HOUSTON", "HB_HOUSTON",    "LZ_HOUSTON", "BUY", "OBL",  9.0, 150),
    ("LSE_HOUSTON", "HB_NORTH",      "LZ_HOUSTON", "BUY", "OBL",  4.5,  80),
    ("LSE_SOUTH",   "RN_RGV_SOLAR",  "LZ_SOUTH",   "BUY", "OPT",  2.0,  60),
    ("WIND_WEST",   "RN_PANHNDL",    "HB_BUSAVG",  "BUY", "OBL",  4.5, 120),
    ("SOLAR_RGV",   "RN_RGV_SOLAR",  "HB_SOUTH",   "BUY", "OPT",  4.0, 150),
    ("CRR_FUND_A",  "HB_WEST",       "LZ_HOUSTON", "BUY", "OPT",  2.5,  50),
    ("CRR_FUND_A",  "HB_BUSAVG",     "LZ_HOUSTON", "SELL","OBL",  1.5,  80),
    ("UTIL_NCNTRL", "RN_NCNTRL_COAL","LZ_NORTH",   "SELL","OBL",  0.8, 120),
])


# -----------------------------------------------------------------------------
#  Auction Runners
# -----------------------------------------------------------------------------

def _build_network(year: int, month: int) -> tuple:
    """Return (network_model, monthly_topology_dict) for the given month."""
    mt = get_monthly_topology(year, month)
    nm = NetworkModel(
        buses=BUSES,
        lines=mt["lines"],
        slack_bus=SLACK_BUS,
        contingencies=mt["contingencies"],
    )
    return nm, mt


def run_annual_crr_auction(
    year: int = 2024,
    summer_month: int = 7,
    capacity_factor: float = 0.75,
    verbose: bool = True,
) -> dict:
    """
    Annual CRR Auction -- uses summer peak topology.

    Parameters
    ----------
    year            : Calendar year for topology overlay.
    summer_month    : Month used to represent summer peak model (default July).
    capacity_factor : Fraction of SFT capacity sold in Annual Auction.
                      ERCOT convention: ~75% in Annual, 90% in Monthly.
    """
    nm, mt = _build_network(year, summer_month)

    if verbose:
        rf = mt["rating_factor"]
        print(f"\n{'='*65}")
        print(f"  ANNUAL CRR AUCTION -- {year} | Summer Model ({year}-{summer_month:02d})")
        print(f"  Rating factor: {rf:.2f}x  |  SFT cap factor: {capacity_factor:.0%}")
        print(f"  Lines: {len(mt['lines'])}  |  Settlement points: {len(SETTLEMENT_POINTS)}")
        if mt["outaged_lines"]:
            print(f"  Outaged lines: {mt['outaged_lines']}")
        print(f"{'='*65}")
        _print_bid_summary(ANNUAL_BIDS, "Annual CRR Bids Submitted")

    result = clear_auction(
        bids_df=ANNUAL_BIDS,
        ptdf=nm.ptdf,
        lines=mt["lines"],
        line_list=list(mt["lines"].keys()),
        bus_list=BUS_LIST,
        bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
        capacity_factor=capacity_factor,
        baseload_crrs=None,
    )

    if verbose:
        _print_auction_results(result, "ANNUAL AUCTION RESULTS", REF_LMP_JULY)

    return result


def run_monthly_crr_auction(
    annual_result: dict,
    year: int = 2024,
    month: int = 1,
    capacity_factor: float = 0.90,
    verbose: bool = True,
) -> dict:
    """
    Monthly Short-Term CRR Auction -- uses January winter topology.
    Annual awards are treated as baseload CRRs that consume SFT capacity.

    Parameters
    ----------
    annual_result   : Output of run_annual_crr_auction().
    year, month     : Delivery month for this auction.
    capacity_factor : Fraction of winter-adjusted SFT sold in Monthly.
    """
    nm, mt = _build_network(year, month)

    # Convert annual awards to baseload CRRs (source, sink, mw, crr_type)
    awarded = annual_result["awarded"]
    buy_awards = awarded[(awarded["side"] == "BUY") & (awarded["mw_awarded"] > 0.1)]
    baseload = list(zip(
        buy_awards["source"], buy_awards["sink"],
        buy_awards["mw_awarded"], buy_awards["crr_type"]
    ))

    if verbose:
        rf = mt["rating_factor"]
        print(f"\n{'='*65}")
        print(f"  MONTHLY SHORT-TERM CRR AUCTION -- {year}-{month:02d}")
        print(f"  Rating factor: {rf:.2f}x  |  SFT cap factor: {capacity_factor:.0%}")
        print(f"  Baseload CRRs (from Annual): {len(baseload)} positions, "
              f"{sum(b[2] for b in baseload):.0f} MW total")
        print(f"{'='*65}")
        _print_bid_summary(MONTHLY_BIDS_JAN, "Monthly CRR Bids Submitted")

    result = clear_auction(
        bids_df=MONTHLY_BIDS_JAN,
        ptdf=nm.ptdf,
        lines=mt["lines"],
        line_list=list(mt["lines"].keys()),
        bus_list=BUS_LIST,
        bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
        capacity_factor=capacity_factor,
        baseload_crrs=baseload,
    )

    if verbose:
        _print_auction_results(result, "MONTHLY AUCTION RESULTS", REF_LMP_JANUARY)

    return result


# -----------------------------------------------------------------------------
#  Hedge Value Analysis
# -----------------------------------------------------------------------------

def compute_hedge_value(result: dict, ref_lmps: dict) -> pd.DataFrame:
    """
    Compute ex-post hedge P&L for each awarded CRR position.

    CRR settlement:
      OBL: settlement = (LMP_sink - LMP_source) x MW_awarded
      OPT: settlement = max(LMP_sink - LMP_source, 0) x MW_awarded

    Hedge value = settlement - ACP x MW_awarded
      (positive means the CRR earned more than it cost)
    """
    awarded = result["awarded"]
    rows = []
    for _, row in awarded.iterrows():
        if row["mw_awarded"] < 0.01:
            continue
        src_lmp = ref_lmps.get(row["source"], 0.0)
        snk_lmp = ref_lmps.get(row["sink"], 0.0)
        dart = snk_lmp - src_lmp
        if row["crr_type"] == "OBL":
            settlement = dart * row["mw_awarded"]
        else:  # OPT
            settlement = max(dart, 0.0) * row["mw_awarded"]
        acp_cost = row["clearing_price"] * row["mw_awarded"]
        rows.append({
            "agent_id":    row["agent_id"],
            "path":        f"{row['source']} -> {row['sink']}",
            "crr_type":    row["crr_type"],
            "side":        row["side"],
            "mw_awarded":  row["mw_awarded"],
            "ref_dart":    dart,
            "settlement":  settlement,
            "acp":         row["clearing_price"],
            "acp_cost":    acp_cost,
            "net_value":   settlement - acp_cost,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
#  Print Utilities
# -----------------------------------------------------------------------------

def _print_bid_summary(bids_df: pd.DataFrame, title: str):
    print(f"\n  {title}")
    print(f"  {'Agent':<14} {'Side':>5} {'Type':>4} {'Source':<18} {'Sink':<16} "
          f"{'Bid $/MWh':>10} {'MW Bid':>8}")
    print("  " + "-" * 80)
    for _, r in bids_df.iterrows():
        print(f"  {r['agent_id']:<14} {r['side']:>5} {r['crr_type']:>4}  "
              f"{r['source']:<18} {r['sink']:<16} "
              f"{r['bid_price']:>9.1f}  {r['mw']:>7.0f}")
    buy_mw  = bids_df.loc[bids_df.side == "BUY",  "mw"].sum()
    sell_mw = bids_df.loc[bids_df.side == "SELL", "mw"].sum()
    print(f"\n  Total BUY: {buy_mw:.0f} MW  |  Total SELL: {sell_mw:.0f} MW")


def _print_auction_results(result: dict, title: str, ref_lmps: dict):
    if result["status"] != "optimal":
        print(f"\n  !! Auction status: {result['status']}")
        return

    awarded  = result["awarded"]
    summary  = result["summary"]
    binding  = result["binding_lines"]

    print(f"\n  {'-'*65}")
    print(f"  {title}")
    print(f"  Total welfare: ${result['total_welfare']:,.1f}  |  "
          f"Binding lines: {len(binding)}")

    # -- Awarded positions ----------------------------------------------------
    print(f"\n  Awarded Positions:")
    print(f"  {'Agent':<14} {'Side':>5} {'Type':>4} {'Source':<18} {'Sink':<16} "
          f"{'Awarded MW':>11} {'ACP $/MWh':>10}")
    print("  " + "-" * 82)
    for _, r in awarded.iterrows():
        if r["mw_awarded"] < 0.1:
            continue
        print(f"  {r['agent_id']:<14} {r['side']:>5} {r['crr_type']:>4}  "
              f"{r['source']:<18} {r['sink']:<16} "
              f"{r['mw_awarded']:>10.1f}  {r['clearing_price']:>9.2f}")

    buy_aw  = summary.get("buy_obl_mw",  0) + summary.get("buy_opt_mw",  0)
    sell_aw = summary.get("sell_obl_mw", 0) + summary.get("sell_opt_mw", 0)
    print(f"\n  Total BUY awarded: {buy_aw:.0f} MW  |  "
          f"Total SELL awarded: {sell_aw:.0f} MW")

    # -- Binding SFT lines ----------------------------------------------------
    if binding:
        print(f"\n  Binding SFT Lines (non-zero dual):")
        print(f"  {'Line':<30} {'Fwd dual':>11} {'Rev dual':>11} {'Net dual':>10}")
        print("  " + "-" * 65)
        for line in binding:
            d = result["line_duals"][line]
            print(f"  {line:<30} {d['forward']:>11.3f} {d['reverse']:>11.3f} "
                  f"{d['net']:>10.3f}")
    else:
        print("\n  No SFT lines binding (all bids within network capacity).")

    # -- Settlement point shadow prices ---------------------------------------
    print(f"\n  SP Shadow Prices ($/MWh):")
    interesting_sps = [
        "RN_PANHNDL", "HB_WEST", "LZ_WEST",
        "HB_NORTH", "LZ_NORTH",
        "HB_HOUSTON", "LZ_HOUSTON",
        "HB_SOUTH", "LZ_SOUTH",
        "RN_RGV_SOLAR", "HB_BUSAVG",
    ]
    print(f"  {'Settlement Point':<22} {'Shadow $':>10} {'Ref LMP':>10} "
          f"{'DART Proxy':>12}")
    print("  " + "-" * 58)
    for sp in interesting_sps:
        shadow = result["sp_shadow"].get(sp, 0.0)
        ref    = ref_lmps.get(sp, float("nan"))
        dart_proxy = ref - ref_lmps.get("HB_BUSAVG", 0)
        print(f"  {sp:<22} {shadow:>10.2f} {ref:>10.1f} {dart_proxy:>12.1f}")

    # -- Hedge value summary --------------------------------------------------
    hv = compute_hedge_value(result, ref_lmps)
    if not hv.empty:
        print(f"\n  Ex-Post Hedge Value (using reference LMPs):")
        print(f"  {'Agent':<14} {'Path':<38} {'Type':>4} {'MW':>6} "
              f"{'DART':>7} {'Settle $':>9} {'ACP $':>8} {'Net $':>8}")
        print("  " + "-" * 98)
        for _, r in hv.iterrows():
            print(f"  {r['agent_id']:<14} {r['path']:<38} {r['crr_type']:>4} "
                  f"{r['mw_awarded']:>6.0f} {r['ref_dart']:>7.1f} "
                  f"{r['settlement']:>9.1f} {r['acp_cost']:>8.1f} "
                  f"{r['net_value']:>8.1f}")
        by_agent = hv.groupby("agent_id")[["settlement", "acp_cost", "net_value"]].sum()
        print(f"\n  By Agent:")
        print(f"  {'Agent':<14} {'Settlement':>12} {'ACP Cost':>10} {'Net Value':>10}")
        print("  " + "-" * 50)
        for agent, row in by_agent.iterrows():
            print(f"  {agent:<14} {row['settlement']:>12.1f} {row['acp_cost']:>10.1f} "
                  f"{row['net_value']:>10.1f}")
        print(f"\n  TOTAL:          {by_agent['settlement'].sum():>12.1f} "
              f"{by_agent['acp_cost'].sum():>10.1f} "
              f"{by_agent['net_value'].sum():>10.1f}")


def _print_crr_economics_explainer():
    """Print a concise market design explainer."""
    sep = "  " + "-" * 69
    lines = [
        "",
        sep,
        "  CRR MARKET DESIGN -- KEY CONCEPTS",
        sep,
        "  CRR (Congestion Revenue Right): Financial instrument that pays/charges",
        "  the difference between two nodal prices.  Provides perfect hedge against",
        "  nodal congestion cost in the Day-Ahead Market.",
        "",
        "  OBL (Obligation): Settles at (LMP_sink - LMP_source) x MW.",
        "    -> Positive when sink > source (you receive money), negative otherwise.",
        "    -> Best for LSEs with predictable directional congestion exposure.",
        "",
        "  OPT (Option): Settles at max(LMP_sink - LMP_source, 0) x MW.",
        "    -> Never negative -- you can let the option expire when spread is adverse.",
        "    -> Best for generators in congested pockets (avoids reverse settlement).",
        "    -> ACP is typically higher than OBL ACP for the same path.",
        "",
        "  ACP (Auction Clearing Price): Shadow price from the SFT constraint dual.",
        "    -> ACP for OBL path (A->B) = shadow(B) - shadow(A)",
        "    -> ACP for OPT = positive-direction SFT dual contribution",
        "    -> Buyers pay ACP; sellers receive ACP.",
        "",
        "  SFT (Simultaneous Feasibility Test): All awarded CRRs must be jointly",
        "    feasible on the DC power flow network.  The LP finds the welfare-",
        "    maximizing allocation subject to forward/reverse line flow limits.",
        "",
        "  Seasonal Topology: ERCOT adjusts the CRR network model monthly for",
        "    line ratings (summer tighter, winter relaxed) and planned outages.",
        "    The Annual auction uses the summer peak model; monthly auctions use",
        "    the actual delivery-month model.",
        sep,
        "",
    ]
    print("\n".join(lines))


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------

def run_crr_auction_example(
    verbose: bool = True,
    save_path: str = "results/ercot_17bus/crr_auction.pkl",
) -> dict:
    """
    Run the full CRR auction example: Annual (summer) + Monthly (January).

    Parameters
    ----------
    save_path : File path for pickle output.  Pass None to skip saving.

    Returns dict with keys 'annual' and 'monthly'.
    """
    if verbose:
        print("\n" + "=" * 65)
        print("  CRR AUCTION EXAMPLE -- ERCOT 17-Bus System")
        print("  Annual (summer peak model) + Monthly (January residual)")
        print("=" * 65)
        _print_crr_economics_explainer()

    # Round 1: Annual CRR Auction
    annual = run_annual_crr_auction(
        year=2024, summer_month=7,
        capacity_factor=0.75,
        verbose=verbose,
    )

    # Round 2: Monthly Short-Term CRR (January, residual capacity)
    monthly = run_monthly_crr_auction(
        annual_result=annual,
        year=2024, month=1,
        capacity_factor=0.90,
        verbose=verbose,
    )

    if verbose:
        _print_auction_comparison(annual, monthly)

    results = {"annual": annual, "monthly": monthly}

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        if verbose:
            print(f"\n  Saved: {save_path}")

    return results


def _print_auction_comparison(annual: dict, monthly: dict):
    """Print side-by-side comparison of Annual vs Monthly results."""
    print(f"\n{'='*65}")
    print("  ANNUAL vs MONTHLY AUCTION COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Metric':<35} {'Annual':>12} {'Monthly':>12}")
    print("  " + "-" * 62)

    def _total_buy(r):
        s = r.get("summary", {})
        return s.get("buy_obl_mw", 0) + s.get("buy_opt_mw", 0)

    def _total_sell(r):
        s = r.get("summary", {})
        return s.get("sell_obl_mw", 0) + s.get("sell_opt_mw", 0)

    ann_binding  = annual.get("binding_lines",  [])
    mon_binding  = monthly.get("binding_lines", [])
    ann_welfare  = annual.get("total_welfare",  0)
    mon_welfare  = monthly.get("total_welfare", 0)

    rows = [
        ("Total BUY awarded (MW)",       _total_buy(annual),    _total_buy(monthly)),
        ("Total SELL awarded (MW)",       _total_sell(annual),   _total_sell(monthly)),
        ("Binding SFT lines",            len(ann_binding),      len(mon_binding)),
        ("Total welfare ($)",            ann_welfare,            mon_welfare),
    ]
    for label, av, mv in rows:
        fmt = "{:>12.0f}" if isinstance(av, float) else "{:>12}"
        print(f"  {label:<35} {fmt.format(av)} {fmt.format(mv)}")

    print(f"\n  Annual binding: {ann_binding}")
    print(f"  Monthly binding: {mon_binding}")
    print()

    # LZ_HOUSTON ACP comparison
    def _lz_hou_acp(result):
        awarded = result.get("awarded", pd.DataFrame())
        if awarded.empty:
            return float("nan")
        mask = (awarded["sink"] == "LZ_HOUSTON") & (awarded["side"] == "BUY")
        sub  = awarded[mask]
        if sub.empty:
            return float("nan")
        return sub["clearing_price"].mean()

    ann_acp = _lz_hou_acp(annual)
    mon_acp = _lz_hou_acp(monthly)
    print(f"  LZ_HOUSTON avg buy ACP  Annual: ${ann_acp:.2f}/MWh  "
          f"Monthly: ${mon_acp:.2f}/MWh")
    if mon_acp > ann_acp:
        print(f"  Monthly ACP > Annual: Annual awards consumed most SFT; residual")
        print(f"  capacity is scarce even with wider winter ratings (1.20x).")
    else:
        print(f"  Annual ACP > Monthly: Summer ratings (0.80x) dominate over the")
        print(f"  relaxed winter ratings available in the monthly auction.")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_crr_auction_example(verbose=True)
