"""Plot load, renewable, and gas price assumptions for the ERCOT 17-bus example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from examples.ercot_17bus.profiles import build_all_profiles
from examples.ercot_17bus.topology import BUSES, GENERATORS


SEASONAL_WEEKS = {
    "Winter": "2023-01-16",
    "Spring": "2023-04-17",
    "Summer": "2023-08-14",
    "Fall": "2023-10-16",
}


def _generator_capacity_by_type_region():
    capacity = {}
    for gd in GENERATORS.values():
        key = (gd["type"], gd["region"])
        capacity[key] = capacity.get(key, 0.0) + float(gd["Pmax"])
    return capacity


def _load_by_region(load_df):
    region_load = {}
    for bus, bd in BUSES.items():
        region = bd.get("region", "Unknown")
        if bus in load_df.columns:
            if region not in region_load:
                region_load[region] = load_df[bus].copy()
            else:
                region_load[region] += load_df[bus]
    return region_load


def _renewable_available_mw(profiles):
    pmax_t = profiles["pmax_t"]
    wind = pmax_t[[g for g, gd in GENERATORS.items() if gd["type"] == "WIND"]]
    solar = pmax_t[[g for g, gd in GENERATORS.items() if gd["type"] == "SOLAR"]]
    return wind.sum(axis=1), solar.sum(axis=1)


def _slice_week(series_or_df, start_date: str):
    start = series_or_df.index.normalize()[0] if start_date is None else None
    start = start or series_or_df.index.__class__([start_date])[0]
    end = start + pd.Timedelta(days=7) - pd.Timedelta(hours=1)
    return series_or_df.loc[start:end]


def _add_full_horizon_traces(fig, profiles, row_offset=0):
    load = profiles["load"]
    wind = profiles["wind"]
    solar = profiles["solar"]
    gas_price = profiles["gas_price"]
    total_load = load.sum(axis=1)
    wind_mw, solar_mw = _renewable_available_mw(profiles)
    region_load = _load_by_region(load)
    capacity = _generator_capacity_by_type_region()

    _add_profile_traces(
        fig=fig,
        total_load=total_load,
        wind_mw=wind_mw,
        solar_mw=solar_mw,
        region_load=region_load,
        wind=wind,
        solar=solar,
        gas_price=gas_price,
        capacity=capacity,
        col=1,
        showlegend=True,
    )


def _add_profile_traces(
    fig,
    total_load,
    wind_mw,
    solar_mw,
    region_load,
    wind,
    solar,
    gas_price,
    capacity,
    col,
    showlegend,
):
    fig.add_trace(
        go.Scatter(
            x=total_load.index,
            y=total_load,
            name="System load",
            mode="lines",
            line=dict(color="#111827", width=1.4),
            hovertemplate="%{x}<br>Load: %{y:.0f} MW<extra></extra>",
            showlegend=showlegend,
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=wind_mw.index,
            y=wind_mw,
            name="Wind available",
            mode="lines",
            line=dict(color="#2563eb", width=1.0),
            hovertemplate="%{x}<br>Wind available: %{y:.0f} MW<extra></extra>",
            showlegend=showlegend,
        ),
        row=1,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=solar_mw.index,
            y=solar_mw,
            name="Solar available",
            mode="lines",
            line=dict(color="#f59e0b", width=1.0),
            hovertemplate="%{x}<br>Solar available: %{y:.0f} MW<extra></extra>",
            showlegend=showlegend,
        ),
        row=1,
        col=col,
    )

    for region, series in sorted(region_load.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=f"Load {region}",
                mode="lines",
                line=dict(width=1.0),
                hovertemplate=f"{region}<br>%{{x}}<br>Load: %{{y:.0f}} MW<extra></extra>",
                showlegend=showlegend,
            ),
            row=2,
            col=col,
        )

    for region in wind.columns:
        cap = sum(
            pmax
            for (gen_type, gen_region), pmax in capacity.items()
            if gen_type == "WIND" and gen_region == region
        )
        fig.add_trace(
            go.Scatter(
                x=wind.index,
                y=wind[region],
                name=f"Wind CF {region}",
                mode="lines",
                line=dict(width=1.1),
                hovertemplate=(
                    f"{region}<br>%{{x}}<br>CF: %{{y:.2%}}"
                    f"<br>Installed wind: {cap:.0f} MW<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=3,
            col=col,
        )

    for region in solar.columns:
        cap = sum(
            pmax
            for (gen_type, gen_region), pmax in capacity.items()
            if gen_type == "SOLAR"
            and (gen_region == region or (gen_region == "South_RGV" and region == "South"))
        )
        fig.add_trace(
            go.Scatter(
                x=solar.index,
                y=solar[region],
                name=f"Solar CF {region}",
                mode="lines",
                line=dict(width=1.1),
                hovertemplate=(
                    f"{region}<br>%{{x}}<br>CF: %{{y:.2%}}"
                    f"<br>Mapped solar: {cap:.0f} MW<extra></extra>"
                ),
                showlegend=showlegend,
            ),
            row=4,
            col=col,
        )

    fig.add_trace(
        go.Scatter(
            x=gas_price.index,
            y=gas_price,
            name="Henry Hub gas",
            mode="lines",
            line=dict(color="#7c2d12", width=1.4),
            hovertemplate="%{x}<br>Gas: $%{y:.2f}/MMBtu<extra></extra>",
            showlegend=showlegend,
        ),
        row=5,
        col=col,
    )


def plot_ercot_17bus_profile_weeks(
    output_path: str | None = "results/ercot_17bus/profile_assumptions_weeks.html",
    show: bool = False,
    seed: int = 42,
):
    """Plot one representative week for each season."""
    profiles = build_all_profiles(seed=seed)
    total_load = profiles["load"].sum(axis=1)
    wind_mw, solar_mw = _renewable_available_mw(profiles)
    capacity = _generator_capacity_by_type_region()

    fig = make_subplots(
        rows=5,
        cols=4,
        shared_xaxes=False,
        shared_yaxes="rows",
        vertical_spacing=0.06,
        horizontal_spacing=0.035,
        subplot_titles=[
            f"{season} week<br>{start}"
            for season, start in SEASONAL_WEEKS.items()
        ],
    )

    for col, (season, start) in enumerate(SEASONAL_WEEKS.items(), start=1):
        week_load = _slice_week(profiles["load"], start)
        week_profiles = {
            "load": week_load,
            "wind": _slice_week(profiles["wind"], start),
            "solar": _slice_week(profiles["solar"], start),
            "gas_price": _slice_week(profiles["gas_price"], start),
        }
        week_region_load = _load_by_region(week_load)
        _add_profile_traces(
            fig=fig,
            total_load=_slice_week(total_load, start),
            wind_mw=_slice_week(wind_mw, start),
            solar_mw=_slice_week(solar_mw, start),
            region_load=week_region_load,
            wind=week_profiles["wind"],
            solar=week_profiles["solar"],
            gas_price=week_profiles["gas_price"],
            capacity=capacity,
            col=col,
            showlegend=(col == 1),
        )

    for col in range(1, 5):
        fig.update_xaxes(title_text="Hour", row=5, col=col)
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="CF", tickformat=".0%", range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text="CF", tickformat=".0%", range=[0, 1], row=4, col=1)
    fig.update_yaxes(title_text="$/MMBtu", row=5, col=1)

    row_labels = [
        "System Load and Renewable MW",
        "Load by Region",
        "Wind Capacity Factor",
        "Solar Capacity Factor",
        "Gas Price",
    ]
    for i, label in enumerate(row_labels, start=1):
        fig.add_annotation(
            text=label,
            x=0,
            y=1 - (i - 0.5) / 5,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            textangle=-90,
            font=dict(size=12),
        )

    fig.update_layout(
        title="ERCOT 17-Bus Seasonal Typical-Week Profile Assumptions",
        template="plotly_white",
        height=1180,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.035, xanchor="left", x=0),
        margin=dict(l=90, r=35, t=120, b=50),
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)

    if show:
        fig.show()

    return fig


def plot_ercot_17bus_profiles(
    output_path: str | None = "results/ercot_17bus/profile_assumptions.html",
    show: bool = False,
    seed: int = 42,
):
    """Build an interactive HTML profile report and return the Plotly figure."""
    profiles = build_all_profiles(seed=seed)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=(
            "System Load and Renewable Availability",
            "Load by Region",
            "Wind Capacity Factor by Region",
            "Solar Capacity Factor by Region",
            "Gas Price Assumption",
        ),
    )
    _add_full_horizon_traces(fig, profiles)

    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="Capacity factor", tickformat=".0%", range=[0, 1], row=3, col=1)
    fig.update_yaxes(title_text="Capacity factor", tickformat=".0%", range=[0, 1], row=4, col=1)
    fig.update_yaxes(title_text="$/MMBtu", row=5, col=1)
    fig.update_xaxes(title_text="Hour", row=5, col=1)

    fig.update_layout(
        title="ERCOT 17-Bus Profile Assumptions",
        template="plotly_white",
        height=1150,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=70, r=40, t=115, b=50),
    )

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True, full_html=True)

    if show:
        fig.show()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ERCOT 17-bus profile assumptions.")
    parser.add_argument(
        "--output",
        default="results/ercot_17bus/profile_assumptions_weeks.html",
        help="HTML output path. Use an empty string to skip writing.",
    )
    parser.add_argument(
        "--view",
        choices=["weeks", "full"],
        default="weeks",
        help="Plot seasonal representative weeks or the full profile horizon.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Profile generation seed.")
    parser.add_argument("--show", action="store_true", help="Display the Plotly figure.")
    args = parser.parse_args()

    output_path = args.output or None
    if args.view == "full":
        plot_ercot_17bus_profiles(output_path=output_path, show=args.show, seed=args.seed)
    else:
        plot_ercot_17bus_profile_weeks(output_path=output_path, show=args.show, seed=args.seed)
    if output_path:
        print(f"Wrote profile assumptions plot: {output_path}")


if __name__ == "__main__":
    main()
