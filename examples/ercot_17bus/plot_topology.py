"""Generate topology visualizations for the ERCOT 17-bus example."""

from __future__ import annotations

import argparse

from viz.topology import plot_topology

from examples.ercot_17bus.topology import (
    BATTERIES,
    BUSES,
    BUS_LOAD_FRACTION,
    GENERATORS,
    GENERIC_TRANSMISSION_CONSTRAINTS,
    LINES,
    SETTLEMENT_POINTS,
    get_monthly_topology,
)


def plot_ercot_17bus_topology(
    year: int | None = None,
    month: int | None = None,
    output_path: str | None = "results/ercot_17bus/topology_map.html",
    show: bool = False,
):
    """Plot the ERCOT 17-bus topology in base or monthly-overlay mode."""
    if year is None or month is None:
        return plot_topology(
            buses=BUSES,
            lines=LINES,
            generators=GENERATORS,
            batteries=BATTERIES,
            settlement_points=SETTLEMENT_POINTS,
            gtcs=GENERIC_TRANSMISSION_CONSTRAINTS,
            bus_load_fraction=BUS_LOAD_FRACTION,
            title="ERCOT 17-Bus Base Topology",
            output_path=output_path,
            show=show,
        )

    monthly = get_monthly_topology(year, month)
    title = (
        f"ERCOT 17-Bus Topology {year}-{month:02d} "
        f"(rating {monthly['rating_factor']:.2f}x)"
    )
    if monthly["outaged_lines"] or monthly["outaged_generators"]:
        title += (
            f" | outaged lines: {len(monthly['outaged_lines'])}, "
            f"outaged generators: {len(monthly['outaged_generators'])}"
        )

    return plot_topology(
        buses=BUSES,
        lines=monthly["lines"],
        generators=monthly["generators"],
        batteries=BATTERIES,
        settlement_points=SETTLEMENT_POINTS,
        gtcs=monthly["gtcs"],
        bus_load_fraction=BUS_LOAD_FRACTION,
        title=title,
        output_path=output_path,
        show=show,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the ERCOT 17-bus topology.")
    parser.add_argument("--year", type=int, default=None, help="Monthly topology year.")
    parser.add_argument("--month", type=int, default=None, help="Monthly topology month.")
    parser.add_argument(
        "--output",
        default="results/ercot_17bus/topology_map.html",
        help="HTML output path. Use an empty string to skip writing.",
    )
    parser.add_argument("--show", action="store_true", help="Display the Plotly figure.")
    args = parser.parse_args()

    output_path = args.output or None
    plot_ercot_17bus_topology(
        year=args.year,
        month=args.month,
        output_path=output_path,
        show=args.show,
    )
    if output_path:
        print(f"Wrote topology map: {output_path}")


if __name__ == "__main__":
    main()
