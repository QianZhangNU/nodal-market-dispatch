"""Generic topology plotting utilities.

The plotting functions in this module intentionally depend only on generic
dictionary-shaped topology data:
  - buses: bus_id -> {lat, lng, ...}
  - lines: line_id -> {from_bus, to_bus, ...}
  - generators/batteries: resource_id -> {bus, ...}
  - gtcs: gtc_id -> {monitored_lines=[(line_id, sign), ...], ...}
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import warnings

import plotly.graph_objects as go


_REGION_COLORS = [
    "#1f77b4",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#ff7f0e",
]

_KV_STYLES = {
    345: {"color": "#4b5563", "width": 3.2},
    230: {"color": "#6b7280", "width": 2.6},
    138: {"color": "#9ca3af", "width": 1.8},
}

_GTC_COLORS = [
    "#ef4444",
    "#2563eb",
    "#16a34a",
    "#f97316",
    "#7c3aed",
    "#0891b2",
]


def plot_topology(
    buses: dict,
    lines: dict,
    generators: dict | None = None,
    batteries: dict | None = None,
    settlement_points: dict | None = None,
    gtcs: dict | None = None,
    bus_load_fraction: dict | None = None,
    title: str = "Topology Map",
    output_path: str | None = None,
    show: bool = False,
):
    """Plot a topology using bus latitude/longitude coordinates.

    Parameters are intentionally generic so this can be reused by any example
    system that follows the common dictionary schema.
    """
    generators = generators or {}
    batteries = batteries or {}
    settlement_points = settlement_points or {}
    gtcs = gtcs or {}
    bus_load_fraction = bus_load_fraction or {}

    valid_buses = _valid_buses_with_coordinates(buses)
    if not valid_buses:
        raise ValueError("No buses with valid 'lat' and 'lng' coordinates were found.")

    gen_by_bus, gen_capacity_by_bus_type = _group_generators_by_bus(generators)
    battery_by_bus = _group_by_bus(batteries)
    sp_by_bus = _group_settlement_points_by_bus(settlement_points)

    fig = go.Figure()

    _add_line_traces(fig, valid_buses, lines)
    _add_gtc_traces(fig, valid_buses, lines, gtcs)
    _add_bus_traces(
        fig,
        valid_buses,
        gen_by_bus,
        gen_capacity_by_bus_type,
        battery_by_bus,
        sp_by_bus,
        bus_load_fraction,
    )

    lats = [float(bd["lat"]) for bd in valid_buses.values()]
    lngs = [float(bd["lng"]) for bd in valid_buses.values()]
    lat_pad = max(0.5, (max(lats) - min(lats)) * 0.12)
    lng_pad = max(0.5, (max(lngs) - min(lngs)) * 0.12)

    fig.update_layout(
        title=title,
        hovermode="closest",
        legend=dict(
            title="Layers",
            orientation="v",
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(l=20, r=230, t=60, b=20),
        plot_bgcolor="#f8fafc",
        xaxis=dict(
            title="Longitude",
            range=[min(lngs) - lng_pad, max(lngs) + lng_pad],
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(
            title="Latitude",
            range=[min(lats) - lat_pad, max(lats) + lat_pad],
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
    )

    if output_path is not None:
        path = Path(output_path)
        if path.parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path), include_plotlyjs=True)

    if show:
        fig.show()

    return fig


def _valid_buses_with_coordinates(buses: dict) -> dict:
    valid = {}
    for bus_id, bus_data in buses.items():
        lat = bus_data.get("lat")
        lng = bus_data.get("lng")
        if lat is None or lng is None:
            warnings.warn(f"Skipping bus {bus_id!r}: missing 'lat' or 'lng'.", stacklevel=2)
            continue
        try:
            lat_f = float(lat)
            lng_f = float(lng)
        except (TypeError, ValueError):
            warnings.warn(f"Skipping bus {bus_id!r}: invalid 'lat' or 'lng'.", stacklevel=2)
            continue
        if math.isnan(lat_f) or math.isnan(lng_f):
            warnings.warn(f"Skipping bus {bus_id!r}: NaN 'lat' or 'lng'.", stacklevel=2)
            continue
        valid[bus_id] = {**bus_data, "lat": lat_f, "lng": lng_f}
    return valid


def _group_by_bus(items: dict) -> dict:
    grouped = defaultdict(list)
    for item_id, item_data in items.items():
        bus = item_data.get("bus")
        if bus is not None:
            grouped[bus].append((item_id, item_data))
    return grouped


def _group_generators_by_bus(generators: dict) -> tuple[dict, dict]:
    gen_by_bus = _group_by_bus(generators)
    capacity_by_bus_type = defaultdict(lambda: defaultdict(float))
    for bus, entries in gen_by_bus.items():
        for _, gen_data in entries:
            gen_type = str(gen_data.get("type", "UNKNOWN"))
            capacity_by_bus_type[bus][gen_type] += float(gen_data.get("Pmax", 0.0) or 0.0)
    return gen_by_bus, capacity_by_bus_type


def _group_settlement_points_by_bus(settlement_points: dict) -> dict:
    grouped = defaultdict(list)
    for sp_id, sp_data in settlement_points.items():
        direct_bus = sp_data.get("bus")
        if direct_bus is not None:
            grouped[direct_bus].append(sp_id)
            continue

        member_buses = sp_data.get("member_buses")
        if isinstance(member_buses, dict):
            for bus, weight in member_buses.items():
                grouped[bus].append(f"{sp_id} ({weight:.2f})")
        elif isinstance(member_buses, (list, tuple, set)):
            for bus in member_buses:
                grouped[bus].append(sp_id)
    return grouped


def _add_line_traces(fig: go.Figure, buses: dict, lines: dict) -> None:
    shown_kv = set()
    for line_id, line_data in lines.items():
        from_bus = line_data.get("from_bus")
        to_bus = line_data.get("to_bus")
        if from_bus not in buses or to_bus not in buses:
            warnings.warn(
                f"Skipping line {line_id!r}: endpoint bus is missing or has no coordinates.",
                stacklevel=2,
            )
            continue

        kv = line_data.get("kV", "unknown")
        style = _line_style(kv)
        showlegend = kv not in shown_kv
        shown_kv.add(kv)

        fig.add_trace(
            go.Scatter(
                x=[buses[from_bus]["lng"], buses[to_bus]["lng"]],
                y=[buses[from_bus]["lat"], buses[to_bus]["lat"]],
                mode="lines",
                line=style,
                name=f"{kv} kV lines",
                legendgroup=f"line-{kv}",
                showlegend=showlegend,
                hoverinfo="text",
                text=_line_hover(line_id, line_data, from_bus, to_bus),
                opacity=0.85,
            )
        )


def _add_gtc_traces(fig: go.Figure, buses: dict, lines: dict, gtcs: dict) -> None:
    for idx, (gtc_id, gtc_data) in enumerate(gtcs.items()):
        color = _GTC_COLORS[idx % len(_GTC_COLORS)]
        lon = []
        lat = []
        text = []
        monitored = gtc_data.get("monitored_lines", [])
        for monitored_item in monitored:
            line_id = monitored_item[0] if isinstance(monitored_item, (list, tuple)) else monitored_item
            sign = monitored_item[1] if isinstance(monitored_item, (list, tuple)) and len(monitored_item) > 1 else 1
            line_data = lines.get(line_id)
            if not line_data:
                continue
            from_bus = line_data.get("from_bus")
            to_bus = line_data.get("to_bus")
            if from_bus not in buses or to_bus not in buses:
                continue
            hover = _gtc_hover(gtc_id, gtc_data, line_id, sign)
            lon.extend([buses[from_bus]["lng"], buses[to_bus]["lng"], None])
            lat.extend([buses[from_bus]["lat"], buses[to_bus]["lat"], None])
            text.extend([hover, hover, None])

        if lon:
            fig.add_trace(
            go.Scatter(
                    x=lon,
                    y=lat,
                    mode="lines",
                    line=dict(color=color, width=5, dash="dot"),
                    name=f"GTC: {gtc_id}",
                    legendgroup=f"gtc-{gtc_id}",
                    hoverinfo="text",
                    text=text,
                    opacity=0.8,
                )
            )


def _add_bus_traces(
    fig: go.Figure,
    buses: dict,
    gen_by_bus: dict,
    gen_capacity_by_bus_type: dict,
    battery_by_bus: dict,
    sp_by_bus: dict,
    bus_load_fraction: dict,
) -> None:
    region_to_buses = defaultdict(list)
    for bus_id, bus_data in buses.items():
        region_to_buses[str(bus_data.get("region", "Unknown"))].append((bus_id, bus_data))

    max_capacity = max(
        [sum(cap_by_type.values()) for cap_by_type in gen_capacity_by_bus_type.values()] + [0.0]
    )
    max_load_fraction = max([float(v) for v in bus_load_fraction.values()] + [0.0])
    region_colors = {
        region: _REGION_COLORS[i % len(_REGION_COLORS)]
        for i, region in enumerate(sorted(region_to_buses))
    }

    for region, entries in sorted(region_to_buses.items()):
        bus_ids = [bus_id for bus_id, _ in entries]
        fig.add_trace(
            go.Scatter(
                x=[bus_data["lng"] for _, bus_data in entries],
                y=[bus_data["lat"] for _, bus_data in entries],
                mode="markers+text",
                text=[str(bus_id) for bus_id in bus_ids],
                textposition="top center",
                name=f"Bus: {region}",
                legendgroup=f"bus-{region}",
                hoverinfo="text",
                hovertext=[
                    _bus_hover(
                        bus_id,
                        buses[bus_id],
                        gen_by_bus.get(bus_id, []),
                        gen_capacity_by_bus_type.get(bus_id, {}),
                        battery_by_bus.get(bus_id, []),
                        sp_by_bus.get(bus_id, []),
                        bus_load_fraction.get(bus_id),
                    )
                    for bus_id in bus_ids
                ],
                marker=dict(
                    color=region_colors[region],
                    size=[
                        _bus_marker_size(
                            sum(gen_capacity_by_bus_type.get(bus_id, {}).values()),
                            bus_load_fraction.get(bus_id),
                            max_capacity,
                            max_load_fraction,
                        )
                        for bus_id in bus_ids
                    ],
                    line=dict(color="#111827", width=1),
                    opacity=0.95,
                ),
            )
        )


def _line_style(kv) -> dict:
    try:
        kv_int = int(kv)
    except (TypeError, ValueError):
        kv_int = None
    if kv_int in _KV_STYLES:
        return _KV_STYLES[kv_int]
    return {"color": "#64748b", "width": 2.0}


def _bus_marker_size(capacity: float, load_fraction, max_capacity: float, max_load_fraction: float) -> float:
    gen_component = 0.0 if max_capacity <= 0 else 16.0 * math.sqrt(max(capacity, 0.0) / max_capacity)
    load_value = float(load_fraction or 0.0)
    load_component = 0.0 if max_load_fraction <= 0 else 10.0 * math.sqrt(max(load_value, 0.0) / max_load_fraction)
    return 10.0 + gen_component + load_component


def _line_hover(line_id, line_data, from_bus, to_bus) -> str:
    parts = [
        f"<b>{line_id}</b>",
        f"From/To: {from_bus} -> {to_bus}",
        f"kV: {_format_optional(line_data.get('kV'))}",
        f"x_pu: {_format_optional(line_data.get('x_pu'))}",
        f"flow_limit: {_format_optional(line_data.get('flow_limit'))} MW",
        f"contingency_limit: {_format_optional(line_data.get('contingency_limit'))} MW",
    ]
    if "ocost" in line_data:
        parts.append(f"ocost: {_format_optional(line_data.get('ocost'))}")
    if line_data.get("description"):
        parts.append(str(line_data["description"]))
    return "<br>".join(parts)


def _gtc_hover(gtc_id, gtc_data, line_id, sign) -> str:
    parts = [
        f"<b>{gtc_id}</b>",
        f"Highlighted line: {line_id}",
        f"Direction sign: {sign:+}",
        f"flow_limit: {_format_optional(gtc_data.get('flow_limit'))} MW",
    ]
    if "contingency_limit" in gtc_data:
        parts.append(f"contingency_limit: {_format_optional(gtc_data.get('contingency_limit'))} MW")
    monitored = gtc_data.get("monitored_lines")
    if monitored:
        parts.append(f"monitored_lines: {_format_monitored_lines(monitored)}")
    if gtc_data.get("description"):
        parts.append(str(gtc_data["description"]))
    return "<br>".join(parts)


def _bus_hover(
    bus_id,
    bus_data,
    generators,
    gen_capacity_by_type,
    batteries,
    settlement_points,
    load_fraction,
) -> str:
    parts = [
        f"<b>Bus {bus_id}: {bus_data.get('name', '')}</b>",
        f"Region: {bus_data.get('region', 'Unknown')}",
        f"kV: {_format_optional(bus_data.get('kV'))}",
        f"Coordinates: {bus_data.get('lat'):.4f}, {bus_data.get('lng'):.4f}",
    ]
    if load_fraction is not None:
        parts.append(f"Load fraction: {float(load_fraction):.3f}")

    if gen_capacity_by_type:
        gen_summary = ", ".join(
            f"{gen_type}: {capacity:.0f} MW"
            for gen_type, capacity in sorted(gen_capacity_by_type.items())
        )
        parts.append(f"Generation by type: {gen_summary}")
        parts.append("Generators: " + ", ".join(gen_id for gen_id, _ in generators))
    else:
        parts.append("Generation: none")

    if batteries:
        battery_summary = ", ".join(
            f"{bat_id} ({bat_data.get('power_mw', '?')} MW)"
            for bat_id, bat_data in batteries
        )
        parts.append(f"Batteries: {battery_summary}")

    if settlement_points:
        parts.append("Settlement points: " + ", ".join(str(sp) for sp in settlement_points))

    return "<br>".join(parts)


def _format_optional(value) -> str:
    return "n/a" if value is None else str(value)


def _format_monitored_lines(monitored_lines) -> str:
    pieces = []
    for item in monitored_lines:
        if isinstance(item, (list, tuple)) and len(item) > 1:
            pieces.append(f"{item[0]} ({item[1]:+})")
        else:
            pieces.append(str(item))
    return ", ".join(pieces)
