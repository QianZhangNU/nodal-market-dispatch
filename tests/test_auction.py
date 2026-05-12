"""CRR auction clearing tests."""

import pandas as pd


def _three_bus_auction_case():
    from solvers.network import NetworkModel

    model = NetworkModel(
        buses={1: {"name": "North"}, 2: {"name": "South"}, 3: {"name": "West"}},
        lines={
            "L_NS": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10,
                     "flow_limit": 100, "contingency_limit": 80},
            "L_NW": {"from_bus": 1, "to_bus": 3, "x_pu": 0.15, "b_pu": 6.67,
                     "flow_limit": 80, "contingency_limit": 60},
            "L_WS": {"from_bus": 3, "to_bus": 2, "x_pu": 0.2, "b_pu": 5,
                     "flow_limit": 60, "contingency_limit": 50},
        },
        slack_bus=1,
    )
    settlement_points = {
        "HB_NORTH": dict(type="hub", member_buses=[1]),
        "HB_SOUTH": dict(type="hub", member_buses=[2]),
        "LZ_WEST": dict(type="resource_node", bus=3),
    }
    return model, settlement_points


def test_auction_3bus_buy_obl_and_opt():
    from solvers.auction import clear_auction

    model, settlement_points = _three_bus_auction_case()
    bids = pd.DataFrame([
        {"agent_id": "A1", "source": "HB_NORTH", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 5.0, "mw": 50},
        {"agent_id": "A2", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OPT", "bid_price": 3.0, "mw": 30},
        {"agent_id": "A3", "source": "LZ_WEST", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 4.0, "mw": 40},
    ])

    result = clear_auction(
        bids_df=bids,
        ptdf=model.ptdf,
        lines=model.lines,
        line_list=model.line_list,
        bus_list=model.bus_list,
        bus_idx=model.bus_idx,
        settlement_points=settlement_points,
        capacity_factor=0.90,
    )

    assert result["status"] == "optimal"
    assert len(result["awarded"]) == 3
    assert result["awarded"]["mw_awarded"].sum() > 0
    assert result["summary"]["buy_obl_mw"] > 0
    assert result["summary"]["buy_opt_mw"] > 0


def test_auction_buy_sell_obl_opt_mix():
    from solvers.auction import clear_auction

    model, settlement_points = _three_bus_auction_case()
    bids = pd.DataFrame([
        {"agent_id": "B1", "source": "HB_NORTH", "sink": "HB_SOUTH",
         "side": "BUY", "crr_type": "OBL", "bid_price": 5.0, "mw": 80},
        {"agent_id": "B2", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OPT", "bid_price": 2.0, "mw": 50},
        {"agent_id": "S1", "source": "HB_SOUTH", "sink": "HB_NORTH",
         "side": "SELL", "crr_type": "OBL", "bid_price": 1.0, "mw": 30},
        {"agent_id": "S2", "source": "LZ_WEST", "sink": "HB_SOUTH",
         "side": "SELL", "crr_type": "OPT", "bid_price": 0.5, "mw": 20},
    ])

    result = clear_auction(
        bids_df=bids,
        ptdf=model.ptdf,
        lines=model.lines,
        line_list=model.line_list,
        bus_list=model.bus_list,
        bus_idx=model.bus_idx,
        settlement_points=settlement_points,
        capacity_factor=0.90,
    )

    assert result["status"] == "optimal"
    assert result["summary"]["buy_obl_mw"] > 0
    assert result["summary"]["sell_obl_mw"] >= 0
    assert result["total_welfare"] >= 0
    for key, acp in result["acp_by_path"].items():
        if key[2] == "OPT":
            assert acp >= -0.01


def test_auction_ercot_17bus_all_bid_types_and_backward_compatibility():
    from solvers.auction import clear_auction
    from solvers.network import NetworkModel
    from examples.ercot_17bus.topology import (
        BUSES, BUS_LIST, BUS_IDX, LINES, LINE_LIST, SETTLEMENT_POINTS, SLACK_BUS,
    )

    network = NetworkModel(buses=BUSES, lines=LINES, slack_bus=SLACK_BUS)
    bids = pd.DataFrame([
        {"agent_id": "T1", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OBL", "bid_price": 8.0, "mw": 100},
        {"agent_id": "T2", "source": "HB_NORTH", "sink": "LZ_HOUSTON",
         "side": "BUY", "crr_type": "OPT", "bid_price": 5.0, "mw": 150},
        {"agent_id": "T3", "source": "HB_WEST", "sink": "LZ_WEST",
         "side": "BUY", "crr_type": "OBL", "bid_price": 6.0, "mw": 80},
        {"agent_id": "T4", "source": "HB_SOUTH", "sink": "LZ_SOUTH",
         "side": "SELL", "crr_type": "OBL", "bid_price": 2.0, "mw": 60},
        {"agent_id": "T5", "source": "RN_RGV_SOLAR", "sink": "HB_HOUSTON",
         "side": "BUY", "crr_type": "OPT", "bid_price": 7.0, "mw": 120},
    ])

    result = clear_auction(
        bids_df=bids,
        ptdf=network.ptdf,
        lines=LINES,
        line_list=LINE_LIST,
        bus_list=BUS_LIST,
        bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
        capacity_factor=0.90,
        baseload_crrs=[("HB_NORTH", "LZ_HOUSTON", 50, "OBL")],
    )

    assert result["status"] == "optimal"
    assert result["awarded"]["mw_awarded"].sum() > 0
    assert len(result["acp_by_path"]) > 0

    old_bids = pd.DataFrame([
        {"agent_id": "X1", "source": "HB_NORTH", "sink": "LZ_WEST",
         "side": "BUY", "bid_price": 5.0, "mw": 50},
    ])
    backward_compat = clear_auction(
        bids_df=old_bids,
        ptdf=network.ptdf,
        lines=LINES,
        line_list=LINE_LIST,
        bus_list=BUS_LIST,
        bus_idx=BUS_IDX,
        settlement_points=SETTLEMENT_POINTS,
    )
    assert backward_compat["status"] == "optimal"
