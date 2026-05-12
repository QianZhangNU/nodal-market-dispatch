"""Network model tests, including a hand-checkable 3-bus radial case."""

import numpy as np

from solvers.network import NetworkModel


def test_network_3bus_radial_hand_calculated_ptdf_and_flows():
    """Radial 1-2-3 system with slack at bus 1.

    Line orientations are 1->2 and 2->3. By inspection:
      - 1 MW injected at bus 2 and withdrawn at slack flows -1 MW on L12.
      - 1 MW injected at bus 3 and withdrawn at slack flows -1 MW on L12
        and -1 MW on L23.
      - net injections [100, -40, -60] produce flows [100, 60].
    """
    model = NetworkModel(
        buses={1: {"name": "Slack"}, 2: {"name": "Mid"}, 3: {"name": "End"}},
        lines={
            "L12": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10.0,
                    "flow_limit": 100, "contingency_limit": 80},
            "L23": {"from_bus": 2, "to_bus": 3, "x_pu": 0.2, "b_pu": 5.0,
                    "flow_limit": 80, "contingency_limit": 60},
        },
        slack_bus=1,
    )

    expected_ptdf = np.array([
        [0.0, -1.0, -1.0],
        [0.0,  0.0, -1.0],
    ])
    np.testing.assert_allclose(model.ptdf, expected_ptdf, atol=1e-10)

    flows = model.line_flow(np.array([100.0, -40.0, -60.0]))
    np.testing.assert_allclose(flows, np.array([100.0, 60.0]), atol=1e-10)


def test_network_3bus_loop_hand_calculated_ptdf_and_flows():
    """Triangle 1->2, 2->3, 3->1 with equal reactances and slack at bus 1.

    For 1 MW injected at bus 2 and withdrawn at bus 1, two parallel paths carry
    power back to bus 1:
      - direct path 2->1 through L12 has 2/3 MW opposite the 1->2 orientation
      - indirect path 2->3->1 has 1/3 MW along L23 and L31

    For 1 MW injected at bus 3 and withdrawn at bus 1, symmetry gives:
      - L12: -1/3, L23: -1/3, L31: +2/3

    Therefore, with bus order [1, 2, 3] and line order [L12, L23, L31]:
      PTDF = [[0, -2/3, -1/3],
              [0,  1/3, -1/3],
              [0,  1/3,  2/3]]
    """
    model = NetworkModel(
        buses={1: {"name": "A"}, 2: {"name": "B"}, 3: {"name": "C"}},
        lines={
            "L12": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10.0,
                    "flow_limit": 100, "contingency_limit": 80},
            "L23": {"from_bus": 2, "to_bus": 3, "x_pu": 0.1, "b_pu": 10.0,
                    "flow_limit": 100, "contingency_limit": 80},
            "L31": {"from_bus": 3, "to_bus": 1, "x_pu": 0.1, "b_pu": 10.0,
                    "flow_limit": 100, "contingency_limit": 80},
        },
        slack_bus=1,
    )

    expected_ptdf = np.array([
        [0.0, -2.0 / 3.0, -1.0 / 3.0],
        [0.0,  1.0 / 3.0, -1.0 / 3.0],
        [0.0,  1.0 / 3.0,  2.0 / 3.0],
    ])
    np.testing.assert_allclose(model.ptdf, expected_ptdf, atol=1e-10)

    flows = model.line_flow(np.array([100.0, -40.0, -60.0]))
    expected_flows = np.array([140.0 / 3.0, 20.0 / 3.0, -160.0 / 3.0])
    np.testing.assert_allclose(flows, expected_flows, atol=1e-10)


def test_network_3bus_triangle_shapes_and_contingency():
    model = NetworkModel(
        buses={1: {"name": "A"}, 2: {"name": "B"}, 3: {"name": "C"}},
        lines={
            "L1": {"from_bus": 1, "to_bus": 2, "x_pu": 0.1, "b_pu": 10.0,
                   "flow_limit": 100, "contingency_limit": 80},
            "L2": {"from_bus": 2, "to_bus": 3, "x_pu": 0.2, "b_pu": 5.0,
                   "flow_limit": 80, "contingency_limit": 60},
            "L3": {"from_bus": 1, "to_bus": 3, "x_pu": 0.15, "b_pu": 6.67,
                   "flow_limit": 90, "contingency_limit": 70},
        },
        slack_bus=1,
        contingencies={"CTG_L1": {"outaged_line": "L1"}},
    )

    assert model.ptdf.shape == (3, 3)
    assert model.lodf.shape == (3, 3)
    assert abs(model.ptdf[0, 0]) < 1e-10
    assert "CTG_L1" in model.contingency_ptdf
    assert len(model.line_flow(np.array([100.0, -60.0, -40.0]))) == 3


def test_network_ercot_17bus_shapes():
    from examples.ercot_17bus.topology import BUSES, LINES, SLACK_BUS, CONTINGENCIES

    model = NetworkModel(
        buses=BUSES,
        lines=LINES,
        slack_bus=SLACK_BUS,
        contingencies=CONTINGENCIES,
    )

    assert model.ptdf.shape == (26, 17)
    assert model.n_bus == 17
    assert model.n_line == 26
    assert len(model.contingency_ptdf) > 0
