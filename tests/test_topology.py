"""ERCOT example topology overlay tests."""


def test_monthly_topology_variation():
    from examples.ercot_17bus.topology import LINES, GENERATORS, get_monthly_topology

    january = get_monthly_topology(2023, 1)
    assert january["rating_factor"] == 1.20
    assert len(january["outaged_lines"]) == 0

    march = get_monthly_topology(2023, 3)
    assert "L_NORTH_HOUSTON_1" in march["outaged_lines"]
    assert len(march["lines"]) == len(LINES) - 1

    july = get_monthly_topology(2023, 7)
    assert july["rating_factor"] == 0.80

    april_2024 = get_monthly_topology(2024, 4)
    assert "NUC_SOUTH_001" in april_2024["outaged_generators"]
    assert len(april_2024["generators"]) == len(GENERATORS) - 1
