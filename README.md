# Nodal Market Dispatch

Python solvers and examples for DC-network-based dispatch, commitment, auction
clearing, and topology visualization studies.

## Overview

This repo is a compact research and prototyping environment for nodal power
market studies. It models transmission networks with DC power flow, then uses
that network model inside market-clearing and dispatch workflows.

The current codebase includes:

- A topology-agnostic DC network model that builds PTDF, LODF, and contingency
  PTDF matrices from bus and line dictionaries.
- A SCUC solver for day-ahead unit commitment with generator operating limits,
  ramping, minimum up/down time, reserve, load shedding, and transmission
  constraints.
- A SCED solver for hourly economic dispatch using committed resources and DC
  network limits.
- A CRR auction clearing solver with simultaneous feasibility test constraints.
- A reusable Plotly topology visualization utility for inspecting example
  networks on a geographic map.
- An ERCOT-style 17-bus example system with topology, generator data, load
  profiles, batteries, outages, and example workflows.

The repo is designed to keep market logic and test systems separate. Generic
solver code lives under `solvers/`, reusable visualization code lives under
`viz/`, and concrete systems live under `examples/`.

This repo was completed with assistance from Codex and Claude Code.

## Repository Layout

```text
solvers/
  network/       DC network model, PTDF, LODF, contingency PTDF
  scuc/          Security-constrained unit commitment
  sced/          Security-constrained economic dispatch
  auction/       CRR auction clearing and SFT logic

examples/
  ercot_17bus/   ERCOT-style 17-bus example system and workflows

viz/
  topology.py    Generic topology map visualization

tests/           Unit and smoke tests, including hand-checkable 3-bus cases
```

## Installation

From a Windows Command Prompt:

```cmd
git clone <repo-url>
cd nodal-market-dispatch
uv venv
.venv\Scripts\activate.bat
uv sync --extra dev
```

Confirm the environment:

```cmd
python --version
python -c "import sys; print(sys.executable)"
```

## Run Tests

```cmd
uv run pytest
```

Focused test examples:

```cmd
uv run pytest tests/test_network.py
uv run pytest tests/test_scuc.py
uv run pytest tests/test_sced.py
uv run pytest tests/test_auction.py
```

## Run Examples

Generate the ERCOT 17-bus topology map:

```cmd
uv run python -m examples.ercot_17bus.plot_topology
```

Generate a monthly topology map:

```cmd
uv run python -m examples.ercot_17bus.plot_topology --year 2023 --month 3
```

Run the day-ahead market example:

```cmd
uv run python -m examples.ercot_17bus.run_dam
```

## Dependencies

Dependencies are managed with `uv`:

- `pyproject.toml` lists direct runtime and dev dependencies.
- `uv.lock` records the resolved package graph.

Add a runtime package:

```cmd
uv add <package-name>
uv run pytest
```

Add a dev-only package:

```cmd
uv add --optional dev <package-name>
uv run pytest
```

Commit both `pyproject.toml` and `uv.lock` after dependency changes.
