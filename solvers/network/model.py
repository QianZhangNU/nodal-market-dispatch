"""
DC Power Flow Network Model (topology-agnostic)
=================================================
Builds PTDF and LODF from arbitrary bus/line dicts passed as arguments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class NetworkModel:
    """
    DC power flow model. All topology passed as arguments.

    Parameters
    ----------
    buses : dict  {bus_id: {"name": str, ...}}
    lines : dict  {line_id: {"from_bus", "to_bus", "x_pu", "b_pu", "flow_limit", ...}}
    slack_bus : any hashable bus ID
    contingencies : dict  {ctg_id: {"outaged_line": line_id}}
    """
    buses: Dict[Any, dict]
    lines: Dict[str, dict]
    slack_bus: Any
    contingencies: Dict[str, dict] = field(default_factory=dict)

    bus_list: List = field(init=False)
    bus_idx: Dict = field(init=False)
    line_list: List[str] = field(init=False)
    n_bus: int = field(init=False)
    n_line: int = field(init=False)
    ptdf: np.ndarray = field(init=False)
    lodf: np.ndarray = field(init=False)
    contingency_ptdf: Dict[str, np.ndarray] = field(init=False)

    def __post_init__(self):
        self.bus_list = sorted(self.buses.keys())
        self.bus_idx = {b: i for i, b in enumerate(self.bus_list)}
        self.line_list = list(self.lines.keys())
        self.n_bus = len(self.bus_list)
        self.n_line = len(self.line_list)
        self.ptdf = self._build_ptdf()
        self.lodf = self._build_lodf()
        self.contingency_ptdf = self._build_contingency_ptdfs()

    def _build_ptdf(self) -> np.ndarray:
        B = np.zeros((self.n_bus, self.n_bus))
        for ld in self.lines.values():
            i = self.bus_idx[ld["from_bus"]]
            j = self.bus_idx[ld["to_bus"]]
            b = ld["b_pu"]
            B[i, i] += b; B[j, j] += b
            B[i, j] -= b; B[j, i] -= b
        slack_i = self.bus_idx[self.slack_bus]
        non_slack = [i for i in range(self.n_bus) if i != slack_i]
        X = np.zeros((self.n_bus, self.n_bus))
        X[np.ix_(non_slack, non_slack)] = np.linalg.inv(B[np.ix_(non_slack, non_slack)])
        ptdf = np.zeros((self.n_line, self.n_bus))
        for k, lid in enumerate(self.line_list):
            ld = self.lines[lid]
            i = self.bus_idx[ld["from_bus"]]
            j = self.bus_idx[ld["to_bus"]]
            ptdf[k, :] = ld["b_pu"] * (X[i, :] - X[j, :])
        return ptdf

    def _build_lodf(self) -> np.ndarray:
        lodf = np.zeros((self.n_line, self.n_line))
        for k, lid_k in enumerate(self.line_list):
            ld = self.lines[lid_k]
            fi = self.bus_idx[ld["from_bus"]]
            ti = self.bus_idx[ld["to_bus"]]
            denom = 1.0 - (self.ptdf[k, fi] - self.ptdf[k, ti])
            if abs(denom) < 1e-9:
                continue
            for l in range(self.n_line):
                if l == k:
                    lodf[l, k] = -1.0
                else:
                    lodf[l, k] = (self.ptdf[l, fi] - self.ptdf[l, ti]) / denom
        return lodf

    def _build_contingency_ptdfs(self) -> Dict[str, np.ndarray]:
        result = {}
        for ctg_id, ctg in self.contingencies.items():
            if "outaged_line" not in ctg or ctg["outaged_line"] not in self.line_list:
                continue
            kk = self.line_list.index(ctg["outaged_line"])
            pc = self.ptdf + np.outer(self.lodf[:, kk], self.ptdf[kk, :])
            pc[kk, :] = 0.0
            result[ctg_id] = pc
        return result

    def line_flow(self, net_injections: np.ndarray) -> np.ndarray:
        return self.ptdf @ net_injections

    def gen_at_bus(self, generators: dict) -> Dict[Any, List[str]]:
        result = {b: [] for b in self.bus_list}
        for g, gd in generators.items():
            if gd["bus"] in result:
                result[gd["bus"]].append(g)
        return result
