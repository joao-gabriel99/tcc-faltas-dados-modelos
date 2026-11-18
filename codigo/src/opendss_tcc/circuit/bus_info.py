"""Bus phase introspection. Replaces busPhaseInfo.m."""
from __future__ import annotations


def bus_phases(d, bus_name: str) -> tuple[int, list[int]]:
    """Returns (n_phases, phases) for a bus. phases like [1,2,3] or [1,3] etc."""
    ckt = d.ActiveCircuit
    ckt.SetActiveBus(bus_name)
    nodes = [int(n) for n in ckt.ActiveBus.Nodes]
    phase_nodes = [n for n in nodes if n in (1, 2, 3)]
    return (len(phase_nodes), phase_nodes)


def is_biphasic_valid(phase_pair: tuple[int, int], phases: list[int]) -> bool:
    a, b = phase_pair
    return a in phases and b in phases


def list_buses(d, exclude: list[str] | None = None) -> list[str]:
    ckt = d.ActiveCircuit
    all_buses = [b.lower() for b in ckt.AllBusNames]
    if exclude:
        exclude_lower = {x.lower() for x in exclude}
        return [b for b in all_buses if b not in exclude_lower]
    return all_buses
