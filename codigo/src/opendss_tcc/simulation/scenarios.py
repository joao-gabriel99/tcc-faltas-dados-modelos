"""Generates the cartesian product of simulation scenarios."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Iterator


@dataclass(frozen=True)
class Scenario:
    bus: str
    fault_kind: str            # 'mono' | 'bi' | 'tri'
    phases: tuple[int, ...]    # () for tri, (p,) for mono, (a,b) for bi
    ground: bool
    rf: float
    load_mult: float
    type_id: int               # 1..11
    distribution_id: str = 'P0'


def make_scenarios(
    buses_phases: dict[str, list[int]],
    rf_values: list[float],
    load_mults: list[float],
    bi_pairs: list[tuple[int, int]],
    distribution_ids: list[str] | None = None,
) -> Iterator[Scenario]:
    from .fault_injector import fault_id

    distribution_ids = distribution_ids or ['P0']

    for bus, phases in buses_phases.items():
        # Monophasic
        for ph in phases:
            for rf, lm, distribution_id in product(
                rf_values, load_mults, distribution_ids
            ):
                yield Scenario(
                    bus=bus, fault_kind='mono', phases=(ph,), ground=True,
                    rf=rf, load_mult=lm,
                    type_id=fault_id('mono', ph, True),
                    distribution_id=distribution_id,
                )

        # Biphasic (with and without ground)
        for pair in bi_pairs:
            if not (pair[0] in phases and pair[1] in phases):
                continue
            for ground in (False, True):
                for rf, lm, distribution_id in product(
                    rf_values, load_mults, distribution_ids
                ):
                    yield Scenario(
                        bus=bus, fault_kind='bi', phases=tuple(pair), ground=ground,
                        rf=rf, load_mult=lm,
                        type_id=fault_id('bi', pair, ground),
                        distribution_id=distribution_id,
                    )

        # Triphasic (requires 3 phases on bus)
        if len(phases) == 3:
            for ground in (False, True):
                for rf, lm, distribution_id in product(
                    rf_values, load_mults, distribution_ids
                ):
                    yield Scenario(
                        bus=bus, fault_kind='tri', phases=(1, 2, 3), ground=ground,
                        rf=rf, load_mult=lm,
                        type_id=fault_id('tri', None, ground),
                        distribution_id=distribution_id,
                    )
