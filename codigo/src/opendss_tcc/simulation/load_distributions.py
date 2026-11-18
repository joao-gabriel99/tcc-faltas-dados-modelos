"""Distribuições espaciais estáticas e reproduzíveis das cargas."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class NominalLoad:
    """Potências nominais de um elemento ``Load`` do OpenDSS."""

    name: str
    kw: float
    kvar: float


@dataclass(frozen=True)
class SpatialPattern:
    """Fatores multiplicativos de uma realização espacial de carga."""

    distribution_id: str
    seed: int | None
    factors: dict[str, float]


def capture_nominal_loads(d) -> list[NominalLoad]:
    """Captura a base imutável usada para evitar multiplicação acumulativa."""
    loads = d.ActiveCircuit.Loads
    result: list[NominalLoad] = []
    for name in loads.AllNames:
        loads.Name = name
        result.append(NominalLoad(name=name, kw=float(loads.kW), kvar=float(loads.kvar)))
    if not result:
        raise ValueError('O circuito não possui elementos Load')
    return result


def _project_preserving_totals(
    deviations: np.ndarray,
    nominal_loads: Sequence[NominalLoad],
    amplitude: float,
) -> np.ndarray:
    """Projeta desvios no espaço que preserva simultaneamente kW e kvar."""
    constraints = np.array(
        [
            [load.kw for load in nominal_loads],
            [load.kvar for load in nominal_loads],
        ],
        dtype=float,
    )
    constraints = constraints[np.linalg.norm(constraints, axis=1) > 1e-12]
    correction = constraints.T @ np.linalg.pinv(
        constraints @ constraints.T
    ) @ (constraints @ deviations)
    projected = deviations - correction
    largest = float(np.max(np.abs(projected)))
    if largest > amplitude:
        projected *= amplitude / largest
    return projected


def generate_spatial_patterns(
    nominal_loads: Sequence[NominalLoad],
    heterogeneous_count: int = 5,
    amplitude: float = 0.10,
    seed: int = 42,
) -> dict[str, SpatialPattern]:
    """Gera P0 uniforme e P1..Pn com sementes fixas e limites verificáveis."""
    if heterogeneous_count < 0:
        raise ValueError('heterogeneous_count deve ser não negativo')
    if not 0 < amplitude < 1:
        raise ValueError('amplitude deve estar no intervalo aberto (0, 1)')

    names = [load.name for load in nominal_loads]
    patterns = {
        'P0': SpatialPattern('P0', None, {name: 1.0 for name in names})
    }
    for index in range(1, heterogeneous_count + 1):
        pattern_seed = seed + index
        rng = np.random.default_rng(pattern_seed)
        raw = rng.uniform(-amplitude, amplitude, len(nominal_loads))
        deviations = _project_preserving_totals(raw, nominal_loads, amplitude)
        factors = {
            name: float(1.0 + deviation)
            for name, deviation in zip(names, deviations, strict=True)
        }
        pattern_id = f'P{index}'
        patterns[pattern_id] = SpatialPattern(pattern_id, pattern_seed, factors)
    return patterns


def apply_spatial_pattern(
    d,
    nominal_loads: Sequence[NominalLoad],
    pattern: SpatialPattern,
) -> None:
    """Aplica fatores à base nominal, preservando o fator de potência individual."""
    loads = d.ActiveCircuit.Loads
    for load in nominal_loads:
        factor = pattern.factors[load.name]
        loads.Name = load.name
        loads.kW = load.kw * factor
        loads.kvar = load.kvar * factor


def patterns_metadata(
    nominal_loads: Sequence[NominalLoad],
    patterns: dict[str, SpatialPattern],
    amplitude: float,
    seed: int,
) -> dict:
    """Produz artefato auditável com fatores, sementes e erros de conservação."""
    base_kw = sum(load.kw for load in nominal_loads)
    base_kvar = sum(load.kvar for load in nominal_loads)
    entries = []
    for pattern in patterns.values():
        total_kw = sum(
            load.kw * pattern.factors[load.name] for load in nominal_loads
        )
        total_kvar = sum(
            load.kvar * pattern.factors[load.name] for load in nominal_loads
        )
        entries.append(
            {
                'distribution_id': pattern.distribution_id,
                'seed': pattern.seed,
                'factors': pattern.factors,
                'total_kw': total_kw,
                'total_kvar': total_kvar,
                'kw_error': total_kw - base_kw,
                'kvar_error': total_kvar - base_kvar,
            }
        )
    return {
        'method': 'uniform_random_projected_kw_kvar',
        'amplitude': amplitude,
        'base_seed': seed,
        'nominal_loads': [asdict(load) for load in nominal_loads],
        'base_total_kw': base_kw,
        'base_total_kvar': base_kvar,
        'patterns': entries,
    }
