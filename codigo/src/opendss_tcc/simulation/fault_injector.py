"""Fault injection on OpenDSS circuits."""
from __future__ import annotations


# Canonical fault type IDs (matches MATLAB Tipo.m)
FAULT_TYPE_MAP = {
    ('mono', 1, 'g'): 1,    # A-G
    ('mono', 2, 'g'): 2,    # B-G
    ('mono', 3, 'g'): 3,    # C-G
    ('bi', (1, 2), 'ng'): 4,  # A-B
    ('bi', (1, 3), 'ng'): 5,  # A-C
    ('bi', (2, 3), 'ng'): 6,  # B-C
    ('bi', (1, 2), 'g'): 7,   # A-B-G
    ('bi', (1, 3), 'g'): 8,   # A-C-G
    ('bi', (2, 3), 'g'): 9,   # B-C-G
    ('tri', None, 'ng'): 10,  # A-B-C
    ('tri', None, 'g'): 11,   # A-B-C-G
}


def fault_id(kind: str, phases, ground: bool) -> int:
    key_ground = 'g' if ground else 'ng'
    if kind == 'mono':
        return FAULT_TYPE_MAP[('mono', phases, 'g')]
    if kind == 'bi':
        ph = tuple(sorted(phases))
        return FAULT_TYPE_MAP[('bi', ph, key_ground)]
    if kind == 'tri':
        return FAULT_TYPE_MAP[('tri', None, key_ground)]
    raise ValueError(f'unknown kind {kind}')


_counter = [0]


def _next_id() -> int:
    _counter[0] += 1
    return _counter[0]


def apply_monophasic(d, bus: str, phase: int, r: float) -> str:
    name = f'1ph_{bus}_{phase}_{_next_id()}'
    d.Text.Command = 'Reset faults'
    d.Text.Command = (
        f'New Fault.{name} phases=1 bus1={bus}.{phase} r={r}'
    )
    return name


def apply_biphasic(
    d,
    bus: str,
    phases: tuple[int, int],
    r: float,
    ground: bool,
    legacy_ground_mapping: bool = False,
) -> str:
    a, b = phases
    tag = 'g' if ground else 'ng'
    name = f'2ph_{bus}_{a}{b}_{tag}_{_next_id()}'
    d.Text.Command = 'Reset faults'
    if legacy_ground_mapping:
        # Convenção histórica do dataset de 15.300 amostras. Mantida somente
        # para reprodução; eletricamente, os rótulos terra/não terra se invertem.
        bus2 = f' bus2={bus}.4.4.4' if ground else ''
        command = f'New Fault.{name} phases=2 bus1={bus}.{a}.{b}{bus2} r={r}'
    elif ground:
        command = (
            f'New Fault.{name} phases=2 bus1={bus}.{a}.{b} '
            f'bus2={bus}.0.0 r={r}'
        )
    else:
        command = (
            f'New Fault.{name} phases=1 bus1={bus}.{a} '
            f'bus2={bus}.{b} r={r}'
        )
    d.Text.Command = command
    return name


def apply_triphasic(
    d,
    bus: str,
    r: float,
    ground: bool,
    legacy_ground_mapping: bool = False,
) -> str:
    tag = 'g' if ground else 'ng'
    name = f'3ph_{bus}_{tag}_{_next_id()}'
    if legacy_ground_mapping:
        bus1 = bus
        bus2 = f' bus2={bus}.4.4.4' if ground else ''
    else:
        bus1 = f'{bus}.1.2.3'
        bus2 = f' bus2={bus}.0.0.0' if ground else f' bus2={bus}.4.4.4'
    d.Text.Command = 'Reset faults'
    d.Text.Command = (
        f'New Fault.{name} phases=3 bus1={bus1}{bus2} r={r}'
    )
    return name


def disable_fault(d, name: str) -> None:
    d.Text.Command = f'Fault.{name}.enabled=false'


def create_reusable_fault(
    d,
    bus: str,
    phase: int,
    r: float,
    name: str = 'scenario_fault',
) -> str:
    """Cria o único elemento de falta reutilizado pela geração em lote."""
    d.Text.Command = (
        f'New Fault.{name} phases=1 bus1={bus}.{phase} '
        f'bus2={bus}.0 r={r} enabled=false'
    )
    return name


def configure_reusable_fault(
    d,
    name: str,
    kind: str,
    bus: str,
    phases: tuple[int, ...],
    r: float,
    ground: bool,
    legacy_ground_mapping: bool = False,
) -> str:
    """Edita a falta em vez de acumular milhares de elementos desabilitados."""
    if kind == 'mono':
        command = (
            f'Edit Fault.{name} phases=1 bus1={bus}.{phases[0]} '
            f'bus2={bus}.0 r={r} enabled=true'
        )
    elif kind == 'bi':
        a, b = phases
        if legacy_ground_mapping:
            bus2 = f'{bus}.4.4.4' if ground else f'{bus}.0.0'
            command = (
                f'Edit Fault.{name} phases=2 bus1={bus}.{a}.{b} '
                f'bus2={bus2} r={r} enabled=true'
            )
        elif ground:
            command = (
                f'Edit Fault.{name} phases=2 bus1={bus}.{a}.{b} '
                f'bus2={bus}.0.0 r={r} enabled=true'
            )
        else:
            command = (
                f'Edit Fault.{name} phases=1 bus1={bus}.{a} '
                f'bus2={bus}.{b} r={r} enabled=true'
            )
    elif kind == 'tri':
        if legacy_ground_mapping:
            bus1 = bus
            bus2 = f'{bus}.4.4.4' if ground else f'{bus}.0.0.0'
        else:
            bus1 = f'{bus}.1.2.3'
            bus2 = f'{bus}.0.0.0' if ground else f'{bus}.4.4.4'
        command = (
            f'Edit Fault.{name} phases=3 bus1={bus1} bus2={bus2} '
            f'r={r} enabled=true'
        )
    else:
        raise ValueError(kind)
    d.Text.Command = 'Reset faults'
    d.Text.Command = command
    return name
