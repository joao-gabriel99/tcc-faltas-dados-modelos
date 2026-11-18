"""Captures circuit state: phasors V, I per bus."""
from __future__ import annotations
import numpy as np


def capture_bus_voltage_phasors(d) -> dict[str, complex]:
    """Returns dict bus.phase -> complex V (line-ground)."""
    ckt = d.ActiveCircuit
    result: dict[str, complex] = {}
    for bus in ckt.AllBusNames:
        ckt.SetActiveBus(bus)
        nodes = list(ckt.ActiveBus.Nodes)
        voltages = list(ckt.ActiveBus.Voltages)  # alternating real, imag
        for i, node in enumerate(nodes):
            if node in (1, 2, 3):
                re, im = voltages[2 * i], voltages[2 * i + 1]
                result[f'{bus}.{node}'] = complex(re, im)
    return result


def capture_substation_currents(d, source_element: str = 'Vsource.source') -> dict[int, complex]:
    """Captures currents flowing from the feeder source on each phase."""
    ckt = d.ActiveCircuit
    ckt.SetActiveElement(source_element)
    currents = list(ckt.ActiveCktElement.Currents)
    # currents alternates real/imag for each terminal/phase
    out: dict[int, complex] = {}
    for ph in (1, 2, 3):
        idx = (ph - 1) * 2
        if idx + 1 < len(currents):
            out[ph] = complex(currents[idx], currents[idx + 1])
    return out


def capture_element_phasors(
    d,
    element_name: str,
    terminal: int = 1,
) -> dict[int, tuple[complex, complex]]:
    """Captura ``(V, I)`` por fase em um terminal explícito do elemento."""
    ckt = d.ActiveCircuit
    available = {name.lower() for name in ckt.AllElementNames}
    if element_name.lower() not in available:
        raise ValueError(f'Elemento de medição inexistente: {element_name}')
    ckt.SetActiveElement(element_name)

    element = ckt.ActiveCktElement
    if terminal < 1 or terminal > element.NumTerminals:
        raise ValueError(
            f'Terminal {terminal} inválido para {element_name}; '
            f'o elemento possui {element.NumTerminals} terminal(is)'
        )

    voltages = list(element.Voltages)
    currents = list(element.Currents)
    conductors = element.NumConductors
    offset = (terminal - 1) * conductors * 2
    phasors: dict[int, tuple[complex, complex]] = {}
    for ph in (1, 2, 3):
        idx = offset + (ph - 1) * 2
        if idx + 1 < len(voltages) and idx + 1 < len(currents):
            v = complex(voltages[idx], voltages[idx + 1])
            i = complex(currents[idx], currents[idx + 1])
            phasors[ph] = (v, i)
    return phasors


def capture_feeder_line_phasors(
    d,
    line_name: str = 'Line.650632',
    terminal: int = 1,
) -> dict[int, tuple[complex, complex]]:
    """Compatibilidade: captura fasores de uma linha explicitamente nomeada."""
    return capture_element_phasors(d, line_name, terminal)


def snapshot_substation(
    d,
    measurement_element: str = 'Line.650632',
    measurement_terminal: int = 1,
) -> dict[str, float]:
    """Returns flat dict of substation-only phasors (mag + angle)."""
    feeder = capture_element_phasors(
        d,
        measurement_element,
        measurement_terminal,
    )
    flat: dict[str, float] = {}
    for ph in (1, 2, 3):
        if ph in feeder:
            v, i = feeder[ph]
        else:
            v = i = complex(0, 0)
        flat[f'V{ph}_mag'] = abs(v)
        flat[f'V{ph}_ang'] = np.angle(v)
        flat[f'I{ph}_mag'] = abs(i)
        flat[f'I{ph}_ang'] = np.angle(i)
    return flat
