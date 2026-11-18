"""Simulações independentes, com controles estabilizados antes da falta."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from ..circuit.loader import load_circuit, solve, reset_faults
from ..circuit.bus_info import bus_phases, list_buses
from . import fault_injector as fi
from .snapshot import snapshot_substation
from .scenarios import make_scenarios, Scenario
from .load_distributions import (
    NominalLoad,
    SpatialPattern,
    apply_spatial_pattern,
    capture_nominal_loads,
    generate_spatial_patterns,
)


def set_load_multiplier(d, mult: float) -> None:
    d.Text.Command = f'Set LoadMult={mult}'


def regulator_state(d) -> dict:
    """Lê posições discretas e relações de transformação dos reguladores."""
    regs = d.ActiveCircuit.RegControls
    result = {}
    for name in regs.AllNames:
        if name.lower() == 'none':
            continue
        regs.Name = name
        transformer = d.ActiveCircuit.Transformers
        transformer.Name = regs.Transformer
        transformer.Wdg = regs.TapWinding
        result[name] = {'tap_number': int(regs.TapNumber), 'ratio': float(transformer.Tap)}
    return result


def checked_solution(d, stage: str) -> dict:
    """Interrompe a geração se a solução elétrica ou de controles falhar."""
    s = d.ActiveCircuit.Solution
    result = dict(converged=bool(s.Converged), controls_done=bool(s.ControlActionsDone),
                  iterations=int(s.Iterations), control_iterations=int(s.ControlIterations),
                  control_mode=int(s.ControlMode), taps=regulator_state(d))
    if not result['converged'] or not result['controls_done']:
        raise RuntimeError(f'Solução não concluída em {stage}: {result}')
    return result


def run_scenario(
    d,
    sc: Scenario,
    measurement_element: str,
    measurement_terminal: int = 1,
    legacy_fault_mapping: bool = False,
    nominal_loads: list[NominalLoad] | None = None,
    spatial_pattern: SpatialPattern | None = None,
    reusable_fault_name: str | None = None,
    freeze_controls: bool = False,
    audit: dict | None = None,
) -> dict:
    # False preserva a API histórica; a geração vigente passa True explicitamente.
    if freeze_controls:
        d.Text.Command = 'Set ControlMode=STATIC'
    if nominal_loads is not None and spatial_pattern is not None:
        apply_spatial_pattern(d, nominal_loads, spatial_pattern)
    set_load_multiplier(d, sc.load_mult)
    solve(d)
    pre_state = checked_solution(d, 'pré-falta') if freeze_controls else None
    pre = snapshot_substation(d, measurement_element, measurement_terminal)

    # apply fault
    name = None
    try:
        if freeze_controls:
            d.Text.Command = 'Set ControlMode=OFF'
        if reusable_fault_name is not None:
            name = fi.configure_reusable_fault(
                d,
                reusable_fault_name,
                sc.fault_kind,
                sc.bus,
                sc.phases,
                sc.rf,
                sc.ground,
                legacy_fault_mapping,
            )
        elif sc.fault_kind == 'mono':
            name = fi.apply_monophasic(d, sc.bus, sc.phases[0], sc.rf)
        elif sc.fault_kind == 'bi':
            name = fi.apply_biphasic(
                d, sc.bus, sc.phases, sc.rf, sc.ground, legacy_fault_mapping
            )
        elif sc.fault_kind == 'tri':
            name = fi.apply_triphasic(
                d, sc.bus, sc.rf, sc.ground, legacy_fault_mapping
            )
        else:
            raise ValueError(sc.fault_kind)

        solve(d)
        fault_state = checked_solution(d, 'falta') if freeze_controls else None
        if freeze_controls and pre_state['taps'] != fault_state['taps']:
            raise RuntimeError('Posição do regulador mudou durante a falta.')
        during = snapshot_substation(d, measurement_element, measurement_terminal)
    finally:
        if name is not None:
            fi.disable_fault(d, name)
        reset_faults(d)

    if audit is not None:
        audit.update(pre=pre_state, fault=fault_state)

    row = {
        'bus': sc.bus,
        'fault_kind': sc.fault_kind,
        'phases': '-'.join(str(p) for p in sc.phases),
        'ground': sc.ground,
        'rf': sc.rf,
        'load_mult': sc.load_mult,
        'type_id': sc.type_id,
    }
    if spatial_pattern is not None:
        row['distribution_id'] = sc.distribution_id
    for k, v in pre.items():
        row[f'pre_{k}'] = v
    for k, v in during.items():
        row[f'fault_{k}'] = v
    return row


def run_all(
    dss_file: str,
    rf_values,
    load_mults,
    bi_pairs,
    exclude_buses,
    measurement_element: str = 'Line.650632',
    measurement_terminal: int = 1,
    legacy_fault_mapping: bool = False,
    system_id: str | None = None,
    continue_on_error: bool = False,
    spatial_distribution_config: dict | None = None,
    max_control_iterations: int = 100,
    independent_scenarios: bool = True,
    freeze_controls: bool = True,
    audit_callback=None,
) -> pd.DataFrame:
    d = load_circuit(dss_file)
    d.Text.Command = f'Set MaxControlIter={max_control_iterations}'
    buses = list_buses(d, exclude=exclude_buses)
    buses_phases = {b: bus_phases(d, b)[1] for b in buses}
    # remove buses with no phase nodes
    buses_phases = {b: p for b, p in buses_phases.items() if p}

    nominal_loads = None
    patterns = None
    distribution_ids = None
    if spatial_distribution_config is not None:
        nominal_loads = capture_nominal_loads(d)
        spatial_cfg = spatial_distribution_config
        patterns = generate_spatial_patterns(
            nominal_loads,
            int(spatial_cfg.get('heterogeneous_count', 0)),
            float(spatial_cfg.get('amplitude', 0.10)),
            int(spatial_cfg.get('seed', 42)),
        )
        distribution_ids = list(patterns)
    scenarios = list(make_scenarios(
        buses_phases,
        rf_values,
        load_mults,
        bi_pairs,
        distribution_ids,
    ))
    first = scenarios[0]
    reusable_fault_name = fi.create_reusable_fault(
        d, first.bus, first.phases[0], first.rf
    )
    rows = []
    for sc in tqdm(scenarios, desc='simulations'):
        try:
            if independent_scenarios:
                d = load_circuit(dss_file)
                d.Text.Command = f'Set MaxControlIter={max_control_iterations}'
                reusable_fault_name = fi.create_reusable_fault(d, sc.bus, sc.phases[0], sc.rf)
            audit = {}
            row = run_scenario(
                d,
                sc,
                measurement_element,
                measurement_terminal,
                legacy_fault_mapping,
                nominal_loads,
                None if patterns is None else patterns[sc.distribution_id],
                reusable_fault_name,
                freeze_controls=freeze_controls,
                audit=audit,
            )
            if system_id:
                row['system_id'] = system_id
            rows.append(row)
            if audit_callback is not None:
                audit_callback({'case_id': len(rows)-1, 'bus': sc.bus,
                                'type_id': sc.type_id, 'load_mult': sc.load_mult,
                                'distribution_id': sc.distribution_id, **audit})
        except Exception as e:
            if not continue_on_error:
                raise RuntimeError(f'Falha no cenário {sc}') from e
            print(f'Failed {sc}: {e}')
    return pd.DataFrame(rows)
