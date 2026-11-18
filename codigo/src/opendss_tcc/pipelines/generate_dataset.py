"""Planejamento e geração dos datasets de simulação por sistema."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import yaml

from ..circuit.bus_info import bus_phases, list_buses
from ..circuit.loader import load_circuit
from ..features.build import build_features
from ..simulation.runner import run_all
from ..simulation.scenarios import make_scenarios
from ..simulation.load_distributions import (
    capture_nominal_loads,
    generate_spatial_patterns,
    patterns_metadata,
)


BI_PAIRS = [(1, 2), (1, 3), (2, 3)]


def load_config(cfg_path: str | Path) -> tuple[Path, dict]:
    path = Path(cfg_path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    spatial_cfg = cfg.get('spatial_distributions')
    if spatial_cfg is not None:
        if spatial_cfg.get('baseline_id', 'P0') != 'P0':
            raise ValueError('O baseline espacial deve ser identificado como P0')
        supported = 'uniform_random_projected_kw_kvar'
        if spatial_cfg.get('method', supported) != supported:
            raise ValueError(
                f'Método espacial não suportado: {spatial_cfg.get("method")}'
            )
        try:
            heterogeneous_count = int(spatial_cfg['heterogeneous_count'])
            amplitude = float(spatial_cfg['amplitude'])
            int(spatial_cfg['seed'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Configuração espacial incompleta ou inválida') from exc
        if heterogeneous_count < 0 or not 0 < amplitude < 1:
            raise ValueError('Contagem ou amplitude espacial inválida')
    return path, cfg


def resolve_paths(cfg_path: str | Path, cfg: dict) -> dict[str, Path]:
    path = Path(cfg_path).resolve()
    project_root = path.parent.parent
    output = cfg['output']
    interim_value = output.get('interim_path')
    if interim_value is None:
        interim_value = str(Path(output['interim_dir']) / 'raw_simulations.parquet')
    metadata_value = output.get('distribution_metadata_path')
    if metadata_value is None:
        metadata_value = str(Path(interim_value).with_name('spatial_distributions.json'))
    return {
        'project_root': project_root,
        'dss': (project_root / cfg['circuit']['dss_file']).resolve(),
        'interim': (project_root / interim_value).resolve(),
        'processed': (project_root / output['processed_path']).resolve(),
        'results': (project_root / output.get('results_dir', 'results')).resolve(),
        'distribution_metadata': (project_root / metadata_value).resolve(),
    }


def describe_plan(cfg_path: str | Path) -> dict:
    """Compila o circuito e calcula o plano sem executar faltas ou gravar dados."""
    path, cfg = load_config(cfg_path)
    paths = resolve_paths(path, cfg)
    d = load_circuit(paths['dss'])
    max_control_iterations = int(
        cfg['circuit'].get('max_control_iterations', 100)
    )
    d.Text.Command = f'Set MaxControlIter={max_control_iterations}'
    excluded = cfg['circuit'].get('exclude_buses', [])
    buses = list_buses(d, exclude=excluded)
    phases = {bus: bus_phases(d, bus)[1] for bus in buses}
    phases = {bus: values for bus, values in phases.items() if values}
    rf_values = cfg['variations']['fault_resistance_ohms']
    load_mults = cfg['variations']['load_multipliers']
    nominal_loads = capture_nominal_loads(d)
    spatial_cfg = cfg.get('spatial_distributions')
    patterns = None
    if spatial_cfg is not None:
        patterns = generate_spatial_patterns(
            nominal_loads,
            int(spatial_cfg['heterogeneous_count']),
            float(spatial_cfg['amplitude']),
            int(spatial_cfg['seed']),
        )
    distribution_ids = None if patterns is None else list(patterns)
    scenarios = list(make_scenarios(
        phases, rf_values, load_mults, BI_PAIRS, distribution_ids
    ))
    type_counts = Counter(scenario.type_id for scenario in scenarios)
    distribution_counts = (
        Counter(scenario.distribution_id for scenario in scenarios)
        if patterns is not None else Counter()
    )
    phase_counts = {
        str(n): sum(1 for values in phases.values() if len(values) == n)
        for n in (1, 2, 3)
    }
    return {
        'system_id': cfg.get('system_id', 'legacy'),
        'dss_file': str(paths['dss']),
        'converged': bool(d.ActiveCircuit.Solution.Converged),
        'eligible_buses': list(phases),
        'eligible_bus_count': len(phases),
        'phase_counts': phase_counts,
        'rf_values': rf_values,
        'load_multipliers': load_mults,
        'spatial_enabled': patterns is not None,
        'distribution_ids': [] if patterns is None else list(patterns),
        'distribution_counts': dict(distribution_counts),
        'baseline_scenario_count': (
            distribution_counts['P0'] if patterns is not None else len(scenarios)
        ),
        'scenario_count': len(scenarios),
        'type_counts': {str(key): type_counts[key] for key in sorted(type_counts)},
        'measurement_element': cfg['circuit'].get(
            'measurement_element', 'Line.650632'
        ),
        'measurement_terminal': int(cfg['circuit'].get('measurement_terminal', 1)),
        'max_control_iterations': max_control_iterations,
        'legacy_fault_mapping': bool(
            cfg['circuit'].get('legacy_fault_mapping', False)
        ),
        'interim_path': str(paths['interim']),
        'processed_path': str(paths['processed']),
        'results_dir': str(paths['results']),
        'distribution_metadata_path': (
            str(paths['distribution_metadata']) if patterns is not None else None
        ),
    }


def main(
    cfg_path: str = 'configs/simulation_ieee13.yaml',
    overwrite: bool = False,
    confirm_generation: bool = False,
    allow_legacy_generation: bool = False,
) -> Path:
    path, cfg = load_config(cfg_path)
    paths = resolve_paths(path, cfg)
    plan = describe_plan(path)
    if not plan['converged']:
        raise RuntimeError(f'Circuito não convergiu: {paths["dss"]}')
    if not confirm_generation:
        raise PermissionError(
            'Geração bloqueada pelo portão humano. '
            'Use --confirm-generation somente após autorização explícita.'
        )
    if plan['legacy_fault_mapping'] and not allow_legacy_generation:
        raise PermissionError(
            'Geração legada bloqueada para preservar o dataset histórico. '
            'Use --allow-legacy-generation apenas para reprodução deliberada.'
        )

    guarded_paths = [paths['interim'], paths['processed']]
    if 'spatial_distributions' in cfg:
        guarded_paths.append(paths['distribution_metadata'])
    existing = [p for p in guarded_paths if p.exists()]
    if existing and not overwrite:
        joined = ', '.join(str(p) for p in existing)
        raise FileExistsError(
            f'Saída já existente: {joined}. Use --overwrite somente após autorização.'
        )

    circuit = cfg['circuit']
    audit_records = []
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    df_raw = run_all(
        str(paths['dss']),
        cfg['variations']['fault_resistance_ohms'],
        cfg['variations']['load_multipliers'],
        BI_PAIRS,
        circuit.get('exclude_buses', []),
        circuit.get('measurement_element', 'Line.650632'),
        int(circuit.get('measurement_terminal', 1)),
        bool(circuit.get('legacy_fault_mapping', False)),
        cfg.get('system_id'),
        spatial_distribution_config=cfg.get('spatial_distributions'),
        max_control_iterations=int(circuit.get('max_control_iterations', 100)),
        independent_scenarios=bool(circuit.get('independent_scenarios', True)),
        freeze_controls=bool(circuit.get('freeze_controls_during_fault', True)),
        audit_callback=audit_records.append,
    )
    if len(df_raw) != plan['scenario_count']:
        raise RuntimeError(
            f'Contagem inesperada: {len(df_raw)} != {plan["scenario_count"]}'
        )

    paths['interim'].parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_parquet(paths['interim'])
    paths['interim'].with_name('auditoria_controles.json').write_text(
        json.dumps({'policy': ('fresh_STATIC_pre_OFF_fault' if
                   circuit.get('independent_scenarios', True) and
                   circuit.get('freeze_controls_during_fault', True) else 'configuracao_alternativa'),
                   'cases': audit_records},
                   ensure_ascii=False), encoding='utf-8')
    print(f'Raw: {len(df_raw)} amostras -> {paths["interim"]}')

    if 'spatial_distributions' in cfg:
        d = load_circuit(paths['dss'])
        nominal_loads = capture_nominal_loads(d)
        spatial_cfg = cfg['spatial_distributions']
        amplitude = float(spatial_cfg.get('amplitude', 0.10))
        spatial_seed = int(spatial_cfg.get('seed', cfg.get('seed', 42)))
        patterns = generate_spatial_patterns(
            nominal_loads,
            int(spatial_cfg.get('heterogeneous_count', 0)),
            amplitude,
            spatial_seed,
        )
        metadata = patterns_metadata(
            nominal_loads, patterns, amplitude, spatial_seed
        )
        paths['distribution_metadata'].parent.mkdir(parents=True, exist_ok=True)
        paths['distribution_metadata'].write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )

    df_feat = build_features(df_raw)
    paths['processed'].parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(paths['processed'])
    print(
        f'Processed: {len(df_feat)} amostras, {df_feat.shape[1]} colunas '
        f'-> {paths["processed"]}'
    )
    return paths['processed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='configs/simulation_ieee13.yaml')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--confirm-generation', action='store_true')
    parser.add_argument('--allow-legacy-generation', action='store_true')
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps(describe_plan(args.config), ensure_ascii=False, indent=2))
    else:
        main(
            args.config,
            overwrite=args.overwrite,
            confirm_generation=args.confirm_generation,
            allow_legacy_generation=args.allow_legacy_generation,
        )
