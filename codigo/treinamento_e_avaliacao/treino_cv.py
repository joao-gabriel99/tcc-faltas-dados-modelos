"""Trains classifier (type_id) and localizer (bus) with RF, XGB, MLP."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from opendss_tcc.features.build import FEATURE_GROUPS, select_features
from opendss_tcc.models.evaluation import cv_evaluate, fit_and_report, metrics
from opendss_tcc.features.cartesian import RAW_COLUMNS
from opendss_tcc.models.factories import corrected_factory


EXPECTED_MODELS = ('RandomForest', 'XGBoost', 'MLP')


class IncompleteTrainingError(RuntimeError):
    """Indica que ao menos uma combinação não pode ser publicada como resultado."""

    def __init__(self, failures: list[str], partial_results: dict[str, dict]) -> None:
        self.failures = failures
        self.partial_results = partial_results
        details = '\n'.join(f'  - {failure}' for failure in failures)
        super().__init__(
            'Treinamento incompleto; all_results.json não foi escrito.\n'
            f'{details}'
        )


def _training_failures(all_results: dict[str, dict]) -> list[str]:
    """Consolida falhas preservadas por tarefa antes de permitir a publicação."""
    failures: list[str] = []
    for task_key, models in all_results.items():
        if not isinstance(models, dict):
            failures.append(f'{task_key}: bloco de modelos ausente ou inválido')
            continue
        missing = [name for name in EXPECTED_MODELS if name not in models]
        if missing:
            failures.append(f'{task_key}: modelos ausentes: {", ".join(missing)}')
        for model_name in EXPECTED_MODELS:
            result = models.get(model_name)
            if isinstance(result, dict) and 'error' in result:
                failures.append(f'{task_key}/{model_name}: {result["error"]}')
            elif not isinstance(result, dict):
                failures.append(f'{task_key}/{model_name}: resultado ausente ou inválido')
    return failures


def _prep(df: pd.DataFrame, feature_groups: list[str]):
    X = df[RAW_COLUMNS].to_numpy()
    return X


def run_task(
    df: pd.DataFrame,
    target_col: str,
    feature_groups: list[str],
    task_name: str,
    seed: int = 42,
    group_col: str | None = None,
    group_mode: str | None = None,
    test_group=None,
) -> dict:
    X = _prep(df, feature_groups)
    le = LabelEncoder()
    y = le.fit_transform(df[target_col].astype(str))
    groups = None if group_col is None else df[group_col].to_numpy()
    print(f'\n=== {task_name} ===')
    print(f'Leitura: {X.shape}; entradas cartesianas do modelo: {12*len(feature_groups)}; classes: {len(le.classes_)}')

    results: dict[str, dict] = {}

    for name, factory in [
        ('RandomForest', lambda: corrected_factory('RandomForest', feature_groups, seed)),
        ('XGBoost', lambda: corrected_factory('XGBoost', feature_groups, seed)),
        ('MLP', lambda: corrected_factory('MLP', feature_groups, seed)),
    ]:
        validation_label = (
            'Leave-One-Group-Out (6 distribuições)'
            if group_mode == 'leave_one_group_out'
            else 'validação cruzada de 5 partições'
        )
        print(f'  -> {name}: {validation_label}...')
        try:
            cv = cv_evaluate(
                factory,
                X,
                y,
                n_splits=5,
                seed=seed,
                groups=groups,
                group_mode=group_mode,
            )
            print(f'     accuracy={cv["accuracy"][0]:.4f} +/- {cv["accuracy"][1]:.4f}  f1_macro={cv["f1_macro"][0]:.4f}')
            # Also a single fit for confusion matrix
            single = fit_and_report(
                factory(),
                X,
                y,
                seed=seed,
                groups=groups,
                test_group=test_group,
            )
            results[name] = {
                'cv': cv,
                'final': single['metrics'],
                'confusion_matrix': single['confusion_matrix'],
                'classes': le.classes_.tolist(),
                'held_out_group': single['held_out_group'],
                'group_column': group_col,
                'group_mode': group_mode,
                'test_group': None if test_group is None else str(test_group),
            }
        except Exception as e:
            print(f'     FAILED: {e}')
            results[name] = {'error': str(e)}

    return results


def main(dataset_path: str = 'data/processed/dataset.parquet', out_dir: str = 'results') -> dict:
    df = pd.read_parquet(dataset_path)
    print(f'Loaded {len(df)} samples, {df.shape[1]} cols')

    all_results: dict[str, dict] = {}

    # Compare feature group impact: phase-only vs +symmetric vs +delta
    feature_configs = {
        'fault_phase': ['fault_phase'],
        'fault_sym': ['fault_sym'],
        'phase+sym': ['fault_phase', 'fault_sym'],
        'phase+sym+delta': ['fault_phase', 'fault_sym', 'delta_phase', 'delta_sym'],
    }

    if 'distribution_id' in df.columns:
        regimes = {
            'controle_p0': (
                df[df['distribution_id'] == 'P0'].copy(),
                'load_mult',
                'stratified_group',
                1.15,
            ),
            'enriquecido_p0_p5': (
                df.copy(),
                'distribution_id',
                'leave_one_group_out',
                'P5',
            ),
        }
    else:
        regimes = {'legado': (df, None, None, None)}

    for regime_name, (
        regime_df, group_col, group_mode, test_group
    ) in regimes.items():
        for feat_name, feat_groups in feature_configs.items():
            prefix = f'{regime_name}__{feat_name}'
            all_results[f'type__{prefix}'] = run_task(
                regime_df,
                'type_id',
                feat_groups,
                f'TYPE: {regime_name}: {feat_name}',
                group_col=group_col,
                group_mode=group_mode,
                test_group=test_group,
            )
            all_results[f'bus__{prefix}'] = run_task(
                regime_df,
                'bus',
                feat_groups,
                f'BUS: {regime_name}: {feat_name}',
                group_col=group_col,
                group_mode=group_mode,
                test_group=test_group,
            )

    failures = _training_failures(all_results)
    if failures:
        raise IncompleteTrainingError(failures, all_results)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / 'all_results.json'
    json_path.write_text(
        json.dumps(all_results, indent=2, default=str), encoding='utf-8'
    )
    print(f'\nSaved: {json_path}')
    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', nargs='?', default='data/processed/dataset.parquet')
    parser.add_argument('out_dir', nargs='?', default='results')
    args = parser.parse_args()
    main(args.dataset_path, args.out_dir)
