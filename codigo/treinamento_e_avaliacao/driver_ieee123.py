"""Treino IEEE123 com protocolo vigente e retomada por ajuste, sem alterar 13/34."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codigo/src'))

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from opendss_tcc.features.cartesian import RAW_COLUMNS, CartesianFeatures
from opendss_tcc.models.evaluation import metrics
from opendss_tcc.models.factories import corrected_factory
from treino_cv import _training_failures, IncompleteTrainingError

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / 'sistemas/ieee123/dados/treino_cv/dataset.parquet'
DEFAULT_OUT = ROOT / 'execucoes/ieee123/desenvolvimento'
DEFAULT_FINAL_OUT = ROOT / 'execucoes/ieee123/final'
DEFAULT_TEST_DATASET = DEFAULT_FINAL_OUT / 'test/dataset.parquet'
FEATURES = {
    'fault_phase': ['fault_phase'],
    'fault_sym': ['fault_sym'],
    'phase+sym': ['fault_phase', 'fault_sym'],
    'phase+sym+delta': ['fault_phase', 'fault_sym', 'delta_phase', 'delta_sym'],
}
MODELS = ('RandomForest', 'XGBoost', 'MLP')
REGIMES = ('controle_p0', 'enriquecido_p0_p5')
TASKS = ('type', 'bus')
METRIC_NAMES = ('accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro')
PLAN_PIN = '62496ff0a2cd83ab3e6826bc9e1f81d557f66bb21d308e6d29ab8c083f4a6943'
_STATUS_LOCK = threading.Lock()
_STATUS = {}
_STATUS_PATH = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False, default=json_default)


def hash_value(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def replace_atomic(temp, target):
    for attempt in range(20):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=json_default) + '\n', encoding='utf-8')
    replace_atomic(temp, path)


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    replace_atomic(temp, path)


def resource_status():
    process = psutil.Process()
    info = process.memory_info()
    return dict(rss_bytes=info.rss,
                process_peak_working_set_bytes=int(getattr(info, 'peak_wset', info.rss)),
                available_memory_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(ROOT).free)


def status(**changes):
    global _STATUS
    with _STATUS_LOCK:
        _STATUS.update(changes)
        _STATUS.update(updated_utc=utc(), **resource_status())
        if _STATUS_PATH is not None:
            try:
                dump(_STATUS_PATH, _STATUS)
            except PermissionError:
                # Uma leitura breve do painel não invalida um ajuste demorado.
                pass


class Heartbeat:
    """Observa um único ajuste; não limita épocas nem interrompe treinamento."""
    def __init__(self, model):
        self.model = model
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.observe, daemon=True)

    def observe(self):
        while not self.stop.wait(10):
            estimator = self.model.named_steps['model']
            observed = {}
            for name in ('n_iter_', 'loss_', 'best_validation_score_'):
                if hasattr(estimator, name):
                    value = getattr(estimator, name)
                    if np.isscalar(value) and (not isinstance(value, float) or np.isfinite(value)):
                        observed[name] = value
            status(observed_training_state=observed)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=2)


def parameter_strings(model):
    return {k: str(v) for k, v in model.get_params(deep=True).items()}


def factory(name, feature):
    return corrected_factory(name, FEATURES[feature], seed=42, n_jobs=2)


def sources():
    names = [
        'codigo/src/opendss_tcc/models/factories.py',
        'codigo/src/opendss_tcc/models/evaluation.py',
        'codigo/src/opendss_tcc/features/cartesian.py',
        'codigo/src/opendss_tcc/features/symmetric.py',
        'codigo/treinamento_e_avaliacao/treino_cv.py',
    ]
    return {name: sha(ROOT/name) for name in names}


def dataset_frame(path, expected_rows):
    frame = pd.read_parquet(path)
    assert len(frame) == expected_rows, (len(frame), expected_rows)
    assert frame.shape[1] == 81, frame.shape
    assert frame['system_id'].eq('ieee123').all()
    assert frame.bus.astype(str).nunique() == 124
    assert frame.type_id.nunique() == 11
    assert set(frame.distribution_id) == {f'P{i}' for i in range(6)}
    assert set(frame.rf) == {.0001}
    keys = ['bus', 'type_id', 'rf', 'load_mult', 'distribution_id']
    assert not frame[keys].duplicated().any()
    assert np.isfinite(frame[RAW_COLUMNS].to_numpy(dtype=float)).all()
    return frame


def training_manifest(dataset, out):
    path = out/'treino_manifesto.json'
    payload = {
        'schema': 'tcc.ieee123.treino-manifesto.v1', 'system': 'ieee123',
        'dataset': str(dataset.resolve()), 'dataset_sha256': sha(dataset),
        'protocol_reference_hash': PLAN_PIN, 'source_hashes': sources(),
        'driver_sha256': sha(__file__), 'seed': 42,
        'parameters': {f'{feature}/{name}': parameter_strings(factory(name, feature))
                       for feature in FEATURES for name in MODELS},
        'runtime': {'python': sys.version, 'executable': Path(sys.executable).name,
                    'platform': platform.platform(), 'blas_threads': 1,
                    'factory_n_jobs': 2,
                    'versions': {name: importlib.metadata.version(name) for name in
                                 ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl']}},
        'fit_policy': 'Um ajuste ativo; modelos CV descartados após salvar métricas/previsões/scales.',
    }
    fingerprint = hash_value(payload)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['fingerprint'] == fingerprint, 'Fonte/dados/ambiente mudaram; retomada não automática.'
        return saved
    result = dict(payload, fingerprint=fingerprint, created_utc=utc())
    dump(path, result)
    return result


def bundle(frame, regime, task):
    mask = (frame.distribution_id.to_numpy() == 'P0'
            if regime == 'controle_p0' else np.ones(len(frame), dtype=bool))
    global_rows = np.flatnonzero(mask)
    part = frame.iloc[global_rows]
    target = 'type_id' if task == 'type' else 'bus'
    encoder = LabelEncoder()
    y = encoder.fit_transform(part[target].astype(str))
    X = part[RAW_COLUMNS].to_numpy()
    group_col = 'load_mult' if regime == 'controle_p0' else 'distribution_id'
    groups = part[group_col].to_numpy()
    mode = 'stratified_group' if regime == 'controle_p0' else 'leave_one_group_out'
    heldout = 1.15 if regime == 'controle_p0' else 'P5'
    return dict(X=X, y=y, classes=encoder.classes_.tolist(), groups=groups,
                global_rows=global_rows, group_column=group_col, group_mode=mode,
                test_group=heldout, frame=part)


def splits(data):
    X, y, groups = data['X'], data['y'], data['groups']
    if data['group_mode'] == 'leave_one_group_out':
        iterator = LeaveOneGroupOut().split(X, y, groups)
    else:
        iterator = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(X, y, groups)
    result = []
    for fold, (tr, te) in enumerate(iterator):
        assert not (set(groups[tr]) & set(groups[te]))
        result.append((f'cv_{fold:02d}', tr, te,
                       sorted({str(v) for v in groups[te]})))
    test = groups == data['test_group']
    assert test.any() and (~test).any()
    result.append(('matriz', np.flatnonzero(~test), np.flatnonzero(test),
                   [str(data['test_group'])]))
    return result


def warning_record(items):
    return [{'category': type(v.message).__name__, 'message': str(v.message),
             'filename': v.filename, 'lineno': v.lineno} for v in items]


def fit_signature(manifest, job_id, fit_name, data, tr, te, feature, name):
    return hash_value({'manifest': manifest['fingerprint'], 'job_id': job_id,
                       'fit': fit_name, 'train_rows': data['global_rows'][tr].tolist(),
                       'test_rows': data['global_rows'][te].tolist(),
                       'classes': data['classes'], 'parameters': parameter_strings(factory(name, feature))})


def run_fit(out, manifest, job_id, fit_name, data, tr, te, feature, name):
    folder = out/'checkpoints'/job_id
    checkpoint = folder/f'{fit_name}.json'
    prediction = folder/f'{fit_name}.npz'
    signature = fit_signature(manifest, job_id, fit_name, data, tr, te, feature, name)
    if checkpoint.exists():
        record = json.loads(checkpoint.read_text(encoding='utf-8'))
        assert record['status'] == 'PASS' and record['signature'] == signature
        assert prediction.exists() and sha(prediction) == record['predictions_sha256']
        return record, True
    model = factory(name, feature)
    begin = time.perf_counter()
    started = utc()
    resources_before = resource_status()
    status(state='fit_em_andamento', active_job=job_id, active_fit=fit_name,
           fit_started_utc=started, observed_training_state={})
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with threadpool_limits(limits=1, user_api='blas'), Heartbeat(model):
                model.fit(data['X'][tr], data['y'][tr])
                pred = model.predict(data['X'][te])
        assert pred.shape == data['y'][te].shape and np.isfinite(pred).all()
        scores = {k: float(v) for k, v in metrics(data['y'][te], pred).items()}
        scales = model.named_steps['features'].scales_
        expected_scales = CartesianFeatures(tuple(FEATURES[feature])).fit(data['X'][tr]).scales_
        np.testing.assert_array_equal(scales, expected_scales)
        assert set(data['y'][te]) == set(range(len(data['classes'])))
        cm = confusion_matrix(data['y'][te], pred).tolist()
        save_npz(prediction, train_rows=data['global_rows'][tr], test_rows=data['global_rows'][te],
                 y_true=data['y'][te], y_pred=pred, classes=np.asarray(data['classes'], dtype=str))
        estimator = model.named_steps['model']
        record = {'schema': 'tcc.ieee123.fit-cv.v1', 'status': 'PASS', 'signature': signature,
                  'job_id': job_id, 'fit': fit_name, 'model': name, 'feature': feature,
                  'metrics': scores, 'confusion_matrix': cm, 'classes': data['classes'],
                  'train_rows': len(tr), 'test_rows': len(te), 'scales': scales.tolist(),
                  'scales_from_train_verified': True,
                  'train_groups': sorted({str(v) for v in data['groups'][tr]}),
                  'test_groups': sorted({str(v) for v in data['groups'][te]}),
                  'predictions': str(prediction), 'predictions_sha256': sha(prediction),
                  'parameters': parameter_strings(model), 'warnings': warning_record(captured),
                  'epochs': int(estimator.n_iter_) if hasattr(estimator, 'n_iter_') else None,
                  'started_utc': started, 'completed_utc': utc(),
                  'elapsed_seconds': time.perf_counter()-begin,
                  'resources_before': resources_before, 'resources_after': resource_status()}
        dump(checkpoint, record)
        return record, False
    except BaseException as exc:
        dump(folder/f'{fit_name}.falha.json', {'status': 'FAIL', 'signature': signature,
             'job_id': job_id, 'fit': fit_name, 'exception': str(exc),
             'traceback': traceback.format_exc(), 'started_utc': started, 'failed_utc': utc(),
             'resources': resource_status()})
        raise
    finally:
        del model
        gc.collect()


def aggregate_job(records, data):
    cv = [v for v in records if v['fit'].startswith('cv_')]
    single = next(v for v in records if v['fit'] == 'matriz')
    result = {k: [float(np.mean([v['metrics'][k] for v in cv])),
                  float(np.std([v['metrics'][k] for v in cv]))] for k in METRIC_NAMES}
    result['fold_count'] = len(cv)
    result['held_out_groups'] = [v['test_groups'] for v in cv]
    return {'cv': result, 'final': single['metrics'],
            'confusion_matrix': single['confusion_matrix'], 'classes': data['classes'],
            'held_out_group': str(data['test_group']), 'group_column': data['group_column'],
            'group_mode': data['group_mode'], 'test_group': str(data['test_group']),
            'checkpoint_fits': len(records), 'warning_count': sum(len(v['warnings']) for v in records),
            'epochs': [v['epochs'] for v in records],
            'checkpoints': [{'fit': v['fit'], 'signature': v['signature'],
                             'predictions_sha256': v['predictions_sha256']} for v in records]}


def development(args):
    global _STATUS_PATH
    dataset, out = args.dataset.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _STATUS_PATH = out/'status_treino.json'
    frame = dataset_frame(dataset, 128466)
    assert len(frame[frame.distribution_id == 'P0']) == 21411
    assert len(frame.load_mult.unique()) == 27
    manifest = training_manifest(dataset, out)
    status(phase='desenvolvimento', state='iniciando', total_fits=312, completed_fits=0,
           run_fingerprint=manifest['fingerprint'], started_or_resumed_utc=utc())
    all_results, completed, resumed = {}, 0, 0
    observed_fit_seconds = []
    for regime in REGIMES:
        for feature in FEATURES:
            for task in TASKS:
                data = bundle(frame, regime, task)
                partition = splits(data)
                key = f'{task}__{regime}__{feature}'
                all_results[key] = {}
                for name in MODELS:
                    job_id = key+'__'+name
                    records = []
                    for fit_name, tr, te, _ in partition:
                        record, was_resumed = run_fit(out, manifest, job_id, fit_name,
                                                     data, tr, te, feature, name)
                        records.append(record)
                        completed += 1
                        resumed += int(was_resumed)
                        if not was_resumed:
                            observed_fit_seconds.append(record['elapsed_seconds'])
                        mean_seconds = (float(np.mean(observed_fit_seconds))
                                        if observed_fit_seconds else None)
                        status(state='checkpoint_concluido', completed_fits=completed,
                               resumed_fits=resumed, last_fit_seconds=record['elapsed_seconds'],
                               estimated_remaining_seconds=(None if mean_seconds is None
                                                            else mean_seconds*(312-completed)),
                               estimate_basis='Média observada dos ajustes novos desta execução; provisória, pois modelos e regimes têm custos diferentes.',
                               active_job=job_id, active_fit=fit_name)
                        print(f'{completed}/312 {job_id}/{fit_name} '
                              f'{"retomado" if was_resumed else "concluído"} '
                              f'{record["elapsed_seconds"]:.1f}s', flush=True)
                    all_results[key][name] = aggregate_job(records, data)
                    dump(out/'conditions'/f'{job_id}.json', all_results[key][name])
                del data, partition
                gc.collect()
    failures = _training_failures(all_results)
    if completed != 312 or len(all_results) != 16 or failures:
        raise IncompleteTrainingError(failures or ['Número de ajustes/condições incompleto.'], all_results)
    assert manifest['dataset_sha256'] == sha(dataset) and manifest['source_hashes'] == sources()
    dump(out/'all_results.json', all_results)
    complete = {'schema': 'tcc.ieee123.treino-concluido.v1', 'status': 'PASS',
                'fits': completed, 'conditions': 48, 'resumed_fits': resumed,
                'dataset_sha256': manifest['dataset_sha256'], 'run_fingerprint': manifest['fingerprint'],
                'all_results_sha256': sha(out/'all_results.json'), 'completed_utc': utc()}
    dump(out/'treino_concluido.json', complete)
    status(state='concluido', completed_fits=completed, active_job=None, active_fit=None,
           all_results_sha256=complete['all_results_sha256'])


def freeze(args):
    dataset, out = args.dataset.resolve(), args.out.resolve()
    result_path = out/'all_results.json'
    complete = json.loads((out/'treino_concluido.json').read_text(encoding='utf-8'))
    assert complete['status'] == 'PASS' and complete['fits'] == 312
    assert sha(dataset) == complete['dataset_sha256']
    assert sha(result_path) == complete['all_results_sha256']
    path = out/'escolhas_finais_congeladas.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        assert saved['source_results_sha256'] == sha(result_path)
        assert saved['contentHash'] == hash_value({k:v for k,v in saved.items() if k != 'contentHash'})
        print(f'Plano congelado existente: {path}', flush=True)
        return
    planned_test = None if args.planned_test_dataset is None else args.planned_test_dataset.resolve()
    if planned_test is not None:
        assert not planned_test.exists(), 'O plano deve ser congelado antes da geração da prova.'
    results = json.loads(result_path.read_text(encoding='utf-8'))
    choices, champions = [], []
    for regime in REGIMES:
        for task in TASKS:
            candidates = []
            for name in MODELS:
                feature = max(FEATURES, key=lambda f: results[f'{task}__{regime}__{f}'][name]['cv']['f1_macro'][0])
                record = results[f'{task}__{regime}__{feature}'][name]
                job = {'id': '__'.join(['ieee123', task, regime, name]), 'system': 'ieee123',
                       'task': task, 'regime': regime, 'model': name, 'feature': feature,
                       'groups': FEATURES[feature], 'cv_f1': record['cv']['f1_macro'][0],
                       'cv_accuracy': record['cv']['accuracy'][0],
                       'parameters': parameter_strings(factory(name, feature))}
                choices.append(job)
                candidates.append(job)
            champions.append(max(candidates, key=lambda j:j['cv_f1'])['id'])
    for job in choices:
        job['primary'] = job['id'] in champions
    levels = sorted(pd.read_parquet(dataset, columns=['load_mult']).load_mult.unique().tolist())
    plan = {'schema': 'tcc.ieee123.escolhas-finais.v1', 'estado': 'congelado_antes_teste',
            'created_utc': utc(), 'source_results_sha256': sha(result_path),
            'dataset_sha256': sha(dataset), 'training_manifest_sha256': sha(out/'treino_manifesto.json'),
            'choices': choices, 'primary_champions': champions, 'seed': 42,
            'planned_test_dataset': None if planned_test is None else str(planned_test),
            'load_multipliers': [round((a+b)/2, 10) for a,b in zip(levels, levels[1:])],
            'expected_test_rows': 123708, 'source_hashes': sources(),
            'selection': 'Maior F1 macro médio CV por modelo/objetivo/regime; empates pela ordem FEATURES; campeão por RF/XGBoost/MLP.',
            'scope': 'Interpolação nos mesmos circuito/perfis, sem validação de campo e sem reescolha pela prova.'}
    assert len(choices) == 12 and len(champions) == 4 and len(plan['load_multipliers']) == 26
    plan['contentHash'] = hash_value(plan)
    dump(path, plan)
    print(f'CONGELADAS 12 escolhas antes da prova: {path}', flush=True)


def final_model(args, plan, job, train, output):
    identifier = job['id']
    model_path = output/'models'/f'{identifier}.joblib'
    fit_path = output/'fits'/f'{identifier}.json'
    mask = train.distribution_id.eq('P0').to_numpy() if job['regime'] == 'controle_p0' else np.ones(len(train), dtype=bool)
    train_rows = np.flatnonzero(mask)
    part = train.iloc[train_rows]
    target = 'type_id' if job['task'] == 'type' else 'bus'
    encoder = LabelEncoder()
    y = encoder.fit_transform(part[target].astype(str))
    X = part[RAW_COLUMNS].to_numpy()
    signature = hash_value({'plan': plan['contentHash'], 'job': job,
                            'train_rows': train_rows.tolist(), 'classes': encoder.classes_.tolist()})
    if fit_path.exists():
        fit = json.loads(fit_path.read_text(encoding='utf-8'))
        assert fit['signature'] == signature and fit['status'] == 'PASS'
        assert sha(model_path) == fit['model_sha256']
        return model_path, fit
    model = factory(job['model'], job['feature'])
    assert parameter_strings(model) == job['parameters']
    started, timer = utc(), time.perf_counter()
    resources_before = resource_status()
    status(state='ajuste_final_em_andamento', active_job=identifier, fit_started_utc=started)
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with threadpool_limits(limits=1, user_api='blas'), Heartbeat(model):
                model.fit(X, y)
        scales = model.named_steps['features'].scales_
        np.testing.assert_array_equal(CartesianFeatures(tuple(job['groups'])).fit(X).scales_, scales)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temp = model_path.with_name(model_path.name+'.tmp')
        joblib.dump(model, temp, compress=3)
        replace_atomic(temp, model_path)
        estimator = model.named_steps['model']
        fit = {'schema': 'tcc.ieee123.fit-final.v1', 'status': 'PASS', 'signature': signature,
               'job': job, 'train_rows': len(part), 'train_indices': train_rows.tolist(),
               'classes': encoder.classes_.tolist(), 'scales': scales.tolist(),
               'parameters': parameter_strings(model), 'warnings': warning_record(captured),
               'epochs': int(estimator.n_iter_) if hasattr(estimator, 'n_iter_') else None,
               'model_sha256': sha(model_path), 'started_utc': started, 'completed_utc': utc(),
               'elapsed_seconds': time.perf_counter()-timer,
               'resources_before': resources_before, 'resources_after': resource_status()}
        dump(fit_path, fit)
        return model_path, fit
    except BaseException as exc:
        dump(output/'fits'/f'{identifier}.falha.json', {'status': 'FAIL', 'exception': str(exc),
             'traceback': traceback.format_exc(), 'signature': signature, 'failed_utc': utc()})
        raise
    finally:
        del model, X
        gc.collect()


def evaluate_final(plan, job, model_path, fit, test, output):
    identifier = job['id']
    evaluation = output/'evaluation'/f'{identifier}.json'
    prediction_path = output/'predictions'/f'{identifier}.parquet'
    if evaluation.exists():
        saved = json.loads(evaluation.read_text(encoding='utf-8'))
        assert saved['model_sha256'] == fit['model_sha256']
        assert saved['plan_contentHash'] == plan['contentHash']
        assert sha(prediction_path) == saved['predictions_sha256']
        return saved
    frame = test[test.distribution_id == 'P0'] if job['regime'] == 'controle_p0' else test
    target = 'type_id' if job['task'] == 'type' else 'bus'
    mapping = {v:i for i,v in enumerate(fit['classes'])}
    labels = frame[target].astype(str)
    assert set(labels) == set(mapping)
    y = labels.map(mapping).to_numpy(dtype=int)
    X = frame[RAW_COLUMNS].to_numpy()
    model = joblib.load(model_path)
    try:
        assert parameter_strings(model) == job['parameters']
        np.testing.assert_array_equal(model.named_steps['features'].scales_, fit['scales'])
        with threadpool_limits(limits=1, user_api='blas'):
            pred = model.predict(X)
        scores = {k:float(v) for k,v in metrics(y,pred).items()}
        cm = confusion_matrix(y,pred,labels=np.arange(len(mapping)))
        assert int(cm.sum()) == len(frame)
        by_groups = []
        for column in ['distribution_id', 'load_mult']:
            for value in sorted(frame[column].unique()):
                mask = frame[column].to_numpy() == value
                by_groups.append({'group': column, 'value': str(value), 'n': int(mask.sum()),
                                  **{k:float(v) for k,v in metrics(y[mask],pred[mask]).items()}})
        predictions = frame[['bus','type_id','load_mult','distribution_id','rf']].copy()
        predictions.insert(0, 'dataset_row', frame.index.to_numpy())
        predictions['real'] = y
        predictions['predicted'] = pred
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temp = prediction_path.with_name(prediction_path.name+'.tmp')
        predictions.to_parquet(temp,index=False)
        replace_atomic(temp,prediction_path)
        report = classification_report(y,pred,labels=np.arange(len(mapping)),
                                       target_names=fit['classes'],output_dict=True,zero_division=0)
        result = {**job, 'status': 'PASS', 'plan_contentHash': plan['contentHash'],
                  'metrics': scores, 'test_rows': len(frame), 'train_rows': fit['train_rows'],
                  'classes': fit['classes'], 'confusion_matrix': cm.tolist(), 'by_groups': by_groups,
                  'per_class': report, 'warnings': fit['warnings'], 'epochs': fit['epochs'],
                  'scales_from_training_verified': True, 'model_sha256': fit['model_sha256'],
                  'predictions_sha256': sha(prediction_path), 'completed_utc': utc()}
        dump(evaluation,result)
        return result
    finally:
        del model,X
        gc.collect()


def final(args):
    global _STATUS_PATH
    dataset, out, test_path = args.dataset.resolve(), args.out.resolve(), args.test_dataset.resolve()
    output = args.final_out.resolve()
    output.mkdir(parents=True,exist_ok=True)
    _STATUS_PATH = out/'status_treino.json'
    plan_path = out/'escolhas_finais_congeladas.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['schema'] == 'tcc.ieee123.escolhas-finais.v1' and plan['estado'] == 'congelado_antes_teste'
    assert plan['contentHash'] == hash_value({k:v for k,v in plan.items() if k != 'contentHash'})
    assert plan['dataset_sha256'] == sha(dataset) and plan['source_hashes'] == sources()
    assert plan['source_results_sha256'] == sha(out/'all_results.json')
    if plan['planned_test_dataset'] is not None:
        assert Path(plan['planned_test_dataset']).resolve() == test_path
    created = datetime.fromisoformat(plan['created_utc']).timestamp()
    assert test_path.stat().st_mtime >= created, 'Prova deve ser gerada após congelamento.'
    if args.generation_manifest:
        generation = json.loads(args.generation_manifest.read_text(encoding='utf-8'))
        references = canonical(generation)
        assert plan['contentHash'] in references or sha(plan_path) in references
    train, test = dataset_frame(dataset,128466), dataset_frame(test_path,123708)
    assert not set(train.load_mult) & set(test.load_mult)
    assert sorted(test.load_mult.unique()) == plan['load_multipliers']
    opening = {'schema':'tcc.ieee123.abertura-prova.v1','plan_contentHash':plan['contentHash'],
               'plan_sha256':sha(plan_path),'test_dataset_sha256':sha(test_path),
               'development_dataset_sha256':sha(dataset),'created_utc':utc()}
    opening_path = output/'abertura_prova.json'
    if opening_path.exists():
        previous = json.loads(opening_path.read_text(encoding='utf-8'))
        assert all(previous[k] == opening[k] for k in opening if k != 'created_utc')
    else:
        dump(opening_path,opening)
    status(phase='final',state='iniciando',total_fits=12,completed_fits=0,
           plan_contentHash=plan['contentHash'])
    # Os ajustes finais só recebem desenvolvimento. A prova nunca é passada a fit.
    fitted=[]
    for index,job in enumerate(plan['choices']):
        file,fit = final_model(args,plan,job,train,output)
        fitted.append((job,file,fit))
        status(completed_fits=index+1,state='ajuste_final_concluido',active_job=job['id'])
        print(f'{index+1}/12 ajuste final {job["id"]} concluído',flush=True)
    results=[]
    for job,file,fit in fitted:
        status(state='avaliando_prova',active_job=job['id'])
        result=evaluate_final(plan,job,file,fit,test,output)
        results.append(result)
        print(job['id'],result['metrics'],flush=True)
    assert len(results)==12 and all(v['status']=='PASS' for v in results)
    assert sha(test_path)==opening['test_dataset_sha256'] and sha(dataset)==plan['dataset_sha256']
    dump(output/'resultados.json',results)
    pd.DataFrame([{k:r[k] for k in ['id','system','task','regime','model','feature','primary']}
                  | r['metrics'] for r in results]).to_csv(output/'resultados.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'id':r['id'],**g} for r in results for g in r['by_groups']]).to_csv(
        output/'por_perfil_e_carga.csv',index=False,encoding='utf-8-sig')
    complete={'schema':'tcc.ieee123.avaliacao-concluida.v1','status':'PASS',
              'models':12,'completed_utc':utc(),'plan_contentHash':plan['contentHash'],
              'results_sha256':sha(output/'resultados.json'),
              'test_dataset_sha256':opening['test_dataset_sha256'],
              'scope':'Teste de cargas intermediárias após escolhas fixadas; não validação em campo.'}
    dump(output/'avaliacao_concluida.json',complete)
    status(state='concluido',completed_fits=12,active_job=None,results_sha256=complete['results_sha256'])


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    sub=result.add_subparsers(dest='command',required=True)
    for command in ['desenvolvimento','congelar','final']:
        p=sub.add_parser(command)
        p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
        p.add_argument('--out',type=Path,default=DEFAULT_OUT)
        if command=='congelar':
            p.add_argument('--planned-test-dataset',type=Path,default=DEFAULT_TEST_DATASET)
        if command=='final':
            p.add_argument('--test-dataset',type=Path,required=True)
            p.add_argument('--final-out',type=Path,default=DEFAULT_FINAL_OUT)
            p.add_argument('--generation-manifest',type=Path)
    return result


def main():
    args=parser().parse_args()
    try:
        {'desenvolvimento':development,'congelar':freeze,'final':final}[args.command](args)
    except BaseException as exc:
        if not isinstance(exc,SystemExit) or exc.code:
            status(state='falha',exception=str(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
