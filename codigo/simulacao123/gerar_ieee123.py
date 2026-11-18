"""Geração IEEE 123 em processos isolados, adaptada para os caminhos deste pacote."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import shutil
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
CODE = ROOT / 'codigo'
OUTPUT = Path()
CONFIG = ROOT / 'sistemas/ieee123/metadados/configuracao_treino.yaml'
RAW = ROOT / 'circuitos/ieee123'
sys.path.insert(0, str(CODE / 'src'))
import numpy as np
import pandas as pd
import yaml
from opendss_tcc.circuit.loader import load_circuit
from opendss_tcc.simulation.scenarios import make_scenarios, Scenario
from opendss_tcc.simulation.runner import run_scenario
from opendss_tcc.simulation.fault_injector import create_reusable_fault
from opendss_tcc.simulation.load_distributions import (
    NominalLoad, SpatialPattern, capture_nominal_loads, generate_spatial_patterns,
    patterns_metadata)
from opendss_tcc.features.build import build_features

RAW_KEYS = [f'{s}_{k}{p}_{t}' for s in ('pre', 'fault') for p in (1,2,3)
            for k in ('V','I') for t in ('mag','ang')]
BI_PAIRS = [(1,2),(1,3),(2,3)]


def utc(): return datetime.now(timezone.utc).isoformat()
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
def atomic_json(p, obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)


def paths(stage):
    if stage not in ('dev', 'test', 'smoke'):
        raise ValueError(stage)
    run = OUTPUT / stage
    return {'run': run, 'raw': run / 'raw_simulations.parquet',
            'processed': run / 'dataset.parquet', 'metadata': run / 'spatial_distributions.json',
            'validation': run / 'validation_report.json'}


def immutable_write(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data: raise RuntimeError(f'Arquivo existente diverge: {path}')
    else:path.write_bytes(data)


def prepare():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    params = json.loads((BASE / 'parametros_geracao.json').read_text(encoding='utf-8'))
    metadata = json.loads((ROOT / 'sistemas/ieee123/metadados/perfis_espaciais.json').read_text(encoding='utf-8'))
    source_manifest = json.loads((RAW / 'fonte_manifesto.json').read_text(encoding='utf-8'))
    sources = {}
    for entry in source_manifest['arquivos']:
        path = RAW / entry['arquivo']
        assert digest(path) == entry['sha256'], path
        sources[path.relative_to(ROOT).as_posix()] = entry['sha256']
    master = RAW / 'originais/trunk/Distrib/IEEETestCases/123Bus/IEEE123Master.dss'
    modules = [CODE / 'src/opendss_tcc' / name for name in (
        'circuit/loader.py', 'simulation/runner.py', 'simulation/scenarios.py',
        'simulation/fault_injector.py', 'simulation/load_distributions.py', 'features/build.py')]
    frozen = {'schema': 'simulacao.ieee123.inputs.v1', 'created_utc': utc(),
              'config': cfg, 'config_sha256': digest(CONFIG), 'master': str(master),
              'source_hashes': sources,
              'pipeline_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in modules},
              'buses_phases': params['buses_phases'], 'metadata': metadata,
              'expected_dev': params['expected_dev'], 'expected_test': params['expected_test']}
    stable = {k:v for k,v in frozen.items() if k != 'created_utc'}
    frozen['input_fingerprint'] = hashlib.sha256(canonical(stable)).hexdigest()
    target = OUTPUT / 'inputs_congelados.json'
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        assert previous['input_fingerprint'] == frozen['input_fingerprint'], 'Parâmetros divergentes'
        return previous
    atomic_json(target, frozen)
    return frozen


_CTX={}
def init_worker(frozen, run):
    global _CTX
    run=Path(run);private=run/'dss_outputs'/str(os.getpid());private.mkdir(parents=True,exist_ok=True)
    wrapper=private/'worker_master.dss'
    wrapper.write_text(f'Set DataPath="{private.as_posix()}"\n'
                       f'Redirect "{Path(frozen["master"]).as_posix()}"\n'
                       f'Set MaxControlIter=100\nSet DataPath="{private.as_posix()}"\n',encoding='utf-8')
    md=frozen['metadata']
    _CTX={'frozen':frozen,'run':run,'wrapper':wrapper,
          'nominal':[NominalLoad(**v) for v in md['nominal_loads']],
          'patterns':{v['distribution_id']:SpatialPattern(v['distribution_id'],v['seed'],v['factors'])
                      for v in md['patterns']}}


def chunk_paths(run, index):
    name=f'{index:06d}';run=Path(run)/'chunks'
    return run/f'{name}.parquet',run/f'{name}.audit.jsonl',run/f'{name}.json'


def work_chunk(index, entries, fingerprint):
    ctx=_CTX;cfg=ctx['frozen']['config'];rows=[];audits=[];start=time.perf_counter()
    for case_id, item in entries:
        sc=Scenario(**{**item,'phases':tuple(item['phases'])})
        d=load_circuit(ctx['wrapper'])
        fault_name=create_reusable_fault(d,sc.bus,sc.phases[0],sc.rf)
        audit={}
        row=run_scenario(d,sc,'Line.L115',1,legacy_fault_mapping=False,
            nominal_loads=ctx['nominal'],spatial_pattern=ctx['patterns'][sc.distribution_id],
            reusable_fault_name=fault_name,freeze_controls=True,audit=audit)
        assert audit['pre']['control_mode']==0 and audit['fault']['control_mode']==-1
        assert audit['pre']['taps']==audit['fault']['taps']
        assert np.isfinite(np.array([row[k] for k in RAW_KEYS],dtype=float)).all()
        row['system_id']='ieee123';rows.append(row)
        audits.append({'case_id':case_id,**asdict(sc),**audit})
    frame=pd.DataFrame(rows);assert frame.shape==(len(entries),33)
    parquet,audit_path,manifest_path=chunk_paths(ctx['run'],index)
    parquet.parent.mkdir(parents=True,exist_ok=True)
    tmp=parquet.with_suffix(f'.tmp.{os.getpid()}');frame.to_parquet(tmp,index=False);tmp.replace(parquet)
    tmp=audit_path.with_suffix(f'.tmp.{os.getpid()}')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        for value in audits:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(audit_path)
    result={'chunk_id':index,'rows':len(entries),'case_start':entries[0][0],
        'case_end':entries[-1][0],'inputs_fingerprint':fingerprint,
        'case_keys_sha256':hashlib.sha256(canonical(entries)).hexdigest(),
        'raw_sha256':digest(parquet),'audit_sha256':digest(audit_path),
        'worker_pid':os.getpid(),'elapsed_seconds':time.perf_counter()-start,'finished_utc':utc()}
    atomic_json(manifest_path,result)
    return result


def load_frozen():
    frozen=json.loads((OUTPUT/'inputs_congelados.json').read_text(encoding='utf-8'))
    assert digest(CONFIG)==frozen['config_sha256']
    for name,h in {**frozen['source_hashes'],**frozen['pipeline_hashes']}.items():assert digest(ROOT/name)==h,name
    return frozen


def stages_plan(frozen, stage, selection):
    loads = frozen['config']['variations']['load_multipliers']
    if stage == 'test':
        p = Path(selection) if selection else ROOT / 'metadados/modelos.json'
        choices = json.loads(p.read_text(encoding='utf-8'))
        if not isinstance(choices, list):
            choices = choices['choices']
        choices = [v for v in choices if v.get('system') == 'ieee123']
        assert len(choices) == 12 and len({v['id'] for v in choices}) == 12
        midpoint = [round((a+b)/2, 10) for a,b in zip(loads, loads[1:])]
        params = json.loads((ROOT / 'sistemas/ieee123/metadados/niveis_teste_final.json').read_text(encoding='utf-8'))
        assert midpoint == params['load_multipliers']
        loads = midpoint
        selection_hash = digest(p)
    else:
        selection_hash = None
    expected = frozen['expected_dev'] if stage == 'dev' else (frozen['expected_test'] if stage == 'test' else 4)
    return loads, expected, selection_hash


def chunks(frozen, loads, chunk_size, stage):
    generator=make_scenarios(frozen['buses_phases'],[.0001],loads,BI_PAIRS,['P0','P1','P2','P3','P4','P5'])
    if stage=='smoke':generator=islice(generator,4)
    numbered=enumerate(generator);index=0
    while True:
        entries=[(i,asdict(sc)) for i,sc in islice(numbered,chunk_size)]
        if not entries:break
        yield index,entries;index+=1


def valid_chunk(run,index,entries,fingerprint):
    parquet,audit,meta=chunk_paths(run,index)
    if not meta.exists():
        if parquet.exists() or audit.exists():raise RuntimeError(f'Chunk incompleto preservado: {index}')
        return None
    result=json.loads(meta.read_text(encoding='utf-8'))
    assert result['inputs_fingerprint']==fingerprint and result['rows']==len(entries)
    assert result['case_keys_sha256']==hashlib.sha256(canonical(entries)).hexdigest()
    assert digest(parquet)==result['raw_sha256'] and digest(audit)==result['audit_sha256']
    return result


def assemble(frozen, stage, pp, all_chunks, expected, fingerprint):
    frames=[pd.read_parquet(chunk_paths(pp['run'],i)[0]) for i in sorted(all_chunks)]
    raw=pd.concat(frames,ignore_index=True)
    assert len(raw)==expected and raw.shape[1]==33
    keys=['bus','type_id','rf','load_mult','distribution_id'];assert not raw.duplicated(keys).any()
    if stage!='smoke':assert set(raw.bus)==set(frozen['buses_phases'])
    assert np.isfinite(raw[RAW_KEYS].to_numpy()).all()
    if stage!='smoke':
        assert set(raw.type_id)==set(range(1,12)) and set(raw.distribution_id)=={f'P{i}' for i in range(6)}
        assert raw.groupby('distribution_id').size().nunique()==1
    assert all(all(int(p) in frozen['buses_phases'][bus] for p in phases.split('-'))
               for bus,phases in raw[['bus','phases']].itertuples(index=False,name=None))
    pp['raw'].parent.mkdir(parents=True,exist_ok=True);pp['processed'].parent.mkdir(parents=True,exist_ok=True)
    tmp=pp['raw'].with_suffix('.tmp');raw.to_parquet(tmp,index=False);tmp.replace(pp['raw'])
    atomic_json(pp['metadata'],frozen['metadata'])
    # Conserva o formato da auditoria oficial, sem carregar todos os dicionários em RAM.
    audit_file=pp['run']/'auditoria_controles.json';tmp=audit_file.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as stream:
        stream.write('{"policy":"fresh_STATIC_pre_OFF_fault","cases":[');first=True
        for i in sorted(all_chunks):
            for line in chunk_paths(pp['run'],i)[1].open(encoding='utf-8'):
                if not first:stream.write(',')
                stream.write(line.strip());first=False
        stream.write(']}\n')
    tmp.replace(audit_file)
    features=build_features(raw);assert features.shape==(expected,81)
    numeric=features.select_dtypes(include=[np.number]);assert np.isfinite(numeric.to_numpy()).all()
    tmp=pp['processed'].with_suffix('.tmp');features.to_parquet(tmp,index=False);tmp.replace(pp['processed'])
    report={'schema':'tcc.ieee123.validation.v1','status':'PASSOU','system_id':'ieee123','stage':stage,
        'rows':expected,'columns':81,'raw_columns':33,'inputs_fingerprint':fingerprint,
        'raw_sha256':digest(pp['raw']),'processed_sha256':digest(pp['processed']),
        'metadata_sha256':digest(pp['metadata']),'audit_sha256':digest(audit_file),
        'eligible_buses':124,'class_count':int(raw.bus.nunique()),
        'type_counts':{str(k):int(v) for k,v in raw.type_id.value_counts().items()},
        'profile_counts':{str(k):int(v) for k,v in raw.distribution_id.value_counts().items()},
        'load_counts':{str(k):int(v) for k,v in raw.load_mult.value_counts().items()},
        'uniqueness_checked':True,'finite_checked':True,'phase_coverage_checked':True,
        'all_controls_and_taps_checked_per_case':True,'finished_utc':utc()}
    atomic_json(pp['validation'],report)
    return report


def run(stage,workers,chunk_size,selection):
    frozen=load_frozen();loads,expected,selection_hash=stages_plan(frozen,stage,selection)
    pp=paths(stage);pp['run'].mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(canonical({'inputs':frozen['input_fingerprint'],'stage':stage,
        'loads':loads,'selection_sha256':selection_hash,'chunk_size':chunk_size})).hexdigest()
    checkpoint=pp['run']/'generation_plan.json'
    plan={'inputs_fingerprint':fingerprint,'expected':expected,'stage':stage,'workers':workers,
          'chunk_size':chunk_size,'selection_sha256':selection_hash,'loads':loads}
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding='utf-8'))
        assert previous['inputs_fingerprint']==fingerprint,'Plano da retomada diverge.'
    else:atomic_json(checkpoint,plan)
    if pp['validation'].exists():
        report=json.loads(pp['validation'].read_text(encoding='utf-8'))
        assert report['status']=='PASSOU' and report['inputs_fingerprint']==fingerprint
        for key,path in [('raw_sha256',pp['raw']),('processed_sha256',pp['processed']),('metadata_sha256',pp['metadata'])]:assert digest(path)==report[key]
        print('COMPLETO: resultados e hashes já conferidos; nenhuma regeneração.',flush=True);return
    for target in [pp['raw'],pp['processed']]:
        if target.exists():raise RuntimeError(f'Saída sem recibo completo preservada: {target}')
    start=time.monotonic();done={};completed_rows=0;pending={}
    iterator=iter(chunks(frozen,loads,chunk_size,stage));exhausted=False
    status_file=pp['run']/'status.json'
    def status(state,extra=None):
        elapsed=time.monotonic()-start;rate=completed_rows/elapsed if elapsed else 0
        value={'state':state,'stage':stage,'completed_rows':completed_rows,'expected_rows':expected,
            'completed_chunks':len(done),'elapsed_seconds':elapsed,'rate_cases_second':rate,
            'eta_seconds':(expected-completed_rows)/rate if rate else None,'updated_utc':utc(),
            'inputs_fingerprint':fingerprint,'workers':workers}
        if extra:value.update(extra)
        atomic_json(status_file,value);atomic_json(OUTPUT/f'status_{stage}.json',value)
        print(json.dumps(value,ensure_ascii=False),flush=True)
    status('running')
    try:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(frozen,str(pp['run']))) as pool:
            while not exhausted or pending:
                while not exhausted and len(pending)<workers*2:
                    try:index,entries=next(iterator)
                    except StopIteration:exhausted=True;break
                    existing=valid_chunk(pp['run'],index,entries,fingerprint)
                    if existing:done[index]=existing;completed_rows+=existing['rows'];continue
                    future=pool.submit(work_chunk,index,entries,fingerprint);pending[future]=index
                if not pending:continue
                completed,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in completed:
                    index=pending.pop(future);result=future.result();done[index]=result;completed_rows+=result['rows']
                status('running')
        assert completed_rows==expected
        status('assembling')
        report=assemble(frozen,stage,pp,done,expected,fingerprint)
        status('complete',{'processed_sha256':report['processed_sha256']})
    except Exception as exc:
        status('failed',{'error':repr(exc)});raise


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'dev', 'test', 'smoke'])
    parser.add_argument('--saida', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--selection-plan', type=Path)
    parser.add_argument('--confirmar-geracao', action='store_true')
    args = parser.parse_args()
    if not args.confirmar_geracao:
        parser.error('Use --confirmar-geracao para executar simulações.')
    OUTPUT = args.saida.resolve()
    if OUTPUT == ROOT or ROOT in OUTPUT.parents:
        parser.error('A pasta de saída deve ficar fora do pacote distribuído.')
    if args.workers < 1 or args.workers > 4:
        parser.error('--workers deve ficar entre1 e4.')
    if args.chunk_size < 1:
        parser.error('--chunk-size deve ser positivo.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare':
        prepare()
        return
    prepare()
    run(args.stage, args.workers, args.chunk_size, args.selection_plan)


if __name__=='__main__':main()
