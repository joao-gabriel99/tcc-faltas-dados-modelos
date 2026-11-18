"""Treino e avaliação finais 13/34 com escolhas previamente fixadas."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,sys,time,warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
sys.path.insert(0,str(PACKAGE/'codigo/src'))
from opendss_tcc.models.factories import corrected_factory
from opendss_tcc.models.evaluation import metrics
from opendss_tcc.features.cartesian import RAW_COLUMNS,CartesianFeatures
OUT=Path(os.environ.get('TCC_OUTPUT_DIR',str(PACKAGE/'execucoes/ieee13_34'))).resolve()
def plan():
    return json.loads((HERE/'plano_final_ieee13_34.json').read_text(encoding='utf-8'))
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def training_dataset(system):
    return PACKAGE/'sistemas'/system/'dados/treino_cv/dataset.parquet'
def test_dataset(system):
    return PACKAGE/'sistemas'/system/'dados/teste_final/dataset.parquet'

def train(j):
    result=OUT/'fits'/f"{j['id']}.json"
    if result.exists():
        return j['id'], 'concluído anteriormente'
    df=pd.read_parquet(training_dataset(j['system']))
    if j['regime']=='controle_p0': df=df[df.distribution_id=='P0']
    le=LabelEncoder().fit(df['type_id' if j['task']=='type' else 'bus'].astype(str))
    y=le.transform(df['type_id' if j['task']=='type' else 'bus'].astype(str))
    model=corrected_factory(j['model'],j['groups'],seed=42,n_jobs=2)
    assert {k:str(v) for k,v in model.get_params(deep=True).items()}==j['parameters']
    start=time.time()
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(df[RAW_COLUMNS].to_numpy(),y)
    file=OUT/'models'/f"{j['id']}.joblib"
    joblib.dump(model,file,compress=3)
    est=model.named_steps['model']
    record=dict(id=j['id'],completed_utc=datetime.now(timezone.utc).isoformat(),
                model_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),train_rows=len(df),
                train_indices=df.index.tolist(),classes=le.classes_.tolist(),
                scales=model.named_steps['features'].scales_.tolist(),
                warnings=[str(w.message) for w in caught],seconds=time.time()-start)
    if j['model']=='MLP': record.update(epochs=int(est.n_iter_),loss=est.loss_curve_,validation=est.validation_scores_)
    dump(result,record)
    return j['id'],round(time.time()-start,1)

def independent_metrics(y,pred,n):
    cm=np.bincount(y*n+pred,minlength=n*n).reshape(n,n)
    tp=np.diag(cm).astype(float); support=cm.sum(axis=1); positives=cm.sum(axis=0)
    precision=np.divide(tp,positives,out=np.zeros(n),where=positives!=0)
    recall=np.divide(tp,support,out=np.zeros(n),where=support!=0)
    f1=np.divide(2*tp,support+positives,out=np.zeros(n),where=(support+positives)!=0)
    return dict(accuracy=float(tp.sum()/cm.sum()),f1_macro=float(f1.mean()),
                f1_weighted=float((f1*support).sum()/support.sum()),
                precision_macro=float(precision.mean()),recall_macro=float(recall.mean())),cm

def evaluate(j):
    identifier=j['id']
    record=json.loads((OUT/'fits'/f'{identifier}.json').read_text(encoding='utf-8'))
    file=OUT/'models'/f'{identifier}.joblib'
    assert sha(file)==record['model_sha256']
    model=joblib.load(file)
    assert {k:str(v) for k,v in model.get_params(deep=True).items()}==j['parameters']
    df=pd.read_parquet(test_dataset(j['system']))
    train=pd.read_parquet(training_dataset(j['system']))
    if j['regime']=='controle_p0':
        df=df[df.distribution_id=='P0']; train=train[train.distribution_id=='P0']
    assert record['train_rows']==len(train) and record['train_indices']==train.index.tolist()
    target='type_id' if j['task']=='type' else 'bus'
    assert sorted(train[target].astype(str).unique())==record['classes']
    assert sorted(df[target].astype(str).unique())==record['classes']
    mapping={label:i for i,label in enumerate(record['classes'])}
    y=df[target].astype(str).map(mapping).to_numpy(dtype=int)
    x=df[RAW_COLUMNS].to_numpy()
    # Diagnóstico de medições exatamente repetidas entre desenvolvimento e prova.
    # Não exclui linhas nem interpreta similaridade física como erro automaticamente.
    input_overlap=df[RAW_COLUMNS].merge(train[RAW_COLUMNS].drop_duplicates(),how='inner',on=RAW_COLUMNS)
    fresh=CartesianFeatures(tuple(j['groups'])).fit(train[RAW_COLUMNS].to_numpy())
    np.testing.assert_array_equal(fresh.scales_,model.named_steps['features'].scales_)
    np.testing.assert_array_equal(fresh.scales_,record['scales'])
    with threadpool_limits(limits=2):
        pred=model.predict(x)
        # Segunda carga do modelo deve reproduzir todas as previsões.
        np.testing.assert_array_equal(joblib.load(file).predict(x),pred)
        if j['model']=='MLP':
            est=model.named_steps['model']
            a=fresh.transform(x)
            for w,b in zip(est.coefs_[:-1],est.intercepts_[:-1]): a=np.maximum(a@w+b,0)
            logits=a@est.coefs_[-1]+est.intercepts_[-1]
            np.testing.assert_array_equal(est.classes_[np.argmax(logits,axis=1)],pred)
            assert est.max_iter==600 and est.n_iter_no_change==60
            assert est.n_iter_==record['epochs']
    scores=metrics(y,pred)
    independent,cm=independent_metrics(y,pred,len(mapping))
    for k in scores: assert abs(scores[k]-independent[k])<1e-12
    np.testing.assert_array_equal(cm,confusion_matrix(y,pred))
    rows=[]
    for column in ['distribution_id','load_mult']:
        for value in sorted(df[column].unique()):
            mask=df[column].to_numpy()==value
            subset=metrics(y[mask],pred[mask])
            checked,_=independent_metrics(y[mask],pred[mask],len(mapping))
            for k in subset: assert abs(subset[k]-checked[k])<1e-12
            rows.append(dict(id=identifier,group=column,value=str(value),n=int(mask.sum()),**subset))
    predictions=df[['bus','type_id','load_mult','distribution_id','rf']].copy()
    predictions['real']=y; predictions['predicted']=pred
    predictions.to_parquet(OUT/'predictions'/f'{identifier}.parquet')
    result=dict(**j,metrics=scores,test_rows=len(df),train_rows=len(train),
                classes=record['classes'],confusion_matrix=cm.tolist(),by_groups=rows,
                per_class=classification_report(y,pred,target_names=record['classes'],output_dict=True,zero_division=0),
                warnings=record['warnings'],epochs=record.get('epochs'),
                scales_from_training_verified=True,metrics_recomputed=True,predictions_replayed=True,
                independent_mlp_inference=j['model']=='MLP',model_sha256=record['model_sha256'])
    result['exact_raw_input_overlap_rows']=len(input_overlap)
    dump(OUT/'evaluation'/f'{identifier}.json',result)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    args=parser.parse_args()
    p=plan()
    for name in ['fits','models','evaluation','predictions']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    if args.command=='train':
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(train,j) for j in p['jobs']]
            for index,future in enumerate(as_completed(futures),1):
                print(index,len(futures),future.result(),flush=True)
        dump(OUT/'treino_concluido.json',{'fits':len(p['jobs']),'completed_utc':datetime.now(timezone.utc).isoformat()})
    else:
        assert (OUT/'treino_concluido.json').exists(),'Execute train primeiro; não usa os modelos públicos como novos ajustes.'
        results=[evaluate(j) for j in p['jobs']]
        dump(OUT/'resultados.json',results)
        print('Avaliação individual concluída; agrupamento é uma etapa posterior documentada nos metadados públicos.',flush=True)
