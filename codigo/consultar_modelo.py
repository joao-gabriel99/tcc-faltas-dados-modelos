"""Consulta pontual aos modelos congelados; não treina nem altera arquivos."""
from pathlib import Path
import argparse
import json
import sys
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from opendss_tcc.features.cartesian import RAW_COLUMNS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('id', help='Identificador em metadados/modelos.json')
parser.add_argument('--linhas', type=int, default=5)
args = parser.parse_args()
if args.linhas < 1:
    parser.error('--linhas deve ser positivo')
items = json.loads((ROOT / 'metadados/modelos.json').read_text(encoding='utf-8'))
item = next((x for x in items if x['id'] == args.id), None)
if item is None:
    parser.error('Identificador não encontrado')
# Carregue somente os arquivos joblib de origem verificada.
model = joblib.load(ROOT / item['path'])
frame = pd.read_parquet(ROOT / 'sistemas' / item['system'] / 'dados/teste_final/dataset.parquet')
if item['regime'] == 'controle_p0':
    frame = frame[frame.distribution_id.eq('P0')]
frame = frame.head(args.linhas)
indices = np.asarray(model.predict(frame[RAW_COLUMNS].to_numpy(dtype=float)), dtype=int)
predicted = np.asarray(item['classes'])[indices]
actual = frame['bus' if item['task'] == 'bus' else 'type_id'].astype(str).to_numpy()
for real, pred in zip(actual, predicted):
    print(f'Rótulo individual esperado: {real}; identificado: {pred}')
