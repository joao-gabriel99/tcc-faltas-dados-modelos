"""Fábricas empregadas nos modelos finais RF, XGBoost e MLP."""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def rf_factory(seed=42):
    return RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=seed,
    )

def xgb_factory(seed=42, num_class: int | None = None):
    if not HAS_XGB:
        raise RuntimeError('xgboost not installed')
    return XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.08,
        n_jobs=-1, random_state=seed, tree_method='hist',
        objective='multi:softprob' if num_class else 'multi:softprob',
    )

def corrected_factory(name, groups, seed=42, n_jobs=2):
    """Configuração fixada antes da execução oficial corrigida de setembro."""
    from ..features.cartesian import CartesianFeatures
    if name == 'RandomForest':
        model = rf_factory(seed)
        model.set_params(n_jobs=n_jobs)
    elif name == 'XGBoost':
        model = xgb_factory(seed)
        model.set_params(n_jobs=n_jobs)
    elif name == 'MLP':
        model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                              solver='adam', max_iter=600, early_stopping=True,
                              n_iter_no_change=60, random_state=seed)
    else:
        raise ValueError(f'Modelo desconhecido: {name}')
    return Pipeline([('features', CartesianFeatures(tuple(groups))), ('model', model)])
