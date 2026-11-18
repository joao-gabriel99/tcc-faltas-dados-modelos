"""Fasores cartesianos com duas escalas comuns ajustadas somente no treino."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from .symmetric import phase_to_012

RAW_COLUMNS = [f'{stage}_{kind}{phase}_{part}' for stage in ('pre', 'fault')
               for phase in (1, 2, 3) for kind in ('V', 'I') for part in ('mag', 'ang')]
GROUPS = ('fault_phase', 'fault_sym', 'delta_phase', 'delta_sym')


def complex_phases(X):
    """Retorna amostra × instante × fase × grandeza, em unidades físicas."""
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or x.shape[1] != 24 or not np.isfinite(x).all():
        raise ValueError('São exigidos 24 campos finitos de fase pré/falta.')
    pairs = x.reshape(-1, 2, 3, 2, 2)
    if np.any(pairs[..., 0] < 0):
        raise ValueError('Módulos fasoriais não podem ser negativos.')
    return pairs[..., 0] * np.exp(1j * pairs[..., 1])


class CartesianFeatures(TransformerMixin, BaseEstimator):
    """Mantém pequenas componentes pequenas, sem padronização individual."""
    def __init__(self, groups=GROUPS):
        self.groups = groups

    def fit(self, X, y=None):
        if not self.groups or any(g not in GROUPS for g in self.groups):
            raise ValueError('Grupo de atributos inválido.')
        z = complex_phases(X)
        # Uma referência por grandeza: percentil 95 do RMS entre as três fases.
        self.scales_ = np.quantile(np.sqrt(np.mean(np.abs(z)**2, axis=2)).reshape(-1, 2), .95, axis=0)
        if np.any(self.scales_ <= 0):
            raise ValueError('Escala física de treino nula.')
        self.n_features_in_ = 24
        return self

    def transform(self, X):
        check_is_fitted(self, 'scales_')
        z = complex_phases(X) / self.scales_
        sym = np.stack([phase_to_012(z[..., k]) for k in range(2)], axis=-1)
        groups = {'fault_phase': z[:, 1], 'fault_sym': sym[:, 1],
                  'delta_phase': z[:, 1]-z[:, 0], 'delta_sym': sym[:, 1]-sym[:, 0]}
        parts = [np.stack((groups[g].real, groups[g].imag), axis=-1).reshape(len(z), 12)
                 for g in self.groups]
        return np.concatenate(parts, axis=1)
