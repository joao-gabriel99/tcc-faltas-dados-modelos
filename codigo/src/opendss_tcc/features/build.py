"""Builds final feature table from raw snapshots."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .symmetric import symmetric_features


def _abc_from_row(row: pd.Series, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        row[f'{prefix}_V1_mag'] * np.exp(1j * row[f'{prefix}_V1_ang']),
        row[f'{prefix}_V2_mag'] * np.exp(1j * row[f'{prefix}_V2_ang']),
        row[f'{prefix}_V3_mag'] * np.exp(1j * row[f'{prefix}_V3_ang']),
    ])
    i = np.array([
        row[f'{prefix}_I1_mag'] * np.exp(1j * row[f'{prefix}_I1_ang']),
        row[f'{prefix}_I2_mag'] * np.exp(1j * row[f'{prefix}_I2_ang']),
        row[f'{prefix}_I3_mag'] * np.exp(1j * row[f'{prefix}_I3_ang']),
    ])
    return v, i


def build_features(df: pd.DataFrame, circular_deltas: bool = True) -> pd.DataFrame:
    """Adds symmetric components and delta features to raw simulation df."""
    feats = []
    for _, row in df.iterrows():
        f = {}
        pre_v, pre_i = _abc_from_row(row, 'pre')
        flt_v, flt_i = _abc_from_row(row, 'fault')
        sym_pre = symmetric_features(pre_v, pre_i)
        sym_flt = symmetric_features(flt_v, flt_i)
        for k, v in sym_pre.items():
            f[f'pre_sym_{k}'] = v
        for k, v in sym_flt.items():
            f[f'fault_sym_{k}'] = v
        # Delta (phase-domain)
        for ph in (1, 2, 3):
            for kind in ('V', 'I'):
                for typ in ('mag', 'ang'):
                    pre_val = row[f'pre_{kind}{ph}_{typ}']
                    flt_val = row[f'fault_{kind}{ph}_{typ}']
                    difference = flt_val - pre_val
                    if typ == 'ang' and circular_deltas:
                        difference = (difference + np.pi) % (2*np.pi) - np.pi
                    f[f'delta_{kind}{ph}_{typ}'] = difference
        # Delta (symmetric)
        for seq in ('0', '1', '2'):
            for kind in ('V', 'I'):
                for typ in ('mag', 'ang'):
                    difference = sym_flt[f'{kind}{seq}_{typ}'] - sym_pre[f'{kind}{seq}_{typ}']
                    if typ == 'ang' and circular_deltas:
                        difference = (difference + np.pi) % (2*np.pi) - np.pi
                    f[f'delta_sym_{kind}{seq}_{typ}'] = difference
        feats.append(f)

    feat_df = pd.DataFrame(feats, index=df.index)
    return pd.concat([df, feat_df], axis=1)


FEATURE_GROUPS = {
    'fault_phase':       [f'fault_{kind}{ph}_{typ}' for ph in (1, 2, 3) for kind in ('V', 'I') for typ in ('mag', 'ang')],
    'fault_sym':         [f'fault_sym_{kind}{seq}_{typ}' for seq in ('0', '1', '2') for kind in ('V', 'I') for typ in ('mag', 'ang')],
    'delta_phase':       [f'delta_{kind}{ph}_{typ}' for ph in (1, 2, 3) for kind in ('V', 'I') for typ in ('mag', 'ang')],
    'delta_sym':         [f'delta_sym_{kind}{seq}_{typ}' for seq in ('0', '1', '2') for kind in ('V', 'I') for typ in ('mag', 'ang')],
}


def select_features(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    cols: list[str] = []
    for g in groups:
        cols.extend(FEATURE_GROUPS[g])
    return df[cols]
