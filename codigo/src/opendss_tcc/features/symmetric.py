"""Fortescue symmetric components transform."""
from __future__ import annotations
import numpy as np

# a = 1 / 120 deg
A = np.exp(1j * 2 * np.pi / 3)
A2 = A * A

# Fortescue matrix (phase -> 012)
FORTESCUE = (1 / 3) * np.array([
    [1,  1,   1],
    [1,  A,   A2],
    [1,  A2,  A],
])


def phase_to_012(phasors_abc: np.ndarray) -> np.ndarray:
    """phasors_abc: shape (..., 3) complex -> (..., 3) complex (0, +, -)"""
    return phasors_abc @ FORTESCUE.T


def symmetric_features(v_abc: np.ndarray, i_abc: np.ndarray) -> dict[str, float]:
    """Returns flat dict with V0/V1/V2 and I0/I1/I2 magnitudes and angles."""
    v012 = phase_to_012(v_abc)
    i012 = phase_to_012(i_abc)
    out: dict[str, float] = {}
    labels = ['0', '1', '2']
    for idx, lbl in enumerate(labels):
        out[f'V{lbl}_mag'] = float(np.abs(v012[idx]))
        out[f'V{lbl}_ang'] = float(np.angle(v012[idx]))
        out[f'I{lbl}_mag'] = float(np.abs(i012[idx]))
        out[f'I{lbl}_ang'] = float(np.angle(i012[idx]))
    return out
