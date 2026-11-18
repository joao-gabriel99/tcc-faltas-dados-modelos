"""OpenDSS circuit loader via dss-python."""
from __future__ import annotations
import os
from pathlib import Path
import dss


def get_dss():
    d = dss.DSS
    d.Start(0)
    return d


def load_circuit(dss_file: str | Path) -> "dss.IDSS":
    d = get_dss()
    original_cwd = Path.cwd()
    try:
        d.Text.Command = 'Clear'
        d.Text.Command = f'Compile "{Path(dss_file).resolve()}"'
        d.Text.Command = 'Solve'
    finally:
        # O comando Compile do OpenDSS altera o diretório de trabalho global.
        # Restaurá-lo evita que uma execução multissistema resolva caminhos
        # relativos a partir da pasta do primeiro circuito carregado.
        os.chdir(original_cwd)
    return d


def solve(d) -> None:
    d.Text.Command = 'Solve'


def reset_faults(d) -> None:
    d.Text.Command = 'Reset faults'
