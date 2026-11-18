"""Gera novas bases em uma pasta separada, usando o núcleo de simulação."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sistema', choices=['ieee13', 'ieee34', 'ieee123'])
    parser.add_argument('--etapa', choices=['treino_cv', 'teste_final'], required=True)
    parser.add_argument('--saida', type=Path, required=True,
                        help='Pasta nova e separada do pacote distribuído')
    parser.add_argument('--confirmar-geracao', action='store_true')
    args = parser.parse_args()
    if not args.confirmar_geracao:
        parser.error('A geração requer --confirmar-geracao e pode ser demorada.')
    output = args.saida.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error('--saida deve ficar fora da pasta do pacote distribuído.')
    if output.exists():
        parser.error('--saida deve ser uma pasta nova; resultados existentes são preservados.')
    import yaml
    cfg_file = ROOT / 'sistemas' / args.sistema / 'metadados/configuracao_treino.yaml'
    cfg = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
    if args.etapa == 'teste_final':
        test = json.loads((cfg_file.parent / 'niveis_teste_final.json').read_text(encoding='utf-8'))
        cfg['variations']['load_multipliers'] = test['load_multipliers']
    circuit = (ROOT / cfg['circuit']['dss_file']).resolve()
    output.mkdir(parents=True)
    config_folder = output / 'configs'
    config_folder.mkdir()
    (output / 'dss_outputs').mkdir()
    wrapper = output / 'entrada.dss'
    wrapper.write_text(f'Redirect "{circuit.as_posix()}"\nSet DataPath="{(output / "dss_outputs").as_posix()}"\n', encoding='utf-8')
    cfg['circuit']['dss_file'] = str(wrapper)
    cfg['output'] = {'interim_path': 'raw_simulations.parquet',
                     'processed_path': 'dataset.parquet', 'results_dir': 'resultados',
                     'distribution_metadata_path': 'spatial_distributions.json'}
    config_copy = config_folder / 'simulation.yaml'
    config_copy.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    sys.path.insert(0, str(ROOT / 'codigo/src'))
    from opendss_tcc.pipelines.generate_dataset import main as generate
    generate(str(config_copy), confirm_generation=True)

if __name__ == '__main__':
    main()
