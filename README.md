# Bases simuladas, modelos treinados e circuitos OpenDSS

Código e modelos próprios sob MIT; dados, resultados e documentação próprios
sob CC BY 4.0. As condições de terceiros permanecem aplicáveis aos circuitos.
Consulte [LICENCAS.md](LICENCAS.md) e [CITATION.cff](CITATION.cff).

## Downloads

Arquivos da versão 1.0.0: [página de downloads](https://github.com/joao-gabriel99/tcc-faltas-dados-modelos/releases/tag/v1.0.0).

| Arquivo | Conteúdo |
|---|---|
| [apoio.zip](https://github.com/joao-gabriel99/tcc-faltas-dados-modelos/releases/download/v1.0.0/apoio.zip) | Código, documentação, circuitos, licenças e manifesto completo. |
| [ieee13.zip](https://github.com/joao-gabriel99/tcc-faltas-dados-modelos/releases/download/v1.0.0/ieee13.zip) | Bases, 12 modelos, previsões e resultados do IEEE 13. |
| [ieee34.zip](https://github.com/joao-gabriel99/tcc-faltas-dados-modelos/releases/download/v1.0.0/ieee34.zip) | Bases, 12 modelos, previsões e resultados do IEEE 34. |
| [ieee123.zip](https://github.com/joao-gabriel99/tcc-faltas-dados-modelos/releases/download/v1.0.0/ieee123.zip) | Bases, 12 modelos, previsões e resultados do IEEE 123. |

Extraia os quatro arquivos na mesma pasta. As bases e os modelos binários
estão nos downloads; este repositório mantém os códigos e metadados legíveis.
O arquivo SHA256SUMS.txt da Release permite conferir os quatro downloads.

## Conteúdo

- sistemas/ieee13, ieee34 e ieee123: bases de treinamento/validação cruzada e de
  teste final, brutas e processadas; 12 modelos finais por sistema; classes,
  escalas, configurações, previsões salvas e avaliação individual/agrupada.
- circuitos: arquivos OpenDSS efetivamente utilizados e suas dependências,
  com caminhos portáteis na cópia e procedência documentada.
- codigo: transformação das entradas necessária para carregar os modelos,
  exemplos de consulta, scripts técnicos de simulação/treinamento/avaliação
  e verificador de integridade. Os pontos de entrada e suas condições de uso
  estão documentados em codigo/.
- metadados: catálogos completos de bases/modelos, ambientes e verificações.
- licencas/terceiros: condições e avisos de terceiros preservados.

São 36 modelos: 3 sistemas × 2 objetivos (tipo de falta/localização) × 2 regimes
× 3 algoritmos (Random Forest, XGBoost e MLP). Apenas as configurações finais
selecionadas pela validação cruzada foram incluídas. Tentativas e modelos
intermediários não foram incorporados.

As bases completas contêm os seis perfis espaciais. O controle usa apenas P0;
o enriquecido usa P0 a P5. O controle é um subconjunto, não uma sétima base.
Consultar DICIONARIO_DADOS.md antes de usar os dados ou comparar os resultados.

## Código de simulação e treinamento

- [Simulação e geração de bases](codigo/SIMULACAO.md).
- [Treinamento, validação cruzada e avaliação final](codigo/treinamento_e_avaliacao/TREINAMENTO.md).
- Exemplo de consulta aos modelos: codigo/consultar_modelo.py.

Os códigos científicos e os dados publicados permitem examinar os procedimentos.
As cópias portáteis foram verificadas por leitura, sintaxe e ajuda das interfaces.
Esta publicação não repetiu simulações, treinamento ou uma reprodução numérica
integral. As versões originais e as diferenças de ambiente estão documentadas.
As rotinas de geração exigem uma pasta nova fora do pacote; os novos treinos
gravam suas saídas em execucoes/, preservando os artefatos distribuídos.

## Leitura dos modelos

Consulte metadados/ambientes.json e os arquivos requirements_*.txt. Os modelos
de 13/34 barras registram scikit-learn 1.9.0; os de 123 barras, 1.8.0. Use
ambientes separados e compatíveis com as execuções originais. As listas
refletem registros históricos, sem afirmar reconstrução completa do ambiente
nesta preparação. A versão de Python do plano antigo não foi confirmada.

O teste pontual desta preparação carregou 36 modelos e conferiu duas previsões
por modelo contra as previsões salvas. Houve avisos de diferença de versão nos
24 modelos de 13/34 barras, preservados em metadados/verificacao_modelos.json.
Esse teste pequeno não equivale a reproduzir integralmente o treinamento.

Joblib deve ser carregado apenas de origem confiável, pois o formato pode
executar objetos Python durante a leitura. Verifique os hashes do pacote.

Extraia os quatro ZIPs na mesma pasta, preservando os diretórios. Execute:

    python codigo/verificar_integridade.py
    python codigo/consultar_modelo.py ieee123__bus__enriquecido_p0_p5__RandomForest --linhas 5

O exemplo usa a transformação já embutida e traduz os índices pela lista de
classes do respectivo modelo. Não treina, não seleciona nova configuração e
não altera os arquivos. Grupos de barras são aplicados somente à avaliação.

## Preservação e limites

Bases, modelos e previsões canônicas foram copiados com conferência SHA-256.
A resolução individual permanece disponível. Este pacote reúne somente
artefatos de simulação, modelos de ML e documentação técnica para seu uso.
Não houve novas simulações ou treinamentos para prepará-lo.

Os perfis espaciais foram copiados da execução canônica. As configurações
YAML mantêm os parâmetros científicos, com entrada DSS relativa à raiz deste
pacote e sem os destinos operacionais antigos. Os multiplicadores do teste
final constam no JSON de cada sistema. Não executar o pipeline histórico
selecionando apenas colunas derivadas: estes modelos usam o transformador
cartesiano distribuído aqui.

O manifesto identifica tamanho e hash de cada arquivo. Integridade não
demonstra validação operacional em campo. Respeitar as condições descritas em
LICENCAS.md e circuitos/ORIGEM.md ao reutilizar ou redistribuir os materiais.
