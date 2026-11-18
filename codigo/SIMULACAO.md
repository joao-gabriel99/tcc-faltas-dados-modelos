# Código de simulação

O núcleo em `src/opendss_tcc` contém os procedimentos usados para inspecionar
as fases, criar as faltas, aplicar os perfis espaciais, medir fasores e montar
as colunas derivadas. Os arquivos do núcleo são cópias do código canônico,
sem alterações das equações, da enumeração de faltas ou da política de controles.

Cada caso recompila o circuito, estabiliza controles em condição normal e os
congela durante a falta. As configurações e os parâmetros publicados conservam
os dados de referência. A ordem exata das fases/barras usada pelo gerador
paralelo de123barras está em `simulacao123/parametros_geracao.json`.

## Geração sequencial

Instale as dependências em ambiente separado, observando as versões originais
registradas em `metadados/ambientes.json`. Para gerar uma nova base de13barras:

    python codigo/gerar_dados.py ieee13 --etapa treino_cv --saida CAMINHO_NOVO --confirmar-geracao

Troque por `ieee34` ou `ieee123` para outro circuito. Use `--etapa teste_final`
para os níveis intermediários publicados. A pasta de saída deve ser nova e
ficar fora deste pacote. Essa interface usa o mesmo núcleo sequencial; o
experimento original de123barras utilizou a orquestração paralela abaixo.

## Geração paralela de123barras

    python codigo/simulacao123/gerar_ieee123.py dev --saida CAMINHO_NOVO --workers 4 --confirmar-geracao
    python codigo/simulacao123/gerar_ieee123.py test --saida CAMINHO_NOVO --workers 4 --confirmar-geracao

O driver é uma adaptação portátil do executor original. Mantém os processos
isolados, checkpoints por hash, parâmetros elétricos, enumeração dos casos,
congelamento de controles e montagem dos arquivos. A preparação agora lê
somente os parâmetros públicos; na geração do teste final confere o catálogo
das12configurações fixadas antes desse teste. Não seleciona modelos novamente.
O estágio `smoke` executa4casos quando solicitado pelo usuário.

As duas interfaces produzem novos arquivos e auditorias na pasta indicada.
Não modificam as bases distribuídas. Os arquivos Parquet publicados continuam
sendo os canônicos. Alterações de versão, caminhos, interface e paralelismo
podem alterar o ambiente, a serialização e os tempos; a preparação deste
pacote não refez as simulações e não garante hashes/tempos idênticos.
