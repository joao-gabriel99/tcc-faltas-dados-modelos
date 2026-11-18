# Treinamento e avaliação

Estas são cópias portáteis das rotinas usadas nas avaliações de13/34/123 barras.
Não houve novos ajustes ou novas previsões durante a preparação desta publicação.
Os módulos de fábricas, cálculo de métricas e transformação de entradas mantêm
as funções científicas originais; os caminhos de entrada/saída foram adaptados.

## Entradas e ambientes

Extraia todos os ZIPs na mesma pasta. Use os ambientes separados documentados em
metadados/ambientes.json e codigo/requirements_*.txt. Para o driver123 instale
também psutil. Executar treinamento pode levar muitas horas. Os comandos abaixo
não foram executados nesta publicação. As saídas novas ficam em execucoes/;
os modelos e dados distribuídos em sistemas/ não são sobrescritos.

## Validação cruzada de13/34

    python codigo/treinamento_e_avaliacao/treino_cv.py sistemas/ieee13/dados/treino_cv/dataset.parquet execucoes/ieee13/cv
    python codigo/treinamento_e_avaliacao/treino_cv.py sistemas/ieee34/dados/treino_cv/dataset.parquet execucoes/ieee34/cv

O controle usa5partições por multiplicador de carga e o enriquecido usa6rodadas
por perfil espacial. O ajuste auxiliar para matriz de validação não é o teste
final. O catálogo escolhas_finais_publicadas.json identifica as36escolhas
históricas, fixadas antes do teste final, sem nova seleção nesta publicação.

## Ajustes e teste final de13/34

    python codigo/treinamento_e_avaliacao/final_ieee13_34.py train
    python codigo/treinamento_e_avaliacao/final_ieee13_34.py evaluate

O plano reduzido conserva os24jobs e seus parâmetros. As funções de ajuste,
LabelEncoder, escalas, métricas, conferência independente e gravação de previsões
são as funções originais, com caminhos portáteis. As auditorias DSS por repetição de cenários pertencem à
simulação; esta entrada executa treino e teste sobre as bases já distribuídas.

## Desenvolvimento e avaliação123

    python codigo/treinamento_e_avaliacao/driver_ieee123.py desenvolvimento

O driver contém as312rodadas/ajustes de desenvolvimento e as rotinas congelar e
final. Para uma reexecução completa, a ordem é: gerar a base de treinamento,
executar desenvolvimento, congelar as escolhas, gerar um novo teste final no
caminho planejado e executar final. O código exige que o teste tenha sido gerado
após o congelamento e verifica hashes e fontes. Não contorne essas verificações
para escolher entradas usando o resultado do teste final.

    python codigo/treinamento_e_avaliacao/driver_ieee123.py congelar --planned-test-dataset ../nova_execucao_ieee123_teste/dataset.parquet
    python codigo/treinamento_e_avaliacao/driver_ieee123.py final --test-dataset ../nova_execucao_ieee123_teste/dataset.parquet

Os modelos são treinados por barra individual. Os agrupamentos da avaliação
principal são posteriores e seus mapeamentos/limitações continuam nos metadados;
não representam retreinamento nem seleção nova nesta publicação.
