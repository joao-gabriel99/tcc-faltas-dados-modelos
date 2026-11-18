# Dicionário e organização das bases

Cada registro representa um caso de falta simulado. Os arquivos brutos e
processados representam os mesmos casos, em dois estágios; não devem ser
concatenados como observações independentes. A relação completa de colunas,
tipos, contagens e níveis de carga está em metadados/bases.json.

| Campo ou família | Significado |
|---|---|
| system_id | Sistema de distribuição IEEE utilizado. |
| bus | Identificador original da barra de aplicação da falta. É rótulo, não grandeza de entrada. |
| type_id | Classe do tipo de falta, entre 1 e 11; ver catálogo abaixo. |
| fault_kind, phases, ground | Categoria mono/bi/tri, fases envolvidas e conexão à terra. |
| rf | Resistência da falta, em ohms. |
| load_mult | Multiplicador global adimensional de carga. |
| distribution_id | Perfil espacial P0 a P5; no texto do TCC, PC₀ a PC₅. |
| pre_V{1,2,3}_{mag,ang}, pre_I{1,2,3}_{mag,ang} | Fasores de tensão e corrente em condição normal; módulo em V/A, ângulo em radianos. |
| fault_V..., fault_I... | Mesmas medições com falta. |
| pre_sym_..., fault_sym_... | Componentes de sequência zero, positiva e negativa (índices 0, 1 e 2). |
| delta_..., delta_sym_... | Campos derivados históricos conservados nos Parquets. Para os modelos corrigidos, as diferenças complexas são recalculadas a partir das 24 colunas de fase; esses campos derivados não são sua entrada. |

O pipeline recebe 24 colunas brutas, na ordem RAW_COLUMNS do código distribuído.
Ele constrói internamente partes real e imaginária, seleciona os grupos de
variáveis e aplica as duas escalas V/I aprendidas no treino. Não fornecer as
81 colunas diretamente nem aplicar uma segunda normalização.

Não existe classe normal: as 12 medições pre_* acompanham cada caso de falta
como referência. São simulações estacionárias, sem ruído adicionado, e não
séries temporais coletadas em campo.

## Tipos de falta

| type_id | Fases e terra |
|---|---|
| 1 | A-terra |
| 2 | B-terra |
| 3 | C-terra |
| 4 | A-B |
| 5 | A-C |
| 6 | B-C |
| 7 | A-B-terra |
| 8 | A-C-terra |
| 9 | B-C-terra |
| 10 | A-B-C |
| 11 | A-B-C-terra |

## Separação experimental

Treino_cv contém os casos usados para treinamento e validação cruzada.
Teste_final contém 26 níveis intermediários de carga, distintos dos 27 níveis
usados no treinamento. Os seis perfis espaciais são os mesmos. O controle
usa somente P0; o enriquecido usa P0 a P5. Não somar os regimes como bases
disjuntas. As diferenças de amostragem e divisão não permitem atribuir a
diferença entre regimes exclusivamente à redistribuição espacial das cargas.

## Classes e previsões

Cada modelo utiliza LabelEncoder com a ordem exata registrada em classes no
seu JSON. Essa ordem é lexical, não necessariamente numérica. As colunas
real e predicted dos Parquets de previsões são índices inteiros codificados,
não IDs físicos de barras ou números originais dos tipos de falta. Sempre
decodificar por classes[i] do modelo correspondente. A coluna bus representa
a barra real individual e type_id representa o tipo original da falta.

O arquivo de previsões de 123 barras contém dataset_row explícito. Em 13/34,
o alinhamento usa o índice/ordem do conjunto após o filtro do regime e os
metadados do caso (barra, tipo, carga, perfil e resistência). Não concatenar
índices de modelos com codificações diferentes.

## Barras individuais e grupos

Os modelos de localização foram treinados com 13, 33 e 124 rótulos individuais,
respectivamente. A avaliação principal utiliza 12 grupos no IEEE 13, 33 barras
no IEEE 34 e 119 grupos no IEEE 123. Mapeamentos completos constam nos
metadados de cada sistema. A definição dos grupos foi posterior à observação
dos resultados; reduz a resolução e não equivale a um novo treinamento nem
a uma validação independente dessa definição.

Para agrupar: índice previsto → classe individual → grupo. Não usar um índice
das 124 barras diretamente na lista dos 119 grupos. As previsões individuais
e as métricas originais foram preservadas. Os resultados incluem acurácia
convencional, acurácia balanceada e F₁ macro, com valores na escala 0–1;
as três métricas têm definições distintas.
