# Origem e uso dos circuitos OpenDSS

Esta pasta reúne as cópias dos circuitos usadas no experimento, separadas por sistema. Seus parâmetros elétricos não foram redesenhados ou recalibrados nesta preparação.

## Entradas de reprodução

- IEEE 13: `ieee13/IEEE13Nodeckt.dss`.
- IEEE 34: `ieee34/ieee34Mod1.dss`, modelo padrão Mod 1.
- IEEE 123: `ieee123/IEEE123_prod.dss`, carregador portátil para o master original em `ieee123/originais/trunk/Distrib/IEEETestCases/123Bus/`.

Preserve estas pastas. Os três arquivos chamados `IEEELineCodes.DSS` possuem conteúdos diferentes e não podem ser reunidos em um único arquivo. O CSV do sistema 13 é referenciado por BusCoords e deve acompanhar o master. O CSV do sistema 34 e o DAT do sistema 123 são recursos auxiliares de coordenadas.

O executor deve compilar a entrada correspondente, definir um DataPath gravável para suas saídas e aplicar a configuração experimental documentada no pacote principal. O carregador portátil do IEEE 123 mantém o mesmo master e MaxControlIter=100 e remove os caminhos absolutos da máquina de origem.

## IEEE 13

Fonte: exemplo IEEE 13 da distribuição OpenDSS. O cabeçalho preserva a atribuição a Tyler Patton, Jon Wood e David Woods, estudantes da Tennessee Tech University, abril de 2009. A cópia usada mantém as definições elétricas, usa CalcVoltageBases por extenso e deixa o Solve demonstrativo comentado. As soluções são comandadas pelo programa Python do experimento. Comentários demonstrativos ao final foram preservados.

Não foi comprovada uma revisão SVN histórica exata desta cópia. Não se atribui a ela o snapshot r4176 do outro sistema nem a versão atual do aplicativo gráfico como versão de origem. Os hashes identificam os bytes efetivamente usados.

## IEEE 34

Modelo IEEE 34 Mod 1, copiado da distribuição oficial OpenDSS em 17/08/2026. As diferenças observadas frente à cópia da instalação usada como referência local são terminações de linha e um espaço final. Não foi comprovada uma revisão SVN histórica exata.

O Mod 2, que cria barras intermediárias para representar cargas distribuídas, não é usado neste experimento.

## IEEE 123

Fontes obtidas do repositório oficial EPRI/OpenDSS no SourceForge em 06/11/2026, fixadas no snapshot SVN **r4176**. O master, suas três dependências elétricas e recursos originais preservam seus bytes.

- Página oficial: https://opendss.epri.com/Downloads.html
- Repositório: https://svn.code.sf.net/p/electricdss/code/
- Circuito no snapshot: https://svn.code.sf.net/p/electricdss/code/!svn/bc/4176/trunk/Distrib/IEEETestCases/123Bus/
- Licença no snapshot: https://svn.code.sf.net/p/electricdss/code/!svn/bc/4176/trunk/License.txt

O arquivo `fonte_manifesto.json` registra URLs, ETags, datas e hashes originais. Os hashes dos sete arquivos originais permanecem disponíveis para conferência.

`Run_IEEE123Bus.DSS` está preservado somente como exemplo da fonte. **Não o use como entrada de reprodução.** Ele contém duas execuções, comandos de plotagem e relatórios e referencia um script de plotagem que não integra este conjunto e não é necessário ao master.

## Direitos e atribuições

Os exemplos integram a distribuição OpenDSS/EPRI, com suas atribuições preservadas. Textos integrais de licença estão em `../licencas/terceiros/`. A licença do snapshot também permanece em `ieee123/originais/trunk/License.txt`.

A licença geral permite redistribuição e uso, com ou sem modificação, sob as condições de preservar avisos, condições e disclaimer e não usar nomes da EPRI ou de contribuidores como endosso sem autorização. Os arquivos lidos não têm identificador SPDX individual. Este pacote não apresenta exemplos ou dados de referência IEEE como criação exclusiva do autor do TCC.

Versão do motor DSS C-API, versão de dss-python e snapshot do circuito são informações diferentes. A instalação atual do aplicativo gráfico não estabelece equivalência numérica com o ambiente registrado do experimento.
