
# Projeto de Análise de Dados e Machine Learning Aplicado à Nutrição

## Descrição
Este projeto tem como objetivo analisar a qualidade nutricional de diferentes alimentos e classificar sua densidade nutricional em **Baixa, Média ou Alta** utilizando técnicas de **Machine Learning**.  
A análise combina exploração de dados, visualizações gráficas e modelo de **árvore de decisão**, permitindo entender quais nutrientes mais influenciam a densidade nutricional.

---

## Coleta de Dados
Os dados utilizados foram coletados em cinco arquivos CSV, representando diferentes grupos de alimentos:  

- `FOOD-DATA-GROUP1.csv`  
- `FOOD-DATA-GROUP2.csv`  
- `FOOD-DATA-GROUP3.csv`  
- `FOOD-DATA-GROUP4.csv`  
- `FOOD-DATA-GROUP5.csv`  

Cada arquivo contém informações sobre calorias, macronutrientes, vitaminas e minerais. Os datasets foram concatenados, limpos (remoção de duplicatas e valores nulos) e as colunas padronizadas para análise.

---

## Modelagem
### Pré-processamento
- Remoção de colunas irrelevantes (`Unnamed: 0` e `Unnamed: 0.1`).  
- Padronização das variáveis numéricas usando média e desvio padrão (`_std`).  
- Criação da variável categórica `Categoria_Nutricional` baseada na densidade nutricional:  
  - Baixa: até o primeiro tercil  
  - Média: entre o primeiro e segundo tercil  
  - Alta: acima do segundo tercil  

### Machine Learning
- Modelo escolhido: **DecisionTreeClassifier** (Árvore de Decisão)  
- Critério: `entropy`  
- Parâmetros:
  - `max_depth=20`  
  - `min_samples_split=3`  
  - `min_samples_leaf=3`  
  - `random_state=42`  
- Divisão treino/teste: 80% treino / 20% teste  

---

## Resultados

### Visualizações
1. **Distribuição de Calorias**
   ![Histograma de Calorias](imagens/distribuicao_calorias.png)

2. **Correlação entre Nutrientes**
   ![Heatmap de Correlação](imagens/correlacao_nutrientes.png)

3. **Relação entre Nutrientes por Categoria Nutricional**
   ![Pairplot](imagens/pairplot_categoria.png)

4. **Importância das Variáveis**
   ![Importância das Variáveis](imagens/importancia_variaveis.png)

### Avaliação do Modelo
- **Acurácia:** ~93,5%  
- **Matriz de Confusão:**  
  Permite verificar acertos e erros do modelo na classificação das categorias nutricionais.

### Predição de Novos Alimentos
- Teste com alimentos fictícios mostrou que o modelo consegue classificar corretamente novos casos, reforçando sua capacidade de generalização.

---

## Conclusões
- Nutrientes como **Proteínas, Cálcio e Vitamina C** possuem forte relação com a densidade nutricional dos alimentos.  
- A árvore de decisão fornece insights sobre quais variáveis são mais relevantes na classificação nutricional.  
- Visualizações exploratórias, como histogramas, heatmaps e pairplots, facilitam a interpretação dos padrões nutricionais.  
- Este estudo evidencia o potencial da **ciência de dados aplicada à nutrição**, sendo útil para educação alimentar, planejamento nutricional e desenvolvimento de sistemas de recomendação de alimentos.  
