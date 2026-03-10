import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

#Carregando os dados:
group1_data = pd.read_csv(r'dataset\FOOD-DATA-GROUP1.csv', sep=',')
group2_data = pd.read_csv(r'dataset\FOOD-DATA-GROUP2.csv', sep=',')
group3_data = pd.read_csv(r'dataset\FOOD-DATA-GROUP3.csv', sep=',')
group4_data = pd.read_csv(r'dataset\FOOD-DATA-GROUP4.csv', sep=',')
group5_data = pd.read_csv(r'dataset\FOOD-DATA-GROUP5.csv', sep=',')

#Concatenando os dataframes:
data = pd.concat([group1_data, group2_data, group3_data, group4_data, group5_data], ignore_index=True)
data = data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])
data = data.drop_duplicates()
print(data.head())

#Limpando valores nulos e padronizando nomes das colunas:
data = data.dropna()
data.columns = data.columns.str.strip()

#Análise Exploratória:

#Distribuição de Calorias nos Alimentos:
plt.figure(figsize=(8,5))
sns.histplot(data["Caloric Value"], bins=30)
plt.title("Distribuição de Calorias nos Alimentos")
plt.show()

#Análise da Distribuição de Calorias nos Alimentos:
#O histograma apresenta a distribuição do valor calórico dos alimentos presentes no dataset. Observa-se que a maior parte dos alimentos possui valores calóricos relativamente baixos, concentrando-se nas primeiras faixas do gráfico.
#À medida que o valor calórico aumenta, a quantidade de alimentos nessa faixa diminui significativamente, indicando que alimentos muito calóricos são menos frequentes no conjunto de dados. Também é possível notar a presença de alguns valores extremos (outliers) com calorias muito altas, o que pode representar alimentos altamente energéticos ou preparações específicas.
#De forma geral, a distribuição é assimétrica à direita, mostrando que a maioria dos alimentos tem menor densidade calórica, enquanto poucos alimentos apresentam valores calóricos muito elevados. Essa análise ajuda a compreender o perfil energético dos alimentos presentes no dataset.

#Correlação entre Nutrientes:
plt.figure(figsize=(14,12))
corr = data.select_dtypes(include="number").corr()
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": .8})
plt.title("Correlação entre Nutrientes")
plt.show()

#Análise da Correlação entre Nutrientes:
#O mapa de calor apresenta as correlações entre os diferentes nutrientes presentes nos alimentos. Observa-se que o valor calórico possui forte correlação com gorduras, pois lipídios têm alta densidade energética. Também há correlação positiva entre carboidratos e açúcares, já que os açúcares fazem parte da composição dos carboidratos.
#Alguns minerais, como fósforo, potássio e magnésio, apresentam correlação entre si e com proteínas, indicando que certos alimentos concentram vários nutrientes importantes ao mesmo tempo. As vitaminas do complexo B também mostram relações moderadas entre si, pois frequentemente aparecem juntas em diversos alimentos.
#Por fim, a densidade nutricional apresenta correlação positiva com vários nutrientes, principalmente proteínas, vitaminas e minerais, sugerindo que alimentos mais nutritivos tendem a possuir maior concentração desses componentes.

#Padronização da escala dos valores:
#Colunas numéricas:
colunas_numericas = [
    'Caloric Value','Fat','Carbohydrates','Protein','Sugars','Dietary Fiber','Nutrition Density','Sodium',
    'Water','Vitamin A','Vitamin B1','Vitamin B11','Vitamin B12','Vitamin B2','Vitamin B3','Vitamin B5','Vitamin B6',
    'Vitamin C','Vitamin D','Vitamin E','Vitamin K','Calcium','Copper','Iron','Magnesium','Manganese',
    'Phosphorus','Potassium','Selenium','Zinc'
]

medias = {}
desvios = {}
for col in colunas_numericas:
    medias[col] = data[col].mean()
    desvios[col] = data[col].std()
    data[col + '_std'] = (data[col] - medias[col]) / desvios[col]

colunas_padronizadas = [col + "_std" for col in colunas_numericas]
data = data[colunas_padronizadas]
print(data.head())

#Criando as categorias de densidade nutricional:
q1 = data['Nutrition Density_std'].quantile(0.33)
q2 = data['Nutrition Density_std'].quantile(0.66)

def classificar_densidade(x):
    if x <= q1:
        return "Baixa"
    elif x <= q2:
        return "Media"
    else:
        return "Alta"

data['Categoria_Nutricional'] = data['Nutrition Density_std'].apply(classificar_densidade)
print(data.head())

#Distribuição de Proteínas por Categoria Nutricional:
nutrientes_principais = [
    'Caloric Value_std','Protein_std','Fat_std','Carbohydrates_std','Calcium_std','Vitamin C_std','Categoria_Nutricional'
]

data_pairplot = data[nutrientes_principais]

with sns.axes_style('whitegrid'):
    pairplot = sns.pairplot(
        data_pairplot,
        hue="Categoria_Nutricional",
        palette="Set2",
        diag_kind="kde",
        corner=False,
        plot_kws={'alpha':0.7, 's':40}
    )

pairplot.fig.suptitle("Relações entre Nutrientes por Categoria Nutricional", y=1.02)
plt.show()

#Análise das relações entre Nutrientes por Categoria Nutricional:
#O pairplot mostra como os principais nutrientes (Proteínas, Cálcio, Calorias, Gorduras, Carboidratos e Vitamina C) se relacionam com a categoria nutricional dos alimentos (Baixa, Média, Alta).
#As diagonais exibem a distribuição de cada nutriente, permitindo ver concentrações e outliers.
#Os scatter plots indicam como pares de nutrientes se correlacionam e destacam que alimentos de alta densidade nutricional tendem a ter mais proteínas e cálcio.
#A separação por cores evidencia quais nutrientes diferenciam melhor as categorias, confirmando os insights do modelo de árvore de decisão.
#Em resumo, o gráfico facilita a visualização de padrões nutricionais e relações entre nutrientes, reforçando a interpretação do modelo.

#Machine Learning:
#Separando as variáveis para machine learning
x = data[
    [
        'Calcium_std','Caloric Value_std','Vitamin C_std','Protein_std','Sugars_std','Iron_std','Fat_std',
        'Carbohydrates_std','Dietary Fiber_std','Vitamin E_std','Magnesium_std','Vitamin B1_std','Water_std',
        'Vitamin B2_std','Vitamin B12_std'
    ]
]

y = data['Categoria_Nutricional']

#Separar treino e teste:
x_train, x_test, y_train, y_test = train_test_split(
  x,
  y,
  test_size=0.20,
  random_state=42
)

#Treinar a árvore de decisão:
modelo = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=3,
    random_state=42
)

modelo.fit(x_train, y_train)

#Visualização da Matriz de confusão:
plt.figure(figsize=(20,15))

tree.plot_tree(
modelo,
feature_names=x.columns,
class_names=modelo.classes_,
filled=True
)

plt.show()

#Influência dos nutrientes na densidade nutricional:
importancias = pd.Series(modelo.feature_importances_, index=x.columns)

with sns.axes_style('whitegrid'):
    importancias.sort_values().plot(kind='barh')
    plt.title("Importância das Variáveis")
    plt.show()

#Fazer predição:
y_pred = modelo.predict(x_test)
print(y_pred[0:10])

#Cálculo da acurácia:
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia de: {round(100 * accuracy, 2)}%')

#Matriz de confusão:
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.show()

#Resultado do modelo:
#O modelo de árvore de decisão obteve uma acurácia de aproximadamente 93.5%, indicando que foi capaz de classificar corretamente a maioria dos alimentos em relação à sua densidade nutricional. Esse resultado demonstra que nutrientes como proteínas, vitaminas e minerais possuem forte relação com a qualidade nutricional dos alimentos. Além disso, a árvore de decisão permite interpretar quais variáveis são mais relevantes para a classificação.

#Predição de alimentos fictícios para teste:
#Criação dos alimentos:
novos_alimentos = pd.DataFrame([
    [50,120,30,5,10,2,3,20,4,1.2,30,0.3,70,0.2,0.1],
    [20,300,5,12,20,3,15,40,2,0.5,25,0.4,60,0.3,0.2],
    [80,90,40,3,8,1.5,2,15,5,1.5,40,0.2,80,0.1,0.0],
    [2,25,12,0,1,4,18,18,3,2.0,50,0.5,55,0.4,0.3],
    [15,350,2,8,25,2.5,20,50,1,0.3,20,0.6,50,0.2,0.1]
], columns=[
    'Calcium','Caloric Value','Vitamin C','Protein','Sugars','Iron','Fat','Carbohydrates','Dietary Fiber',
    'Vitamin E','Magnesium','Vitamin B1','Water','Vitamin B2','Vitamin B12'
])

#Padronização da escala dos valores e limpeza das colunas:
for col in novos_alimentos.columns:
    if col in medias:
        novos_alimentos[col + "_std"] = (novos_alimentos[col] - medias[col]) / desvios[col]

colunas_novos_alimentos_manter = [col for col in novos_alimentos.columns if '_std' in col]
novos_alimentos = novos_alimentos[colunas_novos_alimentos_manter]
novos_alimentos = novos_alimentos[x.columns]

print(novos_alimentos.head())

#Predição dos alimentos criados:
previsao = modelo.predict(novos_alimentos)
print(previsao)

#Conclusão:
#O projeto demonstrou que a análise de dados combinada com técnicas de machine learning é uma ferramenta eficaz para entender a qualidade nutricional dos alimentos. Através da exploração dos dados, foi possível identificar padrões entre macronutrientes, vitaminas e minerais, evidenciando que nutrientes como proteínas, cálcio e vitamina C possuem forte relação com a densidade nutricional.
#O modelo de árvore de decisão apresentou acurácia de aproximadamente 93,5%, mostrando que consegue classificar corretamente a maioria dos alimentos em categorias de baixa, média e alta densidade nutricional. A análise de importância das variáveis destacou quais nutrientes mais influenciam essa classificação, fornecendo informações valiosas para educação alimentar, planejamento nutricional e escolhas mais saudáveis.
#Além disso, os gráficos exploratórios, como histogramas, heatmaps e pairplots, facilitaram a visualização de padrões nutricionais e a relação entre os principais nutrientes, complementando a interpretação do modelo.
#Em resumo, este estudo evidencia o potencial da ciência de dados aplicada à nutrição, oferecendo insights que podem apoiar decisões em saúde, educação alimentar e desenvolvimento de sistemas de recomendação de alimentos. Futuras melhorias podem incluir modelos mais sofisticados, integração de novos grupos alimentares e visualizações interativas para análise ainda mais detalhada.