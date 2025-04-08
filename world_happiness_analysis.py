import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import HeatMap

# Carregamento dos dados do relatório mundial de felicidade
# encoding: codificação de caracteres usada para garantir que caracteres especiais sejam lidos corretamente
# sep=',': define que os dados estão separados por vírgula
# iso-8859-1: codificação comum para evitar problemas com caracteres acentuados
df = pd.read_csv('C:/Dados para analise git/World-Happiness-Analysis-/world_happiness_report.csv',
                 sep=',', encoding='iso-8859-1')
# encoding: codificação de caracteres, normalmente utiliza-se o iso-8859-1, utf-8, latin-1))

#visualisa e retorna as cinco primeiras linhas do DataFrame
df.head()

# Exibir os números de linhas e colunas do DataFrame
df.shape

# Análise dos tipos de atributos.
# object: strings
# int64: inteiros
# float64: reais
# complex: complexos
df.dtypes

# Resumo estatístico das colunas numéricas do DataFrame
# Inclui estatísticas como média, desvio padrão, mínimo e máximo
print(df.describe())

# Informações gerais sobre o DataFrame, incluindo quantidade de valores nulos
print(df.info())

# Verificar a quantidade de valores ausentes por coluna
missing_values = df.isnull().sum()
print("\nValores ausentes por coluna:\n", missing_values)

# Cálculo das medidas centrais: média, mediana e moda
print("\nMédias:\n", df.mean(numeric_only=True))
print("\nMedianas:\n", df.median(numeric_only=True))
print("\nModa:\n", df.mode().iloc[0])

# Estatísticas descritivas com cálculos adicionais: assimetria (skewness) e curtose (kurtosis)
# Assimetria indica o grau de distribuição simétrica dos dados
# Curtose mede a concentração dos valores ao redor da média

descriptive_stats = df.describe()
descriptive_stats.loc['skewness'] = df.skew(numeric_only=True)
descriptive_stats.loc['kurtosis'] = df.kurt(numeric_only=True)
print("\nEstatísticas Descritivas com Assimetria e Curtose:\n", descriptive_stats)

# Visualização das distribuições das principais variáveis numéricas
features = ['Happiness_Score', 'GDP_per_Capita', 'Social_Support', 'Healthy_Life_Expectancy']
for feature in features:
    plt.figure(figsize=(8,6))
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribuição de {feature}')
    plt.show()

    # Teste de normalidade de Shapiro-Wilk
    stat, p = stats.shapiro(df[feature].dropna())
    print(f'Teste de Shapiro-Wilk para {feature}: Estatística={stat:.4f}, p={p:.4f}')
    if p > 0.05:
        print(f'{feature} segue distribuição normal.')
    else:
        print(f'{feature} não segue distribuição normal.')

# Análise exploratória inicial
print("Resumo estatístico dos dados:")
print(df.describe())  # Estatísticas descritivas das colunas numéricas
print("\nInformações sobre os dados:")
print(df.info())  # Estrutura do dataset (colunas, tipos e valores nulos)

# Verificação da quantidade de valores únicos por coluna
print("\nNúmero de valores únicos por coluna:\n", df.nunique())

# Tratamento de valores ausentes substituindo-os pela mediana de cada coluna
# Isso evita que caso exista valores nulos, estes prejudiquem análises estatísticas e modelagens preditivas
df.fillna(df.median(numeric_only=True), inplace=True)
print("\nValores ausentes após tratamento:\n", df.isnull().sum().sum())

# Intervalo de confiança para média da felicidade
confidence_interval = stats.t.interval(0.95, len(df['Happiness_Score'])-1, loc=np.mean(df['Happiness_Score']), scale=stats.sem(df['Happiness_Score']))
print("\nIntervalo de Confiança (95%) para a Média da Felicidade:\n", confidence_interval)

#Tratamento de valores ausentes (substituindo pela mediana)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\nValores ausentes após tratamento:\n", df.isnull().sum().sum())

# Removendo possíveis duplicatas do conjunto de dados
df.drop_duplicates(inplace=True)
print("\nNúmero de linhas após remoção de duplicatas:\n", df.shape[0])

# Detecção de outliers por meio de boxplot
plt.figure(figsize=(10,6))
sns.boxplot(y=df['Happiness_Score'])
plt.title('Detecção de Outliers no Índice de Felicidade')
plt.show()

# Correlação entre variáveis
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Correlação: O que influencia a felicidade?')
plt.show()

# Modelagem preditiva: prever o índice de felicidade a partir de variáveis explicativas
y = df['Happiness_Score']
X = df.drop(columns=['Happiness_Score', 'Year', 'Country'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Utilizando Regressão Linear (modelo estatístico que está buscando
#a relação entre variáveis independentes e dependentes
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Random Forest Regressor: modelo baseado em árvores de decisão que melhora
# a precisão da previsão
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Função de avaliação para medir o desempenho dos modelos de regressão
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)  # Erro médio quadrático
    rmse = np.sqrt(mse)  # Raiz do erro médio quadrático
    r2 = r2_score(y_true, y_pred)  # Coeficiente de determinação R²
    print(f'{model_name} Performance:')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2: {r2:.4f}')
    print('-' * 30)

# Aplicação de PCA (Análise de Componentes Principais)
# Redução de dimensionalidade para visualizar os dados em um espaço de menor dimensão
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]).drop(columns=['Happiness_Score', 'Year']))
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

# Visualização dos dados após PCA
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Happiness_Score'], palette='coolwarm')
plt.title('Análise de Componentes Principais (PCA)')
plt.show()

# Clusterização com K-Means: Agrupamento de países com base nas variáveis principais
o_modelo = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = o_modelo.fit_predict(df[['PCA1', 'PCA2']])

# Visualização dos clusters
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='viridis')
plt.title('Clusters Baseados em PCA')
plt.show()

# Distribuição da felicidade
plt.figure(figsize=(10,6))
sns.histplot(df['Happiness_Score'], bins=30, kde=True, color='blue')
plt.title('Distribuição do Índice de Felicidade')
plt.xlabel('Happiness Score')
plt.ylabel('Frequência')
plt.show()

# Média da felicidade por ano
avg_happiness_score = df.groupby('Year')['Happiness_Score'].mean().reset_index()
plt.figure(figsize=(14,10))
plt.plot(avg_happiness_score['Year'], avg_happiness_score['Happiness_Score'], marker='o', color='green')
plt.title('Média da Felicidade por Ano')
plt.xlabel('Ano')
plt.ylabel('Média da Felicidade')
plt.grid(True)
plt.show()

# Comparação da felicidade ao longo dos anos em países selecionados
selected_countries = ['United States', 'Brazil', 'China', 'Germany', 'India']
df_filtered = df[df['Country'].isin(selected_countries)]
plt.figure(figsize=(12,6))
sns.lineplot(data=df_filtered, x='Year', y='Happiness_Score', hue='Country', marker='o')
plt.title('Evolução da Felicidade ao Longo dos Anos')
plt.ylabel('Happiness Score')
plt.show()

latest_year = df['Year'].max() # Obtendo o último ano disponível no dataset
latest_data = df[df['Year'] == latest_year] # Filtrando os dados para incluir apenas o último ano disponível

# Selecionando os 10 países com os maiores índices de felicidade
top_10 = latest_data.sort_values(by='Happiness_Score', ascending=False).head(10)

# Remover esses países do dataframe antes de pegar os menos felizes
restantes = latest_data[~latest_data['Country'].isin(top_10['Country'])]

# Selecionando os 10 países com os menores índices de felicidade
bottom_10 = restantes.sort_values(by='Happiness_Score').head(10)

# Criar uma figura com dois gráficos lado a lado (com 1 linha, 2 colunas)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico dos 10 países mais felizes
sns.barplot(y='Country', x='Happiness_Score', data=top_10, palette='Blues_r', ax=axes[0])
axes[0].set_title('Top Países Mais Felizes')
axes[0].set_xlabel('Índice de Felicidade')
axes[0].set_ylabel('País')

# Gráfico dos 10 países menos felizes
sns.barplot(y='Country', x='Happiness_Score', data=bottom_10, palette='Reds', ax=axes[1])
axes[1].set_title('Top Países Menos Felizes')
axes[1].set_xlabel('Índice de Felicidade')
axes[1].set_ylabel('País')

# Ajustando espaçamento entre os gráficos
plt.tight_layout()
plt.show()

# Gráfico de dispersão entre PIB per capita e felicidade
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['GDP_per_Capita'], y=df['Happiness_Score'], alpha=0.6)
plt.xlabel('PIB per Capita')
plt.ylabel('Índice de Felicidade')
plt.title('Relação entre PIB per Capita e Felicidade')
plt.show()