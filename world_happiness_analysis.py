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


# Carregamento dos dados
df = pd.read_csv('C:/Dados para analise git/world_happiness_report.csv',
                 sep=',', encoding='iso-8859-1')
# encoding: codificação de caracteres, normalmente utiliza-se o iso-8859-1, utf-8, latin-1))


df.head()
#visualisa e retorna as cinco primeiras linhas do DataFrame

df.shape

# Análise dos tipos de atributos.
# object: strings
# int64: inteiros
# float64: reais
# complex: complexos
df.dtypes

# Exibição de informações gerais sobre o dataset
# object: strings
# int64: inteiros
# float64: reais
# complex: complexos
print("Informações gerais sobre o dataset:\n")
df.info()

# Verificando se houve valores ausentes
missing_values = df.isnull().sum()
print("\nValores ausentes por coluna:\n", missing_values)

# Medidas centrais
print("\nMédias:\n", df.mean(numeric_only=True))
print("\nMedianas:\n", df.median(numeric_only=True))
print("\nModa:\n", df.mode().iloc[0])

# Estatísticas descritivas com curtose e assimetria
descriptive_stats = df.describe()
descriptive_stats.loc['skewness'] = df.skew(numeric_only=True)
descriptive_stats.loc['kurtosis'] = df.kurt(numeric_only=True)
print("\nEstatísticas Descritivas com Assimetria e Curtose:\n", descriptive_stats)

# Distribuições e Histogramas
features = ['Happiness_Score', 'GDP_per_Capita', 'Social_Support', 'Healthy_Life_Expectancy']
for feature in features:
    plt.figure(figsize=(8,6))
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribuição de {feature}')
    plt.show()

    stat, p = stats.shapiro(df[feature].dropna())
    print(f'Teste de Shapiro-Wilk para {feature}: Estatística={stat:.4f}, p={p:.4f}')
    if p > 0.05:
        print(f'{feature} segue distribuição normal.')
    else:
        print(f'{feature} não segue distribuição normal.')

# Análise de valores únicos
unique_values = df.nunique()
print("\nNúmero de valores únicos por coluna:\n", unique_values)

# Teste de normalidade (Shapiro-Wilk)
shapiro_test = stats.shapiro(df['Happiness_Score'].dropna())
print("\nTeste de normalidade (Shapiro-Wilk) para Happiness Score:\n", shapiro_test)

# Intervalo de confiança para média da felicidade
confidence_interval = stats.t.interval(0.95, len(df['Happiness_Score'])-1, loc=np.mean(df['Happiness_Score']), scale=stats.sem(df['Happiness_Score']))
print("\nIntervalo de Confiança (95%) para a Média da Felicidade:\n", confidence_interval)

#Tratamento de valores ausentes (substituindo pela mediana)
df.fillna(df.median(numeric_only=True), inplace=True)
print("\nValores ausentes após tratamento:\n", df.isnull().sum().sum())

#Removendo duplicatas
df.drop_duplicates(inplace=True)
print("\nNúmero de linhas após remoção de duplicatas:\n", df.shape[0])

# Análise de outliers
plt.figure(figsize=(10,6))
sns.boxplot(y=df['Happiness_Score'])
plt.title('Detecção de Outliers no Índice de Felicidade')
plt.show()

# Análise exploratória
plt.figure(figsize=(18,12))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Mapa de Correlação: O que influencia a felicidade?')
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

# Último ano disponível
latest_year = df['Year'].max()
latest_data = df[df['Year'] == latest_year]

# Top 10 países mais e menos felizes
top_10 = latest_data.sort_values(by='Happiness_Score', ascending=False).head(10)
bottom_10 = latest_data.sort_values(by='Happiness_Score').head(10)

fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.barplot(y='Country', x='Happiness_Score', data=top_10, ax=axes[0], palette='Blues_r')
axes[0].set_title('Os 10 Países Mais Felizes do Mundo')
sns.barplot(y='Country', x='Happiness_Score', data=bottom_10, ax=axes[1], palette='Reds')
axes[1].set_title('Os 10 Países Menos Felizes do Mundo')
plt.show()

# Modelagem preditiva
y = df['Happiness_Score']
X = df.drop(columns=['Happiness_Score', 'Year', 'Country'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Random Forest Regressor
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Função de avaliação
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'{model_name} Performance:')
    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'R2: {r2:.4f}')
    print('-' * 30)

evaluate_model(y_test, y_pred_lr, 'Regressão Linear')
evaluate_model(y_test, y_pred_rf, 'Random Forest')

# Gráfico de dispersão entre PIB per capita e felicidade
plt.figure(figsize=(10,6))
sns.scatterplot(x=df['GDP_per_Capita'], y=df['Happiness_Score'], alpha=0.6)
plt.xlabel('PIB per Capita')
plt.ylabel('Índice de Felicidade')
plt.title('Relação entre PIB per Capita e Felicidade')
plt.show()

# Aplicação de PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df.select_dtypes(include=[np.number]).drop(columns=['Happiness_Score', 'Year']))
df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]

plt.figure(figsize=(10,6))
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Happiness_Score'], palette='coolwarm')
plt.title('Análise de Componentes Principais (PCA)')
plt.show()

# Clusterização com K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['PCA1', 'PCA2']])
sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='viridis')
plt.title('Clusters Baseados em PCA')
plt.show()

# Modelagem preditiva simples
X = df[['GDP_per_Capita', 'Social_Support', 'Healthy_Life_Expectancy', 'Freedom', 'Generosity', 'Corruption_Perception']]
y = df['Happiness_Score']

# Separação dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Erro Quadrático Médio (MSE): {mse:.2f}')
print(f'Coeficiente de Determinação (R²): {r2:.2f}')