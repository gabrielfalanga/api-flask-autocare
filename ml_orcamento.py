import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# importando base de dados
df = pd.read_csv('dataset_carros_orcamento_5000.csv')
print(df.head())

"""## 1. Pré-processamento dos dados"""

# Variáveis categóricas a serem transformadas
colunas_categoricas = ['Marca', 'Modelo', 'Combustível', 'Cidade', 'Problema']

# Usando One-Hot Encoding para transformar variáveis categóricas em numéricas
df_encoded = pd.get_dummies(df, columns=colunas_categoricas)
print(df_encoded.head())

# Separando as features (X) e a variável alvo (y)
X = df_encoded.drop(columns=['Custo Estimado'])  # Removemos a coluna do custo, que é o target
y = df_encoded['Custo Estimado']  # O target é o custo estimado

"""## 2. Dividindo os dados em treino e teste (80% treino, 20% teste)"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## 3. Treinando o modelo Random Forest Regressor"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

"""## 4. Fazendo previsões no conjunto de teste"""

y_pred = rf_model.predict(X_test)

"""## 5. Avaliando o desempenho do modelo"""

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

"""## Importância das variáveis"""

importances = rf_model.feature_importances_

# Convertendo para DataFrame para melhor visualização
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False, ignore_index=True)

print(feature_importance_df)  # Mostra as variáveis mais importantes

"""# Aplicando"""

print(list(rf_model.predict(X_test.iloc[[10, 20, 996]])))
print(list(y_test.iloc[[10, 20, 996]]))
