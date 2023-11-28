import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Carregue seus dados a partir de um arquivo CSV
data = pd.read_csv("used_cars.csv")

# Especifique as colunas que você deseja remover
colunas_para_remover = ['accident', 'engine', 'transmission', 'ext_col', 'int_col']

# Use o método drop para remover as colunas
data = data.drop(colunas_para_remover, axis=1)

# Filtrar o DataFrame para manter apenas as linhas onde 'clean_title' seja igual a 'Yes'
data = data[data['clean_title'] == 'Yes']
# Use o método drop para remover a coluna 'clean_title'
data = data.drop('clean_title', axis=1)

# Ordene o DataFrame primeiro pela coluna 'brand' e, em seguida, pela coluna 'model'
data = data.sort_values(by=['brand', 'model'])

# Define uma função personalizada para remover vírgulas, '$' e outros não-numéricos de 'milage' e 'price'
def clean_numeric(value_str):
    return int(''.join(filter(str.isdigit, str(value_str))))

# Aplica a função personalizada para limpar as colunas 'milage' e 'price'
data['milage'] = data['milage'].apply(clean_numeric)
data['price'] = data['price'].apply(clean_numeric)

# Divida os dados em características (X) e rótulo (y)
X = data[['brand', 'model', 'model_year', 'milage', 'fuel_type']]
y = data['price']

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construa os modelos de Árvore de Decisão, Redes Neurais e SVM
dt_model = DecisionTreeRegressor()
nn_model = MLPRegressor(hidden_layer_sizes=(100, 50, 50), max_iter=200)
svm_model = SVR(kernel='linear')

# Crie uma instância do ColumnTransformer para lidar com variáveis categóricas e numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['model_year', 'milage']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'model', 'fuel_type'])
    ])

# Crie um pipeline para unir o pré-processamento e o modelo
dt_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', dt_model)])
nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', nn_model)])
svm_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', svm_model)])

# Ajuste os modelos aos dados de treinamento
dt_pipeline.fit(X_train, y_train)
nn_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

# Avalie os modelos nos dados de teste
dt_score = dt_pipeline.score(X_test, y_test)
nn_score = nn_pipeline.score(X_test, y_test)
svm_score = svm_pipeline.score(X_test, y_test)

# Imprima os scores
print(f"Decision Tree Score: {dt_score}")
print(f"Neural Network Score: {nn_score}")
print(f"SVM Score: {svm_score}")

# Calcule o preço médio dos carros na base de dados
media_preco = data['price'].mean()

# Imprima o preço médio
print(f"Preço Médio dos Carros: ${media_preco:.2f}")

# Calcule as métricas MAE, MSE e RMSE para cada modelo

# Decision Tree
dt_predictions = dt_pipeline.predict(X_test)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = mean_squared_error(y_test, dt_predictions, squared=False)

# Neural Network
nn_predictions = nn_pipeline.predict(X_test)
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_rmse = mean_squared_error(y_test, nn_predictions, squared=False)

# SVM
svm_predictions = svm_pipeline.predict(X_test)
svm_mae = mean_absolute_error(y_test, svm_predictions)
svm_mse = mean_squared_error(y_test, svm_predictions)
svm_rmse = mean_squared_error(y_test, svm_predictions, squared=False)

# Imprima as métricas
print("\nMétricas de Avaliação:")
print(f"\nDecision Tree - MAE: {dt_mae:.2f}, MSE: {dt_mse:.2f}, RMSE: {dt_rmse:.2f}")
print(f"Neural Network - MAE: {nn_mae:.2f}, MSE: {nn_mse:.2f}, RMSE: {nn_rmse:.2f}")
print(f"SVM - MAE: {svm_mae:.2f}, MSE: {svm_mse:.2f}, RMSE: {svm_rmse:.2f}")

# Compare o MAE e MSE em relação à média de preço
mae_proporcao = dt_mae / media_preco
mse_proporcao = dt_mse / media_preco

print("\nProporção em Relação à Média de Preço:")
print(f"Decision Tree - MAE Proporção: {mae_proporcao:.2f}")
print(f"Decision Tree - MSE Proporção: {mse_proporcao:.2f}")