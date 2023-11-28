# Predicting-Car-Prices-with-ML

Este script em Python utiliza a biblioteca scikit-learn para criar e avaliar modelos de regressão para prever preços de carros com base em diferentes características. O conjunto de dados utilizado é carregado de um arquivo CSV chamado "used_cars.csv". Abaixo, um resumo dos processos realizados:

#Processos Realizados:
###Carregamento dos Dados:

Utilização da biblioteca pandas para carregar dados de um arquivo CSV chamado "used_cars.csv".
###Pré-processamento dos Dados:

Remoção de colunas não desejadas, especificadas na lista colunas_para_remover.
Filtragem do DataFrame para manter apenas as linhas onde a coluna 'clean_title' é igual a 'Yes'.
Remoção da coluna 'clean_title'.
Ordenação do DataFrame pelas colunas 'brand' e 'model'.
Definição de uma função clean_numeric para remover caracteres não numéricos das colunas 'milage' e 'price'.
Aplicação da função para limpar as colunas 'milage' e 'price'.
###Divisão dos Dados:

Separação dos dados em características (X) e rótulo (y).
Divisão dos dados em conjuntos de treinamento e teste usando train_test_split.
###Construção de Modelos:

Utilização de três modelos de regressão: Árvore de Decisão, Redes Neurais e Máquinas de Vetores de Suporte (SVM).
###Pré-processamento e Pipeline:

Criação de um ColumnTransformer para lidar com variáveis categóricas e numéricas.
Construção de pipelines para unir o pré-processamento e os modelos.
###Treinamento dos Modelos:

Ajuste dos modelos aos dados de treinamento.
###Avaliação dos Modelos:

Avaliação dos modelos nos dados de teste utilizando o método score.
Cálculo do preço médio dos carros na base de dados.
###Métricas de Avaliação:

Cálculo de métricas como MAE, MSE e RMSE para cada modelo.
Comparação dessas métricas em relação à média de preço.
#Como Executar o Código:
Certifique-se de ter a biblioteca pandas e scikit-learn instaladas (pip install pandas scikit-learn).
Baixe o arquivo CSV "used_cars.csv" e ajuste o caminho no código, se necessário.
Execute o script.
Resultados Impressos:
Pontuação (score) de cada modelo nos dados de teste.
Preço médio dos carros na base de dados.
Métricas de Avaliação (MAE, MSE, RMSE) para cada modelo.
Proporção do MAE e MSE em relação à média de preço.
#Observações:
Os modelos foram treinados e avaliados para prever preços de carros com base nas características fornecidas no conjunto de dados.
A análise das métricas em relação à média de preço fornece uma perspectiva sobre o desempenho relativo dos modelos.
