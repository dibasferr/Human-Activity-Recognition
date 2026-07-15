# Human Activity Recognition (HAR) using Wearable Sensors

Este repositório contém o projeto de **Classificação de Atividades Humanas** desenvolvido no âmbito da unidade curricular de **Engenharia de Aprendizagem / Extração de Conhecimento a partir de Dados (EA/ECAC)** na **Universidade de Coimbra**.

O objetivo é desenvolver uma pipeline completa de Machine Learning capaz de classificar 7 atividades físicas humanas comuns (como caminhar, correr, subir escadas, etc.) utilizando dados de sensores inerciais (acelerómetro, giroscópio e magnetómetro) de dispositivos vestíveis (wearables).

---

## Visão Geral do Projeto

O projeto aborda o ciclo completo de uma pipeline de Ciência de Dados aplicada a sinais de sensores corporais:

1. **Pré-processamento & Limpeza de Dados:** Análise exploratória, tratamento de outliers (métodos IQR, Z-Score e K-Means) e cálculo de métricas de densidade de ruído.
2. **Engenharia de Características (Feature Engineering):**
   * **Abordagem Clássica (FEATURES):** Extração manual de características estatísticas no domínio do tempo e da frequência (janelas de 5s com 50% de overlap).
   * **Abordagem de Deep Learning (EMBEDDINGS):** *Transfer Learning* utilizando o modelo pré-treinado **HARNet5** (projeto *ssl-wearables*) aplicado aos dados brutos do acelerómetro.
3. **Seleção e Redução de Dimensionalidade:** Aplicação de **PCA** (mantendo 90% da variância) e **ReliefF** (seleção das 15 melhores características).
4. **Validação e Modelação:**
   * Estratégias de divisão de dados: **Intra-Subject** (avaliação interna) vs. **Inter-Subject** (generalização para novos utilizadores).
   * Algoritmo de classificação **K-Nearest Neighbors (KNN)** com otimização de hiperparâmetros ($k$).
   * Testes de hipóteses estatísticas para validação de significância entre modelos.

---

## Como Executar e Fazer Previsões (Deployment)

Para testar o melhor modelo identificado no projeto sem ter de reexecutar toda a pipeline de treino, siga estas instruções:

1. Localize a secção demarcada com `# 6. Deployment` no ficheiro `src/Alternativa_otimizada.py` (ou no script correspondente).
2. Descomente **apenas** essa secção do código.
3. Forneça os dados de entrada para previsão no formato de um **NumPy Array com shape `(256, 9)`** correspondente a:
   * Colunas: `[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]`.
4. Execute o script. O modelo fará todo o pré-processamento, extração de características e classificação de forma autónoma.

```python
# Exemplo de código na secção de Deployment:
features = np.load("data/cache_vetor_features.npy", allow_pickle=True)

idx_inicio = # Defina o índice de início desejado
idx_fim    = # Defina o índice de fim desejado

# Seleção de um segmento de teste (ex: primeiras 256 amostras do dispositivo 2 do participante 1)
data = dadosParticionados[0][idx_inicio:idx_fim] 

previsao(data)
