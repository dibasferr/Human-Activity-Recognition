# Human Activity Recognition using Wearable Sensors
### Machine Learning • Feature Engineering • Deep Learning • KNN • HARNet5

Este repositório contém o projeto de **Human Activity Recognition (HAR)** desenvolvido no âmbito da unidade curricular de **Engenharia de Aprendizagem / Extração de Conhecimento a partir de Dados (EA/ECAC)** da **Universidade de Coimbra**.

O objetivo consiste em desenvolver uma pipeline completa de **Machine Learning** capaz de reconhecer automaticamente **7 atividades físicas humanas** recorrendo a dados provenientes de sensores inerciais de dispositivos *wearable*.

As atividades são classificadas utilizando informação de:

-  Acelerómetro
-  Giroscópio
-  Magnetómetro

O projeto compara diferentes estratégias de engenharia de características, redução de dimensionalidade e validação, analisando o seu impacto na capacidade de generalização do modelo.

---

# Índice

- [Visão Geral](#-visão-geral)
- [Pipeline](#-pipeline-de-machine-learning)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Como Executar](#-como-executar)
- [Deployment](#-deployment)
- [Visualização dos Resultados](#-visualização-dos-resultados)
- [Modelos Avaliados](#-modelos-avaliados)
- [Tecnologias](#-tecnologias-utilizadas)

---

#  Visão Geral

O projeto explora todo o ciclo de desenvolvimento de um sistema de reconhecimento de atividades humanas, desde o tratamento dos dados até à avaliação estatística dos modelos.

As principais etapas incluem:

- limpeza de dados;
- deteção de outliers;
- engenharia de características;
- redução de dimensionalidade;
- treino de modelos;
- validação;
- análise estatística dos resultados;
- deployment do modelo final.

---

#  Pipeline de Machine Learning

```text
Sensores Wearable
        │
        ▼
Pré-processamento
        │
        ▼
Remoção de Outliers
        │
        ▼
Feature Engineering
   ├── Features Clássicas
   └── HARNet5 Embeddings
        │
        ▼
Redução de Dimensionalidade
   ├── PCA
   └── ReliefF
        │
        ▼
Classificação (KNN)
        │
        ▼
Validação
        │
        ▼
Análise Estatística
```

---

# Pré-processamento

Antes do treino foram efetuadas várias etapas de limpeza e preparação dos dados.

Incluem:

- análise exploratória;
- normalização;
- deteção de outliers;
- avaliação da densidade de ruído.

Foram comparadas diferentes abordagens:

- IQR
- Z-Score
- K-Means

---

# Engenharia de Características

O projeto compara duas abordagens distintas.

## Features Clássicas

Extração manual de características estatísticas no domínio:

- temporal;
- frequencial.

Características:

- janelas de **5 segundos**
- **50% de overlap**

---

## Embeddings (Deep Learning)

Utilização de **Transfer Learning** através do modelo pré-treinado **HARNet5** (*ssl-wearables*).

Neste caso, os embeddings são extraídos diretamente dos sinais brutos do acelerómetro.

---

# Redução de Dimensionalidade

Foram avaliadas duas estratégias.

## PCA

- preservação de 90% da variância.

## ReliefF

- seleção automática das **15 características mais relevantes**.

---

# Modelo de Classificação

Foi utilizado o algoritmo:

- **K-Nearest Neighbors (KNN)**

Foram efetuados testes para diferentes valores de:

- `k`

---

# Estratégias de Validação

Foram comparadas duas formas de divisão dos dados.

## Intra-Subject

Treino e teste utilizando dados do mesmo utilizador.

Objetivo:

Avaliar a capacidade do modelo aprender padrões individuais.

---

## Inter-Subject

Treino e teste utilizando participantes diferentes.

Objetivo:

Avaliar a capacidade de generalização para novos utilizadores.

---

# Estrutura do Projeto

```text
data/
│
├── matriz_de_resultados.npy
├── matriz_de_f1Score.npy
├── cache_vetor_features.npy
│

src/
│
├── mainActivity.py
├── deployment.py
├── utils.py
└── ...
```

---

# Como Executar

Após clonar o repositório:

```bash
pip install -r requirements.txt
```

Depois execute:

```bash
python src/mainActivity.py
```

---

# Deployment

O modelo final pode ser utilizado sem necessidade de repetir todo o processo de treino.

## Passos

1. Abrir `src/mainActivity.py`

2. Localizar:

```python
# 6. Deployment
```

3. Descomentar apenas essa secção.

4. Fornecer um array NumPy com dimensão:

```python
(256, 9)
```

correspondente às colunas:

```text
acc_x
acc_y
acc_z
gyr_x
gyr_y
gyr_z
mag_x
mag_y
mag_z
```

---

## Exemplo

```python
features = np.load(
    "data/cache_vetor_features.npy",
    allow_pickle=True
)

idx_inicio = 0
idx_fim = 256

data = dadosParticionados[0][idx_inicio:idx_fim]

previsao(data)
```

---

# Visualização dos Resultados

Os resultados das experiências encontram-se armazenados em ficheiros `.npy`, evitando a necessidade de repetir todo o treino.

---

## Matrizes de Confusão

### Utilizando índices

```python
plot_single_confusion_matrix(
    "data/matriz_de_resultados.npy",
    0,
    2
)
```

---

### Utilizando descrições

```python
visualizar_modelo(
    "data/matriz_de_resultados.npy",
    tipo_dataset="embeddings",
    tipo_split="inter",
    tipo_version="pca"
)
```

ou

```python
visualizar_modelo(
    "data/matriz_de_resultados.npy",
    tipo_dataset="features",
    tipo_split="intra",
    tipo_version="relief"
)
```

---

# Modelos Avaliados

Foram avaliadas **12 configurações**.

| Dataset | Redução |
|----------|----------|
| Features | All |
| Features | PCA |
| Features | ReliefF |
| Embeddings | All |
| Embeddings | PCA |
| Embeddings | ReliefF |

Cada configuração foi testada utilizando:

- Intra-Subject
- Inter-Subject

---

# Significância Estatística

O projeto inclui testes estatísticos para validar se o melhor modelo apresenta diferenças significativas relativamente aos restantes.

Exemplo:

```python
models = [
    "FEATURES",
    "FEATURES PCA",
    "FEATURES RELIEFF",
    "EMBEDDINGS",
    "EMBEDDINGS PCA",
    "EMBEDDINGS RELIEFF"
]

matriz = np.load(
    "data/matriz_de_f1Score.npy",
    allow_pickle=True
)

dados = matriz[1, :]
dados = np.vstack(dados)

idx = melhor_modelo(dados)

print(models[idx])

significance_test(dados, idx)
```

---

# Tecnologias Utilizadas

## Linguagem

- Python 3

## Data Science

- NumPy
- SciPy
- Pandas

## Machine Learning

- Scikit-Learn
  - KNN
  - PCA
  - ReliefF
  - SMOTE

## Deep Learning

- HARNet5
- ssl-wearables

## Visualização

- Matplotlib
- Seaborn

---

# Contexto Académico

Projeto desenvolvido no âmbito da unidade curricular de **Engenharia de Aprendizagem / Extração de Conhecimento a partir de Dados (EA/ECAC)** da **Universidade de Coimbra**.
