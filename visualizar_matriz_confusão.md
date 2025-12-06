Este conjunto de funções permite visualizar qualquer matriz de confusão produzida durante as 10 repetições dos 12 modelos testados no projeto:

- FEATURES
- FEATURES PCA
- FEATURES ReliefF
- EMBEDDINGS
- EMBEDDINGS PCA
- EMBEDDINGS ReliefF

Com splits:
- Intra-Subject
- Inter-Subject


As matrizes foram salvas em matriz_de_resultados.npy.

-------------------------------------------------------------------------

O ficheiro matriz_de_resultados.npy tem o formato:
- matriz[linha][coluna]

Onde:

linha:

0 → intra-subject
1 → inter-subject

coluna:

coluna	    dataset           método

  0	        features             all
  1	        features             PCA
  2	        features            ReliefF
  3	        embeddings           all
  4	        embeddings           PCA
  5	        embeddings          ReliefF

-----------------------------------------------------------------------------

Como visualizar uma matriz manualmente

Use:

plot_single_confusion_matrix("matriz_de_resultados.npy", linha, coluna)


Exemplo:
➤ plot_single_confusion_matrix("matriz_de_resultados.npy", 0, 2)


Mostra:
intra-subject, features ReliefF


--------------------------------------------------------------------------------

Como visualizar uma matriz por descrição do modelo

Use:

visualizar_modelo(
    "matriz_de_resultados.npy",
    tipo_dataset="features" ou "embeddings",
    tipo_split="intra" ou "inter",
    tipo_version="all", "pca" ou "relief"
)


Exemplos:

➤ Ver matriz do modelo Embeddings PCA inter-subject
visualizar_modelo("matriz_de_resultados.npy", "embeddings", "inter", "pca")

➤ Ver matriz do modelo Features ReliefF intra-subject
visualizar_modelo("matriz_de_resultados.npy", "features", "intra", "relief")

----------------------------------------------------------------------------------

O que aparece?

A função irá:

- Carregar a matriz correta
- Mostrar um heatmap 7×7
- Mostrar os valores numéricos no terminal