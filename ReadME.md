# Como fazer previsao com o melhor modelo ja identificado?
-> Identificar o codigo no main identificado com #deployment;
-> disponibilizar os dados para previsao no formato : numpy array 
de formato 256 linhas and 9 colunas (acc x y z, gyr x y z, mag x y z);
-> corra o codigo com  a secção deployment descomentada(só essa secção).



## Visualizar matrizes de confusão 
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


As matrizes foram salvas em matriz_de_resultados.npy, como soma das matrizes de confusão geradas nas 10 iterações para cada modelo. 
Fez a soma para depois ser dividida pelo numero de iterações para obter a media das métricas.

-------------------------------------------------------------------------

O ficheiro matriz_de_resultados.npy tem o formato:
- matriz[linha][coluna]

Onde:

linha:

0 → intra-subject
1 → inter-subject

coluna:

coluna	    dataset           método <<---------------------------------------------------------------                
                                                                                                      | 
  0	        features             all                                                                  |
  1	        features             PCA                                                                  |
  2	        features            ReliefF                                                               |
  3	        embeddings           all                                                                  |
  4	        embeddings           PCA                                                                  |
  5	        embeddings          ReliefF                                                               |
                                                                                                      |
-----------------------------------------------------------------------------                         |
                                                                                                      |
# Como visualizar uma matriz manualmente ?                                                            |
                                                                                                      |
# Setup                                                                                               |
## Se ainda nao tem a matriz_de_resultados.npy                                                        |
                                                                                                    
Adicione esta linha de codigo a seguir a avaliacao de cada modelo:                                    |
                                                                                                      |
matrizConfusao[a][b]= matrizConfusao[a][b] + resultados["metricas_test"]["confusion_matrix"]          |
                                                                                                      |
a = 0, se for split intra-subject e a = 1, se for inter-subject                                       |
                                                                                                      |
o indice b corresponde aos indices especificados na coluna acima -------------------------------------

## Se já tem a matriz_de_resultados.py

Use:

plot_single_confusion_matrix("matriz_de_resultados.npy", linha, coluna)


Exemplo:
➤ plot_single_confusion_matrix("matriz_de_resultados.npy", 0, 2)


Mostra:
intra-subject, features ReliefF


--------------------------------------------------------------------------------

# Como visualizar uma matriz por descrição do modelo?

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



## Visualizar o melhor modelo 
Utilizou-se a mesma abordagem que matriz de confusão, mas neste caso guardou-se a distribuição de f1_score de cada modelo.

## Se ainda nao tem a matriz_de_f1Score.npy                                                        
                                                                                                    
Adicione esta linha de codigo a seguir a avaliacao de cada modelo:                                    
                                                                                                      
matriz_de_resultados[a][b]= np.append(distribution[a][b], resultados["metricas_test"]["confusion_matrix"])         
                                                                                                      
a = 0, se for split intra-subject e a = 1, se for inter-subject                                       
                                                                                                      
o indice b corresponde aos indices especificados na coluna acima


## Se já tem a matriz_de_f1Score.py

# Setup(adicione essas linhas de codigo)
models= ["FEATURES", "FEATURES PCA", "FEATURES RELIEFF", "EMBEDDINGS", "EMBEDDINGS PCA", "EMBEDDINGS RELIEFF"]
    
matriz= np.load("matriz_de_f1Score.npy", allow_pickle=True)

print(matriz.shape)
#Para a estrategia intra-subject
dados= matriz[1,:]
dados= np.vstack(dados) # matriz com 6 linhas q representam os 6 modelos
idx=melhor_modelo(dados)
    
print(f"==melhor modelo==\n {models[idx]} \n")

# Significancia estatística entre o melhor modelo e o resto
significance_test(dados, idx)