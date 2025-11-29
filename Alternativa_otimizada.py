import itertools
import numpy as np, matplotlib.pyplot as plt
import csv
from scipy import stats
from scipy.stats import kstest
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from embeddings_extractor import embedding_main
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

MAX_REPS= 42
SWITCH_VALUE=3

Max_Part_num=14
dev_num=5
activity_num=17

ACELERACAO = 0
GIROSCOPIO= 1
MAGNETOMETRO=2

FEATURES= []

############################################################## META 1 #########################################################################

#2
def descarregar_dados():
    array = []
    for i in range(0,Max_Part_num+1):
        caminho = 'sample/part'
        caminho= f"{caminho}{i}/"
        array_aux=[]
        
        for j in range(1,dev_num+1):
            caminho2 = f"{caminho}part{i}dev{j}.csv"
            with open(caminho2,newline='') as csv_file:
                linha = csv.reader(csv_file, delimiter=',')
                for row in linha:
                    array_aux.append(row)
        
        array.append(array_aux)
            
    dados = np.array(array, dtype=object)
    return dados


#3.1
def representacao_grafica():
    modulos_atividades= FEATURES 
    nomes_variaveis = ["Aceleração", "Giroscópio", "Magnetómetro"]
    i=1
    for modulo in modulos_atividades:
        plt.boxplot(modulo)
        plt.title("Boxplot atividade " + nomes_variaveis[i-1])
        i+=1
        plt.tight_layout()
        plt.show()      


#funcao auxiliar    
def calculo_modulo(dados): 
    #ciclo para o cálculo da densidade das 3 caracteristicas - aceleração, giroscópio e magnetómetro
    for j in range(1, 8, 3):
        aux=[]
        for i in range(1,activity_num):
            condition = (dados[:, -1] == str(i))

            modulo = np.sqrt( dados[condition, j].astype(float) **2 +
            dados[condition, j+1].astype(float) **2 +
            dados[condition, j+2].astype(float) **2
            )
            aux.append(modulo)    
        FEATURES.append(aux) 
            
    # Nota: o cálculo do módulo dos vetores foi implementado manualmente (x² + y² + z²)^(1/2).
    # Alternativamente, poderíamos usar a função np.linalg.norm(person[condition, 1:4], axis=1). 
    # No entanto, optou-se pela versão manual por ser ligeiramente mais direta e com desempenho equivalente.   
                
#3.2 Para todos os parametros
def outlier_density(dados):
    #ciclo para o cálculo da densidade das 3 caracteristicas - aceleração, giroscópio e magnetómetro
    for j in range(1, 8, 3):
        with open("Density_file",'a+') as file:
            if j == 1 : file.write("\n--- " + "aceleracao ---\n")
            if j == 4 : file.write("\n--- " + "giroscopio ---\n")
            if j == 7 : file.write("\n--- " + "magnetometro ---\n")

        #calculo de densidade so no Divice ID 2
        for i in range(1,activity_num):
            #condição para obtenção das atividades do pulso direito
            condition = ((dados[:, -1] == str(i)) & (dados[:, 0].astype(int) == 2))

            modulo = np.sqrt( dados[condition, j].astype(float) **2 +
            dados[condition, j+1].astype(float) **2 +
            dados[condition, j+2].astype(float) **2
            )
                
            Q1 = np.percentile(modulo, 25)
            Q3 = np.percentile(modulo, 75)
                
            IQR = Q3 - Q1
            lower_lim = Q1 - 1.5 * IQR
            upper_lim = Q3 + 1.5 * IQR
            nr = len(modulo)
            n0 = len(modulo[ (modulo < lower_lim) | (modulo > upper_lim) ])
            d = (n0/nr)*100
            # -> Fim de calculo
            
            with open("Density_file",'a+') as file:
                file.write("Densidade de outlier do dataset da " + str(i) + " atividade: " + str(d) +"\n")


# 3.3 e 3.4
def z_score_test(modulos, k):
    #plt.figure(figsize=(14,6))
    x_labels = []
    x_data = []
    y_data = []
    colors_all = []

    atividade = 1
    for modulo in modulos:
        modulo = np.array(modulo, dtype=float)

        media = np.mean(modulo)
        desvio_padrao = np.std(modulo)

        if desvio_padrao > 0:
            z_scores = (modulo - media) / desvio_padrao
        else:
            z_scores = np.zeros_like(modulo)
        
        outlier_indices = np.where(np.abs(z_scores) > k)
        
        colors = np.array(['b'] * len(modulo))
        colors[outlier_indices] = 'r'

        # eixo X fixo = número da atividade repetido
        x_vals = np.full(len(modulo), atividade)

        x_data.extend(x_vals)
        y_data.extend(modulo)
        colors_all.extend(colors)
        
        x_labels.append(f"Atv {atividade}")
        atividade += 1

    plt.scatter(x_data, y_data, c=colors_all, s=10)
    plt.xticks(range(1, len(modulos)+1), x_labels, rotation=45)
    plt.title("Z-Score Outliers por Atividade")
    plt.xlabel("Atividades")
    plt.ylabel("Valores")
    plt.grid(axis='x')

def plot_zscore(K):
    sensores = ["Acelerômetro", "Giroscópio", "Magnetômetro"]

    for i in range(len(FEATURES)):
        plt.figure(num=sensores[i], figsize=(15,6))
        z_score_test(FEATURES[i], K)

    plt.show()
 

#3.6 e 3.7
def k_means(N):
    #device deve ser a coluna da coordenada x do tipo de dispositivo
    
    # Para os clusters, faz mais sentido repartir o array em n clusters, em vez de escolher aleatoriamente n centroides. 
    # Escolhendo N centroides random, sou capaz de escolher um centroide outlier e ele agrupar todos os outros outliers.
    # Daí, a distancia ao centroide pode nao ser muito superior ao treshold, nao identificando assim o outlier
        
    colors= []
    dados_transf=[]
    for i in range (1, activity_num):
        
        dados_a_tratar= np.array(FEATURES[:,i-1])
        dados_a_tratar= np.vstack(dados_a_tratar)
        dados_a_tratar= dados_a_tratar.T
        dados_transf.append(dados_a_tratar)
    centroides= []
    pos=0 #posicao do ponterio copia
    dados_transf=np.array(dados_transf, dtype=object)
    dados_transf=np.vstack(dados_transf)
    
    
    media = np.mean(dados_transf, axis=0)
    desvio = np.std(dados_transf, axis=0)
    dados_transf = (dados_transf - media) / desvio #normalização de dados_trans pelo z_score
    
    
    step=int(len(dados_transf)/N)
        
    for j in range(0,N):
            
        if j == N-1 :  cluster= dados_transf[pos:] #Assegurar que nao sao ultrapasados os limites do array  
        else: 
            cluster= dados_transf[pos:pos+step,:]
            pos= pos+step
            
        centroides.append(np.median(cluster, axis=0)) # Calculo de centroides desta forma para garantir a nao escolha de outlier como centroide

    track=1 #monitora o numero de iterações
    flag=1 #houve mudança nos clusters
        
    group= np.zeros(len(dados_transf), dtype=int) # array q define o grupo q cada ponto pertence
    centroides = np.array(centroides)
        
    #calculo de clusters
    while((track <= MAX_REPS) and (flag==1) ):
            
        track+=1 # Mais uma iteracao
        flag=0
            
        dist = np.linalg.norm(dados_transf[:, np.newaxis, :] - centroides[np.newaxis, :, :], axis=2)
        
        for j in range(0,len(dados_transf)):
            recent_group= np.argmin(dist[j]) # o cluster a q pertence
                
            if(group[j] != recent_group):
                group[j] = recent_group
            
        for j in range(0,N):
            aux= dados_transf[group==j]
            new_center= np.median(aux, axis=0)
            if(np.linalg.norm(new_center-centroides[j]) > SWITCH_VALUE ):
                centroides[j] = new_center
                flag=1
        
    resultos=[]
    for j in range(0,N):
        aux= dados_transf[group==j]
            
        if(len(aux)==0) : continue
        resultos.append(aux)
        dist = np.linalg.norm(aux - centroides[j], axis=1)
            
        Q1 = np.percentile(dist, 25)
        Q3 = np.percentile(dist, 75)
            
        IQR = Q3 - Q1
        lower_lim = Q1 - 1.5 * IQR
        upper_lim = Q3 + 1.5 * IQR
        
        dist = np.linalg.norm(aux - centroides[j], axis=1)
        color = np.where((dist > upper_lim) | (dist < lower_lim), 'r', 'b')
        
        colors.extend(color)
        
        
    resultos=np.vstack(resultos)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    limite= int(len(resultos)/2) #diminuir em metade o numero de pontos. Como eles estao distrbuidos aleatoriamente, a probabilidade de aparecer um outlier
    #na primeira metade ou na outra é igual. Entao o plot representada, com dados reduzidos, a imagem esperada
    ax.scatter(resultos[0:limite,0],resultos[0:limite,1],resultos[0:limite,2],c= colors[0:limite])
                
    plt.tight_layout()     
    plt.show()
     
    
#4.1
def sig_est(modulos):
    # Junta os arrays equivalentes (por posição)
    # considerando que a estrutura dos modulos é 3x16
    norm=1 # 1 se forem todos normais. 0 se pelo menos 1 for nao normal
    aux= ["Acelerometro", "Grioscopio", "Magnetometro"]
    a=0
    for sensor in modulos:
        print(aux[a]+"\n")
        
        a+=1
        #Analise de normalidade para cada sensor
        for i in range(1, activity_num):
            ks_stat, p_norm = kstest(sensor[i], 'norm', args=(np.mean(sensor[i]), np.std(sensor[i])))
            if p_norm <=0.05 : 
                norm=0
                break
            
        if norm == 0:
            # Teste de Kruskal-Wallis
            h_stat, p_value = stats.kruskal(*sensor)
            metodo = "Kruskal–Wallis (não paramétrico)"
        else:
            f_stat, p_value = stats.f_oneway(*sensor)
            metodo = "ANOVA (paramétrico)"
            
        print(f"→ Teste usado: {metodo}")
        if p_value < 0.05:
            print("Rejeitamos H₀: existem diferenças significativas entre as atividades.\n\n")
        else:
            print("Não rejeitamos H₀: as médias não diferem significativamente.\n\n")
            

#ChatGPT

# 1. Mean
def feature_mean(mod1, mod2, mod3):
    return [np.mean(mod1), np.mean(mod2), np.mean(mod3)]

# 2. Median
def feature_median(mod1, mod2, mod3):
    return [np.median(mod1), np.median(mod2), np.median(mod3)]

# 3. Standard Deviation
def feature_std(mod1, mod2, mod3):
    return [np.std(mod1), np.std(mod2), np.std(mod3)]

# 4. Variance
def feature_variance(mod1, mod2, mod3):
    return [np.var(mod1), np.var(mod2), np.var(mod3)]

# 5. Root Mean Square (RMS)
def feature_rms(mod1, mod2, mod3):
    return [np.sqrt(np.mean(mod1**2)),
            np.sqrt(np.mean(mod2**2)),
            np.sqrt(np.mean(mod3**2))]

# 6. Averaged Derivatives (mean of first derivative)
def feature_avg_derivative(mod1, mod2, mod3):
    return [np.mean(np.diff(mod1)),
            np.mean(np.diff(mod2)),
            np.mean(np.diff(mod3))]

# 7. Skewness
def feature_skewness(mod1, mod2, mod3):
    return [stats.skew(mod1),
            stats.skew(mod2),
            stats.skew(mod3)]

# 8. Kurtosis
def feature_kurtosis(mod1, mod2, mod3):
    return [stats.kurtosis(mod1),
            stats.kurtosis(mod2),
            stats.kurtosis(mod3)]

# 9. Interquartile Range (IQR)
def feature_iqr(mod1, mod2, mod3):
    return [np.percentile(mod1, 75) - np.percentile(mod1, 25),
            np.percentile(mod2, 75) - np.percentile(mod2, 25),
            np.percentile(mod3, 75) - np.percentile(mod3, 25)]

# 10. Zero Crossing Rate (ZCR)
def feature_zero_crossing_rate(mod1, mod2, mod3):
    zc1 = ((mod1[:-1] * mod1[1:]) < 0).sum() / len(mod1)
    zc2 = ((mod2[:-1] * mod2[1:]) < 0).sum() / len(mod2)
    zc3 = ((mod3[:-1] * mod3[1:]) < 0).sum() / len(mod3)
    return [zc1, zc2, zc3]

# 11. Mean Crossing Rate (MCR)
def feature_mean_crossing_rate(mod1, mod2, mod3):
    mean1 = np.mean(mod1)
    mean2 = np.mean(mod2)
    mean3 = np.mean(mod3)
    mcr1 = ((mod1[:-1] - mean1) * (mod1[1:] - mean1) < 0).sum() / len(mod1)
    mcr2 = ((mod2[:-1] - mean2) * (mod2[1:] - mean2) < 0).sum() / len(mod2)
    mcr3 = ((mod3[:-1] - mean3) * (mod3[1:] - mean3) < 0).sum() / len(mod3)
    return [mcr1, mcr2, mcr3]

# 12. Spectral Entropy
def feature_spectral_entropy(mod1, mod2, mod3):
    def entropy(window):
        fft_vals = np.fft.fft(window)
        psd = np.abs(fft_vals)**2
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        return -np.sum(psd_norm * np.log2(psd_norm))
    
    return [entropy(mod1), entropy(mod2), entropy(mod3)]

#ChatGPT end


#4.2
#4.2 Funcao explicada no relatório 
def feature_extraction(dados):
    resultado =[]
    
    dados_por_pessoa=dados
        
    count=1
    for person in dados_por_pessoa:
        
        person= np.array(person, dtype=float)
        TIMESTAMP_COL = 10
        MIN_SEGMENT_SIZE = 20
        win_size = 5000
        start_time = person[0,TIMESTAMP_COL]
        end_time = start_time + win_size

    
       
        while end_time < person[-1,TIMESTAMP_COL]:
            mask = (person[:,TIMESTAMP_COL] >= start_time) & (person[:,TIMESTAMP_COL] < end_time)

            if np.sum(mask) <= MIN_SEGMENT_SIZE or np.any(person[mask, -1] != person[mask, -1][0]):
                start_time = end_time - win_size/2
                end_time = start_time + win_size
                continue
                
            acc = person[mask, 1:4]
            gyro = person[mask, 4:7]
            mag = person[mask, 7:10]

            start_time = end_time - win_size/2
            end_time = start_time + win_size
            
            # Calcular módulo de cada vetor
            mod_acc = np.linalg.norm(acc, axis=1)
            mod_gyro = np.linalg.norm(gyro, axis=1)
            mod_mag = np.linalg.norm(mag, axis=1)
            
            
            
            #Chamar as funcoes de calculos
            #Na ultima posicao de cada linha está o tipo de atividade
            # 1. Mean
            mean_feat = feature_mean(mod_acc, mod_gyro, mod_mag)

            # 2. Median
            median_feat = feature_median(mod_acc, mod_gyro, mod_mag)

            # 3. Standard Deviation
            std_feat = feature_std(mod_acc, mod_gyro, mod_mag)

            # 4. Variance
            var_feat = feature_variance(mod_acc, mod_gyro, mod_mag)

            # 5. Root Mean Square (RMS)
            rms_feat = feature_rms(mod_acc, mod_gyro, mod_mag)

            # 6. Averaged Derivatives
            avg_deriv_feat = feature_avg_derivative(mod_acc, mod_gyro, mod_mag)

            # 7. Skewness
            skew_feat = feature_skewness(mod_acc, mod_gyro, mod_mag)

            # 8. Kurtosis
            kurt_feat = feature_kurtosis(mod_acc, mod_gyro, mod_mag)

            # 9. Interquartile Range (IQR)
            iqr_feat = feature_iqr(mod_acc, mod_gyro, mod_mag)

            # 10. Zero Crossing Rate (ZCR)
            zcr_feat = feature_zero_crossing_rate(mod_acc, mod_gyro, mod_mag)

            # 11. Mean Crossing Rate (MCR)
            mcr_feat = feature_mean_crossing_rate(mod_acc, mod_gyro, mod_mag)

            # 12. Spectral Entropy
            entropy_feat = feature_spectral_entropy(mod_acc, mod_gyro, mod_mag)
            
            segmento = (
                mean_feat + 
                median_feat + 
                std_feat + 
                var_feat + 
                rms_feat + 
                avg_deriv_feat + 
                skew_feat + 
                kurt_feat + 
                iqr_feat + 
                zcr_feat + 
                mcr_feat + 
                entropy_feat +
                [ person[mask, -1][0]] +  #1ª atvidade, como todos têm a mesma atividade... <- confirmado la em cima
                [count]
            )
            resultado.append(segmento)
        
        count+=1
 
    return np.array(resultado, dtype=object)

#4.3 e 4.4
def aplicar_pca(estrutura, explainability):
    features = np.array(estrutura[:, :-2], dtype=float) # elimina colunas de classes (últimas colunas)
    
    # Normalização (z-score)
    media = np.mean(features, axis=0)
    desvio_padrao = np.std(features, axis=0)
    z_scores = (features - media) / (desvio_padrao + 1e-8) # 1e-8 para evitar divisão por 0
    
    # PCA
    # Funcionamento básico do PCA:
    # Cria novos eixos (componentes principais) no espaço das features.
    # Cada novo eixo é uma combinação linear das features originais que maximiza a variância dos dados ao longo dele.
    # O primeiro vetor principal (PC1) é o eixo que captura a maior parte da variância dos dados.
    # O segundo vetor principal (PC2) é perpendicular ao primeiro e captura a maior parte da variância restante, e assim por diante.
    
    pca = PCA()
    pca.fit_transform(z_scores)
    
    var_exp = pca.explained_variance_ratio_ # Indica quanto cada componente explica da variância total
    if(explainability != None): return var_exp #devolve o resultado para a operação de split
    
    cum_var_exp = np.cumsum(var_exp)

    """
    # ---------- PLOTS ----------
    plt.figure(figsize=(12,5))

    # Gráfico 1: Variância explicada por componente
    plt.subplot(1,2,1)
    plt.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7, align='center')
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada')
    plt.title('Variância Explicada por Componente')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # Gráfico 2: Variância acumulada
    plt.subplot(1,2,2)
    plt.plot(range(1, len(cum_var_exp)+1), cum_var_exp, marker='o')
    plt.axhline(y=0.75, color='r', linestyle='--', label='75% da Variância')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Acumulada')
    plt.title('Variância Acumulada das Componentes')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()
    """
    
    # ---------- RESULTADOS ----------
    print("Variância explicada por componente:", np.round(var_exp, 4))
    print("\nVariância acumulada:", np.round(cum_var_exp, 4))

    # índice da primeira vez que a variância acumulada >= 0.75 by default
    num_dim=0
    #alterado para a segunda meta
    
    num_dim = np.argmax(cum_var_exp >= 0.75) + 1
    print("Número de dimensões necessárias para 75% da variância:", num_dim)
        
        
    # Projetando os dados originais nas 'num_dim' primeiras componentes
    pca_reduced = PCA(n_components=num_dim)
    X_reduced = pca_reduced.fit_transform(z_scores)

    # Exemplo: pegar features reduzidas da primeira amostra
    sample_index = 0
    #print("Features reduzidas da primeira amostra:", X_reduced[sample_index])
    
    labels = np.array(estrutura[:, -2:], dtype=float)
    X_reduced = np.hstack([X_reduced, labels])
    return X_reduced
    # Essa abordagem é vantajosa pois reduz a dimensão dos dados, facilitando análise e visualização.
    # Remove redundância entre features correlacionadas, e ainda mantém a maior parte da informação (variância).
    # Tem limitações, pois como o PCA assume linearidade padrões não lineares podem ser perdidos.
    # Há perda de informações, mesmo com 75% da variância os 25% ainda se perdem (são descartados).
    

#4.5 e 4.6
def fisher_score(estrutura):
    # Extrair features e classes
    features = np.array(estrutura[:, :-1], dtype=float)
    classes = np.array(estrutura[:, -1], dtype=int)
    
    N, M = features.shape

    classes_unicas = np.unique(classes)
    fisher_scores = np.zeros(M)
    media_geral = np.mean(features, axis=0)

    for c in classes_unicas:
        features_c = features[classes == c]
        num_amostras_c = N
        if num_amostras_c == 0:
            continue
        
        media_c = np.mean(features_c, axis=0)
        variancia_c = np.var(features_c, axis=0)
        
        fisher_scores += num_amostras_c * ((media_c - media_geral) ** 2 / (variancia_c + 1e-8))         

    return fisher_scores 

    # Essa abordagem é vantajosa pois: é simples, sua implementação é direta (baseada em estatísticas básicas - média e variânicia);
    # sua interpretação é clara já que cada feature é avaliada isoladamente das outras; funciona muito bem quando classes são separáveis por
    # padrões lineares (como no caso, pela diferença de média); é extremamente eficiente - roda em tempo linear O(N.M).
    # Suas limitações podem ser justamente as suas vantagens, pelo fato de apenas assumir padrões lineares ele acaba ignorando as relações não lineares entre as features;
    # Como ele avalia cada feature isoladamente pode acabar por ignorar correlações entre features.


def reliefF(estrutura, n_vizinhos=10):
    features = np.array(estrutura[:, :-1], dtype=float)
    classes = np.array(estrutura[:, -1], dtype=int)
    
    N, M = features.shape
    pesos = np.zeros(M)
    
    nn = NearestNeighbors(n_neighbors=n_vizinhos + 1)
    nn.fit(features)
    
    for i in range(N):
        amostra = features[i]
        classe = classes[i]
        vizinhos = nn.kneighbors([amostra])[1] # [1] pega apenas os indíces dos vizinhos correspondentes
        vizinhos = vizinhos[0][1:]  # Excluir o próprio 

        mesma_classe = [] 
        for j in vizinhos: 
            if classes[j] == classe: 
                mesma_classe.append(j) 
                
        dif_classe = [] 
        for j in vizinhos: 
            if classes[j] != classe: 
                dif_classe.append(j)
                
        if mesma_classe:
            pesos -= np.mean(np.abs(amostra - features[mesma_classe]), axis=0)
        if dif_classe:
            pesos += np.mean(np.abs(amostra - features[dif_classe]), axis=0)
    
    return pesos

    # Vantagens: captura relações não lineares pois considera o espaço multidimensional e as distâncias entre as amostras;
    # avalia importância contextual (não paneas individual), pois as distâncias são calculadas usando todas as features;
    # Não aasume nenhuma forma de distribuição (ao contrário de métodos baseados em média/variância).
    # Limitações: custo computacional pode ser alti, ja que é preciso calcular os vizinhos mais próximos para cada amostra (O(N^2) no pior caso);
    # sensível a desbalanceamento, pois classes com poucas amostras podem ter poucos vizinhos da mesma classe ("hits"). 

def dez_melhores_features(pesos, f_scores):
    top10_fisher_idx = np.argsort(f_scores)[-10:][::-1]
    top10_relief_idx = np.argsort(pesos)[-10:][::-1]

    print("Top10 Fisher indices:", top10_fisher_idx)
    print("Top10 ReliefF indices:", top10_relief_idx)
    print("Intersecção:", np.intersect1d(top10_fisher_idx, top10_relief_idx))


def quinze_melhores_features(pesos):
    top15_relief_idx = np.argsort(pesos)[-15:][::-1]
    return top15_relief_idx


############################################################## META 2 #########################################################################
def verificar_balanceamento(dados):
    # Contar as ocorrências
    atividades, contagens = np.unique(dados[:, -2], return_counts=True)
    
    # Plot do gráfico de barras
    plt.bar(atividades, contagens, color='skyblue')
    plt.xlabel('Atividade')
    plt.ylabel('Contagem')
    plt.title('Contagem de Atividades (1 a 7)')
    plt.show()

    return atividades, contagens


def smote(dados, num_amostras_sinteticas, atividade, num_vizinhos):
    atividades = dados[:, -2]
    
    # Filtrar atividade alvo
    dados_atividade = dados[atividades == atividade]
    num_amostras = len(dados_atividade)
    
    # Ajustar os vizinhos
    vizinhos = NearestNeighbors(n_neighbors = num_vizinhos).fit(dados_atividade)
    amostras_sinteticas = []
    
    for __ in range(num_amostras_sinteticas):
        # Escolhe aleatóriamente um ponto
        idx = np.random.randint(0, num_amostras)
        x_i = dados_atividade[idx]
        
        # Encontra vizinhos e escolhe um aleatoriamente
        escolhido = vizinhos.kneighbors([x_i], return_distance = False)
        indice_escolhido = np.random.choice(escolhido[0][1:]) # Evitar o próprio ponto
        x_escolhido = dados_atividade[indice_escolhido]
        
        # Cria um novo ponto (interpolação)
        diferenca = x_escolhido - x_i
        gap = np.random.rand() # valor aleatório entre 0 e 1
        x_novo = x_i + gap * diferenca
        
        amostras_sinteticas.append(x_novo)
    
    amostras_sinteticas = np.array(amostras_sinteticas)
    y_novo = np.full(num_amostras_sinteticas, atividade)

    # Junta dados originais + sintéticos
    dados_atualizado = np.vstack([dados, amostras_sinteticas])
    atividades_atualizada = np.concatenate([atividades, y_novo])

    return dados_atualizado, atividades_atualizada, amostras_sinteticas

# Plotar resultado 1.3
def plot_data_augmentation(dados_participante, amostras_sinteticas):
    plt.figure(figsize=(8, 6))

    # (1) Plot de todas as atividades originais (exceto sínteticos)
    atividades_originais = dados_participante[:, -2]

    for ativ in np.unique(atividades_originais):
        mask = atividades_originais == ativ
        plt.scatter(
            dados_participante[mask, 0],   # primeira feature
            dados_participante[mask, 1],   # segunda feature
            label=f"Atividade {int(ativ)}",
            alpha=1,
            s=15
        )

    # (2) Plot das amostras sintéticas
    plt.scatter(
        amostras_sinteticas[:, 0],
        amostras_sinteticas[:, 1],
        color='black',
        marker='x',
        s=80,
        linewidths=2,
        label='Sintéticas (SMOTE)'
    )

    plt.xlabel("Feature 1 (Média do Acelerômetro)")
    plt.ylabel("Feature 2 (Média do Giroscópio)")
    plt.title("SMOTE - 3 amostras sintéticas\nAtividade 4 do Participante 3")
    plt.legend()
    plt.grid(True)
    plt.show()


#2.1 
def retrive_embedding():
    return embedding_main()


def split_para_cada_sujeito(dataset, randState):
    idx_train = []
    idx_val = []
    idx_test = []
    
    features = dataset[:, :-2]
    atividades = dataset[:, -2]
    participantes = dataset[:, -1]
    
    id_unicos = np.unique(participantes)
    
    for i in id_unicos:
        # índices desse participante
        idx = np.where(participantes == i)[0]

        # 60% treino, 40% resto
        idx_treino, idx_resto = train_test_split(
            idx, test_size = 0.4, random_state= randState, shuffle=True
        )

        # 20/20 split do restante (validação e teste)
        idx_validacao, idx_teste = train_test_split(
            idx_resto, test_size = 0.5, random_state= randState, shuffle=True
        )

        idx_train.extend(idx_treino)
        idx_val.extend(idx_validacao)
        idx_test.extend(idx_teste)

    return (
        features[idx_train], atividades[idx_train],
        features[idx_val], atividades[idx_val],
        features[idx_test], atividades[idx_test]
    )
    
  
def split_entre_sujeitos(dataset, randState):
    idx_train = []
    idx_val = []
    idx_test = []
    
    features = dataset[:, :-2]
    atividades = dataset[:, -2]
    participantes = dataset[:, -1]
    
    id_unicos = np.unique(participantes)
    
    """idx_treino = id_unicos[:9]
    idx_validacao   = id_unicos[9:12]
    idx_teste  = id_unicos[12:15]"""
    # 60%(9 participantes) treino, 40% resto
    idx_treino, idx_resto = train_test_split(
        id_unicos, test_size = 0.4, random_state= randState, shuffle=True
    )

    # 3/3 split do restante (validação e teste)
    idx_validacao, idx_teste = train_test_split(
        idx_resto, test_size = 0.5, random_state= randState, shuffle=True
    )


    idx_train = np.isin(participantes, idx_treino)
    idx_val   = np.isin(participantes, idx_validacao)
    idx_test  = np.isin(participantes, idx_teste)
    
    return (
        features[idx_train], atividades[idx_train],
        features[idx_val], atividades[idx_val],
        features[idx_test], atividades[idx_test]
    )
    
    
def preparar_datasets(X_train, X_val, X_test, y_train):
    # X - features, y - atividades
    
    # 1 - ALL FEATURES
    all_treino = X_train.copy()
    all_validacao   = X_val.copy()
    all_teste  = X_test.copy()
    
    
    # 2 - PCA (90% variance)
    # Normalização - usamos o .fit_transform apenas no treino pois o scaler e PCA calculam a média, desvio padrão e componentes só no treino.
    # Validação e teste nunca são usados para ajustar nada, apenas são transformados usando os parâmetros aprendidos no treino.
    scaler_pca = StandardScaler()
    X_treino_norm = scaler_pca.fit_transform(X_train)
    X_validacao_norm   = scaler_pca.transform(X_val)
    X_teste_norm  = scaler_pca.transform(X_test)
    
    # PCA (90%)
    pca = PCA(n_components=0.90)
    X_treino_pca = pca.fit_transform(X_treino_norm)
    X_validacao_pca   = pca.transform(X_validacao_norm)
    X_teste_pca  = pca.transform(X_teste_norm)
    
    
    # 3 - ReliefF (top 15 features) 
    # Normalização novamente (não compartilhar scaler com PCA)
    scaler_relief = StandardScaler()
    X_treino_norm_r = scaler_relief.fit_transform(X_train)
    X_validacao_norm_r   = scaler_relief.transform(X_val)
    X_teste_norm_r  = scaler_relief.transform(X_test)
    
    # Concatenamos as colunas das atividades no final do array pois a função reliefF que desenvolvemos utiliza essa formatação
    X_treino_norm_r_conc = np.concatenate((X_treino_norm_r, y_train.reshape(-1, 1)), axis = 1)
    
    # ReliefF
    pesos = reliefF(X_treino_norm_r_conc, 5)
    
    # índices das 15 melhores features
    top_features = quinze_melhores_features(pesos)
    
    # aplicar seleção
    X_treino_relief = X_treino_norm_r[:, top_features]
    X_validacao_relief   = X_validacao_norm_r[:, top_features]
    X_teste_relief  = X_teste_norm_r[:, top_features]
    
    
    return {
        "all":     (all_treino, all_validacao, all_teste),
        "pca":     (X_treino_pca, X_validacao_pca, X_teste_pca),
        "relief":  (X_treino_relief, X_validacao_relief, X_teste_relief),
        "relief_idx": top_features   # para registrar as colunas escolhidas
    }
         

class K_Nearest_Neighbors:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict_one(self, x):
        distancias = np.linalg.norm(self.X_train - x, axis=1) # calcula as distâncias entre x e todos os pontos do treino
        k_idx = np.argsort(distancias)[:self.k] # ordena a distância do menor para o maior e pega apenas as k primeiras distâncias
        k_labels = self.y_train[k_idx]
        
        # Contar classe que aparece mais vezes
        label_count = {}
        for label in k_labels:
            if label not in label_count:
                label_count[label] = 1
            else:
                label_count[label] += 1

        # Encontrar classe mais frequente
        mais_frequente = None
        maior_contagem = -1

        for label, count in label_count.items():
            if count > maior_contagem:
                maior_contagem = count
                mais_frequente = label
        
        return mais_frequente
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        previsoes = np.empty(len(X), dtype=self.y_train.dtype)

        for i in range(len(X)):
            previsoes[i] = self.predict_one(X[i])

        return previsoes
    

def avaliar_modelo(y_true, y_pred):
    resultados = {}

    resultados["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    resultados["accuracy"] = accuracy_score(y_true, y_pred)
    resultados["precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    resultados["recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    resultados["f1_score"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return resultados


def escolher_melhor_k(X_train, y_train, X_val, y_val):
    melhor_k = None
    melhor_f1_score = -1
    
    k_testes = [1,3,5,7,9,11,13,15,17,19,21]
    
    for k in k_testes:
        knn = K_Nearest_Neighbors(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        metricas = avaliar_modelo(y_val, y_pred)
        f1_score = metricas["f1_score"]
        
        if f1_score > melhor_f1_score:
            melhor_f1_score = f1_score
            melhor_k = k
    
    return melhor_k


def avaliar_teste(X_train, y_train, X_test, y_test, k):
    knn = K_Nearest_Neighbors(k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metricas = avaliar_modelo(y_test, y_pred)
    return metricas


def avaliar_dataset(modelo, y_train, y_val, y_test):
    resultado_final = {}
    
    #for nome, (X_train, X_val, X_test) in splits.items():
        #if nome == "relief_idx":
            #continue
        
    X_train = modelo[0]
    X_val = modelo[1]
    X_test = modelo[2]
        
    melhor_k = escolher_melhor_k(X_train, y_train, X_val, y_val)

    metricas_teste = avaliar_teste(
        np.vstack((X_train, X_val)),
        np.concatenate((y_train, y_val)),
        X_test,
        y_test,
        melhor_k
    )

    resultado_final = {
        "melhor_k": melhor_k,
        "metricas_test": metricas_teste
    }
    
    return resultado_final



#pode ser da estrategia intra-subject(0) or inter-subjet(1). 
#Se for intra-subject o valor será adicionado na primeira linha e se for a inter-subject será adicionada na segunda linha
def dados_De_Analise(estrategia): 
   
    for i in range(0,7):
        # para pessoa 1  
        indx= np.array(7*[True])
        indx[i]= False
        fp = np.sum(resultados["metricas_test"]["confusion_matrix"][indx,i])
        total=  np.sum(resultados["metricas_test"]["confusion_matrix"][indx,:])
        
        matriz_de_fp[estrategia,i] = np.append(matriz_de_fp[estrategia,i] , fp/total)
        
    
    
    

def melhor_modelo(distributions):#usa matriz_de_resultados
    # separar colunas
    c1, c2, c3, c4, c5, c6 = distributions

    # -----------------------
    # 2) Teste de Friedman
    # -----------------------
    stat, p = stats.friedmanchisquare( c1, c2, c3, c4, c5, c6)

    print("=== Teste de Friedman ===")
    print("Estatística:", stat)
    print("p-valor:", p)

    # -----------------------
    # 3) Calcular postos médios -> permite calcular o melhor modelo
    # -----------------------
    # rank por linha
    # Criar matriz linha=experimento, coluna=modelo
    matriz = np.column_stack(distributions)

    # rank por linha
    ranks = np.array([stats.rankdata(row) for row in matriz])

    mean_ranks = ranks.mean(axis=0)

    #=== Postos Médios ===
    postos_med= []
    for i, r in enumerate(mean_ranks, start=1):
        postos_med.append(r)
        
    return postos_med.index(min(postos_med))


def significance_test(distributions, indx_melhor_modelo):
   
    # -----------------------
    # 3) Pós-hoc Wilcoxon
    # -----------------------

    print("=== Wilcoxon Pareado (sem correção) ===")
    for j in range(0,len(distributions)):
        if j == indx_melhor_modelo:
            continue  # não compara consigo mesmo
        stat, p = stats.wilcoxon(distributions[indx_melhor_modelo], distributions[j])
        if(p < 0.05 ):
            print(f"{models[indx_melhor_modelo]} é significativamente melhor que {models[j]}: ")
        else: 
            print(f"{models[indx_melhor_modelo]} não é significativamente melhor que {models[j]}: ")
    
    
###################################################################################################################################################

if __name__ == "__main__":
    """
    ############################################################## META 1 #########################################################################
    #2
    dadosParticinado = descarregar_dados()
    
    #3.1
    dados = np.vstack(dadosParticinado)
    
    
    calculo_modulo(dados)
    FEATURES = np.array(FEATURES, dtype=object)
    
    #representacao_grafica()
    
    #3.2 (Densidade outliers)
    #outlier_density(dados)
      
    #3.3 e 3.4 (Z-score)
    K=3
    #plot_zscore(K)
    
    #3.6 e 3.7 (K-means)
    N=5
    #k_means(N)
    
    #4.1 (Significância estatística)
    #sig_est(FEATURES)
    
    #4.2 (Extrair as features) considerando somente dados de divice 1
    dadosParticinadoUpdated = []
    for sub in dadosParticinado:
        arr = np.array(sub)
        dadosParticinadoUpdated.append(arr[(arr[:, 0].astype(int) == 1) & (arr[:, -1].astype(int) <= 7)]) # Exclui as atividades superiores a 7
    dadosParticinadoUpdated = np.array(dadosParticinadoUpdated, dtype=object)

    vetor_features = feature_extraction(dadosParticinadoUpdated)
    vetor_features = np.vstack(vetor_features)
    
    np.save("cache_vetor_features.npy", vetor_features) #Guardar em cache
    
    print("FEATURES GUARDADAS")

    #4.3 e 4.4 (PCA)
    PCA_data=aplicar_pca(vetor_features)
    
    print("\n")
    
    #4.5 e 4.6 (fisher_score e reliefF)
    f_scores = fisher_score(vetor_features)
    print(f_scores[0])
    print("Ficher Scores: ", f_scores)
    
    print("\n")
    
    pesos = reliefF(vetor_features, 5)
    print("Pesos ReliefF: ", pesos)
    
    print("\n")
    
    dez_melhores_features(pesos,f_scores)

    vetor_features = np.load("cache_vetor_features.npy", allow_pickle=True)
    ############################################################## META 2 #########################################################################
    """
    
    vetor_features = np.load("cache_vetor_features.npy", allow_pickle=True)
    vetor_features = vetor_features.astype(float)
    
    # 1.1 Balanço entre quantidade de amostras das atividades
    #atividades, contagens = verificar_balanceamento(vetor_features)
    
    # 1.2 Implementação do método SMOTE
    # Esse bloco realiza o balancemanto da quantidade dos dados, ele deixa as atividades 2 a 7 com a mesma quantidade de dados da atividade 1
    #dados_filtrados = vetor_features
    #for i in range(2, 8):
        #num_amostras_sinteticas = contagens[0] - contagens[i-1] # Insere dados na atividade desejada até que fique com a mesma quantidade de amostras da atividade 1 (a que possue mais amostras)
        #dados_filtrados, atividades_atualizada, amostras_sinteticas = smote(dados_filtrados, num_amostras_sinteticas, atividade = i, num_vizinhos = 5)
    #atividades, contagens = verificar_balanceamento(dados_filtrados)
    
    # 1.3 Gerar e visualizar 3 amostras da atividade 4 para o particapante 3
    num_amostras = 3
    atividade = 4
    dados_participante_3 = vetor_features[vetor_features[:, -1] == 3]
    #dados_filtrados, atividades_atualizada, amostras_sinteticas = smote(dados_participante_3, num_amostras, atividade, num_vizinhos = 5)
    
    #plot_data_augmentation(dados_participante_3, amostras_sinteticas)
    
    
    #2.1
    embeddings = retrive_embedding()
    
    lista_de_arrays = [
    [np.array([]) for _ in range(6)]  # Linha 1
    for _ in range(2)                 # Repete para 2 linhas
    ]

    # Converte a lista de listas para um array NumPy de objetos
    matriz_de_resultados = np.array(lista_de_arrays, dtype=object)
    
    
    lista_de_arrays = [
    [np.array([]) for _ in range(7)]  # Linha 1
    for _ in range(4)                 # Repete para 4 linhas (1-> features, 2-> embeddings, 3-> feature selection, 4-> embeddings selection)
    ]
    
    matriz_de_fp = np.array(lista_de_arrays, dtype=object) #matriz que irá guardar informação de falsos positivos
    
    
    for i in range(1,11): #10  repeticoes é um numero razoavel para obter uma distribuicao de cada modelo
    
    
        #3.1
        # Split do primeiro dataset (features)
        Xf_train, yf_train, Xf_val, yf_val, Xf_test, yf_test = split_para_cada_sujeito(vetor_features,i)
        
        # Split do segundo dataset (embeddings)
        Xe_train, ye_train, Xe_val, ye_val, Xe_test, ye_test = split_para_cada_sujeito(embeddings,i)
        
        #3.2.
        # Split do primeiro dataset (features)  
        Xf_train2, yf_train2, Xf_val2, yf_val2, Xf_test2, yf_test2 = split_entre_sujeitos(vetor_features,i)
        
        # Split do segundo dataset (embeddings)
        Xe_train2, ye_train2, Xe_val2, ye_val2, Xe_test2, ye_test2 = split_entre_sujeitos(embeddings,i)
        
        # 3.4.
        # FEATURES DATASET
        feat_splits = preparar_datasets(
            Xf_train, Xf_val, Xf_test, yf_train
        )
        
        feat_splits2 = preparar_datasets(
            Xf_train2, Xf_val2, Xf_test2, yf_train2
        )

        # EMBEDDINGS DATASET
        emb_splits = preparar_datasets(
            Xe_train, Xe_val, Xe_test, ye_train
        )
        
        emb_splits2 = preparar_datasets(
            Xe_train2, Xe_val2, Xe_test2, ye_train2
        )

        
       
        # 5.1 e 5.2
        # Realizaremos 12 avaliações para 12 modelos diferentes, pois temos 2 datasets (FEATURES e EMBEDDINGS), 2 splits (para cada sujeito, entre sujeitos) e 3 versões (todos os dados originais, dados com PCA aplicado, dados com ReliefF aplicado)
        
        # 1 - Teste do dataset de FEATURES com split feito em CADA pessoa de TODOS os dados (originais)
        print("\n")
        resultados = avaliar_dataset(feat_splits["all"], yf_train, yf_val, yf_test)
        matriz_de_resultados[0,0] = np.append(matriz_de_resultados[0,0], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(0) 
        
        #print(resultados)
        print("\n")
        
        # 2 - Teste do dataset de FEATURES com split feito em CADA pessoa de dados com PCA aplicado
        print("\n")
        resultados = avaliar_dataset(feat_splits["pca"], yf_train, yf_val, yf_test)
        matriz_de_resultados[0,1] = np.append(matriz_de_resultados[0,1], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(2) 
        #print(resultados)
        print("\n")
        
        # 3 - Teste do dataset de FEATURES com split feito em CADA pessoa de dados com ReliefF aplicado
        print("\n")
        resultados = avaliar_dataset(feat_splits["relief"], yf_train, yf_val, yf_test) 
        matriz_de_resultados[0,2] = np.append(matriz_de_resultados[0,2], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(2) 
        #print(resultados)
        print("\n")
        
        # 4 - Teste do dataset de FEATURES com split feito ENTRE pessoas de TODOS os dados (originais)
        print("\n")
        resultados = avaliar_dataset(feat_splits2["all"], yf_train2, yf_val2, yf_test2) 
        matriz_de_resultados[1,0] = np.append(matriz_de_resultados[1,0], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(0) 
        #print(resultados)
        print("\n")
        
        # 5 - Teste do dataset de FEATURES com split feito ENTRE pessoas de dados com PCA aplicado
        print("\n")
        resultados = avaliar_dataset(feat_splits2["pca"], yf_train2, yf_val2, yf_test2) 
        matriz_de_resultados[1,1] = np.append(matriz_de_resultados[1,1], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(2)
        #print(resultados)
        print("\n")
        
        # 6 - Teste do dataset de FEATURES com split feito ENTRE pessoas de dados com ReliefF aplicado
        print("\n")
        resultados = avaliar_dataset(feat_splits2["relief"], yf_train2, yf_val2, yf_test2) 
        matriz_de_resultados[1,2] = np.append(matriz_de_resultados[1,2], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(2)
        #print(resultados)
        print("\n")
        
        # 7 - Teste do dataset de EMBEDDINGS com split feito em CADA pessoa de TODOS os dados (originais)
        print("\n")
        resultados = avaliar_dataset(emb_splits["all"], ye_train, ye_val, ye_test)
        matriz_de_resultados[0,3] = np.append(matriz_de_resultados[0,3], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(1)
        #print(resultados)
        print("\n")
        
        # 8 - Teste do dataset de EMBEDDINGS com split feito em CADA pessoa de dados com PCA aplicado
        print("\n")
        resultados = avaliar_dataset(emb_splits["pca"], ye_train, ye_val, ye_test)
        matriz_de_resultados[0,4] = np.append(matriz_de_resultados[0,4], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(3)
        #print(resultados)
        print("\n")
        
        # 9 - Teste do dataset de EMBEDDINGS com split feito em CADA pessoa de dados com ReliefF aplicado
        print("\n")
        resultados = avaliar_dataset(emb_splits["relief"], ye_train, ye_val, ye_test) 
        matriz_de_resultados[0,5] = np.append(matriz_de_resultados[0,5], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(3)
        #print(resultados)
        print("\n")
        
        # 10 - Teste do dataset de EMBEDDINGS com split feito ENTRE pessoas de TODOS os dados (originais)
        print("\n")
        resultados = avaliar_dataset(emb_splits2["all"], ye_train2, ye_val2, ye_test2)
        matriz_de_resultados[1,3] = np.append(matriz_de_resultados[1,3], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(1)
        #print(resultados)
        print("\n")
        
        # 11 - Teste do dataset de EMBEDDINGS com split feito ENTRE pessoas de dados com PCA aplicado
        print("\n")
        resultados = avaliar_dataset(emb_splits2["pca"], ye_train2, ye_val2, ye_test2)
        matriz_de_resultados[1,4] = np.append(matriz_de_resultados[1,4], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(3)
        #print(resultados)
        print("\n")
        
        # 12 - Teste do dataset de EMBEDDINGS com split feito ENTRE pessoas de dados com ReliefF aplicado
        print("\n")
        resultados = avaliar_dataset(emb_splits2["relief"], ye_train2, ye_val2, ye_test2) 
        matriz_de_resultados[1,5] = np.append(matriz_de_resultados[1,5], resultados["metricas_test"]["accuracy"])
        dados_De_Analise(3)
        #print(resultados)
        print("\n")
        
        print(f"{i} split done")
    
    np.save("matriz_de_resultados.npy", matriz_de_resultados) #Guardar em cache
    np.save("matriz_de_fp.npy", matriz_de_fp) #Guardar em cache
    
    lista_de_arrays = [
    [np.array([]) for _ in range(7)]  # Linha 1
    for _ in range(4)                 # Repete para 4 linhas (1-> features, 2-> embeddings, 3-> feature selection, 4-> embeddings selection)
    ]
    
    fp_mean= np.array(lista_de_arrays, dtype= object)
    
    for i in range (0,7):
        
        for j in range(0,4):
            
            fp_mean[j,i]= np.mean(matriz_de_fp[j,i])
    
    #fp_mean ira retornar uma matriz com medias de false positives por atividade, 
    #em que cada linha representa features, embeddings, feature selection e embeddings selection
    #o que irá permitir concluir quais as atividades mais dificeis de classificar. Irá permitir comparar features e embeddings 
    #Bem como a relevancia de feature/embeddings selection
    
    models= ["FEATURES", "FEATURES PCA", "FEATURES RELIEFF", "EMBEDDINGS", "EMBEDDINGS PCA", "EMBEDDINGS RELIEFF"]
    
    #Para a estrategia intra-subject
    dados= matriz_de_resultados[0,:]
    dados= np.vstack(dados) # matriz com 6 linhas q representam os 6 modelos
    
    idx=melhor_modelo(dados)
    
    print(f"==melhor modelo==\n {models[idx]} \n")
    
    significance_test(dados, idx)
    


