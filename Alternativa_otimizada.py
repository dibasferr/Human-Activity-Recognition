import numpy as np, scipy, matplotlib.pyplot as plt
import csv
import math, time
import random
from scipy import stats
from scipy.stats import kstest, norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

MAX_REPS= 42
SWITCH_VALUE=3

Max_Part_num=2
dev_num=5
activity_num=17

ACELERACAO = 0
GIROSCOPIO= 1
MAGNETOMETRO=2

#Uma especie de define
FEAT= GIROSCOPIO

FEATURES= []


#2
def descarregar_dados():
    array = []
    for i in range(0,Max_Part_num+1):
        caminho = 'sample/part'
        caminho= f"{caminho}{i}/"
        for j in range(1,dev_num+1):
            caminho2 = f"{caminho}part{i}dev{j}.csv"
            with open(caminho2,newline='') as csv_file:
                linha = csv.reader(csv_file, delimiter=',')
                for row in linha:
                    array.append(row)
    dados = np.array(array, dtype=object)
    return dados


#3.1
#Arranjar uma forma de usar a funcao calculo modulo para esse plot(nao repetir codigo)
def representacao_grafica():
    modulos_atividades= FEATURES # [FEAT]
    nomes_variaveis = ["Aceleração", "Giroscópio", "Magnetómetro"]
    i=1
    print(len(modulos_atividades))
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

#3.3 e 3.4
def z_score_test(modulos, k):
    outliers = []
    i=1
    for modulo in modulos:
        modulo = np.array(modulo, dtype=object)
        
        media = np.mean(modulo)
        desvio_padrao = np.std(modulo)

        if desvio_padrao > 0:
            z_scores = (modulo - media) / desvio_padrao
        else:
            np.zeros_like(modulo)
        
        outlier_indices = np.where(np.abs(z_scores) > k)
        outliers.append(outlier_indices)
        
        colors=np.array(['b']*len(modulo))
        colors[outlier_indices]= 'r'
        plt.subplot(1,16,i)
        
        plt.scatter(np.linspace(0, len(modulo) , len(modulo)) , modulo, c=colors)
        plt.title("Boxplot atividade " + str(i))
        i+=1
    
    
def plot_zscore(K):
    for i in range (0, len(FEATURES)):
            
            if i ==0 : plt.figure(num="Acelerometro")
            if i ==1 : plt.figure(num="Giroscopio")
            if i ==2 : plt.figure(num="Magnetometro")
            
            z_score_test(FEATURES[i] , K )
    plt.show()    

#calcular centroides


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
    dados_transf=np.array(dados_transf)
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
def feature_extraction(dados):
    dados=dados.astype(float)
    resultado=[]
    #Utilizados para navegar a estrutura de dados
    pos_inicial=0
    pos_final=pos_inicial
    flag_final=False
    while(True):
        
        ts_window = dados[pos_inicial:, :]

        # Encontrar o índice relativo do primeiro elemento que excede o intervalo
        relative_indices = np.where(ts_window - dados[pos_inicial] >= 5000)[0]

        if relative_indices.size > 0:
            pos_final = pos_inicial + relative_indices[0]
        else:
            # Caso não encontre nenhum timestamp >= 5000
            pos_final = len(dados) - 1
            flag_final = True
        
        acc = dados[pos_inicial:pos_final, 1:4]  # acelerômetro
        gyro = dados[pos_inicial:pos_final, 4:7] # giroscópio
        mag = dados[pos_inicial:pos_final, 7:10] # magnetômetro

        pos_inicial+= (pos_final-pos_inicial)//2
        
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
            [int(dados[pos_inicial,-1])]
        )
        resultado.append(segmento)
        if(flag_final): break
        
    return np.array(resultado, dtype=object)


#4.3 e 4.4
def aplicar_pca(estrutura):
    features = np.array(estrutura[:, :-1], dtype=float) # elimina coluna de classes (última coluna)
    
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
    cum_var_exp = np.cumsum(var_exp)

    print("Variância explicada por componente:", var_exp)
    print("Variância acumulada:", cum_var_exp)

    # índice da primeira vez que a variância acumulada >= 0.75
    num_dim_75 = np.argmax(cum_var_exp >= 0.75) + 1
    print("Número de dimensões necessárias para 75% da variância:", num_dim_75)
    
    # Projetando os dados originais nas 'num_dim_75' primeiras componentes
    pca_reduced = PCA(n_components=num_dim_75)
    X_reduced = pca_reduced.fit_transform(z_scores)

    # Exemplo: pegar features reduzidas da primeira amostra
    sample_index = 0
    print("Features reduzidas da primeira amostra:", X_reduced[sample_index])

    # Essa abordagem é vantajosa pois reduz a dimensão dos dados, facilitando análise e visualização.
    # Remove redundância entre features correlacionadas, e ainda mantém a maior parte da informação (variância).
    # Tem limitações, pois como o PCA assume linearidade padrões não lineares podem ser perdidos.
    # Há perda de informações, mesmo com 75% da variância os 25% ainda se perdem (são descartados).
    

#4.5 e 4.6
def fisher_score(estrutura):
    # Extrair features e classes
    features = np.array(estrutura[:, :-1], dtype=float)
    classes = np.array(estrutura[:, -1], dtype=int)
    print(classes)
    
    N, M = features.shape
    
    # Normalizar por feature (z-score)
    #media = np.mean(features, axis=0)
    #desvio_padrao = np.std(features, axis=0)
    #z_scores = (features - media) / (desvio_padrao + 1e-8)

    classes_unicas = np.unique(classes)
    print(classes_unicas)
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

    # Normalizar por feature (z-score)
    #media = np.mean(features, axis=0)
    #desvio = np.std(features, axis=0)
    #z_scores = (features - media) / (desvio + 1e-8)
    
    nn = NearestNeighbors(n_neighbors=n_vizinhos + 1)
    nn.fit(features)
    
    print(N)
    
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

def dez_melhores_features():
    top10_fisher_idx = np.argsort(f_scores)[-10:][::-1]
    top10_relief_idx = np.argsort(pesos)[-10:][::-1]

    print("Top10 Fisher indices:", top10_fisher_idx)
    print("Top10 ReliefF indices:", top10_relief_idx)
    print("Intersecção:", np.intersect1d(top10_fisher_idx, top10_relief_idx))


if __name__ == "__main__":
    #2
    dados = descarregar_dados()
    
    #3.1
    calculo_modulo(dados)
    FEATURES = np.array(FEATURES, dtype=object)
    #print(FEATURES.shape)
    
    #representacao_grafica(FEAT)
    
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
    
    #4.2 (Extraira features)
    vetor_features = feature_extraction(dados)
    vetor_features = np.vstack(vetor_features)
    print(vetor_features.shape)
    time.sleep(10)
    
    #4.3 e 4.4 (PCA)
    aplicar_pca(vetor_features)
    
    #4.5 e 4.6 (fisher_score e reliefF)
    f_scores = fisher_score(vetor_features)
    print(f_scores)
    print(f_scores.shape)
    
    pesos = reliefF(vetor_features, 5)
    print(pesos)
    print(pesos.shape)
    
    dez_melhores_features()
