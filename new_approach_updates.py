import numpy as np, scipy as sc, matplotlib.pyplot as plt
import csv
import math, time
import random
from scipy import stats
from scipy.stats import kstest, norm

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

modulos= []

FEATURES= []

def descarregar_dados():
    array = []
    for i in range(0,Max_Part_num+1):
    
        caminho = 'sample/part'
        caminho= f"{caminho}{i}/"
        
        array_aux = []
        for j in range(1,dev_num+1):
            caminho2 = f"{caminho}part{i}dev{j}.csv"
            with open(caminho2,newline='') as csv_file:
                linha = csv.reader(csv_file, delimiter=',')
                for row in linha:
                    array_aux.append(row)
        
        array.append(array_aux) 
    dados= np.array(array, dtype=object)
    return dados


#3.1
#Arranjar uma forma de usar a funcao calculo modulo para esse plot(nao repetir codigo)
def representacao_grafica(FEAT):
    p=1
    for person in modulos:
        plt.figure(num=str(p) + "pessoa")
        p+=1
        modulos_atividades= person[FEAT]
        i=1
        for modulo in modulos_atividades:
            plt.subplot(4,4,i)
            plt.boxplot(modulo)
            plt.title("Boxplot atividade " + str(i))
            i+=1

    plt.show()      


#funcao auxiliar    
def calculo_modulo(dados):   
    for person in dados:
        person = np.array(person)

        #ciclo para o cálculo da densidade das 3 caracteristicas - aceleração, giroscópio e magnetómetro
        aux1=[]
        for j in range(1, 8, 3):
            aux=[]
            #calculo de densidade so no Divice ID 2
            for i in range(1,activity_num):
                
                condition = (person[:, -1] == str(i))

                modulo = np.sqrt( person[condition, j].astype(float) **2 +
                person[condition, j+1].astype(float) **2 +
                person[condition, j+2].astype(float) **2
                )
                #usar z_score aqui para determinar indices dos outliers e fazer um reverse para determinar a linha correspondente nos dados iniciais
                aux.append(modulo) 
            
            aux1.append(aux)      
        
        modulos.append(aux1)  
            
    # Nota: o cálculo do módulo dos vetores foi implementado manualmente (x² + y² + z²)^(1/2).
    # Alternativamente, poderíamos usar a função np.linalg.norm(person[condition, 1:4], axis=1). 
    # No entanto, optou-se pela versão manual por ser ligeiramente mais direta e com desempenho equivalente.   
                            
           
                
#3.2 Para todos os parametros
def outlier_density(dados):
    number=1
    for person in dados:
        person = np.array(person)

        #ciclo para o cálculo da densidade das 3 caracteristicas - aceleração, giroscópio e magnetómetro
        for j in range(1, 8, 3):
            with open("Density_file",'a+') as file:
                if j == 1 : file.write("\n--- " + str(number) + " pessoa - " + "aceleracao ---\n")
                if j == 4 : file.write("\n--- " + str(number) + " pessoa - " + "giroscopio ---\n")
                if j == 7 : file.write("\n--- " + str(number) + " pessoa - " + "magnetometro ---\n")

            #calculo de densidade so no Divice ID 2
            for i in range(1,activity_num):
                #condição para obtenção das atividades do pulso direito
                condition = ((person[:, -1] == str(i)) & (person[:, 0].astype(int) == 2))

                modulo = np.sqrt( person[condition, j].astype(float) **2 +
                person[condition, j+1].astype(float) **2 +
                person[condition, j+2].astype(float) **2
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
        number+=1


#3.4
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
        plt.subplot(4,4,i)
        
        plt.scatter(np.linspace(0, len(modulo) , len(modulo)) , modulo, c=colors)
        plt.title("Boxplot atividade " + str(i))
        i+=1
    
    
def plot_zscore(K):
    for i in range (0, len(modulos)):
        for j in range (0, len(modulos[i])):
            
            if j ==0 : plt.figure(num="Acelerometro " + str(i+1) +" pessoa")
            if j ==1 : plt.figure(num="Giroscopio " + str(i+1) +" pessoa")
            if j ==2 : plt.figure(num="Magnetometro " + str(i+1) +" pessoa")
            
            z_score_test( modulos[i][j] , K )
    plt.show()    

#calcular centroides

def K_means(N, device):
    #device deve ser a coluna da coordenada x do tipo de dispositivo
    
    #Para os clusters, faz mais sentido repartir o array em n clusters, em vez de escolher aleatoriamente n centroides. 
    # Escolhendo N centroides random, sou capaz de escolher um centroide outlier e ele agrupar todos os outros outliers.
    #Daí, a distancia ao centroide pode nao ser muito superior ao treshold, nao identificando assim o outlier

    for person in dados: # a variavel organiza os dados de seguinte forma: [ [], [], [], [] ... , []] em que cada entrada sao os dados de todos os sensores de um individuo
        pessoa= np.array(person)
        
        fig, axes = plt.subplots(4,4, subplot_kw={'projection':'3d'})
        for i in range (1, activity_num):

            dados_a_tratar= pessoa [ pessoa[:,-1] == str(i) ] # ultima pos é activity_id
            
            dados_a_tratar = dados_a_tratar[:,device:device+3].astype(float)
            centroides= []
            pos=0 #posicao do ponterio copia
            

            step=int(len(dados_a_tratar)/N)
           
            for j in range(0,N):
                
                if j ==N-1 :  cluster= dados_a_tratar[pos:] #Assegurar que nao sao ultrapasados os limites do array  
                else: 
                    cluster= dados_a_tratar[pos:pos+step,:]
                    pos= pos+step
                
                centroides.append(np.median(cluster, axis=0)) # Calculo de centroides desta forma para garantir a nao escolha de outlier como centroide

            track=1 #monitora o numero de iterações
            flag=1 #houve mudança nos clusters
            
            group= np.zeros(len(dados_a_tratar), dtype=int) # array q define o grupo q cada ponto pertence
            centroides = np.array(centroides)
            
           #calculo de clusters
            while((track <= MAX_REPS) and (flag==1) ):
                
                track+=1 # Mais uma iteracao
                flag=0
                
                dist = np.linalg.norm(dados_a_tratar[:, np.newaxis, :] - centroides[np.newaxis, :, :], axis=2)
            
                for j in range(0,len(dados_a_tratar)):
                    recent_group= np.argmin(dist[j]) # o cluster a q pertence
                   
                    if(group[j] != recent_group):
                        group[j] = recent_group
                
                for j in range(0,N):
                    aux= dados_a_tratar[group==j]
                    new_center= np.median(aux, axis=0)
                    if(np.linalg.norm(new_center-centroides[j]) > SWITCH_VALUE ):
                        centroides[j] = new_center
                        flag=1
            
            colors= []
            dados_transf=[]
            for j in range(0,N):
                aux= dados_a_tratar[group==j]
                
                if(len(aux)==0) : continue
                dados_transf.append(aux)
                dist = np.linalg.norm(aux - centroides[j], axis=1)
                
                threshold = np.percentile(dist, 95)
                color= np.where(np.linalg.norm(aux-centroides[j], axis=1) > threshold, 'r', 'b')
                colors.extend(color)
            
            
            axes = axes.flatten()

            ax = axes[i-1]
            dados_transf= np.vstack(dados_transf)
            ax.scatter(dados_transf[:,0],dados_transf[:,1],dados_transf[:,2],c= colors)
                
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
        print(aux[a]+"\n\n")
        a+=1
        #Analise de normalidade para cada sensor
        for i in range(1, activity_num):
            ks_stat, p_norm = kstest(sensor[i], 'norm', args=(np.mean(sensor[i]), np.std(sensor[i])))
            if p_norm <=0.05 : 
                norm=0
                break
        
        if norm==0:
            f_stat, p_value = stats.f_oneway(*sensor)
            if (p_value<0.05): print("Rejeitamos H₀: há diferença significativa entre as médias.")
            else:
                print("Não rejeitamos H₀: não há diferença significativa entre as médias.")
        else:
            # Teste de Kruskal-Wallis
            h_stat, p_value = stats.kruskal(*sensor)
            
            # Interpretação
            if p_value < 0.05:
                print("Rejeitamos H₀: pelo menos um grupo é diferente.")
            else:
                print("Não rejeitamos H₀: não há diferença significativa entre os grupos.")
            


# ChatGPT

# 1. Mean
def feature_mean(window):
    return np.mean(window)

# 2. Median
def feature_median(window):
    return np.median(window)

# 3. Standard Deviation
def feature_std(window):
    return np.std(window)

# 4. Variance
def feature_variance(window):
    return np.var(window)

# 5. Root Mean Square (RMS)
def feature_rms(window):
    return np.sqrt(np.mean(window**2))

# 6. Averaged Derivatives (mean of first derivative)
def feature_avg_derivative(window):
    deriv = np.diff(window)
    return np.mean(deriv)

# 7. Skewness
def feature_skewness(window):
    return stats.skew(window)

# 8. Kurtosis
def feature_kurtosis(window):
    return stats.kurtosis(window)

# 9. Interquartile Range (IQR)
def feature_iqr(window):
    return np.percentile(window, 75) - np.percentile(window, 25)

# 10. Zero Crossing Rate (ZCR)
def feature_zero_crossing_rate(window):
    zc = ((window[:-1] * window[1:]) < 0).sum()
    return zc / len(window)

# 11. Mean Crossing Rate (MCR)
def feature_mean_crossing_rate(window):
    mean_val = np.mean(window)
    crossings = ((window[:-1] - mean_val) * (window[1:] - mean_val) < 0).sum()
    return crossings / len(window)


# 12. Spectral Entropy
def feature_spectral_entropy(window):
    # FFT
    fft_vals = np.fft.fft(window)
    psd = np.abs(fft_vals)**2
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # evitar log(0)
    return -np.sum(psd_norm * np.log2(psd_norm))

#ChatGPT end





#4.2
def features_extract(modulos):
    #??????? se nao existe o valor de frequencia de amostragem como saber o numero de amostras a que corresponde 5 segundos
    
    for sensor in modulos:
        #Por sensor(acelerometro, giroscopio, magnetometro)
        todas_janelas = []
        for i in range(1, activity_num):
            todas_janelas.extend(sensor[i])
            todas_janelas.append(100)  #100 significa que é o fim de uma atividade
        
        window_size=100# provisorio, visto que nao sabemos o valor de fs
        overlap=0.5
        step = int(window_size*(1-overlap))
        win_num=0
        for i in range( 0, len(todas_janelas) - window_size +1 , step ):
            window= todas_janelas[i:i+window_size]
            window= np.array(window)
            if(100 in window) :continue
            win_num+=1
            aux=[]
            aux.append( feature_mean(window))
            aux.append( feature_median(window))
            aux.append(feature_std(window))
            aux.append(feature_variance(window))
            aux.append(feature_rms(window))
            aux.append(feature_avg_derivative(window))
            aux.append(feature_skewness(window))
            aux.append(feature_kurtosis(window))
            aux.append(feature_iqr(window))
            aux.append(feature_zero_crossing_rate(window))
            aux.append(feature_mean_crossing_rate(window))
            aux.append(feature_spectral_entropy(window))
            FEATURES.append(aux)
        
        
    
    
    

if __name__=="__main__":
    dados= descarregar_dados()
    
    calculo_modulo(dados)
    
    #representacao_grafica(FEAT)
    
    #outlier_density(dados)
     
      
    modulos= np.array(modulos)
    
    K=3

    #plot_zscore(K)
    
    N=5
    K_means(N,4) #4 significa que estamos a analisar so o giroscopio relativamente a todas as atividades
    
#Coluna 1: Device ID
#Coluna 2: accelerometer x
#Coluna 3: accelerometer y
#Coluna 4: accelerometer z
#Coluna 5: gyroscope x
#Coluna 6: gyroscope y
#Coluna 7: gyroscope z 
#Coluna 8: magnetometer x
#Coluna 9: magnetometer y
#Coluna 10: magnetometer z
#Coluna 11: Timestamp
#Coluna 12: Activity Label
