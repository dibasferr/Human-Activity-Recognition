import numpy as np, scipy, matplotlib.pyplot as plt
import csv
import math, time
import random

MAX_REPS= 42
SWITCH_VALUE=3

Max_Part_num=14
dev_num=5
activity_num=17

ACELERACAO = 0
GIROSCOPIO= 1
MAGNETOMETRO=2

#Uma especie de define
FEAT= GIROSCOPIO

FEATURES= []

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
def representacao_grafica(FEAT):
    modulos_atividades= FEATURES[FEAT]
    i=1
    for modulo in modulos_atividades:
        plt.subplot(4,4,i)
        plt.boxplot(modulo)
        plt.title("Boxplot atividade " + str(i))
        i+=1

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
            #usar z_score aqui para determinar indices dos outliers e fazer um reverse para determinar a linha correspondente nos dados iniciais
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
    for i in range (0, len(FEATURES)):
            
            if i ==0 : plt.figure(num="Acelerometro")
            if i ==1 : plt.figure(num="Giroscopio")
            if i ==2 : plt.figure(num="Magnetometro")
            
            z_score_test(FEATURES[i] , K )
    plt.show()    

#calcular centroides

def K_means(N, device):
    #device deve ser a coluna da coordenada x do tipo de dispositivo
    
    # Para os clusters, faz mais sentido repartir o array em n clusters, em vez de escolher aleatoriamente n centroides. 
    # Escolhendo N centroides random, sou capaz de escolher um centroide outlier e ele agrupar todos os outros outliers.
    # Daí, a distancia ao centroide pode nao ser muito superior ao treshold, nao identificando assim o outlier
        
    fig, axes = plt.subplots(4,4, subplot_kw={'projection':'3d'})
    for i in range (1, activity_num):

        dados_a_tratar= dados [ dados[:,-1] == str(i) ] # ultima pos é activity_id
            
        dados_a_tratar = dados_a_tratar[:,device:device+3].astype(float)
        centroides= []
        pos=0 #posicao do ponterio copia
            
        step=int(len(dados_a_tratar)/N)
           
        for j in range(0,N):
                
            if j == N-1 :  cluster= dados_a_tratar[pos:] #Assegurar que nao sao ultrapasados os limites do array  
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
               

if __name__=="__main__":
    dados= descarregar_dados()
    
    calculo_modulo(dados)
    
    #representacao_grafica(FEAT)
    
    #outlier_density(dados)
      
    FEATURES = np.array(FEATURES, dtype=object)
    #print(FEATURES.shape)
    
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