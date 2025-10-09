import numpy as np, scipy, matplotlib.pyplot as plt
import csv
import math, time

Max_Part_num=2
dev_num=5
activity_num=17

ACELERACAO = 1
GIROSCOPIO= 4
MAGNETOMETRO=7

#Uma especie de define
FEAT= GIROSCOPIO

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
def representacao_grafica(dados):
    for person in dados:
            person = np.array(person)
            #calculo de modulo dos vetores de aceleracao
            plt.figure()
            for i in range(1,activity_num):
                condition = person[:,-1] == str(i)
                modulo = np.sqrt( person[condition, FEAT].astype(float) **2 +
                person[condition, FEAT+1].astype(float) **2 +
                person[condition, FEAT+2].astype(float) **2
                )
                #-> Fim de calculo

                # Nota: o cálculo do módulo dos vetores foi implementado manualmente (x² + y² + z²)^(1/2).
                # Alternativamente, poderíamos usar a função np.linalg.norm(person[condition, 1:4], axis=1). 
                # No entanto, optou-se pela versão manual por ser ligeiramente mais direta e com desempenho equivalente.

                plt.subplot(4,4,i)
                plt.boxplot(modulo)
                plt.title("Boxplot atividade " + str(i))
                
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
        
        FEATURES.append(aux1)  
    
    print(len(FEATURES[0][0]))        
        
                            
           
                
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
                
                FEATURES.append(modulo)
                
                
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
    
    

#calcular o modulo diretamente no nos dados de person(x,y,z). Se for uma linha outlier, guardar logo a posicao da linha no array daquela pessoa para indicar que aquela linha é um outlier. No final,usar esses indices para dizer que aquelas linhas sao outlier num plot tridimensional
#cada pessoa tem 3 plots, acel, girosc, megn
#se cada pessoa tiver 3, cada um vai ter coordenadas de uma atividade atividade(considerada outlier no grupo de sua atividade). Ou seja em vez de guardar modulo é melhor guardar posicoes

#ou usar um plot bidimensional(indice modulo outlier, valor modulo) no array de modulos
#pode misturar os 16 arrays num so e fazer isso -> 3 plots por pessoa
if __name__=="__main__":
    dados= descarregar_dados()
    
    #representacao_grafica(dados)
    
    #outlier_density(dados)
    
    calculo_modulo(dados) 
      
    FEATURES= np.array(FEATURES)
    
    K=3

    for i in range (0, len(FEATURES)):
        for j in range (0, len(FEATURES[i])):
            
            if j ==0 : plt.figure(num="Acelerometro " + str(i+1) +" pessoa")
            if j ==1 : plt.figure(num="Giroscopio " + str(i+1) +" pessoa")
            if j ==2 : plt.figure(num="Magnetometro " + str(i+1) +" pessoa")
            
            Outliers_index= z_score_test( FEATURES[i][j] , K )
    plt.show()    
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
