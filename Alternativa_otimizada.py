import numpy as np, scipy, matplotlib.pyplot as plt
import csv
import math,time

Max_Part_num=2
dev_num=5
activity_num=17

def descarregar_dados():
    array= []
    for i in range(0,Max_Part_num+1):
    
        caminho = 'sample/part'
        caminho= f"{caminho}{i}/"
        
        array_aux= []
        for j in range(1,dev_num+1):
            caminho2= f"{caminho}part{i}dev{j}.csv"
            with open(caminho2,newline='') as csv_file:
                linha= csv.reader(csv_file, delimiter=',')
                for row in linha:
                    array_aux.append(row)
        
        array.append(array_aux) 
    dados= np.array(array,dtype=object)
    return dados

#3.1
def representacao_grafica(dados):
    j=1
    for person in dados:
        person= np.array(person)
        #calculo de modulo dos vetores de aceleracao
        plt.figure()
        for i in range(1,activity_num):
            condition= person[:,-1]== str(i)
            modulo= np.sqrt( person[condition, 1].astype(float) **2 +
            person[condition, 2].astype(float) **2 +
            person[condition, 3].astype(float) **2
            )
            
            #-> Fim de calculo
            
            plt.subplot(4,4,i)
            plt.boxplot(modulo)
            plt.title("Boxplot atividade " + str(i))
            
            
    plt.show()
        

#3.2 So a aceleracao, faltam os outros parametros
def outlier_density(dados):
    number=1
    for person in dados:
        person= np.array(person)
        #calculo de densidade so no Divice ID 2 e 4
        for i in range(1,activity_num):
            condition = ((person[:, -1] == str(i)) & 
             ((person[:, 0].astype(int) == 2) | (person[:, 0].astype(int) == 4)))
            
            modulo= np.sqrt( person[condition, 1].astype(float) **2 +
            person[condition, 2].astype(float) **2 +
            person[condition, 3].astype(float) **2
            )
            
            Q1 = np.percentile(modulo, 25)
            Q3 = np.percentile(modulo, 75)
            IQR = Q3 - Q1
            lower_lim = Q1 - 1.5 * IQR
            upper_lim = Q3 + 1.5 * IQR
            n0= len(modulo[(modulo < lower_lim) | (modulo > upper_lim) ])
            nr= len(modulo)
            d= (n0/nr)*100
            # -> Fim de calculo
            
            with open("Density_file",'a+') as file:
                if i == 1 : file.write(str(number) + " pessoa\n")
                file.write("Densidade de outlier do dataset da " + str(i) + " atividade: " + str(d) +"\n")
        number+=1


if __name__=="__main__":
    dados= descarregar_dados()
    representacao_grafica(dados)
    outlier_density(dados)
                
