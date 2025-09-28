import numpy as np, scipy, matplotlib.pyplot as plt
import csv
import math,time

Max_Part_num=14
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

def representacao_grafica(dados):
    j=1
    for person in dados:
        person= np.array(person)
        #calculo de modulo dos vetores de aceleracao e 
        vetor_modulos= np.array([])
        plt.figure()
        for i in range(1,activity_num):
            condition= person[:,-1]== str(i)
            aux= np.sqrt( person[condition, 1].astype(float) **2 +
            person[condition, 2].astype(float) **2 +
            person[condition, 3].astype(float) **2
            )
            
            plt.subplot(4,4,i)
            plt.boxplot(aux)
            plt.title("Boxplot atividade " + str(i))
        #-> Fim
    plt.show()
        

if __name__=="__main__":
    dados= descarregar_dados()
    representacao_grafica(dados)
#Em teoria, dura menos tempo. Alterei a forma de armazenamento de dados para facilitar o calculo de modulo
