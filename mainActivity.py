import csv
import numpy as np
import matplotlib.pyplot as plt

def descarregar_dados(array):
    for i in range(0, 15):
        caminho = 'samples/FORTH_TRACE_DATASET-master/FORTH_TRACE_DATASET-master/part'
        caminho = f"{caminho}{i}/part{i}dev"

        array_aux = []

        for j in range(1, 6):
            amostras = []
            caminho2 = f"{caminho}{j}.csv"
            with open(caminho2, newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter = ',')
                for row in spamreader:
                    amostras.append(row)
            array_aux.append(amostras)
        
        array.append(array_aux)
    return np.array(array, dtype=object) # Uso de dtype=object pelo fato de o nº de linhas variar 

def calcular_modulo_aceleracao(dados):
    modulos = [[] for _ in range(16)]
    for j in range(0, 15):
        for k in range(0, 5):
            for l in range(0, len(dados[j][k])):
                row = dados[j][k][l]
                try:
                    atividade = int(row[11])
                    if 1 <= atividade <= 16:
                        acc_x = float(row[1])
                        acc_y = float(row[2])
                        acc_z = float(row[3])
                        modulo = np.linalg.norm([acc_x, acc_y, acc_z])
                        modulos[atividade-1].append(modulo)
                except (ValueError, IndexError):
                    continue  # pula linhas mal formatadas
    return np.array(modulos, dtype=object)

def plot_atividade(modulos): 
    plt.boxplot([modulos[0], modulos[1], modulos[2],
                 modulos[3], modulos[4], modulos[5],
                 modulos[6], modulos[7], modulos[8],
                 modulos[9], modulos[10], modulos[11], 
                 modulos[12], modulos[13], modulos[14], 
                 modulos[15]], 
                 labels=["1", "2", "3", 
                         "4", "5", "6", 
                         "7", "8", "9", 
                         "10", "11", "12",
                         "13", "14", "15",
                         "16"])
    plt.title("Boxplot de múltiplos grupos")
    plt.ylabel("Valores de aceleração")
    plt.show()


if __name__=="__main__":
    array = []
    dados = descarregar_dados(array)
    modulos = calcular_modulo_aceleracao(dados)
    plot_atividade(modulos)
    

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