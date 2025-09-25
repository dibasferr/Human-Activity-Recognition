import csv
import numpy as np

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

if __name__=="__main__":
    array = []
    dados = descarregar_dados(array)
    print(len(array[0][0]))





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