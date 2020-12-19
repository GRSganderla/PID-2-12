import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import warnings
import copy
import sys
import os, sys

Limite = 230

def esta_dentro(x_max, y_max, x, y):
    return (x >= 0 and x < x_max and y >= 0 and y < y_max)


def floodfill(imagem, x, y, visitados, cor):

    imagem_nova = np.zeros(imagem.shape, dtype=int)
    limite_x, limite_y = imagem.shape

    fila = [[x,y]]
    visitados[x,y] = False
    percorrido = []

    for vizinho in fila:
        fila.pop(0)
        percorrido.append(vizinho)

        if esta_dentro(limite_x, limite_y, vizinho[0]+1, vizinho[1]+1) and visitados[vizinho[0]+1, vizinho[1]+1] and imagem[vizinho[0]+1, vizinho[1]+1] < Limite:
            fila.append([vizinho[0]+1, vizinho[1]+1])
            imagem_nova[vizinho[0]+1, vizinho[1]+1] = cor
            visitados[vizinho[0]+1, vizinho[1]+1] = False

        if esta_dentro(limite_x, limite_y, vizinho[0]-1, vizinho[1]+1) and visitados[vizinho[0]-1, vizinho[1]+1] and imagem[vizinho[0]-1, vizinho[1]+1] < Limite:
            fila.append([vizinho[0]-1, vizinho[1]+1])
            imagem_nova[vizinho[0]-1, vizinho[1]+1] = cor
            visitados[vizinho[0]-1, vizinho[1]+1] = False

        if esta_dentro(limite_x, limite_y, vizinho[0]-1, vizinho[1]-1) and visitados[vizinho[0]-1, vizinho[1]-1] and imagem[vizinho[0]-1, vizinho[1]-1] < Limite:
            fila.append([vizinho[0]-1, vizinho[1]-1])
            imagem_nova[vizinho[0]-1, vizinho[1]-1] = cor
            visitados[vizinho[0]-1, vizinho[1]-1] = False

        if esta_dentro(limite_x, limite_y, vizinho[0]+1, vizinho[1]-1) and visitados[vizinho[0]+1, vizinho[1]-1] and imagem[vizinho[0]+1, vizinho[1]-1] < Limite:
            fila.append([vizinho[0]+1, vizinho[1]-1])
            imagem_nova[vizinho[0]+1, vizinho[1]-1] = cor
            visitados[vizinho[0]+1, vizinho[1]-1] = False

        if esta_dentro(limite_x, limite_y, vizinho[0]+1, vizinho[1]) and visitados[vizinho[0]+1, vizinho[1]] and imagem[vizinho[0]+1, vizinho[1]] < Limite:
            fila.append([vizinho[0]+1, vizinho[1]])
            imagem_nova[vizinho[0]+1, vizinho[1]] = cor
            visitados[vizinho[0]+1, vizinho[1]] = False

        if esta_dentro(limite_x, limite_y, vizinho[0]-1, vizinho[1]) and visitados[vizinho[0]-1, vizinho[1]] and imagem[vizinho[0]-1, vizinho[1]] < Limite:
            fila.append([vizinho[0]-1, vizinho[1]])
            imagem_nova[vizinho[0]-1, vizinho[1]] = cor
            visitados[vizinho[0]-1, vizinho[1]] = False

        if esta_dentro(limite_x, limite_y, vizinho[0], vizinho[1]+1) and visitados[vizinho[0], vizinho[1]+1] and imagem[vizinho[0], vizinho[1]+1] < Limite:
            fila.append([vizinho[0], vizinho[1]+1])
            imagem_nova[vizinho[0], vizinho[1]+1] = cor
            visitados[vizinho[0], vizinho[1]+1] = False

        if esta_dentro(limite_x, limite_y, vizinho[0], vizinho[1]-1) and visitados[vizinho[0], vizinho[1]-1] and imagem[vizinho[0], vizinho[1]-1] < Limite:
            fila.append([vizinho[0], vizinho[1]-1])
            imagem_nova[vizinho[0], vizinho[1]-1] = cor
            visitados[vizinho[0], vizinho[1]-1] = False


    return percorrido, imagem_nova

def dentro_do_limite(shape, x, y):
    return (x >= 0 and x < shape[0] and y >= 0 and y < shape[1])

def vizinhos_brancos(imagem, x, y):
    soma = 0

    if dentro_do_limite(imagem.shape, x+1, y+1) and imagem[x+1,y+1] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x+1, y-1) and imagem[x+1,y-1] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x-1, y-1) and imagem[x-1,y-1] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x-1, y+1) and imagem[x-1,y+1] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x+1, y) and imagem[x+1,y] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x-1, y) and imagem[x-1,y] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x, y+1) and imagem[x,y+1] != 0:
        soma += 1
    if dentro_do_limite(imagem.shape, x, y-1) and imagem[x,y-1] != 0:
        soma += 1

    return soma

def acha_borda(imagem, cor):

    x, y = imagem.shape
    borda =  np.full(imagem.shape, 255, dtype=int)

    for i in range (0, x):
        for j in range(0, y):
            N_p = vizinhos_brancos(imagem, i, j)
            
            if imagem[i,j] != 0 and N_p >= 2 and N_p <= 6:
                borda[i,j] = 0
            else:
                borda[i,j] = 255


    return np.array(borda)           


def aplica_tags(imagem_total, imagem, visitados, nome_arq, folha_quant, planilha):

    x_max, y_max = imagem.shape
    borda = []
    cor = 15
    folha_quant = folha_quant

    
    for x in range(0, x_max):
        for y in range(0, y_max):
        
            if visitados[x,y] and imagem[x,y] < Limite:
                pontos, imagem_achada = floodfill(imagem, x,y, visitados, cor)

                if len(pontos) > 100:
                    folha_quant += 1

                    x_min, y_min = pontos[0]
                    x_max, y_max = pontos[0]

                    for ponto in pontos:
                        if ponto[0] < x_min:
                            x_min = ponto[0]
                        if ponto[0] > x_max:
                            x_max = ponto[0]
                        if ponto[1] < y_min:
                            y_min = ponto[1]
                        if ponto[1] > y_max:
                            y_max = ponto[1]

                    imagem_final = imagem_total[x_min - 5: x_max+5, y_min -5: y_max+5, :]
                    borda_gray = imagem_achada[x_min - 5: x_max+5, y_min -5: y_max+5]

                    # função para pegar a borda da folha recortada. //foi comentada pq foi utilizado outra função pronta e que é mais eficiente   
                    #borda = acha_borda(borda_gray, cor)
                    #perimetro = len(np.where(borda!=255)[0])

                    bordas = cv2.Canny(imagem_final, 255, 255)

                    bordas = cv2.bitwise_not(bordas)
                    plt.imshow(bordas, cmap='gray')
                    plt.show()

                    perimetro1 = len(np.where(bordas==0)[0])
                    #cv2.imwrite(nome_arq + f"-{folha_quant}.png", imagem_final)
                    #cv2.imwrite(nome_arq + f"-{folha_quant}-P.png", borda)

                    linha_planilha = {
                        'ID Imagem': nome_arq[9:],
                        'ID Folha':  folha_quant,
                        'Perímetro': perimetro1,
                        'Diâmetro Mínimo': 0,
                        'Diâmetro Máximo': 0,
                        'Excentricidade': 0
                    }

                    cor += 15
                    planilha = planilha.append(linha_planilha, ignore_index=True)
            else:
                visitados[x,y] = False
                
    if len(np.where(visitados==True)[0]) > 0:
        aplica_tags(imagem_total, imagem, visitados, nome_arq, folha_quant, planilha)
    


if __name__ == "__main__":
    arquivos = [join("./Folhas/", f) for f in listdir("./Folhas/") if isfile(join("./Folhas", f))]
    filtro_sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    inicio = time.time()

    planilha = pd.DataFrame(columns=['ID Imagem', 'ID Folha', 'Perímetro', 'Diâmetro Mínimo', 'Diâmetro Máximo', 'Excentricidade'])

    for i in range(len(arquivos)):
        imagem_total = cv2.imread(arquivos[i])

        imagem_grayscale = cv2.cvtColor(imagem_total, cv2.COLOR_BGR2GRAY)

        visitados = np.full(imagem_grayscale.shape, True, dtype=bool)

        visitados[np.where(visitados >= Limite)] = False

        aplica_tags(imagem_total, imagem_grayscale, visitados, arquivos[i][:-4], 0, planilha)

        print(f'Para processar as folhas da imagem {arquivos[i][9:-4]} levou {time.time() - inicio}')

    planilha.to_csv('Dados_Folhas.csv', index=False)