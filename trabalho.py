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

#Limite de escala de cinza aceito, mude para aceitar menos ou mais escalas
Limite = 230

colunas = ['ID Imagem', 'ID Folha', 'Perimetro', 'Diametro Minimo', 'Diametro Maximo', 'Excentricidade']

def floodfill(imagem, posicao, visitados, cor, folha_cortada):

    # pega os limites da imagem total
    x_max, y_max = imagem.shape

    # adiciona na fila o pixel inicial
    fila = []
    fila.append(posicao)

    # lista dos pontos percorridos
    percorrido = []

    for i,j in fila:

        # verifica se o pixel inferior direito é pintado e nunca foi visitado
        if (i+1 < x_max and j+1 < y_max) and (visitados[i+1, j+1] == False) and imagem[i+1, j+1] < Limite:
            # adiciona ele a fila
            fila.append([i+1, j+1])
            # atribui cor da id
            folha_cortada[i+1, j+1] = cor
            # diz que ele ja foi visitado
            visitados[i+1, j+1] = True

        # verifica se o pixel inferior esquerdo é pintado e nunca foi visitado
        if (i-1 >= 0 and j+1 < y_max) and (visitados[i-1, j+1]  == False) and imagem[i-1, j+1] < Limite:
            # adiciona ele a fila
            fila.append([i-1, j+1])
            # atribui cor da id
            folha_cortada[i-1, j+1] = cor
            # diz que ele ja foi visitado
            visitados[i-1, j+1] = True

        # verifica se o pixel superior esquerdo é pintado e nunca foi visitado
        if (i-1 >= 0 and j-1 >= 0) and (visitados[i-1, j-1] == False) and imagem[i-1, j-1] < Limite:
            # adiciona ele a fila
            fila.append([i-1, j-1])
            # atribui cor da id
            folha_cortada[i-1, j-1] = cor
            # diz que ele ja foi visitado
            visitados[i-1, j-1] = True

        # verifica se o pixel superior direito é pintado e nunca foi visitado
        if (i+1 < x_max and j-1 >= 0) and (visitados[i+1, j-1] == False) and imagem[i+1, j-1] < Limite:
            # adiciona ele a fila
            fila.append([i+1, j-1])
            # atribui cor da id
            folha_cortada[i+1, j-1] = cor
            # diz que ele ja foi visitado
            visitados[i+1, j-1] = True

        # verifica se o pixel da direita é pintado e nunca foi visitado
        if (i+1 < x_max) and (visitados[i+1, j] == False) and imagem[i+1, j] < Limite:
            # adiciona ele a fila
            fila.append([i+1, j])
            folha_cortada[i+1, j] = cor
            # diz que ele ja foi visitado
            visitados[i+1, j] = True

        # verifica se o pixel da esquerda é pintado e nunca foi visitado
        if (i-1 >= 0) and (visitados[i-1, j] == False) and imagem[i-1, j] < Limite:
            # adiciona ele a fila
            fila.append([i-1, j])
            # atribui cor da id
            folha_cortada[i-1, j] = cor
            # diz que ele ja foi visitado
            visitados[i-1, j] = True

        # verifica se o pixel acima é pintado e nunca foi visitado
        if (j+1 < y_max) and (visitados[i, j+1] == False) and imagem[i, j+1] < Limite:
            # adiciona ele a fila
            fila.append([i, j+1])
            # atribui cor da id
            folha_cortada[i, j+1] = cor
            # diz que ele ja foi visitado
            visitados[i, j+1] = True

        # verifica se o pixel de baixo é pintado e nunca foi visitado
        if (j-1 >= 0) and (visitados[i, j-1] == False) and imagem[i, j-1] < Limite:
            # adiciona ele a fila
            fila.append([i, j-1])
            # atribui cor da id
            folha_cortada[i, j-1] = cor
            # diz que ele ja foi visitado
            visitados[i, j-1] = True

        # adiciona o ponto percorrido
        percorrido.append([i,j])


    return percorrido

# verifica se os vizinhos são fundo ou não 
def vizinhos_brancos(imagem, x, y):
    # soma dos vizinhos que não são brancos
    soma = 0
    x_max, y_max = imagem.shape

    # verifica cada um dos vizinhos do pixel atual
    if (x+1 < x_max and y+1 < y_max) and imagem[x+1,y+1] != 0:
        soma += 1
    if (x+1 < x_max and y-1 >= 0) and imagem[x+1,y-1] != 0:
        soma += 1
    if (x-1 >= 0 and y-1 >= 0) and imagem[x-1,y-1] != 0:
        soma += 1
    if (x-1 >= 0 and y+1 < y_max) and imagem[x-1,y+1] != 0:
        soma += 1
    if (x+1 < x_max) and imagem[x+1,y] != 0:
        soma += 1
    if (x-1 >= 0) and imagem[x-1,y] != 0:
        soma += 1
    if (y+1 < y_max) and imagem[x,y+1] != 0:
        soma += 1
    if (y-1 >= 0) and imagem[x,y-1] != 0:
        soma += 1

    return soma

# algoritmo feito a mão com ideias do afinamento de regiões
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
    # cor da id da folha, para tentar diferenciar cada uma
    cor = 15
    # identificador da sub-imagem atual
    folha_quant = folha_quant
    
    for x in range(0, x_max):
        for y in range(0, y_max):
            
            # se não foi visitado ainda e ta abaixo do limite de cinza permitido
            if visitados[x,y] == False and imagem[x,y] < Limite:
                
                # cria uma matriz de cores brancas, onde vai ser preenchido com a cor da tag aonde a folha se encontra
                folha_cortada = np.full(imagem.shape, 255)
                visitados[x,y] = True
                pontos = floodfill(imagem, [x,y], visitados, cor, folha_cortada)

                # foi feito isso porque teve vezes que ele pegou dois pixels só e disse que era uma folha
                if len(pontos) > 500:
                    folha_quant += 1

                    x_min, y_min = pontos[0]
                    x_max, y_max = pontos[0]

                    # busca o minimo e maximo de cada eixo (x,y)
                    for ponto in pontos:
                        if ponto[0] < x_min:
                            x_min = ponto[0]
                        if ponto[0] > x_max:
                            x_max = ponto[0]
                        if ponto[1] < y_min:
                            y_min = ponto[1]
                        if ponto[1] > y_max:
                            y_max = ponto[1]

                    # corta a imagem tanto no rbg quanto no grayscale, para ter ambas as formas de imagens
                    imagem_final = imagem_total[x_min - 5: x_max+5, y_min -5: y_max+5, :]
                    borda_gray = folha_cortada[x_min - 5: x_max+5, y_min -5: y_max+5]

                    diametro_M = 0
                    diametro_m = 0

                    # função para pegar a borda da folha recortada. //foi comentada pq foi utilizado outra função pronta e que é mais eficiente   
                    #borda = acha_borda(borda_gray, cor)
                    #perimetro = len(np.where(borda!=255)[0])

                    # aplica canny para a imagem cortada, e inverte os valores de pixel da detecção porque é preferivel a borda com cor preta e fundo branco
                    bordas = cv2.Canny(imagem_final, 255, 255)
                    bordas = cv2.bitwise_not(bordas)

                    # pega os dados do perimetro
                    perimetro1 = len(np.where(bordas==0)[0])

                    # define a distancia de pixeis entre cada um dos minimo para os maximos
                    x_dif = abs(x_max - x_min)
                    y_dif = abs(y_max - y_min)

                    # ve qual eixo tem o diametro maior e atribui, e consequentemente atribui o menor
                    if x_dif > y_dif:
                        diametro_M = x_dif
                        diametro_m = y_dif
                    else:
                        diametro_M = y_dif
                        diametro_m = x_dif

                    # calcula a excentricidade do objeto
                    excentricidade = diametro_M/diametro_m

                    # escreve a imagem da sub-imagem e da borda dela
                    cv2.imwrite(nome_arq + f"-{folha_quant}.png", imagem_final)
                    cv2.imwrite(nome_arq + f"-{folha_quant}-P.png", bordas)

                    # insere os dados da planilha
                    linha_planilha = {
                        'ID Imagem': nome_arq[9:],
                        'ID Folha':  folha_quant,
                        'Perímetro': perimetro1,
                        'Diâmetro Mínimo': diametro_m,
                        'Diâmetro Máximo': diametro_M,
                        'Excentricidade': excentricidade
                    }

                    # muda o id da folha seguinte e adiciona na planilha a nova sub-imagem
                    cor += 15
                    planilha = planilha.append(pd.DataFrame(linha_planilha, index=[folha_quant-1], columns=colunas), ignore_index=True)
                    print(planilha)
            else:
                visitados[x,y] = True
                
    # é chamado recursivamente essa função porque de algum jeito os for's ali de cima não passam por todos os pontos, vai saber porque
    if len(np.where(visitados==False)[0]) > 0:
        planilha = planilha.append(aplica_tags(imagem_total, imagem, visitados, nome_arq, folha_quant, planilha), ignore_index=True)
        print(planilha)
    
    print(planilha)
    return planilha
    

# função principal
if __name__ == "__main__":
    # pega todos os arquivos do diretorio Folhas/
    arquivos = [join("./Folhas/", f) for f in listdir("./Folhas/") if isfile(join("./Folhas", f))]

    # inicia o contador de tempo
    inicio = time.time()

    # cria o csv
    planilha = pd.DataFrame(columns=colunas)

    # para cada uma das 15 imagens dentro do diretorio Folhas
    for i in range(1):

        # le o arquivo de entrada
        imagem_total = cv2.imread(arquivos[i])

        # converte uma copia da imagem para escala de cinza
        imagem_grayscale = cv2.cvtColor(imagem_total, cv2.COLOR_BGR2GRAY)

        # cria uma imagem booleana para guardar os pixel que foram visitados
        visitados = np.full(imagem_grayscale.shape, False, dtype=bool)

        # para cada pixel que não é da escala de cinza que foi aceita, é dito que ele ja foi visitado
        visitados[np.where(visitados >= Limite)] = True

        # chama a função para gerar as sub-imagens da imagem atual
        planilha = planilha.append(aplica_tags(imagem_total, imagem_grayscale, visitados, arquivos[i][:-4], 0, planilha), ignore_index=True)

        # exibe o tempo necessario para extrair todas as informações da imagem atual
        print(f'Para processar as folhas da imagem {arquivos[i][9:-4]} levou {time.time() - inicio}')

        # escreve o csv
    planilha.to_csv('Dados_Folhas.csv', index=False)