import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
import time

def convolucao(imagem, kernel, average=False, verbose=False):
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
 
    linhas, colunas = imagem.shape
    linha_kernel, coluna_kernel = kernel.shape
 
    saida = np.zeros(imagem.shape)
 
    height_espacado = int((linha_kernel - 1) / 2)
    width_espacado = int((coluna_kernel - 1) / 2)
 
    imagem_espacada = np.zeros((linhas + (2 * height_espacado), colunas + (2 * width_espacado)))
 
    imagem_espacada[height_espacado:imagem_espacada.shape[0] - height_espacado, width_espacado:imagem_espacada.shape[1] - width_espacado] = imagem
 
    for row in range(linhas):
        for col in range(colunas):
            saida[row, col] = np.sum(kernel * imagem_espacada[row:row + linha_kernel, col:col + coluna_kernel])
            if average:
                saida[row, col] /= kernel.shape[0] * kernel.shape[1]

    if verbose:
            plt.imshow(saida, cmap='gray')
            plt.title("Convolução")
            plt.show()
 
    return saida

def normaliza(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)
 
 
def kernel_gaussiano(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)

    for i in range(size):
        kernel_1D[i] = normaliza(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    return kernel_2D
 
def gaussian_blur(image, kernel_size, verbose=False):
    kernel = kernel_gaussiano(kernel_size, sigma=math.sqrt(kernel_size), verbose=verbose)
    return convolucao(image, kernel, average=True, verbose=verbose)
 
def deteccao_borda(imagem, filtro,  converte_degraus=False, verbose=False):
    Gx = convolucao(imagem, filtro, verbose)

    Gy = convolucao(imagem, np.flip(filtro.T, axis = 0), verbose)
    
    magnetude_gradiente = np.sqrt(np.square(Gx) + np.square(Gy))

    magnetude_gradiente *= 255.0 / magnetude_gradiente.max()

    direcao_gradiente = np.arctan2(Gy, Gx)

    if converte_degraus:
        direcao_gradiente = np.rad2deg(direcao_gradiente)
        direcao_gradiente += 180
    
    if verbose:
            plt.imshow(magnetude_gradiente, cmap='gray')
            plt.title("Deteccão de Borda por Sobel")
            plt.show()

    return magnetude_gradiente, direcao_gradiente

def supressao_non_max(magnetude_gradiente, direcao_gradiente, verbose):

    linhas, colunas = magnetude_gradiente.shape

    saida = np.zeros(magnetude_gradiente.shape)

    PI = 180

    for linha in range(1, linhas-1):
        for coluna in range(1, colunas-1):
            direcao = direcao_gradiente[linha, coluna]

            if(0 <= direcao < PI/8) or (15 * PI / 8 <= direcao <= 2 * PI):
                pixel_ant = magnetude_gradiente[linha, coluna - 1]
                pixel_pro = magnetude_gradiente[linha, coluna + 1]
            
            elif (PI / 8 <= direcao < 3 * PI / 8) or (9 * PI / 8 <= direcao < 11 * PI / 8):
                pixel_ant = magnetude_gradiente[linha + 1, coluna - 1]
                pixel_pro = magnetude_gradiente[linha - 1, coluna + 1]
            
            elif (3 * PI / 8 <= direcao < 5 * PI / 8) or (11 * PI / 8 <= direcao < 13 * PI / 8):
                pixel_ant = magnetude_gradiente[linha - 1, coluna]
                pixel_pro = magnetude_gradiente[linha + 1, coluna]
            
            else:
                pixel_ant = magnetude_gradiente[linha - 1, coluna - 1]
                pixel_pro = magnetude_gradiente[linha + 1, coluna + 1]

            if magnetude_gradiente[linha, coluna] >= pixel_ant and magnetude_gradiente[linha, coluna] >= pixel_pro:
                saida[linha, coluna] = magnetude_gradiente[linha, coluna]

    if verbose:
            plt.imshow(saida, cmap='gray')
            plt.title("Supressão Non Max Aplicado")
            plt.show()

    return saida

def limiar(imagem, baixo, alto, fraco, verbose=False):

    saida = np.zeros(imagem.shape)

    forte = 255

    linha_forte, coluna_forte = np.where(imagem >= alto)
    linha_fraca, coluna_fraca = np.where((imagem <= alto) & (imagem >= baixo))

    saida[linha_forte, coluna_forte] = forte
    saida[linha_fraca, coluna_fraca] = fraco

    if verbose:
            plt.imshow(saida, cmap='gray')
            plt.title("Limiar Aplicado")
            plt.show()
    
    return saida

def hipotese(imagem, fraco):
    linhas, colunas = imagem.shape

    Sup_Inf = imagem.copy()

    for linha in range(1, linhas):
        for coluna in range(1, colunas):
            if (Sup_Inf[linha, coluna] == fraco):
                if Sup_Inf[linha, coluna + 1] == 255 or Sup_Inf[linha, coluna - 1] == 255 or Sup_Inf[linha - 1, coluna] == 255 or Sup_Inf[linha + 1, coluna] == 255 or Sup_Inf[linha - 1, coluna - 1] == 255 or Sup_Inf[linha + 1, coluna - 1] == 255 or Sup_Inf[linha - 1, coluna + 1] == 255 or Sup_Inf[linha + 1, coluna + 1]:
                    Sup_Inf[linha, coluna] = 255
                else:
                    Sup_Inf[linha, coluna] = 0

    Inf_Sup = imagem.copy()    

    for linha in range(linhas-1, 0, -1):
        for coluna in range(colunas-1, 0, -1):
            if (Inf_Sup[linha, coluna] == fraco):
                if Inf_Sup[linha, coluna + 1] == 255 or Inf_Sup[linha, coluna - 1] == 255 or Inf_Sup[linha - 1, coluna] == 255 or Inf_Sup[linha + 1, coluna] == 255 or Inf_Sup[linha - 1, coluna - 1] == 255 or Inf_Sup[linha + 1, coluna - 1] == 255 or Inf_Sup[linha - 1, coluna + 1] == 255 or Inf_Sup[linha + 1, coluna + 1]:
                    Inf_Sup[linha, coluna] = 255
                else:
                    Inf_Sup[linha, coluna] = 0           

    Dir_Esq = imagem.copy()

    for linha in range(1, linhas):
        for coluna in range(colunas-1, 0, -1):
            if (Dir_Esq[linha, coluna] == fraco):
                if Dir_Esq[linha, coluna + 1] == 255 or Dir_Esq[linha, coluna - 1] == 255 or Dir_Esq[linha - 1, coluna] == 255 or Dir_Esq[linha + 1, coluna] == 255 or Dir_Esq[linha - 1, coluna - 1] == 255 or Dir_Esq[linha + 1, coluna - 1] == 255 or Dir_Esq[linha - 1, coluna + 1] == 255 or Dir_Esq[linha + 1, coluna + 1]:
                    Dir_Esq[linha, coluna] = 255
                else:
                    Dir_Esq[linha, coluna] = 0
    
    Esq_Dir = imagem.copy()

    for linha in range(linhas-1, 0, -1):
        for coluna in range(1, colunas):
            if (Esq_Dir[linha, coluna] == fraco):
                if Esq_Dir[linha, coluna + 1] == 255 or Esq_Dir[linha, coluna - 1] == 255 or Esq_Dir[linha - 1, coluna] == 255 or Esq_Dir[linha + 1, coluna] == 255 or Esq_Dir[linha - 1, coluna - 1] == 255 or Esq_Dir[linha + 1, coluna - 1] == 255 or Esq_Dir[linha - 1, coluna + 1] == 255 or Esq_Dir[linha + 1, coluna + 1]:
                    Esq_Dir[linha, coluna] = 255
                else:
                    Esq_Dir[linha, coluna] = 0

    imagem_final = Sup_Inf + Inf_Sup + Dir_Esq + Esq_Dir

    imagem_final[imagem_final > 255] = 255

    return imagem_final

def rec_adiciona_posicao(visitado, x, maximo, posicoes, folha):

    if x >= 0 and x < maximo:

        visitado[x] = 1
        folha.append(posicoes[x])

        for ponto in folha: 
            
            livres = np.where(visitado == 0)[0]
            print(len(livres))

            if len(livres) > 0: 
                for i in livres:
                    if (visitado[i] == 0) and (abs(ponto[0]-posicoes[i][0]) <= 75) and (abs(ponto[1]-posicoes[i][1]) <= 75):
                        visitado[i] = 1
                        folha.append(posicoes[i])
    
    return folha


def acha_folhas(posicoes):

    visitado = np.zeros(posicoes.shape[0], dtype=int)

    maximo = posicoes.shape[0]

    folhas = []

    i = 0

    for x in range(0, maximo):

        print(0, maximo)

        if visitado[x] == 0:
            folha = []
            folha = rec_adiciona_posicao(visitado, x, maximo, posicoes, folha)
            folhas.append(np.array(folha))

    return np.array(folhas)

if __name__ == "__main__":
    arquivos = [join("./Folhas/", f) for f in listdir("./Folhas/") if isfile(join("./Folhas", f))]
    filtro_sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    inicio = time.time()

    for i in range(len(arquivos)):
        image_cru = cv2.imread(arquivos[i])

        imagem_borrada = gaussian_blur(image_cru, 9, verbose=False)

        gradiente, direcao_gradiente = deteccao_borda(imagem_borrada, filtro_sobel, converte_degraus=True, verbose=False)

        imagem_supremida = supressao_non_max(gradiente, direcao_gradiente, verbose=False)

        imagem_limitada = limiar(imagem_supremida, 5, 20, fraco=50, verbose=False)

        imagem = hipotese(imagem_limitada, fraco=50)

        pontos = np.argwhere(imagem==255)

        pontos = np.fliplr(pontos)
        print(f"Levou {time.time() - inicio}")

        folhas = acha_folhas(pontos)
        print(f"Levou {time.time() - inicio}")
        
        folha_atual = 0

        for folha in folhas:

            folha_atual += 1
            x, y, w, h = cv2.boundingRect(folha)
            x, y, w, h = x, y, w, h
            imagem_detectada = imagem[y:y+h, x:x+w]
            imagem_cortada = image_cru[y:y+h, x:x+w]
            
            cv2.imwrite(arquivos[i][:-4] + f"-{folha_atual}.png", imagem_cortada)
            cv2.imwrite(arquivos[i][:-4] + f"-{folha_atual}-P.png", imagem_detectada)