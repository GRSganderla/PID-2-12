import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
import time

if __name__ == "__main__":
    arquivos = [join("./Folhas/", f) for f in listdir("./Folhas/") if isfile(join("./Folhas", f))]
    folhas_canny = [join("./Folhas/FolhasCanny", f) for f in listdir("./Folhas/FolhasCanny") if isfile(join("./Folhas/FolhasCanny", f))]
    filtro_sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    inicio = time.time()

    for i in range(1):#len(folhas_canny)):
        
        canny = cv2.imread(folhas_canny[i])

        if len(canny.shape) == 3:
            canny = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)

        pontos = np.argwhere(canny==255)
        pontos = np.fliplr(pontos)

        folhas = []
        
        print(f"Levou {time.time() - inicio}")
        plt.imshow(canny, cmap='gray')
        plt.title("Deteccão de Borda por Sobel")
        plt.show()


    