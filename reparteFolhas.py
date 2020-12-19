import pandas as pd
import cv2
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import warnings
import sys
warnings.simplefilter(action='ignore', category=DeprecationWarning)

if __name__ == "__main__":

    arquivos = [join("./Folhas/", f) for f in listdir("./Folhas/") if isfile(join("./Folhas", f))]
    folhas_canny = [join("./Folhas/FolhasCanny", f) for f in listdir("./Folhas/FolhasCanny") if isfile(join("./Folhas/FolhasCanny", f))]
    filtro_sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])

    inicio = time.time()

    for i in range(len(folhas_canny)):
        print(folhas_canny[i])
        canny = cv2.imread(folhas_canny[i])

        if len(canny.shape) == 3:
            canny = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)

        canny[np.where(canny > 30)] = 255
        canny[np.where(canny <= 30)] = 0

        x, y, w, h = cv2.boundingRect(np.fliplr(np.argwhere(canny==255)))
        x, y, w, h = x, y, w, h
        imagem_detectada = canny[y:y+h, x:x+w]
        #imagem_cortada = image_cru[y:y+h, x:x+w]

        plt.imshow(imagem_detectada, cmap='gray')
        plt.title("Limiar Aplicado")
        plt.show()