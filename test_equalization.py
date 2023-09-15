from image_processing.equalization import equalization
import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_histogram(titles, histograms, L):
    
    fig, axs = plt.subplots(1, len(histograms), figsize=(12, 5))
    
    for i in range(len(histograms)):
        axs[i].stairs(histograms[i].ravel(), color='k')
        axs[i].set_title(titles[i])
        axs[i].set_xlim([0, L])
        axs[i].set_xlabel('Intensidade do Pixel')
        axs[i].set_ylabel('Número de Pixels')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    # Leitura da imagem como uma matriz de pixels
    image = cv2.imread('data/pout.bmp', cv2.IMREAD_GRAYSCALE)

    # Número de níveis de cinza (256 for 8 bits)
    L = np.iinfo(image.dtype).max + 1 

    histogram = cv2.calcHist(
        [image], # Lista de imagens
        [0], # Lista de Canais a serem considerados
        None, # Máscara, não utilizada
        [L], # Tamanho do histograma para cada canal 
        [0, L] # Intervalo de valores a serem considerados
    )
        
    equalized_image = equalization(image, histogram)
    equalized_histogram = cv2.calcHist(
        [equalized_image], # Lista de imagens
        [0], # Lista de Canais a serem considerados
        None, # Máscara, não utilizada
        [L], # Tamanho do histograma para cada canal 
        [0, L] # Intervalo de valores a serem considerados
    )
    
    combined = np.hstack((image, equalized_image))
    cv2.imshow('Imagem Original vs Equalizada', combined)    
    plot_histogram(
        ['Histograma da Imagem Original', 'Histograma da Imagem Equalizada'],
        [histogram, equalized_histogram],
        L
    )

    # cv2.imwrite('data/pout_equalized.bmp', equalized_image)
    
    print(np.sum((image - equalized_image), axis=-1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
