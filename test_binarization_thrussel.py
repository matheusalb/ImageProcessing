import cv2
import numpy as np
from image_processing.binarization import thrussel_binarization

if __name__=='__main__':
    image = cv2.imread('data/pout_equalized.bmp', cv2.IMREAD_GRAYSCALE)

    # Número de níveis de cinza (256 for 8 bits)
    L = np.iinfo(image.dtype).max + 1 

    histogram = cv2.calcHist(
        [image], # Lista de imagens
        [0], # Lista de Canais a serem considerados
        None, # Máscara, não utilizada
        [L], # Tamanho do histograma para cada canal 
        [0, L] # Intervalo de valores a serem considerados
    )
    
    binary_image  = thrussel_binarization(image)

    combined = np.hstack((image, binary_image))
    cv2.imshow('Imagem Original vs Binarizada', combined)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
