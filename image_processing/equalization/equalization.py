import cv2
import numpy as np


def equalization(image, histogram):
    '''
    Equalizes the histogram of the given image to improve contrast.

    The histogram equalization technique adjusts the distribution of intensities
    of pixels to span the entire color spectrum, potentially enhancing
    image details.    
    '''
    T = image.shape[0] * image.shape[1] # amount of pixels
    
    image_type = image.dtype
    L = np.iinfo(image_type).max + 1 # number of gray levels (256 for 8 bits)
    
    equalized_values = np.zeros(L)
    for j in range(1, L):    
        k = 0
        for i in range(j):
            k += histogram[i]
        k = k/T
        
        equalized_values[j] = k
    
    # Mapea os pixels da imagem original para os valores equalizados
    equalized_image = equalized_values[image]
    
    # Normalize the image to 0-255 range
    equalized_image = equalized_image * (L - 1)
    
    # Atribui o mesmo tipo da imagem original
    equalized_image = equalized_image.astype(image_type)
    
    return equalized_image