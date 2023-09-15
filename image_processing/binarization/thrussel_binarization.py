import cv2
import numpy as np

def thrussel_binarization(image):
    
    pixel_values = image.flatten().astype('int64')
    T_previous = None
    T = (np.max(pixel_values) + np.min(pixel_values))/2
    while T != T_previous:
        T_previous = T
        
        
        pixel_values_b = pixel_values[pixel_values <= T]
        pixel_values_w = pixel_values[pixel_values > T]
        
        print(pixel_values_b)
        print(pixel_values_w)
        Tb = (np.min(pixel_values_b) + np.max(pixel_values_b))/2
        Tw = (np.min(pixel_values_w) + np.max(pixel_values_w))/2
        
        T = (Tb + Tw)/2
    
    binary_image = (image >= T).astype(image.dtype) * 255
    return binary_image