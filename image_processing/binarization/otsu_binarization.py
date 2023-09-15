import cv2
import numpy as np

def otsu_iteration(image, histogram, t):
    # Assume o threshold t e calcula a variância intraclasse
    T = np.iinfo(image.dtype).max + 1 # Número total de níveis de cinza
    
    # 1 Passo: Calcular a probabilidade de ocorrência de cada nível de cinza
    N = image.shape[0] * image.shape[1] # Quantidade de pixels
        
    p = histogram / N
    # 2 passo 
    i_0_t = np.arange(t + 1)
    mi_t = np.dot(i_0_t, p[:t + 1])
    
    i_t_T = np.arange(t+1, T)
    mi_T = np.dot(i_t_T, p[t+1:])
    
    # 3 passo
    w_0 = np.sum(p[:t + 1])
    
    w_1 = np.sum(p[t+1:])
    # w_1 = 1 - w_0 
        
    # 4 passo
    mi_b = mi_t/w_0 if w_0 != 0 else 0 
    mi_w = mi_T/w_1 if w_1 != 0 else 0 
    
    # 5 passo
    
    i = np.arange(t+1)
    var_b = np.dot((i - mi_b)**2, p[:t+1]/w_0)
    
    i = np.arange(t+1, T) 
    var_t = np.dot((i-mi_w)**2, p[t+1:]/w_1)
    
    # 6 passo
    var_w = var_b*w_0 + var_t*w_1
    
    return var_w
    
def otsu_binarization(image, histogram):
    
    thresholds = np.arange(0, np.iinfo(image.dtype).max+1)    
    
    results = [otsu_iteration(image, histogram, t) for t in thresholds]
    
    best_threshold = np.nanargmin(results)
    
    binary_image = (image >= best_threshold).astype(image.dtype) * 255
    print(image)
    print(binary_image)
    print(best_threshold, results[best_threshold])
    return binary_image

