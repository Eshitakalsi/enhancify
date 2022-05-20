import numpy as np
import scipy.misc
import math
import os
import imageio

# Class containing image processing functions
class processing(object):
    
    # Loading the input image characterized an underlying bimodal histogram
    def loadImage(self, target_img_name, pathOut):
    
        image = imageio.imread(target_img_name)

        maxValue = np.max(image)

        hist, _ = np.histogram(image, bins = maxValue+1)

        hist[0] = 0

        posNoZeros = list(np.nonzero(hist)[0])

        imageio.imwrite(pathOut + os.sep + 'imageOriginal.png', image)
        
        # Initializing the threshold to the global mean
        h_dim = len(hist)
        total_pixel_number = np.sum(hist)
        T = total_pixel_number / h_dim
        
        T_k =  self.__optimalThreshold(hist, T, 1)
            
        return image, len(posNoZeros), hist, posNoZeros, maxValue, T_k

    #Iterative Optimal Threshold Selection algorithm
    def __optimalThreshold(self, hist_vals, thresh, optimal_thres):        
        G1 = []
        G2 = []
        for i in hist_vals:
            if(i > thresh):
                G1.append(i) 
            else:
                G2.append(i)
                
        mean1 = np.sum(G1) / len(G1)
        mean2 = np.sum(G2) / len(G2)
        
        main_T = (mean1 + mean2)/2
        
        if abs(main_T - thresh) <= optimal_thres:
            return main_T
        else:
            return self.__optimalThreshold(hist_vals, main_T, optimal_thres)