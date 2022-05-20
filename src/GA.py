import random as rnd
import numpy as np
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

# Class containing the chromosome (individual) structure
class chromosome(object):
    
    def __init__(self, targetHist, noZeroPosHist, numberOfGenes, minGrayLevel, maxGrayLevel, mut_rate, T_k, parent_1=None, parent_2=None, cross_point=None):
        
        self.genes      = []
        self.crossPoint = None
        self.__fitness  = None
        self.__opt_T    = 0
        self.__hist     = None
        self.__matrix   = None
        self.__term1    = None
        self.__term2    = None
        self.__term3    = None

        if parent_1 and parent_2:
            op = geneticOperation()
            self.genes, self.crossPoint = op.crossoverUniform(parent_1, parent_2, cross_point)
            op.mutate(self.genes, minGrayLevel, maxGrayLevel, self.__opt_T, mut_rate)

        else:
            dist = self.__generateUniformDistribution(noZeroPosHist, minGrayLevel, maxGrayLevel)
            for i in range(0, numberOfGenes):
                self.genes.append(gene(dist[i]))
        
        # self.genes.sort(key=lambda x: x.position)
        self.__fitness, self.__opt_T, self.__hist = self.calculateFitness(targetHist, noZeroPosHist, maxGrayLevel, minGrayLevel)

    def __generateUniformDistribution(self, noZeroPosHist, minGrayLevel, maxGrayLevel):

        dist1 =  np.random.uniform(minGrayLevel, maxGrayLevel, len(noZeroPosHist))
        dist = [int(round(j)) for j in dist1]
        return sorted(dist)
    
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

    def saveCurrentImage(self, targetHist, noZeroPosHist, targetMatrix, f_name, f_nameConf):

        self.__matrix = deepcopy(targetMatrix)
        newNoZeros = []
        for i in range(0, len(self.genes)):
            newNoZeros.append(self.genes[i].position)

        for i in range(0, len(noZeroPosHist)):
            ind = noZeroPosHist[i]
            pos = np.where(targetMatrix == ind);
            
            pos_x = pos[0]
            pos_y = pos[1]
            for j in range(0, len(pos_x)):
               self.__matrix[pos_x[j]][pos_y[j]] = newNoZeros[i]

        plt.figure()
        plt.subplot(121)
        plt.imshow(targetMatrix, cmap='Greys_r')
        plt.subplot(122)
        plt.imshow(self.__matrix, cmap='Greys_r')
        plt.tight_layout()
        plt.savefig(f_nameConf)
        plt.close()

        imageio.imwrite(f_name, self.__matrix)

    def saveTermFitness(self, file, mod):
        with open(file, mod) as fo:
            fo.write(str(self.__term1) + "\t")
            fo.write(str(self.__term2) + "\t")
            fo.write(str(self.__term3) + "\n")

    def calculateFitness(self, targetHist, noZeroPosHist, maxGrayLevel, minGrayLevel):

        hist = [0]*(maxGrayLevel+1)
        oldIdx = self.genes[-1].position
        for i in range(len(noZeroPosHist)-1, -1, -1):
            idx = self.genes[i].position
            if idx < minGrayLevel or idx > maxGrayLevel:
                print ('idx', idx)
                exit()
            ind = noZeroPosHist[i]
            if idx == oldIdx:
                hist[idx] += targetHist[ind]
            else:
                hist[idx] = targetHist[ind]
                oldIdx = self.genes[i].position
                
        h_dim = len(hist)
        total_pixel_number = np.sum(hist)
        T = total_pixel_number / h_dim
        opt_T= self.__optimalThreshold(hist, T, 1)

        return opt_T, hist

    def getFitness(self):
        return self.__fitness

    def getOpt_T(self):
        return self.__opt_T

    def getMatrix(self):
        return self.__matrix

# Class representing a gene of each individual      
class gene(object):
    
    def __init__(self, pos = 0):

        # Each gene is a bin of the histogram
        self.position = pos
    
    # Mutate the bin index
    def mutatePosition(self, minGrayLevel, maxGrayLevel, opt_T):

        if self.position <= opt_T:
            value = rnd.randint(minGrayLevel, opt_T)
        else:
            value = rnd.randint(opt_T, maxGrayLevel)
        
        if value > maxGrayLevel:
            value = maxGrayLevel
        elif value < minGrayLevel:
            value = minGrayLevel
        
        self.position = value

# Class containing the genetic operators (i.e., crossover and mutation)            
class geneticOperation(object):

    # Mutation of the genes
    def mutate(self, genes, minGrayLevel, maxGrayLevel, opt_T, rate):
        
        for i in range(0, len(genes)):
            if rnd.uniform(0, 1) < rate:
                genes[i].mutatePosition(minGrayLevel, maxGrayLevel, opt_T)
                
    def crossoverSingle(self, parent_1, parent_2):

        numberGenes = len(parent_1.genes)
        randNum = rnd.randint(0, numberGenes)
        
        list1 = deepcopy(parent_1.genes[0:randNum])
        list2 = deepcopy(parent_2.genes[randNum:numberGenes])

        return list1 + list2

    # Uniform and circular crossover
    def crossoverUniform(self, parent_1, parent_2, cross_point):

        # If the crossover point exists, it is used to mix the genes of the two parents
        if cross_point:
            numberGenes = len(parent_1.genes)
            list1 = []
            half = int(round( numberGenes / 2.0))
            if cross_point >= half:
                list1 = deepcopy(parent_2.genes[0:cross_point-half]) + deepcopy(parent_1.genes[cross_point-half:cross_point]) + deepcopy(parent_2.genes[cross_point:numberGenes])
            else:
                list1 = deepcopy(parent_1.genes[0:cross_point]) + deepcopy(parent_2.genes[cross_point:cross_point+half]) + deepcopy(parent_1.genes[cross_point+half:numberGenes])
            return list1, cross_point
        
        # If the crossover point does not exist, it is randomly selected to mix the genes of the two parents
        else:
            numberGenes = len(parent_1.genes)
            randNum = rnd.randint(0, numberGenes-1)
            list1 = []
            half = int(round(numberGenes / 2.0))

            if randNum >= half:
                list1 = deepcopy(parent_1.genes[0:randNum-half]) + deepcopy(parent_2.genes[randNum-half:randNum]) + deepcopy(parent_1.genes[randNum:numberGenes]) 
            else:
                list1 = deepcopy(parent_2.genes[0:randNum]) + deepcopy(parent_1.genes[randNum:randNum+half]) + deepcopy(parent_2.genes[randNum+half:numberGenes])
            return list1, randNum
        
# def calculateFitness(self, targetHist, noZeroPosHist):
#         marg = list(filter(lambda p: p > 0, np.ravel(targetHist)))
#         entropy = -np.sum(np.multiply(marg, np.log2(marg)))
#         area_of_coverage = noZeroPosHist
#         pixels = sum(targetHist)
#         brightness = scale = len(targetHist)
#         for index in range(0, scale):
#             ratio = targetHist[index] / pixels
#         brightness += ratio * (-scale + index)
#         if brightness == 255:
#             brightness = 1
#         else:
#             brightness = brightness / scale
#         ans = pixels + area_of_coverage + brightness - entropy
#         return ans
