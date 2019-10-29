import os
import torchvision.datasets as datasets
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
class WeightloaderSinglelayer:
    def __init__(self, weightfile, synapsefile = False ):
        self.weight = []


        if synapsefile:
            synapsedata  = pd.read_csv(synapsefile)
            conductance = []
            conductance_match = []
            # get number of states
            statenum = len(synapsedata['pulse'])
            print (statenum)
            min_conductance = synapsedata.iloc[0]['conductance']
            self.minconductance = min_conductance
            for i in range(statenum):
                conductance.append(synapsedata.iloc[i]['conductance'] - min_conductance)
            print (conductance)
            # get floating point weights
            weightfile = np.genfromtxt(weightfile, delimiter=',')  # weightfile[o][i]
            flattenedweight = weightfile.flatten()

            rng = flattenedweight  # deterministic random data
            #a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
            _ = plt.hist(rng, bins='auto')  # arguments are passed to np.histogram
            #plt.title("Histogram with 'auto' bins")
            #plt.show()


            self.weight.append(weightfile)
            self.weight = np.array(self.weight)

            # get maximum absolute value of weight
            max_weight = np.abs(flattenedweight.min())
            if flattenedweight.max() > max_weight:
                max_weight = flattenedweight.max()


            # match conductance to weight
            subtract = 0
            multiply = max_weight / (conductance[-1] - conductance[0]
                                     + 0.5*(conductance[-1] - conductance[-2]))
            for c in conductance:
                conductance_match.append( multiply*(c - subtract))
            print (conductance_match)
            for i in range(len(self.weight[0])):
                print (i)
                for j in range(len(self.weight[0][0])):
                    # find closest conductance_match
                    closestindex = 0
                    mindifference = 10000
                    for idx in range(len(conductance_match)):
                        if np.abs(conductance_match[idx] - np.abs(self.weight[0][i][j])) < mindifference:
                            mindifference = np.abs(conductance_match[idx] - np.abs(self.weight[0][i][j]))
                            closestindex = idx
                    if self.weight[0][i][j] < 0:
                        self.weight[0][i][j] = -1*conductance[closestindex]
                    else:
                        self.weight[0][i][j] = 1 * conductance[closestindex]


        else:
            weightfile = np.genfromtxt(weightfile, delimiter=',')  # weightfile[o][i]
            self.weight.append(weightfile)
            print(weightfile.shape)
            self.weight = np.array(self.weight)
            self.weight = self.weight * 0.001
            self.minconductance = 0


    def get_weight(self):
        return self.weight
    def get_minconductance(self):
        return self.minconductance

class WeightloaderMultilayer:
    def __init__(self, weightfile1, weightfile2, synapsefile = False ):
        self.weight = []


        if synapsefile:
            synapsedata  = pd.read_csv(synapsefile)
            conductance = []
            conductance_match = []
            # get number of states
            statenum = len(synapsedata['pulse'])
            min_conductance = synapsedata.iloc[0]['conductance']
            self.minconductance = min_conductance
            for i in range(statenum):
                conductance.append(synapsedata.iloc[i]['conductance'] - min_conductance)
            # get floating point weights
            tempweight = []
            weightfile1 = np.genfromtxt(weightfile1, delimiter=',')  # weightfile[o][i]
            self.weight.append(weightfile1)
            weightfile2 = np.genfromtxt(weightfile2, delimiter=',')  # weightfile[o][i]
            self.weight.append(weightfile2)
            #self.weight = np.array(self.weight)

            flattenedweight1 = weightfile1.flatten()
            flattenedweight2 = weightfile2.flatten()

            # get maximum absolute value of weight
            max_weight = np.abs(flattenedweight1.min())
            if flattenedweight1.max() > max_weight:
                max_weight = flattenedweight1.max()
            if flattenedweight2.max() > max_weight:
                max_weight = flattenedweight2.max()


            # match conductance to weight
            subtract = 0
            multiply = max_weight / (conductance[-1] - conductance[0]
                                     + 0.5*(conductance[-1] - conductance[-2]))
            for c in conductance:
                conductance_match.append( multiply*(c - subtract))
            for w in range(len(self.weight)):
                print (w)
                for i in range(len(self.weight[w])):
                    for j in range(len(self.weight[w][0])):
                        # find closest conductance_match
                        closestindex = 0
                        mindifference = 10000
                        for idx in range(len(conductance_match)):
                            if np.abs(conductance_match[idx] - np.abs(self.weight[w][i][j])) < mindifference:
                                mindifference = np.abs(conductance_match[idx] - np.abs(self.weight[w][i][j]))
                                closestindex = idx
                        if self.weight[w][i][j] < 0:
                            self.weight[w][i][j] = -1*conductance[closestindex]
                        else:
                            self.weight[w][i][j] = 1 * conductance[closestindex]


        else:
            weightfile1 = np.genfromtxt(weightfile1, delimiter=',')  # weightfile[o][i]
            self.weight.append(weightfile1)
            weightfile2 = np.genfromtxt(weightfile2, delimiter=',')  # weightfile[o][i]
            self.weight.append(weightfile2)
            self.weight = np.array(self.weight)
            self.weight = self.weight * 0.001
            self.minconductance = 0


    def get_weight(self):
        return self.weight

    def get_minconductance(self):

        return self.minconductance


if __name__ == "__main__":
    WeightloaderSinglelayer("weight.csv")
