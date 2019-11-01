import torch
from util import dataloader
import numpy as np
import matplotlib.pyplot as plt
from util import weightloader
error = 0
class IdealSNN:
    def __init__(self, structure = [784, 10], pulsewidth = 100e-9, inputlength = 10e-6, cmem = 10e-12, vt = 0.5, readvoltage =1, spikeenergy=10e-12 ):
        '''

        :param structure:
        :param pulsewidth:
        :param inputlength:
        :param cmem:
        :param vt:
        '''
        self.pulsewidth = pulsewidth
        self.timesteplength = pulsewidth
        self.structure = structure
        self.vt = vt
        self.inputlength = inputlength
        self.readvoltage = readvoltage
        self.spikeenergy = spikeenergy
        self.energyneuron = 0
        self.energysynapse = 0
        self.weight = []
        self.vmem = []
        self.vout = []
        self.input  = None
        self.label = 0
        self.cmem = cmem
        self.reset()
        self.minconductance = 0


    def load_input(self, input):
        self.input = input[0]
        self.label = input[1]

    def set_weight_ideal(self, weightmatrix):
        '''
        :param weightmatrix: list of weight matrix
        :return:
        '''
        assert (len(weightmatrix) == len(self.structure) - 1)
        for w in weightmatrix:
            self.weight.append(torch.tensor(w).type(torch.FloatTensor))

    def run(self):
        '''
        @brief run snn for single image data
        :return: classification result (true/false, label)
        '''
        global error
        time = 0
        score = torch.zeros(len(self.vout[len(self.weight) - 1]))
        # repeat this for the time an input signal is given
        # do  W*x
        while time < self.inputlength:
            for layernum in range(len(self.weight)):
                if layernum == 0: # if first layer, use input signal
                    input = self.input[(int)(time/self.timesteplength)].view(self.structure[0], 1)
                    temp = torch.mm(self.weight[layernum], input).flatten()
                    self.vmem[layernum] = torch.add(self.vmem[layernum], self.pulsewidth*temp/self.cmem)
                    temp = temp.detach().numpy()
                    #self.energysynapse += sum(input.flatten().detach().numpy())*self.minconductance*self.readvoltage*self.pulsewidth
                    self.energysynapse += sum(np.absolute(temp))*self.readvoltage*self.pulsewidth
                    for i in range(len(self.vmem[layernum])):

                        if self.vmem[layernum][i] > self.vt:
                            self.vout[layernum][i] = (int)(self.vmem[layernum][i]/self.vt)
                            self.vmem[layernum][i] = 0
                            self.energyneuron += 0.5*self.cmem*self.vt*self.vt + self.spikeenergy
                        elif self.vmem[layernum][i] < 0:
                            self.vmem[layernum][i] =0
                else:
                    temp = torch.mm(self.weight[layernum], self.vout[layernum - 1].view(128, 1)).flatten()
                    self.vmem[layernum] = torch.add(self.vmem[layernum], self.pulsewidth*temp/self.cmem)
                    temp = temp.detach().numpy()
                    #self.energysynapse += sum(self.vout[layernum - 1].flatten().detach().numpy()) * self.minconductance * self.readvoltage * self.pulsewidth
                    self.energysynapse += sum(np.absolute(temp))*self.readvoltage*self.pulsewidth
                    for i in range(len(self.vmem[layernum])):
                        self.vout[layernum][i] = 0
                        if self.vmem[layernum][i] > self.vt:
                            self.vout[layernum][i] = (int)(self.vmem[layernum][i]/self.vt)
                            self.vmem[layernum][i] = 0
                            self.energyneuron += 0.5 * self.cmem * self.vt * self.vt + self.spikeenergy
                        elif self.vmem[layernum][i] < 0:
                            self.vmem[layernum][i] =0
            score = torch.add(score, self.vout[len(self.weight) - 1])
            time += self.timesteplength
        score *= self.vt

        score += self.vmem[len(self.weight) - 1]
        if score.max() != score[self.label]:
            self.retrain(self.input, self.label)
            error += 1




    def reset(self):

        self.vmem = []
        self.vout = []
        for layer in self.structure[1:]:
            self.vmem.append(torch.zeros((layer)))
            self.vout.append(torch.zeros((layer)))

    def retrain(self, input, label):

        print (len(input), label)
        for i in range(len(input)):
            input[i] = input[i].detach().numpy()
        input = np.array(input)
        input = np.transpose(input)
        input = input.sum(axis=1) # input is in [784] shapex
        print (len(self.weight))
        change = 1e-9 / 20
        for o in range(10):
            if o == label:
                for i in range(784):
                    self.weight[-1][o][i] += input[i]*change
            else:
                for i in range(784):
                    self.weight[-1][o][i] -= input[i] * change




if __name__ == '__main__':
    d = dataloader.Dataloader()
    single = d.__getitem__(1)
    snn = IdealSNN()
    print ("here")


    weight = weightloader.WeightloaderSinglelayer("weight.csv","RRAM.csv")
    snn.set_weight_ideal(weight.get_weight())

    data = dataloader.Dataloader(train=True)
    errorlog = []
    for i, item in enumerate(data):


        snn.reset()
        snn.load_input(item)
        snn.run()
        print("score", i, error)
        errorlog.append(error/(i+1))
        if i > 300: break
    np.savetxt("weight_retrain.csv", snn.weight[0].detach().numpy(), delimiter=',')
    plt.plot(errorlog)

    for y in range(1, 15, 1):
        plt.axhline(color='gray', y=y/100, linewidth=0.5)
    for x in range(100, 2000, 100):
        plt.axvline(color='gray', x=x, linewidth=0.5)


    #plt.show()