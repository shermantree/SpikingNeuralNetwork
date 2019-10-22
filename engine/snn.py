import torch
from util import dataloader

class IdealSNN:
    def __init__(self, structure = [784, 10], pulsewidth = 100e-9, inputlength = 10e-6, cmem = 10e-12, vt = 0.5 ):
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

        self.weight = []
        self.vmem = []
        self.vout = []
        self.input  = None
        self.label = 0
        self.reset()

    def reset(self):

        self.vmem = []
        self.vout = []
        for layer in self.structure[1:]:
            self.vmem.append(torch.zeros((layer)))
            self.vout.append(torch.zeros((layer)))

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
        time = 0
        score = torch.zeros(len(self.vout[len(self.weight) - 1]))
        # repeat this for the time an input signal is given
        # do  W*x
        while time < self.inputlength:
            for layernum in range(len(self.weight)):
                if layernum == 0: # if first layer, use input signal
                    input = self.input[(int)(time/self.timesteplength)].view(self.structure[0], 1)

                    temp = torch.mm(self.weight[layernum], input).flatten()
                    #temp = torch.mul(input, self.weight[layernum])
                    self.vmem[layernum] = torch.add(self.vmem[layernum], temp)

                    for i in range(len(self.vmem[layernum])):

                        if self.vmem[layernum][i] > self.vt:
                            self.vout[layernum][i] = 1
                            self.vmem[layernum][i] = 0
                        elif self.vmem[layernum][i] < 0:
                            self.vmem[layernum][i] = 0
                else:
                    temp = torch.mul(self.vout[layernum - 1], self.weight[layernum])
                    self.vmem[layernum] = torch.add(self.vmem[layernum], temp)
                    for i in range(len(self.vmem[layernum])):
                        self.vout[layernum][i] = 0
                        if self.vmem[layernum][i] > self.vt:
                            self.vout[layernum][i] = 1
                            self.vmem[layernum][i] = 0
                        elif self.vmem[layernum][i] < 0:
                            self.vmem[layernum][i] = 0

            score = torch.add(score, self.vout[len(self.weight) - 1])
            time += self.timesteplength
        score *= self.vt

        score += self.vmem[len(self.weight) - 1]
        if score.max() == score[self.label]:
            return True, self.label
        return False, self.label


if __name__ == '__main__':
    d = dataloader.Dataloader()
    single = d.__getitem__(1)
    snn = IdealSNN()
    print ("here")
    w = 0.0001*torch.ones((10, 784))
    for i in range(784):
        w[2][i] = 1
    snn.set_weight_ideal([w])
    snn.load_input(single)
    print (snn.run())