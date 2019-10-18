import torch


class IdealSNN:
    def __init__(self, structure = [728, 10], pulsewidth = 100e-9, inputlength = 10e-6, cmem = 10e-12, vt = 0.5 ):
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
        self.reset()


    def reset(self):

        self.vmem = []
        self.vout = []
        for layer in self.structure[1:]:
            self.vmem.append(torch.zeros((layer)))
            self.vout.append(torch.zeros((layer)))

    def load_input(self, input):
        self.input = input

    def set_weight_ideal(self, weightmatrix):
        '''
        :param weightmatrix: list of weight matrix
        :return:
        '''
        assert (len(weightmatrix) == len(self.structure) - 1)
        for w in weightmatrix:
            self.weight.append(torch.tensor(w))


    def run(self):
        '''
        @brief run snn for single image data
        :return: classification result
        '''
        time = 0
        # repeat this for the time an input signal is given
        # do x*W (not W*x)
        while time < self.inputlength:
            for layernum in range(len(self.weight)):
                if layernum == 0: # if first layer, use input signal
                    temp = torch.mul(self.input[(int)(time/self.timesteplength)], self.weight[layernum])
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

            time += self.timesteplength

        print (self.o)
        return

if __name__ == '__main__()':
    snn = IdealSNN()
    snn.run()