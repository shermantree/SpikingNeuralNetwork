import torch


class IdealSNN:
    def __init__(self, structure = [728, 10], pulsewidth = 100e-9, inputlength = 10e-6, cmem = 10e-12, vt = 0.5 ):

        self.pulsewidth = pulsewidth
        self.structure = structure
        self.vt = vt
        self.inputlength = inputlength

        self.weight = []

        self.reset()


    def reset(self):

        self.vmem = []
        self.vout = []
        for layer in self.structure[1:]:
            self.vmem.append(torch.zeros((1,layer)))
            self.vout.append(torch.zeros((1, layer)))


    def set_weight_ideal(self, weight):
        pass


