import os
import torchvision.datasets as datasets
import numpy as np
import torch

class Dataloader:
    def __init__(self, pulsewidth = 100e-9, inputlength = 10e-6, train=False):

        self.mnist = datasets.MNIST('./data', download=True, train=train)
        self.timestep = (int)((inputlength+0.9*pulsewidth)/pulsewidth)


    def __getitem__(self, item):
        '''
        @brief make input into left_justified PWM
        '''
        data = np.array(self.mnist[item][0]).astype(np.float32) / 255.
        data = np.reshape(data, -1)
        signal = []
        for i in range(self.timestep):
            temp = torch.zeros((len(data)))
            signal.append(temp)

        for pixel in range(len(data)):
            for time in range((int)(self.timestep * data[pixel])):
                signal[time][pixel] = 1
        return signal, self.mnist[item][1]

    def __len__(self):
        return len(self.mnist)

if __name__ == "__main__":
    d = Dataloader()
    sig = d.__getitem__(1)[0]
