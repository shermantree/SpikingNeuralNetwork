import os
import torchvision.datasets as datasets
import numpy as np
import torch

class Dataloader:
    def __init__(self, pulsewidth = 100e-9, inputlength = 10e-6):
        self.mnist_trainset = datasets.MNIST('./data', download=True, train=False)
        self.timestep = (int)(inputlength/pulsewidth)


    def __getitem__(self, item):
        '''
        @brief make input into left_justified PWM
        '''
        data = np.array(self.mnist_trainset[item][0]).astype(np.float32) / 255.
        data = np.reshape(data, -1)
        signal = []
        for i in range(self.timestep):
            temp = torch.zeros((len(data)))
            signal.append(temp)

        for pixel in range(len(data)):
            for time in range((int)(self.timestep * data[pixel])):
                signal[time][pixel] = 1
        return signal, self.mnist_trainset[item][1]

    def __len__(self):
        return len(self.mnist_trainset)

if __name__ == "__main__":
    d = Dataloader()
    d.__getitem__(1)
