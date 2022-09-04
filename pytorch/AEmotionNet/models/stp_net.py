import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch.common.losses import *


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super().__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, data):
        x = data
        for name, module in self.submodule._modules.items():
            if len(module._modules.items()) != 0:
                for name2, module2 in module._modules.items():
                    x = module2(x)
            else:
                x = module(x)
        return x


class STPNet(nn.Module):
    def __init__(self, num_classes, depth):
        super().__init__()

        # TODO: Реализуйте архитектуру нейронной сети
        # так спектрограммы в батчах имеют разную длину, то необходимо использовать статистический пулинг

    def forward(self, data):
        output = self.net(data)
        return output
