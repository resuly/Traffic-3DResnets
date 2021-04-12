import torch
import torch.nn as nn

# https://faroit.github.io/keras-docs/1.2.1/layers/convolutional/
# [nb_residual_unit] Residual Units

class MinMaxNormalization(object):
    '''MinMax Normalization --> [-1, 1]
       x = (x - min) / (max - min).
       x = x * 2 - 1
    '''

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        # print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class BasicBlock(nn.Module):

    def __init__(self):
        super(BasicBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        x = self.bn1(x) # deafult is None at source code
        x = self.relu(x)
        x = self.conv(x)
        x += residual
        
        return x
    
def get_residual_unit(nb_residual_unit):
    block = BasicBlock
    layers = []
    for _ in range(nb_residual_unit):
        layers.append(block())

    residual_unit = nn.Sequential(*layers)
    return residual_unit

class SingleResnet(nn.Module):
    # one of C, P, T
    def __init__(self, in_channels, params):
        super(SingleResnet, self).__init__()
        self.in_channels = in_channels
        self.params = params
        # Conv1
        # conv1 = Convolution2D(nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        # [nb_residual_unit] Residual Units
        self.residual_unit = get_residual_unit(params.nb_residual_unit)
        
        # Conv2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=params.n_flow, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = x.view(-1, self.in_channels, self.params.map_height, self.params.map_width)
        x = self.conv1(x)
        x = self.residual_unit(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        return x