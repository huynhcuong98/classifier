import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import math
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, Sequential, Module

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        return x

class Depth_Wise(Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)



class Residual2(nn.Module):
  def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, num_channels, 3, 1, 1)
    self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1)
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.bn2 = nn.BatchNorm2d(num_channels)
    if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels, kernel_size=1,
                                   stride=strides)
    else:
            self.conv3 = None
  def forward(self, x):
    y = F.relu(self.bn1(self.conv1(x))) 
    y = self.bn2(self.conv2(y))
    if self.conv3:
            x = self.conv3(x)
    y +=x
    return F.relu(y)


class Network(nn.Module):
    def __init__(self, n_c):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(16)
        
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(1024, 512) # stride 1: 2304, 2:512
        self.fc2 = nn.Linear(512, n_c)

        self.residual1 = Residual2(3, 3, True)
        self.residual2 = Residual2(3, 8, True)

    def forward(self, x):
        # x = self.conv1(x) #48.48.32     #64.64.8
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 4)          
        # x = self.conv2(x) #24.24.32     #32.32.16
        # x = self.bn2(x)
        # x = F.relu(x)

        x= self.residual1(x) #3-8
        x = F.max_pool2d(x, 2)     
        x= self.residual2(x) # 8-16

        x = F.max_pool2d(x, 2)        
        x = self.conv3(x) #12.12.64     #16.16.16
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) #6.6.64  #8.8.16

        x = torch.flatten(x, 1) # 2304  #2048
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Network_arc(nn.Module):
    def __init__(self):
        super(Network_arc, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2048, 512) # stride 1: 2304, 2:512
        # self.fc2 = nn.Linear(1028, 512)

    def forward(self, x):
        x = self.conv1(x) #48.48.32     #64.64.8
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)          
        x = self.conv2(x) #24.24.32     #32.32.16
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2)        
        x = self.conv3(x) #12.12.64     #16.16.32
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2) #6.6.64  #8.8.32
        # x = self.dropout1(x)
        x = torch.flatten(x, 1) # 2304  #2048
        x = self.dropout1(x)
        #x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # x = F.log_softmax(x, dim=1)
        return x