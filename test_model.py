import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import sampler, TensorDataset, Dataset
    
# CNN Model (2 conv layer)
# with bottle neck design
class CNN_1(nn.Module):
    def __init__(self, base=8):
        self.base = int(base)
        super(CNN_1, self).__init__()
        
        self.layer1_1 = nn.Sequential(
            nn.Conv3d(1, self.base, kernel_size=1),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base, kernel_size=3),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base*2, kernel_size=1),
            nn.InstanceNorm3d(self.base*2),
            nn.ReLU(),
            nn.AvgPool3d(2))
        self.layer1_2 = self.conv_block(self.base*2, self.base*4)
        self.layer1_3 = self.conv_block(self.base*4, self.base*6)
        self.layer1_4 = self.conv_block(self.base*6, self.base*8)
                
        self.layer2_1 = nn.Sequential(
            nn.Conv3d(1, self.base, kernel_size=1),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base, kernel_size=3),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base*2, kernel_size=1),
            nn.InstanceNorm3d(self.base*2),
            nn.ReLU(),
            nn.AvgPool3d(2))
        self.layer2_2 = self.conv_block(self.base*2, self.base*4)
        self.layer2_3 = self.conv_block(self.base*4, self.base*6)
        self.layer2_4 = self.conv_block(self.base*6, self.base*8)


        self.fc = nn.Sequential(
            nn.Linear(8192, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )

    def conv_block(self, in_shape, out_shape):
        out = nn.Sequential(
            nn.Conv3d(in_shape, in_shape, kernel_size = 1),
            nn.InstanceNorm3d(in_shape),
            nn.ReLU(),
            nn.Conv3d(in_shape, in_shape, kernel_size = 3),
            nn.InstanceNorm3d(in_shape),
            nn.ReLU(),
            nn.Conv3d(in_shape, out_shape, kernel_size = 1),
            nn.InstanceNorm3d(out_shape),
            nn.ReLU(),nn.AvgPool3d(2)
        )
        return out
    
    def forward(self, x1, x2):
        
        out1 = self.layer1_1(x1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)
        out1 = self.layer1_4(out1)
        out1 = out1.view(out1.size(0), -1)
        
        out2 = self.layer2_1(x2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = self.layer2_4(out2)
        out2 = out2.view(out2.size(0), -1)
        
        # fusion 
        out = torch.cat((out1, out2),1)
        # regression
        out = self.fc(out)
        return out
    
    
# CNN Model (2 conv layer)
# with bottle neck design
class CNN_2(nn.Module):
    def __init__(self, base=8):
        self.base = int(base)
        super(CNN_2, self).__init__()
        
        self.layer1_1 = nn.Sequential(
            nn.Conv3d(1, self.base, kernel_size=1),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base, kernel_size=3),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base*2, kernel_size=1),
            nn.InstanceNorm3d(self.base*2),
            nn.ReLU(),
            nn.AvgPool3d(2))
        self.layer1_2 = self.conv_block(self.base*2, self.base*4)
        self.layer1_3 = self.conv_block(self.base*4, self.base*6)
        self.layer1_4 = self.conv_block(self.base*6, self.base*8)
                
        self.layer2_1 = nn.Sequential(
            nn.Conv3d(1, self.base, kernel_size=1),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base, kernel_size=3),
            nn.InstanceNorm3d(self.base),
            nn.ReLU(),
            nn.Conv3d(self.base, self.base*2, kernel_size=1),
            nn.InstanceNorm3d(self.base*2),
            nn.ReLU(),
            nn.AvgPool3d(2))
        self.layer2_2 = self.conv_block(self.base*2, self.base*4)
        self.layer2_3 = self.conv_block(self.base*4, self.base*6)
        self.layer2_4 = self.conv_block(self.base*6, self.base*8)


        self.fc = nn.Sequential(
            nn.Linear(16384, 1024),
            nn.ReLU(),
#             nn.Dropout(p=0.2),
            nn.Linear(1024, 1)
        )
        
    def conv_block(self, in_shape, out_shape):
        out = nn.Sequential(
            nn.Conv3d(in_shape, in_shape, kernel_size = 1),
            nn.InstanceNorm3d(in_shape),
            nn.ReLU(),
            nn.Conv3d(in_shape, in_shape, kernel_size = 3),
            nn.InstanceNorm3d(in_shape),
            nn.ReLU(),
            nn.Conv3d(in_shape, out_shape, kernel_size = 1),
            nn.InstanceNorm3d(out_shape),
            nn.ReLU(),nn.AvgPool3d(2)
        )
        return out
    
    def forward(self, x1, x2):
        
        out1 = self.layer1_1(x1)
        out1 = self.layer1_2(out1)
        out1 = self.layer1_3(out1)
        out1 = self.layer1_4(out1)
        out1 = out1.view(out1.size(0), -1)
        
        out2 = self.layer2_1(x2)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = self.layer2_4(out2)
        out2 = out2.view(out2.size(0), -1)
        
        # fusion 
        out = torch.cat((out1, out2),1)
        print(out.size())
        # regression
        out = self.fc(out)
        return out    
    
    
