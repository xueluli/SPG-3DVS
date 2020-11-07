import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torchvision.models as models
from torch.autograd import Variable
from biconvlstm_EdgeBranch import ConvBLSTM 
from dense_121_edge_test_revised_3con import Dense
# from dense_121_edge_test_freeze import Dense
import pdb


class Unet_BiCLSTM(nn.Module):
    def __init__(self):
        super(Unet_BiCLSTM, self).__init__()
        # pdb.set_trace(inp)
        self.feature_ext = Dense()
        # self.BiCLSTM_seg = ConvBLSTM(in_channels=24, hidden_channels=[32,32], kernel_size=[(3, 3),(3, 3)], num_layers=2, batch_first=True)
        self.BiCLSTM_seg = ConvBLSTM(in_channels=16, hidden_channels=[16], kernel_size=[(3, 3)], num_layers=1, batch_first=True)
        self.BiCLSTM_edge = ConvBLSTM(in_channels=16, hidden_channels=[16], kernel_size=[(3, 3)], num_layers=1, batch_first=True)
        self.ConvPre = nn.Conv2d(32, 1, kernel_size=1)
        self.Conv = nn.Conv2d(64, 32, kernel_size=1)
        self.Relu = nn.ReLU()
        self.Conv1 = nn.Conv2d(32, 1, kernel_size=1)
        # self.Relu1 = nn.ReLU()
        # self.Conv2 = nn.Conv2d(32, 1, kernel_size=1)
        self.ConvLat = nn.Conv2d(32, 1, kernel_size=1)

        self.ConvE = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, input, noise):
        # internal_state = []
        # # outputs = []
        # if iteration == 429:
        #     pdb.set_trace()
        input_tensor1 = torch.tensor([]).cuda()
        input_tensor2 = torch.tensor([]).cuda()
        T = input.shape[1]
        for step in range(T):
            x = input[:,step,:,:,:]
            
            x1, x2 = self.feature_ext(x)
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            input_tensor1 = torch.cat((input_tensor1,x1),dim=1)
            input_tensor2 = torch.cat((input_tensor2,x2),dim=1)
        
        # pdb.set_trace()
        fseg, fedge = self.feature_ext(noise)
        # pdb.set_trace()
        y_previous, y_cat, y_latter = self.BiCLSTM_seg(input_tensor1[:,0:T//2+1,:,:,:],input_tensor1[:,T//2:,:,:,:])
        z_previous, z_cat, z_latter = self.BiCLSTM_edge(input_tensor2[:,0:T//2+1,:,:,:],input_tensor2[:,T//2:,:,:,:])
        #     # pdb.set_trace()
        #     for i in range(self.num_layers):
        #         # all cells are initialized in the first step
        #         name = 'cell{}'.format(i)
        #         if step == 0:
        #             bsize, _, height, width = x.size()
        #             (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
        #                                                      shape=(height, width))
        #             internal_state.append((h, c))

        #         # do forward
        #         (h, c) = internal_state[i]
        #         x, new_c = getattr(self, name)(x, h, c)
        #         internal_state[i] = (x, new_c)
        #     # pdb.set_trace()
        #     # x = self.conv2(x)
        #     outputs = torch.cat((outputs, x), dim=1) 
        #     # only record effective steps
        #     # if step in self.effective_step:
        #     #     outputs.append(x)
            # outputs.append(x)
        pdb.set_trace()
        yz_cat = torch.cat((y_cat, z_cat), dim=1)
        yz_previous = torch.cat((y_previous, z_previous), dim=1)
        yz_latter = torch.cat((y_latter, z_latter), dim=1)
        # pdb.set_trace()
        yz_previous = self.ConvPre(yz_previous)
        y_cat = self.Conv(yz_cat)
        y_cat = self.Relu(y_cat)
        y_cat = self.Conv1(y_cat)
        # pdb.set_trace()
        # y_cat = self.Relu1(y_cat)
        # y_cat = self.Conv2(y_cat)
        yz_latter = self.ConvLat(yz_latter)

        z_cat = self.ConvE(z_cat)
        return yz_previous, y_cat, yz_latter, z_cat, fseg, fedge