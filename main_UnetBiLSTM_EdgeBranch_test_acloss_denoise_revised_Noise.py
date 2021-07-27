from __future__ import print_function
import argparse, os, time
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from dense_atj import Dense
# from dense_201 import Dense_201
# from dense_121_reduced import Dense
from tqdm import tqdm

from dataset_LSTM import DatasetFromHdf5
from utils import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from metrics import dice_coef, batch_iou, mean_iou, iou_score
from Unet_bidrecionalLSTM_EdgeBranch_test_revised_3con2_Noise import Unet_BiCLSTM
# from Unet_CLSTM import Unet_CLSTM
# from convlstm import ConvLSTM
# from Vgg16 import Vgg16
# from matrices import *
#from reg_contrast import reg_contrast
import pdb

# Training settings
parser = argparse.ArgumentParser(description="Pytorch Unet")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
# parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")
parser.add_argument("--traindata", default="/cvdata/xuelu/Internship_PARC/datasets/VesselNN/H5_files/train/train_low_LSTM_data_patches.h5", type=str, help="Training datapath")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate, Default=0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
parser.add_argument("--nt", type=int, default=3, help="Time step number for each sequence")
parser.add_argument("--aug", action="store_true", help="Use aug?")
# parser.add_argument("--resume", default="/tilde/xli/3D_Vessel_Segmentation/methods/Ours/model/Unet_BiCLSTM_OneCellTrue/model_epoch_107.pth", type=str, help="Path to checkpoint, Default=None")
parser.add_argument("--resume", default="/cvdata/xuelu/Internship_PARC/methods/Ours/model/Unet_BiCLSTM_EdgeBranch_test_revised_Noise/model_epoch_20.pth", type=str, help="Path to checkpoint, Default=None")
# parser.add_argument("--resume", default="", type=str, help="Path to checkpoint, Default=None")
#parser.add_argument("--ID", default="_OUTDOOR", type=str, help="dev ID")
parser.add_argument("--ID", default="EdgeBranch_test_acloss_denoise_revised_Noise", type=str, help="dev ID")
parser.add_argument("--activation", default="no_relu", type=str, help="activation of A and J")

parser.add_argument("--start-epoch", default=1, type = int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.0001, help="Clipping Gradients, Default=0.01")
parser.add_argument("--threads", type=int, default=3, help="Number of threads for data loader to use, Default=1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default=0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default=1e-4")
parser.add_argument("--pretrained", default="", type=str, help='path to pretrained model, Default=None')
parser.add_argument("--model", default="Unet_BiCLSTM", type=str, help="unet or drrn or runet")

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    save_path = os.path.join('.', "model", "{}_{}".format(opt.model, opt.ID))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # cuda = opt.cuda
    # if cuda  and not torch.cuda.is_available():
    #     raise Exception("No GPU found, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    # opt.seed = 4222

    print("Random Seed: ", opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # pdb.set_trace()
    train_set = DatasetFromHdf5(file_path=opt.traindata)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


    # pdb.set_trace()

    print("===> Building model")
    if opt.model == 'dense_121':
        model = Dense()
    elif opt.model == 'Unet_BiCLSTM':
        print("===> Model {}".format(opt.model))
        model = Unet_BiCLSTM()
        # pdb.set_trace()
    elif opt.model == 'Unet_CLSTM':
        # model = Unet_CLSTM(input_channels=3, hidden_channels=[2,2,2])
        model = ConvLSTM()
    else:
        raise ValueError("no known model of {}".format(opt.model))
    criterion = nn.BCEWithLogitsLoss().cuda()

    print("===> Setting GPU")
    model = torch.nn.DataParallel(model).cuda()
    # torch.nn.DataParallel(model).cuda()
    criterion = criterion.cuda()
    # #        global vgg
    #     vgg = Vgg16(requires_grad=False).cuda()
    # #        global vgg

    #    pdb.set_trace()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("===> loading checkpoint: {}".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("===> no checkpoint found at {}".format(opt.resume))

    # optionally copy weights from a checkpoint
#     if opt.pretrained:
#         if os.path.isfile(opt.pretrained):
#             pretrained_dict = torch.load(opt.pretrained)['model'].state_dict()
#             print("===> load model {}".format(opt.pretrained))
#             model_dict = model.state_dict()
#             # filter out unnecessary keys
#             pretrained_dict = {k: v for  k,v in pretrained_dict.items() if k in model_dict}
#             print("\t...loaded parameters:")
#             for k,v in pretrained_dict.items():
#                 print("\t\t+{}".format(k))
#             model_dict.update(pretrained_dict)
#             model.load_state_dict(model_dict)
# #            opt.start_epoch = torch.load(opt.pretrained)['epoch']+1 #69: plus guided filter
#             # weights = torch.load(opt.pretrained)
#             # model.load_state_dict(weights['model'].state_dict())
#         else:
#             print("===> no model found at {}".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #    optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, momentum = opt.momentum)

    print("===> Training")
    writer = SummaryWriter(log_dir='./records/{}_{}/'.format(opt.model, opt.ID))
    for epoch in range(opt.start_epoch, opt.nEpochs + opt.start_epoch):
        # pdb.set_trace()
        train(training_data_loader, optimizer, model, criterion, epoch, writer)
        save_checkpoint(model, epoch, save_path)
        # os.system("python eval.py --cuda --model=model/model_epoch_{}.pth".format(epoch))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch  // opt.step))
    return lr

def active_contour_loss(ori_Input, ori_Output, beta=1, is_biclass=0):
    # pdb.set_trace()
    ori_Output = ori_Output.sigmoid()
    B, C, W, H = ori_Output.shape 
    loss1_total = 0.0
    # loss1_total = loss1_total.cuda()
    loss2_total = 0.0
    # loss2_total = loss2_total.cuda()

    patchSize_w = 64
    patchSize_h = 64

    for i in range(0, W, patchSize_w):
        for j in range(0, W, patchSize_w):
            Input = ori_Input[:,:,i:i+patchSize_w,j:j+patchSize_h]
            Output = ori_Output[:,:,i:i+patchSize_w,j:j+patchSize_h]
    # pdb.set_trace()
    
    # Input = Input*255
            loss1 = torch.sum(Output*((Input-torch.sum(Input*Output)/torch.sum(Output))**2))
            if is_biclass == 0:
                loss2 = beta*torch.sum((1-Output)*((Input-torch.sum(Input*(1-Output))/torch.sum(1-Output))**2))
            else:
                loss2 = beta*torch.sum((1-Output)*((Output*Input-torch.sum(Output*Input*(1-Output))/torch.sum(1-Output))**2))
            loss1_total = loss1_total + loss1
            loss2_total = loss2_total + loss2
    # pdb.set_trace()
    return (loss1_total+loss2_total)/B/64/64
def train(training_data_loader, optimizer, model, criterion, epoch, writer):
# lr policy
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    total_iter = (epoch - 1) * len(training_data_loader)
    for iteration, batch in tqdm(enumerate(training_data_loader, 1), total = len(training_data_loader)):
        input, target, edges, noise = Variable(batch[0]).cuda(), Variable(batch[1]).cuda(), Variable(batch[2]).cuda(), Variable(batch[3]).cuda()
        # pdb.set_trace()
        # input_forward = input[:,]
        # input_reverse = 
        output_pre, output, output_lat, output_edg, fseg, fedge = model(input, noise)
        # pdb.set_trace()
        # output = output1[-1]
        # pdb.set_trace()
        # target = target.unsqueeze(2)
        # pdb.set_trace()
        loss_seg = criterion(output_pre, target[:,0,:,:].unsqueeze(1))+criterion(output, target[:,1,:,:].unsqueeze(1))+criterion(output_lat
            , target[:,2,:,:].unsqueeze(1))
        loss_edg = criterion(output_edg, edges[:,1,:,:].unsqueeze(1))
        # loss = criterion(output, target[:,1,:,:].unsqueeze(1))
        output_total = torch.cat([output_pre, output, output_lat], dim=1)
        loss_ac = 0.001*active_contour_loss(input[:,:,1,:,:], output_total)/opt.batchSize
        loss_denoise = 0.001*torch.sum(torch.abs(output_edg.sigmoid()))/256/256/opt.batchSize
        # loss_ac = active_contour_loss(input[:,1,1,:,:].unsqueeze(1), output)
        loss_noise = 0.001*torch.sum(fseg.relu())+torch.sum(fedge.relu())/opt.batchSize/32/32/16
        # pdb.set_trace()
        # loss = loss_seg+loss_edg+loss_denoise
        loss = loss_seg+loss_edg+loss_ac+loss_denoise+loss_noise
        iou = iou_score(output, target[:,1,:,:].unsqueeze(1))
                 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # writer.add_scalar('data/loss', loss, total_iter)
        # writer.add_scalar('data/iou', iou, total_iter)

        # if iteration%2 == 0:
        #     n = np.random.randint(opt.nt)
        #     display = vutils.make_grid(input[:,n,:,:,:], normalize=False, scale_each=True)
        #     writer.add_image('Image/train_input', display, total_iter)
            
        #     display = vutils.make_grid(target[:,n,:,:,:], normalize=False, scale_each=True)
        #     writer.add_image('Image/train_target', display, total_iter)

        #     display = vutils.make_grid(output[:,n,:,:,:].sigmoid(), normalize=False, scale_each=True)
        #     writer.add_image('Image/train_output', display, total_iter)

           
        #     # psnr_run = 100
        # print("===> Epoch[{}]({}/{}): Loss: {:.10f}, Loss_ac: {:.10f},  loss_denoise: {:.10f}, iou: {:.10f} ".format(epoch, iteration, len(training_data_loader), loss, loss_ac, loss_denoise, iou))
        print("===> Epoch[{}]({}/{}): Loss: {:.10f}, Loss_ac: {:.10f},  loss_denoise: {:.10f}, loss_noise: {:.10f}, iou: {:.10f} ".format(epoch, iteration, len(training_data_loader), loss, loss_ac, loss_denoise, loss_noise, iou))

        # total_iter += 1 

if __name__ == "__main__":
    main()
