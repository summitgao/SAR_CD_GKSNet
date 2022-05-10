import os
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
from dataset import MyDataset_source,MyDataset_target

#from knowledgenet import ResNeXt
from net import ResNeXt


from train import train_epoch, test
import scipy.io as sio



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__=="__main__":



    train_source_loader = torch.utils.data.DataLoader(MyDataset_source('train'), batch_size=128, shuffle=True)

    test_source_loader = torch.utils.data.DataLoader(MyDataset_source('test'), batch_size=100)

    train_target_loader = torch.utils.data.DataLoader(MyDataset_target('train'), batch_size=128, shuffle=True)

    test_target_loader = torch.utils.data.DataLoader(MyDataset_target('test'), batch_size=100)

    net = ResNeXt(2)

    net.cuda()
    optimizer = optim.Adam(net.parameters(), weight_decay=1e-5, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss().cuda()

    log_path = './logs/'
    writer = SummaryWriter(log_path)

    epoch_num = 6 
    lr0 = 1e-3
    for epoch in range(epoch_num):
        current_lr = lr0 / 2**int(epoch/50)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        train_epoch(net, optimizer, train_source_loader, train_target_loader,criterion, epoch, writer=writer)
    out=test(net,test_target_loader,test_target_loader, criterion, epoch, writer=writer)


