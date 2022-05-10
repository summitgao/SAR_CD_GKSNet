import torch
import numpy as np
import scipy.io
import os


def train_epoch(model, optimizer, train_source_loader, train_target_loader,criterion, epoch, writer=None):
    model.train()
    num = len(train_source_loader)
    for i, ((data_source, label_source),(data_target,label_target)) in enumerate(zip(train_source_loader, train_target_loader)):
        model.zero_grad()
        optimizer.zero_grad()
        #print(data.dtype)
        data_source = data_source.to(torch.float32)
        label_source = label_source.to(torch.float32)
        label_source = torch.argmax(label_source,dim=1)
        data_source = data_source.cuda()
        label_source = label_source.cuda().long()

        data_target = data_target.to(torch.float32)
        label_target = label_target.to(torch.float32)
        label_target = torch.argmax(label_target,dim=1)
        data_target = data_target.cuda()
        label_target = label_target.cuda().long()


        
        result = model(source=data_source, target=data_target)

        loss = criterion(result, label_target)
        loss.backward()
        optimizer.step()
        if i%10==0:
            print('epoch {}, [{}/{}], loss {}'.format(epoch, i, num, loss))
            if writer is not None:
                writer.add_scalar('loss', loss.item(), epoch*num + i)

def test(model, test_source_loader,test_target_loader, criterion, epoch, writer=None):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        outlabel = []
        for i, ((data_source, label_source),(data_target,label_target)) in enumerate(zip(test_source_loader, test_target_loader)):
            data_source = data_source.to(torch.float32)
            label_source = label_source.to(torch.float32)
            label_source = torch.argmax(label_source,dim=1)
            data_source = data_source.cuda()
            label_source = label_source.cuda().long()

            data_target = data_target.to(torch.float32)
            label_target = label_target.to(torch.float32)
            label_target = torch.argmax(label_target,dim=1)
            data_target = data_target.cuda()
            label_target = label_target.cuda().long()
            
            result = model(source=data_source, target=data_target)

            test_loss += criterion(result, label_target).item()
            pred = result.data.max(1)[1]

            outlabel.extend(pred.cpu().numpy())
 
            correct += pred.eq(label_target.view_as(pred)).sum().item()

        outlabel = np.array(outlabel,dtype=np.int)

        test_ind = {}
        test_ind['Test_Outlabel'] = outlabel
        scipy.io.savemat(os.path.join(os.getcwd(), 'result/Test_Outlabel.mat'), test_ind)

    print('epoch {}, test loss {}, acc [{}/{}]'.format(epoch, test_loss, correct, len(test_target_loader.dataset)))
    return outlabel

