from os import write
from torch import mode, transpose
import torchvision.transforms as Transforms
import torch
import numpy as np
import glob
from mbhk_dataloader import jia_signal_eye
from mixnet import MixNet
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import time
import datetime
import torch.nn as nn
from tensorboardX import SummaryWriter

# 初始化tensornoard
writer = SummaryWriter("run/jia_signal_eye_24_24")

ttrans = Transforms.Compose([
        Transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.RandomCrop(100),
        # transforms.RandomResizedCrop(112),
        Transforms.ColorJitter(brightness=0.5),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=0, ),
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
    ])
vaild_ttrans = Transforms.Compose([
        # Transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.RandomCrop(100),
        # transforms.RandomResizedCrop(112),
        # Transforms.ColorJitter(brightness=0.5),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=0, ),
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
    ])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
txt_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/train.txt'
vaild_txt = '/media/omnisky/D4T/JSH/faceFenlei/eye/test.txt'
train_data = jia_signal_eye(txt_path,ttrans)
vaild_data = jia_signal_eye(vaild_txt,vaild_ttrans)
valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)

epoch = 60
batchsize = 256

model = MixNet(input_size=(24,24),num_classes=2)
model.cuda()
#定义多GPU训练
model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

#定义损失函数
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
schedule = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

epoch_size = math.ceil(len(train_data)/batchsize)
maxiter = epoch*epoch_size
epoch_count = 0

for iteration in range(maxiter):
    acc = 0.0
    if iteration % epoch_size == 0:
        if epoch_count>0:
            schedule.step()
            model.eval()
            toal_loss = 0
            with torch.no_grad():
                for imgs,label in valid_data_loader:
                    test_result = model(imgs.cuda())
                    loss = loss_function(test_result,label.cuda())
                    result = torch.max(test_result,1)[1]
                    acc += (result == label.to(device)).sum().item()
                    toal_loss += loss
                writer.add_scalars("test_loss_acc",{"loss":toal_loss/len(vaild_data),"access":acc/len(vaild_data)})
                print("valid_loss:{},valid_access:{}".format(toal_loss/len(vaild_data),acc/len(vaild_data)))
            if epoch_count % 10 == 9:
                    torch.save(model.state_dict(),"/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/set_lookdown_as_open_signaleye_24_24_20210301/Mixnet_epoch_{}.pth".format(epoch_count))
                    print("save weight success!!")
        train_data_loader = iter(DataLoader( dataset=train_data,batch_size=batchsize,shuffle=True,num_workers=12))
        epoch_count += 1
    model.train()
    load_t0 = time.time()
    images,labels = next(train_data_loader)
    
    optimizer.zero_grad()
    images.cuda()
    outputs = model(images)
    loss = loss_function(outputs,labels.cuda())
    loss.backward()
    optimizer.step()

    load_t1 = time.time()
    batch_time = load_t1 - load_t0
    eta = int(batch_time * (maxiter - iteration))
    print("Epoch:{}/{} || Epochiter:{}/{} || loss:{:.4f}||Batchtime:{:.4f}||ETA:{}".format(
        epoch_count,epoch,(iteration%epoch_size)+1,epoch_size,loss.item()/images.size()[0],batch_time,
        str(datetime.timedelta(seconds=eta))
    ))
    writer.add_scalar("loss",loss/images.size()[0],iteration)

