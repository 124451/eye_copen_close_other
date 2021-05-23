import torch.nn as nn
import os
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
from mbhk_dataloader import mbhk_data
from mixnet import MixNet
import torch.nn as nn
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
import math
import datetime


def main():
    label_idx = {0:"open_eye",1:"close_eye",2:"other"}
    #tensorboardX初始化
    writer = SummaryWriter("run/change_mix_iniput_24_48")
    train_txt_path = "/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_train.txt"
    valid_txt_path = "/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_valid.txt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #定义预处理
    transforms_function = {'train': Transforms.Compose([
        Transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.RandomCrop(100),
        # transforms.RandomResizedCrop(112),
        Transforms.ColorJitter(brightness=0.5),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=0, ),
        # Transforms.Resize((48, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ]), 'test': Transforms.Compose([
        # Transforms.Resize((48, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])}
    # 定义数据集
    train_data = mbhk_data(train_txt_path,transform=transforms_function['train'])
    valid_data = mbhk_data(valid_txt_path,transform=transforms_function['test'])
    # train_size = int(0.9 * len(train_data))
    # valid_size = len(train_data) - train_size
    # train_dataset,vaild_dataset = torch.utils.data.random_split(train_data,[train_size,valid_size])
    # train_data_loader = torch.utils.data.DataLoader(train_data,batch_size=256,shuffle=True,num_workers=8)
    
    test_data_loader = DataLoader(valid_data, batch_size=128, shuffle=False, num_workers=8)
    #定义模型
    model = MixNet(input_size=(24,48),num_classes=3)
    model.to(device)
    #定义多GPU训练
    model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    #定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # 定义学习率下降
    schedule = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_acc = 0.0

    # 计算一个epoch的步数
    epoch_size = math.ceil(len(train_data)/128)
    #得到迭代次数
    maxiter = 90*epoch_size
    epoch = 0
    for iteration in range(maxiter):
        acc = 0.0
        if iteration % epoch_size == 0:
            if epoch > 0:
                schedule.step()
                model.eval()
                toal_loss = 0
                with torch.no_grad():
                    for timages,tlabels,_ in test_data_loader:
                        test_result = model(timages.cuda())
                        loss = loss_function(test_result,tlabels.cuda())
                        result = torch.max(test_result,1)[1]
                        acc += (result == tlabels.to(device)).sum().item()
                        toal_loss += loss
                    writer.add_scalars("test_loss_acc",{"loss":toal_loss/len(test_data_loader),"access":acc/len(valid_data)},epoch)
                if epoch % 10 == 9:
                    torch.save(model.state_dict(),"./weight/change_mix_data_0202/Mixnet_epoch_{}.pth".format(epoch))
                    print("save weight success!!")
            train_data_loader = iter(DataLoader( dataset=train_data,batch_size=128,shuffle=True,num_workers=12))
            epoch += 1

        model.train()
        load_t0 = time.time()
        images,labels,_ = next(train_data_loader)
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
            epoch,90,(iteration%epoch_size)+1,epoch_size,loss.item(),batch_time,
            str(datetime.timedelta(seconds=eta))
        ))
        writer.add_scalar("loss",loss,iteration)
        # writer.add_scalar("lr",optim.param_groups[0]['lr'],iteration)



if __name__ == "__main__":
    main()