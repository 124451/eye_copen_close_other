from torch import mode, transpose
import torchvision.transforms as Transforms
import torch
import numpy as np
import glob
from mbhk_dataloader import mbhk_get_signal_eye
from mixnet import MixNet
import torch.optim as optim
import math
from torch.utils.data import DataLoader
import time
import datetime
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter

def get_optim_param(optimizer_param, model):
    optimizer_param = dict(lr=0.05, momentum=0.9, weight_decay=4e-5,
                    bias_lr_factor=2, bias_weight_decay_facotr=0)
    base_lr = optimizer_param['lr']
    base_weight_decay = optimizer_param['weight_decay']
    bias_lr_factor = optimizer_param.get('bias_lr_factor', 1)
    bias_weight_decay_factor = optimizer_param.get('bias_weight_decay_facotr', 1)
    # split model by bias params and norml params
    bias_params = []
    normal_params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                normal_params.append(value)

    params = [
        {'params': normal_params, 'lr': base_lr, 'weight_decay': base_weight_decay},
        {'params': bias_params, 'lr': base_lr * bias_lr_factor,
            'weight_decay': base_weight_decay * bias_weight_decay_factor}
    ]

    optimizer_param['params'] = params
    if 'bias_lr_factor' in  optimizer_param.keys():
        optimizer_param.pop('bias_lr_factor')

    if 'bias_weight_decay_facotr' in optimizer_param.keys():
        optimizer_param.pop('bias_weight_decay_facotr')
    return optimizer_param


writer = SummaryWriter("run/relabel_04_mix_SGD_mutillabel_24_24_20210302")
ttrans = Transforms.Compose([
        Transforms.RandomHorizontalFlip(p=0.5),
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
txt_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_train.txt'
vaild_txt = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_valid.txt'
train_data = mbhk_get_signal_eye(txt_path,ttrans)
vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)

epoch = 60
batchsize = 256

model = MixNet(input_size=(24,24),num_classes=3)
optimizer_param = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=4e-5,
                 bias_lr_factor=2, bias_weight_decay_facotr=0)
optimizer = torch.optim.SGD(**get_optim_param(optimizer_param, model))
schedule = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
model.cuda()
#定义多GPU训练
model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()

#定义损失函数
# loss_function = nn.CrossEntropyLoss()
loss_function = nn.MultiLabelSoftMarginLoss()


# optimizer = optim.Adam(model.parameters(),lr = 0.001)

# SGD加余弦退火学习率下降
# schedule = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

epoch_size = math.ceil(len(train_data)/batchsize)-1
maxiter = epoch*epoch_size
epoch_count = 0
#双余弦计数
cosin_copunt = 0
for iteration in range(maxiter):
    acc = 0.0
    schedule.step(epoch_count+cosin_copunt/epoch_size)
    cosin_copunt += 1
    if iteration % epoch_size == 0:
        cosin_copunt = 0
        if epoch_count>0:
            # schedule.step()
            model.eval()
            toal_loss = 0
            with torch.no_grad():
                for imgs,label,_ in valid_data_loader:
                    val_lab = label.clone()
                    label = torch.nn.functional.one_hot(label,num_classes=3)
                    for timg in imgs:
                        test_result = model(timg.cuda())
                        loss = loss_function(test_result,label.cuda())
                        result = torch.max(test_result,1)[1]
                        acc += (result == val_lab.to(device)).sum().item()
                        toal_loss += loss
                print("valid_loss:{},valid_access:{}".format(toal_loss/len(vaild_data)/2,acc/len(vaild_data)/2))
                writer.add_scalars("test_loss_acc",{"loss":toal_loss/len(vaild_data)/2,"access":acc/len(vaild_data)/2},epoch_count)
            # if epoch_count % 10 == 9:
            torch.save(model.state_dict(),"/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_{}.pth".format(epoch_count))
            print("save weight success!!")
        train_data_loader = iter(DataLoader( dataset=train_data,batch_size=batchsize,shuffle=True,num_workers=12,drop_last=True))
        epoch_count += 1
    model.train()
    load_t0 = time.time()
    images,labels,_ = next(train_data_loader)
    labels = torch.nn.functional.one_hot(labels,num_classes=3)
    for train_img in images:
        optimizer.zero_grad()
        train_img.cuda()
        outputs = model(train_img)
        loss = loss_function(outputs,labels.cuda())
        loss.backward()
        optimizer.step()

    load_t1 = time.time()
    batch_time = load_t1 - load_t0
    eta = int(batch_time * (maxiter - iteration))
    print("Epoch:{}/{} || Epochiter:{}/{} || loss:{:.4f}||Batchtime:{:.4f}||ETA:{}".format(
        epoch_count,epoch,(iteration%epoch_size)+1,epoch_size,loss.item()/train_img.size()[0],batch_time,
        str(datetime.timedelta(seconds=eta))
    ))
    writer.add_scalars("train_ad_lr",{"train_loss":loss/train_img.size()[0],"lr":optimizer.param_groups[0]['lr']},iteration)
    # writer.add_scalar("lr",optimizer.param_groups[0]['lr'],iteration)
'''
2021年3月3日09:40:57
SGD加余弦学习率下降
优化器：交叉熵
Epoch:17/60 || Epochiter:2/22 || loss:0.0018||Batchtime:0.2562||ETA:0:04:07
Epoch:17/60 || Epochiter:3/22 || loss:0.0018||Batchtime:0.1210||ETA:0:01:56
Epoch:17/60 || Epochiter:4/22 || loss:0.0015||Batchtime:0.1371||ETA:0:02:12
Epoch:17/60 || Epochiter:5/22 || loss:0.0015||Batchtime:0.1574||ETA:0:02:31
Epoch:17/60 || Epochiter:6/22 || loss:0.0014||Batchtime:0.1258||ETA:0:02:01
Epoch:17/60 || Epochiter:7/22 || loss:0.0017||Batchtime:0.1356||ETA:0:02:10
Epoch:17/60 || Epochiter:8/22 || loss:0.0019||Batchtime:0.2294||ETA:0:03:40
Epoch:17/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1413||ETA:0:02:15
Epoch:17/60 || Epochiter:10/22 || loss:0.0013||Batchtime:0.1273||ETA:0:02:02
Epoch:17/60 || Epochiter:11/22 || loss:0.0015||Batchtime:0.1248||ETA:0:01:59
Epoch:17/60 || Epochiter:12/22 || loss:0.0015||Batchtime:0.1415||ETA:0:02:15
Epoch:17/60 || Epochiter:13/22 || loss:0.0014||Batchtime:0.8906||ETA:0:14:11
Epoch:17/60 || Epochiter:14/22 || loss:0.0016||Batchtime:0.1736||ETA:0:02:45
Epoch:17/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.1702||ETA:0:02:42
Epoch:17/60 || Epochiter:16/22 || loss:0.0017||Batchtime:0.1586||ETA:0:02:31
Epoch:17/60 || Epochiter:17/22 || loss:0.0016||Batchtime:0.1706||ETA:0:02:42
Epoch:17/60 || Epochiter:18/22 || loss:0.0015||Batchtime:0.1708||ETA:0:02:42
Epoch:17/60 || Epochiter:19/22 || loss:0.0015||Batchtime:0.1748||ETA:0:02:46
Epoch:17/60 || Epochiter:20/22 || loss:0.0015||Batchtime:0.1861||ETA:0:02:56
Epoch:17/60 || Epochiter:21/22 || loss:0.0015||Batchtime:0.1795||ETA:0:02:50
Epoch:17/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.2184||ETA:0:03:26
valid_loss:0.003047939855605364,valid_access:0.8531746031746031
Epoch:18/60 || Epochiter:1/22 || loss:0.0016||Batchtime:1.4399||ETA:0:22:42
Epoch:18/60 || Epochiter:2/22 || loss:0.0017||Batchtime:0.8095||ETA:0:12:44
Epoch:18/60 || Epochiter:3/22 || loss:0.0018||Batchtime:0.1650||ETA:0:02:35
Epoch:18/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1603||ETA:0:02:31
Epoch:18/60 || Epochiter:5/22 || loss:0.0016||Batchtime:0.1368||ETA:0:02:08
Epoch:18/60 || Epochiter:6/22 || loss:0.0016||Batchtime:0.1389||ETA:0:02:10
Epoch:18/60 || Epochiter:7/22 || loss:0.0017||Batchtime:0.1494||ETA:0:02:20
Epoch:18/60 || Epochiter:8/22 || loss:0.0015||Batchtime:0.1488||ETA:0:02:19
Epoch:18/60 || Epochiter:9/22 || loss:0.0019||Batchtime:0.1426||ETA:0:02:13
Epoch:18/60 || Epochiter:10/22 || loss:0.0018||Batchtime:0.1465||ETA:0:02:17
Epoch:18/60 || Epochiter:11/22 || loss:0.0018||Batchtime:0.1511||ETA:0:02:21
Epoch:18/60 || Epochiter:12/22 || loss:0.0015||Batchtime:0.1565||ETA:0:02:26
Epoch:18/60 || Epochiter:13/22 || loss:0.0018||Batchtime:0.1667||ETA:0:02:35
Epoch:18/60 || Epochiter:14/22 || loss:0.0018||Batchtime:0.7908||ETA:0:12:17
Epoch:18/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.1281||ETA:0:01:59
Epoch:18/60 || Epochiter:16/22 || loss:0.0014||Batchtime:0.2076||ETA:0:03:13
Epoch:18/60 || Epochiter:17/22 || loss:0.0014||Batchtime:0.1496||ETA:0:02:19
Epoch:18/60 || Epochiter:18/22 || loss:0.0014||Batchtime:0.1500||ETA:0:02:19
Epoch:18/60 || Epochiter:19/22 || loss:0.0018||Batchtime:0.1404||ETA:0:02:10
Epoch:18/60 || Epochiter:20/22 || loss:0.0012||Batchtime:0.1464||ETA:0:02:15
Epoch:18/60 || Epochiter:21/22 || loss:0.0013||Batchtime:0.1592||ETA:0:02:27
Epoch:18/60 || Epochiter:22/22 || loss:0.0019||Batchtime:0.1729||ETA:0:02:39
valid_loss:0.002842211863026023,valid_access:0.8555555555555555
Epoch:19/60 || Epochiter:1/22 || loss:0.0014||Batchtime:2.0321||ETA:0:31:17
Epoch:19/60 || Epochiter:2/22 || loss:0.0014||Batchtime:0.1435||ETA:0:02:12
Epoch:19/60 || Epochiter:3/22 || loss:0.0014||Batchtime:0.1399||ETA:0:02:08
Epoch:19/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1421||ETA:0:02:10
Epoch:19/60 || Epochiter:5/22 || loss:0.0014||Batchtime:0.1317||ETA:0:02:01
Epoch:19/60 || Epochiter:6/22 || loss:0.0016||Batchtime:0.1606||ETA:0:02:27
Epoch:19/60 || Epochiter:7/22 || loss:0.0015||Batchtime:0.1388||ETA:0:02:07
Epoch:19/60 || Epochiter:8/22 || loss:0.0016||Batchtime:0.2671||ETA:0:04:04
Epoch:19/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.3361||ETA:0:05:07
Epoch:19/60 || Epochiter:10/22 || loss:0.0014||Batchtime:0.1393||ETA:0:02:07
Epoch:19/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1415||ETA:0:02:09
Epoch:19/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1520||ETA:0:02:18
Epoch:19/60 || Epochiter:13/22 || loss:0.0019||Batchtime:0.8953||ETA:0:13:36
Epoch:19/60 || Epochiter:14/22 || loss:0.0015||Batchtime:0.1700||ETA:0:02:34
Epoch:19/60 || Epochiter:15/22 || loss:0.0018||Batchtime:0.1662||ETA:0:02:31
Epoch:19/60 || Epochiter:16/22 || loss:0.0014||Batchtime:0.1776||ETA:0:02:41
Epoch:19/60 || Epochiter:17/22 || loss:0.0015||Batchtime:0.1558||ETA:0:02:21
Epoch:19/60 || Epochiter:18/22 || loss:0.0016||Batchtime:0.1615||ETA:0:02:26
Epoch:19/60 || Epochiter:19/22 || loss:0.0016||Batchtime:0.1756||ETA:0:02:39
Epoch:19/60 || Epochiter:20/22 || loss:0.0016||Batchtime:0.1766||ETA:0:02:39
Epoch:19/60 || Epochiter:21/22 || loss:0.0016||Batchtime:0.1653||ETA:0:02:29
Epoch:19/60 || Epochiter:22/22 || loss:0.0016||Batchtime:0.1392||ETA:0:02:05
valid_loss:0.0031674096826463938,valid_access:0.8396825396825397
save weight success!!
Epoch:20/60 || Epochiter:1/22 || loss:0.0014||Batchtime:1.5217||ETA:0:22:52
Epoch:20/60 || Epochiter:2/22 || loss:0.0017||Batchtime:0.1434||ETA:0:02:09
Epoch:20/60 || Epochiter:3/22 || loss:0.0015||Batchtime:0.3195||ETA:0:04:47
Epoch:20/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1406||ETA:0:02:06
Epoch:20/60 || Epochiter:5/22 || loss:0.0016||Batchtime:0.6047||ETA:0:09:03
Epoch:20/60 || Epochiter:6/22 || loss:0.0014||Batchtime:0.1545||ETA:0:02:18
Epoch:20/60 || Epochiter:7/22 || loss:0.0015||Batchtime:0.3211||ETA:0:04:47
Epoch:20/60 || Epochiter:8/22 || loss:0.0016||Batchtime:0.1410||ETA:0:02:06
Epoch:20/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1420||ETA:0:02:06
Epoch:20/60 || Epochiter:10/22 || loss:0.0013||Batchtime:0.1402||ETA:0:02:05
Epoch:20/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1380||ETA:0:02:03
Epoch:20/60 || Epochiter:12/22 || loss:0.0014||Batchtime:0.1367||ETA:0:02:01
Epoch:20/60 || Epochiter:13/22 || loss:0.0017||Batchtime:0.1540||ETA:0:02:17
Epoch:20/60 || Epochiter:14/22 || loss:0.0015||Batchtime:0.1511||ETA:0:02:14
Epoch:20/60 || Epochiter:15/22 || loss:0.0019||Batchtime:0.1487||ETA:0:02:12
Epoch:20/60 || Epochiter:16/22 || loss:0.0016||Batchtime:0.1672||ETA:0:02:28
Epoch:20/60 || Epochiter:17/22 || loss:0.0015||Batchtime:0.3583||ETA:0:05:17
Epoch:20/60 || Epochiter:18/22 || loss:0.0014||Batchtime:0.1666||ETA:0:02:27
Epoch:20/60 || Epochiter:19/22 || loss:0.0014||Batchtime:0.2940||ETA:0:04:19
Epoch:20/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1828||ETA:0:02:41
Epoch:20/60 || Epochiter:21/22 || loss:0.0014||Batchtime:0.1668||ETA:0:02:27
Epoch:20/60 || Epochiter:22/22 || loss:0.0015||Batchtime:0.1694||ETA:0:02:29
valid_loss:0.0027474388480186462,valid_access:0.8714285714285714
Epoch:21/60 || Epochiter:1/22 || loss:0.0016||Batchtime:1.4349||ETA:0:21:02
Epoch:21/60 || Epochiter:2/22 || loss:0.0013||Batchtime:0.8269||ETA:0:12:06
Epoch:21/60 || Epochiter:3/22 || loss:0.0015||Batchtime:0.1324||ETA:0:01:56
Epoch:21/60 || Epochiter:4/22 || loss:0.0015||Batchtime:0.1388||ETA:0:02:01
Epoch:21/60 || Epochiter:5/22 || loss:0.0015||Batchtime:0.4701||ETA:0:06:51
Epoch:21/60 || Epochiter:6/22 || loss:0.0019||Batchtime:0.1393||ETA:0:02:01
Epoch:21/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1575||ETA:0:02:17
Epoch:21/60 || Epochiter:8/22 || loss:0.0013||Batchtime:0.1522||ETA:0:02:12
Epoch:21/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1373||ETA:0:01:59
Epoch:21/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1421||ETA:0:02:03
Epoch:21/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1497||ETA:0:02:10
Epoch:21/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1440||ETA:0:02:05
Epoch:21/60 || Epochiter:13/22 || loss:0.0013||Batchtime:0.1542||ETA:0:02:13
Epoch:21/60 || Epochiter:14/22 || loss:0.0013||Batchtime:0.2899||ETA:0:04:11
Epoch:21/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1761||ETA:0:02:32
Epoch:21/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1726||ETA:0:02:29
Epoch:21/60 || Epochiter:17/22 || loss:0.0016||Batchtime:0.5406||ETA:0:07:47
Epoch:21/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1647||ETA:0:02:22
Epoch:21/60 || Epochiter:19/22 || loss:0.0014||Batchtime:0.1759||ETA:0:02:31
Epoch:21/60 || Epochiter:20/22 || loss:0.0016||Batchtime:0.1840||ETA:0:02:38
Epoch:21/60 || Epochiter:21/22 || loss:0.0018||Batchtime:0.1650||ETA:0:02:21
Epoch:21/60 || Epochiter:22/22 || loss:0.0016||Batchtime:0.1696||ETA:0:02:25
valid_loss:0.0028505504596978426,valid_access:0.861904761904762
Epoch:22/60 || Epochiter:1/22 || loss:0.0013||Batchtime:1.7274||ETA:0:24:42
Epoch:22/60 || Epochiter:2/22 || loss:0.0016||Batchtime:0.1233||ETA:0:01:45
Epoch:22/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1570||ETA:0:02:14
Epoch:22/60 || Epochiter:4/22 || loss:0.0014||Batchtime:0.1479||ETA:0:02:06
Epoch:22/60 || Epochiter:5/22 || loss:0.0013||Batchtime:0.3475||ETA:0:04:56
Epoch:22/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1113||ETA:0:01:34
Epoch:22/60 || Epochiter:7/22 || loss:0.0015||Batchtime:0.6570||ETA:0:09:19
Epoch:22/60 || Epochiter:8/22 || loss:0.0015||Batchtime:0.1437||ETA:0:02:02
Epoch:22/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1577||ETA:0:02:14
Epoch:22/60 || Epochiter:10/22 || loss:0.0015||Batchtime:0.1610||ETA:0:02:16
Epoch:22/60 || Epochiter:11/22 || loss:0.0016||Batchtime:0.1278||ETA:0:01:48
Epoch:22/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1532||ETA:0:02:09
Epoch:22/60 || Epochiter:13/22 || loss:0.0015||Batchtime:0.2652||ETA:0:03:44
Epoch:22/60 || Epochiter:14/22 || loss:0.0016||Batchtime:0.1506||ETA:0:02:07
Epoch:22/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1703||ETA:0:02:23
Epoch:22/60 || Epochiter:16/22 || loss:0.0010||Batchtime:0.1527||ETA:0:02:08
Epoch:22/60 || Epochiter:17/22 || loss:0.0015||Batchtime:0.1467||ETA:0:02:03
Epoch:22/60 || Epochiter:18/22 || loss:0.0015||Batchtime:0.1490||ETA:0:02:05
Epoch:22/60 || Epochiter:19/22 || loss:0.0014||Batchtime:0.2039||ETA:0:02:51
Epoch:22/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1535||ETA:0:02:08
Epoch:22/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1717||ETA:0:02:23
Epoch:22/60 || Epochiter:22/22 || loss:0.0011||Batchtime:0.1732||ETA:0:02:24
valid_loss:0.0031014468986541033,valid_access:0.8476190476190476
Epoch:23/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.3151||ETA:0:18:19
Epoch:23/60 || Epochiter:2/22 || loss:0.0015||Batchtime:0.1974||ETA:0:02:44
Epoch:23/60 || Epochiter:3/22 || loss:0.0015||Batchtime:0.3372||ETA:0:04:41
Epoch:23/60 || Epochiter:4/22 || loss:0.0015||Batchtime:0.2251||ETA:0:03:07
Epoch:23/60 || Epochiter:5/22 || loss:0.0013||Batchtime:0.9683||ETA:0:13:25
Epoch:23/60 || Epochiter:6/22 || loss:0.0013||Batchtime:0.2678||ETA:0:03:42
Epoch:23/60 || Epochiter:7/22 || loss:0.0013||Batchtime:0.1316||ETA:0:01:49
Epoch:23/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1246||ETA:0:01:43
Epoch:23/60 || Epochiter:9/22 || loss:0.0016||Batchtime:0.1431||ETA:0:01:58
Epoch:23/60 || Epochiter:10/22 || loss:0.0016||Batchtime:0.1549||ETA:0:02:08
Epoch:23/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1607||ETA:0:02:12
Epoch:23/60 || Epochiter:12/22 || loss:0.0015||Batchtime:0.1482||ETA:0:02:02
Epoch:23/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.1641||ETA:0:02:15
Epoch:23/60 || Epochiter:14/22 || loss:0.0014||Batchtime:0.1953||ETA:0:02:40
Epoch:23/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1635||ETA:0:02:14
Epoch:23/60 || Epochiter:16/22 || loss:0.0016||Batchtime:0.1499||ETA:0:02:03
Epoch:23/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.4396||ETA:0:06:00
Epoch:23/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1731||ETA:0:02:21
Epoch:23/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1831||ETA:0:02:29
Epoch:23/60 || Epochiter:20/22 || loss:0.0014||Batchtime:0.1672||ETA:0:02:16
Epoch:23/60 || Epochiter:21/22 || loss:0.0013||Batchtime:0.1526||ETA:0:02:04
Epoch:23/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.1727||ETA:0:02:20
valid_loss:0.0028036627918481827,valid_access:0.8626984126984127
Epoch:24/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.5322||ETA:0:20:47
Epoch:24/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.1675||ETA:0:02:16
Epoch:24/60 || Epochiter:3/22 || loss:0.0011||Batchtime:1.0051||ETA:0:13:36
Epoch:24/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1202||ETA:0:01:37
Epoch:24/60 || Epochiter:5/22 || loss:0.0013||Batchtime:0.1443||ETA:0:01:56
Epoch:24/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.5365||ETA:0:07:14
Epoch:24/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1560||ETA:0:02:06
Epoch:24/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1574||ETA:0:02:07
Epoch:24/60 || Epochiter:9/22 || loss:0.0014||Batchtime:0.1542||ETA:0:02:04
Epoch:24/60 || Epochiter:10/22 || loss:0.0016||Batchtime:0.1575||ETA:0:02:06
Epoch:24/60 || Epochiter:11/22 || loss:0.0015||Batchtime:0.1397||ETA:0:01:52
Epoch:24/60 || Epochiter:12/22 || loss:0.0016||Batchtime:0.1500||ETA:0:02:00
Epoch:24/60 || Epochiter:13/22 || loss:0.0014||Batchtime:0.1581||ETA:0:02:06
Epoch:24/60 || Epochiter:14/22 || loss:0.0014||Batchtime:0.1385||ETA:0:01:50
Epoch:24/60 || Epochiter:15/22 || loss:0.0016||Batchtime:0.1907||ETA:0:02:32
Epoch:24/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1583||ETA:0:02:06
Epoch:24/60 || Epochiter:17/22 || loss:0.0015||Batchtime:0.1466||ETA:0:01:56
Epoch:24/60 || Epochiter:18/22 || loss:0.0013||Batchtime:0.2812||ETA:0:03:44
Epoch:24/60 || Epochiter:19/22 || loss:0.0011||Batchtime:0.1666||ETA:0:02:12
Epoch:24/60 || Epochiter:20/22 || loss:0.0015||Batchtime:0.1597||ETA:0:02:06
Epoch:24/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1784||ETA:0:02:21
Epoch:24/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.1752||ETA:0:02:18
valid_loss:0.0026587126776576042,valid_access:0.8777777777777778
Epoch:25/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.4968||ETA:0:19:45
Epoch:25/60 || Epochiter:2/22 || loss:0.0016||Batchtime:0.1598||ETA:0:02:06
Epoch:25/60 || Epochiter:3/22 || loss:0.0012||Batchtime:0.1534||ETA:0:02:01
Epoch:25/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.6512||ETA:0:08:33
Epoch:25/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1400||ETA:0:01:50
Epoch:25/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1409||ETA:0:01:50
Epoch:25/60 || Epochiter:7/22 || loss:0.0012||Batchtime:0.1411||ETA:0:01:50
Epoch:25/60 || Epochiter:8/22 || loss:0.0014||Batchtime:0.1469||ETA:0:01:55
Epoch:25/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.3926||ETA:0:05:07
Epoch:25/60 || Epochiter:10/22 || loss:0.0014||Batchtime:0.1368||ETA:0:01:47
Epoch:25/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1404||ETA:0:01:49
Epoch:25/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1317||ETA:0:01:42
Epoch:25/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.6545||ETA:0:08:30
Epoch:25/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1694||ETA:0:02:11
Epoch:25/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.1609||ETA:0:02:05
Epoch:25/60 || Epochiter:16/22 || loss:0.0016||Batchtime:0.1636||ETA:0:02:07
Epoch:25/60 || Epochiter:17/22 || loss:0.0016||Batchtime:0.1689||ETA:0:02:11
Epoch:25/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1662||ETA:0:02:08
Epoch:25/60 || Epochiter:19/22 || loss:0.0014||Batchtime:0.1684||ETA:0:02:10
Epoch:25/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.2433||ETA:0:03:08
Epoch:25/60 || Epochiter:21/22 || loss:0.0015||Batchtime:0.1664||ETA:0:02:08
Epoch:25/60 || Epochiter:22/22 || loss:0.0011||Batchtime:0.1754||ETA:0:02:15
valid_loss:0.0026347311213612556,valid_access:0.8698412698412699
Epoch:26/60 || Epochiter:1/22 || loss:0.0009||Batchtime:1.3962||ETA:0:17:55
Epoch:26/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.6376||ETA:0:08:10
Epoch:26/60 || Epochiter:3/22 || loss:0.0014||Batchtime:0.1284||ETA:0:01:38
Epoch:26/60 || Epochiter:4/22 || loss:0.0015||Batchtime:0.1429||ETA:0:01:49
Epoch:26/60 || Epochiter:5/22 || loss:0.0013||Batchtime:0.1285||ETA:0:01:38
Epoch:26/60 || Epochiter:6/22 || loss:0.0011||Batchtime:0.1434||ETA:0:01:49
Epoch:26/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1419||ETA:0:01:48
Epoch:26/60 || Epochiter:8/22 || loss:0.0014||Batchtime:0.3632||ETA:0:04:37
Epoch:26/60 || Epochiter:9/22 || loss:0.0013||Batchtime:0.1571||ETA:0:01:59
Epoch:26/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1393||ETA:0:01:45
Epoch:26/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1425||ETA:0:01:48
Epoch:26/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1525||ETA:0:01:55
Epoch:26/60 || Epochiter:13/22 || loss:0.0014||Batchtime:0.3457||ETA:0:04:22
Epoch:26/60 || Epochiter:14/22 || loss:0.0013||Batchtime:0.3921||ETA:0:04:56
Epoch:26/60 || Epochiter:15/22 || loss:0.0011||Batchtime:0.2606||ETA:0:03:16
Epoch:26/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1716||ETA:0:02:09
Epoch:26/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1720||ETA:0:02:09
Epoch:26/60 || Epochiter:18/22 || loss:0.0015||Batchtime:0.1653||ETA:0:02:04
Epoch:26/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1674||ETA:0:02:05
Epoch:26/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1695||ETA:0:02:07
Epoch:26/60 || Epochiter:21/22 || loss:0.0013||Batchtime:0.1711||ETA:0:02:08
Epoch:26/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.1803||ETA:0:02:15
valid_loss:0.0025107075925916433,valid_access:0.8833333333333333
Epoch:27/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.4668||ETA:0:18:17
Epoch:27/60 || Epochiter:2/22 || loss:0.0009||Batchtime:0.1457||ETA:0:01:48
Epoch:27/60 || Epochiter:3/22 || loss:0.0012||Batchtime:0.1354||ETA:0:01:41
Epoch:27/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1441||ETA:0:01:47
Epoch:27/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1893||ETA:0:02:20
Epoch:27/60 || Epochiter:6/22 || loss:0.0011||Batchtime:0.1478||ETA:0:01:49
Epoch:27/60 || Epochiter:7/22 || loss:0.0011||Batchtime:1.1321||ETA:0:13:59
Epoch:27/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1582||ETA:0:01:57
Epoch:27/60 || Epochiter:9/22 || loss:0.0012||Batchtime:0.1510||ETA:0:01:51
Epoch:27/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1587||ETA:0:01:57
Epoch:27/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1338||ETA:0:01:38
Epoch:27/60 || Epochiter:12/22 || loss:0.0010||Batchtime:0.2429||ETA:0:02:58
Epoch:27/60 || Epochiter:13/22 || loss:0.0013||Batchtime:0.1909||ETA:0:02:20
Epoch:27/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1708||ETA:0:02:05
Epoch:27/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1743||ETA:0:02:07
Epoch:27/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1782||ETA:0:02:10
Epoch:27/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1786||ETA:0:02:10
Epoch:27/60 || Epochiter:18/22 || loss:0.0013||Batchtime:0.1724||ETA:0:02:06
Epoch:27/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.2498||ETA:0:03:02
Epoch:27/60 || Epochiter:20/22 || loss:0.0012||Batchtime:0.1865||ETA:0:02:15
Epoch:27/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1846||ETA:0:02:14
Epoch:27/60 || Epochiter:22/22 || loss:0.0013||Batchtime:0.1836||ETA:0:02:13
valid_loss:0.00253820838406682,valid_access:0.8761904761904762
Epoch:28/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.3700||ETA:0:16:34
Epoch:28/60 || Epochiter:2/22 || loss:0.0008||Batchtime:0.1615||ETA:0:01:57
Epoch:28/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1759||ETA:0:02:07
Epoch:28/60 || Epochiter:4/22 || loss:0.0015||Batchtime:0.1756||ETA:0:02:06
Epoch:28/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1766||ETA:0:02:07
Epoch:28/60 || Epochiter:6/22 || loss:0.0011||Batchtime:0.4908||ETA:0:05:53
Epoch:28/60 || Epochiter:7/22 || loss:0.0010||Batchtime:0.2209||ETA:0:02:39
Epoch:28/60 || Epochiter:8/22 || loss:0.0012||Batchtime:0.1618||ETA:0:01:56
Epoch:28/60 || Epochiter:9/22 || loss:0.0009||Batchtime:0.1420||ETA:0:01:41
Epoch:28/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.3809||ETA:0:04:33
Epoch:28/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1220||ETA:0:01:27
Epoch:28/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1529||ETA:0:01:49
Epoch:28/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.2602||ETA:0:03:05
Epoch:28/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1558||ETA:0:01:51
Epoch:28/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.4612||ETA:0:05:28
Epoch:28/60 || Epochiter:16/22 || loss:0.0011||Batchtime:0.1552||ETA:0:01:50
Epoch:28/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1550||ETA:0:01:50
Epoch:28/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1477||ETA:0:01:44
Epoch:28/60 || Epochiter:19/22 || loss:0.0012||Batchtime:0.1422||ETA:0:01:40
Epoch:28/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.1564||ETA:0:01:50
Epoch:28/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.1676||ETA:0:01:58
Epoch:28/60 || Epochiter:22/22 || loss:0.0010||Batchtime:0.1775||ETA:0:02:05
valid_loss:0.0024888673797249794,valid_access:0.8841269841269841
Epoch:29/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.5159||ETA:0:17:47
Epoch:29/60 || Epochiter:2/22 || loss:0.0014||Batchtime:0.1141||ETA:0:01:20
Epoch:29/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1376||ETA:0:01:36
Epoch:29/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1446||ETA:0:01:41
Epoch:29/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.2688||ETA:0:03:08
Epoch:29/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.5147||ETA:0:05:59
Epoch:29/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1118||ETA:0:01:18
Epoch:29/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.3213||ETA:0:03:43
Epoch:29/60 || Epochiter:9/22 || loss:0.0012||Batchtime:0.1379||ETA:0:01:35
Epoch:29/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1387||ETA:0:01:36
Epoch:29/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1186||ETA:0:01:22
Epoch:29/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1429||ETA:0:01:39
Epoch:29/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.5904||ETA:0:06:48
Epoch:29/60 || Epochiter:14/22 || loss:0.0011||Batchtime:0.1574||ETA:0:01:48
Epoch:29/60 || Epochiter:15/22 || loss:0.0009||Batchtime:0.1794||ETA:0:02:03
Epoch:29/60 || Epochiter:16/22 || loss:0.0012||Batchtime:0.1650||ETA:0:01:53
Epoch:29/60 || Epochiter:17/22 || loss:0.0009||Batchtime:0.1753||ETA:0:02:00
Epoch:29/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1651||ETA:0:01:53
Epoch:29/60 || Epochiter:19/22 || loss:0.0010||Batchtime:0.1570||ETA:0:01:47
Epoch:29/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1716||ETA:0:01:57
Epoch:29/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1436||ETA:0:01:38
Epoch:29/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1626||ETA:0:01:51
valid_loss:0.0025020784232765436,valid_access:0.8801587301587301
save weight success!!
Epoch:30/60 || Epochiter:1/22 || loss:0.0008||Batchtime:1.4941||ETA:0:16:59
Epoch:30/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.1525||ETA:0:01:43
Epoch:30/60 || Epochiter:3/22 || loss:0.0014||Batchtime:0.1352||ETA:0:01:31
Epoch:30/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.5475||ETA:0:06:11
Epoch:30/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.2693||ETA:0:03:02
Epoch:30/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.2091||ETA:0:02:21
Epoch:30/60 || Epochiter:7/22 || loss:0.0012||Batchtime:0.1395||ETA:0:01:34
Epoch:30/60 || Epochiter:8/22 || loss:0.0014||Batchtime:0.1501||ETA:0:01:41
Epoch:30/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.1591||ETA:0:01:47
Epoch:30/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1885||ETA:0:02:06
Epoch:30/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1668||ETA:0:01:52
Epoch:30/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1548||ETA:0:01:43
Epoch:30/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.1517||ETA:0:01:41
Epoch:30/60 || Epochiter:14/22 || loss:0.0009||Batchtime:0.2365||ETA:0:02:38
Epoch:30/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1532||ETA:0:01:42
Epoch:30/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.3380||ETA:0:03:45
Epoch:30/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1670||ETA:0:01:51
Epoch:30/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1567||ETA:0:01:44
Epoch:30/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1460||ETA:0:01:36
Epoch:30/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1651||ETA:0:01:49
Epoch:30/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1680||ETA:0:01:51
Epoch:30/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1721||ETA:0:01:53
valid_loss:0.0024881975259631872,valid_access:0.8817460317460317
Epoch:31/60 || Epochiter:1/22 || loss:0.0013||Batchtime:1.4602||ETA:0:16:03
Epoch:31/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.1484||ETA:0:01:37
Epoch:31/60 || Epochiter:3/22 || loss:0.0011||Batchtime:0.1534||ETA:0:01:40
Epoch:31/60 || Epochiter:4/22 || loss:0.0010||Batchtime:1.1523||ETA:0:12:37
Epoch:31/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1389||ETA:0:01:31
Epoch:31/60 || Epochiter:6/22 || loss:0.0011||Batchtime:0.1627||ETA:0:01:46
Epoch:31/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1542||ETA:0:01:40
Epoch:31/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1739||ETA:0:01:53
Epoch:31/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1368||ETA:0:01:29
Epoch:31/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1335||ETA:0:01:26
Epoch:31/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1655||ETA:0:01:47
Epoch:31/60 || Epochiter:12/22 || loss:0.0010||Batchtime:0.1671||ETA:0:01:48
Epoch:31/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.1703||ETA:0:01:50
Epoch:31/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1570||ETA:0:01:41
Epoch:31/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1774||ETA:0:01:54
Epoch:31/60 || Epochiter:16/22 || loss:0.0011||Batchtime:0.2724||ETA:0:02:55
Epoch:31/60 || Epochiter:17/22 || loss:0.0008||Batchtime:0.1705||ETA:0:01:49
Epoch:31/60 || Epochiter:18/22 || loss:0.0011||Batchtime:0.1608||ETA:0:01:43
Epoch:31/60 || Epochiter:19/22 || loss:0.0010||Batchtime:0.1742||ETA:0:01:51
Epoch:31/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1915||ETA:0:02:02
Epoch:31/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1660||ETA:0:01:46
Epoch:31/60 || Epochiter:22/22 || loss:0.0011||Batchtime:0.1707||ETA:0:01:49
valid_loss:0.0025213451590389013,valid_access:0.8825396825396825
Epoch:32/60 || Epochiter:1/22 || loss:0.0010||Batchtime:2.2733||ETA:0:24:10
Epoch:32/60 || Epochiter:2/22 || loss:0.0014||Batchtime:0.1395||ETA:0:01:28
Epoch:32/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.1527||ETA:0:01:37
Epoch:32/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1555||ETA:0:01:38
Epoch:32/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.2513||ETA:0:02:39
Epoch:32/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.1384||ETA:0:01:27
Epoch:32/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1359||ETA:0:01:25
Epoch:32/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1646||ETA:0:01:43
Epoch:32/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.1707||ETA:0:01:47
Epoch:32/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.1457||ETA:0:01:31
Epoch:32/60 || Epochiter:11/22 || loss:0.0014||Batchtime:0.1392||ETA:0:01:27
Epoch:32/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1387||ETA:0:01:26
Epoch:32/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.7399||ETA:0:07:43
Epoch:32/60 || Epochiter:14/22 || loss:0.0009||Batchtime:0.1632||ETA:0:01:42
Epoch:32/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1685||ETA:0:01:45
Epoch:32/60 || Epochiter:16/22 || loss:0.0011||Batchtime:0.1581||ETA:0:01:38
Epoch:32/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1615||ETA:0:01:40
Epoch:32/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1648||ETA:0:01:42
Epoch:32/60 || Epochiter:19/22 || loss:0.0010||Batchtime:0.1622||ETA:0:01:40
Epoch:32/60 || Epochiter:20/22 || loss:0.0012||Batchtime:0.1639||ETA:0:01:41
Epoch:32/60 || Epochiter:21/22 || loss:0.0013||Batchtime:0.1695||ETA:0:01:44
Epoch:32/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1720||ETA:0:01:46
valid_loss:0.002539984416216612,valid_access:0.8825396825396825
Epoch:33/60 || Epochiter:1/22 || loss:0.0012||Batchtime:2.3162||ETA:0:23:46
Epoch:33/60 || Epochiter:2/22 || loss:0.0011||Batchtime:0.1413||ETA:0:01:26
Epoch:33/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.1373||ETA:0:01:24
Epoch:33/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.2424||ETA:0:02:28
Epoch:33/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.1247||ETA:0:01:16
Epoch:33/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1143||ETA:0:01:09
Epoch:33/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.5010||ETA:0:05:05
Epoch:33/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1215||ETA:0:01:14
Epoch:33/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1327||ETA:0:01:20
Epoch:33/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.1443||ETA:0:01:27
Epoch:33/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1437||ETA:0:01:27
Epoch:33/60 || Epochiter:12/22 || loss:0.0013||Batchtime:0.1520||ETA:0:01:31
Epoch:33/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.1934||ETA:0:01:56
Epoch:33/60 || Epochiter:14/22 || loss:0.0012||Batchtime:0.1595||ETA:0:01:36
Epoch:33/60 || Epochiter:15/22 || loss:0.0012||Batchtime:0.1674||ETA:0:01:40
Epoch:33/60 || Epochiter:16/22 || loss:0.0011||Batchtime:0.1713||ETA:0:01:42
Epoch:33/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1718||ETA:0:01:43
Epoch:33/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1703||ETA:0:01:42
Epoch:33/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.4077||ETA:0:04:03
Epoch:33/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1743||ETA:0:01:44
Epoch:33/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1586||ETA:0:01:34
Epoch:33/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1695||ETA:0:01:40
valid_loss:0.0025677643716335297,valid_access:0.8809523809523809
Epoch:34/60 || Epochiter:1/22 || loss:0.0010||Batchtime:2.2304||ETA:0:22:04
Epoch:34/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.1525||ETA:0:01:30
Epoch:34/60 || Epochiter:3/22 || loss:0.0011||Batchtime:0.1339||ETA:0:01:19
Epoch:34/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1467||ETA:0:01:26
Epoch:34/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.2514||ETA:0:02:28
Epoch:34/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1510||ETA:0:01:28
Epoch:34/60 || Epochiter:7/22 || loss:0.0009||Batchtime:0.1783||ETA:0:01:44
Epoch:34/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1160||ETA:0:01:08
Epoch:34/60 || Epochiter:9/22 || loss:0.0014||Batchtime:0.1530||ETA:0:01:29
Epoch:34/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1421||ETA:0:01:23
Epoch:34/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1472||ETA:0:01:25
Epoch:34/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1399||ETA:0:01:21
Epoch:34/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.7067||ETA:0:06:51
Epoch:34/60 || Epochiter:14/22 || loss:0.0014||Batchtime:0.1656||ETA:0:01:36
Epoch:34/60 || Epochiter:15/22 || loss:0.0011||Batchtime:0.1652||ETA:0:01:35
Epoch:34/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1712||ETA:0:01:39
Epoch:34/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1720||ETA:0:01:39
Epoch:34/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1734||ETA:0:01:40
Epoch:34/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1855||ETA:0:01:46
Epoch:34/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1806||ETA:0:01:43
Epoch:34/60 || Epochiter:21/22 || loss:0.0014||Batchtime:0.1811||ETA:0:01:43
Epoch:34/60 || Epochiter:22/22 || loss:0.0013||Batchtime:0.1550||ETA:0:01:28
valid_loss:0.0024748060386627913,valid_access:0.8809523809523809
Epoch:35/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.4401||ETA:0:13:43
Epoch:35/60 || Epochiter:2/22 || loss:0.0011||Batchtime:0.1482||ETA:0:01:24
Epoch:35/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1509||ETA:0:01:26
Epoch:35/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.2754||ETA:0:02:36
Epoch:35/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.8080||ETA:0:07:38
Epoch:35/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1535||ETA:0:01:27
Epoch:35/60 || Epochiter:7/22 || loss:0.0012||Batchtime:0.1351||ETA:0:01:16
Epoch:35/60 || Epochiter:8/22 || loss:0.0014||Batchtime:0.1457||ETA:0:01:22
Epoch:35/60 || Epochiter:9/22 || loss:0.0013||Batchtime:0.1470||ETA:0:01:22
Epoch:35/60 || Epochiter:10/22 || loss:0.0015||Batchtime:0.2257||ETA:0:02:07
Epoch:35/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1496||ETA:0:01:24
Epoch:35/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1409||ETA:0:01:19
Epoch:35/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.2158||ETA:0:02:00
Epoch:35/60 || Epochiter:14/22 || loss:0.0013||Batchtime:0.1390||ETA:0:01:17
Epoch:35/60 || Epochiter:15/22 || loss:0.0011||Batchtime:0.1571||ETA:0:01:27
Epoch:35/60 || Epochiter:16/22 || loss:0.0011||Batchtime:0.1327||ETA:0:01:13
Epoch:35/60 || Epochiter:17/22 || loss:0.0015||Batchtime:0.7994||ETA:0:07:24
Epoch:35/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1659||ETA:0:01:32
Epoch:35/60 || Epochiter:19/22 || loss:0.0007||Batchtime:0.1586||ETA:0:01:27
Epoch:35/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.1798||ETA:0:01:39
Epoch:35/60 || Epochiter:21/22 || loss:0.0014||Batchtime:0.1864||ETA:0:01:42
Epoch:35/60 || Epochiter:22/22 || loss:0.0010||Batchtime:0.1621||ETA:0:01:29
valid_loss:0.002581488573923707,valid_access:0.8777777777777778
Epoch:36/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.5871||ETA:0:14:32
Epoch:36/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.3364||ETA:0:03:04
Epoch:36/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.4193||ETA:0:03:49
Epoch:36/60 || Epochiter:4/22 || loss:0.0016||Batchtime:0.1563||ETA:0:01:25
Epoch:36/60 || Epochiter:5/22 || loss:0.0009||Batchtime:0.1284||ETA:0:01:10
Epoch:36/60 || Epochiter:6/22 || loss:0.0013||Batchtime:0.4833||ETA:0:04:23
Epoch:36/60 || Epochiter:7/22 || loss:0.0014||Batchtime:0.2461||ETA:0:02:13
Epoch:36/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1314||ETA:0:01:11
Epoch:36/60 || Epochiter:9/22 || loss:0.0015||Batchtime:0.1649||ETA:0:01:29
Epoch:36/60 || Epochiter:10/22 || loss:0.0014||Batchtime:0.1609||ETA:0:01:27
Epoch:36/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1485||ETA:0:01:20
Epoch:36/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1314||ETA:0:01:10
Epoch:36/60 || Epochiter:13/22 || loss:0.0013||Batchtime:0.1296||ETA:0:01:09
Epoch:36/60 || Epochiter:14/22 || loss:0.0013||Batchtime:0.7052||ETA:0:06:18
Epoch:36/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1551||ETA:0:01:23
Epoch:36/60 || Epochiter:16/22 || loss:0.0012||Batchtime:0.1468||ETA:0:01:18
Epoch:36/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1678||ETA:0:01:29
Epoch:36/60 || Epochiter:18/22 || loss:0.0015||Batchtime:0.1654||ETA:0:01:28
Epoch:36/60 || Epochiter:19/22 || loss:0.0011||Batchtime:0.1650||ETA:0:01:27
Epoch:36/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1767||ETA:0:01:33
Epoch:36/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1767||ETA:0:01:33
Epoch:36/60 || Epochiter:22/22 || loss:0.0016||Batchtime:0.1800||ETA:0:01:35
valid_loss:0.0026575366500765085,valid_access:0.8746031746031746
Epoch:37/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.7078||ETA:0:15:01
Epoch:37/60 || Epochiter:2/22 || loss:0.0013||Batchtime:0.1430||ETA:0:01:15
Epoch:37/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1325||ETA:0:01:09
Epoch:37/60 || Epochiter:4/22 || loss:0.0013||Batchtime:1.1938||ETA:0:10:26
Epoch:37/60 || Epochiter:5/22 || loss:0.0013||Batchtime:0.1540||ETA:0:01:20
Epoch:37/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.2678||ETA:0:02:20
Epoch:37/60 || Epochiter:7/22 || loss:0.0010||Batchtime:0.1353||ETA:0:01:10
Epoch:37/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1625||ETA:0:01:24
Epoch:37/60 || Epochiter:9/22 || loss:0.0012||Batchtime:0.1407||ETA:0:01:13
Epoch:37/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.1490||ETA:0:01:17
Epoch:37/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1764||ETA:0:01:31
Epoch:37/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1585||ETA:0:01:21
Epoch:37/60 || Epochiter:13/22 || loss:0.0013||Batchtime:0.1857||ETA:0:01:35
Epoch:37/60 || Epochiter:14/22 || loss:0.0015||Batchtime:0.2040||ETA:0:01:45
Epoch:37/60 || Epochiter:15/22 || loss:0.0013||Batchtime:0.1722||ETA:0:01:28
Epoch:37/60 || Epochiter:16/22 || loss:0.0012||Batchtime:0.6135||ETA:0:05:14
Epoch:37/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1607||ETA:0:01:22
Epoch:37/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1566||ETA:0:01:19
Epoch:37/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1529||ETA:0:01:18
Epoch:37/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1627||ETA:0:01:22
Epoch:37/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1730||ETA:0:01:27
Epoch:37/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1741||ETA:0:01:28
valid_loss:0.0030366331338882446,valid_access:0.8611111111111112
Epoch:38/60 || Epochiter:1/22 || loss:0.0010||Batchtime:1.5793||ETA:0:13:19
Epoch:38/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.2164||ETA:0:01:49
Epoch:38/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1403||ETA:0:01:10
Epoch:38/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1453||ETA:0:01:13
Epoch:38/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.1504||ETA:0:01:15
Epoch:38/60 || Epochiter:6/22 || loss:0.0012||Batchtime:1.2941||ETA:0:10:48
Epoch:38/60 || Epochiter:7/22 || loss:0.0012||Batchtime:0.2714||ETA:0:02:15
Epoch:38/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1353||ETA:0:01:07
Epoch:38/60 || Epochiter:9/22 || loss:0.0012||Batchtime:0.1489||ETA:0:01:14
Epoch:38/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1525||ETA:0:01:15
Epoch:38/60 || Epochiter:11/22 || loss:0.0013||Batchtime:0.1691||ETA:0:01:23
Epoch:38/60 || Epochiter:12/22 || loss:0.0013||Batchtime:0.1833||ETA:0:01:30
Epoch:38/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.1758||ETA:0:01:26
Epoch:38/60 || Epochiter:14/22 || loss:0.0012||Batchtime:0.1659||ETA:0:01:21
Epoch:38/60 || Epochiter:15/22 || loss:0.0012||Batchtime:0.1621||ETA:0:01:19
Epoch:38/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1475||ETA:0:01:12
Epoch:38/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1590||ETA:0:01:17
Epoch:38/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.2664||ETA:0:02:10
Epoch:38/60 || Epochiter:19/22 || loss:0.0016||Batchtime:0.1517||ETA:0:01:14
Epoch:38/60 || Epochiter:20/22 || loss:0.0012||Batchtime:0.1569||ETA:0:01:16
Epoch:38/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1661||ETA:0:01:20
Epoch:38/60 || Epochiter:22/22 || loss:0.0013||Batchtime:0.1759||ETA:0:01:25
valid_loss:0.0025000947061926126,valid_access:0.8753968253968254
Epoch:39/60 || Epochiter:1/22 || loss:0.0014||Batchtime:2.0037||ETA:0:16:09
Epoch:39/60 || Epochiter:2/22 || loss:0.0016||Batchtime:0.1506||ETA:0:01:12
Epoch:39/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1533||ETA:0:01:13
Epoch:39/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1556||ETA:0:01:14
Epoch:39/60 || Epochiter:5/22 || loss:0.0012||Batchtime:0.1462||ETA:0:01:10
Epoch:39/60 || Epochiter:6/22 || loss:0.0009||Batchtime:0.2292||ETA:0:01:49
Epoch:39/60 || Epochiter:7/22 || loss:0.0016||Batchtime:0.1401||ETA:0:01:06
Epoch:39/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1462||ETA:0:01:09
Epoch:39/60 || Epochiter:9/22 || loss:0.0013||Batchtime:0.1408||ETA:0:01:07
Epoch:39/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.2669||ETA:0:02:06
Epoch:39/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1385||ETA:0:01:05
Epoch:39/60 || Epochiter:12/22 || loss:0.0010||Batchtime:0.1249||ETA:0:00:59
Epoch:39/60 || Epochiter:13/22 || loss:0.0013||Batchtime:0.6396||ETA:0:05:01
Epoch:39/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1766||ETA:0:01:23
Epoch:39/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.1713||ETA:0:01:20
Epoch:39/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.1767||ETA:0:01:22
Epoch:39/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1733||ETA:0:01:21
Epoch:39/60 || Epochiter:18/22 || loss:0.0015||Batchtime:0.1856||ETA:0:01:26
Epoch:39/60 || Epochiter:19/22 || loss:0.0011||Batchtime:0.1809||ETA:0:01:24
Epoch:39/60 || Epochiter:20/22 || loss:0.0014||Batchtime:0.1704||ETA:0:01:19
Epoch:39/60 || Epochiter:21/22 || loss:0.0014||Batchtime:0.1600||ETA:0:01:14
Epoch:39/60 || Epochiter:22/22 || loss:0.0011||Batchtime:0.1713||ETA:0:01:19
valid_loss:0.0033259245101362467,valid_access:0.8380952380952381
save weight success!!
Epoch:40/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.4224||ETA:0:10:57
Epoch:40/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.4644||ETA:0:03:34
Epoch:40/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1578||ETA:0:01:12
Epoch:40/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.8892||ETA:0:06:48
Epoch:40/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1470||ETA:0:01:07
Epoch:40/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1412||ETA:0:01:04
Epoch:40/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1412||ETA:0:01:04
Epoch:40/60 || Epochiter:8/22 || loss:0.0009||Batchtime:0.1248||ETA:0:00:56
Epoch:40/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.1265||ETA:0:00:57
Epoch:40/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1342||ETA:0:01:00
Epoch:40/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1585||ETA:0:01:11
Epoch:40/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1563||ETA:0:01:10
Epoch:40/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.2367||ETA:0:01:46
Epoch:40/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1601||ETA:0:01:11
Epoch:40/60 || Epochiter:15/22 || loss:0.0015||Batchtime:0.1574||ETA:0:01:10
Epoch:40/60 || Epochiter:16/22 || loss:0.0015||Batchtime:0.5603||ETA:0:04:10
Epoch:40/60 || Epochiter:17/22 || loss:0.0014||Batchtime:0.1532||ETA:0:01:08
Epoch:40/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1531||ETA:0:01:08
Epoch:40/60 || Epochiter:19/22 || loss:0.0016||Batchtime:0.1570||ETA:0:01:09
Epoch:40/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1687||ETA:0:01:14
Epoch:40/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1701||ETA:0:01:15
Epoch:40/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.1446||ETA:0:01:03
valid_loss:0.0027694732416421175,valid_access:0.8658730158730159
Epoch:41/60 || Epochiter:1/22 || loss:0.0013||Batchtime:2.1127||ETA:0:15:29
Epoch:41/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.1474||ETA:0:01:04
Epoch:41/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.1581||ETA:0:01:09
Epoch:41/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1519||ETA:0:01:06
Epoch:41/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1759||ETA:0:01:16
Epoch:41/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1492||ETA:0:01:04
Epoch:41/60 || Epochiter:7/22 || loss:0.0012||Batchtime:0.1454||ETA:0:01:03
Epoch:41/60 || Epochiter:8/22 || loss:0.0012||Batchtime:0.1572||ETA:0:01:08
Epoch:41/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.1486||ETA:0:01:04
Epoch:41/60 || Epochiter:10/22 || loss:0.0014||Batchtime:0.1479||ETA:0:01:03
Epoch:41/60 || Epochiter:11/22 || loss:0.0014||Batchtime:0.1405||ETA:0:01:00
Epoch:41/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1426||ETA:0:01:01
Epoch:41/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.9401||ETA:0:06:42
Epoch:41/60 || Epochiter:14/22 || loss:0.0013||Batchtime:0.1606||ETA:0:01:08
Epoch:41/60 || Epochiter:15/22 || loss:0.0015||Batchtime:0.1587||ETA:0:01:07
Epoch:41/60 || Epochiter:16/22 || loss:0.0013||Batchtime:0.2464||ETA:0:01:44
Epoch:41/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1654||ETA:0:01:10
Epoch:41/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1639||ETA:0:01:09
Epoch:41/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1644||ETA:0:01:09
Epoch:41/60 || Epochiter:20/22 || loss:0.0015||Batchtime:0.1708||ETA:0:01:11
Epoch:41/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.1762||ETA:0:01:14
Epoch:41/60 || Epochiter:22/22 || loss:0.0013||Batchtime:0.1695||ETA:0:01:11
valid_loss:0.0027124981861561537,valid_access:0.8555555555555555
Epoch:42/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.5164||ETA:0:10:33
Epoch:42/60 || Epochiter:2/22 || loss:0.0013||Batchtime:0.2082||ETA:0:01:26
Epoch:42/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.9435||ETA:0:06:32
Epoch:42/60 || Epochiter:4/22 || loss:0.0013||Batchtime:0.1444||ETA:0:00:59
Epoch:42/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1603||ETA:0:01:06
Epoch:42/60 || Epochiter:6/22 || loss:0.0009||Batchtime:0.3330||ETA:0:02:17
Epoch:42/60 || Epochiter:7/22 || loss:0.0014||Batchtime:0.1568||ETA:0:01:04
Epoch:42/60 || Epochiter:8/22 || loss:0.0009||Batchtime:0.1570||ETA:0:01:04
Epoch:42/60 || Epochiter:9/22 || loss:0.0012||Batchtime:0.1651||ETA:0:01:07
Epoch:42/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.1462||ETA:0:00:59
Epoch:42/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1472||ETA:0:01:00
Epoch:42/60 || Epochiter:12/22 || loss:0.0012||Batchtime:0.1761||ETA:0:01:11
Epoch:42/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.1521||ETA:0:01:01
Epoch:42/60 || Epochiter:14/22 || loss:0.0015||Batchtime:0.1619||ETA:0:01:05
Epoch:42/60 || Epochiter:15/22 || loss:0.0012||Batchtime:0.3666||ETA:0:02:28
Epoch:42/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.2657||ETA:0:01:47
Epoch:42/60 || Epochiter:17/22 || loss:0.0012||Batchtime:0.1693||ETA:0:01:08
Epoch:42/60 || Epochiter:18/22 || loss:0.0013||Batchtime:0.1759||ETA:0:01:10
Epoch:42/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1743||ETA:0:01:09
Epoch:42/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.1742||ETA:0:01:09
Epoch:42/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.1794||ETA:0:01:11
Epoch:42/60 || Epochiter:22/22 || loss:0.0008||Batchtime:0.1720||ETA:0:01:08
valid_loss:0.0030495820101350546,valid_access:0.8603174603174604
Epoch:43/60 || Epochiter:1/22 || loss:0.0013||Batchtime:1.3572||ETA:0:08:57
Epoch:43/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.1608||ETA:0:01:03
Epoch:43/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.3030||ETA:0:01:59
Epoch:43/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.3858||ETA:0:02:31
Epoch:43/60 || Epochiter:5/22 || loss:0.0009||Batchtime:0.1423||ETA:0:00:55
Epoch:43/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.1282||ETA:0:00:50
Epoch:43/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.3898||ETA:0:02:32
Epoch:43/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1403||ETA:0:00:54
Epoch:43/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1348||ETA:0:00:52
Epoch:43/60 || Epochiter:10/22 || loss:0.0014||Batchtime:0.1419||ETA:0:00:54
Epoch:43/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.2257||ETA:0:01:27
Epoch:43/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1364||ETA:0:00:52
Epoch:43/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.1134||ETA:0:00:43
Epoch:43/60 || Epochiter:14/22 || loss:0.0011||Batchtime:0.2652||ETA:0:01:41
Epoch:43/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1837||ETA:0:01:10
Epoch:43/60 || Epochiter:16/22 || loss:0.0010||Batchtime:0.3213||ETA:0:02:02
Epoch:43/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.2000||ETA:0:01:15
Epoch:43/60 || Epochiter:18/22 || loss:0.0012||Batchtime:0.1786||ETA:0:01:07
Epoch:43/60 || Epochiter:19/22 || loss:0.0008||Batchtime:0.1665||ETA:0:01:02
Epoch:43/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.2583||ETA:0:01:37
Epoch:43/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1727||ETA:0:01:04
Epoch:43/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1806||ETA:0:01:07
valid_loss:0.0029716293793171644,valid_access:0.8666666666666667
Epoch:44/60 || Epochiter:1/22 || loss:0.0009||Batchtime:1.4684||ETA:0:09:09
Epoch:44/60 || Epochiter:2/22 || loss:0.0011||Batchtime:0.1483||ETA:0:00:55
Epoch:44/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.1514||ETA:0:00:56
Epoch:44/60 || Epochiter:4/22 || loss:0.0014||Batchtime:1.2103||ETA:0:07:29
Epoch:44/60 || Epochiter:5/22 || loss:0.0008||Batchtime:0.1838||ETA:0:01:07
Epoch:44/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.1331||ETA:0:00:49
Epoch:44/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1436||ETA:0:00:52
Epoch:44/60 || Epochiter:8/22 || loss:0.0008||Batchtime:0.1628||ETA:0:00:59
Epoch:44/60 || Epochiter:9/22 || loss:0.0013||Batchtime:0.1518||ETA:0:00:55
Epoch:44/60 || Epochiter:10/22 || loss:0.0009||Batchtime:0.1603||ETA:0:00:58
Epoch:44/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1510||ETA:0:00:54
Epoch:44/60 || Epochiter:12/22 || loss:0.0010||Batchtime:0.1604||ETA:0:00:58
Epoch:44/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.1609||ETA:0:00:58
Epoch:44/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1714||ETA:0:01:01
Epoch:44/60 || Epochiter:15/22 || loss:0.0011||Batchtime:0.1845||ETA:0:01:06
Epoch:44/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.4903||ETA:0:02:56
Epoch:44/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1749||ETA:0:01:02
Epoch:44/60 || Epochiter:18/22 || loss:0.0013||Batchtime:0.1808||ETA:0:01:04
Epoch:44/60 || Epochiter:19/22 || loss:0.0010||Batchtime:0.1807||ETA:0:01:04
Epoch:44/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1867||ETA:0:01:06
Epoch:44/60 || Epochiter:21/22 || loss:0.0008||Batchtime:0.2818||ETA:0:01:39
Epoch:44/60 || Epochiter:22/22 || loss:0.0014||Batchtime:0.1908||ETA:0:01:07
valid_loss:0.002710900269448757,valid_access:0.8706349206349207
Epoch:45/60 || Epochiter:1/22 || loss:0.0007||Batchtime:1.8291||ETA:0:10:43
Epoch:45/60 || Epochiter:2/22 || loss:0.0012||Batchtime:0.1409||ETA:0:00:49
Epoch:45/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.1412||ETA:0:00:49
Epoch:45/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1454||ETA:0:00:50
Epoch:45/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1508||ETA:0:00:52
Epoch:45/60 || Epochiter:6/22 || loss:0.0013||Batchtime:0.1858||ETA:0:01:04
Epoch:45/60 || Epochiter:7/22 || loss:0.0013||Batchtime:0.3792||ETA:0:02:11
Epoch:45/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1451||ETA:0:00:50
Epoch:45/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.7541||ETA:0:04:19
Epoch:45/60 || Epochiter:10/22 || loss:0.0008||Batchtime:0.1656||ETA:0:00:56
Epoch:45/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1643||ETA:0:00:56
Epoch:45/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1865||ETA:0:01:03
Epoch:45/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.1514||ETA:0:00:51
Epoch:45/60 || Epochiter:14/22 || loss:0.0012||Batchtime:0.1580||ETA:0:00:53
Epoch:45/60 || Epochiter:15/22 || loss:0.0011||Batchtime:0.1706||ETA:0:00:57
Epoch:45/60 || Epochiter:16/22 || loss:0.0010||Batchtime:0.1708||ETA:0:00:57
Epoch:45/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1778||ETA:0:00:59
Epoch:45/60 || Epochiter:18/22 || loss:0.0008||Batchtime:0.1727||ETA:0:00:57
Epoch:45/60 || Epochiter:19/22 || loss:0.0012||Batchtime:0.1703||ETA:0:00:56
Epoch:45/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1644||ETA:0:00:54
Epoch:45/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.1697||ETA:0:00:56
Epoch:45/60 || Epochiter:22/22 || loss:0.0008||Batchtime:0.1727||ETA:0:00:57
valid_loss:0.0026065087877213955,valid_access:0.8722222222222222
Epoch:46/60 || Epochiter:1/22 || loss:0.0010||Batchtime:1.7055||ETA:0:09:22
Epoch:46/60 || Epochiter:2/22 || loss:0.0009||Batchtime:0.4555||ETA:0:02:29
Epoch:46/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1436||ETA:0:00:47
Epoch:46/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1462||ETA:0:00:47
Epoch:46/60 || Epochiter:5/22 || loss:0.0008||Batchtime:0.2532||ETA:0:01:22
Epoch:46/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1417||ETA:0:00:46
Epoch:46/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1428||ETA:0:00:46
Epoch:46/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1384||ETA:0:00:44
Epoch:46/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1162||ETA:0:00:37
Epoch:46/60 || Epochiter:10/22 || loss:0.0012||Batchtime:0.4028||ETA:0:02:09
Epoch:46/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1429||ETA:0:00:45
Epoch:46/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1731||ETA:0:00:55
Epoch:46/60 || Epochiter:13/22 || loss:0.0008||Batchtime:0.1368||ETA:0:00:43
Epoch:46/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.5617||ETA:0:02:58
Epoch:46/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1638||ETA:0:00:51
Epoch:46/60 || Epochiter:16/22 || loss:0.0008||Batchtime:0.1710||ETA:0:00:53
Epoch:46/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1720||ETA:0:00:54
Epoch:46/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1726||ETA:0:00:54
Epoch:46/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1650||ETA:0:00:51
Epoch:46/60 || Epochiter:20/22 || loss:0.0008||Batchtime:0.1502||ETA:0:00:46
Epoch:46/60 || Epochiter:21/22 || loss:0.0009||Batchtime:0.1592||ETA:0:00:49
Epoch:46/60 || Epochiter:22/22 || loss:0.0010||Batchtime:0.1749||ETA:0:00:54
valid_loss:0.002581855747848749,valid_access:0.8785714285714286
Epoch:47/60 || Epochiter:1/22 || loss:0.0009||Batchtime:2.2601||ETA:0:11:36
Epoch:47/60 || Epochiter:2/22 || loss:0.0008||Batchtime:0.1478||ETA:0:00:45
Epoch:47/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1322||ETA:0:00:40
Epoch:47/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1354||ETA:0:00:41
Epoch:47/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1507||ETA:0:00:45
Epoch:47/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1529||ETA:0:00:46
Epoch:47/60 || Epochiter:7/22 || loss:0.0008||Batchtime:0.1340||ETA:0:00:40
Epoch:47/60 || Epochiter:8/22 || loss:0.0009||Batchtime:0.1371||ETA:0:00:41
Epoch:47/60 || Epochiter:9/22 || loss:0.0007||Batchtime:0.1859||ETA:0:00:55
Epoch:47/60 || Epochiter:10/22 || loss:0.0009||Batchtime:0.1636||ETA:0:00:48
Epoch:47/60 || Epochiter:11/22 || loss:0.0008||Batchtime:0.1466||ETA:0:00:43
Epoch:47/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1513||ETA:0:00:44
Epoch:47/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.9154||ETA:0:04:30
Epoch:47/60 || Epochiter:14/22 || loss:0.0011||Batchtime:0.1681||ETA:0:00:49
Epoch:47/60 || Epochiter:15/22 || loss:0.0007||Batchtime:0.1640||ETA:0:00:48
Epoch:47/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.1655||ETA:0:00:48
Epoch:47/60 || Epochiter:17/22 || loss:0.0007||Batchtime:0.1485||ETA:0:00:43
Epoch:47/60 || Epochiter:18/22 || loss:0.0008||Batchtime:0.1831||ETA:0:00:53
Epoch:47/60 || Epochiter:19/22 || loss:0.0008||Batchtime:0.1854||ETA:0:00:53
Epoch:47/60 || Epochiter:20/22 || loss:0.0008||Batchtime:0.1775||ETA:0:00:51
Epoch:47/60 || Epochiter:21/22 || loss:0.0007||Batchtime:0.1873||ETA:0:00:53
Epoch:47/60 || Epochiter:22/22 || loss:0.0008||Batchtime:0.1745||ETA:0:00:50
valid_loss:0.002569845411926508,valid_access:0.8801587301587301
Epoch:48/60 || Epochiter:1/22 || loss:0.0008||Batchtime:1.9695||ETA:0:09:23
Epoch:48/60 || Epochiter:2/22 || loss:0.0008||Batchtime:0.1415||ETA:0:00:40
Epoch:48/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1753||ETA:0:00:49
Epoch:48/60 || Epochiter:4/22 || loss:0.0007||Batchtime:0.1346||ETA:0:00:38
Epoch:48/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1426||ETA:0:00:40
Epoch:48/60 || Epochiter:6/22 || loss:0.0007||Batchtime:0.3559||ETA:0:01:40
Epoch:48/60 || Epochiter:7/22 || loss:0.0009||Batchtime:0.7396||ETA:0:03:27
Epoch:48/60 || Epochiter:8/22 || loss:0.0008||Batchtime:0.1454||ETA:0:00:40
Epoch:48/60 || Epochiter:9/22 || loss:0.0009||Batchtime:0.1281||ETA:0:00:35
Epoch:48/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1410||ETA:0:00:39
Epoch:48/60 || Epochiter:11/22 || loss:0.0007||Batchtime:0.1395||ETA:0:00:38
Epoch:48/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1573||ETA:0:00:43
Epoch:48/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1501||ETA:0:00:41
Epoch:48/60 || Epochiter:14/22 || loss:0.0011||Batchtime:0.1688||ETA:0:00:46
Epoch:48/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.3073||ETA:0:01:23
Epoch:48/60 || Epochiter:16/22 || loss:0.0008||Batchtime:0.1733||ETA:0:00:46
Epoch:48/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1710||ETA:0:00:46
Epoch:48/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1793||ETA:0:00:48
Epoch:48/60 || Epochiter:19/22 || loss:0.0007||Batchtime:0.4695||ETA:0:02:05
Epoch:48/60 || Epochiter:20/22 || loss:0.0007||Batchtime:0.1765||ETA:0:00:47
Epoch:48/60 || Epochiter:21/22 || loss:0.0008||Batchtime:0.1751||ETA:0:00:46
Epoch:48/60 || Epochiter:22/22 || loss:0.0008||Batchtime:0.1750||ETA:0:00:46
valid_loss:0.002570601413026452,valid_access:0.8785714285714286
Epoch:49/60 || Epochiter:1/22 || loss:0.0009||Batchtime:1.3976||ETA:0:06:08
Epoch:49/60 || Epochiter:2/22 || loss:0.0008||Batchtime:1.0779||ETA:0:04:43
Epoch:49/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1593||ETA:0:00:41
Epoch:49/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1586||ETA:0:00:41
Epoch:49/60 || Epochiter:5/22 || loss:0.0008||Batchtime:0.1216||ETA:0:00:31
Epoch:49/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1335||ETA:0:00:34
Epoch:49/60 || Epochiter:7/22 || loss:0.0009||Batchtime:0.1529||ETA:0:00:39
Epoch:49/60 || Epochiter:8/22 || loss:0.0008||Batchtime:0.1481||ETA:0:00:38
Epoch:49/60 || Epochiter:9/22 || loss:0.0006||Batchtime:0.2877||ETA:0:01:13
Epoch:49/60 || Epochiter:10/22 || loss:0.0007||Batchtime:0.1388||ETA:0:00:35
Epoch:49/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1619||ETA:0:00:41
Epoch:49/60 || Epochiter:12/22 || loss:0.0007||Batchtime:0.1707||ETA:0:00:43
Epoch:49/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1621||ETA:0:00:40
Epoch:49/60 || Epochiter:14/22 || loss:0.0008||Batchtime:0.5798||ETA:0:02:25
Epoch:49/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.1700||ETA:0:00:42
Epoch:49/60 || Epochiter:16/22 || loss:0.0007||Batchtime:0.1584||ETA:0:00:39
Epoch:49/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1848||ETA:0:00:45
Epoch:49/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1727||ETA:0:00:42
Epoch:49/60 || Epochiter:19/22 || loss:0.0011||Batchtime:0.1551||ETA:0:00:38
Epoch:49/60 || Epochiter:20/22 || loss:0.0008||Batchtime:0.1540||ETA:0:00:37
Epoch:49/60 || Epochiter:21/22 || loss:0.0006||Batchtime:0.1526||ETA:0:00:37
Epoch:49/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1598||ETA:0:00:38
valid_loss:0.002599739469587803,valid_access:0.8801587301587301
save weight success!!
Epoch:50/60 || Epochiter:1/22 || loss:0.0009||Batchtime:2.0261||ETA:0:08:10
Epoch:50/60 || Epochiter:2/22 || loss:0.0009||Batchtime:0.1412||ETA:0:00:34
Epoch:50/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.1528||ETA:0:00:36
Epoch:50/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1410||ETA:0:00:33
Epoch:50/60 || Epochiter:5/22 || loss:0.0009||Batchtime:0.1432||ETA:0:00:34
Epoch:50/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1494||ETA:0:00:35
Epoch:50/60 || Epochiter:7/22 || loss:0.0007||Batchtime:0.1435||ETA:0:00:33
Epoch:50/60 || Epochiter:8/22 || loss:0.0008||Batchtime:0.1507||ETA:0:00:35
Epoch:50/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.2344||ETA:0:00:54
Epoch:50/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1398||ETA:0:00:32
Epoch:50/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1456||ETA:0:00:33
Epoch:50/60 || Epochiter:12/22 || loss:0.0008||Batchtime:0.1290||ETA:0:00:29
Epoch:50/60 || Epochiter:13/22 || loss:0.0007||Batchtime:0.6173||ETA:0:02:21
Epoch:50/60 || Epochiter:14/22 || loss:0.0009||Batchtime:0.1421||ETA:0:00:32
Epoch:50/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1518||ETA:0:00:34
Epoch:50/60 || Epochiter:16/22 || loss:0.0005||Batchtime:0.1716||ETA:0:00:38
Epoch:50/60 || Epochiter:17/22 || loss:0.0009||Batchtime:0.2666||ETA:0:01:00
Epoch:50/60 || Epochiter:18/22 || loss:0.0006||Batchtime:0.1705||ETA:0:00:38
Epoch:50/60 || Epochiter:19/22 || loss:0.0008||Batchtime:0.1755||ETA:0:00:39
Epoch:50/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1791||ETA:0:00:39
Epoch:50/60 || Epochiter:21/22 || loss:0.0007||Batchtime:0.1824||ETA:0:00:40
Epoch:50/60 || Epochiter:22/22 || loss:0.0007||Batchtime:0.1789||ETA:0:00:39
valid_loss:0.0025712414644658566,valid_access:0.8825396825396825
Epoch:51/60 || Epochiter:1/22 || loss:0.0006||Batchtime:1.4867||ETA:0:05:27
Epoch:51/60 || Epochiter:2/22 || loss:0.0009||Batchtime:0.1501||ETA:0:00:32
Epoch:51/60 || Epochiter:3/22 || loss:0.0009||Batchtime:0.2128||ETA:0:00:46
Epoch:51/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1441||ETA:0:00:31
Epoch:51/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.2498||ETA:0:00:53
Epoch:51/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1499||ETA:0:00:32
Epoch:51/60 || Epochiter:7/22 || loss:0.0009||Batchtime:0.6104||ETA:0:02:10
Epoch:51/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1302||ETA:0:00:27
Epoch:51/60 || Epochiter:9/22 || loss:0.0008||Batchtime:0.2837||ETA:0:01:00
Epoch:51/60 || Epochiter:10/22 || loss:0.0006||Batchtime:0.1393||ETA:0:00:29
Epoch:51/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1628||ETA:0:00:34
Epoch:51/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1450||ETA:0:00:30
Epoch:51/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.1622||ETA:0:00:33
Epoch:51/60 || Epochiter:14/22 || loss:0.0007||Batchtime:0.1607||ETA:0:00:33
Epoch:51/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.3523||ETA:0:01:12
Epoch:51/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.1256||ETA:0:00:25
Epoch:51/60 || Epochiter:17/22 || loss:0.0008||Batchtime:0.1807||ETA:0:00:36
Epoch:51/60 || Epochiter:18/22 || loss:0.0008||Batchtime:0.1772||ETA:0:00:35
Epoch:51/60 || Epochiter:19/22 || loss:0.0007||Batchtime:0.4765||ETA:0:01:36
Epoch:51/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1655||ETA:0:00:33
Epoch:51/60 || Epochiter:21/22 || loss:0.0007||Batchtime:0.1788||ETA:0:00:35
Epoch:51/60 || Epochiter:22/22 || loss:0.0007||Batchtime:0.1906||ETA:0:00:37
valid_loss:0.002618425292894244,valid_access:0.8785714285714286
Epoch:52/60 || Epochiter:1/22 || loss:0.0009||Batchtime:1.4794||ETA:0:04:52
Epoch:52/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.1989||ETA:0:00:39
Epoch:52/60 || Epochiter:3/22 || loss:0.0008||Batchtime:0.1398||ETA:0:00:27
Epoch:52/60 || Epochiter:4/22 || loss:0.0008||Batchtime:0.6221||ETA:0:02:01
Epoch:52/60 || Epochiter:5/22 || loss:0.0006||Batchtime:0.1365||ETA:0:00:26
Epoch:52/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.6793||ETA:0:02:11
Epoch:52/60 || Epochiter:7/22 || loss:0.0008||Batchtime:0.1403||ETA:0:00:26
Epoch:52/60 || Epochiter:8/22 || loss:0.0012||Batchtime:0.1412||ETA:0:00:26
Epoch:52/60 || Epochiter:9/22 || loss:0.0008||Batchtime:0.1465||ETA:0:00:27
Epoch:52/60 || Epochiter:10/22 || loss:0.0008||Batchtime:0.1525||ETA:0:00:28
Epoch:52/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1515||ETA:0:00:28
Epoch:52/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1601||ETA:0:00:29
Epoch:52/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1867||ETA:0:00:34
Epoch:52/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1715||ETA:0:00:31
Epoch:52/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.1589||ETA:0:00:29
Epoch:52/60 || Epochiter:16/22 || loss:0.0007||Batchtime:0.1675||ETA:0:00:30
Epoch:52/60 || Epochiter:17/22 || loss:0.0009||Batchtime:0.1554||ETA:0:00:28
Epoch:52/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.3614||ETA:0:01:05
Epoch:52/60 || Epochiter:19/22 || loss:0.0007||Batchtime:0.1767||ETA:0:00:31
Epoch:52/60 || Epochiter:20/22 || loss:0.0007||Batchtime:0.1753||ETA:0:00:31
Epoch:52/60 || Epochiter:21/22 || loss:0.0012||Batchtime:0.1803||ETA:0:00:32
Epoch:52/60 || Epochiter:22/22 || loss:0.0011||Batchtime:0.1784||ETA:0:00:31
valid_loss:0.0027672152500599623,valid_access:0.8753968253968254
Epoch:53/60 || Epochiter:1/22 || loss:0.0008||Batchtime:2.3415||ETA:0:06:52
Epoch:53/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.1573||ETA:0:00:27
Epoch:53/60 || Epochiter:3/22 || loss:0.0012||Batchtime:0.2581||ETA:0:00:44
Epoch:53/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1467||ETA:0:00:25
Epoch:53/60 || Epochiter:5/22 || loss:0.0011||Batchtime:0.1688||ETA:0:00:29
Epoch:53/60 || Epochiter:6/22 || loss:0.0007||Batchtime:0.1410||ETA:0:00:24
Epoch:53/60 || Epochiter:7/22 || loss:0.0009||Batchtime:0.1505||ETA:0:00:25
Epoch:53/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1519||ETA:0:00:25
Epoch:53/60 || Epochiter:9/22 || loss:0.0011||Batchtime:0.1500||ETA:0:00:25
Epoch:53/60 || Epochiter:10/22 || loss:0.0007||Batchtime:0.1454||ETA:0:00:24
Epoch:53/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1580||ETA:0:00:26
Epoch:53/60 || Epochiter:12/22 || loss:0.0008||Batchtime:0.1402||ETA:0:00:23
Epoch:53/60 || Epochiter:13/22 || loss:0.0005||Batchtime:0.7285||ETA:0:01:59
Epoch:53/60 || Epochiter:14/22 || loss:0.0008||Batchtime:0.1711||ETA:0:00:27
Epoch:53/60 || Epochiter:15/22 || loss:0.0006||Batchtime:0.1640||ETA:0:00:26
Epoch:53/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.1819||ETA:0:00:29
Epoch:53/60 || Epochiter:17/22 || loss:0.0007||Batchtime:0.1693||ETA:0:00:27
Epoch:53/60 || Epochiter:18/22 || loss:0.0007||Batchtime:0.1758||ETA:0:00:27
Epoch:53/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1740||ETA:0:00:27
Epoch:53/60 || Epochiter:20/22 || loss:0.0011||Batchtime:0.2002||ETA:0:00:31
Epoch:53/60 || Epochiter:21/22 || loss:0.0008||Batchtime:0.1893||ETA:0:00:29
Epoch:53/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1845||ETA:0:00:28
valid_loss:0.002667673397809267,valid_access:0.8690476190476191
Epoch:54/60 || Epochiter:1/22 || loss:0.0009||Batchtime:2.4879||ETA:0:06:23
Epoch:54/60 || Epochiter:2/22 || loss:0.0007||Batchtime:0.1321||ETA:0:00:20
Epoch:54/60 || Epochiter:3/22 || loss:0.0008||Batchtime:0.1517||ETA:0:00:23
Epoch:54/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1517||ETA:0:00:22
Epoch:54/60 || Epochiter:5/22 || loss:0.0008||Batchtime:0.1349||ETA:0:00:20
Epoch:54/60 || Epochiter:6/22 || loss:0.0008||Batchtime:0.1739||ETA:0:00:25
Epoch:54/60 || Epochiter:7/22 || loss:0.0010||Batchtime:0.2847||ETA:0:00:42
Epoch:54/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1523||ETA:0:00:22
Epoch:54/60 || Epochiter:9/22 || loss:0.0009||Batchtime:0.1522||ETA:0:00:22
Epoch:54/60 || Epochiter:10/22 || loss:0.0009||Batchtime:0.1411||ETA:0:00:20
Epoch:54/60 || Epochiter:11/22 || loss:0.0008||Batchtime:0.2859||ETA:0:00:41
Epoch:54/60 || Epochiter:12/22 || loss:0.0008||Batchtime:0.1373||ETA:0:00:19
Epoch:54/60 || Epochiter:13/22 || loss:0.0011||Batchtime:0.4594||ETA:0:01:05
Epoch:54/60 || Epochiter:14/22 || loss:0.0008||Batchtime:0.1727||ETA:0:00:24
Epoch:54/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.1586||ETA:0:00:22
Epoch:54/60 || Epochiter:16/22 || loss:0.0010||Batchtime:0.1722||ETA:0:00:23
Epoch:54/60 || Epochiter:17/22 || loss:0.0013||Batchtime:0.1748||ETA:0:00:24
Epoch:54/60 || Epochiter:18/22 || loss:0.0011||Batchtime:0.1747||ETA:0:00:23
Epoch:54/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1760||ETA:0:00:23
Epoch:54/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1816||ETA:0:00:24
Epoch:54/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.1740||ETA:0:00:23
Epoch:54/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1804||ETA:0:00:23
valid_loss:0.0031203092075884342,valid_access:0.8587301587301587
Epoch:55/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.4832||ETA:0:03:15
Epoch:55/60 || Epochiter:2/22 || loss:0.0008||Batchtime:0.8422||ETA:0:01:50
Epoch:55/60 || Epochiter:3/22 || loss:0.0008||Batchtime:0.1665||ETA:0:00:21
Epoch:55/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1413||ETA:0:00:18
Epoch:55/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1794||ETA:0:00:22
Epoch:55/60 || Epochiter:6/22 || loss:0.0010||Batchtime:0.1467||ETA:0:00:18
Epoch:55/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1352||ETA:0:00:17
Epoch:55/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1412||ETA:0:00:17
Epoch:55/60 || Epochiter:9/22 || loss:0.0009||Batchtime:0.1151||ETA:0:00:14
Epoch:55/60 || Epochiter:10/22 || loss:0.0011||Batchtime:0.1511||ETA:0:00:18
Epoch:55/60 || Epochiter:11/22 || loss:0.0008||Batchtime:0.1398||ETA:0:00:17
Epoch:55/60 || Epochiter:12/22 || loss:0.0013||Batchtime:0.1536||ETA:0:00:18
Epoch:55/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1447||ETA:0:00:17
Epoch:55/60 || Epochiter:14/22 || loss:0.0007||Batchtime:0.9254||ETA:0:01:50
Epoch:55/60 || Epochiter:15/22 || loss:0.0008||Batchtime:0.1626||ETA:0:00:19
Epoch:55/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.1740||ETA:0:00:20
Epoch:55/60 || Epochiter:17/22 || loss:0.0011||Batchtime:0.1873||ETA:0:00:21
Epoch:55/60 || Epochiter:18/22 || loss:0.0011||Batchtime:0.1773||ETA:0:00:20
Epoch:55/60 || Epochiter:19/22 || loss:0.0009||Batchtime:0.1762||ETA:0:00:20
Epoch:55/60 || Epochiter:20/22 || loss:0.0008||Batchtime:0.1762||ETA:0:00:19
Epoch:55/60 || Epochiter:21/22 || loss:0.0010||Batchtime:0.2716||ETA:0:00:30
Epoch:55/60 || Epochiter:22/22 || loss:0.0013||Batchtime:0.1746||ETA:0:00:19
valid_loss:0.0029390992131084204,valid_access:0.8642857142857143
Epoch:56/60 || Epochiter:1/22 || loss:0.0011||Batchtime:2.0856||ETA:0:03:49
Epoch:56/60 || Epochiter:2/22 || loss:0.0009||Batchtime:0.1603||ETA:0:00:17
Epoch:56/60 || Epochiter:3/22 || loss:0.0013||Batchtime:0.4258||ETA:0:00:45
Epoch:56/60 || Epochiter:4/22 || loss:0.0011||Batchtime:0.1134||ETA:0:00:12
Epoch:56/60 || Epochiter:5/22 || loss:0.0009||Batchtime:0.1351||ETA:0:00:14
Epoch:56/60 || Epochiter:6/22 || loss:0.0009||Batchtime:0.1191||ETA:0:00:12
Epoch:56/60 || Epochiter:7/22 || loss:0.0013||Batchtime:0.1566||ETA:0:00:16
Epoch:56/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.1428||ETA:0:00:14
Epoch:56/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1415||ETA:0:00:14
Epoch:56/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1434||ETA:0:00:14
Epoch:56/60 || Epochiter:11/22 || loss:0.0010||Batchtime:0.1604||ETA:0:00:16
Epoch:56/60 || Epochiter:12/22 || loss:0.0013||Batchtime:0.1406||ETA:0:00:13
Epoch:56/60 || Epochiter:13/22 || loss:0.0009||Batchtime:0.7366||ETA:0:01:12
Epoch:56/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1795||ETA:0:00:17
Epoch:56/60 || Epochiter:15/22 || loss:0.0012||Batchtime:0.2767||ETA:0:00:26
Epoch:56/60 || Epochiter:16/22 || loss:0.0007||Batchtime:0.1589||ETA:0:00:15
Epoch:56/60 || Epochiter:17/22 || loss:0.0008||Batchtime:0.1638||ETA:0:00:15
Epoch:56/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1761||ETA:0:00:16
Epoch:56/60 || Epochiter:19/22 || loss:0.0013||Batchtime:0.1772||ETA:0:00:16
Epoch:56/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1743||ETA:0:00:15
Epoch:56/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.1797||ETA:0:00:16
Epoch:56/60 || Epochiter:22/22 || loss:0.0009||Batchtime:0.1869||ETA:0:00:16
valid_loss:0.0031623276881873608,valid_access:0.8571428571428571
Epoch:57/60 || Epochiter:1/22 || loss:0.0011||Batchtime:1.5237||ETA:0:02:14
Epoch:57/60 || Epochiter:2/22 || loss:0.0011||Batchtime:0.1256||ETA:0:00:10
Epoch:57/60 || Epochiter:3/22 || loss:0.0010||Batchtime:0.2882||ETA:0:00:24
Epoch:57/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1627||ETA:0:00:13
Epoch:57/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1813||ETA:0:00:15
Epoch:57/60 || Epochiter:6/22 || loss:0.0009||Batchtime:0.1712||ETA:0:00:14
Epoch:57/60 || Epochiter:7/22 || loss:0.0010||Batchtime:0.1638||ETA:0:00:13
Epoch:57/60 || Epochiter:8/22 || loss:0.0011||Batchtime:0.6697||ETA:0:00:54
Epoch:57/60 || Epochiter:9/22 || loss:0.0009||Batchtime:0.3190||ETA:0:00:25
Epoch:57/60 || Epochiter:10/22 || loss:0.0013||Batchtime:0.1415||ETA:0:00:11
Epoch:57/60 || Epochiter:11/22 || loss:0.0011||Batchtime:0.1740||ETA:0:00:13
Epoch:57/60 || Epochiter:12/22 || loss:0.0010||Batchtime:0.1398||ETA:0:00:10
Epoch:57/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1652||ETA:0:00:12
Epoch:57/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1487||ETA:0:00:11
Epoch:57/60 || Epochiter:15/22 || loss:0.0014||Batchtime:0.2050||ETA:0:00:15
Epoch:57/60 || Epochiter:16/22 || loss:0.0009||Batchtime:0.1470||ETA:0:00:10
Epoch:57/60 || Epochiter:17/22 || loss:0.0012||Batchtime:0.1735||ETA:0:00:12
Epoch:57/60 || Epochiter:18/22 || loss:0.0009||Batchtime:0.1616||ETA:0:00:11
Epoch:57/60 || Epochiter:19/22 || loss:0.0012||Batchtime:0.1734||ETA:0:00:12
Epoch:57/60 || Epochiter:20/22 || loss:0.0012||Batchtime:0.1701||ETA:0:00:11
Epoch:57/60 || Epochiter:21/22 || loss:0.0011||Batchtime:0.3652||ETA:0:00:24
Epoch:57/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1826||ETA:0:00:12
valid_loss:0.0029387727845460176,valid_access:0.8650793650793651
Epoch:58/60 || Epochiter:1/22 || loss:0.0012||Batchtime:1.5340||ETA:0:01:41
Epoch:58/60 || Epochiter:2/22 || loss:0.0013||Batchtime:0.2101||ETA:0:00:13
Epoch:58/60 || Epochiter:3/22 || loss:0.0011||Batchtime:0.2906||ETA:0:00:18
Epoch:58/60 || Epochiter:4/22 || loss:0.0009||Batchtime:0.1522||ETA:0:00:09
Epoch:58/60 || Epochiter:5/22 || loss:0.0010||Batchtime:0.1299||ETA:0:00:08
Epoch:58/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.1366||ETA:0:00:08
Epoch:58/60 || Epochiter:7/22 || loss:0.0010||Batchtime:1.1649||ETA:0:01:09
Epoch:58/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1387||ETA:0:00:08
Epoch:58/60 || Epochiter:9/22 || loss:0.0013||Batchtime:0.2748||ETA:0:00:15
Epoch:58/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1542||ETA:0:00:08
Epoch:58/60 || Epochiter:11/22 || loss:0.0009||Batchtime:0.1504||ETA:0:00:08
Epoch:58/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1416||ETA:0:00:07
Epoch:58/60 || Epochiter:13/22 || loss:0.0012||Batchtime:0.1761||ETA:0:00:09
Epoch:58/60 || Epochiter:14/22 || loss:0.0012||Batchtime:0.1705||ETA:0:00:09
Epoch:58/60 || Epochiter:15/22 || loss:0.0010||Batchtime:0.1921||ETA:0:00:09
Epoch:58/60 || Epochiter:16/22 || loss:0.0012||Batchtime:0.1756||ETA:0:00:08
Epoch:58/60 || Epochiter:17/22 || loss:0.0012||Batchtime:0.1677||ETA:0:00:08
Epoch:58/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1699||ETA:0:00:08
Epoch:58/60 || Epochiter:19/22 || loss:0.0012||Batchtime:0.2108||ETA:0:00:10
Epoch:58/60 || Epochiter:20/22 || loss:0.0009||Batchtime:0.1646||ETA:0:00:07
Epoch:58/60 || Epochiter:21/22 || loss:0.0013||Batchtime:0.1638||ETA:0:00:07
Epoch:58/60 || Epochiter:22/22 || loss:0.0007||Batchtime:0.1587||ETA:0:00:07
valid_loss:0.002816356485709548,valid_access:0.861904761904762
Epoch:59/60 || Epochiter:1/22 || loss:0.0013||Batchtime:1.4024||ETA:0:01:01
Epoch:59/60 || Epochiter:2/22 || loss:0.0007||Batchtime:0.6391||ETA:0:00:27
Epoch:59/60 || Epochiter:3/22 || loss:0.0008||Batchtime:0.1415||ETA:0:00:05
Epoch:59/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1457||ETA:0:00:05
Epoch:59/60 || Epochiter:5/22 || loss:0.0008||Batchtime:0.1523||ETA:0:00:06
Epoch:59/60 || Epochiter:6/22 || loss:0.0012||Batchtime:0.4769||ETA:0:00:18
Epoch:59/60 || Epochiter:7/22 || loss:0.0013||Batchtime:0.1577||ETA:0:00:05
Epoch:59/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1309||ETA:0:00:04
Epoch:59/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1621||ETA:0:00:05
Epoch:59/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1454||ETA:0:00:05
Epoch:59/60 || Epochiter:11/22 || loss:0.0012||Batchtime:0.1551||ETA:0:00:05
Epoch:59/60 || Epochiter:12/22 || loss:0.0009||Batchtime:0.1480||ETA:0:00:04
Epoch:59/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1196||ETA:0:00:03
Epoch:59/60 || Epochiter:14/22 || loss:0.0009||Batchtime:0.2489||ETA:0:00:07
Epoch:59/60 || Epochiter:15/22 || loss:0.0009||Batchtime:0.1548||ETA:0:00:04
Epoch:59/60 || Epochiter:16/22 || loss:0.0012||Batchtime:0.1530||ETA:0:00:04
Epoch:59/60 || Epochiter:17/22 || loss:0.0010||Batchtime:0.1523||ETA:0:00:04
Epoch:59/60 || Epochiter:18/22 || loss:0.0011||Batchtime:0.5114||ETA:0:00:13
Epoch:59/60 || Epochiter:19/22 || loss:0.0010||Batchtime:0.1703||ETA:0:00:04
Epoch:59/60 || Epochiter:20/22 || loss:0.0013||Batchtime:0.1470||ETA:0:00:03
Epoch:59/60 || Epochiter:21/22 || loss:0.0009||Batchtime:0.1697||ETA:0:00:04
Epoch:59/60 || Epochiter:22/22 || loss:0.0010||Batchtime:0.1768||ETA:0:00:04
valid_loss:0.002729610074311495,valid_access:0.8666666666666667
save weight success!!
Epoch:60/60 || Epochiter:1/22 || loss:0.0009||Batchtime:1.3898||ETA:0:00:30
Epoch:60/60 || Epochiter:2/22 || loss:0.0010||Batchtime:0.1788||ETA:0:00:03
Epoch:60/60 || Epochiter:3/22 || loss:0.0011||Batchtime:0.7952||ETA:0:00:15
Epoch:60/60 || Epochiter:4/22 || loss:0.0010||Batchtime:0.1392||ETA:0:00:02
Epoch:60/60 || Epochiter:5/22 || loss:0.0009||Batchtime:0.1575||ETA:0:00:02
Epoch:60/60 || Epochiter:6/22 || loss:0.0011||Batchtime:0.1419||ETA:0:00:02
Epoch:60/60 || Epochiter:7/22 || loss:0.0011||Batchtime:0.1360||ETA:0:00:02
Epoch:60/60 || Epochiter:8/22 || loss:0.0010||Batchtime:0.1373||ETA:0:00:02
Epoch:60/60 || Epochiter:9/22 || loss:0.0010||Batchtime:0.1600||ETA:0:00:02
Epoch:60/60 || Epochiter:10/22 || loss:0.0010||Batchtime:0.1416||ETA:0:00:01
Epoch:60/60 || Epochiter:11/22 || loss:0.0008||Batchtime:0.1399||ETA:0:00:01
Epoch:60/60 || Epochiter:12/22 || loss:0.0011||Batchtime:0.1431||ETA:0:00:01
Epoch:60/60 || Epochiter:13/22 || loss:0.0010||Batchtime:0.1128||ETA:0:00:01
Epoch:60/60 || Epochiter:14/22 || loss:0.0010||Batchtime:0.1461||ETA:0:00:01
Epoch:60/60 || Epochiter:15/22 || loss:0.0012||Batchtime:0.8907||ETA:0:00:07
Epoch:60/60 || Epochiter:16/22 || loss:0.0010||Batchtime:0.1629||ETA:0:00:01
Epoch:60/60 || Epochiter:17/22 || loss:0.0012||Batchtime:0.1614||ETA:0:00:00
Epoch:60/60 || Epochiter:18/22 || loss:0.0010||Batchtime:0.1728||ETA:0:00:00
Epoch:60/60 || Epochiter:19/22 || loss:0.0011||Batchtime:0.1601||ETA:0:00:00
Epoch:60/60 || Epochiter:20/22 || loss:0.0010||Batchtime:0.1788||ETA:0:00:00
Epoch:60/60 || Epochiter:21/22 || loss:0.0014||Batchtime:0.1553||ETA:0:00:00
Epoch:60/60 || Epochiter:22/22 || loss:0.0012||Batchtime:0.1720||ETA:0:00:00
'''