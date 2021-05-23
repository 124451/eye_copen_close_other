import torch.nn as nn
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
from mbhk_dataloader import mbhk_data
from mixnet import MixNet
import torch.nn as nn
import torch.optim as optim
import time
from collections import OrderedDict
# from torchstat import stat
# img_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/vaild_avg_onely_clos_other.txt'
img_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_valid.txt'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_trans = Transforms.Compose([
        # Transforms.Resize((24, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    test_data = mbhk_data(img_path,
                                       transform=data_trans)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    # 定义模型 输入图像24*48
    net = MixNet(input_size=(24,48), num_classes=3)
    weight_dict = torch.load("weight/change_mix_data_0202/Mixnet_epoch_79.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    net.to(device)
    net.eval()
    acc = 0.0
    val_num = len(test_data)
    with torch.no_grad():
        for i, data in enumerate(test_data_loader):
            img,label,_ = data
            outputs = net(img.to(device))
            result = torch.max(outputs,1)[1]
            acc += (result == label.to(device)).sum().item()
        print("access:%.3f"%(acc/val_num))



if __name__ == "__main__":
    main()