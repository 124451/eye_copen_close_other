import torch.nn as nn
import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
from mbhk_dataloader import mbhk_get_signal_eye
from mixnet import MixNet
import torch.nn as nn
import torch.optim as optim
import time
from collections import OrderedDict
import shutil
from tqdm import tqdm
# from torchstat import stat
# img_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/vaild_avg_onely_clos_other.txt'
img_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_train_and_valid.txt'
label_idx = {0:"open_eye",1:"close_eye",2:"other"}
#预测结果与标签不一致先，保存在对应预测错误的文件夹。
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_trans = Transforms.Compose([
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
    ])
    test_data = mbhk_get_signal_eye(img_path,transform=data_trans)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    # 定义模型 输入图像24*48
    # net = MixNet(input_size=(24,48), num_classes=3)
    # weight_dict = torch.load("weight/change_mix_data_0202/Mixnet_epoch_59.pth")
    net = MixNet(input_size=(24,24), num_classes=3)
    weight_dict = torch.load("weight/mix_mbhk_change_signal_eye_24_24/Mixnet_epoch_59.pth")
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
    #创建文件夹存储错误识别的图片
    class_img = ["close_eye","open_eye","other"]
    error_class_mbhk_img_path = "./error_class/mbhk_img"
    error_class_change_img_path = "./error_class/change_img"
    if not os.path.exists(error_class_mbhk_img_path):
        os.makedirs(error_class_mbhk_img_path)
    if not os.path.exists(error_class_change_img_path):
        os.makedirs(error_class_change_img_path)
    for tp in class_img:
        if not os.path.exists(os.path.join(error_class_mbhk_img_path,tp)):
            os.mkdir(os.path.join(error_class_mbhk_img_path,tp))
        if not os.path.exists(os.path.join(error_class_change_img_path,tp)):
            os.mkdir(os.path.join(error_class_change_img_path,tp))
    #创建错误日志保存到txt文件
    error_lod_mbhk = open(os.path.join(error_class_mbhk_img_path,"error_mbhk_log.txt"),'w')
    error_lod_change = open(os.path.join(error_class_change_img_path,"error_change_log.txt"),'w')
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            # img,res_label,[img_path,json_path]
            timg,label, tpath= test_data.__getitem__(i)
            # timg = timg.unsqueeze(0)
            #用于计数，防止重复操作
            count = 0
            for img in timg:
                #增加维度
                img = img.unsqueeze(0)
                # label = label.unsqueeze(0)
                outputs = net(img.to(device))
                result = torch.max(outputs,1)[1]
                if result.item() != label and count == 0:
                    count += 1
                    if result == 0:
                        if "/imge" in tpath[0]:
                            #则说明是mbhk数据
                            shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"open_eye"))
                            shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"open_eye"))
                            error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        else:
                            error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                            shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"open_eye"))
                            shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"open_eye"))
                    elif result == 1:
                        if "/imge" in tpath[0]:
                            #则说明是mbhk数据
                            shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"close_eye"))
                            shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"close_eye"))
                            error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        else:
                            error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                            shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"close_eye"))
                            shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"close_eye"))
                    elif result == 2:
                        if "/imge" in tpath[0]:
                            #则说明是mbhk数据
                            shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"other"))
                            shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"other"))
                            error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        else:
                            error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                            shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"other"))
                            shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"other"))
    error_lod_mbhk.close()
    error_lod_change.close()
            # torch.tensor([123]).unsqueeze(0)
            # print("dsw")
        # for i, data in enumerate(test_data_loader):
        #     img,label = data
        #     outputs = net(img.to(device))
        #     result = torch.max(outputs,1)[1]
        #     acc += (result == label.to(device)).sum().item()
        # print("access:%.3f"%(acc/val_num))
def select_error_img():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_trans = Transforms.Compose([
        # Transforms.Resize((24, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    test_data = mbhk_get_signal_eye(img_path,
                                       transform=data_trans)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    # 定义模型 输入图像24*48
    net = MixNet(input_size=(24,48), num_classes=3)
    weight_dict = torch.load("weight/mix_data_0129/Mixnet_epoch_59.pth")
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
    #创建文件夹存储错误识别的图片
    class_img = ["close_eye","open_eye","other"]
    error_class_mbhk_img_path = "./error_class/mbhk_img"
    error_class_change_img_path = "./error_class/change_img"
    if not os.path.exists(error_class_mbhk_img_path):
        os.makedirs(error_class_mbhk_img_path)
    if not os.path.exists(error_class_change_img_path):
        os.makedirs(error_class_change_img_path)
    for tp in class_img:
        if not os.path.exists(os.path.join(error_class_mbhk_img_path,tp)):
            os.mkdir(os.path.join(error_class_mbhk_img_path,tp))
        if not os.path.exists(os.path.join(error_class_change_img_path,tp)):
            os.mkdir(os.path.join(error_class_change_img_path,tp))
    #创建错误日志保存到txt文件
    error_lod_mbhk = open(os.path.join(error_class_mbhk_img_path,"error_mbhk_log.txt"),'w')
    error_lod_change = open(os.path.join(error_class_change_img_path,"error_change_log.txt"),'w')
    with torch.no_grad():
        for i in tqdm(range(len(test_data))):
            # img,res_label,[img_path,json_path]
            img,label, tpath= test_data.__getitem__(i)
            img = img.unsqueeze(0)
            # label = label.unsqueeze(0)
            outputs = net(img.to(device))
            result = torch.max(outputs,1)[1]
            if result.item() != label:
                if result == 0:
                    if "/imge" in tpath[0]:
                        #则说明是mbhk数据
                        shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"open_eye"))
                        shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"open_eye"))
                        error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                    else:
                        error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"open_eye"))
                        shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"open_eye"))
                elif result == 1:
                    if "/imge" in tpath[0]:
                        #则说明是mbhk数据
                        shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"close_eye"))
                        shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"close_eye"))
                        error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                    else:
                        error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"close_eye"))
                        shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"close_eye"))
                elif result == 2:
                    if "/imge" in tpath[0]:
                        #则说明是mbhk数据
                        shutil.copy(tpath[1],os.path.join(error_class_mbhk_img_path,"other"))
                        shutil.copy(tpath[0],os.path.join(error_class_mbhk_img_path,"other"))
                        error_lod_mbhk.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                    else:
                        error_lod_change.write("{} {} {}\n".format(tpath[0],label_idx[result.item()],label_idx[label]))
                        shutil.copy(tpath[0],os.path.join(error_class_change_img_path,"other"))
                        shutil.copy(tpath[1],os.path.join(error_class_change_img_path,"other"))
    error_lod_mbhk.close()
    error_lod_change.close()


if __name__ == "__main__":
    main()
    # select_error_img()