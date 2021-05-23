from numpy.lib.function_base import interp
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
from collections import OrderedDict
from sklearn.metrics import roc_curve,auc,f1_score, precision_recall_curve, average_precision_score,precision_score,recall_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix,roc_auc_score 
from sklearn.utils.multiclass import unique_labels
def MultiClass_ROC(y_true, y_pred, classes, title=None):
    """
    :param y_true: list [0, 1, 2]
    :param y_pred: list [0, 1, 2]
    :param classes: list ["0", "1", "2"]
    :param title: roc
    :return: save roc image in path "metric/"
   
    """
    # y_true=[]
    y_true = label_binarize(y_true, classes=[i for i in range(len(classes))])
    # y_pred = label_binarize(y_pred, classes=[i for i in range(len(classes))])
   

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['cyan', 'blue', 'green','orange','pink','red'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("metric/ROC_{}.png".format(len(classes)))

    plt.show()

def main():
    class_num = 3
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
    # model
    mixnet = MixNet(input_size=(24,24), num_classes=3)

    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()
    
    vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
    valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)
    score_list = [] #存储预测得分
    label_list = [] #存储真实标签 
    with torch.no_grad():
        for imgs,labels,_ in valid_data_loader:
            for timg in imgs:
                test_result = mixnet(timg.cuda())
                # result = torch.max(test_result,1)[1]
                result = torch.nn.functional.softmax(test_result,dim=1)
                
                score_list.extend(result.cpu().numpy())
                label_list.extend(torch.nn.functional.one_hot(labels,num_classes=3).numpy())
    tlabel_list = np.array(label_list).reshape((-1,3))
    tscore_list = np.array(score_list).reshape((-1,3))
    # 调用sklearn,计算每个类别对应的fpr和tpr
    fpr_dict = dict()
    tpr_dict = dict()
    roc_auc_dict = dict()
    for i in range(class_num):
        fpr_dict[i],tpr_dict[i],_ = roc_curve(tlabel_list[:,i],tscore_list[:,i])
        roc_auc_dict[i] = auc(fpr_dict[i],tpr_dict[i])

    # Compute micro-average ROC curve and ROC area
    fpr_dict["micro"],tpr_dict["micro"],_ = roc_curve(tlabel_list.ravel(),tscore_list.ravel())
    roc_auc_dict["micro"] = auc(fpr_dict["micro"],tpr_dict["micro"])
    #绘制所有类别平均的roc曲线
    # macro
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(class_num)]))
    # Then interpolate all ROC curves at this points
    mean_ptr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_ptr += interp(all_fpr,fpr_dict[i],tpr_dict[i])
    # Finally average it and compute AUC
    mean_ptr /= class_num
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_ptr
    roc_auc_dict["macro"] = auc(fpr_dict['macro'],tpr_dict["macro"])

    plt.figure()
    lw = 2
    plt.plot(fpr_dict["micro"], tpr_dict["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
 
    plt.plot(fpr_dict["macro"], tpr_dict["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc_dict["macro"]),
             color='navy', linestyle=':', linewidth=4)
 
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(class_num), colors):
        plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc_dict[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('set113_roc.jpg')
    plt.show()

def multi_class_metric(y_true,y_pred):
    
    from sklearn.metrics import f1_score, accuracy_score

    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1_score = f1_score(y_true, y_pred, average='micro')
    accuracy_score = accuracy_score(y_true, y_pred)
    print("Precision_score:", precision)
    print("Recall_score:", recall)
    print("F1_score:", f1_score)
    print("Accuracy_score:", accuracy_score)
def draw_ROC_Line():
    class_num = 3
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
    # model
    mixnet = MixNet(input_size=(24,24), num_classes=3)

    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()
    
    vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
    valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)
    score_list = [] #存储预测得分
    label_list = [] #存储真实标签 
    with torch.no_grad():
        for imgs,labels,_ in valid_data_loader:
            for timg in imgs:
                test_result = mixnet(timg.cuda())
                # result = torch.max(test_result,1)[1]
                result = torch.nn.functional.softmax(test_result,dim=1)
                
                score_list.extend(result.cpu().numpy())
                label_list.extend(labels.cpu().numpy())
    label_list = np.array(label_list).reshape((-1,1))
    score_list = np.array(score_list).reshape((-1,3))
    MultiClass_ROC(label_list,score_list,['0','1','2'])




    # 标签为独热编码
    # 预测结果为softmax
def get_mutil_macro():
    class_num = 3
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
    # model
    mixnet = MixNet(input_size=(24,24), num_classes=3)

    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()
    
    vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
    valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)
    score_list = [] #存储预测得分
    label_list = [] #存储真实标签 
    with torch.no_grad():
        for imgs,labels,_ in valid_data_loader:
            for timg in imgs:
                test_result = mixnet(timg.cuda())
                result = torch.max(test_result,1)[1]
                # result = torch.nn.functional.softmax(test_result,dim=1)
                
                score_list.extend(result.cpu().numpy())
                label_list.extend(labels.cpu().numpy())
    label_list = np.array(label_list).reshape((-1,1))
    score_list = np.array(score_list).reshape((-1,1))
    multi_class_metric(label_list,score_list)
'''
得到结果
Precision_score: 0.8849206349206349
Recall_score: 0.8849206349206349
F1_score: 0.8849206349206349
Accuracy_score: 0.8849206349206349
'''

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.hot_r):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # y_true = np.array(y_true)[indices.astype(int)]
    # y_pred = np.array(y_pred)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    classes = np.array(classes)[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig("metric/{}.png".format("_".join(title.split(" "))))
    return ax
#得到混淆矩阵
def confuse_matrix():
    class_num = 3
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
    # model
    mixnet = MixNet(input_size=(24,24), num_classes=3)

    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()
    
    vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
    valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)
    score_list = [] #存储预测得分
    label_list = [] #存储真实标签 
    with torch.no_grad():
        for imgs,labels,_ in valid_data_loader:
            for timg in imgs:
                test_result = mixnet(timg.cuda())
                result = torch.max(test_result,1)[1]
                # result = torch.nn.functional.softmax(test_result,dim=1)
                
                score_list.extend(result.cpu().numpy())
                label_list.extend(labels.cpu().numpy())
    label_list = np.array(label_list).reshape((-1,1))
    score_list = np.array(score_list).reshape((-1,1))
    plot_confusion_matrix(label_list,score_list,['0','1','2'],normalize=True)

def plot_RR(y_true, y_pred, classes, title=None):
    """
    :param y_true: list [0, 1, 2]
    :param y_pred: list [0, 1, 2]
    :param classes: list ["0", "1", "2"]
    :param title: roc
    :return: save roc image in path "metric/"
    
    """

    y_true = label_binarize(y_true, classes=[i for i in range(len(classes))])
    # y_pred = np.asarray(y_pred)
  
    # �����꣺�����ʣ�False Positive Rate , FPR��

    # Compute RR curve and RR area for each class
    fpr = dict()
    tpr = dict()
    pr_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
        # temp = np.sort(fpr[i])
        fpr[i][0] = 0.0
        pr_auc[i] = auc(np.sort(fpr[i]), tpr[i])
    #
    # Compute micro-average RR curve and RR area
    fpr["micro"], tpr["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    fpr["micro"][0] = 0
    pr_auc["micro"] = auc(np.sort(fpr["micro"]) , tpr["micro"])

    # Compute macro-average RR curve and RR area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

    # Then interpolate all RR curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    pr_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all RR curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average RR curve (area = {0:0.2f})'
                   ''.format(pr_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average RR curve (area = {0:0.2f})'
                   ''.format(pr_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='RR curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], pr_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig("metric/RR_{}.png".format(len(classes)))

    plt.show()

def draw_RR_curve():
    class_num = 3
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
    # model
    mixnet = MixNet(input_size=(24,24), num_classes=3)

    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()
    
    vaild_data = mbhk_get_signal_eye(vaild_txt,vaild_ttrans)
    valid_data_loader = DataLoader(vaild_data,batch_size=128,shuffle=False,num_workers=12)
    score_list = [] #存储预测得分
    label_list = [] #存储真实标签 
    with torch.no_grad():
        for imgs,labels,_ in valid_data_loader:
            for timg in imgs:
                test_result = mixnet(timg.cuda())
                # result = torch.max(test_result,1)[1]
                result = torch.nn.functional.softmax(test_result,dim=1)
                
                score_list.extend(result.cpu().numpy())
                label_list.extend(labels.cpu().numpy())
    label_list = np.array(label_list).reshape((-1,1))
    score_list = np.array(score_list).reshape((-1,3))
    plot_RR(label_list,score_list,['0','1','2'])
if __name__ == "__main__":
    # main()
    draw_ROC_Line()
    # get_mutil_macro()
    # confuse_matrix()
    # draw_RR_curve()