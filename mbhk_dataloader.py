import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torch import transpose
import matplotlib.pyplot as plt
import torch
import json
import numpy as np
import random
random.seed(10)

class mbhk_data(Dataset):
    def __init__(self,txt_file,transform=None) -> None:
        self.root = txt_file
        with open(self.root,'r') as ftxt:
            temp_data = [lb_data.strip() for lb_data in ftxt.readlines()]
        self.lab_data = temp_data
        self.transform = transform
    def __len__(self) -> int:
        return len(self.lab_data)

    def __getitem__(self, index: int):
        img_path = self.lab_data[index]
        path,img_name = os.path.split(img_path)
        img_name = os.path.splitext(img_name)[0]
        if "/imge" in path:
            path = path.replace("/imge","/ann")
        else:
            path = path.replace("/change_img",'/change_json')
        json_path = os.path.join(path,img_name+'.json')
        img = Image.open(img_path)
        img = img.convert("RGB")
        # get label
        label = self.get_label(json_path)
        img = self.get_img(label,img)
        if self.transform is not None:
            img = self.transform(img)
        res_label = label[-1]
        return img,res_label,[img_path,json_path]

    def get_label(self,tpath):
        # print("path:{}".format(tpath))
        with open(tpath,'r') as jt:
            jsdict = json.load(jt)
        eye_point = {'left_eye':[],'right_eye':[]}

        for label_key in jsdict:
            #get 
            # if 'name' not in label_key:
            #     return 0
            if label_key['name'] == "dangerous_driving":
                
                attribute = label_key["attributes"]

                close_eyes = attribute["close_eyes"]
                unclear_eye = attribute["unclear_eyes"]
                sunglass_block = attribute["sunglasses_block"]
                lookdown = attribute["look_down"]
                if close_eyes or lookdown:
                    label = 1
                    # if lookdown:
                    #     pass
                elif unclear_eye or sunglass_block:
                    label = 2
                else:
                    label = 0
               
            elif 'left_eye_' in label_key['name']:
                eye_point['left_eye'].append(np.array(label_key['points'][0]))
            elif 'right_eye_' in label_key['name']:
                eye_point['right_eye'].append(np.array(label_key['points'][0]))

        left_eye_p = np.array(eye_point['left_eye'])
        right_eye_p = np.array(eye_point['right_eye'])

        l_xmin = max(min(left_eye_p[:,0]),0)
        l_ymin = max(min(left_eye_p[:,1]),0)
        l_xmax = max(left_eye_p[:,0])
        l_ymax = max(left_eye_p[:,1])

        r_xmin = max(min(right_eye_p[:,0]),0)
        r_ymin = max(min(right_eye_p[:,1]),0)
        r_xmax = max(right_eye_p[:,0])
        r_ymax = max(right_eye_p[:,1])
        return [l_xmin,l_ymin,l_xmax,l_ymax,r_xmin,r_ymin,r_xmax,r_ymax,label]
    
    def get_img(self,label,image):

    
        img = image
        # 扩大眼睛框
        eye_point = [label[0:4],label[4:8]]
        crop_img = []
        for tdata in eye_point:
            xmin,ymin,xmax,ymax = tdata
            # cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,50),2)
            w,h = xmax - xmin, ymax - ymin
            # 随机扩展大小0.05-0.15
            k = np.random.random_sample()*0.1+0.05
            ratio = h/w
            if ratio > 1:
                ratio = ratio - 1
                xmin -= (ratio/2*w+k*h)
                ymin -= (k*h)
                xmax += (ratio/2*w+k*h)
                ymax += (k*h)
                crop_img.append(img.crop((int(xmin),int(ymin),int(xmax),int(ymax))))
            else:
                ratio = w/h - 1
                xmin -= (k*w)
                ymin -= (ratio/2*h+k*w)
                xmax += (k*w)
                ymax += (ratio/2*h + k*w)
                crop_img.append(img.crop((int(xmin),int(ymin),int(xmax),int(ymax))))
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((int(xmin),int(ymin),int(xmax),int(ymax)))
        # crop_img[0] = crop_img[0].resize((48,48))
        # crop_img[1] = crop_img[1].resize((48,48))
        # target = Image.new("RGB",(96,48))
        # target.paste(crop_img[0],(0,0,48,48))
        # target.paste(crop_img[1],(48,0,96,48))
        crop_img[0] = crop_img[0].resize((24,24))
        crop_img[1] = crop_img[1].resize((24,24))
        target = Image.new("RGB",(48,24))
        target.paste(crop_img[0],(0,0,24,24))
        target.paste(crop_img[1],(24,0,48,24))
        return target
#返回左右两个眼睛
class mbhk_get_signal_eye(Dataset):
    def __init__(self,txt_file,transform=None) -> None:
        self.root = txt_file
        with open(self.root,'r') as ftxt:
            temp_data = [lb_data.strip() for lb_data in ftxt.readlines()]
        self.lab_data = temp_data
        self.transform = transform
    def __len__(self) -> int:
        return len(self.lab_data)

    def __getitem__(self, index: int):
        img_path = self.lab_data[index]
        path,img_name = os.path.split(img_path)
        img_name = os.path.splitext(img_name)[0]
        if "/imge" in path:
            path = path.replace("/imge","/ann")
        else:
            path = path.replace("/change_img",'/change_json')
        json_path = os.path.join(path,img_name+'.json')
        img = Image.open(img_path)
        img = img.convert("RGB")
        # get label
        label = self.get_label(json_path)
        img = self.get_img(label,img)
        if self.transform is not None:
            img[0] = self.transform(img[0])
            img[1] = self.transform(img[1])
        res_label = torch.tensor(label[-1]) 
        return img,res_label,[img_path,json_path]

    def get_label(self,tpath):
        # print("path:{}".format(tpath))
        with open(tpath,'r') as jt:
            jsdict = json.load(jt)
        eye_point = {'left_eye':[],'right_eye':[]}

        for label_key in jsdict:
            #get 
            # if 'name' not in label_key:
            #     return 0
            if label_key['name'] == "dangerous_driving":
                
                attribute = label_key["attributes"]

                close_eyes = attribute["close_eyes"]
                unclear_eye = attribute["unclear_eyes"]
                sunglass_block = attribute["sunglasses_block"]
                lookdown = attribute["look_down"]
                # if close_eyes or lookdown: 闭眼和向下看为闭眼，2021年3月1日12:07:41
                # 向下看和睁眼为睁眼
                if close_eyes or lookdown:
                    label = 1
                    # if lookdown:
                    #     pass
                elif unclear_eye or sunglass_block:
                    label = 2
                else:
                    label = 0
               
            elif 'left_eye_' in label_key['name']:
                eye_point['left_eye'].append(np.array(label_key['points'][0]))
            elif 'right_eye_' in label_key['name']:
                eye_point['right_eye'].append(np.array(label_key['points'][0]))

        left_eye_p = np.array(eye_point['left_eye'])
        right_eye_p = np.array(eye_point['right_eye'])

        l_xmin = max(min(left_eye_p[:,0]),0)
        l_ymin = max(min(left_eye_p[:,1]),0)
        l_xmax = max(left_eye_p[:,0])
        l_ymax = max(left_eye_p[:,1])

        r_xmin = max(min(right_eye_p[:,0]),0)
        r_ymin = max(min(right_eye_p[:,1]),0)
        r_xmax = max(right_eye_p[:,0])
        r_ymax = max(right_eye_p[:,1])
        return [l_xmin,l_ymin,l_xmax,l_ymax,r_xmin,r_ymin,r_xmax,r_ymax,label]
    
    def get_img(self,label,image):

    
        img = image
        # 扩大眼睛框
        eye_point = [label[0:4],label[4:8]]
        crop_img = []
        for tdata in eye_point:
            xmin,ymin,xmax,ymax = tdata
            # cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,50),2)
            w,h = xmax - xmin, ymax - ymin
            # 随机扩展大小0.05-0.15
            k = np.random.random_sample()*0.1+0.05
            ratio = h/w
            if ratio > 1:
                ratio = ratio - 1
                xmin -= (ratio/2*w+k*h)
                ymin -= (k*h)
                xmax += (ratio/2*w+k*h)
                ymax += (k*h)
                crop_img.append(img.crop((int(xmin),int(ymin),int(xmax),int(ymax))))
            else:
                ratio = w/h - 1
                xmin -= (k*w)
                ymin -= (ratio/2*h+k*w)
                xmax += (k*w)
                ymax += (ratio/2*h + k*w)
                crop_img.append(img.crop((int(xmin),int(ymin),int(xmax),int(ymax))))
            # draw = ImageDraw.Draw(img)
            # draw.rectangle((int(xmin),int(ymin),int(xmax),int(ymax)))
        # crop_img[0] = crop_img[0].resize((48,48))
        # crop_img[1] = crop_img[1].resize((48,48))
        # target = Image.new("RGB",(96,48))
        # target.paste(crop_img[0],(0,0,48,48))
        # target.paste(crop_img[1],(48,0,96,48))
        # crop_img[0] = crop_img[0].resize((24,24))
        # crop_img[1] = crop_img[1].resize((24,24))
        # target = Image.new("RGB",(48,24))
        # target.paste(crop_img[0],(0,0,24,24))
        # target.paste(crop_img[1],(24,0,48,24))
        return crop_img

# jia signal eye
# /media/omnisky/D4T/JSH/faceFenlei/eye/
class jia_signal_eye(Dataset):
    def __init__(self,tpath,transform = None):
        with open(tpath,'r') as ftxt:
            t_read_line = ftxt.readlines()
        self.img_label = [label.strip() for label in t_read_line]
        self.transform = transform
    # close:1 open:0
    def __getitem__(self, index: int):
        img_path, label = self.img_label[index].split(' ')
        img = Image.open(img_path)
        img = img.convert("RGB")
        label = int(label)
        if self.transform:
            img = self.transform(img)
        return img, label
    def __len__(self) -> int:
        return len(self.img_label)
        




def main():
    label_idx = {0:"open_eye",1:"close_eye",2:"other"}
    my_data = mbhk_data("/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/train_path_avg.txt")
    dataloader = DataLoader(my_data,batch_size=2,shuffle=False,num_workers=2)
    for i in range(len(my_data)):
        img,label = my_data.__getitem__(i)
        plt.title(label_idx[label])
        plt.imshow(img)
        plt.savefig("view.jpg")

    # for img,data in dataloader:
    #     count = 0
    #     for image,tdata in zip(img,data):
    #         ax1 = plt.subplot(1,2,count)
    #         plt.title(label_idx[tdata])
    #         plt.imshow(image)
    #         count += 1
    #     plt.savefig("view.jpg")

def setect_eye_img():
    import cv2
    txt_save_img_path = "/media/omnisky/D4T/JSH/faceFenlei/eye/mbhlk_hl_0128/mix_train_and_valid.txt"
    with open(txt_save_img_path,'r') as tft:
        txt_data = tft.readlines()
    for img_path in txt_data:
        img = cv2.imread(img_path.strip())
        cv2.imshow("img",img)
        cv2.waitKey(0)



if __name__ == "__main__":
    # main()
    setect_eye_img()