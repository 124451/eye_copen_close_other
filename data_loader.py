from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch import transpose
import torchvision.transforms as Transforms
import torch
import numpy as np
import glob
ttrans = Transforms.Compose([
        Transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(30),
        # transforms.RandomCrop(100),
        # transforms.RandomResizedCrop(112),
        Transforms.ColorJitter(brightness=0.5),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.03), ratio=(0.3, 0.3), value=0, ),
        Transforms.Resize((48, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
    ])

class myself_dataloader(Dataset):
    def __init__(self,txt_label_path, transform=None):
        self.root = txt_label_path
        with open(self.root,'r') as readf:
            temp_data = [lb_data.strip().split(' ') for lb_data in readf.readlines()]
        self.lab_data = temp_data
        
        self.transform = transform
    def __len__(self):
        return len(self.lab_data)

    def __getitem__(self, idx):
        img_path = self.lab_data[idx][0]

        img = Image.open(img_path)
        img = img.convert('RGB')
        label = int(self.lab_data[idx][1])

        if self.transform is not None:
            img = self.transform(img)
        return img, label

def main():
    label_dict = {1:"close_eye",0:"open_eye"}
    txt_path = '/media/omnisky/D4T/JSH/faceFenlei/eye/train.txt'
    my_dataloader = myself_dataloader(txt_path,transform=ttrans)
    train_loader = torch.utils.data.DataLoader(my_dataloader,batch_size=2,shuffle=False,num_workers=0)
    for img1,albel1 in train_loader:
        count = 0
        for img, label in zip(img1,albel1):
            count += 1
            print(count)
        #     get_img = np.transpose(img.numpy(),(1,2,0))
        #     get_img = get_img*np.array([0.082, 0.082, 0.082]).reshape((1,1,3)) + \
        #   np.array([0.45, 0.448, 0.455]).reshape((1,1,3))
        #     get_img *= 255
        #     get_img = get_img.astype("uint8")
        #     get_img = Image.fromarray(get_img)
        #     plt.subplot(1,2,count)
        #     plt.imshow(get_img)
        #     plt.title(label_dict[int(label.numpy())])
        # plt.savefig("view.jpg")
        



    img,label = my_dataloader.__getitem__(1)
    img = np.array(img)*np.array([0.13112923, 0.13112923, 0.13112923]).reshape((1,1,3)) + \
          np.array([0.46848394, 0.46848394, 0.46848394]).reshape((1,1,3))
    img = img*255
    img = np.uint8(img)
    img = Image.fromarray(img)
    # print(img.mode)
    # tpix = img.getpixel((0,0))
    # npimg = np.array(img)
    # rgb_img = img.convert('RGB')
    # nprgbimg = np.array(rgb_img)
    plt.imshow(img)
    plt.title(label_dict[label])
    plt.show()
    print("dsa")

def get_mean_std():
    import cv2
    from tqdm import tqdm
    img_path = "/media/omnisky/D4T/JSH/faceFenlei/eye/data2"
    path_list = [os.path.join(img_path,'train',dir_name) for dir_name in os.listdir(os.path.join(img_path,"train")) \
    if not "other" in dir_name]
    path_list.extend(os.path.join(img_path,'test',dir_name) for dir_name in os.listdir(os.path.join(img_path,"test")) \
    if not "other" in dir_name)
    img_path = []
    for tpath in path_list:
        img_path.extend([os.path.join(tpath,tfname) for tfname in os.listdir(tpath) if tfname.endswith("jpg")])


    means = [0,0,0]
    stdevs = [0,0,0]

    # for curdir,subdir,files in os.walk(r'E:\Coding\eye_class\Dataset_A_Eye_Images'):
    #     for img in (file for file in files if file.endswith("jpg")):
    #         temp_path = os.path.join(curdir,img)
    #         img_path.append(temp_path)
    for tp_img in tqdm(img_path):
        img = np.array(cv2.cvtColor(cv2.imread(tp_img,1),cv2.COLOR_BGR2RGB)).astype(np.float32)/255.0
        for i in range(3):
            means[i] += img[:,:,i].mean()
            stdevs[i] += img[:,:,i].std()

    means = np.asarray(means)/len(img_path)
    stdevs = np.asarray(stdevs)/len(img_path)
    print('means={}'.format(means))
    print('stdevs={}'.format(stdevs))


if __name__ == "__main__":
    # get_mean_std()
    main()