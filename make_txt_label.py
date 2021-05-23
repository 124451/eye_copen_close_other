import random
import os
import numpy as np
random.seed(10)
# 将数据集分为验证集和测试集
#
img_path = r"E:\Coding\eye_class\Dataset_A_Eye_Images"
#0：闭眼
#1：睁眼
def main():
    test_data_close = [[data,0] for data in os.listdir(os.path.join(img_path,r'test\closedEyesTest')) if data.endswith("jpg")]
    print("测试集闭眼:{}".format(len(test_data_close)))
    test_data_open = [[data,1] for data in os.listdir(os.path.join(img_path,r'test\openEyesTest')) if data.endswith("jpg")]
    print("测试集睁眼:{}".format(len(test_data_open)))
    test_data = [*test_data_close,*test_data_open]
    print("测试总数:{}".format(len(test_data)))
    train_data_close = [[data, 0] for data in os.listdir(os.path.join(img_path, r'train\closedEyesTraining')) if
                       data.endswith("jpg")]
    print("训练集闭眼:{}".format(len(train_data_close)))
    train_data_open = [[data, 1] for data in os.listdir(os.path.join(img_path, r'train\openEyesTraining')) if
                      data.endswith("jpg")]
    print("训练集睁眼:{}".format(len(train_data_open)))
    train_data = [*train_data_close, *train_data_open]
    print("训练总数:{}".format(len(train_data)))
    random.shuffle(test_data)
    random.shuffle(train_data)
    with open(os.path.join(img_path,'test_label.txt'),'w') as txtf:
        for temp_data in test_data:
            txtf.write("{} {}\n".format(temp_data[0],temp_data[1]))
    with open(os.path.join(img_path,'train_label.txt'),'w') as txtf:
        for temp_data in train_data:
            txtf.write("{} {}\n".format(temp_data[0],temp_data[1]))
    print('fddwef')



if __name__ == "__main__":
    main()