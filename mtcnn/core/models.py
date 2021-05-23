import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    ''' PNet '''
    def __init__(self, is_train=False, use_cuda=True):
        super(PNet, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # conv1
            nn.BatchNorm2d(10),
            nn.PReLU(),  # PReLU1
            nn.MaxPool2d(kernel_size=2, stride=2),  # pool1
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # conv2
            nn.BatchNorm2d(16),
            nn.PReLU(),  # PReLU2
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # conv3
            nn.BatchNorm2d(32),
            nn.PReLU()  # PReLU3
        )
        # detection
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        # bounding box regresion
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        # landmark localization
        self.conv4_3 = nn.Conv2d(32, 5 * 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.pre_layer(x)
        label = F.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return label, offset


class RNet_5_old(nn.Module):
    ''' RNet '''
    def __init__(self, is_train=False, use_cuda=True):
        super(RNet_5_old, self).__init__()
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU()

        )
        self.conv4 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()
        # detection
        self.conv5_1 = nn.Linear(128, 1)
        # bounding box regression
        self.conv5_2 = nn.Linear(128, 4)
        # lanbmark localization
        self.conv5_3 = nn.Linear(128, 10)

    def forward(self, x):
        # backend
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        # detection
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        return det, box


class ONet(nn.Module):
    def __init__(self, channel=3, pool='None', fcn=False, use_cuda=True):
        super().__init__()
        self.pool = pool
        self.use_cuda = use_cuda
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 16, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 32x32 ==> 16x16
        self.conv2 = self._make_layer(16, 32)

        # 16x16 ==> 8x8
        self.conv3 = self._make_layer(32, 64)

        # 8x8 ==> 6x6
        self.landmarks_1 = nn.Sequential(
            nn.Conv2d(64, 128, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # 8x8 ==> 6x6
        self.box_1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 8x8 ==> 6x6
        self.attribute_1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.landmarks_2 = nn.Linear(128 * 36, 24 * 2)  # landmarks
        self.box_2 = nn.Linear(64 * 36, 4)  # box
        self.attribute_2 = nn.Linear(64 * 36, 1)  # face

    @staticmethod
    def _make_layer(in_channel, out_channel):
        conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        conv_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, (3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        return nn.Sequential(conv_1, conv_2)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y_landmarks = self.landmarks_1(x)
        y_box = self.box_1(x)
        y_attribute = self.attribute_1(x)

        y_landmarks = y_landmarks.view(y_landmarks.size(0), -1)
        y_box = y_box.view(y_box.size(0), -1)
        y_attribute = y_attribute.view(y_box.size(0), -1)

        y_landmarks = self.landmarks_2(y_landmarks)
        y_box = self.box_2(y_box)
        y_attribute_1 = self.attribute_2(y_attribute)
        y_attribute_1 = F.sigmoid(y_attribute_1)
        return y_attribute_1, y_box, y_landmarks
