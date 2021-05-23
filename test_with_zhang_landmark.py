import os
import sys
import cv2
import numpy as np

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as Transforms
from mbhk_dataloader import mbhk_data
from mixnet import MixNet
import torch.nn as nn
import torch.optim as optim
import time
from collections import OrderedDict
import shutil
from tqdm import tqdm
from PIL import Image
from mixnet import MixNet
# point_nums = 24
# def detect_video(mtcnn_detector, videos_root_path, save_path_root):
#     videos_paths = os.listdir(videos_root_path)
#     if not os.path.exists(save_path_root):
#         os.makedirs(save_path_root)

#     for video_name in videos_paths:
#         cap = cv2.VideoCapture(os.path.join(videos_root_path, video_name))
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#         save_path = os.path.join(save_path_root, video_name)
#         out = cv2.VideoWriter(save_path, fourcc, fps, size)
#         print('save path: =====>', save_path)
#         while True:
#             ret, frame = cap.read()
#             if ret:
#                 bboxs, landmarks, wearmask = mtcnn_detector.detect_face(frame, rgb=True)
#                 # 画人脸框
#                 if bboxs is not None:
#                     for i in range(bboxs.shape[0]):
#                         bbox = np.round(bboxs[i, 0:4]).astype(int)
#                         score = bboxs[i, 4]
#                         cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
#                                       (0, 0, 255), 2)
#                         if wearmask is None:
#                             cv2.putText(frame, '{:.3f}'.format(score), (int(bbox[0]), int(bbox[1])),
#                                         cv2.FONT_HERSHEY_COMPLEX,
#                                         1.3,
#                                         (255, 0, 255), thickness=2)
#                         else:
#                             cv2.putText(frame, '{:.3f}, {:.3f}'.format(score, float(wearmask[i])),
#                                         (int(bbox[0]), int(bbox[1])),
#                                         cv2.FONT_HERSHEY_COMPLEX,
#                                         1.3,
#                                         (255, 0, 255) if float(wearmask[i]) < 0.5 else (0, 255, 0), thickness=2)

#                 if landmarks is not None:
#                     for i in range(landmarks.shape[0]):
#                         landmarks_one = landmarks[i, :]
#                         landmarks_one = landmarks_one.reshape((point_nums, 2))
#                         for j in range(point_nums):
#                             cv2.circle(frame, (int(landmarks_one[j, 0]), int(landmarks_one[j, 1])), 2, (255, 0, 0), -1)
#                 out.write(frame)
#             else:
#                 break

#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()

def main():
    eye_class_dict = {0:"open_eye",1:"close_eye",2:"other"}
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]
    data_trans = Transforms.Compose([
        # Transforms.Resize((24, 48)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    mixnet = MixNet(input_size=(24,48), num_classes=3)
    weight_dict = torch.load("weight/change_mix_data_0202/Mixnet_epoch_59.pth")
    new_state_dict = OrderedDict()
    for k, v in weight_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    mixnet.load_state_dict(new_state_dict)
    # stat(net,(3,48,48))
    mixnet.to('cuda:0')
    mixnet.eval()


    pnet, rnet, onet = create_mtcnn_net(
        p_model_path=r'model_store/final/pnet_epoch_19.pt',
        r_model_path=r'model_store/final/rnet_epoch_7.pt',
        o_model_path=r'model_store/final/onet_epoch_92.pt',
        use_cuda=True)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=threshold)
    videos_root_path = 'test_video/20200522164730261_0.avi'
    save_path_root = 'result_video/20200522164730261_0.avi'

    cap = cv2.VideoCapture(videos_root_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    out = cv2.VideoWriter(save_path_root, fourcc, fps, size)
    while True:
        ret,frame = cap.read()
        
        if ret:
            copy_frame = frame.copy()
            left_right_eye = []
            bboxs, landmarks, wearmask = mtcnn_detector.detect_face(frame, rgb=True)
            
            if landmarks is not None:
                for i in range(landmarks.shape[0]):
                    landmarks_one = landmarks[i, :]
                    landmarks_one = landmarks_one.reshape((point_nums, 2))
                    left_eye = np.array(landmarks_one[[6,8,10,11,14],:])
                    xmin = np.min(left_eye[:,0])
                    ymin = np.min(left_eye[:,1])
                    xmax = np.max(left_eye[:,0])
                    ymax = np.max(left_eye[:,1])
                    left_right_eye.append([xmin,ymin,xmax,ymax])
                    # cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)

                    right_eye = np.array(landmarks_one[[7,9,12,13,15],:])
                    xmin = np.min(right_eye[:,0])
                    ymin = np.min(right_eye[:,1])
                    xmax = np.max(right_eye[:,0])
                    ymax = np.max(right_eye[:,1])
                    left_right_eye.append([xmin,ymin,xmax,ymax])
                    # cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),2)
                    for j in [*left_eye,*right_eye]:
                        cv2.circle(frame, (int(j[0]), int(j[1])), 2, (255, 0, 0), -1)
            
                crop_img = []
                for xmin,ymin,xmax,ymax in left_right_eye:
                    w,h = xmax - xmin, ymax - ymin
                    # 随机扩展大小0.05-0.15
                    k = 0.1
                    ratio = h/w
                    if ratio > 1:
                        ratio = ratio - 1
                        xmin -= (ratio/2*w+k*h)
                        ymin -= (k*h)
                        xmax += (ratio/2*w+k*h)
                        ymax += (k*h)
                        
                    else:
                        ratio = w/h - 1
                        xmin -= (k*w)
                        ymin -= (ratio/2*h+k*w)
                        xmax += (k*w)
                        ymax += (ratio/2*h + k*w)
                    cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,255),2)
                    temp_img = copy_frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                    temp_img = cv2.resize(temp_img,(24,24))
                    crop_img.append(temp_img)
                if len(crop_img) < 2:
                    out.write(frame)
                    continue
                compose_img = np.hstack((crop_img[0],crop_img[1]))
                compose_img = cv2.cvtColor(compose_img,cv2.COLOR_BGR2RGB)

                compose_img = Image.fromarray(compose_img)
                img = data_trans(compose_img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    outputs = mixnet(img.to('cuda:0'))
                    result = torch.max(outputs,1)[1]
                    eye_class = eye_class_dict[result.item()]
                cv2.putText(frame,eye_class,(0,20),cv2.FONT_HERSHEY_COMPLEX,1.3,(255,0,255) \
                    if result.item() == 0 else (255,255,0),thickness=2)
            out.write(frame)
        else:
            print("finish")
            break


    
    


if __name__ == "__main__":
    main()