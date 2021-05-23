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
import glob
from tqdm import tqdm
def main():
    eye_class_dict = {0:"open_eye",1:"close_eye"}
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]
    data_trans = Transforms.Compose([
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
        # Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    mixnet = MixNet(input_size=(24,24), num_classes=3)
    weight_dict = torch.load("weight/mix_mbhk_change_signal_eye_24_24/Mixnet_epoch_59.pth")
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
    # videos_root_path = 'test_video/20200522164730261_0.avi'
    # save_path_root = 'result_video/signal_eye_20200522164730261_0.avi'
    test_video_path = glob.glob(os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/test_video/data(2)","*.mp4"))
    for tpa in tqdm(test_video_path):
        cap = cv2.VideoCapture(tpa)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # tpa
        fname = os.path.splitext(os.path.split(tpa)[1])[0]
        save_path = os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/data(2)",fname+".avi")
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
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
                        #绘制眼睛点
                        # for j in [*left_eye,*right_eye]:
                        #     cv2.circle(frame, (int(j[0]), int(j[1])), 2, (255, 0, 0), -1)
                
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
                        cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,255),1)
                        temp_img = copy_frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                        # temp_img = cv2.resize(temp_img,(24,24))
                        crop_img.append(temp_img)
                    if len(crop_img) < 2:
                        out.write(frame)
                        continue
                    # compose_img = np.hstack((crop_img[0],crop_img[1]))
                    t_result = []
                    for i in crop_img:
                        i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

                        compose_img = Image.fromarray(i)
                        img = data_trans(compose_img)
                        img = img.unsqueeze(0)
                        with torch.no_grad():
                            outputs = mixnet(img.to('cuda:0'))
                            result = torch.max(outputs,1)[1]
                            t_result.append(result.item())
                    # 睁眼优先级最高。其次闭眼，再其次其他
                    if 0 in t_result:
                        eye_class = "open_eye"
                        cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255) \
                        ,thickness=2)
                    elif 1 in t_result:
                        eye_class = "close_eye"
                        cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0) \
                        ,thickness=2)
                    else:
                        eye_class = "other"
                        cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255) \
                        ,thickness=2)
                out.write(frame)
            else:
                print("finish")
                break
def dete_signal_video():
    
    eye_class_dict = {0:"open_eye",1:"close_eye",2:"other"}
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]
    data_trans = Transforms.Compose([
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
        # Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    mixnet = MixNet(input_size=(24,24), num_classes=3)
    # eye_class_dict = {0:"open_eye",1:"close_eye"}
    # weight_dict = torch.load("weight/signal_eye/Mixnet_epoch_29.pth")
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/mix_mbhk_change_signal_eye_24_24/Mixnet_epoch_59.pth")
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
    videos_root_path = 'test_video/DMS_RAW_Nebula_20201201-143038_518.mp4'
    save_path_root = 'result_video/24_24_DMS_RAW_Nebula_20201201-143038_518.avi'
    
    cap = cv2.VideoCapture(videos_root_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # tpa
    # fname = os.path.splitext(os.path.split(tpa)[1])[0]
    # save_path = os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/data(2)",fname+".avi")
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
                    # temp_img = cv2.resize(temp_img,(24,24))
                    crop_img.append(temp_img)
                if len(crop_img) < 2:
                    out.write(frame)
                    continue
                # compose_img = np.hstack((crop_img[0],crop_img[1]))
                t_result = []
                for i in crop_img:
                    i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

                    compose_img = Image.fromarray(i)
                    img = data_trans(compose_img)
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        outputs = mixnet(img.to('cuda:0'))
                        result = torch.max(outputs,1)[1]
                        t_result.append(result.item())
                if 0 in t_result:
                    eye_class = "open_eye"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                    ,thickness=2)
                elif 1 in t_result:
                    eye_class = "close_eye"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                    ,thickness=2)
                else:
                    eye_class = "other"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255) \
                    ,thickness=2)
                # eye_class = "open_eye" if 0 in t_result else "close_eye"

                # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                #     if 0 in t_result else (255,255,0),thickness=2)
            out.write(frame)
        else:

            print("finish")
            break
def show_with_camera():
    
    eye_class_dict = {0:"open_eye",1:"close_eye",2:"other"}
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]
    data_trans = Transforms.Compose([
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
        # Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    mixnet = MixNet(input_size=(24,24), num_classes=3)
    # eye_class_dict = {0:"open_eye",1:"close_eye"}
    # weight_dict = torch.load("weight/signal_eye/Mixnet_epoch_29.pth")
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/mix_mbhk_change_signal_eye_24_24/Mixnet_epoch_59.pth")
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
    videos_root_path = 'test_video/20200506143954001_0.avi'
    save_path_root = 'result_video/camera_test_20210301.avi'
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # tpa
    # fname = os.path.splitext(os.path.split(tpa)[1])[0]
    # save_path = os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/data(2)",fname+".avi")
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
                    # temp_img = cv2.resize(temp_img,(24,24))
                    crop_img.append(temp_img)
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
                if len(crop_img) < 2:
                    cv2.imshow("test",frame)
                    tget_in = cv2.waitKey(10)
                    # print(ord('q'),tget_in)
                    if tget_in == ord('q'):
                        print("get out")
                        break
                    out.write(frame)
                    continue
                # compose_img = np.hstack((crop_img[0],crop_img[1]))
                t_result = []
                for i in crop_img:
                    i = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
                    i = cv2.cvtColor(i,cv2.COLOR_GRAY2RGB)

                    compose_img = Image.fromarray(i)
                    img = data_trans(compose_img)
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        outputs = mixnet(img.to('cuda:0'))
                        result = torch.max(outputs,1)[1]
                        t_result.append(result.item())
                if 0 in t_result:
                    eye_class = "open_eye"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                    ,thickness=2)
                elif 1 in t_result:
                    eye_class = "close_eye"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                    ,thickness=2)
                else:
                    eye_class = "other"
                    cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255) \
                    ,thickness=2)
                cv2.imshow("test",frame)
                tget_in = cv2.waitKey(10)
                if tget_in == ord('q'):
                    print("get out")
                    break
                # eye_class = "open_eye" if 0 in t_result else "close_eye"

                # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                #     if 0 in t_result else (255,255,0),thickness=2)
            out.write(frame)
        else:
            print("finish")
            break

def dete_picture():
    
    eye_class_dict = {0:"open_eye",1:"close_eye",2:"other"}
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]
    data_trans = Transforms.Compose([
        Transforms.Resize((24, 24)),
        Transforms.ToTensor(),
        Transforms.Normalize((0.45, 0.448, 0.455), (0.082, 0.082, 0.082)),
        # Transforms.Normalize((0.407, 0.405, 0.412), (0.087, 0.087, 0.087)),
    ])
    mixnet = MixNet(input_size=(24,24), num_classes=2)
    # eye_class_dict = {0:"open_eye",1:"close_eye"}
    # weight_dict = torch.load("weight/signal_eye/Mixnet_epoch_29.pth")
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/set_lookdown_as_open_signaleye_24_24_20210301/Mixnet_epoch_59.pth")
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
    img_file = "/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/test_video/caiji_0123"
    img_save = "/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/jia_img_result"
    img_path = [os.path.join(img_file, file_name) for file_name in glob.glob(os.path.join(img_file,"*.jpg"))]

    # videos_root_path = 'test_video/DMS_RAW_Nebula_20201201-143038_518.mp4'
    # save_path_root = 'result_video/24_24_DMS_RAW_Nebula_20201201-143038_518.avi'
    
    # cap = cv2.VideoCapture(videos_root_path)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # tpa
    # fname = os.path.splitext(os.path.split(tpa)[1])[0]
    # save_path = os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/data(2)",fname+".avi")
    # out = cv2.VideoWriter(save_path_root, fourcc, fps, size)
    for img_p in tqdm(img_path):
        frame = cv2.imread(img_p)
        
        
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
                # temp_img = cv2.resize(temp_img,(24,24))
                crop_img.append(temp_img)
            if len(crop_img) < 2:
                img_name = os.path.split(img_p)[-1]
                cv2.imwrite(os.path.join(img_save,img_name),frame)
                # out.write(frame)
                continue
            # compose_img = np.hstack((crop_img[0],crop_img[1]))
            t_result = []
            for i in crop_img:
                i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

                compose_img = Image.fromarray(i)
                img = data_trans(compose_img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    outputs = mixnet(img.to('cuda:0'))
                    result = torch.max(outputs,1)[1]
                    t_result.append(result.item())
            if 0 in t_result:
                eye_class = "open_eye"
                cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                ,thickness=2)
            elif 1 in t_result:
                eye_class = "close_eye"
                cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                ,thickness=2)
            else:
                eye_class = "other"
                cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,0,255) \
                ,thickness=2)
            # eye_class = "open_eye" if 0 in t_result else "close_eye"
        img_name = os.path.split(img_p)[-1]
        cv2.imwrite(os.path.join(img_save,img_name),frame)
            # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
            #     if 0 in t_result else (255,255,0),thickness=2)
        
        
        
    


if __name__ == "__main__":
    # main()
    # 检测单个视频
    # dete_signal_video()
    #在两卡服务器外接摄像头采集数据并检测
    # show_with_camera()
    #检测图片
    dete_picture()