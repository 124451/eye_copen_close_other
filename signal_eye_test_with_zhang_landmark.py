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
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
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
    test_video_path = glob.glob(os.path.join("test_video/temp_video","*.avi"))
    for tpa in tqdm(test_video_path):
        img_count = 0
        cap = cv2.VideoCapture(tpa)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # tpa
        fname = os.path.splitext(os.path.split(tpa)[1])[0]
        save_path = os.path.join("result_video/result",fname+".avi")
        # out = cv2.VideoWriter(save_path, fourcc, 2, size)
        while True:
            ret,frame = cap.read()
            
            if ret:
                copy_frame = frame.copy()
                left_right_eye = []
                bboxs, landmarks, wearmask = mtcnn_detector.detect_face(frame, rgb=True)
                get_ids = 0
                twild = 0
                for i in range(bboxs.shape[0]):
                    temp = bboxs[i][2] - bboxs[i][0]
                    if temp>twild:
                        get_ids = i
                        twild = temp



                temp_path,trmp_name = os.path.split(save_path)
                trmp_name = os.path.splitext(trmp_name)[0] + "{:04d}.jpg".format(img_count)
                tsave_path = os.path.join(temp_path, trmp_name)
                if landmarks.size != 0:
                    eye_wild_buf = []
                    # for i in range(landmarks.shape[0]):
                    landmarks_one = landmarks[get_ids, :]
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
                        eye_wild_buf.append(w)
                        cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,255),1)
                        # 输出眼睛像素的长宽

                        temp_img = copy_frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                        # temp_img = cv2.resize(temp_img,(24,24))
                        crop_img.append(temp_img)
                    if len(crop_img) < 2:
                        
                        cv2.imwrite(tsave_path,frame)
                        # out.write(frame)
                        continue
                else:
                    cv2.imwrite(tsave_path,frame)
                        # out.write(frame)
                    continue
                    # compose_img = np.hstack((crop_img[0],crop_img[1]))
                result_buff = []
                score_buff = []
                for i in crop_img:
                    i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
                    t1 = time.time()
                    compose_img = Image.fromarray(i)
                    img = data_trans(compose_img)
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        outputs = mixnet(img.to('cuda:0'))
                        spft_max = torch.nn.functional.softmax(outputs,dim=1)
                        # 左眼右眼，分别三个类别的分数
                        score_buff.append(spft_max.cpu().numpy())
                        # 0,1->data,id
                        score,result = torch.max(spft_max,1)
                        # result:最大值的id score:最大值的分数
                        result_buff.append([result.item(),score])
                    run_time = time.time() - t1
                    #0.005819
                bias = 30
                eye_bias = 100
                for i in range(2):
                    t_result = result_buff[i][0]
                    #眼睛抠图的宽度
                    eye_w = eye_wild_buf[i]
                    cv2.putText(frame,"w:{}".format(int(eye_w)),(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-50),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255) \
                        ,thickness=2)
                    if 0 == t_result:
                        # eye_class = "close_eye"
                        # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                        # ,thickness=2)
                        eye_class = "open_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                        cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255) \
                        ,thickness=2)
                    elif 1 == t_result:
                        # eye_class = "open_eye"
                        # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                        # ,thickness=2)

                        eye_class = "close_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                        cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0) \
                        ,thickness=2)
                    else:
                        eye_class = "other:{:.2f}".format(result_buff[i][1].cpu().item())
                        cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255) \
                        ,thickness=2)
                    # bias += 30
                    eye_bias = 0
                    # left_eye
                    left_eye_open,left_eye_close,left_eye_other = score_buff[0][0]
                    cv2.putText(frame,"left_open:{:.2f}".format(left_eye_open) ,(10, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)
                    cv2.putText(frame,"left_close:{:.2f}".format(left_eye_close) ,(10, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)
                    cv2.putText(frame,"left_other:{:.2f}".format(left_eye_other) ,(10, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)

                    #right_eye
                    right_eye_open,right_eye_close,right_eye_other = score_buff[1][0]
                    cv2.putText(frame,"left_open:{:.2f}".format(right_eye_open) ,(200, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)
                    cv2.putText(frame,"left_close:{:.2f}".format(right_eye_close) ,(200, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)
                    cv2.putText(frame,"left_other:{:.2f}".format(right_eye_other) ,(200, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                        ,thickness=2)
                # 计算最大概率的标号
                max_id,max_score = (result_buff[0][0],result_buff[0][1].cpu().item()) if \
                    result_buff[0][1].cpu().item()>result_buff[1][1].cpu().item() else (result_buff[1][0],result_buff[1][1].cpu().item())
                # 测试信息
                eye_wild_buf_info = "w:[{:.2f},{:.2f}]".format(eye_wild_buf[0],eye_wild_buf[1])
                # 测试时那个眼镜框最大
                max_wilde_left_right = 0 if eye_wild_buf[0]>eye_wild_buf[1] else 1
                # 获得最大宽度框的id和分数
                # 宽度最大的 id 和分数 宽度第二大的 id和分数
                max_wilde_id,max_wilde_score,max_wiled_second_id,max_wilde_second_score = (result_buff[0][0],result_buff[0][1].cpu().item(),result_buff[1][0],result_buff[1][1].cpu().item()) if \
                    max_wilde_left_right==0 else (result_buff[1][0],result_buff[1][1].cpu().item(),result_buff[0][0],result_buff[0][1].cpu().item())

                score_buff_info = "score:[left: {:.2f}] [right: {:.2f}]".format(score_buff[0][0][2],score_buff[1][0][2])
                cv2.putText(frame,eye_wild_buf_info,(400,80),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
                    ,thickness=2)
                cv2.putText(frame,score_buff_info,(400,100),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
                ,thickness=2)
                
                # 如果

                # if np.any(np.array(eye_wild_buf[:2])<19.0 )and max_score < 0.9 or np.any(np.array(eye_wild_buf[:2])<17.0 ) or np.any(np.array([score_buff[0][0][2],score_buff[1][0][2]])>= 0.5) and \
                #     max_score<0.9 or max_id==2:
                # 添加最大框                                                                                                            概率最大id=2 宽度最大的id=2
                # if (eye_wild_buf[max_wilde_left_right]<17.0 ) or ((max_wilde_score>= 0.5) and \
                #     max_wilde_id==2 and max_wilde_second_score<0.85)  or max_id==2 and (max_wilde_score < 0.8 and max_wilde_id != 2) or (max_id==2 and max_wilde_id == 2 and(max_wilde_second_score<0.8) ) or \
                #         (max_wilde_id == 2 and max_wiled_second_id==2 and (max_wilde_second_score>0.5 or max_wilde_score>0.5)) or ( eye_wild_buf[ 0 if max_wilde_left_right else 1]<17.0 ) or \
                #             ((eye_wild_buf[ 0 if max_wilde_left_right else 1]>23 and max_wilde_second_score>0.8 and max_wilde_id==2) or \
                #                 (eye_wild_buf[max_wilde_left_right]>23 and max_wilde_score >0.8 and max_wiled_second_id==2)):
                # 左眼右眼宽度大于23 且概率大于0.8 且id=2
                # 存在小于17像素的框且最大宽度的分数小于0.8
                # 存在other概率大于0.5
                # 存在小于10像素直接判断为other
                # 当最小眼睛的像素小于23 且 为other的概率大于0.5
                # 
                # if (eye_wild_buf[ 0 if max_wilde_left_right else 1]<23) and ( (max_wilde_score<0.85 and max_wilde_second_score<0.85)):
                #     print("dwqe")

                
                
                if ((eye_wild_buf[ 0 if max_wilde_left_right else 1]>23 and max_wilde_second_score>0.8 and max_wiled_second_id==2) or \
                    (eye_wild_buf[ max_wilde_left_right]>23 and max_wilde_score >0.8 and max_wilde_id==2) or \
                    (np.any(np.array(eye_wild_buf[:2])<17.0) and (max_wilde_score<0.8)) or
                    ((max_wilde_id==2 and max_wilde_score>0.5 and max_wilde_second_score<0.9) or (max_wiled_second_id==2 and max_wilde_second_score>0.5 and max_wilde_score<0.9)) or\
                    (np.any(np.array(eye_wild_buf[:2])<10.0)) or \
                    ((eye_wild_buf[ 0 if max_wilde_left_right else 1]<23) and ((max_wiled_second_id==2 and max_wilde_second_score>0.5) or (max_wilde_score<0.85 and max_wilde_second_score<0.85)))
                        ):
                    # 如果像素小于19且最大概率的眼睛小于0.9 或 任何一个像素小于12 且 max分数小于0.9 或 other
                    # 2.任意一个other>=50
                    cv2.putText(frame,"other",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255) \
                    ,thickness=2)
                # elif np.any(np.array([score_buff[0][0][1],score_buff[1][0][1]])>= 0.85)  \
                #      or (max_id==1 and max_score>0.750):
                elif (max_wilde_id==1 and max_wilde_score>=0.85)  \
                     or (max_id==1 and max_score>0.750):
                # elif (max_wilde_score >= 0.85) and max_wilde_id==1  \
                #      or (max_wilde_id==1 and max_wilde_score>0.750):
                    # 任意一个闭眼概率大于0.9
                    # 最大值是闭眼且概率大于0.75
                    cv2.putText(frame,"close",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0) \
                    ,thickness=2)
                else:
                    cv2.putText(frame,"open",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
                    ,thickness=2)
                
                cv2.imwrite(tsave_path,frame)
                img_count += 1
                # out.write(frame)
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
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_04_mix_SGD_mutillabel_24_24_20210302/Mixnet_epoch_49.pth")
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
    videos_root_path = 'test_video/hhh/02_65_6504_0_be4ba2aeac264ed992aae74c15b91b18.mp4'
    save_path_root = 'result_video/debug_test.avi'
    
    cap = cv2.VideoCapture(videos_root_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # tpa
    fname = os.path.splitext(os.path.split(videos_root_path)[1])[0]
    save_path = os.path.join("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/data(2)",fname+".avi")
    out = cv2.VideoWriter(save_path_root, fourcc, fps, size)
    while True:
        ret,frame = cap.read()
        
        if ret:
            copy_frame = frame.copy()
            left_right_eye = []
            bboxs, landmarks, wearmask = mtcnn_detector.detect_face(frame, rgb=True)
            temp_path,trmp_name = os.path.split(save_path)
            # trmp_name = os.path.splitext(trmp_name)[0] + "{:04d}.jpg".format(img_count)
            # tsave_path = os.path.join(temp_path, trmp_name)
            if landmarks is not None:
                eye_wild_buf = []
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
                    eye_wild_buf.append(w)
                    cv2.rectangle(frame,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,255),1)
                    # 输出眼睛像素的长宽

                    temp_img = copy_frame[int(ymin):int(ymax),int(xmin):int(xmax)]
                    # temp_img = cv2.resize(temp_img,(24,24))
                    crop_img.append(temp_img)
                if len(crop_img) < 2:
                    
                    cv2.imwrite(tsave_path,frame)
                    # out.write(frame)
                    continue
                # compose_img = np.hstack((crop_img[0],crop_img[1]))
            result_buff = []
            score_buff = []
            for i in crop_img:
                i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
                t1 = time.time()
                compose_img = Image.fromarray(i)
                img = data_trans(compose_img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    outputs = mixnet(img.to('cuda:0'))
                    spft_max = torch.nn.functional.softmax(outputs,dim=1)
                    # 左眼右眼，分别三个类别的分数
                    score_buff.append(spft_max.cpu().numpy())
                    # 0,1->data,id
                    score,result = torch.max(spft_max,1)
                    # result:最大值的id score:最大值的分数
                    result_buff.append([result.item(),score])
                run_time = time.time() - t1
                #0.005819
            bias = 30
            eye_bias = 100
            for i in range(2):
                t_result = result_buff[i][0]
                #眼睛抠图的宽度
                eye_w = eye_wild_buf[i]
                cv2.putText(frame,"w:{}".format(int(eye_w)),(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-50),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255) \
                    ,thickness=2)
                if 0 == t_result:
                    # eye_class = "close_eye"
                    # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                    # ,thickness=2)
                    eye_class = "open_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255) \
                    ,thickness=2)
                elif 1 == t_result:
                    # eye_class = "open_eye"
                    # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                    # ,thickness=2)

                    eye_class = "close_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0) \
                    ,thickness=2)
                else:
                    eye_class = "other:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255) \
                    ,thickness=2)
                # bias += 30
                eye_bias = 0
                # left_eye
                left_eye_open,left_eye_close,left_eye_other = score_buff[0][0]
                cv2.putText(frame,"left_open:{:.2f}".format(left_eye_open) ,(10, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_close:{:.2f}".format(left_eye_close) ,(10, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_other:{:.2f}".format(left_eye_other) ,(10, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)

                #right_eye
                right_eye_open,right_eye_close,right_eye_other = score_buff[1][0]
                cv2.putText(frame,"left_open:{:.2f}".format(right_eye_open) ,(200, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_close:{:.2f}".format(right_eye_close) ,(200, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_other:{:.2f}".format(right_eye_other) ,(200, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
            # 计算最大概率的标号
            max_id,max_score = (result_buff[0][0],result_buff[0][1].cpu().item()) if \
                result_buff[0][1].cpu().item()>result_buff[1][1].cpu().item() else (result_buff[1][0],result_buff[1][1].cpu().item())
            # 测试信息
            eye_wild_buf_info = "w:[{:.2f},{:.2f}]".format(eye_wild_buf[0],eye_wild_buf[1])
            # 测试时那个眼镜框最大
            max_wilde_left_right = 0 if eye_wild_buf[0]>eye_wild_buf[1] else 1
            # 获得最大宽度框的id和分数
            # 宽度最大的 id 和分数 宽度第二大的 id和分数
            max_wilde_id,max_wilde_score,max_wiled_second_id,max_wilde_second_score = (result_buff[0][0],result_buff[0][1].cpu().item(),result_buff[1][0],result_buff[1][1].cpu().item()) if \
                max_wilde_left_right==0 else (result_buff[1][0],result_buff[1][1].cpu().item(),result_buff[0][0],result_buff[0][1].cpu().item())

            score_buff_info = "score:[left: {:.2f}] [right: {:.2f}]".format(score_buff[0][0][2],score_buff[1][0][2])
            cv2.putText(frame,eye_wild_buf_info,(400,80),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
                ,thickness=2)
            cv2.putText(frame,score_buff_info,(400,100),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
            ,thickness=2)
            
            # 如果

            # if np.any(np.array(eye_wild_buf[:2])<19.0 )and max_score < 0.9 or np.any(np.array(eye_wild_buf[:2])<17.0 ) or np.any(np.array([score_buff[0][0][2],score_buff[1][0][2]])>= 0.5) and \
            #     max_score<0.9 or max_id==2:
            # 添加最大框                                                                                                            概率最大id=2 宽度最大的id=2
            # if (eye_wild_buf[max_wilde_left_right]<17.0 ) or ((max_wilde_score>= 0.5) and \
            #     max_wilde_id==2 and max_wilde_second_score<0.85)  or max_id==2 and (max_wilde_score < 0.8 and max_wilde_id != 2) or (max_id==2 and max_wilde_id == 2 and(max_wilde_second_score<0.8) ) or \
            #         (max_wilde_id == 2 and max_wiled_second_id==2 and (max_wilde_second_score>0.5 or max_wilde_score>0.5)) or ( eye_wild_buf[ 0 if max_wilde_left_right else 1]<17.0 ) or \
            #             ((eye_wild_buf[ 0 if max_wilde_left_right else 1]>23 and max_wilde_second_score>0.8 and max_wilde_id==2) or \
            #                 (eye_wild_buf[max_wilde_left_right]>23 and max_wilde_score >0.8 and max_wiled_second_id==2)):
            # 左眼右眼宽度大于23 且概率大于0.8 且id=2
            # 存在小于17像素的框且最大宽度的分数小于0.8
            # 存在other概率大于0.5
            # 存在小于10像素直接判断为other
            
            
            if ((eye_wild_buf[ 0 if max_wilde_left_right else 1]>23 and max_wilde_second_score>0.8 and max_wiled_second_id==2) or \
                (eye_wild_buf[ max_wilde_left_right]>23 and max_wilde_score >0.8 and max_wilde_score==2) or \
                (np.any(np.array(eye_wild_buf[:2])<17.0) and (max_wilde_score<0.8)) or
                ((max_wilde_id==2 and max_wilde_score>0.5 and max_wilde_second_score<0.9) or (max_wiled_second_id==2 and max_wilde_second_score>0.5 and max_wilde_score<0.9)) or\
                (np.any(np.array(eye_wild_buf[:2])<10.0))
                    ):
                # 如果像素小于19且最大概率的眼睛小于0.9 或 任何一个像素小于12 且 max分数小于0.9 或 other
                # 2.任意一个other>=50
                cv2.putText(frame,"other",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255) \
                ,thickness=2)
            # elif np.any(np.array([score_buff[0][0][1],score_buff[1][0][1]])>= 0.85)  \
            #      or (max_id==1 and max_score>0.750):
            elif (max_wilde_id==1 and max_wilde_score>=0.80)  \
                    or (max_id==1 and max_score>0.750):
            # elif (max_wilde_score >= 0.85) and max_wilde_id==1  \
            #      or (max_wilde_id==1 and max_wilde_score>0.750):
                # 任意一个闭眼概率大于0.9
                # 最大值是闭眼且概率大于0.75
                cv2.putText(frame,"close",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0) \
                ,thickness=2)
            else:
                cv2.putText(frame,"open",(400,60),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,0) \
                ,thickness=2)
                
                # cv2.imshow("frame",frame)
                
                
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
    mixnet = MixNet(input_size=(24,24), num_classes=3)
    # eye_class_dict = {0:"open_eye",1:"close_eye"}
    # weight_dict = torch.load("weight/signal_eye/Mixnet_epoch_29.pth")
    weight_dict = torch.load("/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/weight/relabel_mix_24_24_20210302/Mixnet_epoch_59.pth")
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
    img_save = "/media/omnisky/D4T/JSH/faceFenlei/Projects/hul_eye_class/result_video/relabel_img_result_adma_01"
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
            result_buff = []
            score_buff = []
            for i in crop_img:
                i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)

                compose_img = Image.fromarray(i)
                img = data_trans(compose_img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    outputs = mixnet(img.to('cuda:0'))
                    spft_max = torch.nn.functional.softmax(outputs,dim=1)
                    score_buff.append(spft_max.cpu().numpy())
                    # 0,1->data,id
                    score,result = torch.max(spft_max,1)
                    result_buff.append([result.item(),score])
            bias = 30
            eye_bias = 100
            for i in range(2):
                t_result = result_buff[i][0]
                if 0 == t_result:
                    # eye_class = "close_eye"
                    # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(0,255,0) \
                    # ,thickness=2)
                    eye_class = "open_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255) \
                    ,thickness=2)
                elif 1 == t_result:
                    # eye_class = "open_eye"
                    # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
                    # ,thickness=2)

                    eye_class = "close_eye:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,255,0) \
                    ,thickness=2)
                else:
                    eye_class = "other:{:.2f}".format(result_buff[i][1].cpu().item())
                    cv2.putText(frame,eye_class,(int(left_right_eye[i][0])-eye_bias, int(left_right_eye[i][1])-bias),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255) \
                    ,thickness=2)
                # bias += 30
                eye_bias = 0
                # left_eye
                left_eye_open,left_eye_close,left_eye_other = score_buff[0][0]
                cv2.putText(frame,"left_open:{:.2f}".format(left_eye_open) ,(10, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_close:{:.2f}".format(left_eye_close) ,(10, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_other:{:.2f}".format(left_eye_other) ,(10, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)

                #right_eye
                right_eye_open,right_eye_close,right_eye_other = score_buff[1][0]
                cv2.putText(frame,"left_open:{:.2f}".format(right_eye_open) ,(200, 20),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_close:{:.2f}".format(right_eye_close) ,(200, 40),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
                cv2.putText(frame,"left_other:{:.2f}".format(right_eye_other) ,(200, 60),cv2.FONT_HERSHEY_COMPLEX,0.6,(20,150,0) \
                    ,thickness=2)
            # eye_class = "open_eye" if 0 in t_result else "close_eye"
        img_name = os.path.split(img_p)[-1]
        cv2.imwrite(os.path.join(img_save,img_name),frame)
            # cv2.putText(frame,eye_class,(int(xmax), int(ymax)-20),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,0,255) \
            #     if 0 in t_result else (255,255,0),thickness=2)
        
        
        
    


if __name__ == "__main__":
    # 检测多视频
    main()
    # 检测单个视频
    # dete_signal_video()
    #在两卡服务器外接摄像头采集数据并检测
    # show_with_camera()
    #检测图片
    # dete_picture()