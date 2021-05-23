# -*- coding: utf-8 -*-
import os
import cv2
import shutil
import copy
import json
from mbhk_get_eye import get_label
from PIL import Image,ImageDraw
import numpy as np
labelDst = {"close_eyes": "闭眼", "yawn": "打哈欠", "phone": "打电话", "smoking": "吸烟", "sunglasses_block": "红外阻断",
            "drink_water": "喝水", "toothpick": "叼牙签", "eat_something": "吃东西", \
            "speaking": "张嘴说话", "wear_mask": "戴口罩", "look_down": "往下看", "unclear_eyes": "眼部不清晰",
            "glasses_noBlocking": "戴眼睛", "face_occlusion": "脸部遮挡", "face_big": "脸部姿态过大"
            }

keys_106 = ['contour_left1', 'contour_left2', 'contour_left3', 'contour_left4', 'contour_left5', 'contour_left6',
            'contour_left7', 'contour_left8',
            'contour_left9', 'contour_left10', 'contour_left11', 'contour_left12', 'contour_left13', 'contour_left14',
            'contour_left15', 'contour_left16',
            'contour_chin', 'contour_right16', 'contour_right15', 'contour_right14', 'contour_right13',
            'contour_right12', 'contour_right11', 'contour_right10',
            'contour_right9', 'contour_right8', 'contour_right7', 'contour_right6', 'contour_right5', 'contour_right4',
            'contour_right3', 'contour_right2',
            'contour_right1',
            'nose_bridge1', 'nose_bridge2',
            'nose_bridge3', 'nose_tip', 'nose_left_contour1', 'nose_left_contour2', 'nose_left_contour3',
            'nose_left_contour4', 'nose_left_contour5',
            'nose_middle_contour', 'nose_right_contour5', 'nose_right_contour4', 'nose_right_contour3',
            'nose_right_contour2', 'nose_right_contour1',
            'left_eyebrow_left_corner', 'left_eyebrow_upper_left_quarter', 'left_eyebrow_upper_middle',
            'left_eyebrow_upper_right_quarter',
            'left_eyebrow_upper_right_corner', 'left_eyebrow_lower_right_corner', 'left_eyebrow_lower_right_quarter',
            'left_eyebrow_lower_middle',
            'left_eyebrow_lower_left_quarter',
            'right_eyebrow_upper_left_corner', 'right_eyebrow_upper_left_quarter',
            'right_eyebrow_upper_middle', 'right_eyebrow_upper_right_quarter', 'right_eyebrow_right_corner',
            'right_eyebrow_lower_right_quarter',
            'right_eyebrow_lower_middle', 'right_eyebrow_lower_left_quarter', 'right_eyebrow_lower_left_corner',
            'left_eye_left_corner', 'left_eye_upper_left_quarter', 'left_eye_top', 'left_eye_upper_right_quarter',
            'left_eye_right_corner',
            'left_eye_lower_right_quarter', 'left_eye_bottom', 'left_eye_lower_left_quarter', 'left_eye_pupil',
            'left_eye_center',
            'right_eye_left_corner', 'right_eye_upper_left_quarter', 'right_eye_top', 'right_eye_upper_right_quarter',
            'right_eye_right_corner',
            'right_eye_lower_right_quarter', 'right_eye_bottom', 'right_eye_lower_left_quarter', 'right_eye_pupil',
            'right_eye_center',
            'mouth_left_corner', 'mouth_upper_lip_left_contour2', 'mouth_upper_lip_left_contour1',
            'mouth_upper_lip_top', 'mouth_upper_lip_right_contour1',
            'mouth_upper_lip_right_contour2', 'mouth_right_corner', 'mouth_lower_lip_right_contour2',
            'mouth_lower_lip_right_contour3',
            'mouth_lower_lip_bottom', 'mouth_lower_lip_left_contour3', 'mouth_lower_lip_left_contour2',
            'mouth_upper_lip_left_contour3',
            'mouth_upper_lip_left_contour4', 'mouth_upper_lip_bottom', 'mouth_upper_lip_right_contour4',
            'mouth_upper_lip_right_contour3',
            'mouth_lower_lip_right_contour1', 'mouth_lower_lip_top', 'mouth_lower_lip_left_contour1']

colorDst = {"face": (0, 0, 255), "body": (0, 255, 0), "phone": (255, 0, 0), "telephone": (255, 255, 255),
            "smoke": (0, 255, 255), "smokewithhand": (255, 255, 0), "SafetyBelt": (255, 0, 255)}


def read_img_path(img_dir):
    # name.jpg
    # {"name":path}
    # n*2数组
    # lst = []
    # with open(txtname,"r") as f:
    #     a = f.readlines()
    #     for line in a:
    #         lst.append(line.strip())
    img_path = []
    for curdir, subdir, files in os.walk(img_dir):
        for img in (file for file in files if file.endswith('jpg')):
            l_path = os.path.join(curdir, img)
            # if os.path.splitext(img)[0] not in lst:
            img_path.append([os.path.splitext(img)[0], l_path])
    return img_path


def read_json_path(json_dir):
    # name.json
    # {name:path}
    json_path = {}
    for curdir, subdir, files in os.walk(json_dir):
        for l_json in (file for file in files if file.endswith('json')):
            l_path = os.path.join(curdir, l_json)
            json_path[os.path.splitext(l_json)[0]] = l_path
    return json_path


def search_img(img_dir):
    dst = {}
    for curdir, subdir, files in os.walk(img_dir):
        for img in (file for file in files if file.endswith('jpg')):
            l_path = os.path.join(curdir, img)
            dst[img] = l_path

    return dst


def read_json(directory):
    sumlst = []
    attrDst, keyDst, infoDst = {}, {}, {}
    try:
        with open(directory, "r", encoding="utf8") as fp:
            json_data = json.load(fp)
            for j in range(len(json_data)):
                sublst = []
                if "name" in json_data[j]:
                    if json_data[j]["name"] == "dangerous_driving":
                        # 属性
                        attr = json_data[j]["attributes"]
                        for key in attr:
                            attrDst[key] = attr[key]
                            # attrDst[labelDst[key]] = attr[key]

                        # 人脸
                        cc = json_data[j]["points"]
                        if cc[0][0] is not None:
                            xmin = int(min(cc[0][0], cc[1][0]))
                            ymin = int(min(cc[0][1], cc[1][1]))
                            xmax = int(max(cc[0][0], cc[1][0]))
                            ymax = int(max(cc[0][1], cc[1][1]))
                            w = xmax - xmin
                            h = ymax - ymin
                            if xmin < 0:
                                xmin = 0
                            if ymin < 0:
                                ymin = 0
                            if w < 0:
                                w = -w
                            if h < 0:
                                h = -h
                            sublst.append(xmin)
                            sublst.append(ymin)
                            sublst.append(w)
                            sublst.append(h)
                            if "face" not in infoDst:
                                infoDst["face"] = []
                            infoDst["face"].append(sublst)

                    if json_data[j]["name"] == "humanbody":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0], cc[1][0]))
                        ymin = int(min(cc[0][1], cc[1][1]))
                        xmax = int(max(cc[0][0], cc[1][0]))
                        ymax = int(max(cc[0][1], cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "body" not in infoDst:
                            infoDst["body"] = []
                        infoDst["body"].append(sublst)

                    if json_data[j]["name"] == "callwithhand":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0], cc[1][0]))
                        ymin = int(min(cc[0][1], cc[1][1]))
                        xmax = int(max(cc[0][0], cc[1][0]))
                        ymax = int(max(cc[0][1], cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "phone" not in infoDst:
                            infoDst["phone"] = []
                        infoDst["phone"].append(sublst)

                    if json_data[j]["name"] == "telephone":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0], cc[1][0]))
                        ymin = int(min(cc[0][1], cc[1][1]))
                        xmax = int(max(cc[0][0], cc[1][0]))
                        ymax = int(max(cc[0][1], cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "telephone" not in infoDst:
                            infoDst["telephone"] = []
                        infoDst["telephone"].append(sublst)

                    if json_data[j]["name"] == "smoke":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0], cc[1][0]))
                        ymin = int(min(cc[0][1], cc[1][1]))
                        xmax = int(max(cc[0][0], cc[1][0]))
                        ymax = int(max(cc[0][1], cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "smoke" not in infoDst:
                            infoDst["smoke"] = []
                        infoDst["smoke"].append(sublst)

                    if json_data[j]["name"] == "smokewithhand":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0], cc[1][0]))
                        ymin = int(min(cc[0][1], cc[1][1]))
                        xmax = int(max(cc[0][0], cc[1][0]))
                        ymax = int(max(cc[0][1], cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "smokewithhand" not in infoDst:
                            infoDst["smokewithhand"] = []
                        infoDst["smokewithhand"].append(sublst)

                    if json_data[j]["name"] == "SafetyBelt":
                        cc = json_data[j]["points"]
                        xmin = int(min(cc[0][0],cc[1][0]))
                        ymin = int(min(cc[0][1],cc[1][1]))
                        xmax = int(max(cc[0][0],cc[1][0]))
                        ymax = int(max(cc[0][1],cc[1][1]))
                        w = xmax - xmin
                        h = ymax - ymin
                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if w < 0:
                            w = -w
                        if h < 0:
                            h = -h
                        sublst.append(xmin)
                        sublst.append(ymin)
                        sublst.append(w)
                        sublst.append(h)
                        if "SafetyBelt" not in infoDst:
                            infoDst["SafetyBelt"] = []
                        infoDst["SafetyBelt"].append(sublst)

                    if json_data[j]["name"] in keys_106:
                        keyDst[json_data[j]["name"]] = [int(x) for x in json_data[j]["points"][0]]


        sumlst.append(infoDst)
        sumlst.append(attrDst)
        sumlst.append(keyDst)
        return sumlst

    except:
        f = open("error.txt", "a")
        f.write("{}\n".format(directory))
        f.close()
        return 0


def plot_img(image, info):
    h, w, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 画框
    infoDst = info[0]
    y_axis_face = 0
    for key in infoDst:
        color = colorDst[key]
        for subInfo in infoDst[key]:
            cv2.putText(image, key, (int(w * (42 / 1280)), int(h * (131 / 720)) + y_axis_face), font, 1.2, color, 2)
            cv2.rectangle(image, (subInfo[0], subInfo[1]), (subInfo[0] + subInfo[2], subInfo[1] + subInfo[3]), color, 1)
            y_axis_face += 50

    # 属性
    attrDst = info[1]
    y_axis = 0
    for key in attrDst:
        if attrDst[key]:
            # key = key.decode('utf8')
            # if not isinstance(key,unicode):
            #     key = key.decode('utf8')
            cv2.putText(image, key, (int(w * (1056 / 1280)), int(h * (159 / 720)) + y_axis), font, 1.2, (0, 255, 0), 2)
            y_axis += 50

    # 关键点
    keyDst = info[2]
    for key in keyDst:
        cv2.circle(image, (keyDst[key][0], keyDst[key][1]), 1, (0, 255, 0))

    return image

def get_change_label(label):
    '''

    '''
    
    # 扩大眼睛框
    eye_point = [label[0:4],label[4:8]]
    label_min_max = []
    for tdata in eye_point:
        xmin,ymin,xmax,ymax = tdata
        # cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,0,50),2)
        w,h = xmax - xmin, ymax - ymin
        k = np.random.random_sample()*0.1+0.05
        ratio = h/w
        if ratio > 1:
            ratio = ratio - 1
            xmin -= (ratio/2*w+k*h)
            ymin -= (k*h)
            xmax += (ratio/2*w+k*h)
            ymax += (k*h)
            label_min_max.extend([int(xmin),int(ymin),int(xmax),int(ymax)])
            # crop_img.append(img.crop((int(xmin),int(ymin),int(xmax),int(ymax))))
        else:
            ratio = w/h - 1
            xmin -= (k*w)
            ymin -= (ratio/2*h+k*w)
            xmax += (k*w)
            ymax += (ratio/2*h + k*w)
            label_min_max.extend([int(xmin),int(ymin),int(xmax),int(ymax)])
    return label_min_max

def main():
    path = "error_class/change_img/close_eye"
    dst = search_img(path)
    string = "error_class/change_img/close_eye"
    a = read_json(string)
    # jpgname = os.path.basename(string)
    # image = cv2.imread(dst[jpgname.replace("json","jpg")])

    # plot_img(image,a)

    # # cc = a[0]["face"]
    # # for i in cc:
    # #     cv2.rectangle(image,(i[0],i[1]),(i[0] + i[2],i[1] + i[3]),(0,255,0),1)
    # cv2.imshow("a",image)
    # cv2.waitKey(0)
    # # print(cc)

def show():
    path = "error_class/change_img/close_eye"
    isEyePath = "error_class/temp"
    isHongPath = "error_class/temp"
    isLookDownPath = "error_class/temp"
    sumPath = "error_class/temp"
    big_head_pose = "error_class/temp"
    if not os.path.exists("error_class/temp"):
        os.mkdir("error_class/temp")
    # if not os.path.exists(isEyePath):
    #     os.mkdir(isEyePath)
    # if not os.path.exists(isHongPath):
    #     os.mkdir(isHongPath)
    # if not os.path.exists(isLookDownPath):
    #     os.mkdir(isLookDownPath)
    # if not os.path.exists(sumPath):
    #     os.mkdir(sumPath)
    # if not os.path.exists(big_head_pose):
    #     os.mkdir(big_head_pose)



    # name path
    # text_save_path = os.path.join(r"C:\Users\Mic_hu\Desktop\headpose_test_data\20210122_data","finish.txt")
    # f_finish = open(text_save_path,"a")
    img_path = read_img_path(path)
    json_path = read_json_path(path)
    count = 0 
    while count < len(img_path):
        img = cv2.imread(img_path[count][1])
        # img_write = cv2.imread(img_path[count][1])
        img_name = img_path[count][0]
        json_info = read_json(json_path[img_name])
        tlabel_min_max = get_label(json_path[img_name])
        label_min_max = get_change_label(tlabel_min_max)
        left_eye,right_eye = label_min_max[0:4],label_min_max[4:8]

        if json_info:
            plot_img(img,json_info)
            cv2.rectangle(img,(int(left_eye[0]),int(left_eye[1])),(int(left_eye[2]),int(left_eye[3])),(0,255,0),2)
            cv2.rectangle(img,(int(right_eye[0]),int(right_eye[1])),(int(right_eye[2]),int(right_eye[3])),(0,255,0),2)
            cv2.imshow("test",img)
            # cv2.imshow("source",img_write)
            get_input = cv2.waitKey(0)
            count += 1
        #     if chr(get_input) == "q":
        #         # f_finish.close()
        #         pass
        #         break
        #     elif chr(get_input) == " ":
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #     elif chr(get_input) == "f":            
        #         count -= 1
        #         if count <= 0:
        #             count = 0
        #     #赋予你选择错误路径的权力
        #     # elif chr(get_input) == "a":
        #     #     f_finish.write("{}\n".format(img_name))
        #     #     count += 1
        #     #     input_mode = input("eye:1,hong:2,look:3,sum:4  --->:")
                

        #     elif chr(get_input) == "v":
        #         cv2.imwrite(os.path.join(isEyePath,img_name + ".jpg"),img_write)
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #         print("eye:{}".format(img_name))
        #     elif chr(get_input) == "b":
        #         cv2.imwrite(os.path.join(isHongPath,img_name + ".jpg"),img_write)
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #         print("red_block:{}".format(img_name))
        #     elif chr(get_input) == "n":
        #         #向下看
        #         cv2.imwrite(os.path.join(isLookDownPath,img_name + ".jpg"),img_write)
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #         print("look_down:{}".format(img_name))
        #     elif chr(get_input) == "m":
        #         #其他
        #         cv2.imwrite(os.path.join(sumPath,img_name + ".jpg"),img_write)
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #         print("sume:{}".format(img_name))
        #     elif chr(get_input) == "c":
        #         #大姿态
        #         cv2.imwrite(os.path.join(big_head_pose,img_name + ".jpg"),img_write)
        #         # f_finish.write("{}\n".format(img_name))
        #         count += 1
        #         print("big_head_pose:{}".format(img_name))
        #     else:
        #         print("input error....")
        #         pass
        #         # input("输错啦,只能是1,2,3,4某个值！--->:")
        # else:
        #     count += 1
        #     # f_finish.write("{}\n".format(img_name))
        # print("img:{}/{}".format(count,len(img_path)))




show()
