import os
import sys
import cv2
import numpy as np

sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector


def compute_iou(gt_box, b_box):
    '''
    计算iou
    :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
    :param b_box: bounding box b_box 表示形式同上
    :return:
    '''
    width0 = gt_box[2] - gt_box[0]
    height0 = gt_box[3] - gt_box[1]
    width1 = b_box[2] - b_box[0]
    height1 = b_box[3] - b_box[1]
    max_x = max(gt_box[2], b_box[2])
    min_x = min(gt_box[0], b_box[0])
    width = width0 + width1 - (max_x - min_x)
    max_y = max(gt_box[3], b_box[3])
    min_y = min(gt_box[1], b_box[1])
    height = height0 + height1 - (max_y - min_y)

    interArea = width * height
    boxAArea = width0 * height0
    boxBArea = width1 * height1
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou


def detect_video(mtcnn_detector, videos_root_path, save_path_root):
    videos_paths = os.listdir(videos_root_path)
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)

    for video_name in videos_paths:
        cap = cv2.VideoCapture(os.path.join(videos_root_path, video_name))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        save_path = os.path.join(save_path_root, video_name)
        out = cv2.VideoWriter(save_path, fourcc, fps, size)
        print('save path: =====>', save_path)
        while True:
            ret, frame = cap.read()
            if ret:
                bboxs, landmarks, wearmask = mtcnn_detector.detect_face(frame, rgb=True)
                # 画人脸框
                if bboxs is not None:
                    for i in range(bboxs.shape[0]):
                        bbox = np.round(bboxs[i, 0:4]).astype(int)
                        score = bboxs[i, 4]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (0, 0, 255), 2)
                        if wearmask is None:
                            cv2.putText(frame, '{:.3f}'.format(score), (int(bbox[0]), int(bbox[1])),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        1.3,
                                        (255, 0, 255), thickness=2)
                        else:
                            cv2.putText(frame, '{:.3f}, {:.3f}'.format(score, float(wearmask[i])),
                                        (int(bbox[0]), int(bbox[1])),
                                        cv2.FONT_HERSHEY_COMPLEX,
                                        1.3,
                                        (255, 0, 255) if float(wearmask[i]) < 0.5 else (0, 255, 0), thickness=2)

                if landmarks is not None:
                    for i in range(landmarks.shape[0]):
                        landmarks_one = landmarks[i, :]
                        landmarks_one = landmarks_one.reshape((point_nums, 2))
                        for j in range(point_nums):
                            cv2.circle(frame, (int(landmarks_one[j, 0]), int(landmarks_one[j, 1])), 2, (255, 0, 0), -1)
                out.write(frame)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    point_nums = 24
    threshold = [0.6, 0.7, 0.7]  # [0.99, 0.1, 0.6]  #
    pnet, rnet, onet = create_mtcnn_net(
        p_model_path=r'model_store/final/pnet_epoch_19.pt',
        r_model_path=r'model_store/final/rnet_epoch_7.pt',
        o_model_path=r'model_store/final/onet_epoch_92.pt',
        use_cuda=True)

    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=threshold)
    videos_root_path = 'test_video/'
    save_path_root = 'result_video'
    detect_video(mtcnn_detector, videos_root_path, save_path_root)
