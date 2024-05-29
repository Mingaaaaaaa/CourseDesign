import os

import cv2
import numpy


def draw_text(img, text, x, y):
    """
    在图像上绘制文本
    :param text:
    :param x:
    :param y:
    :return:
    """
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def draw_rectangle(img, rect):
    """
    函数在图像上绘制矩形
    :param rect: 宽高坐标
    """
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


class VisionTools:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('pretrained/haarcascade_frontalface_default.xml')
        pass

    def lbp_detect_face(self, frame):
        """
        人脸检测
        :param frame: 一帧图像
        :return: 人脸坐标
        """
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(image=gray, scaleFactor=1.2, minNeighbors=5)
        return faces

    def cut_face(self, frame, face):
        """
        截取人脸
        :param frame: 一帧图像
        :param face: 人脸坐标
        :return: 人脸图像
        """
        (x, y, w, h) = face
        return frame[y:y + w, x:x + h]

    def prepare_training_data(self, data_folder_path, putLog, method='lbph'):
        """
        该功能将读取所有人的训练图像，从每个图像检测人脸
        并将返回两个完全相同大小的列表，一个列表
        每张脸的脸部和另一列标签
        :param method: 算法类型
        :param data_folder_path: 包含所有主题的目录的路径
        """

        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []

        print("Preparing data...")
        total = len(dirs)
        current = 0

        for dir_name in dirs:
            current += 1
            print(f"Processing {current}/{total}")
            putLog(f"Processing {current}/{total}")
            if not dir_name.startswith("B"):
                continue

            label = dir_name
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            subject_images_names = os.listdir(subject_dir_path)
            for image_name in subject_images_names:

                if image_name.startswith("."):
                    continue

                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path)

                face = self.lbp_detect_face(image)

                if face is not None:
                    if len(face) > 0:
                        (x, y, w, h) = face[0]
                        if w > 0 and h > 0:
                            rect = image[y:y + w, x:x + h]
                            # cv2.imshow("Training on image...", rect)
                            # cv2.waitKey(100)
                            # 将脸添加到脸部列表
                            if method == 'eigenface':
                                faces.append(cv2.cvtColor(cv2.resize(rect, (160, 160)), cv2.COLOR_BGR2GRAY))
                            else:
                                faces.append(cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY))
                            # 为这张脸添加标签
                            labels.append(int(label.replace('B', '')))

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return faces, numpy.array(labels)