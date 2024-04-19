import os.path
import threading
from queue import Queue

import PIL
import cv2
import mindspore as ms
# MindFace
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from utils.mindface.recognition.models import iresnet50, iresnet100, get_mbf, vit_t, vit_s, vit_b, vit_l
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from store.config import ConfigStore
from utils.common import singleton
from utils.mindface.detection.models import RetinaFace, resnet50, mobilenet025
from utils.mindface.detection.runner import DetectionEngine, read_yaml
from utils.mindface.detection.utils import prior_box
from utils.vision import VisionTools, draw_rectangle, draw_text
from utils.mediapipe.mp import MediaPipe as MP
import multiprocessing
import requests
from requests_toolbelt import MultipartEncoder
from time import sleep


class VisionService(QThread):
    """
    摄像头相关服务
    """
    resultSignal = pyqtSignal(str)  # 识别结果信号状态
    registerResSignal = pyqtSignal(str)  # 注册结果信号状态

    def __init__(self, store):
        super(VisionService, self).__init__()
        self.all_queues = {
            'display': Queue(),
            'video': Queue(),
        }
        self.isRunning = True
        self.config = store
        self.video_stream_in = ReadCameraThread(self.all_queues, self.resultSignal, self.registerResSignal, 0)

    def close_camera(self):
        """
        close camera and release resources
        """
        self.video_stream_in.stop_read_camera()
        self.config.set_config('camera_on', False)

    def start_camera(self):
        """
        start camera
        """
        self.video_stream_in.start()
        self.config.set_config('camera_on', True)

    def vision_face_register(self, name):
        self.video_stream_in.register_face_inner(name)


def classicFaceRecognize(faces, classicPredictor):
    res = classicPredictor.predict(faces)
    return res[1], res[2]


class ReadCameraThread(threading.Thread):
    """
    摄像头读取线程
    """

    def __init__(self, video_queue, resultSignal, registerResultSignal, camera_id=0):
        super(ReadCameraThread).__init__()
        threading.Thread.__init__(self)
        self.read_capture = cv2.VideoCapture(camera_id)
        # limit fps 30
        self.read_capture.set(cv2.CAP_PROP_FPS, 30)
        self.read_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.read_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_queue = video_queue
        self.resultSignal = resultSignal
        self.registerResultSignal = registerResultSignal
        self.vision_tools = VisionTools()
        self.running = True
        self.config = ConfigStore()
        self.recogQueue = self.config.recogQueue
        self.registerQueue = self.config.registerQueue
        self.mindface = None
        self.detect_config = {}
        self.classicFaceRecognizer = ClassicFaceRecognizer()
        self.recogThread = None
        self.mp = MP()
        self.current_face = None
        self.current_frame = None

    def run(self):
        self.mindfaceInit()
        self.recogThread = RecogThread(self.recogQueue, self.resultSignal,
                                       self.classicFaceRecognizer,
                                       self.mindface)
        self.recogThread.start()
        self.registerThread = RegisterThread(self.registerResultSignal)
        self.registerThread.start()
        if not self.read_capture.isOpened():
            print('未检测到摄像头')
            exit(0)

        while self.read_capture.isOpened() and self.running:
            ret, frame = self.read_capture.read()
            res = None
            faces = None
            if not ret:
                break
            self.current_frame = frame
            if self.config.get_config('face_detect'):
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['classic']:
                    faces = self.classicFaceRecognizer.detect_face(frame)
                    res = frame.copy()
                    for face in faces:
                        draw_rectangle(res, face)
                    pass
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['mediapipe']:
                    faces = self.mp.detect_face(frame)
                    res = self.mp.visualize(frame, faces)
                    faces = self.mp.transform_result(faces)
                    pass
                if self.config.get_config('detect_method') == self.config.detect_methods_mapper['mindspore']:
                    faces = self.mindfaceFaceDetect(frame)
                    res = frame.copy()
                    for face in faces:
                        draw_rectangle(res, face)
                    pass
                if len(faces) > 0:
                    self.current_face = faces[0]
            if self.config.get_config('recog_open') and faces is not None and len(faces) > 0:
                face_img = self.vision_tools.cut_face(frame, faces[0])
                if self.recogQueue.empty():
                    self.recogQueue.put(face_img)
            if frame is not None:
                self.video_queue['video'].put(frame)
            if res is not None:
                self.video_queue['display'].put(res)
            else:
                self.video_queue['display'].put(frame)

        self.read_capture.release()
        cv2.destroyAllWindows()

    def stop_read_camera(self):
        self.running = False

    def register_face_inner(self, name):
        self.registerQueue.put((self.vision_tools.cut_face(self.current_frame, self.current_face), name))

    def mindfaceInit(self):
        detect_config = 'utils/mindface/detection/configs/RetinaFace_mobilenet025.yaml'
        self.detect_config = read_yaml(detect_config)
        self.detect_config['val_model'] = 'utils/mindface/detection/pretrained/RetinaFace_MobileNet025.ckpt'
        self.detect_config['conf'] = 0.8
        self.mindface = MindFaceService(self.detect_config)

    def mindfaceFaceDetect(self, frame):
        boxes = self.mindface.face_detect(self.detect_config, frame)
        faces = []
        for box in boxes:
            if box[4] > self.detect_config['conf']:
                faces.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        return faces


@singleton
class ClassicFaceRecognizer:
    """
    传统面部识别能力
    """

    def __init__(self, method='lbph'):
        self.method = method
        self.vision = VisionTools()
        self.method = method
        self.trained = True
        self.predicting = False
        if self.method == 'lbph':
            # 创建我们的LBPH人脸识别器
            self.face_recognizer = cv2.face.LBPHFaceRecognizer.create()
        elif self.method == 'elgenface':
            # 创建我们的EigenFace人脸识别器
            self.face_recognizer = cv2.face.EigenFaceRecognizer.create()
        elif self.method == 'fisherface':
            # 或者使用FisherFaceRecognizer替换上面的行
            self.face_recognizer = cv2.face.FisherFaceRecognizer.create()

    def train(self, putLog):
        self.training_thread = ClassicTrainingDataThread(self.vision, "dataset/B210413/full", self.method)
        self.training_thread.log_signal.connect(putLog)
        self.training_thread.finish_signal.connect(self.finish_training)
        self.training_thread.start()

    def finish_training(self):
        # 训练我们的面部识别器
        self.face_recognizer.train(self.training_thread.faces, self.training_thread.labels)
        self.trained = True

    def detect_face(self, frame):
        """
        人脸检测
        :param frame: 一帧图像
        :return: 人脸坐标
        """
        return self.vision.lbp_detect_face(frame)

    def predict(self, faces):
        """
        识别测试图像中的人脸
        :param input_img: 输入的预测图像
        :return: 预测完成的图像、对应的id和概率
        """
        # 制作图像的副本，因为我们不想更改原始图像
        label_text = (None, None)
        # for face in faces:
        #     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #     # 使用我们的脸部识别器预测图像
        #     label = self.face_recognizer.predict(cv2.cvtColor(cv2.resize(face, (160, 160)), cv2.COLOR_BGR2GRAY))
        #     # 获取由人脸识别器返回的相应标签的名称
        #     label_text = label
        #     # 画预计人的名字
        #     # draw_text(img, f'B{label_text[0]} possibility: {label_text[1]}', face[0], face[1] - 5)
        # 使用我们的脸部识别器预测图像
        label = self.face_recognizer.predict(cv2.cvtColor(cv2.resize(faces, (160, 160)), cv2.COLOR_BGR2GRAY))
        # 获取由人脸识别器返回的相应标签的名称
        label_text = label
        # 画预计人的名字
        # draw_text(img, f'B{label_text[0]} possibility: {label_text[1]}', face[0], face[1] - 5)
        return None, label_text[0], label_text[1]


class ClassicTrainingDataThread(QThread):
    """
    传统方法下的比如说LBPH和PCA方法的训练线程
    """
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal()

    def __init__(self, vision_tools, data_folder_path, method='lbph'):
        super(ClassicTrainingDataThread, self).__init__()
        self.vision_tools = vision_tools
        self.data_folder_path = data_folder_path
        self.method = method
        self.faces = []
        self.labels = []

    def run(self):
        self.log_signal.emit("Preparing data...")
        faces, labels = self.vision_tools.prepare_training_data(self.data_folder_path, self.log_signal.emit,
                                                                self.method)
        self.faces = faces
        self.labels = labels
        self.log_signal.emit("Data prepared")
        self.finish_signal.emit()


class MindFaceService:
    def __init__(self, detectCfg):
        self.detect_options = {}
        self.recog_options = {}
        self.face_detect_prepare(detectCfg)
        self.recog_prepare()

    def face_detect_prepare(self, cfg):
        if cfg['mode'] == 'Graph':
            context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
        if cfg['name'] == 'ResNet50':
            backbone = resnet50(1001)
        elif cfg['name'] == 'MobileNet025':
            backbone = mobilenet025(1000)
        self.detect_options['network'] = RetinaFace(phase='predict', backbone=backbone,
                                                    in_channel=cfg['in_channel'],
                                                    out_channel=cfg['out_channel'])
        backbone.set_train(False)
        self.detect_options['network'].set_train(False)
        # load checkpoint
        assert cfg['val_model'] is not None, 'val_model is None.'
        param_dict = load_checkpoint(cfg['val_model'])
        print(f"Load trained model done. {cfg['val_model']}")
        self.detect_options['network'].init_parameters_data()
        load_param_into_net(self.detect_options['network'], param_dict)
        # testing image
        self.detect_options['conf_test'] = cfg['conf']
        self.detect_options['detection'] = DetectionEngine(nms_thresh=cfg['val_nms_threshold'],
                                                           conf_thresh=cfg['val_confidence_threshold'],
                                                           iou_thresh=cfg['val_iou_threshold'], var=cfg['variance'])
        self.detect_options['target_size'] = 1600
        self.detect_options['max_size'] = 2176
        self.detect_options['priors'] = prior_box(
            image_sizes=(self.detect_options['max_size'], self.detect_options['max_size']),
            min_sizes=[[16, 32], [64, 128], [256, 512]],
            steps=[8, 16, 32],
            clip=False)

    def face_detect(self, cfg, frame):
        """
        基于retinaface做的面部检测
        """
        img_raw = frame.copy()
        img = np.float32(img_raw)
        im_size_min = np.min(img.shape[0:2])
        im_size_max = np.max(img.shape[0:2])
        resize = float(self.detect_options['target_size']) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > self.detect_options['max_size']:
            resize = float(self.detect_options['max_size']) / float(im_size_max)
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        assert img.shape[0] <= self.detect_options['max_size'] and img.shape[1] <= self.detect_options['max_size']
        image_t = np.empty((self.detect_options['max_size'], self.detect_options['max_size'], 3), dtype=img.dtype)
        image_t[:, :] = (104.0, 117.0, 123.0)
        image_t[0:img.shape[0], 0:img.shape[1]] = img
        img = image_t
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = Tensor(img)
        boxes, confs, _ = self.detect_options['network'](img)
        boxes = self.detect_options['detection'].infer(boxes, confs, resize, scale, self.detect_options['priors'])
        return boxes

    def recog_prepare(self, backbone="mobilefacenet", num_features=512,
                      pretrained='utils/mindface/recognition/pretrained/mobile_casia_ArcFace.ckpt'):
        if backbone == 'iresnet50':
            self.recog_options['recog_model'] = iresnet50(num_features=num_features)
            print("Finish loading iresnet50")
        elif backbone == 'iresnet100':
            self.recog_options['recog_model'] = iresnet100(num_features=num_features)
            print("Finish loading iresnet100")
        elif backbone == 'mobilefacenet':
            self.recog_options['recog_model'] = get_mbf(num_features=num_features)
            print("Finish loading mobilefacenet")
        elif backbone == 'vit_t':
            self.recog_options['recog_model'] = vit_t(num_features=num_features)
            print("Finish loading vit_t")
        elif backbone == 'vit_s':
            self.recog_options['recog_model'] = vit_s(num_features=num_features)
            print("Finish loading vit_s")
        elif backbone == 'vit_b':
            self.recog_options['recog_model'] = vit_b(num_features=num_features)
            print("Finish loading vit_b")
        elif backbone == 'vit_l':
            self.recog_options['recog_model'] = vit_l(num_features=num_features)
            print("Finish loading vit_l")
        else:
            raise NotImplementedError
        if pretrained:
            param_dict = load_checkpoint(pretrained)
            load_param_into_net(self.recog_options['recog_model'], param_dict)

    def recog_infer(self, img):
        """
        The inference of arcface.

        Args:
            img (NumPy): The input image.
            backbone (Object): Arcface model without loss function. Default: "iresnet50".
            pretrained (Bool): Pretrain. Default: False.
        """
        assert (img.shape[-1] == 112 and img.shape[-2] == 112)
        img = ((img / 255) - 0.5) / 0.5
        img = ms.Tensor(img, ms.float32)
        if len(img.shape) == 4:
            pass
        elif len(img.shape) == 3:
            img = img.expand_dims(axis=0)
        net_out = self.recog_options['recog_model'](img)
        embeddings = net_out.asnumpy()
        return embeddings

    def compare_embedding(self, emb1, emb2):
        """
        计算特征向量差异
        :param emb1: 特征向量1
        :param emb2: 特征向量2
        :return:
        """
        return np.linalg.norm(emb1 - emb2, axis=1)


class RecogThread(QThread):
    """
    识别线程
    """

    def __init__(self, recogQueue, resultSignal, classsicFaceRecognizer, mindface):
        super(RecogThread, self).__init__()
        self.vision_tools = VisionTools()
        self.recogQueue = recogQueue
        self.resultSignal = resultSignal
        self.classicFaceRecognizer = classsicFaceRecognizer
        self.mindface = mindface
        self.config = ConfigStore()
        self.faces = []
        self.labels = []

    def run(self):
        # keep only one element in the queue
        while True:
            if not self.recogQueue.empty():
                faces = self.recogQueue.get()
                if self.config.get_config('recog_method') == self.config.recog_methods_mapper['classic']:
                    label, possibility = classicFaceRecognize(faces, self.classicFaceRecognizer)
                    print(f"possibility: {possibility} name: {label}")
                    self.resultSignal.emit(f"B{label}")
                if self.config.get_config('recog_method') == self.config.recog_methods_mapper['mindspore']:
                    # 发请求到mindfaceServer
                    cv2.imwrite('temp/recog.jpg', faces)
                    url = "http://114.116.250.18:8000/recognize"
                    m = MultipartEncoder(
                        fields={'photo': ('recog.jpg', open('temp/recog.jpg', 'rb'), 'image/jpeg')}
                    )
                    headers = {
                        'Content-Type': m.content_type,
                    }
                    response = requests.request("POST", headers=headers, url=url, data=m)
                    sleep(3)
                    print(response)
                    if response.status_code == 200:
                        self.resultSignal.emit(response.json()['name'])
                    else:
                        # self.resultSignal.emit("未知")
                        pass

class RegisterThread(QThread):
    def __init__(self, registerResultSignal):
        super(RegisterThread, self).__init__()
        self.config = ConfigStore()
        self.registerQueue = self.config.registerQueue
        self.registerResultSignal = registerResultSignal

    def run(self):
        while True:
            if not self.registerQueue.empty():
                faces, name = self.registerQueue.get()
                cv2.imwrite('temp/register.jpg', faces)
                url = "http://114.116.250.18:8000/register"
                m = MultipartEncoder(
                    fields={'name': name, 'photo': ('register.jpg',
                                                    open('temp/register.jpg',
                                                         'rb'), 'image/jpeg')}
                )
                headers = {
                    'Content-Type': m.content_type,
                }
                response = requests.request("POST", headers=headers, url=url, data=m)
                print(response)
                if response.status_code == 200:
                    self.registerResultSignal.emit("注册成功")
                    os.path.exists('temp/register.jpg') and os.remove('temp/register.jpg')
                    # 检查dataset/full下是否有该人的文件夹，没有则创建
                    if not os.path.exists(f'dataset/full/{name}'):
                        os.makedirs(f'dataset/full/{name}')
                    # 将照片保存到dataset/full下
                    cv2.imwrite(f'dataset/full/{name}/{name}.jpg', faces)
                else:
                    self.registerResultSignal.emit("注册失败")
                    os.path.exists('temp/register.jpg') and os.remove('temp/register.jpg')
