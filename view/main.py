import os
import sys
import threading

import cv2
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from datetime import datetime
import multiprocessing

from store.config import ConfigStore
from utils.common import singleton
from view.mainDisplay import Ui_System
from PyQt5.QtWidgets import QMainWindow, QApplication


class MainPage(QMainWindow, Ui_System):
    logQueue = multiprocessing.Queue()  # 日志队列
    receiveLogSignal = pyqtSignal(str)  # LOG 信号状态

    def __init__(self, all_queues, classicPredictor, visionService, config):
        super(MainPage, self).__init__()
        self.setupUi(self)
        self.visionService = visionService
        self.config = config

        # 摄像头开关设置
        self.switchButton.clicked.connect(self.switch_camera)

        # 摄像头帧轮询
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.all_queues = all_queues

        # 人脸检测设置
        self.openDetection.stateChanged.connect(self.open_detection)
        self.detectMethod.currentIndexChanged.connect(self.change_detect_method)

        # 人脸识别设置
        self.openRecognition.stateChanged.connect(self.open_recognition)
        self.recogMethod.currentIndexChanged.connect(self.change_recog_method)
        self.classicPredictor = classicPredictor

        # 日志系统
        self.receiveLogSignal.connect(lambda log: self.logOutput(log))
        self.logOutputThread = threading.Thread(target=self.receiveLog, daemon=True)
        self.logOutputThread.start()

        # 训练按钮
        self.startTrainButton.clicked.connect(self.start_train)

        # 人脸注册按钮
        self.imgRegister.clicked.connect(self.register_face)
        self.visionService.registerResSignal.connect(lambda res: self.handleRegister(res))

        # 识别结果显示
        self.visionService.resultSignal.connect(lambda result: self.recogOutput(result))

    def switch_camera(self):
        if not self.timer.isActive():
            self.visionService.start_camera()
            self.timer.start(1)
            self.switchButton.setText("摄像头已开启")
            self.putLog("摄像头已开启")
            self.switchButton.setDisabled(True)
        else:
            self.visionService.close_camera()
            self.timer.stop()
            self.switchButton.setText("打开摄像头")

    def update_frame(self):
        config = ConfigStore()
        if self.classicPredictor.trained:
            config.set_config('recog_open_disabled', False)
            self.openRecognition.setDisabled(False)
            self.recogMethod.setDisabled(False)
        if not self.all_queues['display'].empty():
            frame = self.all_queues['display'].get()
            self.displayImage(frame, self.displayField)

    def displayImage(self, img, qlabel):
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # default：The image is stored using 8-bit indexes into a colormap， for example：a gray image
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        # img.shape[1]：图像宽度width，img.shape[0]：图像高度height，img.shape[2]：图像通道数
        # QImage.__init__ (self, bytes data, int width, int height, int bytesPerLine, Format format)
        # 从内存缓冲流获取img数据构造QImage类
        # img.strides[0]：每行的字节数（width*3）,rgb为3，rgba为4
        # strides[0]为最外层(即一个二维数组所占的字节长度)，strides[1]为次外层（即一维数组所占字节长度），strides[2]为最内层（即一个元素所占字节长度）
        # 从里往外看，strides[2]为1个字节长度（uint8），strides[1]为3*1个字节长度（3即rgb 3个通道）
        # strides[0]为width*3个字节长度，width代表一行有几个像素

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        qlabel.setPixmap(QPixmap.fromImage(outImage))
        qlabel.setScaledContents(True)  # 图片自适应大小

    def open_detection(self):
        if self.openDetection.isChecked():
            self.putLog("✅️人脸检测已开启")
            self.config.set_config('face_detect', True)
        else:
            self.putLog("🛑人脸检测已关闭")
            self.config.set_config('face_detect', False)

    def change_detect_method(self):
        self.putLog(f"切换人脸检测算法: {self.detectMethod.currentText()}")
        self.config.set_config('detect_method', self.detectMethod.currentIndex())

    def open_recognition(self):
        if self.openRecognition.isChecked():
            self.putLog("✅️人脸识别已开启")
            self.config.set_config('recog_open', True)
        else:
            self.putLog("🛑人脸识别已关闭")
            self.config.set_config('recog_open', False)

    def change_recog_method(self):
        self.putLog(f"切换人脸识别算法: {self.recogMethod.currentText()}")
        self.config.set_config('recog_method', self.recogMethod.currentIndex())

    def putLog(self, log):
        self.logQueue.put(log)

    def receiveLog(self):
        """
        系统日志服务常驻，接收并处理系统日志
        :return:
        """
        while True:
            data = self.logQueue.get()
            if data:
                self.receiveLogSignal.emit(data)
            else:
                continue

    def recogOutput(self, result):
        self.resultLabel.setText(result)
        picList = os.listdir(os.path.join('dataset', 'full', result))
        resPic = cv2.imread(os.path.join('dataset', 'full', result, picList[0]))
        self.displayImage(resPic, self.resultFace)

    def logOutput(self, log):
        """
        LOG输出
        :param log:
        :return:
        """
        # 获取当前系统时间
        time = datetime.now().strftime('[%Y/%m/%d %H:%M:%S]')
        log = time + ' ' + log + '\n'

        self.logText.moveCursor(QTextCursor.End)
        self.logText.insertPlainText(log)
        self.logText.ensureCursorVisible()  # 自动滚屏

    def start_train(self):
        self.classicPredictor.train(self.putLog)
        self.putLog("开始训练")

    def register_face(self):
        """
        异步请求http://114.116.250.18:8000/register接口
        接口参数为 name, photo
        """
        name = self.nameInput.text()
        if not name:
            self.putLog("请输入姓名")
            return
        self.putLog("开始注册")
        self.putLog(f"姓名: {name}")
        self.putLog("请稍后...")
        self.imgRegister.setDisabled(True)
        self.visionService.vision_face_register(name)

    def handleRegister(self, res):
        if res == '注册成功':
            self.putLog("注册成功")
        else:
            self.putLog("注册失败")
        self.imgRegister.setDisabled(False)
