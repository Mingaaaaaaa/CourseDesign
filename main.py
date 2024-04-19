"""
课程设计主模块
:author: 李正楠
:date: 2024-4-14
"""
import sys

import cv2
from PyQt5.QtWidgets import QApplication

from service.vision import VisionService, ClassicFaceRecognizer
from store.config import ConfigStore
from view.main import MainPage

if __name__ == "__main__":
    app = QApplication(sys.argv)
    store = ConfigStore()
    classicPreditor = ClassicFaceRecognizer()
    camera = VisionService(store)
    window = MainPage(camera.all_queues, classicPreditor, camera, store)
    window.show()
    sys.exit(app.exec())
