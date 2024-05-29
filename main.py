import sys

import cv2
from PyQt5.QtWidgets import QApplication

from service.vision import VisionService, ClassicFaceRecognizer
from store.config import ConfigStore
from view.main import MainPage

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        store = ConfigStore()
        classicPreditor = ClassicFaceRecognizer()
        camera = VisionService(store)
        window = MainPage(camera.all_queues, classicPreditor, camera, store)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"An error occurred: {e}")
