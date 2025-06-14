from ultralytics import YOLO
from pathlib import Path

class Detector:
    def __init__(self, model_path: str = "yoloapp/models/yolo11n.pt"):
        self.model = YOLO(model_path)

    def inference(self, img_path: Path):
        return self.model(img_path)