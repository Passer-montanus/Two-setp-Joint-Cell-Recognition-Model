import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') # select your model.pt path
    model.track(source="D:/CV/YOLOv8-DeepSORT/Vedio/Datasets/Junction.mp4",
                project='runs/track',
                name='exp',
                save=True
                )