import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/Steels/runs/detect/train/weights/best.pt')
    # model = YOLO('runs/train/mixup2_yolov8-C2f-DBB_100_32/weights/best.pt')
    # model = YOLO('runs/train/mixup2_yolov8-C2f-ODConv_100_32/weights/best.pt')
    # model = YOLO('runs/train/mixup2_yolov8-C2f-OREPA_100_32/weights/best.pt')
    # model = YOLO('weights/best.pt')
    # model = YOLO('runs/train/mixup2_yolov8-SPPF-LSKA_100_32/weights/best.pt')
    # model.val(data='dataset/data.yaml',
    model.val(data='Dataset_Steel/data.yaml',
              split='test',
              batch=16,
              # rect=False,
              save_json=True, # if you need to cal coco metrice
              project='runs/Steek',
              name='yolov8n_300',
              )