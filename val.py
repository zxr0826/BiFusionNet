import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # Select your trained model
    model.val(data='/root/workspace/PDT-dataset/data.yaml',
              split='test', 
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              project='runs/val',
              name='exp',
              )
