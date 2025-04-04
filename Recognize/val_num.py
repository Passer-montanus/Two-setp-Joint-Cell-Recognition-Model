"""
 批量验证脚本
"""

## 错误不中断
import warnings
import os
from ultralytics import YOLO

warnings.filterwarnings('ignore')

# 定义文件夹
folders = ['train']
base_dir = 'runs'  # 基础目录路径，根据你的实际情况调整

# 定义共用的配置参数
data_yaml = 'change_classes/data.yaml'
img_size = 640
batch_size = 16
project = 'runs/val_coco_better'

# 用于记录验证失败的模型
failed_models = []

# 遍历文件夹，自动识别所有子文件夹（模型名称）
for folder in folders:
    # folder_path = os.path.join(base_dir, folder, 'runs/train')
    folder_path = os.path.join(base_dir, folder)
    if os.path.exists(folder_path):
        models = next(os.walk(folder_path))[1]  # 获取所有子目录名称
        for model_name in models:
            weight_path = os.path.join(folder_path, model_name, 'weights/best.pt')
            if os.path.exists(weight_path):
                try:
                    print(f'Validating model: {model_name} in folder: {folder}')
                    # 加载模型
                    model = YOLO(weight_path)
                    # 执行验证
                    model.val(data=data_yaml,
                              split='test',  # 或 'val' 根据需求
                              imgsz=img_size,
                              batch=batch_size,
                              save_json=True,  # 如果需要计算COCO指标
                              project=project,
                              name=model_name,  # 使用模型名称作为输出目录的一部分
                              )
                    print(f"Validation completed for model: {model_name} in folder: {folder}")
                except Exception as e:
                    print(f"Failed to validate model: {model_name} in folder: {folder}. Error: {e}")
                    failed_models.append((folder, model_name))
            else:
                print(f"Weight file does not exist for model: {model_name} in folder: {folder}")
    else:
        print(f"Folder does not exist: {folder_path}")

# 输出未能成功验证的模型
if failed_models:
    print("The following models could not be validated successfully:")
    for folder, model_name in failed_models:
        print(f"Model: {model_name} in folder: {folder}")
else:
    print("All models were validated successfully.")



# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('results/216_1/runs/train/mixup2_yolov8-C2f-ODConv_100_32/weights/best.pt')
#     model.val(data='datasets/data.yaml',
#               # split='val',
#               split='test', # if you need to cal coco metrice with test
#               imgsz=640,
#               batch=16,
#               # rect=False,
#               save_json=True, # if you need to cal coco metrice
#               project='runs/val_coco',
#               name='yolov8-C2f-ODConv',
#               )