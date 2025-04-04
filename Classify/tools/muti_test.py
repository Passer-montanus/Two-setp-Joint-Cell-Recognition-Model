from argparse import ArgumentParser
import os
import sys
sys.path.insert(0, os.getcwd())
import torch
import time  # 导入time模块以获取时间戳
from PIL import Image
from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    parser = ArgumentParser()
    # parser.add_argument('img_folder', help='包含待识别图片的文件夹路径')
    parser.add_argument(
         '--img_folder', default='D:/yolov8/runs/detect_add/Paper/cutpictures/normal_1', help='包含待识别图片的文件夹路径')
    # parser.add_argument(
    #     '--img_folder', default='D:/yolov8/runs/detect/C2f_ODConv/cutpictures/normal_1', help='包含待识别图片的文件夹路径')
    # parser.add_argument('config', help='配置文件路径')
    parser.add_argument(
        '--config', default='models/res2net/res2net50_w26_s8.py', help='配置文件路径')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='数据集类别映射文件路径')
    parser.add_argument(
        '--device', default='cpu', help='用于推断的设备')
    parser.add_argument(
        '--save-dir',default='results_add',
        help='保存预测结果图片的目录，默认不保存。')
    args = parser.parse_args()

    classes_names, label_names = get_info(args.classes_map)
    # 从配置文件和检查点文件构建模型
    model_cfg, train_pipeline, val_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    # 如果指定了保存目录且目录不存在，则创建保存目录
    # if args.save_dir and not os.path.exists(args.save_dir):
    #     os.makedirs(args.save_dir)
    
    # 创建新的保存目录
    save_dir = create_new_save_dir(args.save_dir)
    print(f"Saving results to: {save_dir}")
    # 处理文件夹中的每张图片
    img_files = [f for f in os.listdir(args.img_folder) if os.path.isfile(os.path.join(args.img_folder, f))]
    for img_file in img_files:
        img_path = os.path.join(args.img_folder, img_file)
        result = inference_model(model, img_path, val_pipeline, classes_names, label_names)
        print(result)
        if args.save_dir:
            save_path = os.path.join(save_dir, f'result_{os.path.splitext(img_file)[0]}_{result["pred_class"]}.jpg')
            # save_path = os.path.join(save_dir, f'result_{img_file}')
            # 获取分类结果
            predicted_class = result['pred_class']
            # 打印保存路径
            print(f"Saving result to: {save_path}_{predicted_class}.jpg")
            # 保存图片，并以分类结果命名
            img = Image.open(img_path)
            # img.save(f"{save_path}_{predicted_class}.jpg")
            img.save(save_path)
             # 确认文件是否成功保存
            if os.path.exists(f"{save_path}_{predicted_class}.jpg"):
                print("Image saved successfully!")
            else:
                print("Failed to save image.")
        else:
            save_path = None
        # show_result_pyplot(model, img_path, result, out_file=save_path)

def create_new_save_dir(base_dir):
    # 获取结果文件夹的名称模板
    dir_template = os.path.join(base_dir, 'result{}')
    # 初始化递增数字为1
    index = 1
    # 检查是否存在目录，如果存在则递增数字
    while os.path.exists(dir_template.format(index)):
        index += 1
    # 创建新的保存目录
    save_dir = dir_template.format(index)
    os.makedirs(save_dir)
    return save_dir

if __name__ == '__main__':
    main()
