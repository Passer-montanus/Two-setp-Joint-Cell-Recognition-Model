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
    parser.add_argument(
       '--img_folder', default='D:/yolov8/runs/detect/C2f-OREPA/cutpictures/normal_1', help='包含待识别图片的文件夹路径')
    # parser.add_argument(
    #     '--img_folder', default='datas/MutiTest', help='包含待识别图片的文件夹路径')
    parser.add_argument(
        '--config', default='models/vision_transformer/vit_base_p16_224.py', help='配置文件路径')
    # parser.add_argument(
    # '--config', default='models/res2net/res2net50_w26_s8.py', help='配置文件路径')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='数据集类别映射文件路径')
    parser.add_argument(
        '--device', default='cpu', help='用于推断的设备')
    parser.add_argument(
        '--save-dir',default='results_5_8',
        help='保存预测结果图片的基础目录，默认为 results。')
    args = parser.parse_args()

    classes_names, label_names = get_info(args.classes_map)
    # 从配置文件和检查点文件构建模型
    model_cfg, train_pipeline, val_pipeline, data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    # 确定新的结果目录名
    result_dir = os.path.join(args.save_dir, f'result{len([d for d in os.listdir(args.save_dir) if d.startswith("result")]) + 1}')
    os.makedirs(result_dir)
    print(f"Saving results to: {result_dir}")

    # 处理文件夹中的每张图片
    img_files = [f for f in os.listdir(args.img_folder) if os.path.isfile(os.path.join(args.img_folder, f))]
    for img_file in img_files:
        img_path = os.path.join(args.img_folder, img_file)
        result = inference_model(model, img_path, val_pipeline, classes_names, label_names)
        print(result)
        # 获取分类结果和保存路径
        predicted_class = result['pred_class']
        save_dir = os.path.join(result_dir, predicted_class)
        
        # 如果分类子目录不存在，则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 保存分类结果图片
        save_path = os.path.join(save_dir, f'result_{os.path.splitext(img_file)[0]}_{result["pred_class"]}.jpg')
        # save_path = os.path.join(save_dir, img_file)
        img = Image.open(img_path)
        img.save(save_path)
        # 确认文件是否成功保存
        if os.path.exists(save_path):
            print("Image saved successfully!")
        else:
            print("Failed to save image.")       

        # show_result_pyplot(model, img_path, result, out_file=save_path) ## 可视化

if __name__ == '__main__':
    main()
