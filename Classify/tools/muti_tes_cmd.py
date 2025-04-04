from argparse import ArgumentParser
import os
import sys
sys.path.insert(0, os.getcwd())
import torch

from utils.inference import inference_model, init_model, show_result_pyplot
from utils.train_utils import get_info, file2dict
from models.build import BuildNet

def main():
    parser = ArgumentParser()
    parser.add_argument('img_folder', nargs='?', default=None, help='包含待识别图片的文件夹路径')
    parser.add_argument('config', nargs='?', default=None, help='配置文件路径')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument(
        '--classes-map', default='datas/annotations.txt', help='数据集类别映射文件路径')
    parser.add_argument(
        '--device', default='cpu', help='用于推断的设备')
    parser.add_argument(
        '--save-dir',
        help='保存预测结果图片的目录，默认不保存。')
    args = parser.parse_args()

    # 如果没有通过命令行参数指定路径，则使用硬编码的路径
    img_folder = args.img_folder if args.img_folder is not None else 'datas/MutiTest'
    config_file = args.config if args.config is not None else 'models/efficientnetv2/efficientnetv2_m.py'
    weights_file = args.weights
    save_dir = args.save_dir

    classes_map = args.classes_map

    classes_names, label_names = get_info(classes_map)
    # 从配置文件和检查点文件构建模型
    model_cfg, train_pipeline, val_pipeline,data_cfg, lr_config, optimizer_cfg = file2dict(args.config)
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = BuildNet(model_cfg)
    model = init_model(model, data_cfg, device=device, mode='eval')

    # 如果指定了保存目录且目录不存在，则创建保存目录
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 处理文件夹中的每张图片
    img_files = [f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        result = inference_model(model, img_path, val_pipeline, classes_names, label_names)
        if save_dir:
            save_path = os.path.join(save_dir, f'result_{img_file}')
        else:
            save_path = None
        show_result_pyplot(model, img_path, result, out_file=save_path)

if __name__ == '__main__':
    main()
