import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('D:/yolov8/weights/yolov8n_best.pt') # select your model.pt path
    model = YOLO('D:/AI腹水/11.22/训练结果/216_1/runs/train/mixup2_yolov8-C2f-OREPA_100_32/weights/best.pt') # select your model.pt path
    # model.predict(source='dataset_mixup/images/test',
    model.predict(source='D:/yolov8/Heatmap/dataset/org',
                project='D:/yolov8/Heatmap/runs',
                # name='yolov8n',
                name='Heatmap',
                save=True,
                # visualize=True # visualize model features maps
                save_txt=True
                )

# import warnings
# from ultralytics import YOLO

# warnings.filterwarnings('ignore')

# if __name__ == '__main__':
#     # 初始化YOLO模型
#     model = YOLO('D:/yolov8/weights/yolov8n_C2f_ODConv_best.pt')

#     # 使用 predict() 方法进行预测并保存标注文件
#     model.predict(
#         source='dataset_mixup/images/test',  # 指定输入图像的路径
#         project='runs/detect',  # 指定项目目录
#         name='C2f_ODConv',  # 指定保存结果的文件夹名称
#         save=True,  # 保存预测结果图片
#         save_txt=True  # 保存标注文件
#     )
