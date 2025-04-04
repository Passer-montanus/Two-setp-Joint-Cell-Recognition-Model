import argparse
import os
import cv2
import numpy as np
import shutil
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="图像处理脚本")
    parser.add_argument('--mode', type=str, default='pre', choices=['pre', 'manual', 'compare'], help='运行模式: pre, manual, compare')
    parser.add_argument('--threshould',type=float,default=0.45,help='iou计算阈值')
    parser.add_argument('--desired_width', type=int, default=256, help='目标图像宽度')
    parser.add_argument('--desired_height', type=int, default=256, help='目标图像高度')
    parser.add_argument('--img_path', type=str, default='D:/yolov8/detect_best/org_pic', help='原始图像路径')
    parser.add_argument('--label_path', type=str, default='D:/yolov8/detect_best/manual', help='标签路径')
    parser.add_argument('--predict_path', type=str, default='D:/yolov8/detect_best/pre', help='预测结果路径')
    parser.add_argument('--save_path', type=str, default='D:/yolov8/detect_best/save_pic', help='保存图像的路径')
    parser.add_argument('--result_file_path', type=str, default='D:/yolov8/detect_best/save_txt/result.txt', help='结果文件路径')
    args = parser.parse_args()
    return args


# 使用parse_args函数获取命令行参数
args = parse_args()

# 根据命令行参数设置变量
postfix = 'jpg'
classes = ['ab','normal']
color_map = {'ab': (0, 0, 255), 'normal': (0, 255, 0)}  # 类别对应的颜色BGR
detect_color, missing_color, error_color  = (0, 255, 0), (0, 0, 255), (255, 0, 0)

# 指定args变量
mode = args.mode
iou_threshold = args.threshould
desired_width = args.desired_width
desired_height = args.desired_height
img_path = args.img_path
label_path = args.label_path
predict_path = args.predict_path
save_path_base = args.save_path  # 基础保存路径
result_file_path_base = args.result_file_path  # 基础结果文件路径



# 动态设置具体的保存路径和结果文件路径
save_path = os.path.join(save_path_base, mode)
result_file_path = os.path.join(save_path, 'result.txt')

# 接下来的代码保持不变，使用上面定义的变量...

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

def xywh2xyxy(box):
    """转换坐标格式从中心宽高到左上右下"""
    box[:, 0] = box[:, 0] - box[:, 2] / 2
    box[:, 1] = box[:, 1] - box[:, 3] / 2
    box[:, 2] = box[:, 0] + box[:, 2]
    box[:, 3] = box[:, 1] + box[:, 3]
    return box

def iou(box1, box2):
    """计算两个框的IoU"""
    x11, y11, x12, y12 = np.split(box1, 4, axis=1)
    x21, y21, x22, y22 = np.split(box2, 4, axis=1)

    xa = np.maximum(x11, np.transpose(x21))
    xb = np.minimum(x12, np.transpose(x22))
    ya = np.maximum(y11, np.transpose(y21))
    yb = np.minimum(y12, np.transpose(y22))

    inter_area = np.maximum(0, xb - xa + 1) * np.maximum(0, yb - ya + 1)
    box1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2_area = (x22 - x21 + 1) * (y22 - y21 + 1)
    union_area = box1_area + np.transpose(box2_area) - inter_area

    iou =  inter_area / union_area
    return iou

def draw_box(img, box, color, class_name=None, iou_value=None, 
             thickness=5, font_scale=1.2):
    """
    绘制识别框及其信息
    img: 图像
    box: 框坐标
    color: 颜色
    class_name: 类别名称
    iou_value: IoU值
    thickness: 框的粗细
    font_scale: 字体大小
    """
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=thickness)

    if class_name is not None:
        text = class_name
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), color, cv2.FILLED)
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=thickness)

    if iou_value is not None:
        text = f'IoU: {iou_value:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y2), (x1 + text_width, y2 + text_height + 10), color, cv2.FILLED)
        cv2.putText(img, text, (x1, y2 + text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=thickness)

    return img

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

# 初始化类别计数字典
category_counts = {cls: 0 for cls in classes}
# 初始化对比结果计数
all_right_num, all_missing_num, all_error_num = 0, 0, 0

with open(result_file_path, 'w') as f_w:
    for path in tqdm.tqdm(os.listdir(label_path)):
        image_path = f'{img_path}/{path[:-4]}.{postfix}'
        if not os.path.exists(image_path):
            print(f'image:{img_path}/{path[:-4]}.{postfix} not found.', file=f_w)
        image = cv2.imread(image_path)
        h, w = image.shape[:2]

        try:
            with open(f'{predict_path}/{path}') as f:
                preds = np.array([np.array(x.strip().split(), dtype=np.float32) for x in f.readlines()])
                preds[:, 1:5] = xywh2xyxy(preds[:, 1:5])
                preds[:, [1, 3]] *= w
                preds[:, [2, 4]] *= h
        except FileNotFoundError:
            preds = np.array([])

        try:
            with open(f'{label_path}/{path}') as f:
                labels = np.array([np.array(x.strip().split(), dtype=np.float32) for x in f.readlines()])
                labels[:, 1:] = xywh2xyxy(labels[:, 1:])
                labels[:, [1, 3]] *= w
                labels[:, [2, 4]] *= h
        except FileNotFoundError:
            labels = np.array([])

        if mode == "compare":
            right_num, missing_num, error_num = 0, 0, 0
            for i in range(len(labels)):
                if len(preds) > 0:
                    ious = iou(labels[i:i+1, 1:], preds[:, 1:5]).flatten()  # 确保ious是一维数组
                    max_iou_index = np.argmax(ious)
                    max_iou = ious[max_iou_index]
                    if max_iou >= iou_threshold:
                        # 这里需要确保比较的双方都是单个值
                        if labels[i, 0] == preds[max_iou_index, 0]:
                            draw_box(image, preds[max_iou_index, 1:5], color_map[classes[int(preds[max_iou_index, 0])]], classes[int(preds[max_iou_index, 0])], max_iou)
                            preds = np.delete(preds, max_iou_index, axis=0)  # 从preds中移除已匹配的预测
                            right_num += 1
                        else:
                            draw_box(image, labels[i, 1:], missing_color, classes[int(labels[i, 0])], 0)
                            missing_num += 1
                    else:
                        draw_box(image, labels[i, 1:], missing_color, classes[int(labels[i, 0])], 0)
                        missing_num += 1
                else:
                    # 如果没有预测框，则所有的标签都被视为漏检
                    draw_box(image, labels[i, 1:], missing_color, classes[int(labels[i, 0])], 0)
                    missing_num += 1

            # 处理剩下的预测框为误检
            for pred in preds:
                draw_box(image, pred[1:5], error_color, classes[int(pred[0])])
                error_num += 1

            all_right_num += right_num
            all_missing_num += missing_num
            all_error_num += error_num
            print(f'name:{path[:-4]} right:{right_num} missing:{missing_num} error:{error_num}', file=f_w)

        elif mode == "pre":
            for pred in preds:
                class_name = classes[int(pred[0])]
                color = color_map.get(class_name, (255, 255, 255))
                draw_box(image, pred[1:5], color, class_name)
                category_counts[class_name] += 1
            pass

        elif mode == "manual":
            for label in labels:
                class_name = classes[int(label[0])]
                color = color_map.get(class_name, (255, 255, 255))
                draw_box(image, label[1:], color, class_name)
                category_counts[class_name] += 1
            pass

        # 自定义图像分辨率
        # image = cv2.resize(image, (desired_width, desired_height))

        save_filename = f'{path[:-4]}_{mode}.{postfix}'
        cv2.imwrite(os.path.join(save_path, save_filename), image)

    with open(result_file_path, 'a') as f:
        print("总计类别数量:")
        f.write("总计类别数量:\n")
        for cls, count in category_counts.items():
            print(f.write(f"{cls}: {count}\n"))
            f.write(f"{cls}: {count}\n")



    if mode == "compare":
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}', file=f_w)
        print(f'all_result: right:{all_right_num} missing:{all_missing_num} error:{all_error_num}')