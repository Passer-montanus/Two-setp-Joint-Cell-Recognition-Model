import os
from PIL import Image
import shutil

"""
    根据detect的结果进行裁剪    
"""

def findSingleFile(path):
    # 创建 cutpictures 文件夹（先判断）
    cutp = os.path.join(path, "cutpictures") 
    # 判断文件夹是否存在
    if os.path.exists(cutp):
        # 如果文件夹存在，先删除再创建
        # 递归删除文件夹
        shutil.rmtree(cutp)
        os.makedirs(cutp)
    else:
        # 如果文件夹不存在，直接创建
        os.makedirs(cutp)

    org_path = os.path.join(path, "org")  # 图像文件所在的文件夹路径
    label_path = os.path.join(path, "labels")  # 标签文件所在的文件夹路径

    # 类别索引到类别名称的映射字典
    category_names = {
        0: "ab_0",
        1: "normal_1",
    }

    for filename in os.listdir(label_path):  # 遍历标签文件夹中的文件
        if filename.endswith(".txt"):  # 只处理以 .txt 结尾的文件
            # 获取图像文件名，去除扩展名 ".txt" 后缀
            img_filename = filename[:-4] + ".jpg"
            # img_filename = filename[:-4] + ".png"
            img_path = os.path.join(org_path, img_filename)  # 构建图像文件路径
            label_file_path = os.path.join(label_path, filename)  # 构建标签文件路径

            if os.path.exists(img_path):  # 如果图像文件存在
                # 读取图像文件
                img = Image.open(img_path)
                w, h = img.size
                
                with open(label_file_path, 'r+', encoding='utf-8') as f:
                    # 读取标签文件中的每一行
                    lines = f.readlines()
                    for index, line in enumerate(lines):
                        msg = line.split(" ")
                        category = int(msg[0])
                        x_center = float(msg[1])
                        y_center = float(msg[2])
                        width = float(msg[3])
                        height = float(msg[4])
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        
                        # 创建分类文件夹（按类别存放图片）
                        category_folder = os.path.join(cutp, category_names.get(category, "unknown"))
                        if not os.path.exists(category_folder):
                            os.makedirs(category_folder)

                        # 保存图片到对应的分类文件夹
                        img_roi = img.crop((x1, y1, x2, y2))
                        save_path = os.path.join(category_folder, "{}_{}_{}.jpg".format(img_filename[:-4], index, category))
                        img_roi.save(save_path)

    print("裁剪图片存放目录：", cutp)

def main():
    findSingleFile("runs/detect_add/Hospital")

if __name__ == '__main__':
    main()
