from xml.etree import ElementTree as ET
from PIL import Image
import os

"""
    根据XML文件 对训练集、验证集、测试集进行crop
    便于第二阶段训练训练

"""

def get_bndbox(xml_path):
    """
    返回xml文件中bndbox
    :param xml_path: xml文件绝对路径
    :return: obj_name_loc_list: [{"name": name, "bndbox": bndbox},...]
        name: 数据集类名
        bndbox: [xmin, ymin, xmax, ymax]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj_name_loc_list = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = bndbox.find("xmin").text
        xmax = bndbox.find("xmax").text
        ymin = bndbox.find("ymin").text
        ymax = bndbox.find("ymax").text
        if (int(xmax) - int(xmin)) == 0 or (int(ymax) - int(ymin)) == 0:
            continue
        box = [int(xmin), int(ymin), int(xmax), int(ymax)]
        name = obj.find("name").text
        project_dict = {"name": name, "bndbox": box}
        obj_name_loc_list.append(project_dict)
    return obj_name_loc_list

# 创建crop_image文件夹
# 把每张图片中标注的物体按照数据集类名保存到images对应目录下
# classes = ["meso", "histo", "ab", "lymph", "neutro"]
# for cls in classes:
#    if not os.path.exists(os.path.join("images\\crop_image", cls)):
#        os.mkdir(os.path.join("images\\crop_image", cls))

# classes = ["ab", "normal"]
classes = ["meso", "neutro","histo", "ab", "lymph"]
output_dir = "crop_images_5_8\\five_val"  # 修改为您想要保存的根目录

for cls in classes:
    cls_dir = os.path.join(output_dir, cls)
    if not os.path.exists(cls_dir):
        os.makedirs(cls_dir, exist_ok=True)
    
image_dir = r"D:/yolov8/change_classes/org_val" # 修改
xml_dir = r"D:/yolov8/change_classes/xml_val" # 修改

image_list = os.listdir(image_dir)
for img in image_list:
    img_prefix = img.split(".")[0]
    img_path = os.path.join(image_dir, img)
    xml_path = os.path.join(xml_dir, img_prefix + ".xml")
    obj_name_loc_list = get_bndbox(xml_path)
    if len(obj_name_loc_list) == 0:
        continue
    image = Image.open(img_path)
    i = 0
    for obj_dict in obj_name_loc_list:
        loc = obj_dict["bndbox"]
        cls_name = obj_dict["name"]
        image_crop = image.crop(loc)
        if cls_name in classes:         # 修改
            save_path = os.path.join("crop_images_5_8/five_val/{}".format(cls_name), img_prefix + "_" + str(i) + ".jpg")
            image_crop.save(save_path)
            i += 1
            print(i)