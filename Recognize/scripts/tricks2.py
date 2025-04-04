import os
import shutil

# 自定义筛选条件
# error_count尽可能多  missing_count==0
def custom_filter(right_count, missing_count, error_count, total_count):
    # 条件1：确保正确检测的数量超过总检测数量的50%
    condition1 = right_count > total_count * 0.5
    # 条件2：确保错误检测的数量为0
    condition2 = error_count >= total_count * 0.1
    # 条件3：确保缺失检测的数量不超过总检测数量的10%
    condition3 = missing_count == 0


    # 返回筛选结果
    return condition1 and condition2 and condition3

# result_file_path = 'change_classes/save_txt/result.txt'
# img_folder_path = 'change_classes/save_pic'
# good_performance_folder_path = 'change_classes/save_pic/good'

result_file_path = 'vis/C2f-OREPA/result.txt'
img_folder_path = 'vis/C2f-OREPA/img'
good_performance_folder_path = 'vis/C2f-OREPA/good'

# 创建goodperformance文件夹
os.makedirs(good_performance_folder_path, exist_ok=True)

# 读取result.txt文件并解析结果
with open(result_file_path, 'r') as file:
    lines = file.readlines()

# 解析result.txt中的结果
for line in lines:
    if line.startswith('name:'):
        parts = line.split()
        name = parts[0].split(':')[-1]
        right_count = int(parts[1].split(':')[-1])
        missing_count = int(parts[2].split(':')[-1])
        error_count = int(parts[3].split(':')[-1])
        total_count = right_count + missing_count + error_count

        # 自定义筛选条件
        if custom_filter(right_count, missing_count, error_count, total_count):
            # 复制符合条件的图片到goodperformance文件夹中
            img_path = os.path.join(img_folder_path, name + '.jpg')
            if os.path.exists(img_path):
                shutil.copy(img_path, good_performance_folder_path)
                print(f"Image '{name}.jpg' has been copied to 'goodperformance' folder.")
            else:
                print(f"Image '{name}.jpg' does not exist in 'img' folder.")
