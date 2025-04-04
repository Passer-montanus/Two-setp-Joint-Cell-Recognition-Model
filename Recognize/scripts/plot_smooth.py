import matplotlib.pyplot as plt
import pandas as pd


def smooth_curve(points, factor=0.75):
    """使用指数移动平均对数据进行平滑处理。"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_PR():
    pr_csv_dict = {
        'C2f-OREPA': r'D:\yolov8\runs\test_plot\C2f-OREPA\PR_curve.csv',
        'YOLOv8n': r'D:\yolov8\runs\test_plot\yolov8n\PR_curve.csv',
        'C2f-ODConv': r'D:\yolov8\runs\test_plot\C2f-ODConv\PR_curve.csv',
        'RCSOSA': r'D:\yolov8\runs\test_plot\RCSOSA\PR_curve.csv',
        'SPPF-LSKA': r'D:\yolov8\runs\test_plot\SPPF-LSKA\PR_curve.csv',
        'C2f-DBB': r'D:\yolov8\runs\test_plot\C2f-DBB\PR_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        y = pd.read_csv(res_path, usecols=[2]).values.ravel()
        y_smooth = smooth_curve(y)  # 对y值进行平滑处理
        ax.plot(x, y_smooth, label=modelname, linewidth='2')  # 使用平滑后的y值绘图

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    fig.savefig("pr_smooth_compare.png", dpi=400)
    plt.show()

def plot_F1():
    f1_csv_dict = {
        'C2f-OREPA': r'D:\yolov8\runs\test_plot\C2f-OREPA\F1_curve.csv',
        'YOLOv8n': r'D:\yolov8\runs\test_plot\yolov8n\F1_curve.csv',
        'C2f-ODConv': r'D:\yolov8\runs\test_plot\C2f-ODConv\F1_curve.csv',
        'RCSOSA': r'D:\yolov8\runs\test_plot\RCSOSA\F1_curve.csv',
        'SPPF-LSKA': r'D:\yolov8\runs\test_plot\SPPF-LSKA\F1_curve.csv',
        'C2f-DBB': r'D:\yolov8\runs\test_plot\C2f-DBB\F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        y = pd.read_csv(res_path, usecols=[2]).values.ravel()
        y_smooth = smooth_curve(y)  # 对y值进行平滑处理
        ax.plot(x, y_smooth, label=modelname, linewidth='2')  # 使用平滑后的y值绘图

    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()
    fig.savefig("F1_smooth_compare.png", dpi=400)
    plt.show()

if __name__ == '__main__':
    plot_PR()
    plot_F1()