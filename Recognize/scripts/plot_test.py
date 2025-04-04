import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pwd = os.getcwd()

names = ['YOLOv8','YOLOv6','YOLOv5']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(12, 8))

for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['   metrics/precision(B)'], label=f'{name} - Precision', color=colors[i], linestyle='-', linewidth=2)
    plt.plot(data['      metrics/recall(B)'], label=f'{name} - Recall', color=colors[i], linestyle='--', linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Precision and Recall for YOLO Models', fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.savefig('5_8poltpic/testprecision_recall_curve.png')
plt.show()