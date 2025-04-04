import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

# names = ['YOLOv8','YOLOv6','YOLOv5']
names = ['YOLOv5','YOLOv6','YOLOv8']
# colors = ['#2ca02c', '#1f77b4', '#d62728']
colors = ['#1f77b4', '#2ca02c', '#ff7f00']


plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['   metrics/precision(B)'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('Precision')
plt.legend()

plt.subplot(2, 2, 2)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['      metrics/recall(B)'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('Recall')
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['       metrics/mAP50(B)'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('mAP@0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['    metrics/mAP50-95(B)'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('mAP@0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('5_8poltpic/test/metrice_curve.png')
print(f'metrice_curve.png saved in {pwd}/metrice_curve.png')

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['         train/cls_loss'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('Train cls_loss')
plt.legend()

plt.subplot(1, 2, 2)
for i, name in enumerate(names):
    data = pd.read_csv(f'runs/train/{name}/results.csv')
    plt.plot(data['           val/cls_loss'], label=name, color=colors[i])
plt.xlabel('Epoch')
plt.title('Validation cls_loss')
plt.legend()

plt.tight_layout()
plt.savefig('5_8poltpic/test/loss_curve.png')
print(f'loss_curve.png save in {pwd}/loss_curve.png')