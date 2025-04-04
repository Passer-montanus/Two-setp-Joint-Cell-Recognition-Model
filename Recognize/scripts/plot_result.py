import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

names = ['YOLOv8','YOLOv6','YOLOv5']
colors = ['red', 'blue', 'green']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['   metrics/precision(B)'], label=i)
plt.xlabel('epoch')
# plt.title('precision')
plt.title('Precision')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['      metrics/recall(B)'], label=i)
plt.xlabel('epoch')
# plt.title('recall')
plt.title('Recall')
plt.legend(loc='lower right')

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['       metrics/mAP50(B)'], label=i)
plt.xlabel('epoch')
# plt.title('mAP_0.5')
plt.title('mAP@0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['    metrics/mAP50-95(B)'], label=i)
plt.xlabel('epoch')
# plt.title('mAP_0.5:0.95')
plt.title('mAP@0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('5_8poltpic/metrice_curve.png')
print(f'metrice_curve.png save in {pwd}/metrice_curve.png')

plt.figure(figsize=(10, 5))

# plt.subplot(2, 3, 1)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     plt.plot(data['         train/box_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('train/box_loss')
# plt.legend()

# plt.subplot(2, 3, 2)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     plt.plot(data['         train/dfl_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('train/dfl_loss')
# plt.legend()

plt.subplot(1,2,1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['         train/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/cls_loss')
plt.legend()

# plt.subplot(2, 3, 4)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     plt.plot(data['           val/box_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('val/box_loss')
# plt.legend()

# plt.subplot(2, 3, 5)
# for i in names:
#     data = pd.read_csv(f'runs/train/{i}/results.csv')
#     plt.plot(data['           val/dfl_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('val/dfl_loss')
# plt.legend()

plt.subplot(1,2,2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    plt.plot(data['           val/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/cls_loss')
plt.legend()

plt.tight_layout()
plt.savefig('5_8poltpic/loss_curve.png')
print(f'loss_curve.png save in {pwd}/loss_curve.png')