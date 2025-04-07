细胞分类模型
===========================

## 简介
本项目提供基于深度学习的细胞分类模型，支持以下骨干网络：
- DaViT
- MobileViT
- Res2Net
- VGG
- Vision Transformer

通过 `tools/single_test.py` 和 `tools/muti_test.py`，可以对经过识别分割处理的正常细胞图像进行分类。

---

### 单张图片分类
下载所需的预训练权重至 `datas` 文件夹下，并运行以下命令：
```bash
python tools/single_test.py datas/cell_image.png models/vgg/vgg16.py --classes-map datas/classes_map.txt
```

### 批量图片分类
将所有待分类的细胞图像放入同一文件夹中，并运行以下命令：
```bash
python tools/muti_test.py datas/cell_images/ models/res2net/res2net50.py --classes-map datas/classes_map.txt
```

---

## 支持的模型与预训练权重

| 模型 | 权重 |
| :---: | :---: |
| **DaViT** | [DaViT-S](https://download.openmmlab.com/mmclassification/v0/davit/davit-small_3rdparty_in1k_20221116-51a849a6.pth)|
| **MobileViT** | [MobileViT-Small](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth) |
| **Res2Net** | [Res2Net-50-26w-8s](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth)|
| **VGG** | [VGG-16](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth)<br>[VGG-19](https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth) |
| **Vision Transformer** | [ViT-Base-P16-224](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth)|

---

## 参考
```
@repo{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```
