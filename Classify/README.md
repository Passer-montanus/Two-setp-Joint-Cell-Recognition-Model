Cell Classification Models
===========================

## Introduction
This project provides deep learning-based cell classification models, supporting the following backbones:
- DaViT
- MobileViT
- Res2Net
- VGG
- Vision Transformer

You can classify normal cell images, which have been preprocessed through recognition and segmentation, using `tools/single_test.py` and `tools/muti_test.py`.

---

### Single Image Classification
Download the required pretrained weights to the `datas` folder and run the following command:
```bash
python tools/single_test.py datas/cell_image.png models/vgg/vgg16.py --classes-map datas/classes_map.txt
```

### Batch Image Classification
Place all the cell images to be classified in the same folder and run the following command:
```bash
python tools/muti_test.py datas/cell_images/ models/res2net/res2net50.py --classes-map datas/classes_map.txt
```

---

## Supported Models and Pretrained Weights

| Model | Weights |
| :---: | :---: |
| **DaViT** | [DaViT-S](https://download.openmmlab.com/mmclassification/v0/davit/davit-small_3rdparty_in1k_20221116-51a849a6.pth)|
| **MobileViT** | [MobileViT-Small](https://download.openmmlab.com/mmclassification/v0/mobilevit/mobilevit-small_3rdparty_in1k_20221018-cb4f741c.pth) |
| **Res2Net** | [Res2Net-50-26w-8s](https://download.openmmlab.com/mmclassification/v0/res2net/res2net50-w26-s8_3rdparty_8xb32_in1k_20210927-f547a94b.pth)|
| **VGG** | [VGG-16](https://download.openmmlab.com/mmclassification/v0/vgg/vgg16_batch256_imagenet_20210208-db26f1a5.pth)|
| **Vision Transformer** | [ViT-Base-P16-224](https://download.openmmlab.com/mmclassification/v0/vit/pretrain/vit-base-p16_3rdparty_pt-64xb64_in1k-224_20210928-02284250.pth)|

---

## Reference
```
@repo{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```
