# Two-Step Joint Cell Recognition Model

Cytological examination of serous effusion is critical for diagnosing malignancies, yet it heavily relies on subjective interpretation by pathologists, leading to inconsistent accuracy and misdiagnosis, especially in regions with limited medical resources. To address this challenge, we propose a two-step deep learning framework to standardize and enhance the diagnostic process.

## Overview

This framework consists of two main modules:

1. **Recognize**: Detects abnormal cells using an improved YOLOv8 model integrated with the Online Convolutional Reparameterization (OREPA) module, achieving a sensitivity of 93.09%.
2. **Classify**: Classifies normal cells (lymphocytes, mesothelial cells, histiocytes, neutrophils) using the Dual Attention Vision Transformer (DaViT), achieving an accuracy of 98.74%.

By jointly deploying these models, our approach reduces missed diagnoses and provides granular insights into cell composition, offering a robust tool for rapid and objective cytopathological diagnosis.

## Modules

### [Recognize](./Recognize/Improvement_Effects.md)
The **Recognize** module focuses on detecting abnormal cells in cytological images. It leverages the improved YOLOv8 model with the OREPA module to enhance feature extraction and spatial information utilization. This module is particularly effective in identifying atypical cells with high sensitivity, making it suitable for resource-constrained environments.

- **Key Features**:
  - Enhanced feature extraction with orthogonal representation and prior attention mechanisms.
  - Multi-scale information capture for precise localization.
  - Efficient deployment with re-parameterization techniques.

For more details, visit the [Recognize module page](./Recognize/Improvement_Effects.md).

---

### [Classify](./Classify/README.md)
The **Classify** module is designed to categorize normal cells into specific types, including lymphocytes, mesothelial cells, histiocytes, and neutrophils. It employs the Dual Attention Vision Transformer (DaViT) to achieve high classification accuracy, ensuring reliable and detailed cell composition analysis.

- **Key Features**:
  - Dual attention mechanisms for robust feature learning.
  - High accuracy in distinguishing between normal cell types.
  - Adaptable to diverse cytological datasets.

For more details, visit the [Classify module page](./Classify/README.md).

---

## Conclusion

This two-step framework bridges the gap between AI-driven automation and clinical needs, particularly in resource-constrained settings. By combining advanced detection and classification techniques, it provides a standardized and efficient solution for cytopathological diagnosis.

For further instructions, navigate to the respective module pages.
