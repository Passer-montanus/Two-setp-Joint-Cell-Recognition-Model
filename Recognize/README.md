## <div align="center">Performance Validation</div>

Below are the instructions to validate the recognition performance of normal and abnormal cells:

#### CLI

Run the following command in the command line to load the model weights and perform detection:

```bash
yolo detect model=weights/normal_abnormal_cells.pt source='path/to/your/image_or_video'
```

#### Python

Run the following code in a Python environment:

```python
from ultralytics import YOLO

# Load the model
model = YOLO("weights/normal_abnormal_cells.pt")

# Perform detection
results = model("path/to/your/image_or_video")
results.show()  # Display the detection results
```

## <div align="center">Directory Structure</div>

Below is the structure of the `Recognize` folder:

```
Recognize/
├── README.md                # Documentation
├── weights/                 # Directory for model weights
│   └── normal_abnormal_cells.pt  # Model weights for normal and abnormal cell recognition
├── dataset/                 # Directory for datasets
│   ├── VOCdevkit/           # Test images or videos
│   ├── split_data.py        # Script to split data into datasets
│   └── ...                  # Other scripts for annotation conversion
├── scripts/                 # Processing scripts
│   ├── detect.py            # Script for detection
│   ├── crop.py              # Script to crop images for the second step based on detection results
│   └── ...                  # Other utility scripts
└── results/                 # Directory for detection results
    └── output/              # Output images or videos
```

Please organize your project according to the above structure for better usage and management.
