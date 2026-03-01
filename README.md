# pneumonia-detection-model
Deep learning model for classifying chest X-ray images into Pneumonia and Normal categories using Convolutional Neural Networks.

Dataset : https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data

## Model Architecture
Input → CConv2D → ReLU → MaxPool → Dense → Output

Training Details
Optimizer: Adam
Loss: Binary Crossentropy
Input size: 180×180

## How to Use Model
from tensorflow.keras.models import load_model
model = load_model("models/mymodel.keras")
