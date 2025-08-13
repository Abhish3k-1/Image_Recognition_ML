# 🐱🐶 Image Recognition using MobileNet

This repository contains an **image classification model** based on **MobileNetV2** for detecting whether an image contains a **cat** 🐱 or a **dog** 🐶.  
The pre-trained model file `mobilenet_cats_cats_.h5` is included via Git LFS for quick loading without re-training.

---
## ⚙️ Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/Abhish3k-1/Image_Recognition_ML.git
   cd Image_Recognition_ML


## Install Python dependencies

pip install tensorflow keras numpy matplotlib



## Enable Git LFS (Large File Storage)
## This project uses Git LFS to store the .h5 model file.

git lfs install
git lfs pull

## 🚀 Usage

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('mobilenet_cats_cats_.h5')

# Load and preprocess an image
img_path = 'path_to_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")
