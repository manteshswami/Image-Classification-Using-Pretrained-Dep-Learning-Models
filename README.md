# 🧠 *Image Classification Using Pretrained Deep Learning Models*
-----------------------------------------------------------
## 1. 🧠 Image Classification using VGG16 & VGG19
-----------------------------------------------------------
 *Pre-trained VGG16 and VGG19 models are used to classify real-world images using deep learning.*

### 🚀 Project Overview

This project uses CNN-based pretrained models trained on ImageNet to detect and classify objects from input images such as flowers, animals, etc.

### 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

### 📌 Step-by-Step Implementation
🔹 Step 1 — Import Libraries
<pre><code>
  import tensorflow as tf 
  from tensorflow.keras.applications import VGG16, VGG19 
  from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions 
  from tensorflow.keras.preprocessing.image import load_img, img_to_array 
  import numpy as np 
  import matplotlib.pyplot as plt </code></pre>
🔹 Step 2 — Load Pretrained Models
<pre><code> 
  model_1 = VGG16(weights="imagenet") 
  model_2 = VGG19(weights="imagenet") 
</code></pre>
🔹 Step 3 — Create Prediction Function
<pre><code> 
  def predict_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array) 
    prediction = model.predict(img_array) 
    return decode_predictions(prediction, top=5)[0] 
</code></pre>
🔹 Step 4 — Predict Using VGG16
<pre><code> 
  image_path = "/content/green_mamba.jpg" 
  predictions = predict_image(model_1, image_path) 
  for i, (img_id, label, score) in enumerate(predictions): 
  print(f"{i+1}. {label} - {score*100:.2f}%") 
</code></pre>
🔹 Step 5 — Display Result
<pre><code> 
  img = load_img(image_path) 
  plt.imshow(img) 
  plt.title(f"Prediction: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)") 
  plt.axis("off") 
  plt.show() 
</code></pre>
🔹 Step 6 — Predict Using VGG19
<pre><code> 
  image_path = "/content/ele.jpg" 
  predictions = predict_image(model_2, image_path) 
  for i, (img_id, label, score) in enumerate(predictions):
  print(f"{i+1}. {label} - {score*100:.2f}%") 
</code></pre>
🔹 Step 7 — Display Result
<code><pre>
  img = load_img(image_path) 
  plt.imshow(img) 
  plt.title(f"Prediction: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)") 
  plt.axis("off") 
  plt.show() 
</code></pre>

-----------------------------------------------------------
## 2. 🧠 Image Classification Using Pretrained InceptionV3
-----------------------------------------------------------
*This section demonstrates image classification using the InceptionV3 deep learning model trained on the ImageNet dataset.*

### 🚀 Model Overview

InceptionV3 is a powerful CNN architecture designed for high-accuracy image recognition.
Unlike VGG models, InceptionV3 requires 299 × 299 input images and uses factorized convolutions for better performance.

### 📌 Implementation Using InceptionV3
🔹 Step 1 — Import InceptionV3 Modules
<pre><code>
  from tensorflow.keras.applications import InceptionV3
  from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions     
  from tensorflow.keras.preprocessing.image import load_img, img_to_array 
  import numpy as np 
  import matplotlib.pyplot as plt 
</code></pre>
🔹 Step 2 — Load Pretrained InceptionV3 Model
<pre><code> inception_model = InceptionV3(weights="imagenet") </code></pre>
🔹 Step 3 — Define Prediction Function for InceptionV3
<pre><code> 
  def predict_inception(model, img_path):
    img = load_img(img_path, target_size=(299, 299)) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array) 
    predictions = model.predict(img_array) 
    return decode_predictions(predictions, top=5)[0]
</code></pre>
🔹 Step 4 — Predict Image Using InceptionV3
<pre><code> 
  image_path = "/content/dog.jpg" 
  predictions = predict_inception(inception_model, image_path) 
  for i, (img_id, label, score) in enumerate(predictions):
    print(f"{i+1}. {label} - {score*100:.2f}%") 
</code></pre>
🔹 Step 5 — Display InceptionV3 Result
<pre><code> 
  img = load_img(image_path)
  plt.imshow(img)
  plt.title(f"Prediction: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)") 
  plt.axis("off") 
  plt.show() 
</code></pre>

### 📌 Model Input Size Comparison

| Model       | Input Size  |
|-------------|-------------|
| VGG16       | 224 × 224   |
| VGG19       | 224 × 224   |
| InceptionV3 | 299 × 299   |




