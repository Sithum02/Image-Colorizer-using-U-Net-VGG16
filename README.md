# 🌈 Image Colorization using U-Net + VGG16 Encoder (TensorFlow)

This project implements a **deep learning–based automatic image colorization model**, which takes **grayscale images** as input and predicts the corresponding **color channels (a,b in Lab space)** using a **U-Net decoder** with a **pre-trained VGG16 encoder** as the feature extractor.

## 🚀 Features
- ✅ Converts grayscale images (L channel) to color (a,b) using deep learning  
- 🧠 Uses **VGG16 (ImageNet)** as a frozen encoder to leverage pre-trained semantic features  
- 🧱 Decoder follows a U-Net–style structure with skip connections for better spatial detail  
- 📊 Includes TensorFlow data pipelines for efficient preprocessing & batching  
- 👀 Live visualization callback to preview colorized outputs after each epoch

---

## 🧠 Model Architecture
- **Encoder:** Pre-trained VGG16 (ImageNet), frozen weights  
- **Bottleneck:** Deep semantic features (16×16 resolution)  
- **Decoder:** U-Net style with Conv2DTranspose + skip connections to reconstruct (a,b)  
- **Loss:** MSE between predicted and ground-truth (a,b) channels  
- **Input:** L channel `(256×256×1)`  
- **Output:** ab channels `(256×256×2)`

---

## 🧰 Tech Stack
- **Framework:** TensorFlow / Keras  
- **Languages:** Python  
- **Libraries:** OpenCV, NumPy, Matplotlib  

---

## 📂 Project Structure
