# Plant Disease Detection System for Sustainable Agriculture

A Deep Learning-based web application to detect and classify plant leaf diseases, promoting sustainable agriculture.

## Overview

This project uses a **Convolutional Neural Network (CNN)** to identify plant diseases from leaf images. The model predicts the type of disease or whether the plant is healthy, helping farmers and agricultural experts take timely action.

---

## 🛠️ Technologies Used

- **TensorFlow/Keras**: To build and train the CNN model for plant disease classification.
- **OpenCV**: For image preprocessing (resizing, color conversion, and normalization).
- **Streamlit**: To create an interactive web application for disease detection.
- **NumPy**: For numerical operations and array manipulations.

---

## 📋 Features

✅ Identify diseases in plants from uploaded images.<br>
✅ Supports multiple plants such as Apple, Corn, Potato, and Tomato.<br>
✅ Provides real-time, accurate disease classification.<br>
✅ User-friendly interface built with Streamlit.

---

## 📷 Model Architecture

- Input layer expects images of size **128x128x3**.
- The model predicts the disease class using a **Softmax activation**.
- Trained on a diverse dataset of plant leaf images for improved accuracy.

---

**Clone the repository:**

```bash
git clone https://github.com/Or4cle404/Plant-Disease-Detectiongit
cd plant-disease-detection
```

---

## 🖼️ Usage

1. Go to the **"Disease Recognition"** section of the app.
2. Upload an image of a plant leaf.
3. Click the **"Predict"** button.
4. View the disease classification and suggested actions.

---

## 📊 Class Labels

The model supports a wide range of plant diseases, including:

- Apple___Apple_scab
- Corn_(maize)___Common_rust
- Potato___Early_blight
- Tomato___Late_blight
- And many more!

---

## ⭐ Acknowledgments

Special thanks to open-source datasets and libraries that made this project possible!

---

✨ **Let's build a healthier, more sustainable future for agriculture!** ✨

