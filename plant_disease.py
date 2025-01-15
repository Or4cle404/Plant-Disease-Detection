import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

def model_predict(image_path):
    # Load the trained model
    model = tf.keras.models.load_model(r"D:\Jupyter\plant_disease_model.keras")
    
    # Read the image
    img = cv2.imread(image_path)
    H, W, C = 128, 128, 3  # Resize to 128x128 if your model expects this size
    img = cv2.resize(img, (H, W))  # Resize the image to the model's expected input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB (if it is BGR)
    img = np.array(img) / 255.0  # Normalize the image to [0, 1] range
    img = img.reshape(1, H, W, C)  # Reshape to (1, 128, 128, 3), as expected by the model

    # Predict using the model
    prediction = model.predict(img)  # Get prediction from model
    result_index = np.argmax(prediction, axis=-1)[0]  # Get the predicted class index

    return result_index

# Streamlit app code
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition"])

# Display an image for the homepage
img = Image.open(r"C:\Users\ASUS\Downloads\smart-agriculture-iot-with-hand-planting-tree-background_53876-124626.png")
st.image(img)

if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == "Disease Recognition":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an image:")

    if test_image is not None:
        # Save uploaded image to disk
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if st.button("Show image"):
        st.image(test_image, width=4, use_column_width=True)
    
    if st.button("Predict"):
        st.write("Our Prediction")

        try:
            # Get the predicted class index from the model
            result_index = model_predict(save_path)
            
            # List of class names
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                          'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                          'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                          'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                          'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                          'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                          'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            # Ensure the predicted index is within the bounds of the class list
            if result_index < len(class_name):
                st.success(f"Model is predicting it's a {class_name[result_index]}")
            else:
                st.error("Error: Predicted index is out of range. Please check the model's output.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
