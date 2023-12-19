import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO

  
# Streamlit uygulamasının arka plan rengini ayarla
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f4f4;  /* Açık gri renk */
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit başlıklarını ve görselleri ekle
st.markdown('<p style="background-color: #3366cc; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">🏠 E-Commerce (Amazon) Product Recognation 🏠</p>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #3366cc; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📌 Products 📌</p>', unsafe_allow_html=True)
st.image("resim2.jpeg", use_column_width=True)
# Kullanıcıdan resim yükleme yöntemini seçmesini iste
st.sidebar.title("Image Upload Method")
upload_method = st.sidebar.radio("Select an Upload Method:", ["From Computer", "From Internet"])

uploaded_image = None  # Kullanıcının yüklediği resmi saklamak için

if upload_method == "From Computer":
    uploaded_image = st.file_uploader("Please upload a product image:", type=["jpg", "png", "jpeg"])
elif upload_method == "From Internet":
    st.write("Please Enter an URL:")
    image_url = st.text_input("Image URL")

    
# Model seçimi
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio("Select an Model:", ["InceptionV3", "MobilNet",  "VGG16"])          
        
# Resmi yükle ve tahmin et butonları
if uploaded_image is not None or (upload_method == "From Internet" and image_url):
    st.markdown('<p style="background-color: #3366cc; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📸 Seçtiğiniz Resim 📸</p>', unsafe_allow_html=True)
    if uploaded_image is not None:
        st.image(uploaded_image, caption='', use_column_width=True)
    elif upload_method == "From Internet" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='', use_column_width=True)
        except Exception as e:
            st.error("Please Enter a Valid URL")


# Model bilgisi düğmesi
if st.sidebar.button("About Model"):
    st.markdown(f'<p style="background-color: #3366cc; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📜 {selected_model} Modeli Hakkında 📜</p>', unsafe_allow_html=True)
    if selected_model == "InceptionV3":
        st.write("InceptionV3 is an advanced model developed by Google for high-accuracy visual classification. It uses multi-path convolution blocks to learn multi-dimensional features.")
    elif selected_model == "MobilNet":
        st.write("MobileNet is a lightweight and efficient model optimised for mobile devices. It can perform fast operations with a small number of parameters.")
    elif selected_model == "VGG16":  # VGG16 için bilgi
        st.write("VGG16 is a deep convolutional neural network architecture known for its simplicity and deep layers, widely used in image recognition tasks.")
    
# Tahmin yap butonu
if st.button("Predict"):
    if upload_method == "From Computer" and uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif upload_method == "From Internet" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("Please Enter a Valid URL")
            
            
    # Kullanıcının seçtiği modele göre modeli yükle
    if selected_model == "InceptionV3":
        model_path = 'InceptionV3.h5'
    elif selected_model == "MobilNet":
        model_path = 'MobilNet.h5'
    elif selected_model == "VGG16":
        model_path = 'VGG16.h5'


    # Seçilen modeli yükle
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'tf': tf}, safe_mode=False)

    # Resmi model için hazırla ve tahmin yap
    if 'image' in locals():
        image = image.resize((224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        #image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # Tahmin yap
        prediction = model.predict(image)

        # Tahmin sonuçlarını göster
        class_names = ["Area Rugs", "Coffee Cups & Mugs", "Paints", "Yarn"]  # Modelin tahmin sınıfları
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        st.markdown(f'<p style="background-color: #3366cc; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📸Prdiction of Model 📸</p>', unsafe_allow_html=True)

        st.write(f"Prediction Result: {predicted_class}")
        st.write(f"Prediction Confidence: {confidence:.2f}")
        
        st.markdown('<p style="background-color: #3366cc; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Prediction Probs  📊</p>', unsafe_allow_html=True)
        # Tahmin olasılıklarını bir grafikte göster
        prediction_df = pd.DataFrame({'Categories': class_names, 'Prob': prediction[0]})
        st.bar_chart(prediction_df.set_index('Categories'))
