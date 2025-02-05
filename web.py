import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_extras.let_it_rain import rain 
from PIL import Image
import os

# Load Model Once (Prevents Reloading)
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        st.error("🚨 Model file not found! Please check the file path.")
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Function to Predict Disease
def model_prediction(test_image):
    if model is None:
        return None
    
    image = Image.open(test_image).convert("RGB")  # Convert to RGB (ensures compatibility)
    image = image.resize((128, 128))  # Resize
    input_arr = np.array(image) / 255.0  # Normalize pixel values (important!)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Streamlit Sidebar
st.sidebar.title("🌱 Plant Disease Prediction")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Display an Introductory Image
img = Image.open('potatoplant.jpg')
st.image(img, caption="Healthy Potato Plant", use_column_width=True)

# Home Page
if app_mode == 'Home':
    st.markdown(
        "<h1 style='text-align: center; color: green;'>🌿 Plant Disease Detection System for Sustainable Agriculture 🌿</h1>", 
        unsafe_allow_html=True
    )
    st.markdown("### How it Works:")
    st.write(
        """
        ✅ Upload an image of a potato leaf  
        ✅ Click "Predict" to check for diseases  
        ✅ AI model will identify if the plant is **healthy** or has **Early/Late Blight**  
        """
    )

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('📷 Upload an Image for Disease Detection')

    test_image = st.file_uploader('Choose an Image:', type=['jpg', 'png', 'jpeg'])

    if test_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button('Predict'):
            if model is None:
                st.error("🚨 Model not loaded. Please check the model file.")
            else:
                with st.spinner("🔍 Analyzing... Please wait"):
                    result_index = model_prediction(test_image)
                    if result_index is None:
                        st.error("⚠️ Prediction failed due to model issue.")
                    else:
                        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                        prediction = class_name[result_index]

                        # Rain Effect for Fun Animation
                        rain( 
                            emoji="🍃",  
                            font_size=35,  
                            falling_speed=3,  
                            animation_length="infinite"
                        )

                        with col2:
                            if prediction == 'Potato___healthy':
                                st.success(f"✅ The plant is **Healthy!** 🍀")
                            else:
                                st.error(f"⚠️ The plant has **{prediction.replace('_', ' ')}**! 🚨")
