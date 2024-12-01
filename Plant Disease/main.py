import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
MODEL_PATH = "E:/Plant Disease/bestmodel1.h5"  # Update this path if needed
model = load_model(MODEL_PATH)

# Define the 38 classes from your dataset
DISEASES = [
    'Tomato___Late_blight', 'Tomato___healthy', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Potato___healthy',
    'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Early_blight',
    'Tomato___Septoria_leaf_spot', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Strawberry___Leaf_scorch', 'Peach___healthy', 'Apple___Apple_scab',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Bacterial_spot',
    'Apple___Black_rot', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew', 'Peach___Bacterial_spot',
    'Apple___Cedar_apple_rust', 'Tomato___Target_Spot',
    'Pepper,_bell___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Potato___Late_blight', 'Tomato___Tomato_mosaic_virus',
    'Strawberry___healthy', 'Apple___healthy', 'Grape___Black_rot',
    'Potato___Early_blight', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Common_rust_', 'Grape___Esca_(Black_Measles)',
    'Raspberry___healthy', 'Tomato___Leaf_Mold',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Pepper,_bell___Bacterial_spot', 'Corn_(maize)___healthy'
]


def load_image(image_file):
    """Load and return an image file."""
    img = Image.open(image_file)
    return img


def preprocess_image(image, target_size=(128, 128)):
    """
    Preprocess the image for prediction:
    - Resizes to (128, 128) to match the training input size.
    - Normalizes pixel values to [0, 1].
    """
    image = image.resize(target_size)  # Resize the image to (128, 128)
    image = img_to_array(image)       # Convert to NumPy array
    image = image / 255.0             # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def predict_disease(image):
    """
    Predict the disease using the loaded model.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image, target_size=(128, 128))

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Find the most confident class
    predicted_class_idx = np.argmax(predictions[0])  # Get the class index
    predicted_class = DISEASES[predicted_class_idx]  # Map to class name
    confidence = predictions[0][predicted_class_idx]  # Confidence score

    return predicted_class, confidence



# Set page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Main app header
st.title("🌿 Plant Disease Detection")
st.markdown("""
    Upload a photo of your plant's leaves to detect potential diseases.
    Our AI model will analyze the image and provide potential disease diagnoses.
""")

# Sidebar with information
with st.sidebar:
    st.header("About")
    st.markdown("""
        This application helps identify common plant diseases using machine learning.

        **Supported Plants:**
        - Tomato, Grape, Orange, Soybean, Squash, Potato
        - Corn, Strawberry, Peach, Apple, Blueberry, Cherry, Pepper, Raspberry

        **How to use:**
        1. Upload a clear image of the affected plant leaf
        2. Wait for the analysis
        3. Review the results
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an image of a plant leaf", type=['jpg', 'jpeg', 'png'])

# Create two columns for layout
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Display the uploaded image
    with col1:
        st.subheader("Uploaded Image")
        image = load_image(uploaded_file)
        st.image(image, caption="Uploaded Plant Image", use_container_width=True)


    # Make prediction and display results
    with col2:
        st.subheader("Analysis Results")
        with st.spinner("Analyzing image..."):
            # Get prediction
            predicted_class, confidence = predict_disease(image)

            # Split plant and disease
            plant, disease = predicted_class.split('___')

            # Display the results
            st.markdown(f"### **Plant**: {plant}")
            if disease == "healthy":
                st.markdown("### 🌱 The plant appears **healthy**!")
            else:
                st.markdown(f"### 🌡️ Detected Disease: **{disease}**")

            # Show confidence
            confidence_percentage = f"{confidence * 100:.2f}%"
            st.progress(float(confidence))
            st.write(f"Confidence: {confidence_percentage}")
            st.markdown("---")

