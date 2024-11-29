# 🌿 Plant Disease Detection

This is a *Streamlit-based web application* that uses a trained deep learning model to detect plant diseases from leaf images. The model identifies both the plant and its disease with high confidence.

---

## 🌟 Features

- *User-Friendly Interface*: Upload plant leaf images through a simple drag-and-drop interface.
- *Wide Disease Coverage*: Detects 38 plant diseases, including healthy plant states.
- *Interactive Analysis*: Provides confidence scores and disease-specific information.
- *Real-time Results*: Processes and predicts disease in seconds.

---

## 🛠 Tech Stack

- *Frontend*: Streamlit
- *Backend*: TensorFlow (Keras), NumPy
- *Data Preprocessing*: PIL (Pillow), TensorFlow Image Processing
- *Deployment*: Local or cloud-based platforms

---

## 📚 Dataset

The model was trained on a comprehensive dataset of labeled images for plant disease detection, containing multiple classes for various crops and diseases. The dataset includes crops like *Tomato, Grape, Corn, Apple*, and more.

---
## 🚀 Installation Guide
### 1. Clone the Repository
bash
git clone https://github.com/Chayan1729/Plant-Disease-Detection_.git
cd plant-disease-detection


### 2. Set Up a Virtual Environment
To isolate the project's dependencies, create and activate a virtual environment.

#### For Windows:
bash
python -m venv venv
venv\Scripts\activate

### 3. Install Dependencies
bash

pip install -r requirements.txt

### 4. Run the App
bash

streamlit run main.py



## 🌱 Supported Plants and Diseases

- *Tomato*: Late Blight, Early Blight, Yellow Leaf Curl Virus, and more  
- *Corn (Maize)*: Northern Leaf Blight, Common Rust, Gray Leaf Spot  
- *Apple*: Apple Scab, Black Rot, Cedar Apple Rust  
- *Strawberry*: Leaf Scorch  
- *Grape*: Black Rot, Esca (Black Measles), Isariopsis Leaf Spot  
- And many more...
