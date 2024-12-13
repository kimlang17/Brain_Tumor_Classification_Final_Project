import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

# Hide deprecation warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="Brain.jpg",
    initial_sidebar_state='auto'
)

# Hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

# Load the model function
# @st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cnn_model_ML_50_layer4.h5')
    return model

# Load the model
with st.spinner('Model is being loaded..'):
    model = load_model()

# Define a function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    size = (150, 150)  # Changed to match the model's expected input size
    image = Image.open(uploaded_file)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0  # Rescale the image
    return img_reshape

# Define the function to get the class label from prediction
def prediction_cls(prediction, class_names):
    return class_names[np.argmax(prediction)]

# Sidebar contents
with st.sidebar:
    st.markdown("""
    <h1 style='text-align: center;'>Brain Tumor Detection</h1>""", unsafe_allow_html=True)
    st.image('Brain.jpg')

# Justified subheader using custom HTML and CSS
    st.markdown("""
    <h3 style='text-align: justify;'>
        Accurately classify the Brain as healthy or Tumor. This helps a doctor to classify it faster and easier.
    </h3>""", unsafe_allow_html=True)

# Main content
st.markdown("""
    <h2 style='text-align: center;'>Brain Tumor Detection</h2>""", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png"])

if file is None:
    st.markdown("""
    <h4 style='text-align: center;'>Please upload an image MRI of a brain.</h4>""", unsafe_allow_html=True)
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    processed_image = preprocess_image(file)
    predictions = model.predict(processed_image)

    class_names = ['Healthy', 'Brain Tumor']
    predicted_class = prediction_cls(predictions, class_names)

    result_message = f"Detected Result: {predicted_class}"

    if predicted_class == 'Healthy':
        st.balloons()
        st.sidebar.success(result_message)
        st.markdown("## Congratulation, Healthy Brain!!!")
    elif predicted_class == 'Brain Tumor':
        st.sidebar.warning(result_message)
        st.markdown("## Detected Tumor!")
        st.markdown("## Please get immediate treatment!")
st.set_option('deprecation.showfileUploaderEncoding', False)