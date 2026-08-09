#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 05:37:45 2024

@author: excellus

PlantGuard v1 -- the original, deliberately minimal Streamlit front-end.

Kept as the reference implementation of the first version: plain Streamlit
widgets, no custom CSS, one prediction, no frills. Only correctness fixes have
been applied (current caching/image APIs, an upload guard, shared class list);
the layout and copy are unchanged on purpose.

Loads `newplantdis.keras`, the better of the two checkpoints (0.9767 validation
accuracy vs 0.8860 for trainedv3.keras).

Run with:  streamlit run plantpred.py

See `newplantpred.py` for the redesigned light "field report" UI and
`plantpredv2.py` for the dark-themed variant.
"""

import os

import numpy as np
import streamlit as st
import tensorflow as tf

from plant_classes import (
    ALLOWED_UPLOAD_TYPES,
    BEST_MODEL_PATH,
    CLASS_NAMES,
    HOME_IMAGE_PATH,
    IMAGE_SIZE,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(BEST_MODEL_PATH)


def model_prediction(test_image):
    model = load_model()
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=IMAGE_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_ind = np.argmax(prediction)

    return result_ind


st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home','About','Disease Recognition'])

if(app_mode=='Home'):
    st.header('PLANT DISEASE PREDICTION MODEL')
    if os.path.exists(HOME_IMAGE_PATH):
        st.image(HOME_IMAGE_PATH, use_container_width=True)
    else:
        st.info('Home image `home_page.jpg` not found — skipping.')
    st.markdown('''
    # Welcome to PlantGuard

    ## Protect Your Plants with AI-Powered Disease Detection

    PlantGuard helps you identify and manage plant diseases quickly and accurately using advanced artificial intelligence.

    ### How It Works

    1. Take a photo of your plant
    2. Upload the image to our app
    3. Receive instant disease identification

    ### Why Choose Us

    - **High Accuracy**: Our model reaches 97.7% accuracy on the validation split
    - **Fast Results**: Get instant diagnoses
    - **User-Friendly**: Simple interface designed for gardeners and farmers of all skill levels
    - **Broad Coverage**: 38 conditions across 14 crops

    ### Our Mission

    We're committed to empowering gardeners, farmers, and plant enthusiasts to maintain healthier plants and improve crop yields through accessible technology.

    Get started now and keep your plants thriving!

''')

elif(app_mode=='About'):
    st.header('About')
    st.markdown('''
    # About PlantGuard

    ## Our Mission

    At PlantGuard, we aim to revolutionize plant health management by putting advanced disease detection tools in the hands of gardeners, farmers, and plant enthusiasts worldwide.

    ## Our Technology

    ### AI-Powered Disease Recognition

    Our plant disease recognition model reaches 97.7% accuracy on the validation split, allowing users to:

    - Instantly identify plant diseases from photos
    - See which condition the model considers most likely

    Note: accuracy is measured on the validation split, which was also used to
    monitor training, so treat it as an optimistic estimate.

    ### Our Dataset

    The backbone of our accurate model is our robust dataset:

    - **Size**: Approximately 87,000 RGB images of crop leaves
    - **Content**: Includes both healthy and diseased plant leaves
    - **Diversity**: Categorized into 38 different classes
    - **Structure**:
      - 70,295 image training set
      - 17,572 image validation set
      - Additional test set of 33 images for prediction purposes
    - **Origin**: Recreated using offline augmentation from an original dataset [(available on GitHub)](https://github.com/spMohanty/PlantVillage-Dataset)

    This comprehensive dataset ensures our model can recognize a wide variety of plant diseases across numerous crop types, providing you with reliable and accurate diagnoses.

    ## Join Us in Nurturing a Greener World

    Whether you're a seasoned farmer, a hobby gardener, or just starting your plant journey, PlantGuard is here to help you grow healthier, happier plants.
''')

elif(app_mode=='Disease Recognition'):
    #st.header('Disease Recognition')
    st.markdown('''
    # Plant Disease Recognition
    Identify Plant Diseases in Seconds

    ### Supported Plants

    Our model can identify diseases in a wide range of crops, including:
    - Tomato
    - Potato
    - Corn
    - Apple
    - Grape
    - (and many more...)

    ### Tips for Best Results

    - Ensure good lighting when taking photos
    - Focus on the affected area
    - Include both healthy and diseased parts for comparison
    - Take multiple photos from different angles if needed


    Start protecting your plants today with PlantGuard's cutting-edge disease recognition technology!
''')
    test_image = st.file_uploader('Choose an Image', type=ALLOWED_UPLOAD_TYPES)
    if test_image is None:
        st.info('Please upload an image to proceed with prediction.')
    else:
        if st.button('Show Image'):
            st.image(test_image, use_container_width=True)
        if st.button('Predict'):
            result_index = model_prediction(test_image)
            st.success(f'Model is Predicting it is a {CLASS_NAMES[result_index]}')
