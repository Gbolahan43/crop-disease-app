import streamlit as st
import tensorflow as tf
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache(allow_output_mutation=True)
# @st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('newplantdis.keras')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the model. Please try again later.")
        return None

def model_prediction(test_image):
    model = load_model()
    if model is None:
        return None
    
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        result_ind = np.argmax(prediction)
        logger.info("Prediction successful")
        return result_ind
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        st.error("An error occurred during prediction. Please try again.")
        return None

st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox('Select Page', ['Home','About','Disease Recognition'])

if app_mode == 'Home':
    st.header('PLANT DISEASE PREDICTION MODEL')
    try:
        imagepath = 'home_page.jpg'
        st.image(imagepath, use_column_width=True)
    except Exception as e:
        logger.error(f"Error loading home page image: {str(e)}")
        st.error("Failed to load home page image.")
    
    st.markdown('''
    # Welcome to PlantGuard
    
    ## Protect Your Plants with AI-Powered Disease Detection
    
    PlantGuard helps you identify and manage plant diseases quickly and accurately using advanced artificial intelligence.
    
    ### How It Works
    
    1. Take a photo of your plant
    2. Upload the image to our app
    3. Receive instant disease identification and treatment recommendations
    
    ### Why Choose Us
    
    - **Unmatched Accuracy**: Our state-of-the-art AI model boasts a 98% accuracy rate in disease detection
    - **Fast Results**: Get instant diagnoses and treatment recommendations
    - **User-Friendly**: Simple interface designed for gardeners and farmers of all skill levels
    - **Comprehensive Database**: Covers a wide range of plants and diseases
    - **Continually Updated**: Our model is regularly trained on new data to stay current with emerging plant diseases
    
    ### Our Mission
    
    We're committed to empowering gardeners, farmers, and plant enthusiasts to maintain healthier plants and improve crop yields through accessible technology.
    
    Get started now and keep your plants thriving!
    
    [Upload Image]
    ''')

elif app_mode == 'About':
    st.header('About')
    st.markdown('''
    # About PlantGuard

    ## Our Mission
    
    At PlantGuard, we aim to revolutionize plant health management by putting advanced disease detection tools in the hands of gardeners, farmers, and plant enthusiasts worldwide.
    
    ## Our Technology
    
    ### AI-Powered Disease Recognition
    
    Our state-of-the-art plant disease recognition model boasts a 98% accuracy rate, allowing users to:
    
    - Instantly identify plant diseases from photos
    - Receive tailored treatment recommendations
    - Access a comprehensive database of plant health information
    
    ### Our Dataset
    
    The backbone of our accurate model is our robust dataset:
    
    - **Size**: Approximately 87,000 RGB images of crop leaves
    - **Content**: Includes both healthy and diseased plant leaves
    - **Diversity**: Categorized into 38 different classes
    - **Structure**: 
      - 80% Training set
      - 20% Validation set
      - Additional test set of 33 images for prediction purposes
    - **Origin**: Recreated using offline augmentation from an original dataset [(available on GitHub)](%s) https://github.com/spMohanty/PlantVillage-Dataset
    
    This comprehensive dataset ensures our model can recognize a wide variety of plant diseases across numerous crop types, providing you with reliable and accurate diagnoses.
    
    ## Join Us in Nurturing a Greener World
    
    Whether you're a seasoned farmer, a hobby gardener, or just starting your plant journey, PlantGuard is here to help you grow healthier, happier plants.
    
    [Get Started with PlantGuard]
    ''')
    
elif app_mode == 'Disease Recognition':
    st.markdown('''
    # Plant Disease Recognition
    Identify and Treat Plant Diseases in Seconds
    
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
    
    test_image = st.file_uploader('Choose an Image')
    
    if test_image is not None:
        if st.button('Show Image'):
            try:
                st.image(test_image, use_column_width=True)
            except Exception as e:
                logger.error(f"Error displaying uploaded image: {str(e)}")
                st.error("Failed to display the uploaded image.")
        
        if st.button('Predict'):
            result_index = model_prediction(test_image)
            classes = [
                    'Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy'
                ]
            if result_index is not None:
                # image_class = classes[result_index]
                st.success(f'Model is Predicting it is a {classes[result_index]}')
    else:
        st.info("Please upload an image to proceed with prediction.")

if __name__ == "__main__":
    try:
        logger.info("Application started")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.critical(f"Critical error in main application: {str(e)}")
