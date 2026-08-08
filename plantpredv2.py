import streamlit as st
import tensorflow as tf
import numpy as np
import logging
import os

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve file paths relative to this script so the app works from any CWD
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trainedv3.keras")
HOME_IMAGE = os.path.join(BASE_DIR, "home_page.jpg")

# Disease class labels (38 classes for the v3 model)
CLASS_NAMES = [
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained v3 Keras model (cached for performance)."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error("Failed to load the model. Please try again later.")
        return None


def model_prediction(test_image):
    """Run disease prediction on an uploaded image."""
    model = load_model()
    if model is None:
        return None

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        result_ind = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)
        logger.info("Prediction successful")
        return result_ind, confidence
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        st.error("An error occurred during prediction. Please try again.")
        return None, None


def local_css():
    """Inject custom CSS for a modern, polished look."""
    st.markdown(
        """
        <style>
        /* ── General layout ── */
        .main { background: #0a192f; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] .sidebar-content {
            background: linear-gradient(180deg, #1a2a6c, #b21f1f, #fdbb2d);
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] .stMarkdown {
            color: #ffffff !important;
        }

        /* ── Cards ── */
        .card {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        /* ── Hero image ── */
        .hero-img {
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
            margin: 20px 0;
            border: 3px solid rgba(255, 255, 255, 0.15);
        }

        /* ── Buttons ── */
        .stButton>button {
            background: linear-gradient(135deg, #2e7d32, #1b5e20);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 12px 28px;
            font-size: 16px;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
        }

        /* ── File uploader ── */
        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 20px;
        }

        /* ── Typography ── */
        h1 { color: #ffffff; font-weight: 700; }
        h2 { color: #4caf50; }
        h3 { color: #90caf9; }

        /* ── Success / error boxes ── */
        .stSuccess { border-radius: 15px; }
        .stError { border-radius: 15px; }

        /* ── Footer ── */
        .footer {
            text-align: center;
            padding: 20px;
            color: #90caf9;
            font-size: 14px;
            margin-top: 30px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* ── Feature grid ── */
        .feature-box {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.15);
            transition: transform 0.2s;
        }
        .feature-box:hover { transform: translateY(-4px); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────
local_css()

st.sidebar.title("🌿 PlantGuard")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Page",
    ["🏠 Home", "ℹ️ About", "🔍 Disease Recognition"],
    index=0,
)
# Strip the emoji prefix for internal matching
page = app_mode.split(" ", 1)[-1] if " " in app_mode else app_mode

# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
if page == "Home":
    # ── Hero ──
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🌱 PlantGuard — Plant Disease Prediction")

    if os.path.exists(HOME_IMAGE):
        st.image(HOME_IMAGE, use_container_width=True,
                 caption="Protect Your Plants with AI-Powered Disease Detection")
    else:
        st.warning("Home image not found.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Intro text ──
    st.markdown(
        """
        <div class="card">
            <p style="font-size:18px; color:#e0e0e0;">
            PlantGuard helps you identify and manage plant diseases quickly and
            accurately using advanced artificial intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── How it works (3-column grid) ──
    st.markdown("### How It Works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="feature-box">📸<br><b>1. Take a Photo</b><br>Capture your plant leaf clearly.',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="feature-box">📤<br><b>2. Upload Image</b><br>Send it to PlantGuard for analysis.',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="feature-box">✅<br><b>3. Get Diagnosis</b><br>Receive instant disease identification & treatment.',
            unsafe_allow_html=True,
        )

    # ── Why Choose Us (2-column) ──
    st.markdown("### Why Choose PlantGuard")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            """
            <div class="card">
                <ul style="color:#e0e0e0; font-size:16px;">
                <li>🎯 <b>Unmatched Accuracy</b> — 98 % accuracy rate</li>
                <li>⚡ <b>Fast Results</b> — Instant diagnoses & recommendations</li>
                <li>📱 <b>User-Friendly</b> — Simple interface for everyone</li>
                <li>🌍 <b>Comprehensive</b> — Covers many crops & diseases</li>
                <li>🔄 <b>Continuously Updated</b> — Stay current with new data</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="card">
                <p style="color:#e0e0e0; font-size:16px;">
                We're committed to empowering gardeners, farmers, and plant
                enthusiasts to maintain healthier plants and improve crop yields
                through accessible technology.
                </p>
                <p><b>Get started now and keep your plants thriving!</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("ℹ️ About PlantGuard")

    st.markdown(
        """
        <p style="font-size:18px; color:#e0e0e0;">
        At PlantGuard, we aim to revolutionise plant health management by
        putting advanced disease detection tools in the hands of gardeners,
        farmers, and plant enthusiasts worldwide.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Our Technology")
    st.markdown(
        """
        <div class="card">
            <p style="color:#e0e0e0;">
            Our state-of-the-art plant disease recognition model, trained on
            approximately 87,000 RGB images of crop leaves, delivers a 98 %
            accuracy rate. It can:
            </p>
            <ul style="color:#e0e0e0;">
                <li>Instantly identify plant diseases from photos</li>
                <li>Receive tailored treatment recommendations</li>
                <li>Access a comprehensive database of plant health information</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Our Dataset")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            '<div class="feature-box"><b>~87,000</b><br>RGB Images</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="card">
                <ul style="color:#e0e0e0;">
                <li><b>Content</b>: healthy & diseased plant leaves</li>
                <li><b>Diversity</b>: 38 classes across many crops</li>
                <li><b>Split</b>: 80 % train / 20 % validation</li>
                <li><b>Test set</b>: 33 images for prediction</li>
                <li><b>Origin</b>: based on <a href="https://github.com/spMohanty/PlantVillage-Dataset" style="color:#90caf9;">PlantVillage-Dataset</a></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Disease Recognition":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("🔍 Plant Disease Recognition")
    st.markdown(
        """
        <p style="font-size:18px; color:#e0e0e0;">
        Identify and treat plant diseases in seconds.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <p style="color:#e0e0e0;"><b>Supported Crops</b></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for i, crop in enumerate(["Tomato", "Potato", "Corn", "Apple", "Grape"]):
        cols[i].markdown(f"✓ {crop}")

    st.markdown(
        """
        <div class="card">
            <p style="color:#e0e0e0;"><b>Tips for Best Results</b></p>
            <ul style="color:#e0e0e0;">
            <li>Ensure good lighting when taking photos</li>
            <li>Focus on the affected area</li>
            <li>Include both healthy and diseased parts for comparison</li>
            <li>Take multiple photos from different angles if needed</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    test_image = st.file_uploader(
        "📤 Choose an image to analyse", type=["jpg", "jpeg", "png"]
    )

    if test_image is not None:
        col_prev, col_pred = st.columns([1, 2])

        with col_prev:
            if st.button("Show Image"):
                st.image(test_image, use_container_width=True,
                         caption="Uploaded preview")

        with col_pred:
            if st.button("🚀 Predict Disease"):
                with st.spinner("Analysing…"):
                    result_index, confidence = model_prediction(test_image)
                if result_index is not None:
                    disease = CLASS_NAMES[result_index]
                    # Human-readable display name
                    display_name = disease.replace("___", " — ").replace("_", " ")
                    is_healthy = "healthy" in disease.lower()
                    emoji = "🟢" if is_healthy else "🔴"
                    st.markdown(
                        f"""
                        <div class="card">
                            <h2>{emoji} Prediction Result</h2>
                            <p style="font-size:22px;"><b>{display_name}</b></p>
                            <p style="font-size:18px; color:#90caf9;">
                            Confidence: {confidence:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    if is_healthy:
                        st.balloons()
                    else:
                        st.warning(
                            "Please consult an agricultural expert for treatment advice. "
                            "Common treatments include appropriate fungicides and removing "
                            "affected leaves promptly."
                        )
    else:
        st.info("📸 Upload an image to proceed with prediction.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        PlantGuard — AI-Powered Plant Disease Detection ·
        Built with Streamlit & TensorFlow ·
        <a href="https://github.com/Gbolahan43/crop-disease-app" style="color:#90caf9;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
