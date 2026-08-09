"""PlantGuard — light "field report" front-end.

A redesign of the original app with a light, print-friendly agronomy-report look:
top-3 ranked predictions with confidence bars, a low-confidence warning, and the
model's coverage listed from the shared class registry.

Loads `newplantdis.keras`, the better of the two checkpoints (0.9767 validation
accuracy vs 0.8860 for trainedv3.keras).

Run with:  streamlit run newplantpred.py

See `plantpred.py` for the original, deliberately minimal v1 UI, and
`plantpredv2.py` for the dark-themed variant.
"""

import logging
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
    NUM_CLASSES,
    crops,
    diseases_by_crop,
    is_healthy,
    pretty_label,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Below this softmax probability the model's top guess is not worth reporting
# as an answer -- PlantVillage images are lab-style, so field photos and
# off-domain uploads often land here.
CONFIDENCE_FLOOR = 60.0

st.set_page_config(
    page_title="PlantGuard — Field Report",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────────────
# Styling — light, paper-like, distinct from the dark plantpredv2.py theme
# ──────────────────────────────────────────────────────────────────────────────
def inject_css():
    # This page forces a light background, so every piece of text needs an
    # explicit dark colour: under a dark Streamlit theme the inherited text
    # colour is near-white and vanishes. Streamlit's own theme rules are more
    # specific than a bare `h1`, hence the data-testid selectors and !important.
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background: #f5f7f2;
            color-scheme: light;
        }

        /* ── Base text colours ── */
        .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
            color: #2f3a34;
        }
        .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp h5, .stApp h6 {
            color: #14301f !important;
            font-weight: 700;
        }
        /* Streamlit wraps markdown headings in its own container */
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4,
        [data-testid="stMarkdownContainer"] h5,
        [data-testid="stMarkdownContainer"] h6 { color: #14301f !important; }
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li { color: #2f3a34; }
        [data-testid="stHeaderActionElements"] { display: none; }

        /* ── Captions and metrics ── */
        [data-testid="stCaptionContainer"], .stCaption,
        [data-testid="stCaptionContainer"] p, .stCaption p {
            color: #55655a !important;
        }
        [data-testid="stMetricValue"] { color: #14301f !important; }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p { color: #55655a !important; }

        /* ── Sidebar: white panel, so the same dark-text treatment ── */
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e0e5db;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 { color: #14301f !important; }
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] div { color: #2f3a34; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] .stCaption,
        [data-testid="stSidebar"] .stCaption p { color: #55655a !important; }
        /* The model-name code block keeps its dark chip styling */
        [data-testid="stSidebar"] pre,
        [data-testid="stSidebar"] code,
        [data-testid="stSidebar"] pre * ,
        [data-testid="stSidebar"] code * { color: #eaf1ea !important; }

        /* ── Widget labels (radio, uploader) ── */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label { color: #14301f !important; font-weight: 600; }
        [data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 1px dashed #b9c6bb; }
        [data-testid="stFileUploaderDropzone"] * { color: #2f3a34 !important; }

        /* ── Progress bars (used for the ranked candidates) ── */
        [data-testid="stProgress"] p,
        [data-testid="stProgress"] div { color: #2f3a34 !important; }

        /* ── Alerts: force light panels so dark-theme text stays readable ── */
        [data-testid="stAlert"], .stAlert, [data-testid="stAlertContainer"] {
            background: #ffffff !important;
            border: 1px solid #d9e2db !important;
            border-radius: 6px;
        }
        [data-testid="stAlert"] *, .stAlert *,
        [data-testid="stAlertContainer"] * { color: #2f3a34 !important; }

        .paper {
            background: #ffffff;
            border: 1px solid #e0e5db;
            border-left: 4px solid #3f7d4f;
            border-radius: 6px;
            padding: 22px 26px;
            margin: 14px 0;
            box-shadow: 0 1px 3px rgba(27, 58, 43, 0.08);
        }
        .paper p, .paper li, .paper span {
            color: #2f3a34 !important;
            font-size: 15px;
            line-height: 1.6;
        }
        .paper b, .paper strong { color: #14301f !important; }
        .paper a { color: #2c6b40 !important; text-decoration: underline; }

        /* Page title — set explicitly rather than relying on the theme */
        .page-title {
            font-size: 42px;
            line-height: 1.15;
            font-weight: 700;
            color: #14301f !important;
            margin: 4px 0 18px;
        }

        .kicker {
            text-transform: uppercase;
            letter-spacing: 1.4px;
            font-size: 12px !important;
            font-weight: 700;
            color: #55655a !important;
            margin-bottom: 6px;
        }

        .verdict-healthy { border-left-color: #3f7d4f; }
        .verdict-disease { border-left-color: #b5462f; }
        .verdict-unsure  { border-left-color: #c98a1a; }

        .verdict-name {
            font-size: 26px !important;
            font-weight: 700;
            color: #14301f !important;
            margin: 2px 0 4px;
        }
        .verdict-sub { font-size: 14px !important; color: #55655a !important; }

        .stButton>button {
            background: #3f7d4f;
            color: #ffffff;
            border: none;
            border-radius: 4px;
            padding: 10px 22px;
            font-weight: 600;
        }
        .stButton>button:hover { background: #346942; }

        .step {
            background: #ffffff;
            border: 1px solid #e0e5db;
            border-radius: 6px;
            padding: 18px;
            text-align: center;
            height: 100%;
        }
        .step, .step b { color: #14301f !important; }
        .step span { color: #55655a !important; font-size: 14px; }

        .foot {
            text-align: center;
            color: #55655a !important;
            font-size: 13px;
            padding: 18px;
            margin-top: 26px;
            border-top: 1px solid #e0e5db;
        }
        .foot a { color: #2c6b40 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load and cache the best-performing checkpoint."""
    try:
        model = tf.keras.models.load_model(BEST_MODEL_PATH)
        logger.info("Model loaded from %s", BEST_MODEL_PATH)
        return model
    except Exception as exc:
        logger.error("Error loading model: %s", exc)
        st.error(
            f"Failed to load the model from `{os.path.basename(BEST_MODEL_PATH)}`. "
            "Check that the checkpoint file is present, then reload the page."
        )
        return None


def predict(test_image, top_k=3):
    """Return the top-k ``(label, confidence_pct)`` pairs, or None on failure."""
    model = load_model()
    if model is None:
        return None

    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=IMAGE_SIZE)
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])

        probs = model.predict(input_arr, verbose=0)[0]
        ranked = np.argsort(probs)[::-1][:top_k]
        logger.info("Prediction successful: %s", CLASS_NAMES[ranked[0]])
        return [(CLASS_NAMES[i], float(probs[i]) * 100.0) for i in ranked]
    except Exception as exc:
        logger.error("Error during prediction: %s", exc)
        st.error("An error occurred during prediction. Please try another image.")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Shared fragments
# ──────────────────────────────────────────────────────────────────────────────
def show_home_image():
    """Render the hero image, warning rather than crashing if it is missing."""
    if os.path.exists(HOME_IMAGE_PATH):
        st.image(
            HOME_IMAGE_PATH,
            use_container_width=True,
            caption="AI-assisted leaf disease screening",
        )
    else:
        logger.warning("Home image missing at %s", HOME_IMAGE_PATH)
        st.info("Hero image `home_page.jpg` not found — skipping.")


def render_verdict(results):
    """Render the top prediction plus the runners-up."""
    label, confidence = results[0]
    healthy = is_healthy(label)
    uncertain = confidence < CONFIDENCE_FLOOR

    if uncertain:
        tone, badge = "verdict-unsure", "⚠️ Low confidence"
    elif healthy:
        tone, badge = "verdict-healthy", "🟢 Healthy"
    else:
        tone, badge = "verdict-disease", "🔴 Disease detected"

    st.markdown(
        f"""
        <div class="paper {tone}">
            <div class="kicker">{badge}</div>
            <div class="verdict-name">{pretty_label(label)}</div>
            <div class="verdict-sub">Top-1 confidence: {confidence:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uncertain:
        st.warning(
            f"The model is only {confidence:.1f}% confident, below the "
            f"{CONFIDENCE_FLOOR:.0f}% reporting threshold. This often means the "
            "photo is out of scope (a crop the model does not cover, a whole "
            "plant rather than a single leaf, or a non-leaf subject). Treat the "
            "result as a hint, not a diagnosis."
        )

    st.markdown("##### Ranked candidates")
    for rank, (candidate, score) in enumerate(results, start=1):
        st.progress(
            min(max(score / 100.0, 0.0), 1.0),
            text=f"{rank}. {pretty_label(candidate)} — {score:.1f}%",
        )

    if not healthy and not uncertain:
        st.info(
            "Next step: confirm with an agronomist or extension officer before "
            "treating. This tool reports a label only — it does not prescribe "
            "pesticide or dosage."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
inject_css()

st.sidebar.title("🌿 PlantGuard")
st.sidebar.caption("Field report edition")
page = st.sidebar.radio("Section", ["Home", "About", "Disease Recognition"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("**Active model**")
st.sidebar.code(os.path.basename(BEST_MODEL_PATH), language=None)
st.sidebar.caption(
    f"{NUM_CLASSES} classes · {len(crops())} crops · "
    "0.977 validation accuracy · 128×128 input"
)

# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────
if page == "Home":
    st.markdown('<div class="kicker">Plant disease screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-title">Know what is wrong with your leaf</div>',
        unsafe_allow_html=True,
    )

    show_home_image()

    st.markdown(
        """
        <div class="paper">
            <p>PlantGuard screens a photograph of a single crop leaf against a
            convolutional neural network trained on the PlantVillage dataset, and
            reports the most likely condition with a confidence score. It covers
            <b>%d conditions across %d crops</b>.</p>
        </div>
        """
        % (NUM_CLASSES, len(crops())),
        unsafe_allow_html=True,
    )

    st.markdown("### How it works")
    steps = [
        ("📸", "1. Photograph", "One leaf, good light, plain background."),
        ("📤", "2. Upload", "JPG or PNG, on the Disease Recognition page."),
        ("📄", "3. Read the report", "Ranked candidates with confidence scores."),
    ]
    for column, (icon, title, body) in zip(st.columns(3), steps):
        with column:
            st.markdown(
                f'<div class="step">{icon}<br><b>{title}</b><br><span>{body}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### What it is, and is not")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="paper verdict-healthy">
                <div class="kicker">It does</div>
                <ul>
                <li>Rank the likeliest conditions with confidence scores</li>
                <li>Flag results below the reporting threshold as uncertain</li>
                <li>Run offline, in seconds, on a CPU</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="paper verdict-disease">
                <div class="kicker">It does not</div>
                <ul>
                <li>Prescribe treatment, pesticide or dosage</li>
                <li>Recognise crops outside the %d it was trained on</li>
                <li>Replace a lab test or an agronomist's judgement</li>
                </ul>
            </div>
            """
            % len(crops()),
            unsafe_allow_html=True,
        )

elif page == "About":
    st.markdown('<div class="page-title">About PlantGuard</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="paper">
            <div class="kicker">Purpose</div>
            <p>Put leaf-disease screening in the hands of growers who cannot
            reach a plant clinic quickly. Upload a leaf, get a ranked diagnosis
            and a confidence score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    accuracy, coverage, dataset = st.columns(3)
    accuracy.metric("Validation accuracy", "97.7%")
    coverage.metric("Classes", NUM_CLASSES)
    dataset.metric("Training images", "70,295")

    st.markdown(
        """
        <div class="paper">
            <div class="kicker">Model</div>
            <p>A VGG-style CNN (~5.0M parameters) built with
            <code>tf.keras.Sequential</code>: five Conv/BatchNorm blocks of
            32→512 filters, global average pooling, a 512-unit dense layer with
            dropout, and a %d-way softmax. Trained with Adam at a 1e-4 learning
            rate on 128×128 RGB crops.</p>
            <p><b>Reported accuracy is measured on the validation split</b>,
            which was also used to monitor training, so there is no untouched
            test set — read 97.7%% as an optimistic figure. Per-class
            precision, recall and F1 average 0.98; late blight is the weakest
            case, being visually similar across tomato and potato.</p>
        </div>
        """
        % NUM_CLASSES,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="paper">
            <div class="kicker">Dataset</div>
            <p>The offline-augmented
            <a href="https://github.com/spMohanty/PlantVillage-Dataset">PlantVillage</a>
            collection — roughly 87,000 RGB leaf photographs, healthy and
            diseased, in %d classes: 70,295 for training and 17,572 for
            validation.</p>
            <p>The images are lab-style: one leaf, even light, uniform
            background. Accuracy on real field photography is materially lower,
            which is what the confidence floor on the results page is for.</p>
        </div>
        """
        % NUM_CLASSES,
        unsafe_allow_html=True,
    )

    st.markdown("### Coverage")
    grouped = diseases_by_crop()
    for column, crop_list in zip(st.columns(2), (crops()[:7], crops()[7:])):
        with column:
            for crop in crop_list:
                conditions = ", ".join(grouped[crop])
                st.markdown(f"**{crop}** — {conditions}")

elif page == "Disease Recognition":
    st.markdown('<div class="page-title">Disease recognition</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="paper">
            <div class="kicker">Before you upload</div>
            <ul>
            <li>Fill the frame with a <b>single leaf</b>, focused on the affected area</li>
            <li>Even, indirect light — avoid glare and hard shadows</li>
            <li>Plain background where possible</li>
            <li>Include some healthy tissue alongside the damage for contrast</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    test_image = st.file_uploader(
        "Leaf photograph", type=ALLOWED_UPLOAD_TYPES, label_visibility="visible"
    )

    if test_image is None:
        st.info("Upload a JPG or PNG leaf photograph to run a screening.")
    else:
        preview, report = st.columns([2, 3])

        with preview:
            st.markdown("##### Uploaded image")
            try:
                st.image(test_image, use_container_width=True)
            except Exception as exc:
                logger.error("Error displaying uploaded image: %s", exc)
                st.error("Could not display that file — is it a valid image?")
            run = st.button("Run screening", use_container_width=True)

        with report:
            st.markdown("##### Report")
            if run:
                with st.spinner("Analysing leaf…"):
                    results = predict(test_image)
                if results:
                    render_verdict(results)
            else:
                st.caption("Press **Run screening** to analyse this image.")

st.markdown(
    """
    <div class="foot">
        PlantGuard · TensorFlow + Streamlit ·
        <a href="https://github.com/Gbolahan43/crop-disease-app">GitHub</a><br>
        Screening aid only — not a substitute for professional plant diagnosis.
    </div>
    """,
    unsafe_allow_html=True,
)
