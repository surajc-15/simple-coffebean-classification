import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="☕ Coffee Bean Classifier",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for enhanced styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #8B4513 0%, #D2691E 50%, #A0522D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #6B4423;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #8B4513 0%, #D2691E 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(139, 69, 19, 0.3);
        margin: 1rem 0;
    }
    
    .grade-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        margin: 0.5rem;
    }
    
    .grade-a { background-color: #28a745; color: white; }
    .grade-b { background-color: #17a2b8; color: white; }
    .grade-c { background-color: #ffc107; color: black; }
    .grade-d { background-color: #dc3545; color: white; }
    
    .upload-section {
        border: 3px dashed #D2691E;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: rgba(210, 105, 30, 0.1);
        margin: 2rem 0;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #D2691E;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Load model function
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("coffee_bean_classifier.h5")
    except Exception as e:
        st.error(f"Model not found: {e}")
        return None


# Initialize
model = load_model()
class_names = ["Dark", "Green", "Light", "Medium"]
class_descriptions = {
    "Dark": "Highest quality - Rich, bold flavor with perfect roasting.",
    "Green": "High quality - Fresh unroasted beans with excellent potential.",
    "Light": "Good quality - Bright, acidic flavor with floral notes.",
    "Medium": "Standard quality - Balanced but lower grade beans.",
}

grade_info = {
    "A": {"color": "#28a745", "description": "Premium Quality - Exceptional beans"},
    "B": {"color": "#17a2b8", "description": "High Quality - Very good beans"},
    "C": {"color": "#ffc107", "description": "Good Quality - Standard beans"},
    "D": {"color": "#dc3545", "description": "Fair Quality - Basic beans"},
}

# Header
st.markdown(
    '<h1 class="main-header">☕ Coffee Bean Classifier</h1>', unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">AI-Powered Coffee Bean Quality Assessment</p>',
    unsafe_allow_html=True,
)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
    <div class="upload-section">
        <h3>📸 Upload Your Coffee Bean Image</h3>
        <p>Get instant classification and quality assessment</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

if uploaded_file is not None and model is not None:
    # Create main layout
    image_col, results_col = st.columns([1, 1])

    with image_col:
        st.markdown("### 📷 Your Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Coffee Bean", use_column_width=True)

        # Image info
        st.markdown(
            f"""
        <div class="info-card">
            <strong>Image Details:</strong><br>
            📏 Size: {image.size[0]} x {image.size[1]} pixels<br>
            📁 Format: {image.format}<br>
            💾 File size: {len(uploaded_file.getvalue())} bytes
        </div>
        """,
            unsafe_allow_html=True,
        )

    with results_col:
        st.markdown("### 🔬 Analysis Results")

        # Preprocess and predict
        with st.spinner("Analyzing your coffee bean..."):
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            pred_idx = np.argmax(prediction)
            label = class_names[pred_idx]
            confidence = float(np.max(prediction)) * 100

            # Grade mapping based on quality order (Dark=highest, Medium=lowest)
            grade_map = {"Dark": "A", "Green": "B", "Light": "C", "Medium": "D"}
            grade = grade_map.get(label, "N/A")

        # Main prediction display
        st.markdown(
            f"""
        <div class="prediction-box">
            <h2>🏆 {label} Roast</h2>
            <h3>Confidence: {confidence:.1f}%</h3>
            <p>{class_descriptions.get(label, "")}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Grade badge
        grade_color = grade_info.get(grade, {}).get("color", "#666")
        grade_desc = grade_info.get(grade, {}).get("description", "")

        st.markdown(
            f"""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="background-color: {grade_color}; color: white; 
                        padding: 1rem 2rem; border-radius: 50px; 
                        display: inline-block; font-weight: bold; font-size: 1.2rem;">
                Grade: {grade}
            </div>
            <p style="margin-top: 0.5rem; color: #666;">{grade_desc}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Confidence chart
    st.markdown("---")
    st.markdown("### 📊 Confidence Breakdown")

    # Create confidence chart
    confidence_data = {
        "Bean Type": class_names,
        "Confidence": [float(pred * 100) for pred in prediction[0]],
    }

    fig = px.bar(
        confidence_data,
        x="Bean Type",
        y="Confidence",
        title="Classification Confidence Scores",
        color="Confidence",
        color_continuous_scale="earth",
    )

    fig.update_layout(
        title_x=0.5,
        showlegend=False,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")

    st.plotly_chart(fig, use_container_width=True)

    # Additional metrics
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <h2>{confidence:.1f}%</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with metrics_col2:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>🏅 Quality Grade</h3>
            <h2>{grade}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with metrics_col3:
        second_best_idx = np.argsort(prediction[0])[-2]
        second_confidence = prediction[0][second_best_idx] * 100
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>🥈 2nd Choice</h3>
            <h2>{class_names[second_best_idx]}</h2>
            <p>{second_confidence:.1f}%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

elif uploaded_file is not None and model is None:
    st.error(
        "❌ Model not loaded. Please ensure 'coffee_bean_classifier.h5' is available."
    )

# Footer with tips
st.markdown("---")
st.markdown("### 💡 Tips for Best Results")

tip_col1, tip_col2, tip_col3 = st.columns(3)

with tip_col1:
    st.markdown(
        """
    <div class="info-card">
        <h4>📸 Image Quality</h4>
        <ul>
            <li>Use clear, well-lit photos</li>
            <li>Focus on single beans</li>
            <li>Avoid blurry images</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with tip_col2:
    st.markdown(
        """
    <div class="info-card">
        <h4>🔍 Best Angles</h4>
        <ul>
            <li>Top-down view preferred</li>
            <li>Fill the frame with the bean</li>
            <li>Natural lighting works best</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

with tip_col3:
    st.markdown(
        """
    <div class="info-card">
        <h4>📏 File Requirements</h4>
        <ul>
            <li>JPG, JPEG, or PNG format</li>
            <li>Reasonable file size</li>
            <li>Minimum 128x128 pixels</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )

# About section
with st.expander("ℹ️ About This Classifier"):
    st.markdown(
        """
    This AI-powered coffee bean classifier uses deep learning to identify and grade coffee beans based on their roast level and visual characteristics.
    
    **Bean Types (Quality Order - Highest to Lowest):**
    - **Dark Roast (Grade A)**: Highest quality - Bold, rich flavor with perfect roasting
    - **Green Beans (Grade B)**: High quality - Premium unroasted beans 
    - **Light Roast (Grade C)**: Good quality - Bright, acidic with complex flavors
    - **Medium Roast (Grade D)**: Standard quality - Balanced but lower grade beans
    
    **Grading System:**
    - **Grade A**: Premium quality beans
    - **Grade B**: High quality beans  
    - **Grade C**: Standard quality beans
    - **Grade D**: Basic quality beans
    """
    )

st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #666;">☕ Powered by AI • Built with Streamlit • Made for Coffee Lovers</p>',
    unsafe_allow_html=True,
)
