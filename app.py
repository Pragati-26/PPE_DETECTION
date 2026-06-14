import streamlit as st
from PIL import Image
from src.detector import PPEDetector

# 1. Page Config & Title Layout
st.set_page_config(
    page_title="PPE Compliance Monitor", 
    page_icon="🛡️", 
    layout="centered"
)

st.title("🛡️ AI PPE Safety Compliance Monitor")
st.markdown(
    "Upload a workspace image below to automatically audit personnel for mandatory "
    "safety gear (Hardhats, Safety Vests, Gloves, Goggles, etc.)."
)

# 2. Sidebar Configuration Controls
st.sidebar.header("⚙️ Model Configuration")
st.sidebar.markdown("Fine-tune the detection sensitivity below:")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Lower values catch more objects but increase false positives. Higher values show only confident detections."
)

iou_threshold = st.sidebar.slider(
    "NMS IoU Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="Controls overlapping bounding box suppression. Adjust if objects are clustered closely."
)

# 3. Cached Model Loader
@st.cache_resource
def load_ppe_detector():
    """
    Loads the custom-trained weights into memory exactly once.
    Using cache_resource prevents reload delays on interface clicks.
    """
    # Use relative path for cross-platform compatibility
    return PPEDetector(model_path="models/best.pt")

try:
    detector = load_ppe_detector()
except Exception as e:
    st.error(f"Failed to load model weights. Ensure 'models/best.pt' exists. Error: {e}")
    st.stop()

# 4. Image Upload Mechanism
uploaded_file = st.file_uploader(
    "Choose a workplace snapshot...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read the uploaded file into a PIL Image format
    input_image = Image.open(uploaded_file)
    
    # Layout two columns to visually compare before and after execution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Safety Compliance Audit")
        
        # Add an interactive action button to trigger inference
        if st.button("Run Safety Scan", type="primary"):
            with st.spinner("Analyzing equipment compliance metrics..."):
                # Pass UI parameters straight to our modular backend logic
                processed_img = detector.detect(
                    image=input_image, 
                    conf_threshold=conf_threshold, 
                    iou_threshold=iou_threshold
                )
                
                # Display processed frame with labels, bounding boxes, and metadata
                st.image(processed_img, use_container_width=True)
                st.success("Audit complete!")