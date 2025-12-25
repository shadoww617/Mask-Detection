import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import io

MODEL_PATH = "runs/detect/mask_detector2/weights/best.pt"
VAL_DIR = "dataset/images/val"

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Face Mask Detector", layout="wide")
st.title("😷 Smart Face Mask Detection System")
st.write("Upload an image or pick a sample image to detect mask-wearing status.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- INPUT MODE ----------------
st.sidebar.header("Input Mode")
mode = st.sidebar.radio("Choose Input Type", ["Upload Image", "Use Sample Image"])

input_image = None

# Upload Mode
if mode == "Upload Image":
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")    # FORCE RGB
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

# Sample Mode
else:
    samples = sorted(os.listdir(VAL_DIR))
    choice = st.selectbox("Select Sample Image", samples)
    image_path = os.path.join(VAL_DIR, choice)
    input_image = Image.open(image_path).convert("RGB")      # FORCE RGB
    st.image(input_image, caption="Sample Image", use_column_width=True)

# ---------------- RUN DETECTION ----------------
if input_image and st.button("Run Detection"):
    with st.spinner("Detecting..."):
        results = model.predict(
            source=np.array(input_image),
            conf=0.25,
            imgsz=416,
            show_labels=True,
            show_conf=False,
            line_thickness=1
        )
        r = results[0]

        # Render result in memory
        output_image = r.plot()
        st.success("Detection Complete!")

        col1, col2 = st.columns([3, 1])

        # Show Image
        with col1:
            st.image(output_image, caption="Detection Result", use_column_width=True)

        # Summary panel
        with col2:
            st.subheader("📊 Detection Summary")
            if len(r.boxes) == 0:
                st.write("No faces detected.")
            else:
                classes = {0: "With Mask", 1: "Without Mask", 2: "Mask Incorrect"}
                counts = {c: 0 for c in classes.values()}

                for b in r.boxes:
                    c = int(b.cls)
                    counts[classes[c]] += 1

                for k, v in counts.items():
                    st.write(f"**{k}:** {v}")

        # Download button
        buf = io.BytesIO()
        Image.fromarray(output_image).save(buf, format="PNG")
        st.download_button(
            label="Download Result",
            data=buf.getvalue(),
            mime="image/png",
            file_name="mask_detection_result.png"
        )

st.write("---")
st.caption("Deep Learning Project | YOLO Mask Detector")
