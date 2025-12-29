import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

# ==============================
# 🔧 CONFIG
# ==============================
MODEL_DIR = "models/waste_model"  # thư mục SavedModel (model.export)
LABEL_FILE = "models/labels.pkl"  # file nhãn

# ==============================
# 🔶 STYLE BOX
# ==============================
def intro_box(text: str):
    st.markdown(
        f"""
        <div style="
            background-color:#fff7cc;
            padding:20px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:18px;
            line-height:1.6;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def result_box(label: str, conf: float):
    """
    Hiển thị kết quả dự đoán trong một box đẹp với màu xanh thành công.
    """
    st.markdown(
        f"""
        <div style="
            background-color:#e6ffe6;
            padding:20px;
            border-radius:10px;
            border:1px solid #66cc66;
            font-size:18px;
            line-height:1.6;
            text-align:center;">
            <h3 style="color:#006600; margin-bottom:10px;">Kết quả dự đoán</h3>
            <p style="font-size:24px; font-weight:bold; margin:5px 0;">Loại rác: {label}</p>
            <p style="font-size:20px; margin:5px 0;">Độ tự tin: {round(conf * 100, 2)}%</p>
            <span style="font-size:30px;">✅</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================
# 🔶 LOAD SAVEDMODEL + LABELS
# ==============================
@st.cache_resource
def load_infer_and_labels():
    # Kiểm tra model
    if not os.path.exists(MODEL_DIR):
        st.error("❌ Không tìm thấy thư mục SavedModel: models/waste_model.\nHãy chạy train_autokeras.py trước.")
        st.stop()
    # Load SavedModel (KHÔNG dùng keras.models.load_model)
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures["serving_default"]
    # Load labels
    if not os.path.exists(LABEL_FILE):
        st.error("❌ Không tìm thấy labels.pkl trong thư mục models/.")
        st.stop()
    labels = joblib.load(LABEL_FILE)
    return infer, labels

infer, LABELS = load_infer_and_labels()

# ==============================
# 🔶 HÀM DỰ ĐOÁN AUTO-KERAS
# ==============================
def predict_image(pil_img: Image.Image):
    """
    Nhận ảnh PIL, resize và gọi SavedModel.
    AutoKeras SavedModel yêu cầu input: uint8, shape (1, 224, 224, 3)
    """
    # 1. Resize về 224x224
    img = pil_img.resize((224, 224))
    # 2. Chuyển sang numpy uint8 (0–255)
    arr = np.array(img, dtype=np.uint8)
    # 3. Thêm chiều batch → (1, 224, 224, 3)
    arr = np.expand_dims(arr, axis=0)
    # 4. Chuyển sang tensor uint8
    tensor = tf.convert_to_tensor(arr, dtype=tf.uint8)
    # 5. Gọi SavedModel
    output = infer(tensor)
    # AutoKeras trả dict, thường key là "output_0"
    probs = list(output.values())[0].numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return LABELS[idx], conf

# ==============================
# 🔶 TRANG ANALYSIS
# ==============================
def show():
    st.markdown(
        "<h2 style='color:#2b6f3e;'>Analysis – Data Analysis & Image Classification Demo (AutoKeras SavedModel)</h2>",
        unsafe_allow_html=True,
    )
    dataset_path = "images_raw"
    # ------------------------------
    # 1. THỐNG KÊ DATASET
    # ------------------------------
    intro_box("""
    <h3 style="color:#b30000;">1. Dataset statistics</h3>
    Automatic directory reading system <b>images_raw/</b> and compile statistics on the number of images for each category of waste.
    """)
    if not os.path.exists(dataset_path):
        st.error("⚠ Không tìm thấy thư mục images_raw/.")
        return
    classes = sorted(
        [c for c in os.listdir(dataset_path)
         if os.path.isdir(os.path.join(dataset_path, c))]
    )
    stats = {}
    for cls in classes:
        folder = os.path.join(dataset_path, cls)
        count = len([
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        stats[cls] = count
    st.table({"Lớp": list(stats.keys()), "Số ảnh": list(stats.values())})
    st.write("---")
    # ------------------------------
    # 2. ẢNH MẪU NGẪU NHIÊN
    # ------------------------------
    intro_box("""
    <h3 style="color:#b30000;">2. Random sample image in the dataset</h3>
    """)
    cols = st.columns(3)
    for i, cls in enumerate(classes):
        folder = os.path.join(dataset_path, cls)
        imgs = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not imgs:
            continue
        img_path = os.path.join(folder, random.choice(imgs))
        with cols[i % 3]:
            st.image(img_path, caption=cls)
    st.write("---")
    # ------------------------------
    # 3. DEMO PHÂN LOẠI ẢNH
    # ------------------------------
    intro_box("""
    <h3 style="color:#b30000;">3. Image Classification Demo Using AutoKeras SavedModel</h3>
    Upload one or more images, and the system will predict the corresponding type of waste.
    """)
    uploaded_files = st.file_uploader(
        "📤 Select an image to classify",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Ảnh: {file.name}")
            img = Image.open(file).convert("RGB")
            st.image(img, width=250, caption="Upload image")
            if st.button(f"🔍 Predict {file.name}"):
                label, conf = predict_image(img)
                result_box(label, conf)
            st.write("---")
