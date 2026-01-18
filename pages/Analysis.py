import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2 as mobilenetv2  # Thêm import này để fix NameError

# ==============================
# 🔧 CONFIG
# ==============================
MODEL_DIR = "models/waste_mobilenetv2.keras"  # Đường dẫn file .keras
LABEL_FILE = "models/labels.pkl"  # file nhãn

# ==============================
# 🔶 DICTIONARY CHO HƯỚNG DẪN VÀ MẸO
# ==============================
RECYCLE_INSTRUCTIONS = {
    "plastic": "Đặt vào thùng tái chế màu vàng dành cho vật liệu nhựa.",
    "paper": "Đặt vào thùng tái chế màu xanh dành cho giấy và bìa carton.",
    "glass": "Đặt vào thùng tái chế màu xanh dương dành cho thủy tinh.",
    "metal": "Đặt vào thùng tái chế màu xám dành cho kim loại.",
    "organic": "Đổ vào thùng phân hủy sinh học hoặc ủ phân hữu cơ.",
    "others": "Đặt vào thùng rác thông thường không tái chế."
}

ENV_TIPS = {
    "plastic": "Tái chế 1 kg nhựa tiết kiệm năng lượng tương đương 0.7 lít dầu.",
    "paper": "Tái chế 1 tấn giấy tiết kiệm 17 cây xanh và 26.000 lít nước.",
    "glass": "Tái chế thủy tinh tiết kiệm 30% năng lượng so với sản xuất mới.",
    "metal": "Tái chế kim loại giảm 95% năng lượng so với khai thác quặng.",
    "organic": "Ủ phân hữu cơ giảm khí metan và tạo phân bón tự nhiên.",
    "others": "Giảm thiểu rác không tái chế để bảo vệ môi trường."
}

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
    Hiển thị kết quả dự đoán với cấu trúc mới: label lớn, confidence, badge màu, hướng dẫn, tip, nút thử lại, và link feedback.
    """
    # Xác định màu badge dựa trên confidence
    if conf > 0.8:
        badge_color = "#28a745"  # Xanh (high)
        badge_text = "Cao"
    elif conf > 0.5:
        badge_color = "#ffc107"  # Vàng (medium)
        badge_text = "Trung bình"
    else:
        badge_color = "#fd7e14"  # Cam (low)
        badge_text = "Thấp"

    # Lấy hướng dẫn và tip dựa trên label (lowercase vì label có thể capitalize)
    lower_label = label.lower()
    instruction = RECYCLE_INSTRUCTIONS.get(lower_label, "Không có hướng dẫn cụ thể.")
    tip = ENV_TIPS.get(lower_label, "Không có mẹo cụ thể.")

    # Box chính với kết quả ở giữa
    st.markdown(
        f"""
        <div style="
            background-color:#e6ffe6;
            padding:20px;
            border-radius:10px;
            border:1px solid #66cc66;
            font-size:18px;
            line-height:1.6;
            text-align:center;
            margin: 20px 0;">
            <h2 style="color:#006600; font-size:36px; font-weight:bold; margin:10px 0;">{label}</h2>
            <p style="font-size:24px; margin:5px 0;">{round(conf * 100, 2)}% tự tin</p>
            <span style="background-color:{badge_color}; color:white; padding:5px 10px; border-radius:5px; font-weight:bold;">{badge_text}</span>
            <p style="margin-top:20px; font-size:18px;"><strong>Hướng dẫn tái chế:</strong> {instruction}</p>
            <p style="font-size:16px; color:#555;"><em>Mẹo môi trường: {tip}</em></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Nút "Try Another Photo"
    if st.button("Thử ảnh khác"):
        st.experimental_rerun()  # Reload trang để upload ảnh mới

    # Link "Report Wrong Prediction"
    st.markdown(
        '<p style="text-align:center; font-size:14px;"><a href="https://forms.gle/EXAMPLE_FORM_LINK" target="_blank">Báo cáo dự đoán sai</a></p>',
        unsafe_allow_html=True,
    )

# ==============================
# 🔶 LOAD SAVEDMODEL + LABELS
# ==============================
@st.cache_resource
def load_infer_and_labels():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Không tìm thấy file mô hình: {MODEL_DIR}.")
        st.stop()
    # Load model bằng Keras (thay vì SavedModel)
    model = load_model(MODEL_DIR)  # Thay tf.saved_model.load()
    # Không cần signatures nữa, vì Keras model dùng trực tiếp model.predict()

    if not os.path.exists(LABEL_FILE):
        st.error("Không tìm thấy labels.pkl trong thư mục models/.")
        st.stop()
    labels = joblib.load(LABEL_FILE)
    return model, labels  # Trả về model thay vì infer_fn

# ==============================
# 🔶 HÀM DỰ ĐOÁN AUTO-KERAS
# ==============================
infer, LABELS = load_infer_and_labels()
def predict_image(pil_img: Image.Image):
    img = pil_img.resize((224, 224))
    arr = np.array(img)  # [0,255]
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    arr = mobilenetv2.preprocess_input(arr)  # [-1,1]
    probs = infer.predict(arr)[0]
    idx = np.argmax(probs)
    conf = probs[idx]
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
        st.error("Không tìm thấy thư mục images_raw/.")
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
        "Select an image to classify",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Ảnh: {file.name}")
            img = Image.open(file).convert("RGB")
            st.image(img, width=250, caption="Upload image")
            if st.button(f"Predict {file.name}"):
                label, conf = predict_image(img)
                result_box(label, conf)  # Gọi hàm result_box đã chỉnh sửa
            st.write("---")
