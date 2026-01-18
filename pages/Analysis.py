import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2 as mobilenetv2

# ==============================
# 🔧 CONFIG
# ==============================
MODEL_DIR = "models/waste_mobilenetv2.keras"
LABEL_FILE = "models/labels.pkl"

# ==============================
# 🔶 LOAD SAVEDMODEL + LABELS
# ==============================
@st.cache_resource
def load_infer_and_labels():
    if not os.path.exists(MODEL_DIR):
        st.error(f"Không tìm thấy file mô hình: {MODEL_DIR}.")
        st.stop()
    model = load_model(MODEL_DIR)
    if not os.path.exists(LABEL_FILE):
        st.error("Không tìm thấy labels.pkl trong thư mục models/.")
        st.stop()
    labels = joblib.load(LABEL_FILE)
    return model, labels

infer, LABELS = load_infer_and_labels()

RECYCLE_INSTRUCTIONS = {
    "plastic": "Rửa sạch và bỏ vào thùng tái chế màu <strong>vàng</strong> dành cho nhựa. Ở Việt Nam, thùng nhựa thường có màu vàng hoặc xanh dương.",
    "paper": "Rửa sạch, gấp phẳng và bỏ vào thùng tái chế màu <strong>xanh dương</strong> dành cho giấy & carton. Ở Việt Nam, thùng giấy thường có màu xanh dương hoặc xanh lá.",
    "glass": "Rửa sạch và bỏ vào thùng tái chế màu <strong>xanh dương</strong> dành cho thủy tinh. Tránh vỡ để dễ tái chế.",
    "metal": "Rửa sạch và bỏ vào thùng tái chế màu <strong>xám</strong> dành cho kim loại. Ở Việt Nam, thùng kim loại thường có màu xám hoặc xanh.",
    "organic": "Bỏ vào thùng hữu cơ màu <strong>nâu</strong> hoặc ủ phân tại nhà. Tránh trộn với rác không phân hủy.",
    "others": "Bỏ vào thùng rác thông thường màu <strong>đen</strong>. Cố gắng giảm thiểu loại rác này."
}

ENV_TIPS = {
    "plastic": "Tái chế 1 tấn nhựa giúp tiết kiệm năng lượng tương đương 700 lít dầu và giảm khí thải CO2.",
    "paper": "Tái chế 1 tấn giấy/cartons giúp tiết kiệm khoảng 17 cây trưởng thành và giảm lượng nước thải đáng kể!",
    "glass": "Tái chế thủy tinh tiết kiệm 30% năng lượng so với sản xuất mới và có thể tái chế vô hạn.",
    "metal": "Tái chế kim loại giảm 95% năng lượng khai thác và giảm ô nhiễm không khí.",
    "organic": "Ủ phân hữu cơ giảm khí metan từ bãi rác và tạo phân bón tự nhiên cho cây trồng.",
    "others": "Giảm rác không tái chế giúp bảo vệ đại dương và động vật hoang dã khỏi ô nhiễm nhựa."
}

# ==============================
# 🎨 CUSTOM COMPONENTS
# ==============================
def intro_box(text: str):
    st.markdown(
        f"""
        <div style="background-color:#ffffdd; padding:20px; border-radius:10px; border-left:6px solid #e6d784; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def result_box(label: str, conf: float):
    lower_label = label.lower()
    instruction = RECYCLE_INSTRUCTIONS.get(lower_label, "Không có hướng dẫn cụ thể.")
    tip = ENV_TIPS.get(lower_label, "Không có mẹo cụ thể.")

    if conf > 0.8:
        badge_color = "#28a745"
        badge_text = "Cao"
    elif conf > 0.5:
        badge_color = "#ffc107"
        badge_text = "Trung bình"
    else:
        badge_color = "#dc3545"
        badge_text = "Thấp"

    st.markdown(
        f"""
        <div style="background-color:#f0fff4; padding:20px; border-radius:10px; border-left:6px solid #28a745; box-shadow:0 4px 10px rgba(0,0,0,0.1);">
            <h2 style='text-align:center; color:#2b6f3e; margin:0;'>{label}</h2>
            <p style='text-align:center; font-size:18px; margin:10px 0;'>
                {round(conf * 100, 1)}% tự tin
                <span style='background-color:{badge_color}; color:white; padding:4px 12px; border-radius:20px; font-size:14px;'>{badge_text}</span>
            </p>
            <h4 style='color:#2b6f3e; margin-top:20px;'>Hướng dẫn tái chế:</h4>
            <p style='font-size:16px;'>{instruction}</p>
            <h4 style='color:#2b6f3e; margin-top:15px;'>Mẹo môi trường:</h4>
            <p style='font-size:16px;'><em>{tip}</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Thử ảnh khác", use_container_width=True):
        st.experimental_rerun()
    st.markdown(
        """
        <div style="text-align:center; margin-top:20px; color:#666; font-size:14px;">
            Dự đoán sai? 
            <a href="https://forms.gle/EXAMPLE" target="_blank">Báo cáo để cải thiện mô hình!</a> 🙏
        </div>
        """,
        unsafe_allow_html=True
    )

# ==============================
# 🔶 PREDICTION FUNCTION
# ==============================
def predict_image(pil_img: Image.Image):
    img = pil_img.resize((224, 224))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = mobilenetv2.preprocess_input(arr)
    probs = infer.predict(arr)[0]
    idx = np.argmax(probs)
    conf = probs[idx]
    return LABELS[idx], conf

# ==============================
# 📊 MAIN PAGE CONTENT
# ==============================
def show():
    intro_box(
        """
        Trang này cung cấp phân tích dữ liệu và demo phân loại hình ảnh.  
        Chúng tôi sẽ hiển thị thống kê dataset, ảnh mẫu ngẫu nhiên, và công cụ upload ảnh để phân loại rác thải.
        """
    )

    # PART 1: DATASET STATISTICS
    st.markdown("### 1. Thống kê Dataset")
    data_dir = "images_raw"
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])  # FIX: Chỉ lấy directories
    stats = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}
    st.table(stats)

    # PART 2: RANDOM SAMPLE IMAGES
    st.markdown("### 2. Ảnh Mẫu Ngẫu Nhiên")
    cols = st.columns(3)
    for i, cls in enumerate(classes):
        imgs = os.listdir(os.path.join(data_dir, cls))
        if imgs:
            sample_img = random.choice(imgs)
            cols[i % 3].image(os.path.join(data_dir, cls, sample_img), caption=cls.capitalize(), use_column_width=True)

    # PART 3: IMAGE CLASSIFICATION DEMO
    st.markdown("### 3. Demo Phân Loại Hình Ảnh")
    uploaded_files = st.file_uploader("Chọn ảnh để phân loại", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown(f"#### Ảnh: {uploaded_file.name}")
            pil_img = Image.open(uploaded_file).convert("RGB")
            st.image(pil_img, caption="Ảnh đã upload", use_column_width=False, width=300)

            if st.button(f"Predict {uploaded_file.name}", use_container_width=True):
                predicted_class, confidence = predict_image(pil_img)
                result_box(predicted_class, confidence)

if __name__ == "__main__":
    show()
