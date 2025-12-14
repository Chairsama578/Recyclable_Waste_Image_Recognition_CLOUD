import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib


# ==============================
#  🔧 CONFIG
# ==============================
MODEL_DIR = "models/waste_model"   # thư mục SavedModel (model.export)
LABEL_FILE = "models/labels.pkl"   # file nhãn


# ==============================
#  🔶 STYLE BOX
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


# ==============================
#  🔶 LOAD SAVEDMODEL + LABELS
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
#  🔶 HÀM DỰ ĐOÁN AUTO-KERAS
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
#  🔶 TRANG ANALYSIS
# ==============================
def show():

    st.markdown(
        "<h2 style='color:#2b6f3e;'>Analysis – Phân tích dữ liệu & Demo phân loại ảnh (AutoKeras SavedModel)</h2>",
        unsafe_allow_html=True,
    )

    dataset_path = "images_raw"

    # ------------------------------
    # 1. THỐNG KÊ DATASET
    # ------------------------------
    intro_box("""
    <h3 style="color:#b30000;">1. Thống kê dataset</h3>
    Hệ thống tự động đọc thư mục <b>images_raw/</b> và thống kê số lượng ảnh của từng lớp rác.
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
    <h3 style="color:#b30000;">2. Ảnh mẫu ngẫu nhiên trong dataset</h3>
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
    <h3 style="color:#b30000;">3. Demo phân loại ảnh bằng AutoKeras SavedModel</h3>
    Tải lên một hoặc nhiều ảnh, hệ thống sẽ dự đoán lớp rác tương ứng.
    """)

    uploaded_files = st.file_uploader(
        "📤 Chọn ảnh để phân loại",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for file in uploaded_files:
            st.subheader(f"Ảnh: {file.name}")

            img = Image.open(file).convert("RGB")
            st.image(img, width=250, caption="Ảnh tải lên")

            if st.button(f"🔍 Predict {file.name}"):
                label, conf = predict_image(img)
                st.success("Kết quả dự đoán:")
                st.json({
                    "prediction": label,
                    "confidence": round(conf, 4),
                })
            st.write("---")
