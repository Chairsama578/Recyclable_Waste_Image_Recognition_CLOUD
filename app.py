import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import mobilenet_v2 as mobilenetv2
# ================================
# 🔧 CẤU HÌNH TRANG (Must be first!)
# ================================
st.set_page_config(
    page_title="Waste Classification – Detect Recyclable Waste with AI.",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stFileUploader label {
        background-color: #2b6f3e;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        cursor: pointer;
    }
    .stFileUploader div {
        text-align: center;
    }
    .stImage {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================
# 🎨 HEADER
# ================================
with st.container():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image("https://asset-2.tstatic.net/jogja/foto/bank/images/Sampah-dan-pengelolaannya.jpg", width=110)
    with col2:
        st.markdown(
            """
            <h2 style='text-align:center; color:#2b6f3e;'>
                Waste Classification – Sort Recyclable Waste with AI.
            </h2>
            <h4 style='text-align:center; color:#4b4b4b;'>
                Upload a photo of any waste item and get instant classification + recycling instructions.
            </h4>
            """,
            unsafe_allow_html=True
        )
    with col3:
        pass

st.write("---")

# ================================
# 🧭 SIDEBAR NAVIGATION
# ================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Introduction Topic",
        "Analysis – Analysis Data & Demo Classification",
        "Training Info – AutoML Training Information"
    ]
)

# ================================
# 🔶 CONFIG MODEL (THÊM MỚI)
# ================================
MODEL_DIR = "models/waste_mobilenetv2.keras"
LABEL_FILE = "models/labels.pkl"

# Dictionary hướng dẫn và tip (dịch VN để khớp Home)
RECYCLE_INSTRUCTIONS_VN = {
    "plastic": "Rửa sạch và bỏ vào thùng tái chế màu <strong>vàng</strong> dành cho nhựa. Ở Việt Nam, thùng nhựa thường có màu vàng hoặc xanh dương.",
    "paper": "Rửa sạch, gấp phẳng và bỏ vào thùng tái chế màu <strong>xanh dương</strong> dành cho giấy & carton. Ở Việt Nam, thùng giấy thường có màu xanh dương hoặc xanh lá.",
    "glass": "Rửa sạch và bỏ vào thùng tái chế màu <strong>xanh dương</strong> dành cho thủy tinh. Tránh vỡ để dễ tái chế.",
    "metal": "Rửa sạch và bỏ vào thùng tái chế màu <strong>xám</strong> dành cho kim loại. Ở Việt Nam, thùng kim loại thường có màu xám hoặc xanh.",
    "organic": "Bỏ vào thùng hữu cơ màu <strong>nâu</strong> hoặc ủ phân tại nhà. Tránh trộn với rác không phân hủy.",
    "others": "Bỏ vào thùng rác thông thường màu <strong>đen</strong>. Cố gắng giảm thiểu loại rác này."
}

ENV_TIPS_VN = {
    "plastic": "Tái chế 1 tấn nhựa giúp tiết kiệm năng lượng tương đương 700 lít dầu và giảm khí thải CO2.",
    "paper": "Tái chế 1 tấn giấy/cartons giúp tiết kiệm khoảng 17 cây trưởng thành và giảm lượng nước thải đáng kể!",
    "glass": "Tái chế thủy tinh tiết kiệm 30% năng lượng so với sản xuất mới và có thể tái chế vô hạn.",
    "metal": "Tái chế kim loại giảm 95% năng lượng khai thác và giảm ô nhiễm không khí.",
    "organic": "Ủ phân hữu cơ giảm khí metan từ bãi rác và tạo phân bón tự nhiên cho cây trồng.",
    "others": "Giảm rác không tái chế giúp bảo vệ đại dương và động vật hoang dã khỏi ô nhiễm nhựa."
}

# ================================
# 🔶 HÀM LOAD MODEL + LABELS (COPY TỪ ANALYSIS)
# ================================
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

# ================================
# 🔶 HÀM DỰ ĐOÁN (FIX PREPROCESS)
# ================================
def predict_image(pil_img: Image.Image):
    img = pil_img.resize((224, 224))
    arr = np.array(img)  # (224,224,3) uint8 [0,255]
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    arr = mobilenetv2.preprocess_input(arr)  # Scale về [-1,1] cho MobileNetV2
    probs = infer.predict(arr)[0]
    idx = np.argmax(probs)
    conf = probs[idx]
    return LABELS[idx], conf
# ================================
# 🔶 HÀM RESULT_BOX (COPY VÀ ĐIỀU CHỈNH TỪ ANALYSIS)
# ================================
def result_box(label: str, conf: float):
    lower_label = label.lower()
    instruction = RECYCLE_INSTRUCTIONS_VN.get(lower_label, "Không có hướng dẫn cụ thể.")
    tip = ENV_TIPS_VN.get(lower_label, "Không có mẹo cụ thể.")
    
    # Map label sang tiếng Việt (để khớp placeholder cũ)
    vn_label_map = {
        "plastic": "Nhựa (Plastic)",
        "paper": "Giấy (Paper/Cardboard)",
        "glass": "Thủy tinh (Glass)",
        "metal": "Kim loại (Metal)",
        "organic": "Hữu cơ (Organic)",
        "others": "Khác (Others)"
    }
    vn_label = vn_label_map.get(lower_label, label)
    
    if conf > 0.8:
        badge_color = "#28a745"
        badge_text = "Rất cao"
    elif conf > 0.5:
        badge_color = "#ffc107"
        badge_text = "Trung bình"
    else:
        badge_color = "#dc3545"
        badge_text = "Thấp"

    st.markdown(
        f"""
        <div style="background:#f8fff9; padding:25px; border-radius:15px; border-left:8px solid #2b6f3e; box-shadow:0 6px 15px rgba(0,0,0,0.1);">
            <h1 style='text-align:center; color:#2b6f3e; margin:0;'>{vn_label}</h1>
            <p style='text-align:center; font-size:20px; margin:15px 0;'>
                Confidence: <strong>{round(conf * 100)}%</strong> 
                <span style='background:{badge_color}; color:white; padding:6px 15px; border-radius:30px; font-size:16px;'>{badge_text}</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(f"<h4 style='color:#2b6f3e;'>Hướng dẫn tái chế ở Việt Nam:</h4><p style='font-size:16px; line-height:1.6;'>{instruction}</p>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:#2b6f3e;'>Mẹo môi trường:</h4><p style='font-size:16px;'><em>{tip}</em></p>", unsafe_allow_html=True)

# ================================
# 📌 ROUTING ĐẾN TRANG
# ================================
if page.startswith("Home"):
    st.markdown("""
    The Waste Classification app is an intelligent web-based system designed to automatically identify and classify common recyclable waste materials into five primary categories: **paper**, **plastic**, **metal**, **glass**, and **organic**.
    
    Built using open-source tools like TensorFlow, Keras, and Streamlit, it provides instant analysis and practical recycling instructions tailored to local practices in Vietnam. 🌍♻️
    """)

    st.markdown("### Description for the Type of Image")
    st.markdown("""
    - The system can process photographs of everyday waste items, such as crumpled paper, plastic bottles, metal cans, glass jars, or organic scraps like food waste.
    - For best results, ensure the photo is clear, well-lit, and focuses on a single item against a simple background.
    - Supported formats: JPG, JPEG, PNG.
    """)

    st.markdown("### Brief Instructions")
    st.markdown("""
    - **Supported categories:** Paper, Plastic, Metal, Glass, Organic.
    - Upload your photo below to get started!
    - If the confidence score is below 70%, try retaking the photo from a different angle.
    """)

    # Upload button
    cols = st.columns(3)
    with cols[1]:
        uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg"], label_visibility="visible")

    if uploaded_file is not None:
        # ========================
        # KẾT QUẢ PHÂN LOẠI (SỬ DỤNG MODEL THẬT)
        # ========================
        st.markdown("<h2 style='text-align:center; color:#2b6f3e; margin-top:30px;'>Kết Quả Phân Loại Rác</h2>", unsafe_allow_html=True)
        
        col_left, col_right = st.columns([1, 1.2])
        
        with col_left:
            st.image(uploaded_file, caption="Ảnh vật thải đã upload", use_column_width=True)
            st.markdown("<div style='text-align:center; margin-top:10px;'><small>Click vào ảnh để zoom</small></div>", unsafe_allow_html=True)
        
        with col_right:
            # Dự đoán thật sử dụng model
            pil_img = Image.open(uploaded_file).convert("RGB")
            predicted_class, confidence = predict_image(pil_img)
            confidence = confidence  # Đã là float 0-1, nhân 100 ở display
            
            # Hiển thị với result_box
            result_box(predicted_class, confidence)
        
        # Ảnh minh họa thùng rác (giữ nguyên)
        st.markdown("<div style='margin:30px 0; text-align:center;'>", unsafe_allow_html=True)
        st.image("https://asset-2.tstatic.net/jogja/foto/bank/images/Sampah-dan-pengelolaannya.jpg", use_column_width=True)
        st.markdown("<p style='font-size:12px; color:#666; margin-top:8px; text-align:center;'>Minh họa các thùng phân loại rác tái chế</p></div>", unsafe_allow_html=True)
        
        # Nút thử lại
        st.markdown("<br>", unsafe_allow_html=True)
        btn_cols = st.columns([1, 2, 1])
        with btn_cols[1]:
            if st.button("Try Another Photo", use_container_width=True, type="primary"):
                st.experimental_rerun()
        
        # Báo cáo sai
        st.markdown("""
        <div style="text-align:center; margin-top:30px; color:#666; font-size:14px;">
            Incorrect prediction? 
            <a href="mailto:nganvhk22@uef.edu.vn?subject=Report%20Wrong%20Prediction%20-%20Waste%20Classification">
                Report it to us to improve the model! 🙏
            </a>
        </div>
        """, unsafe_allow_html=True)

elif page.startswith("Analysis"):
    try:
        from pages.Analysis import show
        show()
    except ImportError:
        st.error("Page Analysis not found.")

elif page.startswith("Training Info"):
    try:
        from pages.Training_Info import show
        show()
    except ImportError:
        st.error("Page Training Info not found.")

# ================================
# 📝 FOOTER
# ================================
st.write("---")
st.markdown(
    """
    <div style="padding:18px; background:#ffffdd; border-radius:10px; border:1px solid #e6d784; margin-bottom:10px;">
        <b>Students:</b><br>
        - Student 1: Võ Hoàng Kim Ngân - Email: <a href="mailto:nganvhk22@uef.edu.vn">nganvhk22@uef.edu.vn</a><br>
        - Student 2: Nhan Gia Huy - Email: <a href="mailto:huyng222@uef.edu.vn">huyng222@uef.edu.vn</a><br><br>
        <b>GitHub Repository:</b> <a href="https://github.com/Chairsama578/Recyclable_Waste_Image_Recognition_CLOUD">Link to Repository</a><br><br>
        <b>Contact:</b> For inquiries, email the students above.
    </div>
    """,
    unsafe_allow_html=True
)
