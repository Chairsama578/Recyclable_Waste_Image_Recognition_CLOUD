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
DATA_DIR = "images_raw"
MODEL_DIR = "models/waste_model"
LABEL_FILE = "models/labels.pkl"


# ==============================
#  🔶 STYLE BOX
# ==============================
def yellow_box(text: str):
    st.markdown(
        f"""
        <div style="
            background-color:#fff7cc;
            padding:18px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:17px;
            line-height:1.6;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
#  🔶 LOAD MODEL + LABELS
# ==============================
@st.cache_resource
def load_infer_and_labels():
    if not os.path.exists(MODEL_DIR):
        st.error("❌ Không tìm thấy SavedModel trong thư mục models/waste_model.")
        st.stop()

    model = tf.saved_model.load(MODEL_DIR)
    infer_fn = model.signatures["serving_default"]

    if not os.path.exists(LABEL_FILE):
        st.error("❌ Không tìm thấy models/labels.pkl.")
        st.stop()

    labels = joblib.load(LABEL_FILE)
    return infer_fn, labels


infer, LABELS = load_infer_and_labels()


def predict_path(img_path: str):
    """Dự đoán 1 ảnh theo đường dẫn (dùng cho phần đánh giá)."""

    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))

    arr = np.array(img, dtype=np.uint8)
    arr = np.expand_dims(arr, axis=0)
    tensor = tf.convert_to_tensor(arr, dtype=tf.uint8)

    out = infer(tensor)
    probs = list(out.values())[0].numpy()[0]

    idx = int(np.argmax(probs))
    conf = float(probs[idx])

    return LABELS[idx], conf


# ==============================
#  🔶 PAGE
# ==============================
def show():
    st.markdown(
        "<h2 style='color:#2b6f3e;'>Training Info – Thông tin huấn luyện AutoKeras</h2>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------
    # 1. Hiện dữ liệu thô
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">1. Hiện dữ liệu thô</h3>
        Dataset gốc được lưu trong thư mục <b>images_raw/</b>, gồm các lớp:
        <code>glass, metal, organic, others, paper, plastic</code>.
        Hệ thống sẽ thống kê số lượng ảnh ban đầu của từng lớp.
        """
    )

    if not os.path.exists(DATA_DIR):
        st.error("⚠ Không tìm thấy thư mục images_raw/.")
        return

    raw_stats = {}
    classes = sorted(
        [c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))]
    )

    for cls in classes:
        folder = os.path.join(DATA_DIR, cls)
        files = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith("aug_")
        ]
        raw_stats[cls] = len(files)

    st.write("**📊 Số lượng ảnh gốc (chưa augment):**")
    st.table({"Lớp": list(raw_stats.keys()), "Số ảnh gốc": list(raw_stats.values())})

    st.write("---")

    # -------------------------------------------------------
    # 2. Hiện xử lý dữ liệu thô đã được xử lý (Tiền xử lý dữ liệu)
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">2. Tiền xử lý dữ liệu & Augmentation</h3>
        Các ảnh được <b>resize về 224×224</b> và lưu thêm các phiên bản augment
        (xoay, lật, thay đổi độ sáng, thêm nhiễu, ...). Các ảnh augment được đặt
        tên bắt đầu bằng <code>aug_*.jpg</code>.
        """
    )

    aug_stats = {}
    total_stats = {}

    for cls in classes:
        folder = os.path.join(DATA_DIR, cls)
        all_imgs = [
            f
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        aug_imgs = [f for f in all_imgs if f.startswith("aug_")]
        aug_stats[cls] = len(aug_imgs)
        total_stats[cls] = len(all_imgs)

    st.write("**📊 Số lượng ảnh sau khi augment:**")
    st.table(
        {
            "Lớp": classes,
            "Ảnh gốc": [raw_stats.get(c, 0) for c in classes],
            "Ảnh augment (aug_*)": [aug_stats.get(c, 0) for c in classes],
            "Tổng ảnh": [total_stats.get(c, 0) for c in classes],
        }
    )

    st.write("---")

    # -------------------------------------------------------
    # 3. Hiện đường dẫn tương đối lưu model
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">3. Đường dẫn lưu mô hình đã huấn luyện</h3>
        Mô hình tốt nhất do AutoKeras chọn được export theo định dạng
        <b>SavedModel</b> và lưu tại:
        """
    )

    st.code(
        f"""
models/
    waste_model/      # SavedModel (export từ AutoKeras)
        saved_model.pb
        variables/
        assets/
    labels.pkl        # Danh sách nhãn theo thứ tự index softmax
""",
        language="text",
    )

    st.write("---")

    # -------------------------------------------------------
    # 4. Đọc thông tin model object
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">4. Thông tin về mô hình SavedModel</h3>
        Dưới đây là thông tin input/output của signature
        <code>serving_default</code> trong SavedModel, dùng cho việc suy luận
        (inference) trong ứng dụng.
        """
    )

    st.write("**📥 Input signature:**")
    st.code(str(infer.structured_input_signature), language="text")

    st.write("**📤 Output signature:**")
    st.code(str(infer.structured_outputs), language="text")

    # -------------------------------------------------------
    # 5–7. Kết quả train & đánh giá độ tin cậy (đánh giá nhanh trên dataset)
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">5–7. Kết quả train & Đánh giá độ tin cậy mô hình</h3>
        Để minh họa, hệ thống sẽ chạy <b>đánh giá nhanh</b> trên toàn bộ
        dataset hiện có (gồm cả ảnh gốc và ảnh augment) và tính:
        <ul>
            <li>Độ chính xác (accuracy) theo từng lớp và toàn bộ.</li>
            <li>Độ tin cậy trung bình (mean confidence) của các dự đoán đúng.</li>
        </ul>
        Lưu ý: đây chỉ là đánh giá tham khảo, không thay thế cho đánh giá trên
        tập kiểm tra độc lập.
        """
    )

    if st.button("▶ Chạy đánh giá nhanh trên dataset"):
        per_class_total = {c: 0 for c in classes}
        per_class_correct = {c: 0 for c in classes}
        per_class_conf_sum = {c: 0.0 for c in classes}

        image_paths = []

        for cls in classes:
            folder = os.path.join(DATA_DIR, cls)
            files = [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for f in files:
                image_paths.append((cls, os.path.join(folder, f)))

        progress = st.progress(0.0)
        n = len(image_paths)

        for i, (true_cls, path) in enumerate(image_paths, start=1):
            pred_cls, conf = predict_path(path)

            per_class_total[true_cls] += 1
            if pred_cls == true_cls:
                per_class_correct[true_cls] += 1
                per_class_conf_sum[true_cls] += conf

            progress.progress(i / n)

        # Tính bảng kết quả
        rows = []
        total_correct = 0
        total_images = 0

        for cls in classes:
            total = per_class_total[cls]
            correct = per_class_correct[cls]
            acc = correct / total * 100 if total > 0 else 0.0
            mean_conf = per_class_conf_sum[cls] / correct if correct > 0 else 0.0

            rows.append(
                {
                    "Lớp": cls,
                    "Số ảnh": total,
                    "Dự đoán đúng": correct,
                    "Accuracy (%)": round(acc, 2),
                    "Mean confidence (đúng)": round(mean_conf, 4),
                }
            )

            total_correct += correct
            total_images += total

        st.write("**📊 Kết quả theo từng lớp:**")
        st.dataframe(rows, hide_index=True)

        if total_images > 0:
            overall_acc = total_correct / total_images * 100
            st.success(
                f"🎯 Độ chính xác tổng thể trên toàn bộ dataset: **{overall_acc:.2f}%**"
            )

    st.write("---")

    # -------------------------------------------------------
    # 8. Gợi ý so sánh với các mô hình khác
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">8. So sánh kết quả với các mô hình khác</h3>
        Trong đề tài này, AutoKeras đã tự động thử nhiều kiến trúc CNN khác nhau
        (ResNet, Xception, v.v.) và chọn ra mô hình có độ chính xác cao nhất.
        <br><br>
        Để mở rộng, sinh viên có thể:
        <ul>
            <li><b>8.1 Huấn luyện thêm một mô hình thủ công</b> (ví dụ: CNN thuần Keras).</li>
            <li><b>8.2 So sánh accuracy, thời gian train, kích thước mô hình</b> giữa AutoKeras và CNN thủ công.</li>
        </ul>
        """
    )

    # 8.1 – Ví dụ code CNN thuần Keras
    st.markdown("### 8.1 Huấn luyện thêm một mô hình CNN thuần Keras (minh hoạ)")

    with st.expander("📌 Xem ví dụ code CNN thuần Keras"):
        st.code(
            """
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

IMG_SIZE = (224, 224)
NUM_CLASSES = 6  # glass, metal, organic, others, paper, plastic

# 1. Load ảnh thành numpy array (X) và nhãn (y) giống phần train_autokeras.py
#    Giả sử đã có X.shape = (N, 224, 224, 3), y là nhãn dạng số 0..5

# 2. Chuẩn hoá
X = X.astype("float32") / 255.0

# 3. Xây dựng CNN đơn giản
model = keras.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 4. Train mô hình
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)

# 5. Lưu model để so sánh kích thước với AutoKeras
model.save("models/manual_cnn.keras")  # hoặc .h5
            """,
            language="python",
        )

    st.markdown(
        """
        👉 Sinh viên có thể copy đoạn code trên ra file riêng
        (ví dụ <code>train_cnn_manual.py</code>), chỉnh sửa lại phần đọc dữ liệu
        giống với <code>train_autokeras.py</code> và chạy để thu được:
        <ul>
            <li>Accuracy trên tập validation/test.</li>
            <li>Thời gian huấn luyện (tổng thời gian chạy script).</li>
            <li>Kích thước file mô hình <code>manual_cnn.keras</code>.</li>
        </ul>
        """
    )

    # 8.2 – Bảng khung so sánh
    st.markdown("### 8.2 Khung so sánh AutoKeras vs CNN thuần Keras")

    st.write(
        """
        Sau khi huấn luyện xong cả hai mô hình, sinh viên ghi lại các số liệu
        (accuracy, thời gian train, kích thước file) và điền vào bảng dưới đây
        trong báo cáo. Ở ứng dụng demo, bảng chỉ mang tính minh họa.
        """
    )

    # Bảng khung (sinh viên tự cập nhật số liệu thật trong báo cáo)
    st.table(
        {
            "Mô hình": ["AutoKeras ImageClassifier", "CNN thuần Keras"],
            "Accuracy trên tập đánh giá (%)": ["...", "..."],
            "Thời gian train (phút)": ["...", "..."],
            "Kích thước file model (MB)": ["...", "..."],
        }
    )
