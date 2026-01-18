import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model  # Import để load .keras


# ==============================
#  CONFIG
# ==============================
DATA_DIR = "images_raw"
MODEL_DIR = "models/waste_mobilenetv2.keras"
LABEL_FILE = "models/labels.pkl"
AUG_DIR = "images_augmented"

# ==============================
#  STYLE BOX
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
#  LOAD MODEL + LABELS
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


infer, LABELS = load_infer_and_labels()


 # Bây giờ infer là model

def predict_image(pil_img: Image.Image):
    img = pil_img.resize((224, 224))
    arr = np.array(img) / 255.0  # Normalize nếu cần (tùy model)
    arr = np.expand_dims(arr, axis=0)
    # Dự đoán bằng Keras model
    probs = infer.predict(arr)[0]  # infer giờ là model
    idx = np.argmax(probs)
    conf = probs[idx]
    return LABELS[idx], conf  # Sửa: Bỏ conf lặp


# Thêm hàm predict_path (vì được gọi trong evaluation nhưng chưa định nghĩa)
def predict_path(path: str):
    img = Image.open(path).convert("RGB")
    return predict_image(img)


# ==============================
#  PAGE
# ==============================
def show():
    st.markdown(
        "<h2 style='color:#2b6f3e;'>Training Info – AutoKeras Training Overview</h2>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------
    # 1. Display raw data
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">1. Display Raw Dataset</h3>
        The original dataset is stored in the <b>images_raw/</b> directory and includes the following classes:
        <code>glass, metal, organic, others, paper, plastic</code>.
        The system counts the number of original images for each class.
        """
    )

    if not os.path.exists(DATA_DIR):
        st.error(" images_raw/ directory not found.")
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

    st.write("** Number of original images (before augmentation):**")
    st.table({"Class": list(raw_stats.keys()), "Original Images": list(raw_stats.values())})

    st.write("---")

    # -------------------------------------------------------
    # 2. Data preprocessing & augmentation
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">2. Data Preprocessing & Augmentation</h3>
        All images are <b>resized to 224×224</b>. Additional augmented images are generated
        (rotation, flipping, brightness adjustment, noise, etc.).
        Augmented images are saved with filenames starting with <code>aug_*.jpg</code>.
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

    st.write("** Dataset size after augmentation:**")
    st.table(
        {
            "Class": classes,
            "Original Images": [raw_stats.get(c, 0) for c in classes],
            "Augmented Images (aug_*)": [aug_stats.get(c, 0) for c in classes],
            "Total Images": [total_stats.get(c, 0) for c in classes],
        }
    )

    st.write("---")

    # -------------------------------------------------------
    # 3. Model storage path
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">3. Trained Model Storage Path</h3>
        The best model selected by AutoKeras is exported in
        <b>SavedModel</b> format and stored at:
        """
    )

    st.code(
        """
models/
    waste_model/      # SavedModel exported from AutoKeras
        saved_model.pb
        variables/
        assets/
    labels.pkl        # Class labels in softmax index order
""",
        language="text",
    )

    st.write("---")

    # -------------------------------------------------------
    # 4. Model signature info
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">4. SavedModel Information</h3>
        Below is the input/output information of the
        <code>serving_default</code> signature used for inference.
        """
    )

    st.write("** Input signature:**")
    st.code(str(infer.input_shape), language="text")  # Sửa cho Keras model: Hiển thị input shape thay vì signature

    st.write("** Output signature:**")
    st.code(str(infer.output_shape), language="text")  # Sửa cho Keras model: Hiển thị output shape

    # -------------------------------------------------------
    # 5–7. Training results & quick evaluation
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">5–7. Training Results & Model Reliability Evaluation</h3>
        For demonstration purposes, the system performs a <b>quick evaluation</b>
        on the entire dataset (including original and augmented images) to compute:
        <ul>
            <li>Accuracy per class and overall accuracy.</li>
            <li>Mean confidence of correct predictions.</li>
        </ul>
        Note: This is only a reference evaluation and does not replace
        testing on an independent test set.
        """
    )

    if st.button(" Run quick evaluation on dataset"):
        # Lọc classes chỉ những lớp có trong LABELS (bỏ 'others')
        filtered_classes = [c for c in classes if c in LABELS]
        
        per_class_total = {c: 0 for c in filtered_classes}
        per_class_correct = {c: 0 for c in filtered_classes}
        per_class_conf_sum = {c: 0.0 for c in filtered_classes}

        image_paths = []

        for cls in filtered_classes:  # Chỉ dùng filtered_classes
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
            pred_cls, conf = predict_path(path)  # Sửa: Chỉ return 2 giá trị

            per_class_total[true_cls] += 1
            if pred_cls == true_cls:
                per_class_correct[true_cls] += 1
                per_class_conf_sum[true_cls] += conf

            progress.progress(i / n)

        rows = []
        total_correct = 0
        total_images = 0

        for cls in filtered_classes:  # Dùng filtered_classes
            total = per_class_total[cls]
            correct = per_class_correct[cls]
            acc = correct / total * 100 if total > 0 else 0.0
            mean_conf = per_class_conf_sum[cls] / correct if correct > 0 else 0.0

            rows.append(
                {
                    "Class": cls,
                    "Images": total,
                    "Correct Predictions": correct,
                    "Accuracy (%)": round(acc, 2),
                    "Mean Confidence (Correct)": round(mean_conf, 4),
                }
            )

            total_correct += correct
            total_images += total

        st.write("** Per-class evaluation results:**")
        st.dataframe(rows, hide_index=True)

        if total_images > 0:
            overall_acc = total_correct / total_images * 100
            st.success(
                f" Overall accuracy on the entire dataset: **{overall_acc:.2f}%**"
            )

    st.write("---")

    # -------------------------------------------------------
    # 8. Model comparison suggestion
    # -------------------------------------------------------
    yellow_box(
        """
        <h3 style="color:#b30000;">8. Comparison with Other Models</h3>
        In this project, AutoKeras automatically explores multiple CNN architectures
        (ResNet, Xception, etc.) and selects the best-performing model.
        <br><br>
        For further study, students can:
        <ul>
            <li><b>8.1 Train a manual CNN model</b> (pure Keras).</li>
            <li><b>8.2 Compare accuracy, training time, and model size</b> between AutoKeras and the manual CNN.</li>
        </ul>
        """
    )
