import streamlit as st
import os
import random
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
#  🔧 CONFIG
# ==============================
DATA_DIR = "images_raw"
MODEL_DIR = "models/waste_model"
LABEL_FILE = "models/labels.pkl"

# ==============================
#  🎨 IMPROVED STYLING
# ==============================
st.markdown("""
    <style>
    .main {
        background-color: #f0f8f5;
    }
    .stButton>button {
        background-color: #2b6f3e;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stDataFrame {
        border: 1px solid #ddd;
        border-radius: 8px;
        overflow: hidden;
    }
    .yellow-box {
        background-color: #fff7cc;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e6d784;
        margin-bottom: 20px;
    }
    h3 {
        color: #b30000;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
#  🔶 LOAD MODEL + LABELS
# ==============================
@st.cache_resource
def load_infer_and_labels():
    if not os.path.exists(MODEL_DIR):
        st.error("❌ SavedModel not found in models/waste_model directory.")
        st.stop()

    model = tf.saved_model.load(MODEL_DIR)
    infer_fn = model.signatures["serving_default"]

    if not os.path.exists(LABEL_FILE):
        st.error("❌ models/labels.pkl not found.")
        st.stop()

    labels = joblib.load(LABEL_FILE)
    return infer_fn, labels

infer, LABELS = load_infer_and_labels()

def predict_path(img_path: str):
    """Predict an image by its path (used for evaluation)."""

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
        "<h1 style='color:#2b6f3e; text-align: center;'>Training Info – AutoKeras Training Overview</h1>",
        unsafe_allow_html=True,
    )

    if not os.path.exists(DATA_DIR):
        st.error("⚠ images_raw/ directory not found.")
        return

    classes = sorted(
        [c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))]
    )

    # -------------------------------------------------------
    # 1. Display raw data
    # -------------------------------------------------------
    with st.expander("### 1. Display Raw Dataset", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            The original dataset is stored in the <b>images_raw/</b> directory and includes the following classes:
            <code>glass, metal, organic, others, paper, plastic</code>.
            The system counts the number of original images for each class.
            </div>
        """, unsafe_allow_html=True)

        raw_stats = {}
        for cls in classes:
            folder = os.path.join(DATA_DIR, cls)
            files = [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and not f.startswith("aug_")
            ]
            raw_stats[cls] = len(files)

        st.subheader("📊 Number of Original Images (Before Augmentation)")
        df_raw = pd.DataFrame({"Class": list(raw_stats.keys()), "Original Images": list(raw_stats.values())})
        st.dataframe(df_raw, use_container_width=True)

        # Add bar chart for visualization
        fig_raw, ax_raw = plt.subplots()
        sns.barplot(x="Class", y="Original Images", data=df_raw, ax=ax_raw, palette="Greens_d")
        ax_raw.set_title("Original Images per Class")
        ax_raw.set_ylabel("Count")
        st.pyplot(fig_raw)

    # -------------------------------------------------------
    # 2. Data preprocessing & augmentation
    # -------------------------------------------------------
    with st.expander("### 2. Data Preprocessing & Augmentation", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            All images are <b>resized to 224×224</b>. Additional augmented images are generated
            (rotation, flipping, brightness adjustment, noise, etc.).
            Augmented images are saved with filenames starting with <code>aug_*.jpg</code>.
            </div>
        """, unsafe_allow_html=True)

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

        st.subheader("📊 Dataset Size After Augmentation")
        df_aug = pd.DataFrame(
            {
                "Class": classes,
                "Original Images": [raw_stats.get(c, 0) for c in classes],
                "Augmented Images (aug_*)": [aug_stats.get(c, 0) for c in classes],
                "Total Images": [total_stats.get(c, 0) for c in classes],
            }
        )
        st.dataframe(df_aug, use_container_width=True)

        # Stacked bar chart
        fig_aug, ax_aug = plt.subplots()
        df_aug.plot(kind='bar', x='Class', stacked=True, ax=ax_aug, color=['#2b6f3e', '#b30000', '#e6d784'])
        ax_aug.set_title("Dataset Composition per Class")
        ax_aug.set_ylabel("Count")
        st.pyplot(fig_aug)

        # Display sample images
        st.subheader("🖼️ Sample Images from Dataset")
        cols = st.columns(3)
        for i, cls in enumerate(random.sample(classes, min(3, len(classes)))):
            folder = os.path.join(DATA_DIR, cls)
            files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if files:
                sample_img = random.choice(files)
                img_path = os.path.join(folder, sample_img)
                with cols[i]:
                    st.image(img_path, caption=f"Sample from {cls}", use_column_width=True)

    # -------------------------------------------------------
    # 3. Model storage path
    # -------------------------------------------------------
    with st.expander("### 3. Trained Model Storage Path", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            The best model selected by AutoKeras is exported in
            <b>SavedModel</b> format and stored at:
            </div>
        """, unsafe_allow_html=True)

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

    # -------------------------------------------------------
    # 4. Model signature info
    # -------------------------------------------------------
    with st.expander("### 4. SavedModel Information", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            Below is the input/output information of the
            <code>serving_default</code> signature used for inference.
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📥 Input Signature")
        st.code(str(infer.structured_input_signature), language="text")

        st.subheader("📤 Output Signature")
        st.code(str(infer.structured_outputs), language="text")

    # -------------------------------------------------------
    # 5–7. Training results & quick evaluation
    # -------------------------------------------------------
    with st.expander("### 5–7. Training Results & Model Reliability Evaluation", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            For demonstration purposes, the system performs a <b>quick evaluation</b>
            on the entire dataset (including original and augmented images) to compute:
            <ul>
                <li>Accuracy per class and overall accuracy.</li>
                <li>Mean confidence of correct predictions.</li>
            </ul>
            Note: This is only a reference evaluation and does not replace
            testing on an independent test set.
            </div>
        """, unsafe_allow_html=True)

        if st.button("▶ Run Quick Evaluation on Dataset"):
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
                        "Class": cls,
                        "Images": total,
                        "Correct Predictions": correct,
                        "Accuracy (%)": round(acc, 2),
                        "Mean Confidence (Correct)": round(mean_conf, 4),
                    }
                )

                total_correct += correct
                total_images += total

            st.subheader("📊 Per-Class Evaluation Results")
            df_eval = pd.DataFrame(rows)
            st.dataframe(df_eval, use_container_width=True)

            if total_images > 0:
                overall_acc = total_correct / total_images * 100
                st.success(
                    f"🎯 Overall accuracy on the entire dataset: **{overall_acc:.2f}%**"
                )

            # Add visualization for accuracy
            fig_eval, ax_eval = plt.subplots()
            sns.barplot(x="Class", y="Accuracy (%)", data=df_eval, ax=ax_eval, palette="Reds_d")
            ax_eval.set_title("Accuracy per Class")
            ax_eval.set_ylim(0, 100)
            st.pyplot(fig_eval)

    # -------------------------------------------------------
    # 8. Model comparison suggestion
    # -------------------------------------------------------
    with st.expander("### 8. Comparison with Other Models", expanded=True):
        st.markdown("""
            <div class='yellow-box'>
            In this project, AutoKeras automatically explores multiple CNN architectures
            (ResNet, Xception, etc.) and selects the best-performing model.
            <br><br>
            For further study, students can:
            <ul>
                <li><b>8.1 Train a manual CNN model</b> (pure Keras).</li>
                <li><b>8.2 Compare accuracy, training time, and model size</b> between AutoKeras and the manual CNN.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)

# Run the page
show()
