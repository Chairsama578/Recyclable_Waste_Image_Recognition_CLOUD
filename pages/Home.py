import streamlit as st

# ==========================
# 🎨 HỘP HIỂN THỊ NỘI DUNG
# ==========================
def intro_box(text):
    st.markdown(f"""
        <div style="
            background-color:#fff7cc;
            padding:20px;
            border-radius:10px;
            border:1px solid #e6d784;
            font-size:18px;
            line-height:1.7;
        ">
        {text}
        </div>
    """, unsafe_allow_html=True)


# ==========================
# 🎯 TRANG HOME
# ==========================
def show():

    st.markdown(
        "<h3 style='color:#2b6f3e;'>Introduction Topic</h3>",
        unsafe_allow_html=True
    )

    # ====== MỤC 1 ======
    intro_box("""
    <h3 style="color:#b30000;">1. Context and Reasons for Choosing the Topic </h3>
    The issue of waste management and classification plays an important role in environmental protection, 
    especially in large urban areas where the amount of household waste is rapidly increasing. 
    Manual waste sorting is often time-consuming, inaccurate, and labor-intensive.

    The development of Artificial Intelligence, particularly Google's <b>AutoML Vision</b> technology, allows 
    for the automatic creation of image recognition models without complex programming. 
    This enables students to effectively and practically implement waste classification models.
    """)

    # ====== MỤC 2 ======
    intro_box("""
    <h3 style="color:#b30000;">2. Mục tiêu Đề tài</h3>

    Mục tiêu chính của đề tài:
    <ul>
        <li>Xây dựng hệ thống nhận diện hình ảnh rác tái chế sử dụng Google AutoML Vision.</li>
        <li>Phân loại tự động các loại rác phổ biến:</li>
    </ul>

    <ul style="margin-left:30px;">
        <li>Plastic (Plastic)</li>
        <li>Paper (Paper)</li>
        <li>Glass (Glass)</li>
        <li>Metal (Metal)</li>
        <li>Organic (Organic)</li>
        <li>Others (Other)</li>
    </ul>

    After training, the system will be integrated into the Streamlit web application to demonstrate its waste classification capabilities.
    This is an important step towards <b>an automatic waste sorting solution (Automated Waste Sorting System)</b>.
    """)

    # ====== MỤC 3 ======
    intro_box("""
    <h3 style="color:#b30000;">3. Phạm vi và Nội dung thực hiện</h3>

    <ul>
        <li>Collect and standardize waste image data.</li>
        <li>Prepare the dataset structure according to the AutoML Vision standard.</li>
        <li>Training a waste classification model using AutoML Vision.</li>
        <li>Evaluating the model through various metrics: Accuracy, Precision, Recall, F1-score.</li>
        <li>Deploy the predictive model within the Streamlit interface.</li>
        <li>Propose an automated waste classification process based on the developed model.</li>
    </ul>
    """)

    # ====== MỤC 4 ======
    intro_box("""
    <h3 style="color:#b30000;">4. Ý nghĩa khoa học và thực tiễn</h3>

    <ul>
        <li>Applying AI to waste sorting – a field with significant social impact.</li>
        <li>Reducing the burden on sanitation workers.</li>
        <li>Increasing recycling rates through precise identification.</li>
        <li>Has the potential to develop into an automatic waste sorting system in smart cities.</li>
    </ul>

    The topic is highly applicable and aligns with the digital transformation trends in the environmental sector.
    """)

