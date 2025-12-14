import streamlit as st
import os


# ================================
# 🔧 CẤU HÌNH TRANG
# ================================
st.set_page_config(
    page_title="Recyclable Waste Image Classification – AutoML Vision",
    layout="wide"
)


# ================================
# 🎨 HEADER (KHÔNG ghi chữ header)
# ================================
with st.container():
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        if os.path.exists("rose.png"):
            st.image("rose.png", width=110)

    with col2:
        st.markdown(
            """
            <h2 style='text-align:center; color:#2b6f3e;'>
                Topic 3: Recyclable Waste Image Recognition Using AutoML Vision
            </h2>
            <h4 style='text-align:center; color:#4b4b4b;'>
                Proposing an Automated Classification Solution
            </h4>
            """,
            unsafe_allow_html=True
        )

    with col3:
        pass  # Ô bên phải để trống như Topic 2

st.write("---")


# ================================
# 🧭 SIDEBAR NAVIGATION
# ================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to:",
    [
        "Home – Giới thiệu đề tài",
        "Analysis – Phân tích dữ liệu & Demo phân loại",
        "Training Info – Thông tin huấn luyện AutoML"
    ]
)


# ================================
# 📌 ROUTING ĐẾN TRANG TRONG /pages
# ================================
if page.startswith("Home"):
    from pages.Home import show
    show()

elif page.startswith("Analysis"):
    from pages.Analysis import show
    show()

elif page.startswith("Training Info"):
    from pages.Training_Info import show
    show()


# ================================
# 📝 FOOTER (KHÔNG ghi chữ footer)
# ================================
st.write("---")

# --- STUDENTS BOX ---
st.markdown(
    """
    <div style="
        padding:18px; 
        background:#ffffdd; 
        border-radius:10px;
        border:1px solid #e6d784;
        margin-bottom:10px;
    ">
        <b>Students:</b><br>
        - Student 1: ... email<br>
        - Student 2: ... email<br>
        - Student 3: ... email<br>
        - Student 4: ... email<br>
    </div>
    """,
    unsafe_allow_html=True
)

# --- INSTRUCTOR BOX ---
st.markdown(
    """
    <div style="
        padding:18px;
        background:#fafafa;
        border-radius:12px;
        border:1px solid #ddd;
        font-size:16px;
    ">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg"
             width="22"
             style="vertical-align:middle; margin-right:6px;">
        <b>Bùi Tiến Đức</b> –
        <a href="https://orcid.org/0000-0001-5174-3558"
           target="_blank"
           style="text-decoration:none; color:#0073e6;">
           ORCID: 0000-0001-5174-3558
        </a>
    </div>
    """,
    unsafe_allow_html=True
)
