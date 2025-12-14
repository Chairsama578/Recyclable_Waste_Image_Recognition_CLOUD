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
        "<h3 style='color:#2b6f3e;'>Giới thiệu Đề tài</h3>",
        unsafe_allow_html=True
    )

    # ====== MỤC 1 ======
    intro_box("""
    <h3 style="color:#b30000;">1. Bối cảnh và Lý do chọn đề tài</h3>
    Vấn đề xử lý và phân loại rác thải đóng vai trò quan trọng trong việc bảo vệ môi trường,
    đặc biệt tại các đô thị lớn nơi lượng rác sinh hoạt tăng nhanh.
    Việc phân loại rác thủ công thường tốn thời gian, thiếu chính xác và chi phí nhân công cao.

    Sự phát triển của Trí tuệ Nhân tạo, đặc biệt là công nghệ <b>AutoML Vision</b> của Google,
    cho phép tạo ra các mô hình nhận diện hình ảnh một cách tự động, không cần lập trình phức tạp.
    Điều này giúp sinh viên có thể triển khai mô hình phân loại rác một cách hiệu quả và thực tế.
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
        <li>Plastic (Nhựa)</li>
        <li>Paper (Giấy)</li>
        <li>Glass (Thủy tinh)</li>
        <li>Metal (Kim loại)</li>
        <li>Organic (Hữu cơ)</li>
        <li>Others (Khác)</li>
    </ul>

    Hệ thống sau khi huấn luyện sẽ được tích hợp vào ứng dụng web Streamlit để demo khả năng phân loại rác.
    Đây là bước quan trọng hướng đến <b>giải pháp phân loại rác tự động (Automated Waste Sorting System)</b>.
    """)

    # ====== MỤC 3 ======
    intro_box("""
    <h3 style="color:#b30000;">3. Phạm vi và Nội dung thực hiện</h3>

    <ul>
        <li>Thu thập và chuẩn hóa dữ liệu hình ảnh rác.</li>
        <li>Chuẩn bị cấu trúc dataset theo đúng chuẩn AutoML Vision.</li>
        <li>Huấn luyện mô hình phân loại rác bằng AutoML Vision.</li>
        <li>Đánh giá mô hình qua các chỉ số: Accuracy, Precision, Recall, F1-score.</li>
        <li>Triển khai mô hình dự đoán trong giao diện Streamlit.</li>
        <li>Đề xuất quy trình phân loại rác tự động dựa trên mô hình đã xây dựng.</li>
    </ul>
    """)

    # ====== MỤC 4 ======
    intro_box("""
    <h3 style="color:#b30000;">4. Ý nghĩa khoa học và thực tiễn</h3>

    <ul>
        <li>Ứng dụng AI vào công tác phân loại rác – lĩnh vực có ý nghĩa xã hội lớn.</li>
        <li>Giảm gánh nặng cho công nhân môi trường.</li>
        <li>Tăng tỷ lệ tái chế nhờ nhận diện chính xác.</li>
        <li>Có thể phát triển thành hệ thống phân loại rác tự động trong các đô thị thông minh.</li>
    </ul>

    Đề tài mang tính ứng dụng cao và phù hợp xu hướng chuyển đổi số trong lĩnh vực môi trường.
    """)

