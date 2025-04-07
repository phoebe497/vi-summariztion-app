import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import os

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Đang sử dụng CPU")

# Khởi tạo session state để lưu mô hình và tokenizer
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

# Hàm load mô hình (chỉ gọi khi cần)
def load_model_and_tokenizer():
    model_name = "NlpHUST/t5-small-vi-summarization"
    try:
        if st.session_state.tokenizer is None:
            st.session_state.tokenizer = T5Tokenizer.from_pretrained(model_name)
        if st.session_state.model is None:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            model = model.to(device)
            # Load mô hình fine-tuned
            best_model_path = "D:/NLP/Research_T5/checkpoints/best_model.pt"
            if os.path.exists(best_model_path):
                if device.type == "cuda":
                    model.load_state_dict(torch.load(best_model_path))
                else:
                    model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
            else:
                st.error(f"Không tìm thấy file checkpoint tại: {best_model_path}!")
                st.stop()
            model.eval()
            st.session_state.model = model
    except Exception as e:
        st.error(f"Lỗi khi load mô hình: {e}")
        st.stop()

# Hàm tóm tắt
def summarize(text, model, tokenizer, device, max_length=150):
    model.eval()
    prefix = "Tóm tắt: "
    inputs = tokenizer(prefix + text, return_tensors="pt", max_length=512, truncation=True)
    
    prefix_tokens = len(tokenizer(prefix, return_tensors="pt")["input_ids"][0])
    adjusted_max_length = max_length + prefix_tokens
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=adjusted_max_length,
            min_length=int(max_length * 0.6),  # Giảm min_length
            num_beams=2,  # Giảm num_beams để tăng tốc
            early_stopping=True,  # Bật early stopping
            pad_token_id=tokenizer.pad_token_id
        )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    # Post-processing: Cắt ở cuối câu
    if summary and summary[-1] not in ['.', '!', '?']:
        last_sentence_end = max(summary.rfind('.'), summary.rfind('!'), summary.rfind('?'))
        if last_sentence_end != -1:
            summary = summary[:last_sentence_end + 1]
    
    return summary

# Giao diện Streamlit
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@700&display=swap');
        .title-card {
            background-color: #FAFAFA;
            padding: 10px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 20px;
        }
        .title-text {
            margin-top: 0;
        }
        .title-text h1 {
            color: #FF4B4B;
            font-family: 'Quicksand', sans-serif;
            font-size: 36px;
            margin: 0;
            text-shadow: 2px 2px 5px #aaa;
        }
        .title-text p {
            font-size: 16px;
            color: #666;
            margin: 5px 0 0 0;
        }
        .summary-box {
            background-color: #F9F9F9;
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #FF4B4B;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
    <div class="title-card">
        <div class="title-text">
            <h1>AI Tóm tắt văn bản Tiếng Việt</h1>
            <p>Ứng dụng tóm tắt văn bản tự động sử dụng mô hình T5-small</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("### 📌 Nhập văn bản cần tóm tắt:")

# Ô nhập văn bản
input_text = st.text_area("Văn bản đầu vào", height=200)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Tùy chọn tóm tắt")
    length_option = st.selectbox("Độ dài tóm tắt", ["Ngắn", "Dài"])
    max_length_map = {
        "Ngắn": 80,  # Giảm để tăng tốc
        "Dài": 150   # Giảm để tăng tốc
    }
    max_length = max_length_map[length_option]
    
    st.markdown("---")
    st.markdown("### 📌 Thông tin mô hình")
    st.write("- **Mô hình:** T5-small (NlpHUST/t5-small-vi-summarization)")
    st.write("- **Fine-tuned trên:** 190 bài báo khoa học tiếng Việt")
    st.write("- **Batch size:** 128, **Epochs:** 200")
    st.write("- **Max input length:** 512")

# Nút tóm tắt
if st.button("🚀 Tóm tắt ngay"):
    if input_text.strip() == "":
        st.error("Vui lòng nhập văn bản để tóm tắt!")
    else:
        # Load mô hình nếu chưa load
        if st.session_state.model is None or st.session_state.tokenizer is None:
            with st.spinner("⏳ Đang load mô hình..."):
                load_model_and_tokenizer()
        
        # Tóm tắt
        with st.spinner("⏳ Đang xử lý..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            summary = summarize(
                input_text,
                st.session_state.model,
                st.session_state.tokenizer,
                device,
                max_length=max_length
            )

        st.success("✅ Tóm tắt thành công!")
        st.write("### ✨ Kết quả tóm tắt:")
        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

        # Thông tin độ dài
        st.write("### 📏 Thông tin độ dài:")
        st.write(f"- Độ dài văn bản gốc: **{len(input_text.split())} từ**, **{len(input_text)} ký tự**")
        st.write(f"- Độ dài văn bản tóm tắt: **{len(summary.split())} từ**, **{len(summary)} ký tự**")
        # summary_tokens = len(st.session_state.tokenizer(summary, return_tensors="pt")["input_ids"][0])
        # st.write(f"- Số token tóm tắt: **{summary_tokens} tokens**")