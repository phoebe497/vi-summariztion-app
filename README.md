# vi-summariztion-app
## Hướng dẫn chạy giao diện tóm tắt

```python
# 1. Cài đặt thư viện cần thiết
!pip install streamlit transformers torch

# 2. Tải mô hình đã fine-tune và đặt vào thư mục checkpoints/
# Đảm bảo tồn tại file checkpoints/best_model.pt

# 3. Chạy ứng dụng Streamlit
!streamlit run app.py