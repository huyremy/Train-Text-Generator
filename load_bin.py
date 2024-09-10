from transformers import TFGPT2LMHeadModel

# Khởi tạo mô hình
model = TFGPT2LMHeadModel.from_pretrained("gpt2")  # Hoặc mô hình bạn đã huấn luyện

# Tải trọng số từ file .bin
model.load_weights("my_model_weights.bin")
