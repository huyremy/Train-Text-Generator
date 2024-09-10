from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Khởi tạo mô hình
model = TFGPT2LMHeadModel.from_pretrained("gpt2")  # Hoặc mô hình bạn đã huấn luyện

# Tải trọng số từ file .bin
model.load_weights("my_model_weights.bin")

# Khởi tạo tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Hàm để tạo ra văn bản từ mô hình
def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='tf')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sử dụng mô hình để tạo văn bản
prompt = "Write a function to add two numbers."
generated_text = generate_text(prompt)
print(generated_text)
