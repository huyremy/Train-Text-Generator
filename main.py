import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Tải mô hình và tokenizer đã huấn luyện
model = GPT2LMHeadModel.from_pretrained('./chatbot_model')
tokenizer = GPT2Tokenizer.from_pretrained('./chatbot_model')

# Đặt mô hình vào chế độ đánh giá
model.eval()

def generate_response(prompt):
    # Tokenize đầu vào
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Tạo attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()  # Tạo attention mask từ input_ids

    # Tạo câu trả lời
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,  # Thêm attention_mask
            max_length=150,  # Độ dài tối đa của câu trả lời
            num_return_sequences=1,  # Số lượng câu trả lời muốn nhận
            no_repeat_ngram_size=2,  # Tránh lặp lại các n-gram
            pad_token_id=tokenizer.eos_token_id  # ID của token padding
        )

    # Giải mã kết quả
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
