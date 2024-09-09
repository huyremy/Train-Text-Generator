import os
import json
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

# Load the dataset
with open("data_instruct_clean.json", "r") as f:
    dataset = json.load(f)

# Kiểm tra nếu dataset là danh sách hay dictionary
if isinstance(dataset, dict):
    instructions = [dataset["instruction"]]  # Nếu chỉ có một cặp instruction-output
    outputs = [dataset["output"]]  # Lấy output
else:
    instructions = [entry["instruction"] for entry in dataset]  # Nếu là danh sách các cặp
    outputs = [entry["output"] for entry in dataset]  # Lấy output cho từng cặp

# Preprocess the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Thêm token padding nếu chưa có
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Sử dụng token EOS làm token padding

# Tokenize và chuẩn bị dữ liệu
max_length = 50  # Giảm độ dài tối đa của đầu vào
train_encodings = tokenizer(instructions, truncation=True, padding=True, max_length=max_length, return_tensors='tf')

# Create the model (sử dụng mô hình nhỏ hơn)
model = TFGPT2LMHeadModel.from_pretrained("gpt2")  # Thay đổi mô hình

# Prepare the dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((
    train_encodings["input_ids"],
    train_encodings["input_ids"]  # Labels are the same as input_ids for language modeling
))
train_dataset = train_dataset.shuffle(1000).batch(1)  # Giảm kích thước batch xuống 1 max là 4

# Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

# Compile the model
model.compile(optimizer=optimizer)

# Train the model
model.fit(train_dataset, epochs=10)

# Save the model
model.save_pretrained("my_gpt2_model")
