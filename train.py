import json
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Tải dữ liệu từ file JSON
# Use json.loads to load from string
# Remove trailing commas and extra characters from the file
with open('data_instruct_clean.json', 'r') as f:
    # Read the file as a string
    file_content = f.read()
    # Remove any trailing commas
    file_content = file_content.replace(",]", "]").replace(",}", "}")
    # Load the JSON data from the modified string
    data = json.loads(file_content)

# Chuyển đổi dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Tạo câu hỏi và câu trả lời
#df['text'] = df['instruction'] + ' ' + df['input'] + ' ' + df['output']
df['text'] = df['instruction'] + ' ' + df['output']
# Tải mô hình và tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Thêm token padding
tokenizer.pad_token = tokenizer.eos_token  # Sử dụng token kết thúc làm token padding

model = GPT2LMHeadModel.from_pretrained(model_name)

# Tạo Dataset tùy chỉnh
class TextDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']  # Đặt labels bằng input_ids
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Tokenize dữ liệu
encodings = tokenizer('\n'.join(df['text'].tolist()), return_tensors='pt', max_length=512, truncation=True, padding=True)

# Tạo dataset từ encodings
dataset = TextDataset(encodings)

# Huấn luyện mô hình
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3000,
    per_device_train_batch_size=10,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Lưu mô hình đã huấn luyện
model.save_pretrained('./chatbot_model')
tokenizer.save_pretrained('./chatbot_model')
