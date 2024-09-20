import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm  # Thêm thư viện tqdm

# Kiểm tra xem GPU có sẵn không
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải tập dữ liệu
dataset = load_dataset("wanhin/VinAI_test")

# Tải mô hình và tokenizer DialoGPT
model_name = "HuyRemy/TiengViet"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)  # Di chuyển mô hình lên GPU

# Thêm pad token nếu chưa có
tokenizer.pad_token = tokenizer.eos_token  # Sử dụng eos_token làm pad_token

def encode(text):
    # Sử dụng padding và truncation
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs["input_ids"].to(device)  # Di chuyển tensor lên GPU

def get_vectors_from_dataset(dataset):
    vectors = []
    for item in tqdm(dataset['validation'], desc="Đang tạo vectors", unit=" câu"):  # Thay đổi thành 'validation'
        text = item['vi']  # Thay đổi từ 'question' thành 'vi'
        input_ids = encode(text)
        
        with torch.no_grad():  # Tắt tính toán gradient
            outputs = model(input_ids)
            logits = outputs.logits  # Lấy logits

            # Tính vector trung bình từ logits
            vector = logits.mean(dim=1).detach().cpu().numpy()  # Chuyển về CPU trước khi chuyển thành numpy
        
        vectors.append(vector)
    return np.vstack(vectors)  # Chuyển đổi danh sách thành mảng 2D

def create_faiss_index(vectors):
    dimension = vectors.shape[1]  # Kích thước vector
    index = faiss.IndexFlatL2(dimension)  # Tạo index FAISS
    index.add(vectors)  # Thêm vectors vào index
    return index

# Lấy vectors từ dataset
vectors = get_vectors_from_dataset(dataset)  
faiss_index = create_faiss_index(vectors)  # Tạo index FAISS

# Lưu mô hình, tokenizer và FAISS index
output_dir = './saved_model'
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
faiss.write_index(faiss_index, "huyremy_vector.bin")
print("Completed")
# Implement here 
def chatbot(text):
    # Mã hóa đầu vào và lấy vector
    input_vector = encode(text)
    
    with torch.no_grad():  # Tắt tính toán gradient
        outputs = model(input_vector)
        logits = outputs.logits  # Lấy logits
        input_vector = logits.mean(dim=1).detach().cpu().numpy()  # Chuyển về CPU trước khi chuyển thành numpy

    # Tìm kiếm trong FAISS index
    D, I = faiss_index.search(input_vector, k=5)  # k là số lượng kết quả muốn tìm

    # Tạo phản hồi dựa trên các ví dụ tương tự
    responses = [dataset['validation'][i]['en'] for i in I[0]]  # Thay đổi từ 'answer' thành 'en'

    # Tạo phản hồi cuối cùng
    final_response = " ".join(responses)
    return final_response
    
abc=chatbot("who are you?")
abc = str(abc)
print(abc)
