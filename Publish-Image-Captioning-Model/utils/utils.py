import os
import gdown
import torch
import torch

torch.backends.quantized.engine = 'fbgemm' 

def load_vocab():
    if not os.path.exists("vocab.pth"):
        print("Downloading vocab...")
        file_id = "1zbmKzRx-6UbQwdD8dncLyLzZO3KniGdc"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "vocab.pth", quiet=False)
    
    vocab = torch.load("vocab.pth", weights_only=False)
    return vocab


def load_model():
    if not os.path.exists("best_model.pth"):
        print("Downloading model...")
        file_id = "1husywK1QEBkClPBhtRO6nZ7qvKE61ige"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, "best_model.pth", quiet=False)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state_dict = torch.load('best_model.pth', weights_only=False,  map_location=device)
    return model_state_dict, device