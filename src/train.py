import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    AutoTokenizer, 
    ViTImageProcessor, 
    AutoModel, 
    ViTModel
)

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
class MemeDataset(Dataset):
    def __init__(self, csv_file, img_dir, max_len=128, img_size=512):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.max_len = max_len
        self.img_size = img_size
        
        # Inisialisasi Processor & Tokenizer di sini biar aman
        self.tokenizer = AutoTokenizer.from_pretrained("indolem/indobertweet-base-uncased")
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k", 
            size={"height": img_size, "width": img_size}
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['combined_text'])
        
        # Tokenisasi
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Image Processing (512x512)
        img_path = os.path.join(self.img_dir, row['local_path'])
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': pixel_values,
            'label': torch.tensor(int(row['label']), dtype=torch.long)
        }

# ==========================================
# 2. MODEL DEFINITION (CROSS-ATTENTION)
# ==========================================
class MultimodalMemeModel(nn.Module):
    def __init__(self, num_labels=2):
        super(MultimodalMemeModel, self).__init__()
        
        # IndoBERTweet
        self.text_encoder = AutoModel.from_pretrained("indolem/indobertweet-base-uncased")
        # ViT Base
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        self.hidden_dim = 768
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        # Text Feats
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feats = text_outputs.last_hidden_state # [B, L, 768]
        
        # Image Feats (FIXED: Support 512 resolution with interpolation)
        image_outputs = self.image_encoder(pixel_values=pixel_values, interpolate_pos_encoding=True)
        image_feats = image_outputs.last_hidden_state # [B, P, 768]
        
        # Cross-Attention: Text (Q) looks at Image (K, V)
        attn_output, _ = self.cross_attn(query=text_feats, key=image_feats, value=image_feats)
        
        # Take [CLS] token (index 0)
        fused_vector = attn_output[:, 0, :]
        logits = self.classifier(fused_vector)
        return logits

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def run_training():
    # --- Config ---
    CSV_TRAIN = './data/train_cleaned.csv'
    IMG_DIR = './data/train/'
    BATCH_SIZE = 2  # Kecilkan jika OOM karena resolusi 512
    EPOCHS = 5
    LR = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Memulai IndoMeme-XAI Training di: {DEVICE}")

    # --- Dataset ---
    train_dataset = MemeDataset(CSV_TRAIN, IMG_DIR, img_size=512)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Model & Tools ---
    model = MultimodalMemeModel().to(DEVICE)
    
    # Handle Imbalance 65:35
    weights = torch.tensor([1.0, 1.8]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            # Pindahkan data
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            pix = batch['pixel_values'].to(DEVICE)
            lbl = batch['label'].to(DEVICE)

            # Forward & Backward
            optimizer.zero_grad()
            logits = model(ids, mask, pix)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Record metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbl.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # --- Metrics Akhir Epoch ---
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"\n✅ Epoch {epoch+1} Result | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")

        # --- Save Model ---
        if not os.path.exists('../result'): os.makedirs('../result')
        torch.save(model.state_dict(), f"./result/indomeme_512_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    run_training()