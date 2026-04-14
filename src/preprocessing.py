import os
import pandas as pd
import re
from PIL import Image
from tqdm import tqdm

class MemePreprocessor:
    def __init__(self):
        # Kamus normalisasi slang dari hasil audit Notebook 01
        self.slang_dict = {
            'gw': 'gue', 'ga': 'tidak', 'gak': 'tidak', 'lo': 'kamu',
            'lu': 'kamu', 'udah': 'sudah', 'aja': 'saja', 'sih': 'saja',
            'yg': 'yang', 'jd': 'jadi', 'kalo': 'kalau', 'bgt': 'banget',
            'skg' : 'sekarang', 'dgn': 'dengan', 'dr': 'dari', 'tp': 'tapi',
            'klo': 'kalau', 'gmn': 'gimana',
        }

    def remove_icons_and_symbols(self, text):
        """
        Menghapus emoji, simbol ikon, dan karakter non-ASCII.
        """
        if not isinstance(text, str): return ""
        
        # 1. Hapus Emoji dan simbol Unicode (Ikon)
        # Menggunakan regex untuk menangkap karakter di luar jangkauan standar teks
        text = text.encode('ascii', 'ignore').decode('ascii')
        
        # 2. Hapus URL jika ada (sering nempel di teks meme)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 3. Hapus simbol/punctuation kecuali spasi (Menghapus 'ikon' tekstual)
        # Kita hanya sisakan karakter a-z dan angka 0-9
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 4. Hapus whitespace berlebih hasil pembersihan simbol
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def clean_text(self, text):
        """Pipeline pembersihan teks lengkap."""
        # Kecilkan huruf dulu
        text = text.lower()
        
        # Sikat simbol dan ikon
        text = self.remove_icons_and_symbols(text)
        
        # Normalisasi slang
        words = text.split()
        normalized_words = [self.slang_dict.get(w, w) for w in words]
        
        return " ".join(normalized_words)

    def pipeline(self, csv_path, save_path):
        """Menjalankan proses pembersihan pada metadata."""
        df = pd.read_csv(csv_path)
        
        print(f"Memulai pembersihan teks untuk {csv_path}...")
        
        # Terapkan pembersihan pada kolom OCR dan Caption
        df['ocr_cleaned'] = df['ocr'].apply(self.clean_text)
        df['caption_cleaned'] = df['caption'].apply(self.clean_text)
        
        # Gabungkan sesuai spesifikasi: [OCR] [SEP] [CAPTION]
        df['combined_text'] = df.apply(
            lambda x: f"{x['ocr_cleaned']} [SEP] {x['caption_cleaned']}", axis=1
        )
        
        # Simpan kolom yang kita butuhkan saja untuk training agar ringan
        final_columns = ['local_path', 'combined_text', 'hate_final', 'topic']
        df_final = df[final_columns].copy()
        
        # Konversi label ke numerik
        df_final['label'] = df_final['hate_final'].apply(lambda x: 1 if x == 'hate' else 0)
        
        df_final.to_csv(save_path, index=False)
        print(f"Preprocessing Selesai! Data 'bersih' disimpan di: {save_path}")

if __name__ == "__main__":
    pp = MemePreprocessor()
    
    # Jalankan untuk Train dan Test
    pp.pipeline('./data/train/metadata.csv', './data/train_cleaned.csv')
    pp.pipeline('./data/test/metadata.csv', './data/test_cleaned.csv')