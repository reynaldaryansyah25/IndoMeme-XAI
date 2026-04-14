import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm

def extract_huggingface_schema(parquet_file, target_dir):
    """
    Mengekstrak file parquet ke folder lokal dengan mempertahankan 
    seluruh kolom asli dari Hugging Face.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Membaca data parquet
    df = pd.read_parquet(parquet_file)
    
    # List untuk menampung baris metadata
    metadata_rows = []

    print(f"Mengekstrak {len(df)} data ke {target_dir}...")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Proses Gambar
            # Kita simpan gambar secara fisik untuk memudahkan inspeksi & OCR manual nanti
            img_data = row['image']['bytes']
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Gunakan index atau image_path asli untuk nama file
            # Saya sarankan menggunakan index agar unik dan rapi
            file_name = f"meme_{index}.jpg"
            image_save_path = os.path.join(target_dir, file_name)
            img.save(image_save_path, "JPEG")

            # 2. Ambil semua kolom asli + local_path
            record = {
                'local_path': file_name,            # Kolom tambahan untuk pointer file lokal
                'image_path': row['image_path'],      # Nama file asli dari dataset
                'hate_final': row['hate_final'],      # Target Label
                'appropriate_final': row['appropriate_final'],
                'topic': row['topic'],
                'topic_id': row['topic_id'],
                'ocr': row['ocr'],                    # Teks hasil Qwen2-VL
                'caption': row['caption']             # Deskripsi hasil Gemini 2.5 Flash
            }
            metadata_rows.append(record)

        except Exception as e:
            print(f"Error pada baris {index}: {e}")
            continue

    # 3. Simpan ke CSV dengan kolom yang sama persis seperti Hugging Face
    new_df = pd.DataFrame(metadata_rows)
    new_df.to_csv(os.path.join(target_dir, "metadata.csv"), index=False)
    print(f"Berhasil! Metadata disimpan di {target_dir}/metadata.csv\n")

# --- Eksekusi Konfigurasi ---
config = [
    {
        "src": "INDOMEME/data/train-00000-of-00001.parquet",
        "dst": "data/train"
    },
    {
        "src": "INDOMEME/data/test-00000-of-00001.parquet",
        "dst": "data/test"
    }
]

for split in config:
    if os.path.exists(split['src']):
        extract_huggingface_schema(split['src'], split['dst'])
    else:
        print(f"File {split['src']} tidak ditemukan. Pastikan path git clone benar.")