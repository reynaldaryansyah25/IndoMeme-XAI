"""
========================================================================
PROJECT: INDOMEME-XAI (Data Validation Tool)
AUTHOR : Reynald Aryansyah
DESC   : Tool untuk memvalidasi 'Weak Labels' menjadi 'Gold Labels'
========================================================================
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from pathlib import Path

# KONFIGURASI
FOLDER_UTAMA = Path(r"D:\PROYEK ML DAN AI\IndoMeme-XAI\data\raw\final_dataset")
FILE_MENTAH = FOLDER_UTAMA / "dataset.csv"
FILE_FIX = FOLDER_UTAMA / "dataset_reynald_VALIDATED.csv"
FOLDER_GAMBAR = FOLDER_UTAMA / "images"

def validasi_data():
    # 1. Load Data
    if not os.path.exists(FILE_MENTAH):
        print("File dataset belum ada!")
        return

    df = pd.read_csv(FILE_MENTAH)
    
    # Cek kalau kita mau lanjutin validasi sebelumnya
    if os.path.exists(FILE_FIX):
        df_fix = pd.read_csv(FILE_FIX)
        id_sudah = set(df_fix['tweet_id'].tolist())
        # Filter data yang BELUM divalidasi
        df_sisa = df[~df['tweet_id'].isin(id_sudah)]
    else:
        df_sisa = df.copy()
        # Buat file baru dengan header
        df_sisa.head(0).to_csv(FILE_FIX, index=False)

    total = len(df)
    sisa = len(df_sisa)
    print(f"\n[INFO] Total Data: {total}. Sudah Divalidasi: {total - sisa}. Sisa: {sisa}")
    
    plt.ion() # Mode interaktif
    fig = plt.figure(figsize=(10, 8))

    # 2. Loop Validasi
    for index, row in df_sisa.iterrows():
        img_path = FOLDER_GAMBAR / row['image_filename']
        
        # Skip kalau gambar fisik gak ada (error download)
        if not img_path.exists():
            continue

        try:
            # Tampilkan Gambar
            plt.clf()
            img = mpimg.imread(str(img_path))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"KATEGORI MESIN: [{row['category'].upper()}]\nKeyword: {row['keyword']}", fontsize=14, color='blue')
            plt.draw()
            plt.pause(0.01)
            
            # Tampilkan Teks di Terminal
            print("\n" + "="*50)
            print(f"ID: {row['tweet_id']}")
            print(f"CAPTION: {row['full_text']}")
            print("-" * 20)
            print(f"PREDIKSI MESIN: {row['category']}")
            print("-" * 20)
            print("[ENTER] = Setuju (Benar)")
            print("[1] = Ganti jadi NON_HATE")
            print("[2] = Ganti jadi SATIRE")
            print("[3] = Ganti jadi HATE_SPEECH")
            print("[x] = HAPUS (Sampah/Iklan/Bukan Meme)")
            print("[q] = STOP & KELUAR")
            
            pilihan = input(">> Pilihan Anda: ").lower().strip()

            if pilihan == 'q':
                print("Menyimpan progress...")
                break
            
            # Logika Update Label
            label_baru = row['category'] # Default: ikut mesin
            
            if pilihan == '1': label_baru = 'non_hate'
            elif pilihan == '2': label_baru = 'satire'
            elif pilihan == '3': label_baru = 'hate_speech'
            elif pilihan == 'x': 
                print("Data dihapus (tidak disimpan ke file fix).")
                continue # Skip, jangan simpan ke file fix

            # Update row dan simpan
            row['category'] = label_baru
            
            # Append ke file CSV Validated (Real-time saving)
            pd.DataFrame([row]).to_csv(FILE_FIX, mode='a', header=False, index=False)
            print(f"[OK] Disimpan sebagai: {label_baru}")

        except Exception as e:
            print(f"Error menampilkan gambar: {e}")
            continue

    plt.close()
    print("\n[SELESAI] Sesi validasi berakhir.")

if __name__ == "__main__":
    validasi_data()