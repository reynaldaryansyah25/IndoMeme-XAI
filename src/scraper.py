"""
========================================================================
PROJECT: INDOMEME-XAI (Data Acquisition Module)
AUTHOR : Reynald Aryansyah
DATE   : 2024
LICENSE: Private Research (skripsi/tesis purpose)

Deskripsi:
Script ini dirancang khusus oleh Reynald Aryansyah untuk mengambil dataset 
meme multimodal dari X (Twitter) dengan mekanisme 'Safety First'.
Dilengkapi fitur anti-duplikat, auto-resume, dan batch monitoring.
========================================================================
"""

import os
import csv
import time
import random
import requests
import hashlib
from pathlib import Path
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ==============================================================================
# KONFIGURASI PROYEK (Reynald's Config)
# ==============================================================================

# Folder penyimpanan data (sesuaikan path laptop Reynald)
FOLDER_UTAMA = Path(r"D:\PROYEK ML DAN AI\IndoMeme-XAI\data\raw\final_dataset")
FOLDER_GAMBAR = FOLDER_UTAMA / "images"
FILE_METADATA = FOLDER_UTAMA / "dataset.csv"

# BATAS AMAN: Script akan 'Pause' setiap X download agar akun tidak kena ban
JATAH_PER_SESI = 500 

# TARGET SCRAPING (Keyword Rahasia Dapur - UPDATED)
TARGET_OPERASI = {
    "non_hate": {
        "target": 4000,
        "kata_kunci": [
            "meme lucu", "meme kocak", "meme receh", "asupan meme", 
            "meme kegoblokan", "meme warga +62", "meme ngakak", 
            "meme random", "shitpost indonesia", "meme tuman",
            "meme kucing", "meme anabul", # Tambahan
            "meme kaum rebahan", "meme sobat miskin", "meme beban keluarga",
            "meme budak korporat", "meme mahasiswa akhir", "meme skripsi",
            "meme gaji umr", "meme emak-emak", "jokes bapak-bapak", 
            "meme thr", "meme puasa", "meme lebaran", "meme wibu indonesia", 
            "meme kpop indo", "meme mobile legends", "meme epyepe", 
            "meme dark jokes indo", "meme twitter", "meme kearifan lokal"
        ]
    },
    "satire": {
        "target": 3000,
        "kata_kunci": [
            "meme pejabat", "meme wakil rakyat", "meme dpr", "meme ketua dpr",
            "meme mulyono", "meme raja jawa", "meme paman usman",
            "meme dinasti", "meme nepotisme", "meme mahkamah keluarga",
            "meme kaesang", "meme gibran", "meme fufufafa", "meme samsul",
            "meme tapera", "meme ukt mahal", "meme pajak", "meme bea cukai",
            "meme kominfo", "meme blokir", "meme ikn", "meme jalan rusak",
            "meme banjir", "meme polusi jakarta", "meme hukum tumpul",
            "meme korupsi", "meme bansos", "meme konoha", "meme wakanda"
        ]
    },
    "hate_speech": {
        "target": 3000,
        "kata_kunci": [
            "meme cebong", "meme kampret", "meme kadrun", "meme buzzeRp",
            "meme penjilat", "meme pengkhianat", "meme antek asing",
            "meme pki", "meme komunis", "meme khilafah", "meme radikal",
            "meme syiah", "meme liberal", "meme wahabi", "meme tolol", 
            "meme goblok", "meme bangsat", "meme anjing", "meme babi", 
            "meme sampah masyarakat", "meme otak udang", "meme mukidi", 
            "meme planga plongo", "meme petugas partai", "meme rezim dzalim", 
            "meme rezim panik", "meme lengser"
        ]
    }
}

# ==============================================================================
# FUNGSI BANTUAN (ENGINEERING TOOLS)
# ==============================================================================

def siapkan_browser_siluman():
    """Membuka Chrome dengan mode samaran agar tidak terdeteksi sebagai bot."""
    opsi = uc.ChromeOptions()
    opsi.add_argument("--no-first-run")
    opsi.add_argument("--start-maximized")
    opsi.add_argument("--disable-notifications") # Matikan notifikasi ganggu
    
    print(f"[SYSTEM] Reynald's Scraper sedang menyiapkan browser...")
    driver = uc.Chrome(options=opsi, use_subprocess=True)
    return driver

def cek_folder_siap():
    """Memastikan folder penyimpanan sudah ada."""
    if not FOLDER_UTAMA.exists():
        os.makedirs(FOLDER_UTAMA)
        print("[SYSTEM] Folder utama dibuat.")
    if not FOLDER_GAMBAR.exists():
        os.makedirs(FOLDER_GAMBAR)
        print("[SYSTEM] Folder gambar dibuat.")

def hitung_progress_saat_ini():
    """Mengecek berapa data yang sudah Reynald dapatkan sejauh ini."""
    statistik = {"non_hate": 0, "satire": 0, "hate_speech": 0}
    
    if not os.path.exists(FILE_METADATA):
        return statistik
    
    with open(FILE_METADATA, 'r', encoding='utf-8-sig') as f:
        pembaca = csv.DictReader(f)
        for baris in pembaca:
            kategori = baris.get('category', 'unknown')
            if kategori in statistik:
                statistik[kategori] += 1
    return statistik

def bersihkan_teks_sampah(teks_mentah):
    """Membersihkan caption dari username dan baris kosong."""
    if not teks_mentah: return ""
    baris_baris = teks_mentah.split('\n')
    # Filter: Hapus baris pendek (1 huruf) dan username (@...)
    bersih = [b.strip() for b in baris_baris if len(b.strip()) > 1 and not b.startswith("@")]
    return " ".join(bersih)

def download_gambar(url, id_tweet):
    """Mendownload gambar ke folder lokal."""
    try:
        if not url: return None
        # Ubah thumbnail video jadi gambar asli
        if "video_thumb" in url: url = url.replace("video_thumb", "media")
        
        headers = {'User-Agent': 'Mozilla/5.0'} # Nyamar jadi browser biasa
        respon = requests.get(url, headers=headers, timeout=10)
        
        if respon.status_code == 200:
            # Hash gambar agar nama file unik
            hash_img = hashlib.md5(respon.content).hexdigest()
            ext = 'jpg' if 'jpg' in url else 'png'
            nama_file = f"{id_tweet}_{hash_img}.{ext}"
            path_lengkap = FOLDER_GAMBAR / nama_file
            
            # Cek duplikat file fisik
            if not path_lengkap.exists():
                with open(path_lengkap, "wb") as f:
                    f.write(respon.content)
            return nama_file
    except Exception as e:
        return None
    return None

# ==============================================================================
# LOGIKA UTAMA (MAIN ENGINE)
# ==============================================================================

def mulai_operasi_reynald():
    cek_folder_siap()
    
    # Setup CSV Database
    file_ada = os.path.isfile(FILE_METADATA)
    mode_tulis = 'a' if file_ada else 'w'
    file_csv = open(FILE_METADATA, mode_tulis, newline='', encoding='utf-8-sig')
    
    kolom = ['tweet_id', 'category', 'keyword', 'full_text', 'image_filename', 'image_url', 'tweet_url']
    penulis = csv.DictWriter(file_csv, fieldnames=kolom)
    if not file_ada: penulis.writeheader()

    # Load ID lama untuk anti-duplikat memori
    id_sudah_diproses = set()
    if file_ada:
        print("[SYSTEM] Membaca database lama punya Reynald...")
        with open(FILE_METADATA, 'r', encoding='utf-8-sig') as f:
            pembaca = csv.DictReader(f)
            for baris in pembaca:
                id_sudah_diproses.add(baris['tweet_id'])
        print(f"[INFO] {len(id_sudah_diproses)} data lama berhasil dimuat. Aman dari duplikat.")

    driver = siapkan_browser_siluman()
    total_download_sesi_ini = 0 

    try:
        # --- PROSES LOGIN ---
        driver.get("https://x.com/i/flow/login")
        print("\n" + "="*60)
        print(f"HI REYNALD! SILAKAN LOGIN MANUAL SEKARANG.")
        print("Pastikan sudah masuk halaman HOME, baru lanjut.")
        print("="*60 + "\n")
        input(">> TEKAN ENTER KALAU SUDAH LOGIN <<")
        
        # --- LOOPING KATEGORI ---
        for kategori, config in TARGET_OPERASI.items():
            progress = hitung_progress_saat_ini()
            target_kategori = config['target']
            
            # Cek apakah target sudah tercapai
            if progress[kategori] >= target_kategori:
                print(f"\n[SKIP] Kategori '{kategori}' sudah aman ({progress[kategori]}/{target_kategori}).")
                continue

            print(f"\n[PHASE] Memulai Misi: {kategori.upper()} (Target: {target_kategori})")
            
            list_keyword = config['kata_kunci']
            random.shuffle(list_keyword) # Acak biar nggak bosen

            for kata_kunci in list_keyword:
                # --- SAFETY CHECK: BATCH LIMIT ---
                if total_download_sesi_ini >= JATAH_PER_SESI:
                    print("\n" + "!"*60)
                    print(f"[SAFETY] Reynald, istirahat dulu! Sudah {total_download_sesi_ini} download.")
                    print("Saran: Matikan script, ngopi dulu 30 menit biar akun aman.")
                    print("!"*60)
                    
                    tanya = input(">> Mau lanjut paksa? (y/n): ").lower()
                    if tanya != 'y':
                        print("[SHUTDOWN] Menyimpan data dan keluar...")
                        return # Keluar dari fungsi utama
                    else:
                        print("[RESUME] Oke, lanjut gaspol...")
                        total_download_sesi_ini = 0 # Reset counter
                # ---------------------------------

                # Cek target lagi (real-time)
                if hitung_progress_saat_ini()[kategori] >= target_kategori:
                    print(f"  [DONE] Target {kategori} tercapai bos!")
                    break

                print(f"\n  [SEARCH] Keyword: {kata_kunci}")
                
                # Query Spesifik: Indo + Gambar Only + No Retweet
                query = f'{kata_kunci} lang:id filter:images -filter:retweets'
                url_cari = f"https://x.com/search?q={requests.utils.quote(query)}&src=typed_query"
                driver.get(url_cari)
                time.sleep(5) # Tunggu loading

                gagal_muat_counter = 0

                # Scroll max 40 kali per keyword
                for scroll in range(40):
                    tweets = driver.find_elements(By.CSS_SELECTOR, "article[data-testid='tweet']")
                    
                    if not tweets:
                        time.sleep(2)
                        continue

                    dapat_baru = 0
                    for tweet in tweets:
                        try:
                            # 1. Ambil ID & URL
                            try:
                                link = tweet.find_element(By.CSS_SELECTOR, "a[href*='/status/']")
                                tweet_url = link.get_attribute("href")
                                id_tweet = tweet_url.split('/status/')[-1]
                            except: continue

                            if id_tweet in id_sudah_diproses: continue # Skip punya lama

                            # 2. Ambil Gambar
                            url_gambar = None
                            try:
                                imgs = tweet.find_elements(By.CSS_SELECTOR, "div[data-testid='tweetPhoto'] img")
                                for img in imgs:
                                    if "pbs.twimg.com/media" in img.get_attribute("src"):
                                        url_gambar = img.get_attribute("src")
                                        break
                            except: pass
                            
                            if not url_gambar: continue 

                            # 3. Ambil Teks (Caption)
                            teks_lengkap = ""
                            try:
                                el_teks = tweet.find_element(By.CSS_SELECTOR, "div[data-testid='tweetText']")
                                teks_lengkap = bersihkan_teks_sampah(el_teks.text)
                            except:
                                # Fallback ke Alt Text kalau caption kosong
                                try: teks_lengkap = "[ALT] " + imgs[0].get_attribute("alt")
                                except: teks_lengkap = ""

                            # 4. Simpan Data
                            nama_file_gambar = download_gambar(url_gambar, id_tweet)
                            if nama_file_gambar:
                                penulis.writerow({
                                    'tweet_id': id_tweet, 
                                    'category': kategori, # Label otomatis
                                    'keyword': kata_kunci, 
                                    'full_text': teks_lengkap,
                                    'image_filename': nama_file_gambar, 
                                    'image_url': url_gambar, 
                                    'tweet_url': tweet_url
                                })
                                file_csv.flush() # Simpan fisik ke harddisk
                                id_sudah_diproses.add(id_tweet)
                                total_download_sesi_ini += 1
                                dapat_baru += 1
                                print(f"    [+] {kategori.upper()} | {nama_file_gambar} | Teks: {teks_lengkap[:20]}...")

                        except Exception: continue
                    
                    # Logika Scroll
                    if dapat_baru == 0:
                        gagal_muat_counter += 1
                        if gagal_muat_counter >= 3:
                            print("    [!] Gak ada barang baru, ganti keyword lain.")
                            break
                    else:
                        gagal_muat_counter = 0

                    # Scroll acak biar kayak manusia
                    driver.execute_script(f"window.scrollBy(0, {random.randint(1200, 2000)});")
                    time.sleep(random.uniform(4, 7))

    except Exception as e:
        print(f"[ERROR] Ada masalah teknis bos: {e}")

    finally:
        file_csv.close()
        print("\n" + "="*60)
        print(f"OPERASI SELESAI. Total download sesi ini: {total_download_sesi_ini}")
        print("Script by: Reynald Aryansyah")
        print("="*60)

if __name__ == "__main__":
    mulai_operasi_reynald()