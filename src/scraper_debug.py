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

# ================= CONFIG =================
BASE_DIR = Path(r"D:\PROYEK ML DAN AI\IndoMeme-XAI\data\raw\x_data")
IMAGE_DIR = BASE_DIR / "images"
METADATA_FILE = BASE_DIR / "metadata_debug.csv"
DEBUG_HTML_FILE = BASE_DIR / "debug_page_source.html"

KEYWORDS = ["meme lucu"] # Coba 1 keyword dulu untuk tes
# ===========================================

def setup_driver():
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--start-maximized")
    # Menonaktifkan notifikasi yang bisa menutupi elemen
    options.add_argument("--disable-notifications")
    driver = uc.Chrome(options=options, use_subprocess=True)
    return driver

def ensure_dirs():
    if not BASE_DIR.exists(): os.makedirs(BASE_DIR)
    if not IMAGE_DIR.exists(): os.makedirs(IMAGE_DIR)

def debug_scrape():
    ensure_dirs()
    driver = setup_driver()

    try:
        # === LOGIN MANUAL ===
        driver.get("https://x.com/i/flow/login")
        print("\n" + "="*50)
        print("!!! SILAKAN LOGIN MANUAL SEKARANG !!!")
        print("Pastikan sudah masuk HOME, lalu kembali ke sini.")
        print("="*50 + "\n")
        input(">> TEKAN ENTER SETELAH LOGIN DAN MASUK BERANDA <<")
        
        # === MULAI DIAGNOSA ===
        keyword = KEYWORDS[0]
        print(f"\n[DIAGNOSTIC] Mencari keyword: {keyword}")
        
        # URL Pencarian
        driver.get(f"https://x.com/search?q={keyword}&src=typed_query&f=media")
        
        print("[WAIT] Menunggu 15 detik agar halaman loading sempurna...")
        time.sleep(15) # Hard sleep untuk memastikan render selesai

        # --- CEK 1: APAKAH ADA ARTIKEL? ---
        articles = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
        print(f"[CHECK 1] Jumlah elemen <article data-testid='tweet'>: {len(articles)}")

        # --- CEK 2: APAKAH ADA GAMBAR? ---
        images = driver.find_elements(By.TAG_NAME, 'img')
        valid_images = [img for img in images if "pbs.twimg.com/media" in img.get_attribute("src") or ""]
        print(f"[CHECK 2] Jumlah total <img>: {len(images)}")
        print(f"[CHECK 2] Jumlah image meme potensial (pbs.twimg): {len(valid_images)}")

        # --- KEPUTUSAN ---
        if len(articles) == 0:
            print("\n[ALERT] Script GAGAL menemukan tweet standar.")
            print("[ACTION] Menyimpan Source Code halaman ke 'debug_page_source.html'...")
            
            # Simpan HTML untuk diperiksa manual
            with open(DEBUG_HTML_FILE, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            
            print(f"[SAVED] Cek file di: {DEBUG_HTML_FILE}")
            print("Tolong beritahu saya apa isi tag <div data-testid='...'> yang membungkus tweet di file tersebut.")
        
        else:
            print("\n[SUCCESS] Elemen ditemukan! Mencoba ambil data sampel...")
            # Coba ambil 1 data sebagai bukti
            first_tweet = articles[0]
            try:
                txt = first_tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
                print(f"Sample Text: {txt[:50]}...")
            except:
                print("Sample Text: (Tidak ada teks)")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        print("[DONE] Diagnosa selesai. Menutup browser...")
        try:
            driver.quit()
        except OSError:
            pass # Abaikan WinError 6

if __name__ == "__main__":
    debug_scrape()