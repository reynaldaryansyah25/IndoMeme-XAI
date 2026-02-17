# 🛡️ IndoMeme-XAI

## Deteksi Multimodal Hate Speech dengan Explainable AI

**IndoMeme-XAI** adalah proyek riset kecerdasan buatan yang berfokus pada deteksi ujaran kebencian (*hate speech*) dalam bentuk meme — konten yang menggabungkan teks dan gambar — menggunakan pendekatan **Explainable AI (XAI)**.

Proyek ini dirancang untuk menjawab tantangan moderasi konten digital di Indonesia dengan sistem yang tidak hanya akurat, tetapi juga transparan dan dapat dipertanggungjawabkan.

---

## 📝 Latar Belakang

Di era media sosial, ujaran kebencian tidak lagi disampaikan secara eksplisit melalui teks saja. Meme sering digunakan untuk menyamarkan:

* Sarkasme
* Penghinaan terselubung
* Dehumanisasi kelompok tertentu
* Simbol visual bermuatan ideologis

Model AI berbasis teks saja sering gagal memahami konteks visual tersebut. Oleh karena itu, dibutuhkan pendekatan **multimodal** yang mampu:

* "Membaca" teks
* "Melihat" gambar
* Memahami hubungan semantik di antara keduanya

**IndoMeme-XAI** menggabungkan kedua modalitas ini untuk menghasilkan prediksi yang lebih akurat dan kontekstual.

---

## 💡 Mengapa Explainable AI (XAI)?

Sebagian besar model Deep Learning bersifat *black box* — menghasilkan prediksi tanpa memberikan alasan yang jelas. Dalam konteks hukum dan moderasi konten, transparansi sangat penting.

Sistem ini menggunakan **Cross-Attention Mechanism** untuk:

1. **Menunjukkan Bukti Visual**
   Menghasilkan *heatmap* yang menandai area gambar paling berkontribusi terhadap klasifikasi.

2. **Menjelaskan Koneksi Semantik**
   Memperlihatkan hubungan antara kata tertentu (misalnya penyebutan kelompok) dengan objek visual dalam meme.

3. **Meningkatkan Transparansi**
   Memberikan dasar rasional bagi moderator sebelum mengambil keputusan penghapusan konten.

---

## 🏗️ Arsitektur Teknologi

Proyek ini mengintegrasikan dua model *state-of-the-art*:

### 🔤 Natural Language Processing (NLP)

Menggunakan **IndoBERT** untuk memahami:

* Struktur bahasa Indonesia
* Bahasa gaul
* Konteks sosial media lokal

### 🖼️ Computer Vision

Menggunakan **Vision Transformer (ViT)** yang:

* Membagi gambar menjadi *patch*
* Mempelajari pola visual secara kontekstual

### 🔀 Multimodal Fusion

Menggabungkan teks dan gambar melalui:

* *Attention Layer*
* *Cross-Attention Mechanism*
* Representasi fitur terpadu untuk klasifikasi akhir

---

## 📁 Manajemen Data & Eksperimen

Struktur proyek dirancang untuk menjaga integritas eksperimen dan replikasi riset:

### 1️⃣ Data Acquisition

* Pengambilan data dari platform media sosial Indonesia
* Kurasi konten multimodal (teks + gambar)

### 2️⃣ Annotation

* Pelabelan manual
* Pengawasan kualitas (*quality control*)
* Validasi *ground truth*

### 3️⃣ Experimentation Pipeline

* Preprocessing lokal
* Training skala besar di lingkungan cloud (Kaggle)
* Evaluasi dan analisis interpretabilitas

---

## 📊 Target dan Kontribusi

Proyek ini menargetkan standar riset internasional dengan fokus pada:

### ✅ Akurasi Tinggi

Membedakan antara:

* Kritik yang sah
* Sarkasme
* Ujaran kebencian eksplisit maupun implisit

### 🔍 Interpretability

Menghasilkan penjelasan visual yang:

* Konsisten
* Dapat dipahami manusia
* Dapat digunakan sebagai bukti pendukung

### 🌍 Dampak Sosial

* Mendukung ekosistem digital Indonesia yang lebih sehat
* Menjadi referensi teknologi moderasi berbasis AI
* Berkontribusi pada pengembangan riset AI nasional

---

## 🚀 Visi Proyek

IndoMeme-XAI bukan hanya sistem klasifikasi, tetapi fondasi menuju sistem moderasi konten yang:

* Adil
* Transparan
* Bertanggung jawab
* Kontekstual terhadap budaya Indonesia