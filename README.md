<p align="center">
  <img src="Logo uns.png" alt="Logo UNS" width="250"/>
</p>

<h1 align="center">  Face Recognition Menggunakan Eigenface </h1>

<p align="center">
  <strong>Kelompok 2 <br> informatika D <br> Universitas Sebelas Maret</strong>
</p>

---
## 👥 Anggota Kelompok
| NIM        | NAMA                           |
|:-----------|:-----------------              |
| L0125076   | Carrera Abi Saputra            |
| L0125129   | Ersandy Fahreza                |
| L0125137   | Regita Ariella Safa Wardani    |

**Dosen Pengampu:** Drs. Bambang Harjito, M.App.Sc., Ph.D.

---

## 🧑‍💻 Deskripsi Program
Sistem Pengenalan Wajah (*Face Recognition*) modern yang dibangun menggunakan implementasi murni konsep **Aljabar Linear**, khususnya **Principal Component Analysis (PCA)** dan **Eigenface**. Proyek ini dilengkapi dengan antarmuka grafis (GUI) berbasis `tkinter` yang interaktif, performa pencocokan berbasis *Cosine Similarity*, sistem *caching* data latih, serta fitur **Threshold Dinamis** untuk menguji akurasi klasifikasi secara *real-time*.

---

## 🗂️ Struktur Folder

```
project_eigenface/
├── main.py                  # File utama, hubungkan semua modul
├── gui.py                   # Antarmuka grafis (CustomTkinter)
├── model_cache.pkl          # Cache model hasil training (auto-generated)
├── dataset/                 # Folder dataset wajah
└── src/
    ├── __init__.py          # Penanda paket Python, ekspor semua fungsi
    ├── preprocessing.py     # Grayscale + resize 100×100 + flatten
    ├── dataset_loader.py    # Baca semua gambar dari subfolder dataset
    ├── eigenface.py         # Algoritma inti PCA & EigenFace (MANUAL)
    ├── distance.py          # Jarak Euclidean (MANUAL)
    ├── cache_manager.py     # Simpan & load model (.pkl)
    ├── trainer.py           # Pipeline training 7 langkah
    └── recognizer.py        # Pipeline pencocokan wajah
```

---
⚙️ Instalasi
1. Download / Clone project
bashgit clone https://github.com/username/project_eigenface.git
cd project_eigenface
2. Install dependencies
bashpip install customtkinter pillow numpy opencv-python
3. Siapkan dataset
Buat folder dataset/ dengan struktur berikut — nama subfolder = nama orang:
dataset/
├── Nama_Orang_1/
│   ├── foto1.jpg
│   └── foto2.jpg
└── Nama_Orang_2/
    └── foto1.jpg
Format gambar yang didukung: .jpg, .jpeg, .png, .bmp

🚀 Cara Menjalankan
Via GUI (disarankan)
bashpython gui.py
Via terminal
bashpython main.py

🖥️ Cara Penggunaan GUI
LangkahTombolKeterangan1Choose Dataset FolderPilih folder dataset wajah2Train ModelTraining eigenface dari dataset3Upload Test ImagePilih foto wajah yang ingin dikenali4Recognize FaceCocokkan wajah dengan dataset
Hasil ditampilkan berupa:

Foto test dan foto paling mirip dari dataset berdampingan
Persentase kemiripan (0% — 100%)
Nama orang, jarak Euclidean, dan status dikenali / tidak dikenali


🔬 Alur Algoritma Training (7 Langkah)
[1] Load gambar dari dataset
         │  load_images_from_folder()
         ▼
[2] Hitung Mean Face
         │  hitung_mean_face()  →  rata-rata tiap pixel
         ▼
[3] Normalisasi / Centering
         │  kurangi_mean()  →  gambar dikurangi mean face
         ▼
[4] Hitung Covariance Matrix
         │  hitung_covariance_matrix()  →  C = A × Aᵀ  (trik Turk & Pentland)
         ▼
[5] Hitung Eigenvalue & Eigenvector  ← MANUAL
         │  power_iteration() + deflasi_matrix()
         ▼
[6] Bentuk Eigenface
         │  bentuk_eigenfaces()  →  konversi ke ruang gambar (D dimensi)
         ▼
[7] Proyeksi Semua Training Image
         │  proyeksi_gambar()  →  koordinat ringkas tiap gambar
         ▼
    Simpan ke cache (.pkl)
Alur Pencocokan Wajah
Gambar Test
     │  preprocessing_gambar()  →  grayscale + resize + flatten
     ▼
Kurangi Mean Face
     ▼
Proyeksikan ke Ruang Eigenface
     ▼
Hitung Jarak Euclidean ke Semua Training Image  ← MANUAL
     ▼
Ambil Jarak Terkecil
     ▼
Bandingkan dengan Threshold
     ├── jarak ≤ threshold  →  DIKENALI ✓  (tampilkan nama + persentase)
     └── jarak > threshold  →  TIDAK DIKENALI ✗
