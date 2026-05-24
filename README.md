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
│   ├── Nama_Orang_1/
│   │   ├── foto1.jpg
│   │   └── foto2.jpg
│   └── Nama_Orang_2/
│       ├── foto1.jpg
│       └── foto2.jpg
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
