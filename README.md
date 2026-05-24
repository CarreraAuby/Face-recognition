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
## Landasan Teori

### 1. Pengenalan Wajah

Pengenalan wajah adalah proses identifikasi seseorang berdasarkan ciri-ciri wajahnya. Secara matematis, sebuah gambar wajah berukuran `100×100` pixel dipandang sebagai sebuah **vektor** di ruang berdimensi tinggi (ℝ¹⁰⁰⁰⁰). Tantangan utamanya adalah membandingkan vektor-vektor berdimensi sangat tinggi ini secara efisien.

### 2. Principal Component Analysis (PCA)

PCA adalah teknik reduksi dimensi yang mengubah data berdimensi tinggi menjadi representasi berdimensi lebih rendah sambil mempertahankan informasi terpenting. PCA efektif untuk wajah karena wajah manusia memiliki struktur berulang (mata, hidung, mulut di posisi relatif sama), sehingga perbedaan antar wajah dapat direpresentasikan dengan sedikit angka saja.

### 3. EigenFace

EigenFace diperkenalkan oleh **Turk & Pentland (1991)**. Setiap wajah dinyatakan sebagai kombinasi linear dari sekumpulan "wajah dasar" yang disebut eigenface:

```
Wajah = mean_face + (w₁×eigenface₁) + (w₂×eigenface₂) + ... + (wₖ×eigenfaceₖ)
```

Dua wajah yang mirip akan memiliki bobot `w₁, w₂, ..., wₖ` yang mirip pula.

### 4. Trik Turk & Pentland

Covariance matrix normal berukuran **D×D** (10000×10000) — terlalu besar. Solusinya dengan membalik perkalian matriks:

```
Normal  : C = Aᵀ × A  →  ukuran D×D  (10000×10000)  ← tidak praktis
Trik    : C = A × Aᵀ  →  ukuran N×N  (misal 200×200) ← jauh lebih kecil
```

Eigenvector hasil kemudian dikonversi ke ruang gambar: `eigenface = Aᵀ × v`

### 5. Power Iteration

Metode numerik untuk mencari eigenvector dominan secara manual:

```
1. Mulai dengan vektor acak b₀
2. Kalikan dengan matriks : b_baru = A × b
3. Normalisasi            : b = b_baru / ‖b_baru‖
4. Hitung eigenvalue      : λ = bᵀ × A × b  (Rayleigh Quotient)
5. Ulangi hingga konvergen
```

### 6. Deflasi Matrix

Setelah satu eigenvector ditemukan, pengaruhnya dihilangkan agar iterasi berikutnya menemukan eigenvector yang berbeda:

```
M_baru = M - λ × (v × vᵀ)
```

### 7. Jarak Euclidean dan Kemiripan

Kemiripan dua wajah diukur dengan jarak Euclidean pada ruang eigenface:

```
d = √( Σ (aᵢ - bᵢ)² )
```

Persentase kemiripan dihitung relatif terhadap threshold:

```
similarity = max(0, (1 - d/threshold) × 100) %
```

- `d ≤ threshold` → wajah **DIKENALI**
- `d > threshold` → wajah **TIDAK DIKENALI**

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
## Instalasi

```bash
pip install customtkinter pillow numpy opencv-python
```

Siapkan folder `dataset/` — nama subfolder = nama orang, isi folder = foto-fotonya.
Format yang didukung: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## Cara Menjalankan

```bash
# Via GUI
python gui.py

# Via terminal
python main.py
```

**Urutan penggunaan GUI:**
1. **Choose Dataset Folder** — pilih folder dataset
2. **Train Model** — training eigenface dari dataset
3. **Upload Test Image** — pilih foto yang ingin dikenali
4. **Recognize Face** — lihat hasil perbandingan dan persentase kemiripan

---

## Konfigurasi

Edit di `main.py`:

```python
DATASET_FOLDER   = "dataset"         # path folder dataset
JUMLAH_KOMPONEN  = 50                # jumlah eigenface (30–100 disarankan)
THRESHOLD        = 8000              # batas jarak pencocokan
CACHE_FILE       = "model_cache.pkl"
```

Untuk training ulang setelah dataset diperbarui:
```python
model = jalankan_training(paksa_ulang=True)
```

---

## Dependencies

| Library | Kegunaan |
|---------|----------|
| `customtkinter` | Antarmuka grafis |
| `Pillow` | Baca & tampilkan gambar |
| `numpy` | Operasi matriks & array |
| `opencv-python` | Preprocessing gambar |

---
## Dokumentasi Program

https://youtu.be/LINK_YOUTUBE_KAMU
