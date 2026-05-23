# eigenface.py  (versi dalam folder src)
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
# Bagian: Algoritma Inti Eigenface
#
# File ini berisi semua perhitungan matematika inti:
# - Mean face
# - Covariance matrix
# - Eigenvalue & Eigenvector (MANUAL - tidak pakai library)
# - Pembentukan Eigenface
# - Proyeksi gambar

import numpy as np


# =============================================================
# BAGIAN A - MEAN FACE
# =============================================================

def hitung_mean_face(images):
    """
    Hitung rata-rata dari semua gambar (mean face).
    Hasilnya adalah 'wajah rata-rata' dari seluruh dataset.

    Ilustrasi:
        Foto 1: [120, 80, 200, ...]
        Foto 2: [100, 90, 180, ...]
        Foto 3: [110, 70, 190, ...]
        Mean  : [110, 80, 190, ...]  <- rata-rata tiap posisi pixel

    Parameter:
        images : array shape (N, D)  N=jumlah gambar, D=jumlah pixel

    Return:
        mean_face : array shape (D,)
    """
    mean_face = np.mean(images, axis=0)
    return mean_face


def kurangi_mean(images, mean_face):
    """
    Kurangi setiap gambar dengan mean face (normalisasi/centering).
    Tujuan: fokus ke perbedaan antar wajah, bukan nilai absolutnya.

    Ilustrasi:
        Foto 1:  [120, 80, 200]
        Mean:    [110, 80, 190]
        Hasil:   [ 10,  0,  10]  <- "penyimpangan dari rata-rata"

    Parameter:
        images    : array shape (N, D)
        mean_face : array shape (D,)

    Return:
        centered : array shape (N, D)
    """
    centered = images - mean_face
    return centered


# =============================================================
# BAGIAN B - COVARIANCE MATRIX
# =============================================================

def hitung_covariance_matrix(centered_images):
    """
    Hitung covariance matrix menggunakan trik Turk & Pentland.

    Kenapa pakai trik ini?
    - Covariance matrix normal ukurannya D x D (10000 x 10000) -> terlalu besar!
    - Dengan trik ini ukurannya N x N (jumlah gambar x jumlah gambar) -> jauh lebih kecil

    Rumus trik: C = A * A^T   (bukan A^T * A seperti biasanya)

    Parameter:
        centered_images : array shape (N, D)

    Return:
        C : covariance matrix shape (N, N)
    """
    A = centered_images          # shape: (N, D)
    C = np.dot(A, A.T)           # shape: (N, N)
    return C


# =============================================================
# BAGIAN C - EIGENVALUE & EIGENVECTOR (MANUAL - WAJIB)
# =============================================================

def normalisasi_vektor(v):
    """
    Normalisasi vektor supaya panjangnya = 1 (unit vector).

    Rumus: v_norm = v / ||v||
    dimana ||v|| = sqrt(v[0]^2 + v[1]^2 + ... + v[n]^2)

    MANUAL - tidak pakai library apapun
    """
    total = 0.0
    for i in range(len(v)):
        total += v[i] * v[i]

    panjang = total ** 0.5

    # hindari pembagian dengan nol
    if panjang < 1e-10:
        return v

    return v / panjang


def power_iteration(matrix, max_iter=500, toleransi=1e-8):
    """
    Cari eigenvector TERBESAR menggunakan metode Power Iteration.

    Cara kerja (ilustrasi):
        Mulai: vektor random  [0.3, 0.7, 0.1, ...]
        Iter 1: kalikan matrix -> normalisasi -> [0.5, 0.4, 0.2, ...]
        Iter 2: kalikan matrix -> normalisasi -> [0.6, 0.3, 0.2, ...]
        ...
        Iter N: [0.71, 0.22, 0.19, ...]  <- sudah konvergen = eigenvector!

    Parameter:
        matrix    : matrix persegi N x N
        max_iter  : batas maksimal iterasi (default 500)
        toleransi : batas konvergensi (default 1e-8)

    Return:
        eigenvalue  : nilai eigen (skalar)
        eigenvector : vektor eigen (array 1D)

    MANUAL - tidak pakai numpy.linalg atau sejenisnya
    """
    n = matrix.shape[0]

    # mulai dari vektor random yang sudah di-seed
    # seed(42) supaya hasilnya sama setiap kali dijalankan
    np.random.seed(42)
    b = np.random.rand(n)
    b = normalisasi_vektor(b)

    eigenvalue_lama = 0.0

    for iterasi in range(max_iter):
        # langkah 1: kalikan matrix dengan vektor
        b_baru = np.dot(matrix, b)

        # langkah 2: hitung eigenvalue pakai Rayleigh quotient
        # lambda = b^T * A * b  (b sudah ternormalisasi jadi penyebut = 1)
        eigenvalue = np.dot(b, b_baru)


        # langkah 3: normalisasi vektor baru
        b_baru = normalisasi_vektor(b_baru)

        # langkah 4: hitung selisih vektor lama vs baru
        # kalau selisihnya sudah sangat kecil = konvergen = berhenti
        selisih = np.linalg.norm(b_baru - b)

        b = b_baru

        if abs(eigenvalue - eigenvalue_lama) < toleransi and selisih < toleransi:
            break

        eigenvalue_lama = eigenvalue

    # hitung eigenvalue final yang lebih akurat
    b_hasil = np.dot(matrix, b)
    eigenvalue_final = 0.0
    for i in range(n):
        eigenvalue_final += b[i] * b_hasil[i]

    return eigenvalue_final, b


def deflasi_matrix(matrix, eigenvalue, eigenvector):
    """
    Deflasi matrix: hapus pengaruh eigenvector yang sudah ditemukan.
    Tujuan: supaya iterasi berikutnya menemukan eigenvector yang BERBEDA.

    Rumus: M_baru = M - lambda * (v * v^T)
    dimana lambda = eigenvalue, v = eigenvector

    Tanpa deflasi, power iteration akan selalu menemukan eigenvector yang sama!

    Parameter:
        matrix      : matrix saat ini
        eigenvalue  : eigenvalue yang baru ditemukan
        eigenvector : eigenvector yang baru ditemukan

    Return:
        matrix_baru : matrix setelah pengaruh eigenvector dihapus

    MANUAL - tidak pakai library
    """
    
    # np.outer melakukan hal yang sama persis
    # tapi jauh lebih cepat karena dioptimasi di level C
    outer = np.outer(eigenvector, eigenvector)
    matrix_baru = matrix - eigenvalue * outer
    return matrix_baru


def hitung_eigenvectors(cov_matrix, jumlah_komponen):
    """
    Hitung sejumlah eigenvector terbesar dari covariance matrix.
    Gabungan power iteration (cari 1 eigenvector) + deflasi (hapus pengaruhnya).

    Parameter:
        cov_matrix       : covariance matrix N x N
        jumlah_komponen  : berapa eigenvector yang mau dicari

    Return:
        eigenvalues  : array shape (k,)
        eigenvectors : array shape (k, N)

    MANUAL - tidak pakai numpy.linalg.eig atau sejenisnya
    """
    eigenvalues  = []
    eigenvectors = []

    # copy supaya matrix asli tidak berubah
    M = cov_matrix.copy().astype(float)

    print(f"  Menghitung {jumlah_komponen} eigenvector...")

    for i in range(jumlah_komponen):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i+1}/{jumlah_komponen}")

        # cari eigenvector terbesar dari matrix saat ini
        eigenval, eigenvec = power_iteration(M)

        eigenvalues.append(eigenval)
        eigenvectors.append(eigenvec)

        # deflasi: hapus pengaruh eigenvector yang baru ditemukan
        M = deflasi_matrix(M, eigenval, eigenvec)

    print(f"  Selesai! {jumlah_komponen} eigenvector berhasil dihitung.")

    return np.array(eigenvalues), np.array(eigenvectors)


# =============================================================
# BAGIAN D - BENTUK EIGENFACE & PROYEKSI
# =============================================================

def bentuk_eigenfaces(centered_images, eigenvectors_kecil):
    """
    Konversi eigenvector dari ruang kecil (N dimensi) ke ruang gambar (D dimensi).
    Hasil konversi inilah yang disebut 'eigenface'.

    Kenapa perlu konversi?
    Eigenvector yang kita dapat dari covariance matrix N x N
    berukuran N (jumlah gambar). Tapi gambar kita berukuran D (10000 pixel).
    Jadi perlu dikonversi supaya ukurannya cocok.

    Rumus: eigenface = A^T * v  (lalu dinormalisasi)
    dimana A = centered_images, v = eigenvector kecil

    Parameter:
        centered_images    : array shape (N, D)
        eigenvectors_kecil : array shape (k, N)

    Return:
        eigenfaces : array shape (k, D)
    """
    eigenfaces = []

    for v in eigenvectors_kecil:
        # A^T shape: (D, N)
        # v shape: (N,)
        # hasil: (D,)
        ef = np.dot(centered_images.T, v)

        # normalisasi
        ef = normalisasi_vektor(ef)

        eigenfaces.append(ef)

    return np.array(eigenfaces)  # shape: (k, D)


def proyeksi_gambar(centered_images, eigenfaces):
    """
    Proyeksikan semua gambar ke ruang eigenface.
    Hasilnya adalah koordinat ringkas tiap gambar.

    Ilustrasi:
        Gambar asli:  10000 angka (pixel)
              ↓ proyeksi
        Koordinat:    50 angka  <- jauh lebih ringkas!

    Inilah yang nanti dibandingkan saat pencocokan wajah.

    Parameter:
        centered_images : array shape (N, D)
        eigenfaces      : array shape (k, D)

    Return:
        koordinat : array shape (N, k)
    """
    # (N, D) dot (D, k) = (N, k)
    koordinat = np.dot(centered_images, eigenfaces.T)
    return koordinat