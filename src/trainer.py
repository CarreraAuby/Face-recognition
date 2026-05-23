# trainer.py
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
# Bagian: Training
#
# File ini berisi fungsi training utama yang menggabungkan
# semua langkah dari load gambar sampai proyeksi eigenface.

from .dataset_loader import load_images_from_folder
from .eigenface import (
    hitung_mean_face,
    kurangi_mean,
    hitung_covariance_matrix,
    hitung_eigenvectors,
    bentuk_eigenfaces,
    proyeksi_gambar
)
from .preprocessing import get_img_size


def training(dataset_folder, jumlah_komponen=50):
    """
    Fungsi utama untuk melatih model eigenface.

    Menggabungkan semua langkah secara berurutan:
    1. Load semua gambar dari dataset
    2. Hitung mean face (wajah rata-rata)
    3. Kurangi mean dari semua gambar (normalisasi)
    4. Hitung covariance matrix
    5. Hitung eigenvector (MANUAL)
    6. Bentuk eigenface
    7. Proyeksikan semua training image ke ruang eigenface

    Parameter:
        dataset_folder  : path ke folder dataset
                          contoh: "dataset" atau "C:/Users/Kamu/dataset"
        jumlah_komponen : berapa eigenface yang dipakai
                          semakin banyak = semakin akurat tapi semakin lambat
                          default: 50, disarankan antara 30-100

    Return:
        model : dictionary berisi semua data hasil training
                {
                    'mean_face'       : array wajah rata-rata,
                    'eigenfaces'      : array eigenface,
                    'eigenvalues'     : array nilai eigen,
                    'proyeksi'        : koordinat semua training image,
                    'labels'          : list nama orang,
                    'image_paths'     : list path gambar,
                    'img_size'        : ukuran gambar (100, 100),
                    'jumlah_komponen' : jumlah komponen yang dipakai
                }
                Return None kalau training gagal.

    Contoh pakai:
        model = training("dataset", jumlah_komponen=50)
        if model is not None:
            print("Training berhasil!")
    """
    print("=" * 50)
    print("MEMULAI PROSES TRAINING")
    print("=" * 50)

    # -----------------------------------------------
    # LANGKAH 1: Load gambar
    # -----------------------------------------------
    print("\n[1/7] Loading gambar dari dataset...")
    images, labels, image_paths = load_images_from_folder(dataset_folder)

    if images is None:
        print("Training gagal! Periksa folder dataset.")
        return None

    N = len(images)         # jumlah total gambar
    D = images.shape[1]     # panjang vektor (100x100 = 10000)
    print(f"      {N} gambar, {D} pixel per gambar")

    # sesuaikan jumlah komponen supaya tidak melebihi jumlah gambar
    if jumlah_komponen >= N:
        jumlah_komponen = N - 1
        print(f"      jumlah_komponen disesuaikan menjadi {jumlah_komponen}")

    # -----------------------------------------------
    # LANGKAH 2: Hitung mean face
    # -----------------------------------------------
    print("\n[2/7] Menghitung mean face...")
    mean_face = hitung_mean_face(images)
    print(f"      Selesai. Shape: {mean_face.shape}")

    # -----------------------------------------------
    # LANGKAH 3: Kurangi mean
    # -----------------------------------------------
    print("\n[3/7] Normalisasi (kurangi mean)...")
    centered = kurangi_mean(images, mean_face)
    print(f"      Selesai. Shape: {centered.shape}")

    # -----------------------------------------------
    # LANGKAH 4: Covariance matrix
    # -----------------------------------------------
    print("\n[4/7] Menghitung covariance matrix...")
    cov_matrix = hitung_covariance_matrix(centered)
    print(f"      Selesai. Shape: {cov_matrix.shape} ({N}x{N})")

    # -----------------------------------------------
    # LANGKAH 5: Eigenvector (ini yang paling lama)
    # -----------------------------------------------
    print(f"\n[5/7] Menghitung {jumlah_komponen} eigenvector (butuh waktu)...")
    eigenvalues, eigenvectors_kecil = hitung_eigenvectors(cov_matrix, jumlah_komponen)
    print(f"      Selesai.")

    # -----------------------------------------------
    # LANGKAH 6: Bentuk eigenface
    # -----------------------------------------------
    print("\n[6/7] Membentuk eigenface...")
    eigenfaces = bentuk_eigenfaces(centered, eigenvectors_kecil)
    print(f"      Selesai. Shape: {eigenfaces.shape}")

    # -----------------------------------------------
    # LANGKAH 7: Proyeksi semua training image
    # -----------------------------------------------
    print("\n[7/7] Memproyeksikan training images...")
    proyeksi = proyeksi_gambar(centered, eigenfaces)
    print(f"      Selesai. Shape: {proyeksi.shape}")

    print("\n" + "=" * 50)
    print("TRAINING SELESAI!")
    print("=" * 50)

    # kumpulkan semua hasil ke dalam dictionary
    model = {
        'mean_face'       : mean_face,
        'eigenfaces'      : eigenfaces,
        'eigenvalues'     : eigenvalues,
        'proyeksi'        : proyeksi,
        'labels'          : labels,
        'image_paths'     : image_paths,
        'img_size'        : get_img_size(),
        'jumlah_komponen' : jumlah_komponen
    }

    return model