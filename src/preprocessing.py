# preprocessing.py
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
# Bagian: Preprocessing Gambar
#
# File ini khusus untuk memproses gambar mentah
# sebelum dimasukkan ke algoritma eigenface.
# Proses: buka gambar -> grayscale -> resize -> flatten

import numpy as np
from PIL import Image

# ukuran standar semua gambar akan diresize ke ukuran ini
# semua gambar harus sama ukurannya supaya bisa dibandingkan
IMG_SIZE = (100, 100)


def preprocessing_gambar(image_path):
    """
    Proses satu gambar dari path menjadi vektor 1D siap pakai.

    Langkah:
    1. Buka gambar dari path
    2. Ubah ke grayscale (hitam putih)
    3. Resize ke ukuran standar (100x100)
    4. Ubah ke array numpy
    5. Flatten jadi vektor 1D panjang 10000

    Parameter:
        image_path : path/lokasi file gambar

    Return:
        vektor : array 1D panjang 10000, atau None kalau gagal

    Contoh pakai:
        vektor = preprocessing_gambar("dataset/orang1/foto1.jpg")
    """
    try:
        # langkah 1: buka gambar
        img = Image.open(image_path)

        # langkah 2: ubah ke grayscale
        # grayscale = hitam putih, nilai tiap pixel 0-255
        # RGB punya 3 channel (merah, hijau, biru)
        # grayscale cukup 1 channel -> lebih ringan
        img = img.convert('L')

        # langkah 3: resize ke ukuran standar
        # semua gambar harus sama ukurannya
        img = img.resize(IMG_SIZE)

        # langkah 4: ubah ke array numpy
        img_array = np.array(img, dtype=float)
        # img_array sekarang shape (100, 100)

        # langkah 5: flatten -> jadi 1D
        # dari matriks 100x100 jadi array [a1, a2, a3, ..., a10000]
        vektor = img_array.flatten()
        # vektor sekarang shape (10000,)

        return vektor

    except Exception as e:
        print(f"  Gagal proses gambar {image_path}: {e}")
        return None


def get_img_size():
    """
    Return ukuran standar gambar yang dipakai.
    Dipanggil oleh file lain yang butuh info ukuran gambar.
    """
    return IMG_SIZE