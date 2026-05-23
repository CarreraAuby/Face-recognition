import numpy as np
from PIL import Image

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
        img = Image.open(image_path)

        img = img.convert('L')

        img = img.resize(IMG_SIZE)

        img_array = np.array(img, dtype=float)
       
        vektor = img_array.flatten()

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
