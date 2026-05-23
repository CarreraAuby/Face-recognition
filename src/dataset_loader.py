# dataset_loader.py
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
# Bagian: Dataset Loader
#
# File ini khusus untuk membaca dan memuat semua gambar
# dari folder dataset ke dalam bentuk array numpy.

import numpy as np
import os
from .preprocessing import preprocessing_gambar, get_img_size


def load_images_from_folder(dataset_folder):
    """
    Baca semua gambar dari folder dataset.

    Struktur folder dataset yang diharapkan:
        dataset/
        ├── nama_orang_1/
        │   ├── foto1.jpg
        │   └── foto2.jpg
        └── nama_orang_2/
            └── foto1.jpg

    Setiap subfolder = 1 orang.
    Nama subfolder = label / nama orang tersebut.

    Parameter:
        dataset_folder : path ke folder dataset

    Return:
        images      : array numpy shape (N, 10000)
        labels      : list nama orang, panjang N
        image_paths : list path asli gambar, panjang N

    Contoh pakai:
        images, labels, paths = load_images_from_folder("dataset")
    """
    images      = []
    labels      = []
    image_paths = []

    # --- cek folder ada atau tidak ---
    if not os.path.exists(dataset_folder):
        print(f"ERROR: Folder '{dataset_folder}' tidak ditemukan!")
        print(f"Pastikan folder dataset ada di lokasi yang benar.")
        return None, None, None

    # --- list semua subfolder ---
    subfolders = sorted(os.listdir(dataset_folder))
    print(f"Ditemukan {len(subfolders)} item di folder dataset")

    for person_name in subfolders:
        person_path = os.path.join(dataset_folder, person_name)

        # skip kalau bukan folder (misal ada file .txt dll)
        if not os.path.isdir(person_path):
            continue

        # baca semua file di folder orang ini
        file_list = os.listdir(person_path)
        count = 0

        for filename in file_list:
            # hanya proses file gambar
            ext = filename.lower().split('.')[-1]
            if ext not in ['jpg', 'jpeg', 'png', 'bmp']:
                continue

            full_path = os.path.join(person_path, filename)

            # panggil fungsi preprocessing dari preprocessing.py
            vektor = preprocessing_gambar(full_path)

            # kalau berhasil, masukkan ke list
            if vektor is not None:
                images.append(vektor)
                labels.append(person_name)
                image_paths.append(full_path)
                count += 1

        if count > 0:
            print(f"  {person_name}: {count} gambar")

    # --- cek apakah ada gambar yang berhasil dibaca ---
    if len(images) == 0:
        print("ERROR: Tidak ada gambar yang berhasil dibaca!")
        print("Cek apakah format gambar sudah benar (jpg/png/bmp)")
        return None, None, None

    jumlah_orang = len(set(labels))
    print(f"\nTotal: {len(images)} gambar dari {jumlah_orang} orang berhasil dimuat")

    return np.array(images), labels, image_paths