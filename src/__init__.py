# __init__.py
# File ini menandakan bahwa folder src adalah "paket" Python
# Dengan file ini, file lain bisa import dari folder src
# Contoh: from src.trainer import training
#
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]

from .preprocessing import preprocessing_gambar
from .dataset_loader import load_images_from_folder
from .eigenface import (
    hitung_mean_face,
    kurangi_mean,
    hitung_covariance_matrix,
    normalisasi_vektor,
    power_iteration,
    deflasi_matrix,
    hitung_eigenvectors,
    bentuk_eigenfaces,
    proyeksi_gambar
)
from .distance import euclidean_distance
from .trainer import training
from .recognizer import kenali_wajah
from .cache_manager import simpan_model, load_model