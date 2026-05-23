# cache_manager.py
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
# Bagian: Cache Manager (Simpan & Load Hasil Training)
#
# File ini berisi fungsi untuk menyimpan dan memuat hasil training.
#
# MASALAH TANPA CACHE:
#   Setiap kali program dibuka -> training ulang dari awal -> lama!
#
# SOLUSI DENGAN CACHE:
#   Training pertama  -> simpan hasil ke file .pkl
#   Training berikutnya -> langsung load dari file .pkl -> cepat!
#
# Ilustrasi:
#   Tanpa cache: buka program -> tunggu 5 menit training -> bisa pakai
#   Dengan cache: buka program -> load 3 detik -> langsung bisa pakai

import pickle
import os


# nama file cache default
CACHE_FILE = "model_cache.pkl"


def simpan_model(model, nama_file=CACHE_FILE):
    """
    Simpan hasil training ke file .pkl supaya tidak perlu training ulang.

    File .pkl adalah file biner Python (pickle format).
    Bisa menyimpan hampir semua tipe data Python, termasuk
    dictionary, array numpy, list, dll.

    Parameter:
        model     : dictionary hasil dari fungsi training()
        nama_file : nama file tempat menyimpan (default: model_cache.pkl)

    Return:
        True  : kalau berhasil disimpan
        False : kalau gagal

    Contoh pakai:
        simpan_model(model)
        simpan_model(model, "backup_model.pkl")
    """
    try:
        # 'wb' = write binary (tulis dalam format biner)
        with open(nama_file, 'wb') as f:
            pickle.dump(model, f)

        # hitung ukuran file yang tersimpan
        ukuran = os.path.getsize(nama_file)
        ukuran_mb = ukuran / (1024 * 1024)  # convert ke MB

        print(f"Model berhasil disimpan ke '{nama_file}'")
        print(f"Ukuran file: {ukuran_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"Gagal menyimpan model: {e}")
        return False


def load_model(nama_file=CACHE_FILE):
    """
    Load hasil training dari file .pkl yang sudah tersimpan.

    Parameter:
        nama_file : nama file yang mau di-load (default: model_cache.pkl)

    Return:
        model : dictionary hasil training, atau None kalau gagal/tidak ada

    Contoh pakai:
        model = load_model()
        model = load_model("backup_model.pkl")
    """
    # cek dulu file-nya ada atau tidak
    if not os.path.exists(nama_file):
        print(f"File cache '{nama_file}' tidak ditemukan.")
        print(f"Perlu training dulu sebelum bisa load model.")
        return None

    try:
        # 'rb' = read binary (baca dalam format biner)
        with open(nama_file, 'rb') as f:
            model = pickle.load(f)

        print(f"Model berhasil di-load dari '{nama_file}'")
        print(f"Info model:")
        print(f"  Jumlah gambar training : {len(model['labels'])}")
        print(f"  Jumlah orang           : {len(set(model['labels']))}")
        print(f"  Jumlah komponen        : {model['jumlah_komponen']}")
        return model

    except Exception as e:
        print(f"Gagal load model: {e}")
        print(f"Coba hapus file '{nama_file}' dan training ulang.")
        return None


def cek_cache_ada(nama_file=CACHE_FILE):
    """
    Cek apakah file cache sudah ada atau belum.

    Return:
        True  : file cache ada
        False : file cache belum ada

    Contoh pakai:
        if cek_cache_ada():
            model = load_model()
        else:
            model = training("dataset")
            simpan_model(model)
    """
    return os.path.exists(nama_file)


def hapus_cache(nama_file=CACHE_FILE):
    """
    Hapus file cache.
    Berguna kalau mau training ulang dengan dataset yang baru/berbeda.

    Parameter:
        nama_file : nama file cache yang mau dihapus

    Return:
        True  : berhasil dihapus
        False : gagal atau file tidak ada
    """
    if not os.path.exists(nama_file):
        print(f"File cache '{nama_file}' tidak ada, tidak perlu dihapus.")
        return False

    try:
        os.remove(nama_file)
        print(f"File cache '{nama_file}' berhasil dihapus.")
        print(f"Jalankan training lagi untuk membuat cache baru.")
        return True

    except Exception as e:
        print(f"Gagal menghapus cache: {e}")
        return False