# main.py
# Tugas Project Aljabar Linear - Face Recognition
# Nama: [Nama Kamu]
# NIM : [NIM Kamu]
#
# File ini adalah file UTAMA yang dijalankan pertama kali.
# File ini menghubungkan semua bagian:
#   - src/trainer.py      -> training model
#   - src/recognizer.py   -> pencocokan wajah
#   - src/cache_manager.py -> simpan & load model
#
# Cara menjalankan:
#   python main.py
#
# Struktur folder project:
#   project_eigenface/
#   ├── main.py              <- FILE INI (jalankan ini)
#   ├── dataset/             <- folder dataset
#   └── src/
#       ├── __init__.py
#       ├── preprocessing.py
#       ├── dataset_loader.py
#       ├── eigenface.py
#       ├── distance.py
#       ├── cache_manager.py
#       ├── trainer.py
#       └── recognizer.py

from src.trainer import training
from src.recognizer import kenali_wajah
from src.cache_manager import simpan_model, load_model, cek_cache_ada, hapus_cache


# ============================================================
# PENGATURAN - UBAH SESUAI KEBUTUHAN
# ============================================================

DATASET_FOLDER   = "dataset"   # nama folder dataset kamu
JUMLAH_KOMPONEN  = 50          # jumlah eigenface (30-100 disarankan)
THRESHOLD        = 8000        # batas jarak pencocokan
CACHE_FILE       = "model_cache.pkl"  # nama file cache


# ============================================================
# FUNGSI UTAMA
# ============================================================

def jalankan_training(paksa_ulang=False):
    """
    Jalankan training dengan fitur cache.

    Kalau cache sudah ada -> load dari cache (cepat!)
    Kalau cache belum ada -> training dari awal, lalu simpan cache

    Parameter:
        paksa_ulang : kalau True, training ulang walau cache sudah ada
                      pakai ini kalau dataset sudah diupdate
    """

    # cek apakah perlu training ulang
    if paksa_ulang and cek_cache_ada(CACHE_FILE):
        print("Menghapus cache lama karena paksa_ulang=True...")
        hapus_cache(CACHE_FILE)

    # kalau cache sudah ada, langsung load
    if cek_cache_ada(CACHE_FILE):
        print("Cache ditemukan! Loading model dari cache...")
        print("(Tidak perlu training ulang)")
        model = load_model(CACHE_FILE)
        if model is not None:
            return model
        else:
            print("Cache rusak, akan training ulang...")

    # kalau tidak ada cache, training dari awal
    print("Belum ada cache. Memulai training dari awal...")
    model = training(DATASET_FOLDER, JUMLAH_KOMPONEN)

    if model is None:
        print("Training gagal!")
        return None

    # simpan hasil training ke cache
    print("\nMenyimpan model ke cache...")
    simpan_model(model, CACHE_FILE)
    print("Cache tersimpan! Lain kali tidak perlu training ulang.")

    return model


def jalankan_pencocokan(model, test_image_path):
    """
    Jalankan pencocokan wajah dari satu gambar test.

    Parameter:
        model           : hasil training
        test_image_path : path gambar yang mau dikenali
    """
    print(f"\nMencocokkan gambar: {test_image_path}")
    print("-" * 40)

    hasil = kenali_wajah(test_image_path, model, THRESHOLD)

    print(f"Hasil     : {'DIKENALI ✓' if hasil['dikenali'] else 'TIDAK DIKENALI ✗'}")
    print(f"Pesan     : {hasil['pesan']}")
    print(f"Jarak     : {hasil['jarak']:.4f}  (threshold: {THRESHOLD})")

    if hasil['dikenali']:
        print(f"Nama      : {hasil['label']}")
        print(f"Foto mirip: {hasil['path_gambar']}")

    return hasil


# ============================================================
# JALANKAN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("PROGRAM FACE RECOGNITION - EIGENFACE")
    print("Tugas Aljabar Linear")
    print("=" * 50)

    # -------------------------------------------------------
    # STEP 1: Training (atau load dari cache)
    # -------------------------------------------------------
    # Ganti paksa_ulang=True kalau mau training ulang dari awal
    # (misalnya kalau dataset sudah diupdate)
    model = jalankan_training(paksa_ulang=False)

    if model is None:
        print("\nProgram berhenti karena training gagal.")
        exit()

    # -------------------------------------------------------
    # STEP 2: Contoh pencocokan wajah
    # -------------------------------------------------------
    # Pakai gambar pertama dari dataset sebagai contoh test
    contoh_gambar = model['image_paths'][0]
    label_asli    = model['labels'][0]

    print(f"\n{'='*50}")
    print("CONTOH UJI PENCOCOKAN")
    print(f"{'='*50}")
    print(f"Label asli gambar test: {label_asli}")

    hasil = jalankan_pencocokan(model, contoh_gambar)

    # -------------------------------------------------------
    # STEP 3: Info untuk anggota GUI (orang ke-2)
    # -------------------------------------------------------
    print(f"\n{'='*50}")
    print("INFO UNTUK ANGGOTA GUI:")
    print(f"{'='*50}")
    print("Import ke file GUI kamu:")
    print("  from src.trainer    import training")
    print("  from src.recognizer import kenali_wajah")
    print("  from src.cache_manager import simpan_model, load_model, cek_cache_ada")
    print("")
    print("Atau pakai main.py ini langsung:")
    print("  from main import jalankan_training, jalankan_pencocokan")

    print("\nProgram selesai!")