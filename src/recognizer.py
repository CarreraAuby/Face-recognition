import numpy as np
from .preprocessing import preprocessing_gambar
from .distance import euclidean_distance


def kenali_wajah(test_image_path, model, threshold=8000):
    """
    Kenali wajah dari gambar test dengan mencocokkannya ke database training.

    Cara kerja:
    1. Load & preprocessing gambar test
    2. Kurangi mean face
    3. Proyeksikan ke ruang eigenface -> dapat koordinat test
    4. Hitung jarak ke SEMUA koordinat training image
    5. Ambil yang jaraknya paling kecil = paling mirip
    6. Bandingkan jarak terkecil dengan threshold:
       - jarak <= threshold -> wajah DIKENALI
       - jarak >  threshold -> wajah TIDAK DIKENALI

    Parameter:
        test_image_path : path ke gambar yang mau dikenali
        model           : dictionary hasil dari fungsi training()
        threshold       : batas jarak maksimal
                          kalau jarak > threshold = tidak dikenali
                          nilai default 8000, bisa diubah sesuai eksperimen

    Return:
        hasil : dictionary berisi:
            {
                'dikenali'   : True/False,
                'pesan'      : keterangan hasil,
                'jarak'      : nilai jarak minimum,
                'label'      : nama orang (kalau dikenali, else None),
                'path_gambar': path foto paling mirip (kalau dikenali, else None)
            }

    Contoh pakai:
        hasil = kenali_wajah("foto_test.jpg", model, threshold=8000)
        if hasil['dikenali']:
            print(f"Ini adalah: {hasil['label']}")
        else:
            print("Wajah tidak dikenali")
    """
   
    img_vektor = preprocessing_gambar(test_image_path)

    if img_vektor is None:
        return {
            'dikenali'   : False,
            'pesan'      : f'Gagal membaca gambar: {test_image_path}',
            'jarak'      : float('inf'),
            'label'      : None,
            'path_gambar': None
        }

    img_vektor = img_vektor.astype(float)

    centered_test = img_vektor - model['mean_face']

    koordinat_test = np.dot(model['eigenfaces'], centered_test)

    jarak_minimum = float('inf')
    label_terbaik = None
    index_terbaik = None
    path_terbaik  = None

    for i in range(len(model['proyeksi'])):
        jarak = euclidean_distance(koordinat_test, model['proyeksi'][i])

        if jarak < jarak_minimum:
            jarak_minimum = jarak
            label_terbaik = model['labels'][i]
            index_terbaik = i
            path_terbaik  = model['image_paths'][i]

    if jarak_minimum > threshold:
        return {
            'dikenali'   : False,
            'pesan'      : 'Wajah tidak ditemukan dalam database',
            'jarak'      : jarak_minimum,
            'label'      : None,
            'path_gambar': None
        }
    else:
        return {
            'dikenali'   : True,
            'pesan'      : f'Wajah dikenali sebagai: {label_terbaik}',
            'jarak'      : jarak_minimum,
            'label'      : label_terbaik,
            'index'      : index_terbaik,
            'path_gambar': path_terbaik
        }
