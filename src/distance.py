def euclidean_distance(vektor1, vektor2):
    """
    Hitung jarak Euclidean antara dua vektor koordinat.

    Rumus:
        d = sqrt( (v1[0]-v2[0])^2 + (v1[1]-v2[1])^2 + ... + (v1[n]-v2[n])^2 )
        d = sqrt( sum( (v1[i] - v2[i])^2 ) )

    Ilustrasi sederhana (2 dimensi):
        v1 = [2, 3]
        v2 = [5, 7]
        d  = sqrt((2-5)^2 + (3-7)^2)
           = sqrt(9 + 16)
           = sqrt(25)
           = 5.0

    Untuk wajah (50 dimensi):
        v1 = [2.1, 5.3, 1.2, ...]  <- koordinat gambar test
        v2 = [2.0, 5.1, 1.3, ...]  <- koordinat gambar training
        d  = sqrt((2.1-2.0)^2 + (5.3-5.1)^2 + ...)
           = 0.245  <- kecil = mirip!

    Parameter:
        vektor1 : koordinat gambar pertama (array 1D)
        vektor2 : koordinat gambar kedua (array 1D)

    Return:
        jarak : bilangan float, semakin kecil = semakin mirip

    MANUAL - tidak pakai scipy.spatial.distance atau sejenisnya
    """
    
    if len(vektor1) != len(vektor2):
        print("ERROR: panjang vektor berbeda!")
        return float('inf')

    total = 0.0

    for i in range(len(vektor1)):
        selisih = vektor1[i] - vektor2[i]
        total += selisih * selisih

    jarak = total ** 0.5

    return jarak
