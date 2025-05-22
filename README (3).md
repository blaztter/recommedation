
# Laporan Proyek Machine Learning - Muhamad Azis

## Project Overview

Dengan meningkatnya jumlah konten anime di platform digital seperti MyAnimeList, Netflix, dan Crunchyroll, pengguna seringkali kesulitan dalam memilih anime yang sesuai dengan minat mereka. Hal ini diperparah dengan semakin banyaknya judul baru setiap musim. Sistem rekomendasi menjadi solusi penting untuk membantu pengguna menjelajahi koleksi anime secara lebih efisien dan personal. Dalam proyek ini, saya mengembangkan sistem rekomendasi anime menggunakan Content-Based Filtering.

**Referensi:**  
- Netflix Technology Blog. (2016). The Netflix Recommender System: Algorithms, Business Value, and Innovation. https://netflixtechblog.com/the-netflix-recommender-system-55838468f429

## Business Understanding
### Problem Statement
1. Bagaimana menemukan anime yang sesuai dengan preferensi mereka karena terlalu banyak pilihan.?
2. Bagaimana personalisasi rekomendasi berdasarkan riwayat tontonan dan rating pengguna?


### Goals

1. Membangun sistem rekomendasi anime berbasis konten (Content-Based Filtering ) untuk pengguna baru atau tanpa riwayat rating.
2. memberikan rekomendasi yang relevan.

### Solution Statements

1. Menggunakan Content-Based Filtering untuk merekomendasikan anime berdasarkan kesamaan fitur seperti judul, genre, dan tipe. 
2. Menggunakan pola rating pengguna-anime untuk memprediksi kemungkinan suka/tidak suka pengguna terhadap anime tertentu. 

## Data Understanding

Dataset digunakan dari kaggle: https://www.kaggle.com/datasets/khioya/recomendador-de-anime

terdapat 3 folder yaitu
1. usuarios berisi 100 baris 3 kolom
2. rating berisi 1504 baris 3 kolom
3. anime berisi 12294 baris 

## Kondisi Data
1. Missing Values: Tidak ditemukan 
2. Duplikasi: Tidak ada data duplikat (`df.duplicated().sum() = 0`)
3. nilai kosong: Tidak ada

### Fitur-fitur dalam dataset:


| Kolom         | Tipe Data | Deskripsi                              |
|---------------|-----------|----------------------------------------|
| `userId`      | `int64`   | ID unik pengguna                       |
| `animeId`     | `int64`   | ID unik anime                          |
| `titulo`      | `object`  | Judul anime                            |
| `genero`      | `object`  | Genre (dipisahkan dengan koma)         |
| `tipo`        | `object`  | Jenis anime (TV, Movie, OVA, dsb.)     |
| `episodios`   | `int64`   | Jumlah episode                         |
| `rating`      | `float64` | Rating rata-rata anime                 |
| `usuarios`    | `int64`   | Jumlah pengguna yang memberikan rating |


### Visualisasi Data (EDA)
1. Distribusi Rating Pengguna

   
![download](https://github.com/user-attachments/assets/d4f0aa58-eeea-4b9b-948d-a62de6f58de2)


   terlihat bahwa rating 4 dan 5 mendominasi dengan jumlah pengguna mendekati 500, menunjukkan kecenderungan positif dalam penilaian anime. Rating rendah seperti 1 (di bawah 100) dan 2 (sekitar 150-200) jauh lebih sedikit, sementara rating 3 berada di tengah (sekitar 300-400), mengindikasikan bahwa pengguna cenderung memberikan skor tinggi, mungkin karena hanya menilai anime yang mereka sukai, yang bisa mencerminkan bias positif dalam data.

2. Top 10 Genre Anime

   ![download](https://github.com/user-attachments/assets/3d2a5422-89ae-4be0-a40b-e062cc5f0579)



   terlihat bahwa genre Comedy mendominasi dengan jumlah tertinggi mendekati 4,000, diikuti oleh Action dan Adventure dengan angka di atas 2,000, menunjukkan popularitas besar kedua genre tersebut. Fantasy, Sci-Fi, dan Drama memiliki jumlah yang lebih seimbang di kisaran 2,000, sementara Shounen, Kids, Romance, dan Slice of Life berada di bawah 2,000, dengan Slice of Life sebagai yang terendah. Ini mengindikasikan kecenderungan pengguna terhadap genre komedi dan aksi, mungkin karena kontennya lebih menghibur atau menarik secara luas.

3. Distribusi tipe anime

![download](https://github.com/user-attachments/assets/fe390cfd-58e8-4c00-9a87-909c62a0b1cd)


   tipe TV mendominasi dengan jumlah mendekati 3,500, diikuti oleh OVA (sekitar 3,000) dan Movie (sekitar 2,500), menunjukkan popularitas besar format seri dan konten tambahan. Tipe Special berada di kisaran 1,500, sementara ONA dan Music memiliki jumlah jauh lebih rendah (di bawah 1,000), mengindikasikan bahwa konten anime utama masih berfokus pada TV, OVA, dan Movie, dengan konten khusus seperti ONA dan Music kurang diminati atau diproduksi.

4. Rata-rata rating tertinggi per genre

![download](https://github.com/user-attachments/assets/71e04ed4-c2b1-4831-8152-aba6cb3537d5)


   genre Josei dan Thriller memimpin dengan rata-rata rating mendekati 7, diikuti oleh Mystery, Police, Shounen, Supernatural, Shounen Ai, Military, School, dan Romance dengan nilai serupa di kisaran 6-7. Ini menunjukkan bahwa genre yang lebih spesifik seperti Josei dan Thriller cenderung mendapatkan apresiasi lebih tinggi, sementara genre populer seperti Romance juga tetap kompetitif, mengindikasikan preferensi pengguna terhadap cerita mendalam atau tematik yang kuat.

## Data Preparation

Data preparation adalah proses penting dalam pengembangan model AI. Proses ini memastikan data yang digunakan berkualitas tinggi, relevan, dan siap untuk melatih model machine learning. Tanpa data preparation yang baik, model dapat menghasilkan performa buruk, bias, atau bahkan gagal mengenali pola yang diinginkan.

### Langkah-langkah:
1. Merge Dataset
  - Gabung dataset rating.csv dengan animes.csv berdasarkan animeId.
2. Data Cleaning
  - Hapus baris dengan nilai kosong.
  - Hilangkan duplikasi.
  - Hapus kolom rating_x, ganti nama rating_y menjadi rating.
3. Mapping ID ke Index Numerik
    
   1   user_ids = cleaned_data['userId'].unique().tolist()
   2   user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}

4. Normalisasi Rating
   - Rating dinormalisasi ke rentang 0–1 agar cocok dengan aktivasi sigmoid.
5. Buat Kolom combined_features
   - Gabungan titulo, genero, dan tipo untuk content-based filtering.
6. Pembagian Data Training dan Validasi
   - 80% untuk training, 20% untuk validasi.

### Alasan:
1. Memastikan kompatibilitas model machine learning.
2. Membersihkan noise dan data tidak relevan.
3. Mempersiapkan data untuk representasi embedding dan prediksi probabilitas.

## Modeling
### Content-Based Filtering
### Mekanisme:
1. Gunakan TF-IDF Vectorizer untuk ekstraksi fitur teks.
2. Hitung cosine similarity antar anime.

### Kelebihan:
1. Tidak memerlukan riwayat pengguna.
2. Cepat dan mudah diimplementasikan.

### Kekurangan:
1. Tidak personal; rekomendasi sama untuk semua orang.
2. Bergantung pada deskripsi dan genre yang tersedia.

## Evaluation

### Metrik Evaluasi:
1. Binary Crossentropy : Untuk loss function.
2. RMSE (Root Mean Squared Error) : Untuk mengukur akurasi prediksi.

### Hasil Evaluasi Model:
Berikut adalah ringkasan performa model berdasarkan metrik Root Mean Square Error (RMSE) untuk data pelatihan (TRAIN RMSE) dan validasi (VAL RMSE) pada dua epoch yang berbeda:

| EPOCH | TRAIN RMSE | VAL RMSE |
|-------|------------|----------|
| 1     | 0.0055     | 0.0769   |
| 100   | 0.0026     | 0.0473   |

## Catatan
- EPOCH 1: Model menunjukkan TRAIN RMSE 0.0055 dan VAL RMSE 0.0769, menandakan performa awal yang cukup baik pada data pelatihan namun ada kesenjangan dengan data validasi.
- EPOCH 100: Setelah 100 epoch, TRAIN RMSE menurun menjadi 0.0026 dan VAL RMSE menjadi 0.0473, menunjukkan peningkatan akurasi dan generalisasi model seiring waktu.
- Penurunan RMSE pada kedua set data menunjukkan bahwa model terus belajar dan mengurangi error selama pelatihan.


### Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi anime menggunakan pendekatan Content-Based Filtering, dengan hasil yang akurat, logis, dan stabil. Hasil evaluasi dan rekomendasi menunjukkan bahwa model siap digunakan dalam sistem rekomendasi nyata, seperti pada platform streaming anime atau aplikasi berbasis pengguna lainnya.

