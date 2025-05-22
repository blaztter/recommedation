
# Laporan Proyek Machine Learning - Muhamad Azis

## Project Overview

Penyakit jantung merupakan penyebab kematian tertinggi di dunia, dengan lebih dari 17 juta kasus setiap tahun menurut WHO (2023). Deteksi dini sangat penting untuk mencegah komplikasi lebih lanjut. Dengan pemanfaatan machine learning, kita dapat membangun sistem prediksi yang membantu tenaga medis mengenali pasien berisiko tinggi lebih awal.

Dalam proyek ini, model machine learning dibangun untuk mengklasifikasikan kemungkinan seseorang terkena penyakit jantung berdasarkan parameter klinis seperti tekanan darah, kolesterol, usia, dan detak jantung.

**Referensi:**  
- WHO. (2023). *Cardiovascular Diseases (CVDs)*. Retrieved from https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

## Business Understanding
### Problem Statement
1. Bagaimana membangun model prediktif akurat untuk mendeteksi risiko penyakit jantung menggunakan data klinis pasien?
2. Bagaimana mengidentifikasi faktor risiko dominan yang berkontribusi pada penyakit jantung?
3. Bagaimana memastikan model memiliki kemampuan generalisasi baik untuk sistem pendukung keputusan medis?


### Goals

1. Membangun model Mencapai akurasi ≥85% untuk prediksi risiko penyakit jantung.
2. Mengidentifikasi faktor risiko utama
3. Menghasilkan model yang dapat diimplementasikan

### Solution Statements

1. Menggunakan Random Forest yang menangani non-linearitas data
2. Menganalisis feature importance dari model Random Forest
3. Menyimpan model dengan joblib dan menyederhanakan preprocessing

## Data Understanding

Dataset digunakan dari Hugging Face: [https://huggingface.co/datasets/muhrafli/heart-diseases](https://huggingface.co/datasets/muhrafli/heart-diseases)

Jumlah data: 918 sampel, 12 kolom.

## Kondisi Data
1. Missing Values: Tidak ditemukan (total 918 sampel, semua kolom lengkap)
2. Duplikasi: Tidak ada data duplikat (`df.duplicated().sum() = 0`)
3. Outlier: Tidak ada outlier yang ditemukan

### Fitur-fitur dalam dataset:
1. Age`: Usia pasien
2. Sex`: Jenis kelamin (M/F)
3. `ChestPainType`: Jenis nyeri dada (ATA, NAP, ASY, TA)
4. `RestingBP`: Tekanan darah istirahat
5. `Cholesterol`: Kadar kolesterol
6. `FastingBS`: Gula darah puasa
7. `RestingECG`: Hasil EKG saat istirahat
8. `MaxHR`: Detak jantung maksimal
9. `ExerciseAngina`: Nyeri saat olahraga (Y/N)
10. `Oldpeak`: Penurunan ST setelah latihan
11. `ST_Slope`: Kemiringan segmen ST
12. `HeartDisease`: Target (0 = sehat, 1 = sakit jantung)

### Visualisasi Data (EDA)
1. Distribusi usia → mayoritas di rentang 40–60 tahun

   ![download](https://github.com/user-attachments/assets/5ec82895-96ef-43be-a004-dcc49652bff2)


   Berdasarkan histogram distribusi usia berdasarkan diagnosis penyakit jantung, terlihat bahwa penderita (biru) dan tidak penderita (oranye) memiliki pola yang berbeda. Puncak distribusi untuk tidak penderita berada di kisaran usia 50-60 tahun dengan jumlah mencapai sekitar 60 orang, sedangkan penderita menunjukkan puncak yang lebih tinggi di usia 55-65 tahun, mencapai hampir 70 orang. Ini menunjukkan bahwa risiko penyakit jantung cenderung meningkat pada kelompok usia 50-an hingga 60-an, dengan jumlah penderita lebih dominan di rentang tersebut dibandingkan tidak penderita.

2. Korelasi `Cholesterol` dan `HeartDisease` tidak kuat

   ![download](https://github.com/user-attachments/assets/63a76b7d-f84a-425a-b5f6-8d6660a9d945)


   Matriks korelasi menunjukkan bahwa Oldpeak (0.4) dan MaxHR (-0.4) memiliki korelasi paling kuat dengan penyakit jantung, menjadikannya prediktor utama, diikuti oleh usia (0.28) dan gula darah puasa (0.27) dengan korelasi positif sedang, serta kolesterol (-0.23) yang menunjukkan korelasi negatif lemah yang tidak biasa dan perlu investigasi lebih lanjut. Fitur seperti tekanan darah istirahat (0.11) memiliki pengaruh minimal, sehingga Oldpeak dan MaxHR dapat diprioritaskan untuk analisis atau model prediksi penyakit jantung.


## Data Preparation

Data preparation adalah proses penting dalam pengembangan model AI. Proses ini memastikan data yang digunakan berkualitas tinggi, relevan, dan siap untuk melatih model machine learning. Tanpa data preparation yang baik, model dapat menghasilkan performa buruk, bias, atau bahkan gagal mengenali pola yang diinginkan.

### Langkah-langkah:
1. **Encoding**: One-hot encoding pada fitur lain.
2. **Pemisahan Fitur dan Target:**   
   - Target: HeartDisease
   - Fitur: Seluruh kolom selain target
3. **Train-test split**: 80% data latih, 20% data uji.
4. **Scaling**: StandardScaler untuk `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`.

### Alasan:
1. Algoritma Random Forest tidak sensitif terhadap skala, tapi scaling membantu stabilitas.
2. Encoding diperlukan agar fitur kategorikal dapat diproses oleh model.
3. Pemisahan Fitur dan target ini sangat penting agar model dapat mempelajari hubungan antara fitur dan target dengan jelas.
4. Train-test split sebelum scaling untuk mencegah data leakage.

## Modeling
### Random Forest Classifier
Random Forest membentuk beberapa pohon keputusan menggunakan teknik bootstrapping, lalu melakukan majority voting untuk menentukan prediksi akhir.
### Mekanisme:
1. Membangun banyak pohon keputusan secara acak (bootstrap samples)
2. Setiap pohon memilih fitur secara acak untuk split
3. Prediksi akhir ditentukan oleh voting majority (untuk klasifikasi)

### Parameter:
- class_weight='balanced': Menyeimbangkan kelas target jika terjadi ketidakseimbangan (imbalanced class).
- random_state=42: Agar eksperimen dapat direproduksi dengan hasil yang konsisten.
- Parameter lainnya seperti n_estimators, max_dep, criterion menggunakan nilai default:
  - n_estimators=100: Jumlah pohon default.
  - max_depth=None: Setiap pohon dibiarkan tumbuh hingga sempurna.
  - criterion='gini': Pengukuran kualitas split default.

### Kelebihan:
1. Menangani non-linearitas dan interaksi fitur
2. Robust terhadap overfitting
3. Menyediakan feature importance

### Kekurangan:
1. Membutuhkan lebih banyak resource komputasi
2. Kurang interpretable dibanding Logistic Regression

## Evaluation

### Metrik yang digunakan:
1. **Accuracy:** `(TP+TN) / total`
2. **Precision:** `TP / (TP + FP)`
3. **Recall:** `TP / (TP + FN)`
4. **F1-score:** `2 * (Precision * Recall) / (Precision + Recall)`

### Hasil Evaluasi Model:
Random Forest Performance:
Accuracy: 0.8696
F1-Score: 0.8698

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.84      | 0.86   | 0.85     | 77      |
| 1     | 0.90      | 0.88   | 0.89     | 107     |

**Accuracy**: 0.87 (184 samples)  
**Macro Avg**: Precision = 0.87, Recall = 0.87, F1-Score = 0.87  
**Weighted Avg**: Precision = 0.87, Recall = 0.87, F1-Score = 0.87


## Analisis Dampak terhadap Business Understanding:
Model memberikan prediksi yang seimbang antara mengenali pasien yang benar-benar berisiko dan meminimalisasi false positives, yang sangat penting dalam konteks klinis. Ini menjawab Problem Statement #1 dan #3, yaitu:

- Membangun model prediktif akurat
- Memastikan generalisasi model yang baik

Dengan akurasi 86.9%, model ini melampaui target goal ≥85%.
Selain itu, feature importance dari model menunjukkan bahwa Oldpeak, MaxHR, dan Age adalah faktor risiko dominan, menjawab Problem Statement #2.

### Kesimpulan
Model Random Forest yang dibangun berhasil mencapai performa yang baik dengan akurasi >86%, memberikan insight terhadap fitur yang berkontribusi besar, dan layak diimplementasikan dalam sistem prediksi risiko penyakit jantung sebagai bagian dari sistem pendukung keputusan medis.


