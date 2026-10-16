# Proyek Mini Deep Learning: Prediksi Diabetes

## 1. Deskripsi Proyek

Proyek ini bertujuan untuk memprediksi kemungkinan diabetes menggunakan *Pima Indians Diabetes Database* dengan pendekatan *Deep Learning*. Model utama yang digunakan adalah **Multilayer Perceptron (MLP)** yang diimplementasikan menggunakan PyTorch.

Proyek ini mencakup prapemrosesan data, pengembangan model jaringan saraf tiruan, eksperimen hyperparameter, evaluasi performa, dan perbandingan dengan beberapa model *machine learning* konvensional.

### Tujuan Proyek

- Memahami karakteristik dataset diabetes.
- Melakukan prapemrosesan data sebelum pelatihan model.
- Mengimplementasikan model MLP menggunakan PyTorch.
- Menguji pengaruh berbagai konfigurasi model dan hyperparameter.
- Mengevaluasi dan membandingkan performa model berdasarkan beberapa metrik klasifikasi.

## 2. Dataset

**Nama dataset:** Pima Indians Diabetes Database

**Sumber:** [Kaggle — Pima Indians Diabetes Database](https://www.kaggle.com/datasets/mragpavan/diabetes)

Dataset terdiri dari 768 observasi, delapan fitur masukan, dan satu variabel target, yaitu `Outcome`.

### Deskripsi Fitur

| Fitur | Deskripsi |
|---|---|
| `Pregnancies` | Jumlah kehamilan |
| `Glucose` | Konsentrasi glukosa plasma |
| `BloodPressure` | Tekanan darah diastolik |
| `SkinThickness` | Ketebalan lipatan kulit trisep |
| `Insulin` | Kadar insulin serum setelah dua jam |
| `BMI` | Indeks massa tubuh |
| `DiabetesPedigreeFunction` | Indikator riwayat diabetes dalam keluarga |
| `Age` | Usia dalam tahun |
| `Outcome` | Status diabetes: 0 = tidak diabetes, 1 = diabetes |

### Prapemrosesan Data

Tahapan prapemrosesan yang dilakukan meliputi:

- Memperlakukan nilai nol pada beberapa pengukuran fisiologis sebagai nilai yang hilang.
- Mengisi nilai yang hilang menggunakan median.
- Melakukan standardisasi fitur numerik.
- Membagi dataset menjadi data latih dan data uji menggunakan pembagian terstratifikasi.
- Menggunakan validasi silang lima lipatan (*5-fold cross-validation*) pada data latih selama eksperimen.

Pembagian dataset terdiri dari **614 data latih** dan **154 data uji**.

## 3. Metodologi

### 3.1 Model Utama: Multilayer Perceptron (MLP)

Multilayer Perceptron merupakan jaringan saraf tiruan *feed-forward* yang digunakan untuk mempelajari pola hubungan antara fitur masukan dan target klasifikasi.

Model dikembangkan menggunakan PyTorch dengan tahapan berikut:

1. Membangun model MLP sebagai model dasar (*baseline*).
2. Menguji berbagai arsitektur jaringan dan jumlah neuron.
3. Membandingkan konfigurasi fungsi aktivasi.
4. Mengevaluasi teknik regularisasi.
5. Menguji optimizer, *learning rate*, ukuran batch, dan jumlah epoch.
6. Melakukan penyetelan hyperparameter.
7. Mengevaluasi model pada data uji.

### 3.2 Eksperimen Hyperparameter

Eksperimen dilakukan untuk mengamati pengaruh beberapa parameter berikut:

- Arsitektur lapisan tersembunyi dan jumlah neuron.
- Fungsi aktivasi.
- Dropout dan *weight decay*.
- Optimizer dan *learning rate*.
- Ukuran batch dan jumlah epoch.
- Pembobotan kelas untuk menangani ketidakseimbangan kelas.

Berdasarkan konfigurasi eksperimen yang tersimpan, model MLP terpilih menggunakan satu lapisan tersembunyi dengan 16 neuron, aktivasi ReLU, dropout 0,2, optimizer RMSprop, *learning rate* 0,0005, ukuran batch 64, dan 200 epoch, dengan pembobotan kelas diaktifkan.

### 3.3 Model Pembanding

Model MLP dibandingkan dengan beberapa model *machine learning* konvensional:

- Logistic Regression
- Support Vector Machine (SVM) dengan kernel RBF
- Random Forest

## 4. Metrik Evaluasi

Performa model dievaluasi menggunakan metrik berikut:

- **Accuracy:** Persentase prediksi yang benar dari seluruh data.
- **Precision:** Proporsi prediksi positif yang benar-benar positif.
- **Recall:** Proporsi kasus positif aktual yang berhasil terdeteksi.
- **F1-score:** Rata-rata harmonik precision dan recall.
- **ROC-AUC:** Kemampuan model membedakan kelas positif dan negatif berdasarkan skor prediksi.

Evaluasi juga dilengkapi dengan visualisasi *confusion matrix*, kurva ROC, kurva pembelajaran, grafik eksperimen hyperparameter, dan analisis kepentingan fitur.

## 5. Hasil Eksperimen

Tabel berikut menunjukkan hasil eksperimen yang tersimpan pada data uji.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| MLP Baseline | 74,03% | 63,46% | 61,11% | 62,26% | 81,83% |
| MLP Tuning | 74,68% | 60,87% | 77,78% | 68,29% | 80,89% |
| Logistic Regression | 70,78% | 60,00% | 50,00% | 54,55% | 81,30% |
| SVM (RBF) | 73,38% | 63,83% | 55,56% | 59,41% | 79,63% |
| Random Forest | 73,38% | 63,27% | 57,41% | 60,19% | 81,13% |

### Analisis Hasil

- MLP setelah penyetelan hyperparameter memperoleh akurasi tertinggi di antara model yang diuji, yaitu 74,68%.
- MLP hasil tuning memperoleh recall sebesar 77,78%, lebih tinggi dibandingkan model pembanding.
- F1-score MLP meningkat dari 62,26% pada model baseline menjadi 68,29% setelah tuning.
- Logistic Regression memperoleh ROC-AUC sedikit lebih tinggi dibandingkan MLP hasil tuning.
- Hasil menunjukkan adanya pertukaran performa antara akurasi keseluruhan, kemampuan mendeteksi kasus positif, dan kemampuan membedakan kedua kelas.

Hasil ini berlaku untuk pembagian data dan konfigurasi eksperimen yang digunakan. Model belum dapat dianggap sebagai alat diagnosis medis.

## 6. Struktur Repository

Struktur repository proyek di GitHub adalah sebagai berikut:

```text
Deep-Learning-Diabetes-Prediction/
├── data/
│   └── diabetes.csv
├── results/
│   ├── plots/
│   │   ├── architecture_auc.png
│   │   ├── batch_size_auc.png
│   │   ├── confusion_matrix.png
│   │   ├── epochs_auc.png
│   │   ├── feature_importance.png
│   │   ├── learning_curve.png
│   │   ├── model_comparison.png
│   │   ├── optimizer_auc.png
│   │   ├── regularization_auc.png
│   │   └── roc_curve.png
│   ├── data_info.json
│   └── results.json
├── Deep_Learning_Group_Mini_Project.ipynb
├── README.md
├── deep_learning_group_mini_project.py
└── requirements.txt
```

### Keterangan File dan Folder

- **`data/`**: Menyimpan dataset yang digunakan dalam proyek.
  - `diabetes.csv`: Dataset Pima Indians Diabetes Database.
- **`results/`**: Menyimpan hasil eksperimen dan evaluasi model.
  - `plots/`: Berisi grafik hasil eksperimen dan evaluasi model.
  - `data_info.json`: Menyimpan informasi dataset dan pembagian data.
  - `results.json`: Menyimpan hasil evaluasi model.
- **`Deep_Learning_Group_Mini_Project.ipynb`**: Notebook untuk menjalankan eksperimen secara interaktif.
- **`README.md`**: Dokumentasi proyek.
- **`deep_learning_group_mini_project.py`**: Script Python untuk menjalankan eksperimen.
- **`requirements.txt`**: Daftar pustaka Python yang diperlukan untuk menjalankan proyek.

## 7. Instalasi dan Cara Menjalankan

### 7.1 Mengunduh Repository

Jalankan perintah berikut pada terminal:

```bash
git clone https://github.com/auliafathustsani/Deep-Learning-Diabetes-Prediction.git
cd Deep-Learning-Diabetes-Prediction
```

### 7.2 Menginstal Pustaka

Pasang pustaka yang tercantum dalam `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 7.3 Menyiapkan Dataset

Dataset tersedia pada folder `data/` dengan nama `diabetes.csv`.

Pastikan file tersebut memiliki kolom fitur yang sesuai dan kolom target bernama `Outcome`.

### 7.4 Menjalankan Notebook

Untuk menjalankan eksperimen melalui Jupyter Notebook:

```bash
jupyter notebook Deep_Learning_Group_Mini_Project.ipynb
```

### 7.5 Menjalankan Script Python

Eksperimen juga dapat dijalankan melalui script Python:

```bash
python deep_learning_group_mini_project.py
```

Pastikan lokasi dataset dan direktori penyimpanan hasil sesuai dengan konfigurasi di dalam kode.

## 8. Anggota Kelompok dan Kontribusi

| No. | Nama | NIM | Kontribusi |
|---|---|---|---|
| 1 | Ananda Auliya Rahma | 24/533691/PA/22608 | Eksplorasi dataset dan prapemrosesan data |
| 2 | Aulia Fathus Tsani | 24/534388/PA/22661 | Implementasi MLP dan penyetelan hyperparameter |
| 3 | Alifa Batrisyia Nariswari | 24/535256/PA/22707 | Evaluasi model dan analisis hasil eksperimen |
| 4 | Herlina Iin Nur Soleha | 24/541333/PA/22962 | Visualisasi hasil dan dokumentasi proyek |

*Pembagian kontribusi perlu disesuaikan dengan pekerjaan yang benar-benar dilakukan oleh masing-masing anggota.*

## 9. Referensi

1. Afiahayati. (2026). *Single Layer Perceptron to Multi Layer Perceptron* [Bahan ajar/slide kuliah]. Departemen Ilmu Komputer dan Elektronika, FMIPA, Universitas Gadjah Mada.

2. Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.

3. Pavan Kumar, D. (2020). *Pima Indians Diabetes Database* [Data set]. Kaggle. https://www.kaggle.com/datasets/mragpavan/diabetes

4. Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research, 15*, 1929–1958.

## 10. Catatan

Proyek ini dibuat untuk tujuan pembelajaran dan penelitian. Hasil prediksi model tidak dimaksudkan untuk menggantikan pemeriksaan, penilaian, atau diagnosis oleh tenaga medis profesional.
