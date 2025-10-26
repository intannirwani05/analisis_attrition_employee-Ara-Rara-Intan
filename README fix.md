 
# Employee Attrition Prediction

---

## Daftar Isi

- [Domain Proyek: Human Resource Analytics](#domain-proyek)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution Statements](#solution-statements)
  - [Project Benefits](#project-benefits)
- [Data Understanding](#data-understanding)
  - [Deskripsi Fitur](#deskripsi-fitur)
  - [Penjelasan Kontekstual Fitur]
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
  - [Pemeriksaan Data Awal](#pemeriksaan-data-awal)
  - [Pemeriksaan Missing Values dan Duplikasi](#pemeriksaan-missing-values-dan-duplikasi)
  - [Analisis Nilai Unik](#analisis-nilai-unik)
  - [Distribusi Target](#distribusi-target)
- [Data Preparation](#data-preparation)
  - [Pemisahan Fitur dan Targetl](#pemisahan-fitur-dan-target)
  - [Transformasi dan Preprocessing](#transformasi-dan-preprocessing)
- [Model Training and Hyperparameter Tuning](#model-training-and-hyperparameter-tuning)
- [Model Evaluation](#model-evaluation)
- [Conclusions](#conclusions)

---

## Domain Proyek

Manajemen sumber daya manusia merupakan salah satu aspek paling krusial dalam menjaga keberlanjutan dan daya saing suatu perusahaan. Karyawan tidak hanya berperan sebagai pelaksana operasional, tetapi juga menjadi aset strategis yang menentukan produktivitas dan inovasi organisasi. Namun, di era bisnis modern yang penuh dinamika, perusahaan menghadapi tantangan besar berupa tingginya tingkat attrition atau turnover, yaitu kondisi ketika karyawan memutuskan untuk meninggalkan perusahaan baik secara sukarela maupun karena kebijakan organisasi.

Fenomena ini menimbulkan konsekuensi seperti meningkatnya **biaya rekrutmen dan pelatihan**, berkurangnya **efisiensi kerja**, serta menurunnya **moral tim**. Faktor pemicu attrition beragam, antara lain **beban kerja dan jam lembur**, **kepuasan kerja**, **lingkungan kerja**, **kompensasi**, dan **dinamika karier**. Memahami faktor-faktor tersebut memungkinkan tim HR melakukan **intervensi preventif** seperti penyesuaian beban kerja, peningkatan kompensasi, atau kebijakan pengembangan karier.

Proyek ini berada dalam domain **Human Resource Analytics**, dengan fokus pada **analisis perilaku karyawan dan prediksi tingkat attrition** menggunakan pendekatan berbasis data.

---

## Business Understanding

### 1. Problem Statements

1. Bagaimana cara mengidentifikasi faktor-faktor yang paling berpengaruh terhadap keputusan seorang karyawan untuk keluar dari perusahaan?  
2. Bagaimana membangun model prediktif yang akurat untuk mengestimasi kemungkinan attrition menggunakan data historis karyawan?  
3. Bagaimana hasil prediksi tersebut dapat digunakan secara strategis untuk menekan angka turnover dan meningkatkan retensi karyawan?

### 2. Goals

- Mengembangkan model machine learning yang mampu **memprediksi probabilitas seorang karyawan akan keluar** dengan akurasi tinggi.  
- Mengidentifikasi variabel-variabel penting yang paling memengaruhi keputusan keluar.  
- Menyediakan hasil analisis dan rekomendasi berbasis data bagi tim HR perusahaan.

### 3. Solution Statements

- Melakukan eksplorasi awal (pemeriksaan missing/duplikasi) dan analisis ringkas untuk memahami distribusi fitur terhadap target *Attrition*.
- Melakukan **data cleaning** minimal, **encoding kategorikal**, **imputasi**, **scaling**, dan **penanganan class imbalance (SMOTE)** di dalam **`Pipeline`** agar reproducible.
- Membangun dan membandingkan beberapa algoritma pembelajaran mesin: **Logistic Regression**, **Random Forest** *(dengan opsi XGBoost/LightGBM jika paket tersedia di lingkungan eksekusi)*.
- Menggunakan **Stratified K-Fold Cross-Validation** dan **RandomizedSearchCV** untuk *hyperparameter tuning* dengan metrik utama **ROC-AUC**.

### 4. Project Benefits

- **Meningkatkan efisiensi HR**: membantu manajemen mengidentifikasi karyawan yang berisiko keluar.  
- **Menekan biaya turnover**: menurunkan kebutuhan rekrutmen dan pelatihan ulang.  
- **Mendukung keputusan strategis**: hasil prediksi bisa digunakan untuk meningkatkan kesejahteraan dan kepuasan karyawan.

---

## Data Understanding

Dataset terdiri dari tiga file utama:  
1. **train.csv** — data pelatihan berisi fitur dan label target (*Attrition*).  
2. **test.csv** — data uji tanpa label, digunakan untuk prediksi akhir.  
3. **sample_submission.csv** — format pengumpulan hasil prediksi untuk diunggah ke Kaggle.

### 1. Deskripsi Fitur
 |Nama Fitur                     |Tipe Data               |Deskripsi                                                 |
 --------------------------------------------------------------------------------------------------------------------
 |`id`                           |`object`                |ID unik karyawan untuk identifikasi.                      |
 |`Age`                          |`int64`                 |Usia karyawan.                                            |
 |`BusinessTravel`               |`object`                |Frekuensi perjalanan dinas karyawan.                      |
 |`DailyRate`                    |`int64`                 |Gaji harian.                                              |
 |`Department`                   |`object`                |Departemen tempat karyawan bekerja.                       |
 |`DistanceFromHome`             |`int64`                 |Jarak tempat tinggal karyawan ke kantor.                  |
 |`Education`                    |`int64`                 |Tingkat Pendidikan Terakhir.                              | 
 |`EducationField`               |`object`                |Bidang studi terakhir karyawan.                           |
 |`EmployeeCount`                |`int64`                 |Jumlah total karyawan dalam perusahaan.                   |
 |`EmployeeNumber`               |`int64`                 |Nomor unik karyawan dalam sistem HR.                      |
 |`EnvironmentSatisfaction`      |`int64`                 |Tingkat kepuasan terhadap lingkungan kerja.               |
 |`Gender`                       |`object`                |Jenis kelamin karyawan.                                   | 
 |`HourlyRate`                   |`int64`                 |Upah per jam.                                             |
 |`JobInvolvement`               |`int64`                 |Tingkat keterlibatan pekerjaan.                           |
 |`JobLevel`                     |`int64`                 |Level jabatan karyawan.                                   |
 |`JobRole`                      |`object`                |Posisi atau jabatan spesifik karyawan.                    |  
 |`JobSatisfaction`              |`int64`                 |Tingkat Kepuasan pekerjaan.                               |
 |`MaritalStatus`                |`object`                |Status Pernikahan.                                        |
 |`MonthlyIncome`                |`int64`                 |Gaji Bulanan Karyawan.                                    |
 |`MonthlyRate`                  |`int64`                 |Tarif kompensasi bulanan.                                 |
 |`NumCompaniesWorked`           |`int64`                 |Jumlah perusahaan tempat karyawan pernah bekerja.         | 
 |`Over18`                       |`object`                |Status usia di atas 18 tahun (selalu Y dalam dataset).    |
 |`OverTime`                     |`object`                |Apakah karyawan sering lembur.                            |
 |`PercentSalaryHike`            |`int64`                 |Persentase kenaikan gaji tahunan terakhir.                |
 |`PerformanceRating`            |`int64`                 |Penilaian kinerja terakhir.                               |
 |`RelationshipSatisfaction`     |`int64`                 |Tingkat kepuasan terhadap hubungan kerja.                 |
 |`StandardHours`                |`int64`                 |Jam kerja standar (selalu 80 dalam dataset).              |
 |`StockOptionLevel`             |`int64`                 |Level kepemilikan saham perusahaan.                       |
 |`TotalWorkingYears`            |`int64`                 |Total tahun pengalaman kerja karyawan.                    |
 |`TrainingTimesLastYear`        |`int64`                 |Jumlah pelatihan yang diikuti dalam setahun terakhir.     |
 |`WorkLifeBalance`              |`int64`                 |Tingkat keseimbangan kerja–hidup.                         |
 |`YearsAtCompany`               |`int64`                 |Total tahun bekerja di perusahaan saat ini.               |
 |`YearsInCurrentRole`           |`int64`                 |Total tahun di posisi atau jabatan saat ini.              |
 |`YearsSinceLastPromotion`      |`int64`                 |Tahun sejak promosi terakhir.                             |
 |`YearsWithCurrManager `        |`int64`                 |Tahun bekerja dengan manajer saat ini.                    |
 |`Attrition`                    |`int64`                 |Target: apakah karyawan keluar dari perusahaan.           |

Berdasarkan karakteristik datanya, fitur dapat dikelompokkan menjadi beberapa kategori utama:

- **Demografi**: `Age`, `Gender`, `MaritalStatus`  
- **Pekerjaan**: `JobRole`, `Department`, `JobLevel`  
- **Kinerja & Kompensasi**: `MonthlyIncome`, `OverTime`, `WorkLifeBalance`  
- **Kepuasan Kerja**: `JobSatisfaction`, `EnvironmentSatisfaction`  
- **Loyalitas & Masa Kerja**: `YearsAtCompany`, `TotalWorkingYears`

### 2. Penjelasan Kontekstual Fitur

- **OverTime**, **JobSatisfaction**, dan **WorkLifeBalance** berhubungan erat dengan keseimbangan kehidupan kerja dan tingkat kepuasan karyawan. Karyawan yang sering lembur dengan tingkat kepuasan kerja rendah cenderung memiliki risiko attrition lebih tinggi.

- **YearsAtCompany**, **YearsInCurrentRole**, dan **YearsSinceLastPromotion** menunjukkan dinamika karier karyawan. Nilai tinggi pada YearsInCurrentRole tetapi besar juga pada YearsSinceLastPromotion bisa menandakan stagnasi karier.

- **MonthlyIncome** dan **JobLevel** memiliki hubungan linear: semakin tinggi level jabatan, umumnya semakin besar gaji. Namun, perbedaan signifikan antar individu bisa menjadi indikator ketidakpuasan terhadap kompensasi.

- **PerformanceRating** dan **JobInvolvement** menggambarkan dedikasi serta pencapaian kerja. Nilai keterlibatan tinggi tanpa peningkatan performa dapat menunjukkan ketidakseimbangan beban atau kurangnya penghargaan.

- **EnvironmentSatisfaction** turut mencerminkan kenyamanan karyawan di lingkungan kerja. Nilai yang rendah bisa memicu keinginan untuk mencari lingkungan kerja baru.

---

## Exploratory Data Analysis (EDA)

### 1. Pemeriksaan Data Awal  
Tahap pertama dilakukan untuk memahami karakteristik dataset secara umum. Dataset berisi **1.470 data karyawan** dengan berbagai informasi demografis, pekerjaan, kepuasan kerja, dan status keluar (*attrition*). Mayoritas karyawan berada pada rentang usia produktif antara **30–40 tahun**, dengan masa kerja rata-rata sekitar **7 tahun**.

### 2. Pemeriksaan Missing Values dan Duplikasi  
Pengecekan awal menunjukkan bahwa **tidak terdapat data hilang (missing value)** maupun **data duplikat** di seluruh kolom. Hal ini berarti dataset sudah lengkap dan siap digunakan tanpa proses imputasi tambahan.

| Jenis Pemeriksaan  | Hasil     |
---------------------------
| Missing Values     | Tidak ada |
| Data Duplikat      | Tidak ada |

### 3. Analisis Nilai Unik  
Setiap kolom diperiksa untuk memastikan variasi nilai informatif. Ditemukan tiga kolom yang hanya memiliki satu nilai unik untuk seluruh baris, yaitu:  
- `EmployeeCount`  
- `Over18`  
- `StandardHours`  

Karena kolom-kolom tersebut **tidak mengandung variasi informasi**, maka dihapus dari dataset pelatihan dan pengujian.

### 4. Distribusi Target (`Attrition`)  
Proporsi karyawan yang keluar dari perusahaan relatif kecil dibandingkan yang bertahan.  
Hal ini menunjukkan adanya **ketidakseimbangan kelas (class imbalance)** yang perlu diperhatikan pada tahap modeling melalui penggunaan metode **SMOTE** atau pengaturan **`class_weight='balanced'`**.

---

## Data Preparation

Tahap ini bertujuan untuk menyiapkan seluruh langkah transformasi data secara sistematis dan konsisten melalui *pipeline* yang dirancang menggunakan `scikit-learn`. Pendekatan ini memastikan tidak terjadi *data leakage* serta memudahkan proses pelatihan dan validasi model.

### 1. Pemisahan Fitur dan Target
Target prediksi adalah kolom **`Attrition`**, yang berisi status apakah karyawan keluar atau tidak. Nilai kategorikal seperti `"Yes"` dan `"No"` dikonversi menjadi biner (1 dan 0) agar dapat digunakan dalam algoritma pembelajaran mesin.

```python
TARGET = 'Attrition'
if train[TARGET].dtype == 'object':
    train[TARGET] = train[TARGET].map(lambda x: 1 if str(x).strip().lower() in ['yes','1','y','true'] else 0)
X = train.drop(columns=[TARGET])
y = train[TARGET]
```

Dataset kemudian dibagi menjadi data latih dan validasi menggunakan **`train_test_split`** dengan parameter `stratify=y` agar proporsi kelas tetap seimbang antara kedua set.

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=RANDOM_STATE
)
```

### 2. Transformasi dan Preprocessing

Langkah preprocessing mencakup penanganan nilai hilang, encoding variabel kategorikal, dan normalisasi fitur numerik. Seluruh proses dikemas dalam `ColumnTransformer` dan `Pipeline`.

```python
num_trans = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())])
cat_trans = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                      ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preproc = ColumnTransformer([('num', num_trans, num_cols),
                             ('cat', cat_trans, cat_cols)])
```

Jika dataset mengalami ketidakseimbangan kelas, maka digunakan **SMOTE (Synthetic Minority Over-sampling Technique)** untuk menyeimbangkan proporsi kelas pada data latih.  
Pipeline dibuat fleksibel agar dapat otomatis menambahkan SMOTE bila diperlukan:

```python
def build_pipeline(model):
    if USE_SMOTE and _HAS_SMOTE:
        return ImbPipeline([
            ('preproc', preproc),
            ('smote', SMOTE(random_state=RANDOM_STATE)),
            ('clf', model)
        ])
    else:
        return Pipeline([
            ('preproc', preproc),
            ('clf', model)
        ])
```

Pendekatan ini membuat seluruh tahapan *preprocessing* terintegrasi langsung dengan model, sehingga reproducible dan efisien.

---
 
## Model Training and Hyperparameter Tuning

Beberapa algoritma diuji untuk membandingkan performa klasifikasi attrition karyawan, yaitu:

- **Logistic Regression**
- **Random Forest**
- **XGBoost**
- **LightGBM**

Setiap model dikombinasikan dengan pipeline preprocessing di atas, lalu dilakukan pencarian parameter terbaik menggunakan **`RandomizedSearchCV`** dengan **Stratified K-Fold Cross-Validation (5 folds)** dan metrik utama **ROC-AUC**.

```python
search = RandomizedSearchCV(estimator=spec['pipeline'],
                            param_distributions=spec['param_dist'],
                            n_iter=N_ITER_SEARCH,
                            scoring='roc_auc',
                            n_jobs=N_JOBS,
                            cv=skf,
                            random_state=RANDOM_STATE)
search.fit(X_train, y_train)
```

Proses tuning menghasilkan hasil sebagai berikut (ringkasan):

| Model                | CV AUC     | Validation AUC | Keterangan            |
------------------------------------------------------------------------------
| Logistic Regression  | 0.8329     | 0.8319         | Model linear baseline |
| Random Forest        | **0.8039** | **0.8448**     | Model terbaik         |
| XGBoost              | 0.8221     | 0.7969         | Sedikit overfitting   |
| LightGBM             | 0.8139     | 0.7726         | Performa lebih rendah |

Model terbaik yang dipilih adalah **Random Forest**, karena memberikan hasil **Validation ROC-AUC = 0.8448** dan performa yang stabil di berbagai subset data.

---

## Model Evaluation

Evaluasi dilakukan menggunakan metrik **ROC-AUC**, yang mengukur kemampuan model dalam membedakan kelas positif (Attrition=1) dan negatif (Attrition=0) di berbagai ambang keputusan.

ROC curve divisualisasikan untuk menggambarkan performa model terbaik:

```python
y_proba = best_model.predict_proba(X_val)[:,1]
best_auc = roc_auc_score(y_val, y_proba)

fpr, tpr, _ = roc_curve(y_val, y_proba)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'{best_name} (AUC={best_auc:.4f})')
plt.plot([0,1],[0,1],'--',color='gray')
plt.legend(); plt.title("ROC Curve")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(True); plt.show()
```

<img src="img/eval_roc_curve.png" alt="ROC Curve" />

Dari hasil pengujian, **Random Forest** menunjukkan keseimbangan baik antara sensitivitas dan spesifisitas, dengan **Validation ROC-AUC sebesar 0.8448**, yang berarti model mampu memisahkan karyawan yang berpotensi keluar dan yang bertahan dengan cukup baik.

---

## Conclusion

- Proses *data preparation* yang meliputi imputasi, encoding, dan scaling berhasil menghasilkan dataset yang siap digunakan untuk pemodelan tanpa adanya missing value atau duplikasi data.  
- Beberapa model telah diuji, dan hasil menunjukkan bahwa **Random Forest** memberikan performa terbaik dengan **Validation ROC-AUC sebesar 0.8448**.  
- Nilai ROC-AUC yang tinggi menunjukkan bahwa model mampu membedakan dengan baik antara karyawan yang berpotensi keluar dan yang tetap bertahan.  
- Tahapan evaluasi menunjukkan bahwa model memiliki keseimbangan yang baik antara sensitivitas dan spesifisitas, serta dapat digunakan sebagai dasar untuk analisis lebih lanjut.  

---

**End of Report — Employee Attrition Prediction**
