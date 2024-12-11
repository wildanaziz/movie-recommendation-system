# Laporan Proyek Machine Learning - Wildan Aziz Hidayat

![movie_banner](images/banner.jpg)  
Figure 1.0: A collection of popular movies

## Project Overview

Sistem rekomendasi film adalah salah satu aplikasi machine learning yang populer dan memiliki dampak signifikan pada industri hiburan. Proyek ini bertujuan untuk mengembangkan sebuah sistem rekomendasi berbasis content-based filtering dan collaborative filtering.

Sistem ini akan membantu pengguna menemukan film yang sesuai dengan preferensi mereka, berdasarkan data rating pengguna dan informasi genre film. Pendekatan ini tidak hanya memberikan pengalaman personalisasi yang lebih baik bagi pengguna, tetapi juga meningkatkan engagement platform streaming.

Metode ini bertujuan untuk memperbaiki masalah saran-saran tertentu, yang terjadi ketika data yang relevan bagi pengguna diabaikan. Metode ini mengekstrak tipe kepribadian pengguna, pengalaman menonton sebelumnya, dan data dari banyak basis data lain yang berkaitan dengan kritikus film. Hal ini difokuskan pada estimasi kesamaan komposit. Metodologi ini adalah sistem hibrida yang menggabungkan teknik penyaringan berbasis konten dan kolaboratif. Hongli LIn dkk. menyarankan sebuah proses yang disebut penyaringan kolaboratif yang didukung konten untuk memprediksi kompleksitas model dari contoh tertentu untuk setiap kandidat (CBCF). Metode ini dibagi menjadi dua tahap: "penyaringan berbasis konten", yang meningkatkan informasi tentang tinjauan skenario peserta pelatihan saat ini, dan "penyaringan kolaboratif", yang memberikan prediksi yang paling akurat. Algoritme CBCF mempertimbangkan pendekatan berbasis konten dan kolaboratif, mengatasi kedua kekurangannya secara bersamaan.[[1]](https://ieeexplore.ieee.org/document/9872515/authors#authors)

**Referensi**: [Movie Recommender System Using Content-based and Collaborative Filtering](https://ieeexplore.ieee.org/document/9872515/authors#authors)

## Business Understanding

### Problem Statements
1. Bagaimana memberikan rekomendasi film kepada pengguna berdasarkan genre yang disukai?
2. Bagaimana memanfaatkan data rating pengguna untuk memberikan rekomendasi yang lebih relevan?

### Goals
1. Mendapatkan rekomendasi film berdasarkan genre atau tema dari movies yang disukai dengan tingkat Recall@K lebih dari 80%.
2. Mendapatkan rekomendasi film berdasarkan movies yang pernah dirating sebelumnya dengan error lebih kecil dari 50%.

### Solution Approach
- Mengembangkan sistem rekomendasi berbasis content-based filtering yang memanfaatkan informasi genre film dengan menggunakan algoritma TF-IDF dan cosine similarity untuk membuat sistem rekomendasi film berdasarkan genre yang mirip dengan film yang disukai.
- Mengembangkan sistem rekomendasi berbasis collaborative filtering menggunakan data rating pengguna untuk meningkatkan relevansi rekomendasi dengan menggunakan pendekatan embedding dengan TensorFlow/Keras untuk membangun model collaborative filtering dengan outcome sistem rekomendasi yang mirip dengan film yang pernah dirating sebelumnya.


## Data Understanding

Dataset yang digunakan adalah [MovieLens](https://grouplens.org/datasets/movielens/), yang terdiri dari dua file utama:
1. **movies.csv**: Berisi informasi mengenai judul film dan genre.
2. **ratings.csv**: Berisi data rating yang diberikan pengguna terhadap film.

Jumlah data:
- **movies.csv**: 10329 baris dan 3 kolom.
- **ratings.csv**: 105339 baris dan 4 kolom.

### Variabel-variabel pada dataset:
#### movies.csv
- **movieId**: ID unik untuk setiap film.
- **title**: Judul film.
- **genres**: Genre film dalam format string.

#### ratings.csv
- **userId**: ID unik untuk setiap pengguna.
- **movieId**: ID unik untuk setiap film.
- **rating**: Rating yang diberikan oleh pengguna.
- **timestamp**: Waktu ketika rating diberikan.

#### Info Data movies.csv

|   # | Column   | Non-Null Count | Dtype   |
| --: | -------- | -------------- | ------- |
|   0 | movieId  | 10329 non-null | int64   |
|   1 | title    | 10329 non-null | object  |
|   2 | genres   | 10329 non-null | object  |


#### Info Data ratings.csv

Dataset ini mengandung 105339 entri dan 4 kolom

|   # | Column      | Dtype |
| --: | ----------- | ----- |
|   0 | userId      | int64 |
|   1 | movieId     | int64 |
|   2 | rating      | int64 |
|   3 | timestamp   | int64 |

#### Visualisasi Data Ratings
![rating_plot](images/ratings.png)  
Gambar 2.0 plot kolom rating dari dataframe "rating"

Di label "count" maksimum nilainya berada kurang dari 30000.

Di label "rating" terdapat bar setelah angka 0 yang artinya banyak user yang memberi rating 0.5 menandakan rating terendah yakni 0.5.

![movies_user_plot](images/users-movies.png)  
Gambar 2.1 plot kolom movieId dan userId dari dataframe "rating"

Terlhat jumlah user unik yang memberi rating berada di angka 6.1% dan jumlah film unik yang diberi rating lebih dari 93.9%. Data ini bersifat **many-to-many** yang berarti bahwa 1 user bisa memberi rating ke banyak movies dan 1 movies bisa diberi rating oleh banyak user

## Data Preparation

### Teknik yang diterapkan:
1. **Genre ke format list**: Mengubah string genre menjadi format list untuk analisis lebih lanjut.
2. **Pengecekan Missing Values**: Karena pada data yang didapatkan tidak terdapat missing values maka tidak ada data yang dihilangkan.
3. **Encoding userId dan movieId**: Mengubah ID menjadi representasi numerik agar dapat digunakan dalam modeling collaborative filtering.
4. **Pembagian data**: Membagi data menjadi training (80%) dan validation (20%).

### Genre ke Format List
- Pada tahap ini genre dari setiap movies di dataframe **movies.csv** akan diubah menjadi bentuk array(list). Hal ini dilakukan untuk mempermudah akses ke genre di kolom "genres". Hasilnya sebagai berikut

|   # | movieId  |                          title   |                                             genres |
| --: | -------: | ---------------------------------: | ------------------------------------------------: |
|   0 |    1     |                 Toy Story (1995)   | [Adventure, Animation, Children, Comedy, Fantasy] |
|   1 |    2     |                   Jumanji (1995)   |                    [Adventure, Children, Fantasy] |
|   2 |    3     |          Grumpier Old Men (1995)   |                                 [Comedy, Romance] |
|   3 |    4     |         Waiting to Exhale (1995)   |                          [Comedy, Drama, Romance] |
|   4 |    5     | Father of the Bride Part II (1995) |                                          [Comedy] |

Proses ini diperlukan untuk memastikan data siap digunakan oleh algoritma machine learning dan meningkatkan performa model.

### Pengecekan Missing Values
- Karena pada dataset yang diberikan tidak terdapat missing values maka tidak ada data yang dihapus.

### Encoding userId dan movieId
- Pada tahap ini dilakukan proses encoding pada kolom userId dan movieId kemudian dimasukkan ke kolom baru masing-masing. Hal ini dilakukan untuk merepresentasikan id user dan movie dalam format yang dapat di proses oleh model machine learning.

### Pembagian data

- Melakukan pemisahan pada dataframe menjadi train dan validasi dengan rasio 80:20, namun data di acak terlebih dahulu sebelum di pisah. Hal ini dilakukan supaya model dapat melakukan evaluasi pada data baru dan mencegah overfitting

## Modeling

### Content-Based Filtering
- **Pendekatan**: Menggunakan TF-IDF untuk mengekstraksi fitur dari genre film, diikuti oleh perhitungan cosine similarity.
- **Output**: Top-N rekomendasi film berdasarkan kesamaan genre dengan input film yang diberikan.

Teknik **Content-Based Filtering** merupakan teknik sistem rekomendasi untuk merekomendasikan suatu produk yang memiliki kemiripan dengan produk yang disukai. Dalam kasus ini sistem akan merekomendasikan movies berdasarkan kemiripan genre dengan movies yang disukai pengguna. Teknik ini menggunakan rumus **Cosine Similarity** untuk mendapatkan kecocokan antara produk 1 dengan yang lain.

Formula untuk **Cosine Similarity** adalah:  
$\displaystyle cos~(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$

Teknik ini menggunakan model **TF-IDF Vectorizer** untuk mendapatkan informasi mengenai genre yang terdapat di setiap movies dan diubah menjadi fitur yang dapat diukur kemiripannya. Contohnya adalah sebagai berikut

| Title                                                       | horror | war | comedy | western | romance | drama | musical | sci  | thriller |
|-------------------------------------------------------------|--------|-----|--------|---------|---------|-------|---------|------|----------|
| Cloudy with a Chance of Meatballs (2009)                    | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Red (2008)                                                 | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.550387 | 0.0  | 0.83491  |
| Davy Crockett, King of the Wild Frontier (1955)             | 0.0    | 0.0 | 0.0    | 0.0     | 0.832344 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Ghost of Frankenstein, The (1942)                           | 0.0    | 1.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |
| Most Hated Family in America, The (2007)                    | 0.0    | 0.0 | 0.0    | 0.0     | 0.000000 | 0.0   | 0.000000 | 0.0  | 0.00000  |

Selanjutnya **Cosine Similarity** akan diterapkan pada dataframe movies yang telah dibersihkan sehingga menghasilkan output sebagai berikut:

| Title                                                      | Guilty as Sin (1993) | Girl Who Kicked the Hornet's Nest, The (Luftslottet som sprängdes) (2009) | Leopard Man, The (1943) | American Haunting, An (2005) | Chain Reaction (1996) | Fellini Satyricon (1969) | Hereafter (2010) | Tetsuo, the Ironman (Tetsuo) (1988) | Attack the Block (2011) | Stone (2010) |
|------------------------------------------------------------|----------------------|----------------------------------------------------------------------------|------------------------|-----------------------------|-----------------------|--------------------------|------------------|-------------------------------------|-------------------------|--------------|
| Kingdom, The (2007)                                        | 0.530410             | 0.339414                                                                   | 0.385744               | 0.385744                    | 0.698413              | 0.167196                 | 0.167196         | 0.489422                            | 0.310663                | 0.739518    |
| Blackadder's Christmas Carol (1988)                         | 0.000000             | 0.000000                                                                   | 0.000000               | 0.000000                    | 0.000000              | 0.000000                 | 0.000000         | 0.000000                            | 0.344642                | 0.000000    |
| Pineapple Express (2008)                                    | 0.452900             | 0.656832                                                                   | 0.000000               | 0.000000                    | 0.343118              | 0.000000                 | 0.000000         | 0.240444                            | 0.437749                | 0.000000    |
| I Confess (1953)                                            | 0.598829             | 0.000000                                                                   | 0.435502               | 0.435502                    | 0.516846              | 0.000000                 | 0.000000         | 0.362186                            | 0.000000                | 0.834910    |
| Ink (2009)                                                  | 0.000000             | 0.206915                                                                   | 0.000000               | 0.000000                    | 0.231229              | 0.502118                 | 0.502118         | 0.670387                            | 0.783547                | 0.000000    |

Di tabel tersebut dapat dilihat kecocokan dari 1 movies dengan yang lain. Nilai-nilai pada tabel tersebut merepresentasikan persentase kecocokan antara kedua movies tersebut.

#### Mendapatkan top-N rekomendasi
Tabel tersebut adalah dataframe cosine similarity yang akan digunakan untuk mendapatkan top-N rekomendasi movies. Dalam kasus ini akan dicoba mendapatkan top-10 rekomendasi movies yang mirip dengan movies **"Sherlock Holmes (2010)"**. Outputnya sebagai berikut

Data untuk uji coba
| # | title                          | genres                           |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Sherlock Holmes (2010)         | [Mystery, Sci-Fi]                |

Hasil rekomendasi

| Title                                                     | Genres                      |
|-----------------------------------------------------------|-----------------------------|
| Andromeda Strain, The (1971)                              | [Mystery, Sci-Fi]            |
| Stalker (1979)                                            | [Drama, Mystery, Sci-Fi]     |
| Solaris (Solyaris) (1972)                                  | [Drama, Mystery, Sci-Fi]     |
| Sound of My Voice (2011)                                  | [Drama, Mystery, Sci-Fi]     |
| Fire in the Sky (1993)                                    | [Drama, Mystery, Sci-Fi]     |
| Big Empty, The (2003)                                     | [Comedy, Mystery, Sci-Fi]    |
| Stepford Wives, The (1975)                                | [Mystery, Sci-Fi, Thriller]  |
| District 9 (2009)                                         | [Mystery, Sci-Fi, Thriller]  |
| Seconds (1966)                                            | [Mystery, Sci-Fi, Thriller]  |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995)                 | [Mystery, Sci-Fi, Thriller]  |

Berdasarkan hasil rekomendasi tersebut dapat dilihat bahwa movies yang direkomendasikan memiliki genre yang mirip dengan input movienya yaitu genre "Mystery" dan "Sci-Fi".


#### Kelebihan Content-Based Filtering:
- Tidak memerlukan data pengguna.
- Cocok untuk sistem dengan data pengguna yang terbatas.

#### Kekurangan Content-Based Filtering:
- Tidak memperhitungkan variasi preferensi antar pengguna.

### Collaborative Filtering
- **Pendekatan**: Membuat embedding untuk pengguna dan film menggunakan TensorFlow/Keras. Model dioptimalkan menggunakan fungsi loss binary crossentropy.
- **Output**: Top-N rekomendasi film untuk pengguna tertentu berdasarkan pola rating.

Teknik **Collaborative Filtering** merupakan teknik sistem rekomendasi untuk merekomendasikan suatu produk berdasarkan kesamaan preferensi antar user. Dalam kasus ini sistem akan menggunakan movie yang dirating tinggi oleh user untuk mencari kesamaan dengan user lain.

Proyek ini menggunakan model **RecommenderNet** yang dibuat dari kelas **Model** milik **Keras**. Dan kemudian dicompile menggunakan metrik **Mean Absolute Error**, loss function **Binary Crossentropy**, dan optimizer **Adam**. Dan kemudian model dapat dilatih.

#### Mendapatkan top-N rekomendasi

Kita ambil user secara acak dari dataframe ratings
```
Showing recommendations for user: 521
========================================
Movies with high ratings from user
----------------------------------------
True Lies (1994) : Action, Adventure, Comedy, Romance, Thriller
For Love or Money (1993) : Comedy, Romance
Much Ado About Nothing (1993) : Comedy, Romance
Sleepless in Seattle (1993) : Comedy, Drama, Romance
Mission: Impossible (1996) : Action, Adventure, Mystery, Thriller
----------------------------------------
```
Kemudian akan diambil semua movies yang belum dilihat oleh user, lalu model akan melakukan prediksi berdasarkan movies dengan rating tinggi oleh user dan kemiripannya dengan user lain. Hasilnya akan mendapatkan rekomendasi sebagai berikut

```
Top 10 movies recommendations
----------------------------------------
Grumpier Old Men (1995) : Comedy, Romance
Turbo: A Power Rangers Movie (1997) : Action, Adventure, Children
Take the Money and Run (1969) : Comedy, Crime
Sixteen Candles (1984) : Comedy, Romance
Fast Food, Fast Women (2000) : Comedy, Romance
```

Berdasarkan hasil rekomendasi tersebut menunjukkan movies yang relevan dengan movies yang telah dirating sebelumnya. Rekomendasi yang diberikan juga bervariasi dan tidak hanya terpatok pada beberapa genres tertentu tidak seperti **Content-Based Filtering**.

#### Kelebihan:
- Memanfaatkan interaksi pengguna untuk menghasilkan rekomendasi yang lebih personal.
- Dapat menangkap pola kompleks dalam data pengguna.

#### Kekurangan:
- Membutuhkan data rating yang cukup besar.
- Tidak dapat merekomendasikan film baru yang belum memiliki rating.


## Evaluation

### Evaluasi Content-Based Filtering

Metrik evaluasi yang digunakan untuk **Content Based Filtering** adalah **Recall@K**.

**Recall@K** adalah metrik yang mengukur proporsi dari item yang relevan di top-K dari keseluruhan item relevan di top-N rekomendasi.

Formula dari Recall@K adalah:

Recall@K = $\displaystyle \frac{\text{item yang relevan di top-K}}{\text{item yang relevan di top-N}}$

Berikut analisa Recall@K untuk hasil rekomendasi **Content-Based Filtering**.

Data untuk uji coba
| # | title                          | genres                           |
|--:|:------------------------------:|:--------------------------------:|
| 0 | Sherlock Holmes (2010)         | [Mystery, Sci-Fi]                |

Hasil rekomendasi

| Title                                                     | Genres                      |
|-----------------------------------------------------------|-----------------------------|
| Andromeda Strain, The (1971)                              | [Mystery, Sci-Fi]            |
| Stalker (1979)                                            | [Drama, Mystery, Sci-Fi]     |
| Solaris (Solyaris) (1972)                                  | [Drama, Mystery, Sci-Fi]     |
| Sound of My Voice (2011)                                  | [Drama, Mystery, Sci-Fi]     |
| Fire in the Sky (1993)                                    | [Drama, Mystery, Sci-Fi]     |
| Big Empty, The (2003)                                     | [Comedy, Mystery, Sci-Fi]    |
| Stepford Wives, The (1975)                                | [Mystery, Sci-Fi, Thriller]  |
| District 9 (2009)                                         | [Mystery, Sci-Fi, Thriller]  |
| Seconds (1966)                                            | [Mystery, Sci-Fi, Thriller]  |
| Twelve Monkeys (a.k.a. 12 Monkeys) (1995)                 | [Mystery, Sci-Fi, Thriller]  |



Seperti di tabel, semua movies memiliki ketiga genre di data uji coba yaitu **"Mystery, Sci-Fi"**. Hal ini menjadikan jumlah item yang relevan di top-N = 10. maka dapat disimpulkan juga untuk jumlah item di top-K akan selalu sama dengan K.

Maka Recall@K untuk

- K = 5 &rarr; 5/10 \* 100% = 50%
- K = 8 &rarr; 8/10 \* 100% = 80%
- K = 10 &rarr; 10/10 \* 100% = 100%

Dapat disimpulkan bahwa rekomendasi yang diberikan memiliki Recall@K sebesar 100%.

### Evaluasi Collaborative Filtering

Metrik evaluasi yang digunakan untuk **Collaborative Filtering** adalah **Mean Absolute Error (MAE)**

MAE atau Mean Absolute Error diterapkan dengan cara mengukur rata-rata dari selisih absolut antara prediksi dan nilai asli (y_asli - y_prediksi).

Formula MAE adalah

MAE = $\displaystyle \sum\frac{|y_i - \hat{y}_i|}{n}$

Dimana:
MAE = nilai Mean Absolute Error
y = nilai aktual
ŷ = nilai prediksi
i = urutan data
n = jumlah data
Berikut plot MAE dari model

![model_plot](images/model_metrics_mae.png  
Gambar 3.0 plot MAE dari model

Dapat dilihat model ini memiliki nilai MAE yang relatif rendah yaitu kurang dari 20% dan tidak mengalami overfitting karena antara data train dan data test tidak memiliki ketimpangan hingga 10% sehingga cocok untuk melakukan prediksi pada data baru.

## References

###### [1] J F Mohammad, B Al Faruq1, S Urolagin, "Movie Recommender System Using Content-based and Collaborative Filtering", 2022_ https://ieeexplore.ieee.org/document/9872515/authors#authors
