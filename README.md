# Lifecycle of Machine-Learning Project

# Sentiment Analysis Review
Memberikan prediksi apakah review yang diberikan oleh pengguna menandakan bahwa mereka puas dengan layanan laundrynya atau malah mereka kecewa.

## 1. Project Planning and Setup
* Goals : 
* User : End user
* Performace constraints : accuracy >= 90%

## 2. Data Colection and Labeling
* Data availability and collection:
  * Data review: Berisikan data-data review dari beberapa laundry yang kemudian diklasifikasikan menjadi review positif dan negatif. Data ini digunakan untuk membuat model machine learning.
  
  * Data laundry: Berisikan informasi detail dari setiap laundry. This data is further used to create dummy dataset for backend database.
* Storage: cloud
* Data preprocessing & respresentation:
  * Mengubah semua karakter menjadi huruf kecil
  ```sh
  #Mengubah menjadi huruf kecil semua
  df['text'] = df['text'].apply(lambda x: x.lower())
  print(df)
  ```
  * Menghapus karakter dan tanda baca yang tidak diinginkan
  * Menghapus spasi berlebih
  * Mengabungkan kata-kata penting agar tidak hilang
  * Stopword removal

## 3. Model Trainig and Debugging
* Model selection: 
  * Classification supervised learning
  * Sentiment analysis
* Model training: link notebook
* Debugging

## 4. Deployment and Monitoring
Menyimpan model dalam bentuk .h5 beserta dengan tokenizer. Kemudian, model di deploy di cloud.
