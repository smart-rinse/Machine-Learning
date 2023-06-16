# Lifecycle of Machine-Learning Project

# Sentiment Analysis Review
Sentiment analysis mengevaluasi kualitas laundry berdasarkan ulasan pengguna sebelumnya. Kemudian memberikan rekomendasi laundry yang berkualitas kepada pengguna. Dalam sistem rekomendasi, sentimen positif dapat menjadi faktor penting dalam menentukan laundry yang direkomendasikan kepada pengguna.

## 1. Project Planning and Setup
* Goals : Meningkatkan kualitas rekomendasi dengan mempertimbangkan sentimen pengguna.
* Performace constraints : accuracy >= 90%

## 2. Data Colection and Labeling
* Data availability and collection:
  * Data review: Berisikan data-data review dari beberapa laundry yang kemudian diklasifikasikan menjadi review positif dan negatif. Data ini digunakan untuk membuat model machine learning.
  
  * Data laundry: Berisikan informasi detail dari setiap laundry. This data is further used to create dummy dataset for backend database.
* Storage: cloud
* Data preprocessing & respresentation:
  * Mengubah semua karakter menjadi huruf kecil
  ```sh
  df['text'] = df['text'].apply(lambda x: x.lower())
  ```
  * Menghapus karakter dan tanda baca yang tidak diinginkan
  ```sh
  df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
  ```
  * Menghapus spasi berlebih
  ```sh
  df['text'] = df['text'].str.replace(r'\s+', ' ')
  ```
  * Mengabungkan kata-kata penting agar tidak hilang
  ```sh
  def combine_important_words(tokens):
    combined_tokens = []
    skip_next = False
    for i in range(len(tokens) - 1):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] in ['tidak', 'kurang']:
            combined_tokens.append(tokens[i] + '_' + tokens[i+1])
            skip_next = True
        else:
            combined_tokens.append(tokens[i])
    if not skip_next:
        combined_tokens.append(tokens[-1])
    return combined_tokens
    
    df['Tokens'] = df['text'].apply(word_tokenize)
    df['Combined_Tokens'] = df['Tokens'].apply(combine_important_words)
  ```
  * Stopword removal
  ```sh
  stopwords_ind = stopwords.words('indonesian')
  df['Clean_Tokens'] = df['Combined_Tokens'].apply(lambda x: [word for word in x if word.lower() not in stopwords_ind])
  
  df['clean_text'] = df['Clean_Tokens'].apply(lambda x: ' '.join(x))
  ```

## 3. Model Trainig and Debugging
* Model selection: Model ini merupakan model classification dan supervised learning dan berfokus pada sentimen analisis untuk mengklasifikasikan review menjadi positif atau negatif.
* Model training: Model ini menggunakan layer embedding. Then at the final layers, a sequential model is used which consists of:
  - `Dense(units=64, activation='relu')` layer
  - `Dense(units=32, activation='relu')` layer
  - `Dropout(0.4)` layer
  - `Dense(units=1, activation='sigmoid')` layer
  /n Result:
  - `Loss: ......`
  - `Accuracy: ......`
* Debugging: Mengatasi masalah overfitting dengan memperbanyak data

## 4. Deployment
The model architecture then deployed to backend service / google cloud and then the model will get the data and process it.
