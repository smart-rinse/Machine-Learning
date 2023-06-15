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
* Model selection: 
  * Classification supervised learning
  * Sentiment analysis
* Model training: link notebook
* Debugging

## 4. Deployment and Monitoring
Menyimpan model dalam bentuk .h5 beserta dengan tokenizer. Kemudian, model di deploy di cloud.
