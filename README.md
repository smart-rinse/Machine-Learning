# Lifecycle of Machine-Learning Project

# Sentiment Analysis Review
Sentiment analysis evaluates the quality of the laundry based on previous user reviews. Then provide quality laundry recommendations to users. In a recommendation system, positive sentiment can be an important factor in determining laundry recommendations to users.

## 1. Project Planning and Setup
* Goals: Improving the quality of recommendations by considering the sentiments of user reviews.
* User: End user
* Evaluation
<br> Create a visualization of all training loss/accuracy and validation loss/accuracy and also create a predict function to predict the sentiment results from new input words. </br>

## 2. Data Colection and Labeling
* Data availability and collection:
  * Data review: Data review terdiri dari 2 kolom yaitu teks review dan label. Data ini selanjutnya digunakan untuk membuat model sentimen analysis.
  
  * Data laundry: Data laundry erisikan informasi detail dari setiap laundry. This data is further used to create dummy dataset for backend database.
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
  <br> Result: </br>
  - `Loss: ......`
  - `Accuracy: ......`
* Debugging: Mengatasi masalah overfitting dengan memperbanyak data

## 4. Deployment
The model architecture then deployed to backend service / google cloud and then the model will get the data and process it.
