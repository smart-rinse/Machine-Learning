# Lifecycle of Machine-Learning Project

# Sentiment Analysis Review
Sentiment analysis evaluates the quality of the laundry based on previous user reviews. Then provide quality laundry recommendations to users. In a recommendation system, positive sentiment can be an important factor in determining laundry recommendations to users.

## 1. Project Planning and Setup
* Goals: Improving the quality of recommendations by considering the sentiments of user reviews.
* User: End user
* Evaluation: Create a visualization of all training loss/accuracy and validation loss/accuracy and also create a predict function to predict the sentiment results from new input words. </br>

## 2. Data Colection and Labeling
* Data availability and collection: Data collection is done by collecting laundry review data on the internet (specifically on google maps).
  * Review dataset
  <br> Review dataset consists of two columns, which is review text and labels. This data is then used to create a sentiment analysis model. </br>
  
  * Laundry dataset
  <br> Laundry dataset contains detailed information for each laundry. This data is further used to create dummy dataset for backend database. </br>
* Data preprocessing:
  * Converts all characters to lowercase
  ```sh
  df['text'] = df['text'].apply(lambda x: x.lower())
  ```
  * Remove unwanted characters and punctuation
  ```sh
  df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
  ```
  * Remove excess spaces
  ```sh
  df['text'] = df['text'].str.replace(r'\s+', ' ')
  ```
  * Modify the tokenizer
  ```sh
  def combine_important_words(tokens):
    combined_tokens = []
    skip_next = False
    for i in range(len(tokens) - 1):
        if skip_next:
            skip_next = False
            continue
        if tokens[i] in ['tidak', 'kurang', 'gak', 'sangat', 'banget', 'kadang', 'terlalu','yg','gk']:
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
## 3. Data Splitting
* Dataset for training consists of more than 2000 row of data. 80% of the data will be used for training set, and 20% of the data will be used for test set.
## 4. Model Training and Debugging
* Tokenized word index data is exported into .pkl file. So we don't need to fit the data anymore when using on production. We only need to load pickle file for production purpose.
* Model selection: This model is a classification and supervised learning model and focuses on sentiment analysis to classify reviews as positive (1) or negative (0).
* Model training: This model uses embedding layer. Then at the final layers, a sequential model is used which consists of:
  - `Dense(units=64, activation='relu')` layer
  - `Dense(units=32, activation='relu')` layer
  - `Dropout(0.4)` layer
  - `Dense(units=1, activation='sigmoid')` layer
  <br> Result: </br>
  - `Loss: 0.3%`
  - `Accuracy: 99%`
* Debugging: Solved the problem of overfitting by multiplying data and using Kfold cross validation.
* Note : although the model having a high accuracy, the model still got some misclassification issue when testing the model. But overall, it works fine.

## 5. Deployment
The model architecture then deployed to backend service / google cloud and then the model will get the data and process it.

## Prerequisites
Library used in this project:
* numpy==1.22.4
* pandas==1.5.3
* tensorflow==2.12.0
* gensim==4.3.1
* nlpaug==1.1.11
* scikit-learn==1.2.2
* nltk==3.8.1
