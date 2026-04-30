# 📰 Fake News Detection Using NLP and Machine Learning

## 📌 1. Problem Statement

With the explosion of digital news sources, fake news has become a major threat to public trust and informed decision-making. The objective of this project is to develop an NLP-based machine learning model to classify news articles as either true or fake, based on their textual content.

## 🧪 2. Methodology (Tasks from Starter Notebook)

### 1. Data Preparation
- Combined `True.csv` and `Fake.csv` datasets, each containing title, text, and publication date.
- Assigned labels:
  - `1` for true
  - `0` for fake

### 2. Text Preprocessing
- Performed tokenization, lowercasing, punctuation removal, and digit removal.
- Used spaCy for lemmatization and stopword removal.

### 3. Train-Validation Split
- Split data using an 80:30 ratio for training and validation.

### 4. Exploratory Data Analysis (EDA)
- Plotted distribution of character lengths before and after preprocessing.
- Generated word clouds to analyze common terms in both true and fake news.
- Displayed top 10 unigrams, bigrams, and trigrams for both fake and true news.

### 5. Feature Extraction
- Applied TF-IDF Vectorization to transform text into numerical format.
- Additionally, used pre-trained embeddings with `word2vec-google-news-300` from the Gensim API:

```python
import gensim.downloader as api
word2vec_model = api.load('word2vec-google-news-300')
