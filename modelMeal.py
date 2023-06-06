#library
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# preprosesing
data = pd.read_csv('MealPlan.csv')
# Preprocessing bahan dasar
def preprosesdata(data):
    def preprocess_bahan(text):
        tokenisasi = word_tokenize(text)
        stop_words = set(stopwords.words('indonesian'))
        filtered_tokens = [token for token in tokenisasi if token.lower() not in stop_words]
        filtered_tokens = [re.sub(r'\W+', '', token) for token in filtered_tokens]
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        return stemmed_tokens
    def preprocess_harga(harga):
        harga = re.sub(r'\D', '', harga)
        harga = int(harga)
        return harga
    def preprocess_alergi(alergi_columns):
        alergi_encoded = alergi_columns.sum(axis=1)
        alergi_encoded = alergi_encoded.apply(lambda x: 1 if x > 0 else 0)
        return alergi_encoded
    data["Bahan Dasar"] = data["Bahan Dasar"].apply(preprocess_bahan)
    data["Harga"] = data["Harga"].apply(preprocess_harga)
    # gabung kolom alergi jadi satu kolom
    alergi_columns = data[["Alergi Telur", "Alergi Kacang", "Alergi Kedelai", "Alergi Seafood", "Alergi udang", "Alergi susu", "Alergi gandum"]]
    data["Alergi"] = preprocess_alergi(alergi_columns)
    data = data.drop(["Alergi Telur", "Alergi Kacang", "Alergi Kedelai", "Alergi Seafood", "Alergi udang", "Alergi susu", "Alergi gandum"], axis=1)
    return data