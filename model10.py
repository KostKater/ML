import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def load_data():
    data = pd.read_csv('MealPlan.csv')
    return data

def preprocess_data(data):
    data['Bahan Dasar'] = data['Bahan Dasar'].apply(lambda x: x.split(','))
    mlb = MultiLabelBinarizer()
    bahan_dasar_encoded = pd.DataFrame(mlb.fit_transform(data['Bahan Dasar']), columns=mlb.classes_)
    data_encoded = pd.concat([data, bahan_dasar_encoded], axis=1)
    data['Bahan Dasar'] = data['Bahan Dasar'].apply(lambda x: ' '.join(x))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['Bahan Dasar'])
    tfidf_matrix.sort_indices()
    tfidf_matrix = tfidf_matrix.toarray()

    return data, vectorizer, tfidf_matrix

def train_model(tfidf_matrix, vectorizer):
    input_shape = len(vectorizer.get_feature_names_out())
    input_layer = Input(shape=(input_shape,))
    hidden_layer_1 = Dense(256, activation='relu')(input_layer)
    hidden_layer_2 = Dense(128, activation='relu')(hidden_layer_1)
    output_layer = Dense(input_shape, activation='sigmoid')(hidden_layer_2)
    model = Model(input_layer, output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(tfidf_matrix, tfidf_matrix, epochs=500)
    model.save('rekomendasi.h5')
    return model

def load_model():
    model = tf.keras.models.load_model('rekomendasi.h5')
    return model

def recommend_meal_plan(data, vectorizer, model, tfidf_matrix, bahan_dasar, alergi, kehalalan, harga):
    bahan_dasar_input_text = ' '.join(bahan_dasar)
    tfidf_input = vectorizer.transform([bahan_dasar_input_text])
    prediction = model.predict(tfidf_input.toarray())
    similarities = cosine_similarity(prediction, tfidf_matrix)
    indeks_item_relevan = np.argsort(similarities.ravel())[::-1][:10]
    makanan_rekomendasi = data['Nama Makanan'].iloc[indeks_item_relevan].tolist()
    filtered_makanan_rekomendasi = filter_meal_plan(data, alergi, kehalalan, harga)
    filtered_makanan_rekomendasi = filtered_makanan_rekomendasi['Nama Makanan'].tolist()
    rekomendasi_final = list(set(makanan_rekomendasi) & set(filtered_makanan_rekomendasi))
    return rekomendasi_final


def filter_meal_plan(data, alergi, kehalalan, harga_max):
    data_filtered = data

    if alergi != '0':
        alergi_list = alergi.split(',')
        for alergi_item in alergi_list:
            data_filtered = data_filtered[~data_filtered['Alergi'].str.contains(alergi_item, case=False, na=False)]
    if kehalalan == '0':
        data_filtered = data_filtered[data_filtered['Kehalalan'] == 0]
    elif kehalalan == '1':
        data_filtered = data_filtered[data_filtered['Kehalalan'] == 1]
    data_filtered.loc[:, 'Harga'] = data_filtered['Harga'].str.replace('Rp', '').str.replace(',', '').astype(int)
    data_filtered = data_filtered[data_filtered['Harga'] <= harga_max]
    return data_filtered