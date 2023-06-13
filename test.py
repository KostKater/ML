from model10 import load_data, preprocess_data, train_model, load_model, recommend_meal_plan

if __name__ == '__main__':
    data = load_data()
    data, vectorizer, tfidf_matrix = preprocess_data(data)
    model = train_model(tfidf_matrix, vectorizer)
    model = load_model()
    bahan_dasar = ['ayam']
    alergi = ''
    kehalalan = '1'
    harga_max = 20000
    harga_min = 0
    recommendations = recommend_meal_plan(data, vectorizer, model, tfidf_matrix, bahan_dasar, alergi, kehalalan, harga_min,harga_max,bahan_dasar)
