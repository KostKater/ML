from model10 import load_data, preprocess_data, train_model, load_model, recommend_meal_plan

if __name__ == '__main__':
    data = load_data()
    data, vectorizer, tfidf_matrix = preprocess_data(data)
    model = train_model(tfidf_matrix, vectorizer)
    model = load_model()
    bahan_dasar = ['telur']
    alergi = ''
    kehalalan = '0'
    harga_max = 20000
    harga_min = 0
    recommendations = recommend_meal_plan(data, vectorizer, model, tfidf_matrix, bahan_dasar, alergi, kehalalan, harga_min,harga_max,bahan_dasar)
    for rekomendasi in recommendations:
        print("Nama Makanan:", rekomendasi['name'])
        print("Deskripsi:", rekomendasi['deskripsi'])
        print("Image URL:", rekomendasi['img_url'])
        print("Kehalalan:", rekomendasi['kehalalan'])
        print("Nutrisi:")
        print("- Kalori:", rekomendasi['nutrisi']['kalori'])
        print("- Lemak:", rekomendasi['nutrisi']['lemak'])
        print("- Karbohidrat:", rekomendasi['nutrisi']['karbohidrat'])
        print("- Protein:", rekomendasi['nutrisi']['protein'])
        print("Harga:", rekomendasi['harga'])
        print("Resep:")
        print("- Bahan Makanan:", rekomendasi['recipe']['bahan Makanan'])
        print("- Resep:", rekomendasi['recipe']['resep'])
        print("=========")
    #print(recommendations)