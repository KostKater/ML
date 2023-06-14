from model10 import load_data, preprocess_data, train_model, load_model, recommend_meal_plan,get_resep
if __name__ == '__main__':
    data = load_data()
    data, vectorizer, tfidf_matrix = preprocess_data(data)
    #model = train_model(tfidf_matrix, vectorizer)
    model = load_model()
    bahan_dasar = ['telur']
    alergi = ''
    kehalalan = '0'
    harga_max = 99999999
    harga_min = 0
    recommendations = recommend_meal_plan(data, vectorizer, model, tfidf_matrix, bahan_dasar, alergi, kehalalan, harga_min,harga_max,bahan_dasar)
    for rekomendasi in recommendations:
        print("Nama Makanan:", rekomendasi['name'])
        print("=========")
    #print(recommendations)

"""data = load_data() 
nama_makanan = 'Babi kecap kentang' 

resep_list, bahan_list = get_resep(data, nama_makanan)

print("Resep:")
print(resep_list)

print("\nBahan Makanan:")
print(bahan_list)
"""