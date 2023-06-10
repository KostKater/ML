import pandas as pd

# Membaca file CSV ke dalam DataFrame
df = pd.read_csv('MealPlan.csv')

# Melihat tipe data kolom
print(df['Resep'].head())