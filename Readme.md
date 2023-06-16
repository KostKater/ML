**Meal Planning Recommendation System**                                                                                           
The Meal Planning Recommendation System is a Python-based model that provides personalized meal recommendations based on user preferences. It utilizes natural language processing techniques, machine learning, and data preprocessing to generate meal suggestions that match the user's dietary requirements, budget, and ingredient preferences.

**Installation**                                            
To run the meal planning recommendation system, follow these steps:
1. Clone the repository
```bash
git clone https://github.com/your-username/meal-planning-recommendation.git
cd meal-planning-recommendation
```
2. Install the required dependencies:
```bash
pip install pandas numpy scikit-learn tensorflow
```
3.Prepare the data:
Place the meal plan dataset file (MealPlan.csv) in the project directory.
4.Run the application: 
Test Model with run Test.py

**Usage**                                                      
The meal planning recommendation system offers the following functionalities:
Personalized Meal Recommendations: Based on your input preferences (ingredient, dietary restrictions, budget, etc.), the model provides a list of recommended meal plans. Each recommendation includes details such as the meal name, description, image URL, halal status, nutrition information, price, and recipe.
Filtering Options: You can further refine the recommendations by specifying additional filters, such as allergens, halal preference, price range, and ingredient preferences. The system will adjust the recommendations accordingly.
Recipe and Ingredients: You can access the recipe and ingredient list for any recommended meal plan. The recipe includes step-by-step instructions for preparing the meal.

**Data**                                            
The model utilizes a dataset (MealPlan.csv) containing information about various meal plans. The dataset includes the following columns:                                     
- Nama Makanan: The name of the meal plan.                                                         
- Deskripsi: A brief description of the meal plan.                                              
- Gambar: URL to an image representing the meal plan.                                        
- Bahan Dasar: Preprosesing from Bahan Makanan
- Kehalalan: Halal status of the meal plan (0 = non-halal, 1 = halal).
- Kalori: Caloric content of the meal plan.
- Lemak: Fat content of the meal plan.
- Karbohidrat: Carbohydrate content of the meal plan.
- Protein: Protein content of the meal plan.
- Harga: Price of the meal plan.
- Bahan Makanan: List of ingredients required for the meal plan.
- Resep: Step-by-step instructions for preparing the meal plan.
 
