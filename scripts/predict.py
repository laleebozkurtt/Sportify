import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Model ve etiket kodlayıcıları yükleme
model_file_path = "../models/exercise_model.pkl"
data_file_path = "../data/exercise_data.csv"
with open(model_file_path, "rb") as model_file:
    model = pickle.load(model_file)

# Veri kümesinden etiket kodlayıcıları hazırlama
df = pd.read_csv(data_file_path)
categorical_columns = ["FitnessLevel", "Goal", "TargetArea", "Exercise"]

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Kullanıcı girdilerini normalize ederek alıyoruz
user_input = {
    "FitnessLevel": input("Fitness Level (Beginner, Intermediate, Advanced): ").strip().capitalize(),
    "Goal": input("Goal (Strength Building, Flexibility Enhancement, etc.): ").strip().title(),
    "TargetArea": input("Target Area (Arms, Legs, Back, etc.): ").strip().capitalize()
}

# Kullanıcı girdilerini sayısal değerlere dönüştürme
try:
    encoded_input = [
        label_encoders["FitnessLevel"].transform([user_input["FitnessLevel"]])[0],
        label_encoders["Goal"].transform([user_input["Goal"]])[0],
        label_encoders["TargetArea"].transform([user_input["TargetArea"]])[0]
    ]
except ValueError as e:
    print("Hata: Geçersiz giriş yaptınız. Lütfen aşağıdaki seçeneklerden birini girin:")
    print("FitnessLevel:", list(label_encoders["FitnessLevel"].classes_))
    print("Goal:", list(label_encoders["Goal"].classes_))
    print("TargetArea:", list(label_encoders["TargetArea"].classes_))
    exit()

# Sütun isimleriyle tahmin verisini oluşturma
encoded_input_df = pd.DataFrame([encoded_input], columns=["FitnessLevel", "Goal", "TargetArea"])
# Egzersiz olasılıklarını alma
exercise_proba = model.estimators_[0].predict_proba(encoded_input_df)

# Olasılık skorlarına göre en iyi 4 egzersizi sıralama
top_4_indices = exercise_proba[0].argsort()[-4:][::-1]
top_4_exercises = [label_encoders["Exercise"].inverse_transform([i])[0] for i in top_4_indices]

# En iyi 4 egzersiz için set ve tekrar sayısını alma
top_4_results = []
for exercise_encoded in top_4_indices:
    temp_input = encoded_input_df.copy()
    predicted = model.predict(temp_input)
    top_4_results.append({
        "Exercise": label_encoders["Exercise"].inverse_transform([exercise_encoded])[0],
        "Sets": predicted[0, 1],
        "Reps": predicted[0, 2]
    })

# Önerileri yazdırma
print("\nEn Uygun 4 Egzersiz:")
for i, result in enumerate(top_4_results, 1):
    print(f"{i}. Egzersiz: {result['Exercise']}, Setler: {result['Sets']}, Tekrarlar: {result['Reps']}")