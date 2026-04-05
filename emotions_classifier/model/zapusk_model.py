import pickle
from emotions_classifier.model.tokenizer import tokenizer


def run_pipeline(text):
    with open("model.pkl", 'rb') as file: 
        loaded_model = pickle.load(file)

    print("Модель успешно загружена!")

    new_data = [text]  # пример: 4 признака

    # Делаем предсказание
    new = tokenizer(new_data)
    prediction = loaded_model.predict(new)

    return prediction