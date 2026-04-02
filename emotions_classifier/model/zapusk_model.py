import pickle
from emotions_classifier.model.tokenizer import tokenizer


def run_pipeline(text):
    with open('model.pkl', 'rb') as file: 
        loaded_model = pickle.load(file)

    print("Модель успешно загружена!")

    new_data = [text]  # пример: 4 признака

    # Делаем предсказание
    new = tokenizer(new_data)
    prediction = loaded_model.predict(new)

    return prediction

if __name__ == "__main__":

    run_pipeline(text = "разочек катком по твоему лицо проедц оно и не изменится хуйня азиатская")