from fastapi import FastAPI
from emotions_classifier.model.tokenizer import tokenizer
from emotions_classifier.model.zapusk_model import run_pipeline
# Создаем экземпляр приложения FastAPI
app = FastAPI(title="Emotions Classifier API", 
              description="API для классификации эмоций в тексте",
              version="1.0.0")

@app.post("/get_prediction")
def predict(text: str):

    prediction = run_pipeline(text = text)

    return {"pred": int(list(prediction)[0])}
    


