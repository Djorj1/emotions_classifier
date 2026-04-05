"""
Модель-классификатор
    1. Подготовка даасета
    2. Токенизация
    3. Сбалансирование классов
    4. Обучение модели
    5. Вывод оценки
"""

import yaml

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from sklearn.utils import resample
from emotions_classifier.model.learning_model import train_model
# from emotions_classifier.model.learning_model_with_optuna import optimize_hyperparameters, train_best_model
from emotions_classifier.model.learning_model_with_rfe_and_optuna import optimize_hyperparameters, train_best_model

with open("C:/Users/Георгий/Desktop/budcemp/emotions_classifier/emotions_classifier/model/params.yaml", "r") as f:
    params = yaml.safe_load(f)
model_name = params['model']['bert']


ds = load_dataset("Xeonil/ru-merged-toxic-comments", token = params['token'])
df = ds['train'].to_pandas()


tokenizer = AutoTokenizer.from_pretrained(model_name, token = params['token'])
model = AutoModel.from_pretrained(model_name, token = params['token'])
model.eval()


df_toxic = df.loc[df["target"] == 1 ][:500]
df_friend = df.loc[df["target"] == 0 ][:500]
df = pd.concat([df_toxic, df_friend]).reset_index(drop = True)


# inputs = tokenizer.encode(df['text'].tolist(), return_tensors='pt', padding = True, truncation = True, n_jobs = -1)
inputs = tokenizer(
    df['text'].tolist(),
    return_tensors='pt',
    padding=True,
    truncation=True,
    max_length=1024
)



with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state[:, 0, :]

embedding_columns = [f'feature_{i}' for i in range(len(embeddings[0]))]
df_embeddings = pd.DataFrame(embeddings, columns=embedding_columns)


df_embeddings['target'] = df['target'].values


print("\nПервые 5 строк датафрейма с эмбеддингами:")
print(df_embeddings.head())

# acc, f1, crr = train_model(df_embeddings)
# print(f"Accuracy: {acc}")
# print(f"F1-score: {f1}")
# print("\nClassification Report:")
# print(crr)

# result = optimize_hyperparameters(df_embeddings, n_trials = 100)
# print(result['best_params'])
# best_model_result = train_best_model(df_embeddings, result['best_params'])
# print(f"Accuracy: {best_model_result['accuracy']}")
# print(f"F1-score: {best_model_result['f1_score']}")
# print("\nClassification Report:")
# print(best_model_result['classification_report'])

result = optimize_hyperparameters(df_embeddings, n_trials = 100)
print(result['best_params'])
best_model_result = train_best_model(df_embeddings, result['best_params'])
print(f"Accuracy: {best_model_result['accuracy']}")
print(f"F1-score: {best_model_result['f1_score']}")
print("\nClassification Report:")
print(best_model_result['classification_report'])