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
from emotions_classifier.learning_model import train_model


with open("emotions_classifier/params.yaml", "r") as f:
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
# print(df.head(3))

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

acc, f1, crr = train_model(df_embeddings)

print(f"Accuracy: {acc}")
print(f"F1-score: {f1}")
print("\nClassification Report:")
print(crr)

