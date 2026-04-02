"""
Модуль по векторизация предложения
    1. Векторизация текста
"""

import yaml

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Взятие констант
with open('emotions_classifier/params.yaml', "r", encoding = "utf-8") as f:
    params = yaml.safe_load(f)
model_name = params["model"]["bert"]

# Название модели, которую мы используем
# model_name = "answerdotai/ModernBERT-base"

# Токенизатор, функция разбивающая текст на токены
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Загрузка модели
model = AutoModel.from_pretrained(model_name)

# Сам текст
text = params["text"]

# Токенизация текста
inputs = tokenizer(text, return_tensors = "pt", padding = True, truncation = True)

# 
with torch.no_grad():
    outputs = model(**inputs)


token_embeddings = outputs.last_hidden_state[0]

print(f"Форма эмбедингов: {token_embeddings.shape}")
print(f"Пример эмбединга первого токена: {token_embeddings[0][:10]}")
print(f"Количество ненулевых значений: {torch.count_nonzero(token_embeddings[0])} из {len(token_embeddings[0])}")

