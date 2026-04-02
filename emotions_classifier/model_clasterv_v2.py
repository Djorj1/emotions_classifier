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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from datasets import load_dataset
from sklearn.utils import resample

device = torch.device("cpu")

# Загрузка датасета
ds = load_dataset("Xeonil/ru-merged-toxic-comments", token="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")
df = ds['train'].to_pandas()

# Загрузка параметров
with open("emotions_classifier/params.yaml", "r") as f:
    params = yaml.safe_load(f)
model_name = params['model']['bert']

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
model = AutoModel.from_pretrained(model_name, token="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")
model = model.to(device)
model.eval()

# Балансировка классов
df_toxic = df.loc[df["target"] == 1][:500]
df_friend = df.loc[df["target"] == 0][:500]
df = pd.concat([df_toxic, df_friend]).reset_index(drop=True)

# Функция для получения эмбеддингов
def get_embeddings(texts, batch_size=32):
    """Получение эмбеддингов для списка текстов"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Токенизация батча
        inputs = tokenizer(
            batch_texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Берем эмбеддинги [CLS] токена (первый токен)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings)

# Получение эмбеддингов
print("Извлечение эмбеддингов...")
texts = df['text'].tolist()
embeddings = get_embeddings(texts, batch_size=32)

print(f"Форма эмбеддингов: {embeddings.shape}")

# Создание DataFrame с эмбеддингами
embedding_columns = [f'feature_{i}' for i in range(embeddings.shape[1])]
df_embeddings = pd.DataFrame(embeddings, columns=embedding_columns)
df_embeddings['target'] = df['target'].values

print("\nПервые 5 строк датафрейма с эмбеддингами:")
print(df_embeddings.head())

# Подготовка данных для обучения
X = df_embeddings.drop(columns='target')
y = df_embeddings['target']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучение модели
print("\nОбучение классификатора...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Предсказание и оценка
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))