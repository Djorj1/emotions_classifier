import yaml

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

def tokenizer(df):
    with open('model\params.yaml') as f:
        params = yaml.safe_load(f)

    print()
    print(params['token'])
    print()

    tokenizer = AutoTokenizer.from_pretrained(params['model']['bert'], token = params['token'])
    model = AutoModel.from_pretrained(params['model']['bert'], token = params['token'])
    model.eval()

    inputs = tokenizer(
        df,
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
    return df_embeddings
