import random
import torch
import numpy as np
import pandas as pd
import os

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
set_seed(2022)

if __name__ == "__main__":
    # ------------ 1. PREPARE DATA --------------
    client_news_df = pd.read_csv('../data/archivos_auxiliares/clientes_noticias.csv')[['nit', 'news_url_absolute','news_id']]
    news_df = pd.read_csv('../data/archivos_auxiliares/noticias.csv')

    client_news_df = client_news_df.merge(news_df, on='news_id')
    client_news_df['text'] = client_news_df.news_title + '. ' + client_news_df.news_text_content
    # Crop only 10000 first characters of news
    client_news_df['text'] = client_news_df['text'].apply(lambda x: x[:10000])

    # ------------ 2. INFER SETFIT MODEL --------------
    # Load SetFit model 
    model = SetFitModel.from_pretrained('../data/archivos_auxiliares/trained_setfix_categoria')

    # Predict
    preds_proba = model.predict_proba(client_news_df.text.values)
    labels = ['Macroeconomía','Sostenibilidad','Innovación','Regulaciones','Alianza','Reputación','Descartable']
    for i, l in enumerate(labels):
        client_news_df[l] = preds_proba[:,i]
    client_news_df['preds'] = np.argmax(preds_proba,1)

    labels_map_inverse = {}
    for i, l in enumerate(labels):
        labels_map_inverse[i]=l
    client_news_df['pred_categoria'] = client_news_df['preds'].replace(labels_map_inverse)

    client_news_df.to_csv('../data/output/pred_news_categoria.csv', index=False)
    