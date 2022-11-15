import random
import torch
import numpy as np
import pandas as pd
import os
import pickle

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from sentence_transformers import SentenceTransformer

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
    # Crop 10000 first characters of news, but reserve sentences
    def crop_news(text):
        if len(text)>10000:
            sentences = sent_tokenize(text[9000:11000])
            cum_sum_len = np.cumsum([len(x) for x in sentences])
            if cum_sum_len[-1]>=1000:
                sentence_idx_to_cut = np.where(cum_sum_len >= 1000)[0][0]
                return text[:9000] + ' '.join(sentences[:sentence_idx_to_cut+1])
            else:
                return text
        else:
            return text
    # client_news_df['text'] = client_news_df['text'].apply(crop_news)
    client_news_df['text'] = client_news_df['text'].apply(lambda x: x[:100])


    # ------------ 2. INFER SENTENCE TRANSFORMER MODEL --------------
    # Load Sentence Transformers multilingual model 
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    # Get embeddings
    news_embeddings = model.encode(client_news_df.text.values)

    output_dir = '../data/intermediate_output'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open('../data/intermediate_output/news_embeddings.pkl', 'wb') as handle:
        pickle.dump(news_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    