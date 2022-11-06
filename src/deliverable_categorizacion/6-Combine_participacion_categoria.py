import random
import torch
import numpy as np
import pandas as pd
import os
import regex

from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer

CLASSIFICATION_THRESHOLD = 0.7

def match_and_extract(row, *, text_col):
    pattern = r'\b(?<=(.{0,100}))('+'(.{0,20})'.join(row['trimmed_name'].split())+r')\b(?=(.{0,100}))'
    matches = regex.findall(pattern, row[text_col], flags=regex.I|regex.DOTALL)
    matches = [' '.join(x) for x in matches]
    return matches

def get_paricipacion(row):
        if row.group == row.pred_group and np.max(row[group_columns].values)>CLASSIFICATION_THRESHOLD:
            if row.name_in_news:
                return 'Cliente'
            else:
                return 'Sector'
        else:
            return 'No aplica'

def get_categoria(row):
    if np.max(row[category_columns].values)>CLASSIFICATION_THRESHOLD and row.pred_categoria != 'Descartable':
        return row.pred_categoria
    else:
        if row.participacion != 'No aplica':
            return 'Otra'
        else:
            return 'Descartable'

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
    # ------ 1. TRIM CLIENTS' NAME ---------
    stopwords = pd.read_csv('../data/archivos_auxiliares/stopwords.txt', names=['stopword'], header=None)
    stopwords = stopwords['stopword']
    stopwords = stopwords.str.replace('.', r'\.', regex=False)

    clients = pd.read_csv('../data/archivos_auxiliares/clientes.csv', usecols=['nombre'])
    clients['trimmed_name'] = clients['nombre']
    for word in stopwords:
        clients['trimmed_name'] = clients['trimmed_name'].str.replace(r'(?:\s|^){}(?:\s|$)'.format(word), ' ', regex=True).str.strip()

    clients.to_csv('../data/archivos_auxiliares/clean_names.csv', index=False)



    # ------ 2. FIND ALL MATCHES OF CLIENTS IN THE NEWS ---------
    regex_chars = ['.','(', ')']
    client_df = pd.read_csv('../data/archivos_auxiliares/clientes.csv')

    # regroup_desc_ciiu_division.csv is a manually created CSV file, 
    # which have bigger groups than desc_ciiu_division
    regroup_desc_ciiu_division = pd.read_csv('../data/archivos_auxiliares/regroup_desc_ciiu_division.csv')
    client_df = pd.merge(client_df, regroup_desc_ciiu_division, how='left', on='desc_ciiu_division')
    client_df['group'] = 'group_' + client_df['group'].astype(str)
    clean_names = pd.read_csv('../data/archivos_auxiliares/clean_names.csv')
    client_df = client_df.merge(clean_names, on='nombre')
    for char in regex_chars:
        client_df['trimmed_name'] = client_df['trimmed_name'].str.replace(char, f'\{char}', regex=False)


    client_news_df = pd.read_csv('../data/archivos_auxiliares/clientes_noticias.csv')[['nit', 'news_url_absolute','news_id']]
    news_df = pd.read_csv('../data/archivos_auxiliares/noticias.csv')


    client_news_df = client_news_df.merge(news_df, on='news_id')
    client_news_df = client_news_df.merge(client_df, on='nit')

    client_news_df['appearance_in_title'] = client_news_df.apply(match_and_extract, text_col='news_title', axis=1)
    client_news_df['appearance_in_body'] = client_news_df.apply(match_and_extract, text_col='news_text_content', axis=1)

    client_news_df['name_in_title'] = client_news_df['appearance_in_title'].map(len) > 0
    client_news_df['name_in_body'] = client_news_df['appearance_in_body'].map(len) > 0
    client_news_df['name_in_news'] = client_news_df['name_in_title'] | client_news_df['name_in_body']

    client_news_df['appearance_in_title'] = client_news_df['appearance_in_title'].map(str)
    client_news_df['appearance_in_body'] = client_news_df['appearance_in_body'].map(str)

    client_news_df = client_news_df[['nit', 'news_id', 'nombre', 'desc_ciiu_division', 'group', 'trimmed_name', 'name_in_news']]

    # ------ 3. LOAD PREDICTIONS OF "paricipacion" AND "categoria" ---------

    pred_group = pd.read_csv('../data/output/pred_news_group.csv')
    pred_category = pd.read_csv('../data/output/pred_news_categoria.csv')

    group_columns = ['group_30', 'group_40', 'group_28', 'group_29', 'group_20', 'group_25', 'group_1', 'group_32', 'group_16', 'group_23', 'group_11', 'group_22', 'group_19', 'group_39', 'group_49', 'group_2', 'group_44', 'group_18', 'group_13', 'group_26', 'group_41', 'group_36', 'group_31', 'group_14', 'group_12', 'group_27', 'group_43', 'group_21', 'group_7', 'group_46', 'group_35', 'group_48', 'group_33', 'group_38', 'group_5', 'group_4', 'group_47', 'group_6', 'group_42']
    pred_group = pred_group[['nit', 'news_id', 'pred_group'] + group_columns]

    category_columns = ['Macroeconomía', 'Sostenibilidad', 'Innovación', 'Regulaciones', 'Alianza', 'Reputación', 'Descartable']
    pred_category = pred_category[['nit', 'news_id', 'preds', 'pred_categoria'] + category_columns]

    pred_df = client_news_df.merge(pred_group, on=['nit','news_id']).merge(pred_category, on=['nit','news_id'])

    # ------ 4. COMBINE PREDICTIONS OF "paricipacion" AND "categoria" AND "name_in_news" ---------     
    pred_df['participacion'] = pred_df.apply(get_paricipacion,1)
    pred_df['categoria'] = pred_df.apply(get_categoria,1)

    pred_df['nombre_equipo'] = 'Latino-Asian Brotherhood'
    pred_df[['nombre_equipo', 'nit', 'news_id', 'participacion', 'categoria']].to_csv('../data/output/categorizacion.csv', index=False)