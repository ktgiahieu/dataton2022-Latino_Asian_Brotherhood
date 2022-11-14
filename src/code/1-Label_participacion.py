import pandas as pd
import regex
import pickle

import pandas as pd
import regex
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import spatial

# THIS FILE CREATE (news, client) PAIR AS LABELS, USING REGEX
# THESE LABELS WILL BE USED TO TRAIN A SentenceTransformers TO CLASSIFY THE SECTOR OF THE NEWS

def match_and_extract(row, *, text_col):
    """
    Find all matches of clients' trimmed names in text
    """
    pattern = r'\b(?<=(.{0,100}))('+'(.{0,20})'.join(row['trimmed_name'].split())+r')\b(?=(.{0,100}))'
    matches = regex.findall(pattern, row[text_col], flags=regex.I|regex.DOTALL)
    matches = [' '.join(x) for x in matches]
    return matches

def get_cosine_similarity(vector1, vector2):
    sim = 1 - spatial.distance.cosine(vector1, vector2)
    return sim

#This dictionary contains client name correction
client_name_correction = {
    'CELSA SAS': 'METALURGICA CELSA',
    'AIR E': 'AIR-E',
    'MARVAL': 'CONSTRUCTORA  MARVAL'
}

# This list contains client's name that have meaning like a normal word,
# which is bad for matching
bad_client_names = ['NIVEL',
            'SI',
            'PROTECCION',
            'EFICACIA',
            'CADENA',
            'P R',
            'CONTINENTE',
            'ACTIVOS',
            'EFECTIVO',
            'MINEROS',
            'ETERNA',
            'MAYORISTA',
            'ADQUIRIR',
            'DIVISA',
            'SURAMERICANA',
            'LISTOS',
            'SERVICIOS PUBLICOS',
            'ULTRA',
            'ARME',
            'SOBERANA',
            'CAMAGUEY',
            'SOLLA',
            'MAYAGUEZ',
            'HADA',
            'CASCADA',
            'NORMANDIA',
            'CRYSTAL',
            'AR',
            'TIEMPOS'
            ]


if __name__ == "__main__":
    # IMPORTANT: Fix missing values in clientes.csv
    client_df = pd.read_csv('../data/archivos_auxiliares/clientes.csv')
    new_row = {'nit':860516431, 
               'nombre':'GRAN TIERRA ENERGY COLOMBIA', 
               'desc_ciiu_division':'EXTRACCION DE PETROLEO CRUDO Y GAS NATURAL', 
               'desc_ciuu_grupo':'EXTRACCION DE PETROLEO CRUDO',
               'desc_ciiuu_clase': 'EXTRACCION DE PETROLEO CRUDO',
               'subsec': 'EXTRACCION DE PETROLEO'
              }
    client_df = client_df.append(new_row, ignore_index=True)

    # ------ 1. TRIM CLIENTS' NAME ---------
    stopwords = pd.read_csv('../data/archivos_auxiliares/stopwords.txt', names=['stopword'], header=None)
    stopwords = stopwords['stopword']
    stopwords = stopwords.str.replace('.', r'\.', regex=False)

    client_df['trimmed_name'] = client_df['nombre']
    for word in stopwords:
        client_df['trimmed_name'] = client_df['trimmed_name'].str.replace(r'(?:\s|^){}(?:\s|$)'.format(word), ' ', regex=True).str.strip()

    # ------ 2. FIND ALL MATCHES OF CLIENTS IN THE NEWS ---------
    regex_chars = ['.','(', ')']
    for char in regex_chars:
        client_df['trimmed_name'] = client_df['trimmed_name'].str.replace(char, f'\{char}', regex=False)
        
    # regroup_desc_ciiu_division.csv is a manually created CSV file, 
    # which have bigger groups than desc_ciiu_division
    regroup_desc_ciiu_division = pd.read_csv('../data/archivos_auxiliares/regroup_desc_ciiu_division.csv')
    client_df = pd.merge(client_df, regroup_desc_ciiu_division, how='left', on='desc_ciiu_division')
    client_df['group'] = 'group_' + client_df['group'].astype(str)
        
    client_news_df = pd.read_csv('../data/archivos_auxiliares/clientes_noticias.csv')[['nit','news_id']]
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

    # Only use (news, client) pair that have client name in news, and (news23736, 800047031) is also a good pair
    matched_news = client_news_df[((client_news_df.name_in_body==True) | (client_news_df.name_in_title==True) | \
                            ((client_news_df.news_id == 'news23736') & (client_news_df.nit == 800047031))) & \
                            (~client_news_df.trimmed_name.isin(bad_client_names))]

    # Only use news that have 2 client (prevent news that lists a lot of companies)
    matched_news = pd.merge(matched_news, matched_news.groupby('news_id').nit.count().to_frame().rename(columns={'nit':'count'}), on='news_id')
    matched_news = matched_news[matched_news['count'] <= 2]

    #Remove special case where Bayer Leverkusen is mistaken as Bayer company
    matched_news = matched_news[~matched_news.apply(lambda row: ('Bayer Leverkusen' in row['news_text_content']) or ('Bayer Leverkusen' in row['news_title']), 1)]
    

    # ------ 3. CREATE PSEUDO-LABEL ---------
    with open('../data/intermediate_output/news_embeddings.pkl', 'rb') as handle:
        news_embeddings = pickle.load(handle)

    # Load Sentence Transformers multilingual model 
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

    desc_ciiu_division_embeddings = model.encode(client_news_df.desc_ciiu_division.unique())
    desc_ciuu_grupo_embeddings = model.encode(client_news_df.desc_ciuu_grupo.unique())
    desc_ciiuu_clase_embeddings = model.encode(client_news_df.desc_ciiuu_clase.unique())

    desc_ciiu_division_embeddings_dict = {k:desc_ciiu_division_embeddings[i] for i, k in enumerate(client_news_df.desc_ciiu_division.unique())}
    desc_ciuu_grupo_embeddings_dict = {k:desc_ciuu_grupo_embeddings[i] for i, k in enumerate(client_news_df.desc_ciuu_grupo.unique())}
    desc_ciiuu_clase_embeddings_dict = {k:desc_ciiuu_clase_embeddings[i] for i, k in enumerate(client_news_df.desc_ciiuu_clase.unique())}
    
    client_news_df['desc_ciiu_division_similarity'] = client_news_df.apply(
        lambda x: get_cosine_similarity(desc_ciiu_division_embeddings_dict[x.desc_ciiu_division], 
                                        news_embeddings[x.name]), 1)
    client_news_df['desc_ciuu_grupo_similarity'] = client_news_df.apply(
        lambda x: get_cosine_similarity(desc_ciuu_grupo_embeddings_dict[x.desc_ciuu_grupo], 
                                        news_embeddings[x.name]), 1)
    client_news_df['desc_ciiuu_clase_similarity'] = client_news_df.apply(
        lambda x: get_cosine_similarity(desc_ciiuu_clase_embeddings_dict[x.desc_ciiuu_clase], 
                                        news_embeddings[x.name]), 1)
    client_news_df['client_similarity'] =   client_news_df['desc_ciiu_division_similarity'] * \
                                            client_news_df['desc_ciuu_grupo_similarity'] * \
                                            client_news_df['desc_ciiuu_clase_similarity']
    max_similarity_each_nombre_dict = client_news_df.groupby('nombre').client_similarity.max().to_dict()
    client_news_df['max_similarity_each_nombre'] = client_news_df['nombre'].replace(max_similarity_each_nombre_dict)


    # For group < 200 samples, add news that related to CIIU description
    to_be_added_group_dict = (matched_news.group.value_counts() < 200).to_dict()
    to_be_added_group_dict['group_3'] = True
    matched_news['to_be_added_group'] = matched_news.group.replace(to_be_added_group_dict)
    
    new_rows = []
    for group in matched_news.group.unique():
        if to_be_added_group_dict[group]:
            for new_row in client_news_df[(client_news_df.group==group)&\
                (client_news_df.client_similarity >= client_news_df.max_similarity_each_nombre)].to_dict(orient="records"):
                new_rows.append(new_row)

    new_rows = pd.DataFrame(new_rows)
    new_rows_sampled = new_rows.groupby('group').apply(lambda x: x.sample(n=min(100,len(x)))).reset_index(drop=True)

    matched_news = pd.concat([matched_news, new_rows_sampled])

    matched_news.to_csv('../data/intermediate_output/matched_news_group.csv', index=False)
