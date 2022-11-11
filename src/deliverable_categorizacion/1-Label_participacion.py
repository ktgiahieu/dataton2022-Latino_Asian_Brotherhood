import pandas as pd
import regex

import pandas as pd
import regex
import numpy as np

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
    # ------ 1. TRIM CLIENTS' NAME ---------
    stopwords = pd.read_csv('../data/archivos_auxiliares/stopwords.txt', names=['stopword'], header=None)
    stopwords = stopwords['stopword']
    stopwords = stopwords.str.replace('.', r'\.', regex=False)

    clients = pd.read_csv('../data/archivos_auxiliares/clientes.csv', usecols=['nombre'])
    for k,v in client_name_correction.items():
        clients_idx = clients[clients.nombre==k].index
        clients.at[clients_idx,'nombre'] = v
    clients['trimmed_name'] = clients['nombre']
    for word in stopwords:
        clients['trimmed_name'] = clients['trimmed_name'].str.replace(r'(?:\s|^){}(?:\s|$)'.format(word), ' ', regex=True).str.strip()

    clients.to_csv('../data/archivos_auxiliares/clean_names.csv', index=False)

    # ------ 2. FIND ALL MATCHES OF CLIENTS IN THE NEWS ---------
    regex_chars = ['.','(', ')']
    client_df = pd.read_csv('../data/archivos_auxiliares/clientes.csv')
    for k,v in client_name_correction.items():
        clients_idx = client_df[client_df.nombre==k].index
        client_df.at[clients_idx,'nombre'] = v

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

    client_news_df['appearance_in_title'] = client_news_df['appearance_in_title'].map(str)
    client_news_df['appearance_in_body'] = client_news_df['appearance_in_body'].map(str)

    # Only use (news, client) pair that have client name in news, and (news23736, 800047031) is also a good pair
    matched_news = client_news_df[((client_news_df.name_in_body==True) | (client_news_df.name_in_title==True) | \
                            ((client_news_df.news_id == 'news23736') & (client_news_df.nit == 800047031))) & \
                            (~client_news_df.trimmed_name.isin(bad_client_names))]

    # Only use news that have 2 client (prevent news that lists a lot of companies)
    matched_news = pd.merge(matched_news, matched_news.groupby('news_id').nit.count().to_frame().rename(columns={'nit':'count'}), on='news_id')
    matched_news = matched_news[matched_news['count'] < 3]

    #Remove special case where Bayer Leverkusen is mistaken as Bayer company
    matched_news = matched_news[~matched_news.apply(lambda row: ('Bayer Leverkusen' in row['news_text_content']) or ('Bayer Leverkusen' in row['news_title']), 1)]

    matched_news.to_csv('../data/archivos_auxiliares/matched_news_group.csv', index=False)
