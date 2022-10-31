import pandas as pd
import regex


def match_and_extract(row, *, text_col):
    pattern = r'\b(?<=(.{0,100}))('+row['trimmed_name']+r')\b(?=(.{0,100}))'
    matches = regex.findall(pattern, row[text_col], flags=regex.I|regex.DOTALL)
    matches = [x+y+z for x,y,z in matches]
    return matches


regex_chars = ['.','(', ')']
client_df = pd.read_csv('../data/clientes.csv')[['nit', 'nombre']]
clean_names = pd.read_excel('../data/clean_names.xlsx')
client_df = client_df.merge(clean_names, on='nombre')
for char in regex_chars:
    client_df['trimmed_name'] = client_df['trimmed_name'].str.replace(char, f'\{char}', regex=False)


client_news_df = pd.read_csv('../data/clientes_noticias.csv')[['nit', 'news_id']]
news_df = pd.read_csv('../data/noticias.csv')[['news_id','news_title','news_text_content']]


client_news_df = client_news_df.merge(news_df, on='news_id')
client_news_df = client_news_df.merge(client_df, on='nit')

client_news_df['appearance_in_title'] = client_news_df.apply(match_and_extract, text_col='news_title', axis=1)
client_news_df['appearance_in_body'] = client_news_df.apply(match_and_extract, text_col='news_text_content', axis=1)

client_news_df['name_in_title'] = client_news_df['appearance_in_title'].map(len) > 0
client_news_df['name_in_body'] = client_news_df['appearance_in_body'].map(len) > 0

client_news_df['appearance_in_title'] = client_news_df['appearance_in_title'].map(str)
client_news_df['appearance_in_body'] = client_news_df['appearance_in_body'].map(str)

print(client_news_df[client_news_df[['name_in_body', 'name_in_body']].any(axis=1)])

client_news_df[client_news_df[['name_in_body', 'name_in_body']].any(axis=1)].drop(columns=['news_title', 'news_text_content', 'trimmed_name']).to_excel('../data/text_matching.xlsx', index=False)