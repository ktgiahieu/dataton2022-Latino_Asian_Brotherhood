import pandas as pd


stopwords = pd.read_csv('../data/stopwords.txt', names=['stopword'], header=None)
stopwords = stopwords['stopword']
stopwords = stopwords.str.replace('.', r'\.', regex=False)

clients = pd.read_csv('../data/clientes.csv', usecols=['nombre'])
clients['trimmed_name'] = clients['nombre']

for word in stopwords:
    clients['trimmed_name'] = clients['trimmed_name'].str.replace(r'(?:\s|^){}(?:\s|$)'.format(word), ' ', regex=True).str.strip()


writer = pd.ExcelWriter('../data/clean_names.xlsx') 
clients.to_excel(writer, index=False, sheet_name='Sheet1')

for column_name in clients.columns:
    column_length = max(clients[column_name].str.len().max(), len(column_name))
    col_idx = clients.columns.get_loc(column_name)
    writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_length)

writer.save()