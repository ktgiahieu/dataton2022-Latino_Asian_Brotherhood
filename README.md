# Datatón 2022 - Team Latino-Asian Brotherhood

This repo contains terminos_y_condiciones.pdf and the final submission code for reproducing the results.

### Environment

To setup the environment:
* Install python >= 3.8
* Install `requirements.txt` in the fresh python environment (Here we use CUDA 11.2. If you use different CUDA version, please change torch version)

## Categoria: Main solution

### Training
To train a **Categoria** classifier, run `src/recomendador/train_categoria.sh`

### Inference
To predict news category using **Categoria** classifier, run `src/recomendador/predict_categoria.sh`

### TLDR
#### Participacion

* We first use word matching to find news related to each client in `clientes_noticias.csv`. This will create a `matched_news_group.csv`.
* We then use this as a pseudo-label to train a [SetFit Model](https://huggingface.co/blog/setfit) to classify which Sector each news is in.
![image](https://user-images.githubusercontent.com/42331617/200150838-12907ea0-d172-47a8-adfd-28cc9baf25cb.png)
SetFit first fine-tunes a Sentence Transformer model (`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) on a small number of labeled examples (typically 16 per class). This is followed by training a classifier head on the embeddings generated from the fine-tuned Sentence Transformer.

#### Categoria
* We first label 28 random news for each category in [this file](https://github.com/ktgiahieu/dataton2022-Latino_Asian_Brotherhood/blob/khuong_categorize/src/data/archivos_auxiliares/category_label_1st_round.csv)
* We train a 1st round [SetFit Model](https://huggingface.co/blog/setfit) to classify which Category each news is in.
* We then re-label wrongly classified news for each category by the previous model, and save it in [this file](https://github.com/ktgiahieu/dataton2022-Latino_Asian_Brotherhood/blob/khuong_categorize/src/data/archivos_auxiliares/category_label_2nd_round.csv)
* We train a 2nd round [SetFit Model](https://huggingface.co/blog/setfit) to classify which Category each news is in.

#### Combine results
##### Pariticipacion
A (client, news) pair is classified as:
* *Cliente* if its name appeared in the news and have the same Sector
* *Sector* if its name did not appear in the news but have the same Sector
* *No aplica* otherwise\

##### Categoria
A (client, news) pair is classified as:
* *Otra* if they have the same Sector and the news is classified as **Descartable**
* *Descartable* if they don't have the same Sector and the news is classified as **Descartable**
* The original category classified by the model: otherwise

## Recomendador: Main solution
(To be added)
