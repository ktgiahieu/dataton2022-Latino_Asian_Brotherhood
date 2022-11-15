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
    # ------------ 1. RANDOM SAMPLING 16 SAMPLES EACH GROUP --------------
    df = pd.read_csv('../data/intermediate_output/matched_news_group.csv')
    df.rename(columns={'group':'label_text'}, inplace=True)
    df['text'] = df['news_title'] + '. ' + df['news_text_content']
    labels = df.label_text.unique()

    labels_map = {}
    for i, l in enumerate(labels):
        labels_map[l]=i

    df['label'] = df.label_text.replace(labels_map)

    df = df.groupby('label_text').apply(lambda x: x.sample(2,replace=False, random_state=2022) if x.news_id.count()>=2 else x).reset_index(drop=True)
    df.to_csv('../data/intermediate_output/matched_news_group_sampled.csv', index=False)

    # ------------ 2. TRAIN SETFIT MODEL --------------
    
    # Load dataset
    dataset = load_dataset("csv", data_files="../data/intermediate_output/matched_news_group_sampled.csv")
    train_ds = dataset["train"]

    # Load SetFit model from Hub
    model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    # Create trainer
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        loss_class=CosineSimilarityLoss,
        batch_size=8,
        num_iterations=20, # Number of text pairs to generate for contrastive learning
        num_epochs=1, # Number of epochs to use for contrastive learning
        seed=2022
    )
    # Train
    trainer.train()

    # Save model
    model.save_pretrained('../data/intermediate_output/trained_setfix_participacion')