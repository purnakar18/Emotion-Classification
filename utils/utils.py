import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
from sklearn.model_selection import train_test_split


def get_text_embeddings(text_sentences, tokenizer, model):
  encoded_inputs = tokenizer(text_sentences, padding=True, truncation=True, return_tensors='pt')

# Model inference
  with torch.no_grad():
    outputs = model(**encoded_inputs)

# Extract the embeddings
  embeddings = outputs.last_hidden_state

# Apply pooling to reduce dimensionality to 1D
  pooled_embeddings = torch.mean(embeddings, dim=1)  # Mean pooling

# Convert to numpy array
  embeddings_1d = pooled_embeddings.numpy()

  return embeddings_1d.flatten()

def write_tags(Tags):

    temp=pd.DataFrame(Tags)
    temp.to_csv('Tags.csv',index=False)

    return 0

def read_tags(text, tokenizer, model):
    Tags = []
    for i in range(len(text)):
        embed = get_text_embeddings(text[i], tokenizer, model)
        Tags.append(embed)
    Tags = np.array(Tags)

    write_tags(Tags)
    return Tags

def read_data(model_type):

    df = pd.read_csv("emotions_final.csv")
    df.head()
    files_list = df['file_name'].tolist()
    text = df['text'].tolist()
    df.drop(['file_name', 'text','style'], axis=1, inplace=True)
    '''df = pd.read_csv("Info.csv")
    df.head()
    files_list=df['ImageName'].tolist()
    text=df['Description'].tolist()
    df.drop(['ImageName','Description'], axis=1, inplace=True)'''
    emotions = list(df.columns)
    emo_ratings=df.to_numpy()

    idx=[i for i in range(len(text))]

    if model_type == 'LP':
        print('LP model')
        return files_list, idx, text, emo_ratings, emotions

    if model_type == 'Custom' or model_type == 'Roberta':
        return files_list, idx, text, emo_ratings, emotions

    #tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    #model = RobertaModel.from_pretrained("roberta-base")

    temp=pd.read_csv('Tags.csv')
    Tags=temp.to_numpy()

    '''Tags=[]
    for i in range(len(text)):
        embed = get_text_embeddings(text[i], tokenizer, model)
        Tags.append(embed)
    Tags = np.array(Tags)
    temp=pd.DataFrame(Tags)
    temp.to_csv('Tags.csv',index=False)'''


    return files_list, idx, Tags, emo_ratings, emotions


def get_datafiles(files_list, idx, Tags, emo_ratings):
    x_files, test_files, x_idx, test_idx = train_test_split(files_list, idx, test_size=0.1, train_size=0.9, shuffle=True)
    train_files, val_files, train_idx, val_idx = train_test_split(x_files, x_idx, test_size=0.1, train_size=0.9, shuffle=True)

    train_text = [Tags[i] for i in train_idx]
    val_text = [Tags[i] for i in val_idx]
    test_text = [Tags[i] for i in test_idx]

    train_emo = [emo_ratings[i] for i in train_idx]
    val_emo = [emo_ratings[i] for i in val_idx]
    test_emo = [emo_ratings[i] for i in test_idx]

    return train_files, val_files, test_files, train_text, val_text, test_text, train_emo, val_emo, test_emo


