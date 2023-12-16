import spacy
import neuralcoref
import ast
from collections import Counter
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.impute import SimpleImputer

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

role_mappings = {
    "nsubj": "s",
    "dobj": "o",
    "pobj": "o",
}


def jsonl_read(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def entity_grid(text):
    transitions = list()
    entities = list()
    sentences_counter = 0
    split_id = len(text)//3    
    doc = nlp(text)
    resolved_ref = doc._.coref_resolved
    doc2 = nlp(resolved_ref)
    sentences = [sent for sent in doc2.sents]
    sentences_counter += len(sentences)
    for sent in sentences:
        dict_sentence = dict()
        for token in sent:
            if token.pos_ in ["PROPN", "NOUN", "PRON"] and token.dep_ != "compound":
                if token.text not in dict_sentence.keys():
                    token_role = token.dep_
                    if token_role in role_mappings.keys():
                        token_role = role_mappings[token_role]
                    else:
                        token_role = "x"
                    dict_sentence[token.text] = token_role
        entities.append(dict_sentence)    
    for i in range(len(entities)-1):
        for key, value in entities[i].items():
            role_1 = value
            if key in entities[i+1].keys():
                role_2 = entities[i+1][key]
            else:
                role_2 = "-"
            
            transitions.append(f"{role_1}->{role_2}")
    count_transitions = Counter(transitions)
    count_transitions = dict(count_transitions)
    weighted_transitions = dict()
    for key, value in count_transitions.items():
        weighted_transitions[key] = value / (sentences_counter-1)
    
    return weighted_transitions


data_train = jsonl_read("SubtaskA/subtaskA_train_monolingual.jsonl")
data_dev = jsonl_read("SubtaskA/subtaskA_dev_monolingual.jsonl")


transitions_data_train = list()
for sample in tqdm(data_train):
    text = sample["text"]
    transitions = entity_grid(text)
    del sample["text"]
    del sample["model"]
    del sample["source"]
    del sample["label"]
    for key, value in transitions.items():
        sample[key] = value
    transitions_data_train.append(sample)


transitions_data_dev = list()
for sample in tqdm(data_dev):
    text = sample["text"]
    transitions = entity_grid(text)
    del sample["text"]
    del sample["model"]
    del sample["source"]
    del sample["label"]
    for key, value in transitions.items():
        sample[key] = value
    transitions_data_dev.append(sample)


df_train = pd.DataFrame(transitions_data_train)

imp=SimpleImputer(missing_values=np.NaN)
idf_train=pd.DataFrame(imp.fit_transform(df_train))
idf_train.columns=df_train.columns
idf_train.index=df_train.index

idf_train.to_csv("/processed_data/entity_grid_train.csv")


df_dev = pd.DataFrame(transitions_data_dev)

idf_dev=pd.DataFrame(imp.fit_transform(df_dev))
idf_dev.columns=df_dev.columns
idf_dev.index=df_dev.index

idf_dev.to_csv("/processed_data/entity_grid_dev.csv")