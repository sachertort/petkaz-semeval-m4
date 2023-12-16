import json
from copy import deepcopy
import pickle

import spacy
import en_core_web_sm
from lexical_diversity import lex_div as ld
from scipy.stats import pointbiserialr
from scipy.sparse._csr import csr_matrix
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm

NLP = spacy.load("en_core_web_sm")
LATIN = ["i.e.", "e.g.", "etc.", "c.f.", "et", "al."]
FEATS = ["ttr", "root_ttr", "log_ttr", "maas_ttr", "msttr", "mattr", "hdd", "mtld", "mtld_ma_wrap", "mtld_ma_bid"]


def jsonl_read(file_path: str) -> list:
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def jsonl_write(data: list, file_path: str) -> None:
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def jsonl_write_lines(entry: dict, file_path: str) -> None:
    with open(file_path, 'a') as f:
        f.write(json.dumps(entry) + "\n")

def style_features_processing(entry: dict) -> tuple:
    text = entry["text"]
    
    doc = NLP(text)
    pos_tokens = []
    shape_tokens = []

    for word in doc:
        if word.is_punct or word.is_stop or word.text in LATIN:
            pos_target = word.text
            shape_target = word.text
        else:
            pos_target = word.pos_
            shape_target = word.shape_

        pos_tokens.append(pos_target)
        shape_tokens.append(shape_target)

    return " ".join(pos_tokens), " ".join(shape_tokens)

def log_counts(texts: list) -> tuple:
    vectorizer = TfidfVectorizer(lowercase=False,
                                 ngram_range=(1, 2),
                                 use_idf=False,
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

def preprocess(text: str, mode: str="spacy") -> list:
    if mode == "spacy":
        doc = NLP(text)
        result = [f"{w.lemma_}_{w.pos_}" for w in doc if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]
    elif mode == "lemmatize":
        result = ld.flemmatize(text)
    else:
        result = ld.tokenize(text)
    return result

def lex_div_feats_extraction(entry: dict, 
                             preprocess_mode: str="spacy", 
                             features: list=FEATS) -> None:
    text = entry["text"]
    preprocessed = preprocess(text, preprocess_mode)
    # entry["text_preprocessed"] = preprocessed
    
    for feature in features:
        entry[feature] = getattr(ld, feature)(preprocessed)

    return entry
    

def features_evaluation(dataset: list) -> None:
    labels = [entry["label"] for entry in dataset]

    for feature in FEATS:
        if feature in dataset[0]:
            feat_values = [entry[feature] for entry in dataset]
            point_biserial_corr, p_value = pointbiserialr(labels, feat_values)
            print(f"{feature}: {round(point_biserial_corr, 2)} (p = {round(p_value, 2)})")

def get_texts_diversities(in_file_name: str,
                          out_file_name: str):
    data = jsonl_read(in_file_name)
    for entry in tqdm(data):
        computed_entry = deepcopy(lex_div_feats_extraction(entry))
        del computed_entry["text"]
        jsonl_write_lines(computed_entry, out_file_name)

def get_texts_styles(in_file_name: str,
                     in_dev_file_name: str):
    data = jsonl_read(in_file_name)
    pos_data = []
    shape_data = []
    for entry in tqdm(data):
        pos, shape = style_features_processing(entry)
        pos_data.append(pos)
        shape_data.append(shape)

    dev_data = jsonl_read(in_dev_file_name)
    dev_pos_data = []
    dev_shape_data = []
    for entry in tqdm(dev_data):
        pos, shape = style_features_processing(entry)
        dev_pos_data.append(pos)
        dev_shape_data.append(shape)


    pos_vectorizer, pos_tf = log_counts(pos_data)
    pos_tf_dev = pos_vectorizer.transform(dev_pos_data)
    with open("processed_data/pos_vectorizer.pkl", "wb") as file:
        pickle.dump(pos_vectorizer, file)
    save_npz("processed_data/pos_tf.npz", pos_tf)
    save_npz("processed_data/pos_tf_dev.npz", pos_tf_dev)

    shape_vectorizer, shape_tf = log_counts(shape_data)
    shape_tf_dev = shape_vectorizer.transform(dev_shape_data)
    with open("processed_data/shape_vectorizer.pkl", "wb") as file:
        pickle.dump(shape_vectorizer, file)
    save_npz("processed_data/shape_tf.npz", shape_tf)
    save_npz("processed_data/shape_tf_dev.npz", shape_tf_dev)

def main():
    get_texts_styles("SubtaskA/subtaskA_train_monolingual.jsonl",
                     "SubtaskA/subtaskA_dev_monolingual.jsonl")

    get_texts_diversities("SubtaskA/subtaskA_train_monolingual.jsonl",
                          "processed_data/train_diversities.jsonl")
    get_texts_diversities("SubtaskA/subtaskA_dev_monolingual.jsonl",
                          "processed_data/dev_diversities.jsonl")

if __name__ == "__main__":
    main()
 