import json

import spacy
import en_core_web_sm
from lexical_diversity import lex_div as ld
from scipy.stats import pointbiserialr
from scipy.sparse._csr import csr_matrix
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

def preprocess(text: str, mode: str ="lemmatize") -> list:
    if mode == "stanza":
        doc = NLP(text)
        result = [f"{w.lemma_}_{w.pos_}" for w in doc if not w.pos_ in ["PUNCT", "SYM", "SPACE"]]
    elif mode == "lemmatize":
        result = ld.flemmatize(text)
    else:
        result = ld.tokenize(text)
    return result

def lex_div_feats_extraction(entry: dict, 
                             preprocess_mode: str="lemmatize", 
                             features: list=FEATS) -> None:
    text = entry["text"]
    preprocessed = preprocess(text, preprocess_mode)
    # entry["text_preprocessed"] = preprocessed
    
    for feature in features:
        entry[feature] = getattr(ld, feature)(preprocessed)

def features_evaluation(dataset: list) -> None:
    labels = [entry["label"] for entry in dataset]

    for feature in FEATS:
        if feature in dataset[0]:
            feat_values = [entry[feature] for entry in dataset]
            point_biserial_corr, p_value = pointbiserialr(labels, feat_values)
            print(f"{feature}: {round(point_biserial_corr, 2)} (p = {round(p_value, 2)})")